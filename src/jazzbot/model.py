"""Decoder-only pre-norm Transformer, with explicit portable GQA and KV caching."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.kv_heads = cfg.heads, cfg.kv_heads
        self.head_dim = cfg.dim // cfg.heads
        self.dropout = cfg.dropout
        self.q = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.kv = nn.Linear(cfg.dim, 2 * cfg.kv_heads * self.head_dim, bias=False)
        self.out = nn.Linear(cfg.dim, cfg.dim, bias=False)
        inv = 1.0 / cfg.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        self.register_buffer("inv_freq", inv, persistent=False)

    def rotate(self, x, offset):
        positions = torch.arange(offset, offset + x.shape[-2], device=x.device).float()
        angle = torch.outer(positions, self.inv_freq.float())
        cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)

    def forward(self, x, past=None, use_cache=False):
        batch, length, _ = x.shape
        offset = 0 if past is None else past[0].shape[-2]
        q = self.q(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k, v = self.kv(x).chunk(2, dim=-1)
        k = k.view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rotate(q, offset), self.rotate(k, offset)
        if past is not None:
            k, v = torch.cat((past[0], k), dim=-2), torch.cat((past[1], v), dim=-2)
        cache = (k, v) if use_cache else None
        # repeat_interleave works on CUDA, MPS and CPU; cache stays in compact KV-head form.
        repeats = self.heads // self.kv_heads
        k, v = k.repeat_interleave(repeats, 1), v.repeat_interleave(repeats, 1)
        mask = None
        if offset and length > 1:
            mask = torch.arange(k.shape[-2], device=x.device)[None, :] <= (
                offset + torch.arange(length, device=x.device)[:, None]
            )
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=offset == 0,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(y.transpose(1, 2).contiguous().view(batch, length, -1)), cache


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm, self.ffn_norm = RMSNorm(cfg.dim), RMSNorm(cfg.dim)
        self.attention = Attention(cfg)
        self.gate = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.hidden_dim, bias=False)
        self.down = nn.Linear(cfg.hidden_dim, cfg.dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, past=None, use_cache=False):
        y, cache = self.attention(self.attn_norm(x), past, use_cache)
        x = x + self.dropout(y)
        y = self.ffn_norm(x)
        return x + self.dropout(self.down(F.silu(self.gate(y)) * self.up(y))), cache


class JazzTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.validate()
        self.config = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.norm = RMSNorm(cfg.dim)
        self.apply(self._init)
        for block in self.blocks:
            nn.init.normal_(block.attention.out.weight, std=0.02 / math.sqrt(2 * cfg.layers))
            nn.init.normal_(block.down.weight, std=0.02 / math.sqrt(2 * cfg.layers))

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, ids, targets=None, cache=None, use_cache=False, last_only=False):
        past_length = 0 if cache is None else cache[0][0].shape[-2]
        if ids.shape[1] + past_length > self.config.context_length:
            raise ValueError("Sequence plus KV cache exceeds context_length")
        x = self.embedding(ids)
        next_cache = []
        for i, block in enumerate(self.blocks):
            x, layer_cache = block(x, None if cache is None else cache[i], use_cache)
            if use_cache:
                next_cache.append(layer_cache)
        if last_only:
            x = x[:, -1:]
        logits = F.linear(self.norm(x), self.embedding.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), ignore_index=-100)
        return logits, loss, next_cache if use_cache else None

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
