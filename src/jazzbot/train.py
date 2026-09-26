import hashlib
import json
import math
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.nn import functional as F

from .augment import NoteAugmenter
from .checkpoint import verify_tokenizer_bytes
from .config import ModelConfig, TrainConfig
from .data import TokenCorpus
from .midi import CHORD_ROOTS, load_tokenizer
from .model import JazzTransformer


def choose_device(name="auto"):
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA unavailable: install a CUDA-enabled PyTorch build or use --device cpu"
        )
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError(
            "MPS unavailable: use an Apple Silicon Mac with an MPS-enabled PyTorch build"
        )
    return torch.device(name)


def precision_context(device, precision):
    if precision == "auto":
        precision = (
            ("bf16" if torch.cuda.is_bf16_supported() else "fp16")
            if device.type == "cuda"
            else "fp32"
        )
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be auto, fp32, bf16 or fp16")
    if device.type != "cuda" and precision != "fp32":
        raise ValueError("V1 uses fp32 on CPU/MPS; mixed precision is supported on CUDA")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision)
    factory = (
        (lambda: torch.autocast(device_type=device.type, dtype=dtype)) if dtype else nullcontext
    )
    return precision, factory


def lr_at(step, cfg):
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    progress = min(1.0, (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps - 1))
    return cfg.learning_rate * (
        cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
    )


@torch.inference_mode()
def evaluate(model, corpus, device, batch_size=4, max_batches=None, autocast=nullcontext):
    was_training = model.training
    model.eval()
    total_loss, total_tokens, correct = 0.0, 0, 0
    try:
        for x, y in corpus.eval_batches(batch_size, max_batches):
            x, y = x.to(device), y.to(device)
            with autocast():
                logits, _, _ = model(x)
            loss = F.cross_entropy(
                logits.float().flatten(0, 1), y.flatten(), ignore_index=-100, reduction="sum"
            )
            mask = y != -100
            total_tokens += int(mask.sum())
            total_loss += float(loss)
            correct += int(((logits.argmax(-1) == y) & mask).sum())
    finally:
        model.train(was_training)
    nll = total_loss / total_tokens
    return {
        "loss": nll,
        "perplexity": math.exp(min(nll, 80)),
        "accuracy": correct / total_tokens,
        "tokens": total_tokens,
    }


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = JazzTransformer(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device), checkpoint


def verify_tokenizer(checkpoint, path):
    verify_tokenizer_bytes(checkpoint, Path(path).read_bytes())


def train(
    data,
    run,
    config,
    resume=None,
    device_override=None,
    max_steps=None,
    init_from=None,
):
    if resume and init_from:
        raise ValueError("Use either --resume or --init-from, not both")
    cfg = TrainConfig.load(config)
    if device_override:
        cfg.device = device_override
    if max_steps:
        cfg.max_steps = max_steps
    device = choose_device(cfg.device)
    precision, autocast = precision_context(device, cfg.precision)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    data, run = Path(data), Path(run)
    run.mkdir(parents=True, exist_ok=True)
    if (run / "last.pt").exists() and not resume:
        raise ValueError("Run already has a checkpoint: use --resume or a new --run directory")
    manifest_bytes = (data / "manifest.json").read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    tokenizer = load_tokenizer(data / "tokenizer.json")
    tokenizer_content = (data / "tokenizer.json").read_bytes()
    tokenizer_hash = hashlib.sha256(tokenizer_content).hexdigest()
    if tokenizer_hash != manifest["tokenizer_sha256"]:
        raise ValueError("Prepared tokenizer differs from manifest")
    cfg.model.vocab_size = len(tokenizer)
    cfg.model.validate()
    train_data = TokenCorpus(data, "train", cfg.model.context_length, tokenizer["PAD_None"])
    val_data = TokenCorpus(data, "val", cfg.model.context_length, tokenizer["PAD_None"])
    model = JazzTransformer(cfg.model).to(device)
    decay, no_decay = [], []
    for p in model.parameters():
        (decay if p.ndim >= 2 else no_decay).append(p)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")
    generator = torch.Generator().manual_seed(cfg.seed)
    step, best, best_step = 0, float("inf"), 0
    initialized_from = None
    if init_from:
        state = torch.load(init_from, map_location="cpu", weights_only=True)
        if state["model_config"] != cfg.to_dict()["model"]:
            raise ValueError("Initialization checkpoint requires the same model configuration")
        verify_tokenizer(state, data / "tokenizer.json")
        model.load_state_dict(state["model"])
        initialized_from = str(Path(init_from).resolve())
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if (
            state["manifest_sha256"] != manifest_hash
            or state["model_config"] != cfg.to_dict()["model"]
        ):
            raise ValueError("Resume requires the same dataset and model configuration")
        verify_tokenizer(state, data / "tokenizer.json")
        # Training schedule/batch changes are explicit in the newly saved config.
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        generator.set_state(state["sampler_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda" and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if device.type == "mps" and state.get("mps_rng") is not None:
            torch.mps.set_rng_state(state["mps_rng"])
        step, best = state["step"], state["best_val_loss"]
        best_step = state.get("best_val_step")
        initialized_from = state.get("initialized_from")
    if step >= cfg.max_steps:
        raise ValueError(f"Checkpoint already at step {step}; increase max_steps to continue")
    shutil.copyfile(data / "tokenizer.json", run / "tokenizer.json")
    (run / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    pitch_lookup = torch.tensor(
        [
            int(token.split("_")[1]) if token.startswith("Pitch_") else -1
            for token in (tokenizer[i] for i in range(len(tokenizer)))
        ]
    )
    augmenter = None
    if cfg.augmentation_probability:
        augmenter = NoteAugmenter(
            tokenizer,
            cfg.augmentation_probability,
            cfg.velocity_shift_bins,
            (cfg.duration_scale_min, cfg.duration_scale_max),
        )
    chord_ids = {}
    for token_id in range(len(tokenizer)):
        token = tokenizer[token_id]
        if token.startswith("Chord|") and token != "Chord|NC_None":
            root, quality = token.removeprefix("Chord|").removesuffix("_None").split(":", 1)
            chord_ids[(root, quality)] = token_id
    chord_remaps = None
    if chord_ids:
        chord_remaps = {}
        for shift in range(-cfg.transpose, cfg.transpose + 1):
            remap = torch.arange(len(tokenizer))
            for (root, quality), token_id in chord_ids.items():
                shifted_root = CHORD_ROOTS[(CHORD_ROOTS.index(root) + shift) % 12]
                remap[token_id] = chord_ids[(shifted_root, quality)]
            chord_remaps[shift] = remap
    print(
        f"{model.parameter_count():,} parameters | {device} | {precision} | "
        f"{cfg.batch_size * cfg.accumulation_steps * cfg.model.context_length:,} tokens/update"
    )
    model.train()
    started = time.perf_counter()
    token_count = 0
    while step < cfg.max_steps:
        optimizer.zero_grad(set_to_none=True)
        lr = lr_at(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        batches = [
            train_data.random_batch(
                cfg.batch_size,
                generator,
                pitch_lookup,
                cfg.transpose,
                chord_remaps,
                augmenter=augmenter,
            )
            for _ in range(cfg.accumulation_steps)
        ]
        valid_tokens = sum(int((y != -100).sum()) for _, y in batches)
        loss_sum = 0.0
        for x, y in batches:
            x, y = x.to(device), y.to(device)
            with autocast():
                _, loss, _ = model(x, y)
                weight = (y != -100).sum() / valid_tokens
                scaled_loss = loss * weight
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite training loss; lower learning rate or use fp32"
                )
            scaler.scale(scaled_loss).backward()
            loss_sum += float(scaled_loss.detach())
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if not scaler.is_enabled() and not torch.isfinite(grad_norm):
            raise FloatingPointError("Non-finite gradient norm")
        scaler.step(optimizer)
        scaler.update()
        step += 1
        token_count += valid_tokens
        event = {
            "step": step,
            "train_loss": loss_sum,
            "lr": lr,
            "grad_norm": float(grad_norm),
            "tokens_per_second": token_count / (time.perf_counter() - started),
        }
        should_eval = step % cfg.eval_interval == 0 or step == cfg.max_steps
        if should_eval:
            result = evaluate(model, val_data, device, cfg.batch_size, cfg.eval_batches, autocast)
            event.update({f"val_{k}": v for k, v in result.items()})
            improved = result["loss"] < best
            if improved:
                best_step = step
            best = min(best, result["loss"])
            event.update(
                best_val_loss=best,
                best_val_step=best_step,
                val_loss_above_best=result["loss"] - best,
            )
            if cfg.eval_train_batches:
                clean = evaluate(
                    model, train_data, device, cfg.batch_size, cfg.eval_train_batches, autocast
                )
                event.update({f"clean_train_{k}": v for k, v in clean.items()})
                event["generalization_gap"] = result["loss"] - clean["loss"]
            payload = {
                "model": model.state_dict(),
                "model_config": cfg.to_dict()["model"],
                "train_config": cfg.to_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "step": step,
                "best_val_loss": best,
                "best_val_step": best_step,
                "sampler_rng": generator.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "mps_rng": torch.mps.get_rng_state() if device.type == "mps" else None,
                "tokenizer_sha256": tokenizer_hash,
                "tokenizer_json": tokenizer_content,
                "manifest_sha256": manifest_hash,
                "preprocessing": manifest["preprocessing"],
                "phrase_boundaries": manifest.get("phrase_boundaries", False),
                "initialized_from": initialized_from,
            }
            atomic_save(payload, run / "last.pt")
            if improved:
                atomic_save(payload, run / "best.pt")
        if step % cfg.log_interval == 0 or should_eval:
            print(json.dumps(event), flush=True)
            with (run / "metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
    return run / "last.pt"
