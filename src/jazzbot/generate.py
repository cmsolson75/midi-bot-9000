import json
import time
from pathlib import Path

import torch

from .midi import PreprocessConfig, decode, encode, load_tokenizer, preprocess_midi
from .train import choose_device, load_checkpoint, verify_tokenizer


class MonophonicGrammar:
    """Stateful REMI mask: complete note tuples, forward time, no overlapping notes."""

    def __init__(self, tokenizer, min_notes=16, min_pitch=48, max_pitch=96):
        self.tokenizer = tokenizer
        self.types = {}
        self.values = {}
        self.resolution = max(tokenizer.config.beat_res.values())
        for i in range(len(tokenizer)):
            kind, value = tokenizer[i].split("_", 1)
            self.types.setdefault(kind, []).append(i)
            self.values[i] = value
        self.bar, self.position, self.end = -1, -1, 0
        self.previous = "BOS"
        self.notes, self.min_notes = 0, min_notes
        self.min_pitch, self.max_pitch = min_pitch, max_pitch

    def consume(self, token):
        kind, value = self.tokenizer[int(token)].split("_", 1)
        if kind == "Bar":
            self.bar += 1
            self.position = -1
        elif kind == "Position":
            self.position = int(value)
        elif kind == "Duration":
            beats, frames, resolution = map(int, value.split("."))
            duration = round((beats + frames / resolution) * self.resolution)
            self.end = self.bar * 4 * self.resolution + self.position + duration
            self.notes += 1
        self.previous = kind

    def allowed(self):
        if self.previous == "BOS":
            return self.types["Bar"]
        if self.previous == "Position":
            return [
                i
                for i in self.types["Pitch"]
                if self.min_pitch <= int(self.values[i]) <= self.max_pitch
            ]
        if self.previous == "Pitch":
            return self.types["Velocity"]
        if self.previous == "Velocity":
            return self.types["Duration"]
        if self.previous in {"Bar", "Duration"}:
            result = list(self.types["Bar"])
            result += [
                i
                for i in self.types["Position"]
                if int(self.values[i]) > self.position
                and self.bar * 4 * self.resolution + int(self.values[i]) >= self.end
            ]
            if self.previous == "Duration" and self.notes >= self.min_notes:
                result += self.types["EOS"]
            return result
        return []


def sample(logits, allowed, temperature=0.9, top_k=32, top_p=0.95, generator=None):
    if temperature <= 0 or not 0 < top_p <= 1 or top_k < 0:
        raise ValueError("Require temperature > 0, 0 < top_p <= 1, top_k >= 0")
    if not allowed:
        raise ValueError("No valid token in grammar state")
    # Sampling on CPU makes the seeded sampling stream portable to MPS.
    values = logits.detach().float().cpu()[allowed] / temperature
    values, order = torch.sort(values, descending=True)
    if top_k:
        values, order = values[:top_k], order[:top_k]
    probabilities = values.softmax(-1)
    remove = probabilities.cumsum(-1) - probabilities >= top_p
    probabilities[remove] = 0
    choice = int(torch.multinomial(probabilities, 1, generator=generator))
    return allowed[int(order[choice])]


@torch.inference_mode()
def generate(
    checkpoint,
    output,
    prompt=None,
    prompt_seconds=None,
    max_new_tokens=512,
    temperature=0.9,
    top_k=32,
    top_p=0.95,
    seed=42,
    device="auto",
    min_notes=16,
):
    if max_new_tokens < 4 or min_notes < 0:
        raise ValueError("max_new_tokens must be >= 4 and min_notes >= 0")
    if prompt_seconds is not None and prompt_seconds <= 0:
        raise ValueError("prompt_seconds must be positive")
    device = choose_device(device)
    model, state = load_checkpoint(checkpoint, device)
    model.eval()
    tokenizer_path = Path(checkpoint).parent / "tokenizer.json"
    verify_tokenizer(state, tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_path)
    cfg = PreprocessConfig(**state["preprocessing"])
    ids = [tokenizer["BOS_None"]]
    if prompt:
        score = preprocess_midi(prompt, cfg, end=prompt_seconds)
        if not score.tracks[0].notes:
            raise ValueError("Prompt contains no notes in the configured extraction range")
        ids += encode(tokenizer, score)
    grammar = MonophonicGrammar(tokenizer, min_notes, cfg.min_pitch, cfg.max_pitch)
    for token in ids:
        grammar.consume(token)
    grammar.min_notes += grammar.notes  # Minimum applies to newly generated notes.
    prompt_tokens, prompt_notes = len(ids), grammar.notes
    random = torch.Generator().manual_seed(seed)
    context = model.config.context_length
    current = torch.tensor([ids[-context:]], device=device)
    cache = None
    complete = len(ids)
    started = time.perf_counter()
    for _ in range(max_new_tokens):
        logits, _, cache = model(current, cache=cache, use_cache=True, last_only=True)
        token = sample(logits[0, -1], grammar.allowed(), temperature, top_k, top_p, random)
        ids.append(token)
        grammar.consume(token)
        if grammar.previous in {"Duration", "EOS"}:
            complete = len(ids)
        if token == tokenizer["EOS_None"]:
            break
        if cache[0][0].shape[-2] >= context:
            # Re-prefill with half a context at rollover, then resume cheap cached steps.
            cache = None
            current = torch.tensor([ids[-max(1, context // 2) :]], device=device)
        else:
            current = torch.tensor([[token]], device=device)
    ids = ids[:complete]  # Never decode an unfinished Pitch/Velocity/Duration tuple.
    result = decode(tokenizer, ids)
    if grammar.notes <= prompt_notes:
        raise RuntimeError("No complete new notes were sampled; increase max_new_tokens")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.dump_midi(output)
    report = {
        "checkpoint": str(checkpoint),
        "prompt": str(prompt) if prompt else None,
        "prompt_tokens": prompt_tokens,
        "new_tokens": len(ids) - prompt_tokens,
        "new_notes": grammar.notes - prompt_notes,
        "total_notes": len(result.tracks[0].notes),
        "seed": seed,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
        "seconds": time.perf_counter() - started,
        "device": str(device),
        "tokens": ids,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "tokens"}, indent=2))
    return result
