"""Portable checkpoints, including the exact tokenizer used during training."""

import hashlib
import tempfile
from pathlib import Path

import torch

from .midi import load_tokenizer


def verify_tokenizer_bytes(state, content):
    if hashlib.sha256(content).hexdigest() != state["tokenizer_sha256"]:
        raise ValueError("Tokenizer does not match checkpoint")


def tokenizer_bytes(state, checkpoint):
    content = state.get("tokenizer_json")
    if content is None:
        source = Path(checkpoint).parent / "tokenizer.json"
        if not source.is_file():
            raise FileNotFoundError(
                f"This older checkpoint needs its original tokenizer at {source}. "
                "Ask the model provider for that file or a self-contained export."
            )
        content = source.read_bytes()
    if not isinstance(content, bytes):
        raise ValueError("Checkpoint tokenizer_json must contain JSON bytes")
    verify_tokenizer_bytes(state, content)
    return content


def checkpoint_tokenizer(state, checkpoint):
    content = tokenizer_bytes(state, checkpoint)
    # MidiTok's public loader takes a path. Close the file before loading on Windows.
    with tempfile.TemporaryDirectory(prefix="jazzbot-tokenizer-") as directory:
        path = Path(directory) / "tokenizer.json"
        path.write_bytes(content)
        return load_tokenizer(path)


def export_checkpoint(checkpoint, output):
    """Save inference weights and tokenizer in one file, without training state."""
    from .train import atomic_save

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    content = tokenizer_bytes(state, checkpoint)
    # Preserve the earlier directory-style CLI as well as explicit .pt destinations.
    target = Path(output)
    if target.suffix != ".pt":
        target = target / "model.pt"
    if target.exists():
        raise FileExistsError(f"Output already exists: {target}; choose another --output")
    keys = (
        "model",
        "model_config",
        "tokenizer_sha256",
        "preprocessing",
        "manifest_sha256",
        "step",
    )
    payload = {key: state[key] for key in keys}
    payload["phrase_boundaries"] = state.get("phrase_boundaries", False)
    payload["tokenizer_json"] = content
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(payload, target)
    print(f"Self-contained checkpoint: {target.resolve()}")
    return target
