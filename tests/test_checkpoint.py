import hashlib
import shutil
import subprocess
import sys
from dataclasses import asdict

import pytest
import torch

from jazzbot.checkpoint import checkpoint_tokenizer, export_checkpoint
from jazzbot.config import ModelConfig
from jazzbot.generate import generate
from jazzbot.midi import PreprocessConfig, make_tokenizer
from jazzbot.model import JazzTransformer


@pytest.fixture(params=["plain", "chords", "phrases"])
def legacy_checkpoint(tmp_path, request):
    source = tmp_path / "legacy"
    source.mkdir()
    tokenizer = make_tokenizer(
        use_chords=request.param != "plain", use_phrases=request.param == "phrases"
    )
    tokenizer.save(source / "tokenizer.json")
    cfg = ModelConfig(
        vocab_size=len(tokenizer),
        dim=32,
        layers=1,
        heads=4,
        kv_heads=2,
        hidden_dim=64,
        context_length=32,
        dropout=0,
    )
    torch.manual_seed(42)
    state = {
        "model": JazzTransformer(cfg).state_dict(),
        "model_config": asdict(cfg),
        "tokenizer_sha256": hashlib.sha256((source / "tokenizer.json").read_bytes()).hexdigest(),
        "preprocessing": asdict(PreprocessConfig()),
        "manifest_sha256": "synthetic-test",
        "step": 1,
        "optimizer": {},
        "phrase_boundaries": request.param == "phrases",
    }
    checkpoint = source / "best.pt"
    torch.save(state, checkpoint)
    return checkpoint, state, tokenizer


def test_export_and_generate_with_only_checkpoint(tmp_path, legacy_checkpoint):
    source, state, tokenizer = legacy_checkpoint
    original = generate(source, tmp_path / "legacy.mid", max_new_tokens=80, device="cpu")
    target = tmp_path / "portable" / "model.pt"
    export_checkpoint(source, target)
    exported = torch.load(target, weights_only=True)
    assert "optimizer" not in exported
    assert exported["phrase_boundaries"] == state["phrase_boundaries"]
    assert exported["tokenizer_json"] == (source.parent / "tokenizer.json").read_bytes()
    shutil.rmtree(source.parent)
    loaded = checkpoint_tokenizer(exported, target)
    assert [loaded[i] for i in range(len(loaded))] == [tokenizer[i] for i in range(len(tokenizer))]
    # No sidecar, dataset, or original checkpoint remains available.
    result = generate(target, tmp_path / "portable.mid", max_new_tokens=80, device="cpu")
    assert result.tracks[0].notes == original.tracks[0].notes
    with pytest.raises(FileExistsError, match="already exists"):
        export_checkpoint(target, target)
    # A self-contained model can be re-exported without reconstructing a sidecar.
    export_checkpoint(target, tmp_path / "second.pt")


def test_missing_or_mismatched_tokenizer_rejected(tmp_path, legacy_checkpoint):
    source, state, _ = legacy_checkpoint
    sidecar = source.parent / "tokenizer.json"
    content = sidecar.read_bytes()
    sidecar.unlink()
    with pytest.raises(FileNotFoundError, match="original tokenizer"):
        export_checkpoint(source, tmp_path / "missing.pt")
    assert not (tmp_path / "missing.pt").exists()
    state["tokenizer_json"] = content + b" "
    with pytest.raises(ValueError, match="does not match"):
        checkpoint_tokenizer(state, source)
    sidecar.write_bytes(content + b" ")
    with pytest.raises(ValueError, match="does not match"):
        export_checkpoint(source, tmp_path / "mismatch.pt")


def test_cli_missing_checkpoint_has_actionable_error(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "jazzbot", "generate", "--checkpoint", str(tmp_path / "missing.pt")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "missing.pt" in result.stderr
    assert "Traceback" not in result.stderr
