import csv
import json
import shutil
import zipfile

import numpy as np
import pytest
import torch
from test_midi import write_midi

from jazzbot.augment import NoteAugmenter
from jazzbot.data import TokenCorpus
from jazzbot.download import safe_extract
from jazzbot.generate import generate
from jazzbot.midi import PreprocessConfig, load_tokenizer
from jazzbot.mix import mix_corpora
from jazzbot.prepare import prepare
from jazzbot.train import train


def make_dataset(root):
    root.mkdir()
    rows = []
    for i, split in enumerate(("train", "val", "test")):
        write_midi(root / f"{i}.mid", offset=i)
        rows.append(
            {
                "id": i,
                "midi_filepath": f"{i}.mid",
                "artist": f"Artist {i}",
                "album": f"Album {i}",
                "split": split,
            }
        )
    with (root / "pijama.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def test_train_resume_generate_and_window_boundaries(tmp_path):
    root, data = tmp_path / "raw", tmp_path / "data"
    make_dataset(root)
    prepare(
        root, data, split_mode="official", cfg=PreprocessConfig(min_notes=4), chord_compatible=True
    )
    tokenizer = load_tokenizer(data / "tokenizer.json")
    corpus = TokenCorpus(data, "train", 32, tokenizer["PAD_None"])
    all_targets = torch.cat([y[y >= 0] for _, y in corpus.eval_batches(2)])
    original = np.fromfile(data / "train.bin", dtype=np.uint16)
    assert all_targets.tolist() == original[1:].tolist()
    cfg = {
        "model": {
            "dim": 32,
            "layers": 1,
            "heads": 4,
            "kv_heads": 2,
            "hidden_dim": 64,
            "context_length": 32,
            "dropout": 0.1,
        },
        "batch_size": 2,
        "accumulation_steps": 2,
        "max_steps": 4,
        "warmup_steps": 1,
        "eval_interval": 1,
        "eval_batches": 1,
        "log_interval": 1,
        "device": "cpu",
        "min_lr_ratio": 1.0,
        "augmentation_probability": 0.9,
        "velocity_shift_bins": 6,
        "duration_scale_min": 0.65,
        "duration_scale_max": 1.35,
        "transpose": 12,
        "eval_train_batches": 1,
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    run = tmp_path / "run"
    train(data, run, config, max_steps=2)
    train(data, run, config, resume=run / "last.pt")
    checkpoint = torch.load(run / "last.pt", weights_only=True)
    assert checkpoint["step"] == 4
    assert checkpoint["tokenizer_json"] == (data / "tokenizer.json").read_bytes()
    assert "optimizer" in checkpoint and "sampler_rng" in checkpoint
    # Constant LR makes an interrupted/resumed run directly comparable to an
    # uninterrupted one, including augmentation and dropout RNG states.
    uninterrupted_run = tmp_path / "uninterrupted"
    train(data, uninterrupted_run, config)
    uninterrupted = torch.load(uninterrupted_run / "last.pt", weights_only=True)
    for key, value in checkpoint["model"].items():
        torch.testing.assert_close(value, uninterrupted["model"][key], rtol=0, atol=0)
    events = [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
    assert all("clean_train_loss" in event and "best_val_step" in event for event in events)
    augmenter = NoteAugmenter(tokenizer, 1, 6, (0.65, 1.35))
    x, y = corpus.random_batch(4, torch.Generator().manual_seed(5), augmenter=augmenter)
    assert torch.equal(x[:, 1:], y[:, :-1])
    # Include padding and confirm it never becomes a prediction target.
    padded = TokenCorpus(data, "train", 4096, tokenizer["PAD_None"])
    x, y = padded.random_batch(1, torch.Generator().manual_seed(5), augmenter=augmenter)
    size = int((y[0] >= 0).sum())
    assert torch.equal(x[0, 1:size], y[0, : size - 1])
    assert (x[0, size:] == tokenizer["PAD_None"]).all()
    assert (y[0, size:] == -100).all()
    validation = TokenCorpus(data, "val", 32, tokenizer["PAD_None"])
    with pytest.raises(ValueError, match="training split"):
        validation.random_batch(1, torch.Generator(), augmenter=augmenter)
    assert np.array_equal(np.fromfile(data / "train.bin", dtype=np.uint16), original)
    standalone = tmp_path / "standalone.pt"
    shutil.copyfile(run / "last.pt", standalone)
    assert not (standalone.parent / "tokenizer.json").exists()
    score = generate(standalone, tmp_path / "solo.mid", max_new_tokens=80, device="cpu")
    assert len(score.tracks[0].notes) > 0
    score = generate(
        run / "last.pt",
        tmp_path / "continued.mid",
        prompt=root / "0.mid",
        prompt_seconds=1.0,
        max_new_tokens=80,
        device="cpu",
    )
    assert len(score.tracks[0].notes) > 5

    initialized_run = tmp_path / "initialized-run"
    train(data, initialized_run, config, max_steps=1, init_from=run / "best.pt")
    initialized = torch.load(initialized_run / "last.pt", weights_only=True)
    assert initialized["step"] == 1
    assert initialized["initialized_from"] == str((run / "best.pt").resolve())


def test_duplicate_recordings_cannot_cross_official_splits(tmp_path):
    root = tmp_path / "raw"
    make_dataset(root)
    (root / "1.mid").write_bytes((root / "0.mid").read_bytes())
    with pytest.raises(ValueError, match="Duplicate recording"):
        prepare(root, tmp_path / "prepared", split_mode="official")


def test_zip_path_traversal_rejected(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("../outside.mid", b"bad")
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract(archive, tmp_path / "extracted")


def test_source_balanced_mixture_weights(tmp_path):
    root = tmp_path / "raw"
    make_dataset(root)
    first, second = tmp_path / "first", tmp_path / "second"
    cfg = PreprocessConfig(min_notes=4)
    prepare(root, first, split_mode="official", cfg=cfg, chord_compatible=True)
    prepare(root, second, split_mode="official", cfg=cfg, chord_compatible=True)
    target = tmp_path / "mixture"
    mix_corpora([first, second], [0.8, 0.2], target)
    manifest = json.loads((target / "manifest.json").read_text())
    for split in ("train", "val", "test"):
        totals = {}
        for record in manifest["records"]:
            if record["split"] == split:
                totals[record["dataset"]] = (
                    totals.get(record["dataset"], 0) + record["sampling_weight"]
                )
        assert totals == pytest.approx({"first": 0.8, "second": 0.2})
