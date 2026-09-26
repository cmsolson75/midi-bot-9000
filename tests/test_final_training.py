import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from test_phrases import make_weimar
from test_pipeline import make_dataset

from jazzbot.midi import PreprocessConfig
from jazzbot.prepare import prepare

PROJECT = Path(__file__).resolve().parents[1]


def test_two_stage_runner_initializes_and_skips_completed_stages(tmp_path):
    raw, source = tmp_path / "raw", tmp_path / "source"
    make_dataset(raw)
    prepare(
        raw, source, split_mode="official", cfg=PreprocessConfig(min_notes=4), chord_compatible=True
    )
    database = tmp_path / "weimar.db"
    connection, _ = make_weimar(database)
    connection.close()
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "model": {
                    "dim": 32,
                    "layers": 1,
                    "heads": 4,
                    "kv_heads": 2,
                    "hidden_dim": 64,
                    "context_length": 32,
                    "dropout": 0,
                },
                "batch_size": 1,
                "accumulation_steps": 1,
                "eval_batches": 1,
                "eval_interval": 1,
                "warmup_steps": 0,
                "device": "mps",
            }
        )
    )
    pretrain, finetune = tmp_path / "pretrain", tmp_path / "finetune"
    command = [sys.executable, str(PROJECT / "scripts/train_pretrained_phrases.py")]
    for key, value in {
        "pretrain-source": source,
        "pretrain-data": tmp_path / "aligned",
        "phrase-data": tmp_path / "phrases",
        "database": database,
        "pretrain-config": config,
        "finetune-config": config,
        "pretrain-run": pretrain,
        "finetune-run": finetune,
        "pretrain-steps": 1,
        "finetune-steps": 1,
        "device": "cpu",
    }.items():
        command += [f"--{key}", str(value)]
    env = {**os.environ, "OMP_NUM_THREADS": "2"}
    subprocess.run(command, cwd=PROJECT, env=env, check=True, capture_output=True, text=True)
    state = torch.load(finetune / "last.pt", weights_only=True)
    assert state["step"] == 1
    assert state["train_config"]["device"] == "cpu"
    assert state["initialized_from"] == str(pretrain / "best.pt")
    assert state["phrase_boundaries"] is True
    assert isinstance(state["tokenizer_json"], bytes)
    before = (finetune / "last.pt").read_bytes()
    result = subprocess.run(
        command, cwd=PROJECT, env=env, check=True, capture_output=True, text=True
    )
    assert result.stdout.count("skipping training") == 2
    assert (finetune / "last.pt").read_bytes() == before
    # Changing the parent weights must not silently reuse the existing fine-tune.
    with (pretrain / "best.pt").open("ab") as handle:
        handle.write(b"changed")
    result = subprocess.run(command, cwd=PROJECT, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Pretraining weights changed" in result.stderr


@pytest.mark.skipif(os.name == "nt" or not shutil.which("bash"), reason="Requires native Bash")
@pytest.mark.parametrize("options", [[], ["--device", "cpu"], ["--prepare-only"]])
def test_bash_runner_forwards_options_on_system_bash(tmp_path, options):
    log = tmp_path / "commands.log"
    stub = tmp_path / "uv"
    stub.write_text('#!/bin/sh\necho "$*" >> "$MIDI_TEST_LOG"\n')
    stub.chmod(0o755)
    subprocess.run(
        ["bash", str(PROJECT / "scripts/train_final.sh"), *options],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "MIDI_TEST_LOG": str(log),
        },
        check=True,
        capture_output=True,
        text=True,
    )
    commands = log.read_text().splitlines()
    expected = "run --locked python scripts/train_pretrained_phrases.py --device "
    expected += "cpu" if "cpu" in options else "auto"
    if "--prepare-only" in options:
        expected += " --prepare-only"
    assert commands[-1] == expected
    assert commands[-2] == "run --locked python scripts/prepare_final_training.py"


@pytest.mark.skipif(os.name == "nt" or not shutil.which("bash"), reason="Requires native Bash")
def test_bash_runner_stops_when_preparation_fails(tmp_path):
    log = tmp_path / "commands.log"
    stub = tmp_path / "uv"
    stub.write_text('#!/bin/sh\necho "$*" >> "$MIDI_TEST_LOG"\nexit 7\n')
    stub.chmod(0o755)
    result = subprocess.run(
        ["bash", str(PROJECT / "scripts/train_final.sh"), "--prepare-only"],
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
            "MIDI_TEST_LOG": str(log),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 7
    assert log.read_text().splitlines() == ["run --locked python scripts/prepare_final_training.py"]
