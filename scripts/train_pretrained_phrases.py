"""Augmented mixed pretraining followed by chord/phrase-aware Weimar fine-tuning."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from jazzbot.config import TrainConfig
from jazzbot.midi import load_tokenizer
from jazzbot.phrase_pretrain import digest, prepare_phrase_pretraining
from jazzbot.weimar import prepare_weimar


def run_stage(data, run, config, budget, initialization=None, device=None):
    last = run / "last.pt"
    active_config = run / "config.json" if last.exists() else config
    cfg = TrainConfig.load(active_config)
    cfg.model.vocab_size = len(load_tokenizer(data / "tokenizer.json"))
    step = 0
    command = [
        sys.executable,
        "-m",
        "jazzbot",
        "train",
        "--data",
        str(data),
        "--run",
        str(run),
        "--config",
        str(active_config),
        "--max-steps",
        str(budget),
    ]
    if device:
        command += ["--device", device]
    if last.exists():
        state = torch.load(last, map_location="cpu", weights_only=True)
        if (
            state["manifest_sha256"] != digest(data / "manifest.json")
            or state["tokenizer_sha256"] != digest(data / "tokenizer.json")
            or state["model_config"] != cfg.to_dict()["model"]
        ):
            raise ValueError(f"{run}: checkpoint does not match this stage's data/model")
        expected = str(initialization.resolve()) if initialization else None
        if state.get("initialized_from") != expected:
            raise ValueError(f"{run}: checkpoint has a different initialization")
        step = state["step"]
        command += ["--resume", str(last)]
    elif initialization:
        command += ["--init-from", str(initialization)]
    if step < budget:
        print(f"{run}: training from step {step} through {budget}", flush=True)
        subprocess.run(command, check=True)
    else:
        print(f"{run}: already at step {step}; skipping training", flush=True)
    if not (run / "best.pt").is_file():
        raise ValueError(f"{run}: missing best checkpoint")
    subprocess.run([sys.executable, "scripts/summarize_training.py", str(run)], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-source", type=Path, default="data/pretrain-mixture")
    parser.add_argument(
        "--pretrain-data", type=Path, default="data/pretrain-mixture-phrase-compatible"
    )
    parser.add_argument("--phrase-data", type=Path, default="data/weimar-phrases")
    parser.add_argument("--database", type=Path, default="data/weimar/wjazzd.db")
    parser.add_argument("--pretrain-run", type=Path, default="runs/augmented-pretrain-phrases-mps")
    parser.add_argument("--finetune-run", type=Path, default="runs/pretrained-weimar-phrases-mps")
    parser.add_argument("--pretrain-config", type=Path, default="configs/phrase-pretrain-mps.json")
    parser.add_argument("--finetune-config", type=Path, default="configs/phrase-finetune-mps.json")
    parser.add_argument("--pretrain-steps", type=int, default=5000)
    parser.add_argument("--finetune-steps", type=int, default=4000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument(
        "--prepare-only", action="store_true", help="Prepare and verify data without training"
    )
    args = parser.parse_args()
    if min(args.pretrain_steps, args.finetune_steps) < 1:
        parser.error("Step budgets must be positive")
    if args.pretrain_run.resolve() == args.finetune_run.resolve():
        parser.error("Pretraining and fine-tuning need different run directories")
    if not (args.pretrain_source / "manifest.json").is_file():
        parser.error("Missing prepared pretraining mixture; supply --pretrain-source")
    pre_config, fine_config = (
        TrainConfig.load(args.pretrain_config),
        TrainConfig.load(args.finetune_config),
    )
    if pre_config.model != fine_config.model:
        parser.error("Both stages must use matching model configurations")
    if not (args.phrase_data / "manifest.json").exists():
        prepare_weimar(args.database, args.phrase_data, phrase_boundaries=True)
    phrase_manifest = json.loads((args.phrase_data / "manifest.json").read_text())
    if not phrase_manifest.get("phrase_boundaries"):
        parser.error("Fine-tuning data must contain annotated phrase boundaries")
    if digest(args.phrase_data / "tokenizer.json") != phrase_manifest["tokenizer_sha256"]:
        parser.error("Phrase tokenizer differs from its manifest")
    prepare_phrase_pretraining(
        args.pretrain_source, args.pretrain_data, args.phrase_data / "tokenizer.json"
    )
    print("Both stages use the same phrase-compatible tokenizer. Data is ready.", flush=True)
    if args.prepare_only:
        return
    run_stage(
        args.pretrain_data,
        args.pretrain_run,
        args.pretrain_config,
        args.pretrain_steps,
        device=args.device,
    )
    initialization = args.pretrain_run / "best.pt"
    provenance = {
        "pretrained_checkpoint": str(initialization.resolve()),
        "pretrained_sha256": digest(initialization),
    }
    record = args.finetune_run / "pretraining_source.json"
    if record.exists():
        if json.loads(record.read_text()) != provenance:
            raise ValueError("Pretraining weights changed; select a new --finetune-run")
    elif (args.finetune_run / "last.pt").exists():
        raise ValueError("Existing fine-tune lacks this pipeline's provenance; use a new run")
    else:
        args.finetune_run.mkdir(parents=True, exist_ok=True)
        record.write_text(json.dumps(provenance, indent=2))
    run_stage(
        args.phrase_data,
        args.finetune_run,
        args.finetune_config,
        args.finetune_steps,
        initialization,
        device=args.device,
    )
    print(f"Checkpoints ready: {args.finetune_run / 'best.pt'} and {args.finetune_run / 'last.pt'}")


if __name__ == "__main__":
    main()
