import argparse
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="MIDI Bot 9000: train and sample tiny jazz Transformers"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser(
        "download", help="Download verified PiJAMA Hawthorne transcriptions"
    )
    download.add_argument("--root", default="data/pijama")
    prepare = commands.add_parser("prepare", help="Extract monophonic voices and tokenize")
    prepare.add_argument("--root", default="data/pijama")
    prepare.add_argument("--output", default="data/processed")
    prepare.add_argument("--split", choices=["album", "official"], default="album")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--limit", type=int)
    prepare.add_argument("--min-pitch", type=int, default=48)
    prepare.add_argument("--max-pitch", type=int, default=96)
    prepare.add_argument("--min-notes", type=int, default=32)
    prepare.add_argument(
        "--chord-compatible",
        action="store_true",
        help="Use the conditioned vocabulary and insert Chord|NC before each note",
    )
    prepare_weimar = commands.add_parser(
        "prepare-weimar", help="Prepare beat-normalized solos with chord tokens"
    )
    prepare_weimar.add_argument("--database", default="data/weimar/wjazzd.db")
    prepare_weimar.add_argument("--output", default="data/weimar-conditioned")
    prepare_weimar.add_argument("--seed", type=int, default=42)
    prepare_weimar.add_argument("--limit", type=int)
    prepare_directory = commands.add_parser(
        "prepare-midi-dir", help="Prepare a directory of MIDI files with Chord|NC tokens"
    )
    prepare_directory.add_argument("--input", required=True)
    prepare_directory.add_argument("--output", required=True)
    prepare_directory.add_argument("--dataset", required=True)
    prepare_directory.add_argument("--seed", type=int, default=42)
    prepare_directory.add_argument("--limit", type=int)
    mix = commands.add_parser("mix", help="Build a source-balanced prepared corpus")
    mix.add_argument("--input", action="append", required=True)
    mix.add_argument("--weight", action="append", type=float, required=True)
    mix.add_argument("--output", required=True)
    train = commands.add_parser("train", help="Train or resume")
    train.add_argument("--data", default="data/processed")
    train.add_argument("--run", default="runs/jazz-v1")
    train.add_argument("--config", default="configs/m4max.json")
    train.add_argument("--resume")
    train.add_argument("--init-from", help="Initialize model weights from a compatible checkpoint")
    train.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    train.add_argument("--max-steps", type=int)
    generate = commands.add_parser("generate", help="Generate MIDI, optionally continuing a prompt")
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--output", default="outputs/solo.mid")
    generate.add_argument("--prompt")
    generate.add_argument("--prompt-seconds", type=float)
    generate.add_argument("--max-new-tokens", type=int, default=512)
    generate.add_argument("--min-notes", type=int, default=16)
    generate.add_argument("--temperature", type=float, default=0.9)
    generate.add_argument("--top-k", type=int, default=32)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    generate.add_argument(
        "--chords", help="Comma-separated forced progression, e.g. Dm7,G7,Cmaj7,A7"
    )
    generate.add_argument("--beats-per-chord", type=float, default=4.0)
    generate.add_argument(
        "--strip-tempo",
        action="store_true",
        help="Remove tempo metadata so imported MIDI follows the DAW project tempo",
    )
    evaluate = commands.add_parser("evaluate", help="Full token-weighted held-out evaluation")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--data", default="data/processed")
    evaluate.add_argument("--split", choices=["val", "test"], default="test")
    evaluate.add_argument("--batch-size", type=int, default=4)
    evaluate.add_argument("--max-batches", type=int)
    evaluate.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    export = commands.add_parser("export", help="Save an inference-only PyTorch bundle")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", default="outputs/jazzbot")
    info = commands.add_parser("info", help="Model size and available hardware")
    info.add_argument("--config", default="configs/m4max.json")
    info.add_argument("--chord-compatible", action="store_true")
    extract = commands.add_parser("extract", help="Export the upper-voice heuristic for listening")
    extract.add_argument("--input", required=True)
    extract.add_argument("--output", default="outputs/extracted.mid")
    extract.add_argument("--seconds", type=float)
    inspect = commands.add_parser("inspect", help="MIDI note, timing and overlap diagnostics")
    inspect.add_argument("midi")
    args = parser.parse_args()
    if args.command == "extract":
        from .midi import preprocess_midi

        score = preprocess_midi(args.input, end=args.seconds)
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        score.dump_midi(target)
        print(f"Extracted MIDI: {target.resolve()}")
    elif args.command == "inspect":
        from .inspect import midi_stats

        print(json.dumps(midi_stats(args.midi), indent=2))
    elif args.command == "download":
        from .download import download

        download(args.root)
    elif args.command == "prepare":
        from .midi import PreprocessConfig
        from .prepare import prepare

        if not 21 <= args.min_pitch <= args.max_pitch <= 108 or args.min_notes < 1:
            parser.error("Require 21 <= min-pitch <= max-pitch <= 108 and min-notes > 0")
        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        prepare(
            args.root,
            args.output,
            args.split,
            args.seed,
            args.limit,
            PreprocessConfig(
                min_pitch=args.min_pitch, max_pitch=args.max_pitch, min_notes=args.min_notes
            ),
            args.chord_compatible,
        )
    elif args.command == "prepare-weimar":
        from .weimar import prepare_weimar

        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        prepare_weimar(args.database, args.output, args.seed, args.limit)
    elif args.command == "prepare-midi-dir":
        from .folder import prepare_midi_directory

        if args.limit is not None and args.limit < 1:
            parser.error("--limit must be positive")
        prepare_midi_directory(args.input, args.output, args.dataset, args.seed, args.limit)
    elif args.command == "mix":
        from .mix import mix_corpora

        if len(args.input) != len(args.weight):
            parser.error("Provide one --weight for each --input")
        mix_corpora(args.input, args.weight, args.output)
    elif args.command == "train":
        from .train import train

        train(
            args.data,
            args.run,
            args.config,
            args.resume,
            args.device,
            args.max_steps,
            args.init_from,
        )
    elif args.command == "generate":
        from .generate import generate

        generate(**{k: v for k, v in vars(args).items() if k != "command"})
    elif args.command == "evaluate":
        import hashlib

        from .data import TokenCorpus
        from .midi import load_tokenizer
        from .train import choose_device, evaluate, load_checkpoint, verify_tokenizer

        device = choose_device(args.device)
        model, state = load_checkpoint(args.checkpoint, device)
        verify_tokenizer(state, Path(args.data) / "tokenizer.json")
        digest = hashlib.sha256((Path(args.data) / "manifest.json").read_bytes()).hexdigest()
        if digest != state["manifest_sha256"]:
            raise ValueError("Evaluation corpus differs from checkpoint dataset")
        tokenizer = load_tokenizer(Path(args.data) / "tokenizer.json")
        corpus = TokenCorpus(
            args.data, args.split, model.config.context_length, tokenizer["PAD_None"]
        )
        print(
            json.dumps(evaluate(model, corpus, device, args.batch_size, args.max_batches), indent=2)
        )
    elif args.command == "export":
        import torch

        from .train import atomic_save, verify_tokenizer

        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        source = Path(args.checkpoint).parent / "tokenizer.json"
        verify_tokenizer(state, source)
        target = Path(args.output)
        target.mkdir(parents=True, exist_ok=True)
        keys = (
            "model",
            "model_config",
            "tokenizer_sha256",
            "preprocessing",
            "manifest_sha256",
            "step",
        )
        atomic_save({key: state[key] for key in keys}, target / "model.pt")
        shutil.copyfile(source, target / "tokenizer.json")
        print(f"Inference bundle: {target.resolve()}")
    elif args.command == "info":
        import torch

        from .config import TrainConfig
        from .midi import make_tokenizer
        from .model import JazzTransformer

        cfg = TrainConfig.load(args.config)
        cfg.model.vocab_size = len(make_tokenizer(use_chords=args.chord_compatible))
        model = JazzTransformer(cfg.model)
        print(
            json.dumps(
                {
                    "parameters": model.parameter_count(),
                    "vocab_size": cfg.model.vocab_size,
                    "fp32_weights_mib": model.parameter_count() * 4 / 2**20,
                    "cuda_available": torch.cuda.is_available(),
                    "mps_available": torch.backends.mps.is_available(),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
