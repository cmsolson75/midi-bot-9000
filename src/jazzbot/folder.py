"""Prepare a directory of MIDI files for harmony-agnostic pretraining."""

import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .midi import (
    PreprocessConfig,
    add_no_chord_tokens,
    encode,
    make_tokenizer,
    preprocess_midi,
)


def prepare_midi_directory(source, output, dataset, seed=42, limit=None, cfg=None):
    source, output = Path(source).resolve(), Path(output)
    cfg = cfg or PreprocessConfig(min_pitch=36, max_pitch=100, min_notes=16)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is not empty; use a new output directory")
    files = sorted(path for path in source.rglob("*") if path.suffix.lower() in {".mid", ".midi"})
    if limit is not None:
        files = files[:limit]
    if not files:
        raise ValueError(f"No MIDI files found in {source}")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = make_tokenizer(cfg.resolution, use_chords=True)
    tokenizer.save(output / "tokenizer.json")
    records, skipped, seen = [], [], set()
    buffers = {split: [] for split in ("train", "val", "test")}
    offsets = {split: 0 for split in buffers}
    for path in tqdm(files, desc=f"Preparing {dataset}"):
        relative = path.relative_to(source).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        fraction = (
            int(hashlib.sha256(f"{seed}|{dataset}|{relative}".encode()).hexdigest()[:8], 16) / 2**32
        )
        split = "train" if fraction < 0.8 else "val" if fraction < 0.9 else "test"
        try:
            if digest in seen:
                raise ValueError("duplicate MIDI content")
            score = preprocess_midi(path, cfg)
            note_count = len(score.tracks[0].notes)
            if note_count < cfg.min_notes:
                raise ValueError(f"only {note_count} extracted notes")
            encoded = add_no_chord_tokens(tokenizer, encode(tokenizer, score))
            ids = [tokenizer["BOS_None"], *encoded, tokenizer["EOS_None"]]
            records.append(
                {
                    "id": relative,
                    "source": relative,
                    "dataset": dataset,
                    "sha256": digest,
                    "split": split,
                    "offset": offsets[split],
                    "length": len(ids),
                    "notes": note_count,
                }
            )
            buffers[split].append(np.asarray(ids, dtype=np.uint16))
            offsets[split] += len(ids)
            seen.add(digest)
        except (ValueError, OSError, RuntimeError, EOFError) as exc:
            skipped.append({"source": relative, "reason": str(exc)})
    for split, arrays in buffers.items():
        (np.concatenate(arrays) if arrays else np.array([], dtype=np.uint16)).tofile(
            output / f"{split}.bin"
        )
    tokenizer_hash = hashlib.sha256((output / "tokenizer.json").read_bytes()).hexdigest()
    manifest = {
        "format_version": 1,
        "dtype": "uint16",
        "dataset": dataset,
        "split_mode": "file",
        "seed": seed,
        "preprocessing": cfg.to_dict(),
        "representation": "REMI with Chord|NC before each note",
        "vocab_size": len(tokenizer),
        "tokenizer_sha256": tokenizer_hash,
        "records": records,
        "skipped": skipped,
        "tokens": offsets,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset": dataset,
                "recordings": len(records),
                "skipped": len(skipped),
                "tokens": offsets,
            },
            indent=2,
        )
    )
    if not offsets["train"] or not offsets["val"]:
        raise ValueError("Need nonempty train and val splits")
    return output
