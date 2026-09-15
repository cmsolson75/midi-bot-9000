import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .midi import PreprocessConfig, add_no_chord_tokens, encode, make_tokenizer, preprocess_midi


def finite_number(value, default):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (ValueError, TypeError):
        return default


def prepare(
    root, output, split_mode="album", seed=42, limit=None, cfg=None, chord_compatible=False
):
    root, output = Path(root).resolve(), Path(output)
    cfg = cfg or PreprocessConfig()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is not empty; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = make_tokenizer(cfg.resolution, use_chords=chord_compatible)
    tokenizer.save(output / "tokenizer.json")
    with (root / "pijama.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # Group albums and duplicate recording IDs before choosing any excerpts.
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    path_map = (
        json.loads((root / "paths.json").read_text(encoding="utf-8"))
        if (root / "paths.json").exists()
        else {}
    )
    keys, paths, hashes = {}, [], []
    for i, row in enumerate(rows):
        relative = row["midi_filepath"].replace("\\", "/")
        path = (root / path_map.get(relative, relative)).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Metadata MIDI path escapes dataset root")
        paths.append(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        hashes.append(digest)
        row_keys = []
        if split_mode == "album":
            row_keys.append(("album", row["artist"], row["album"]))
        if digest:
            row_keys.append(("sha256", digest))
        for field in ("acoust_id", "mb_recording_id"):
            if row.get(field) and row[field].lower() != "nan":
                row_keys.append((field, row[field]))
        for key in row_keys:
            if key in keys:
                parents[find(i)] = find(keys[key])
            else:
                keys[key] = i
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(find(i), []).append(i)
    splits = {}
    for members in groups.values():
        if split_mode == "official":
            values = {rows[i]["split"] for i in members}
            if len(values) != 1:
                raise ValueError("Duplicate recording crosses official splits; use --split album")
            split = values.pop()
        else:
            identity = min(f"{rows[i]['artist']}|{rows[i]['album']}" for i in members)
            fraction = (
                int(hashlib.sha256(f"{seed}|{identity}".encode()).hexdigest()[:8], 16) / 2**32
            )
            split = "train" if fraction < 0.8 else "val" if fraction < 0.9 else "test"
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split {split}")
        splits.update({i: split for i in members})
    records, skipped, seen = [], [], set()
    buffers = {split: [] for split in ("train", "val", "test")}
    offsets = {split: 0 for split in buffers}
    for i, row in enumerate(tqdm(rows[:limit], desc="Extracting and tokenizing")):
        digest = hashes[i]
        try:
            if digest in seen:
                raise ValueError("duplicate MIDI content")
            start = finite_number(row.get("performance_start_sec"), 0.0)
            end = finite_number(row.get("performance_end_sec"), None)
            score = preprocess_midi(paths[i], cfg, start, end)
            note_count = len(score.tracks[0].notes)
            if note_count < cfg.min_notes:
                raise ValueError(f"only {note_count} extracted notes")
            encoded = encode(tokenizer, score)
            if chord_compatible:
                encoded = add_no_chord_tokens(tokenizer, encoded)
            ids = [tokenizer["BOS_None"], *encoded, tokenizer["EOS_None"]]
            split = splits[i]
            records.append(
                {
                    "id": row.get("id", str(i)),
                    "source": row["midi_filepath"],
                    "artist": row["artist"],
                    "album": row["album"],
                    "split": split,
                    "sha256": digest,
                    "offset": offsets[split],
                    "length": len(ids),
                    "notes": note_count,
                }
            )
            buffers[split].append(np.asarray(ids, dtype=np.uint16))
            offsets[split] += len(ids)
            seen.add(digest)
        except (ValueError, OSError, RuntimeError, EOFError) as exc:
            skipped.append({"source": row["midi_filepath"], "reason": str(exc)})
    for split, arrays in buffers.items():
        (np.concatenate(arrays) if arrays else np.array([], dtype=np.uint16)).tofile(
            output / f"{split}.bin"
        )
    manifest = {
        "format_version": 1,
        "dtype": "uint16",
        "split_mode": split_mode,
        "seed": seed,
        "preprocessing": cfg.to_dict(),
        "vocab_size": len(tokenizer),
        "tokenizer_sha256": hashlib.sha256((output / "tokenizer.json").read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256((root / "pijama.csv").read_bytes()).hexdigest(),
        "timing": "Fixed 120-QPM grid preserving seconds; bars are NOT verified beats",
        "representation": (
            "REMI with Chord|NC before each note for conditioned-model pretraining"
            if chord_compatible
            else "REMI"
        ),
        "records": records,
        "skipped": skipped,
        "tokens": offsets,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"recordings": len(records), "skipped": len(skipped), "tokens": offsets}, indent=2
        )
    )
    if not records or not offsets["train"] or not offsets["val"]:
        raise ValueError(
            "Need nonempty train and val splits. Increase --limit or inspect manifest.json"
        )
