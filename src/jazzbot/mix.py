"""Build a source-balanced physical corpus from compatible prepared datasets."""

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def mix_corpora(inputs, weights, output):
    if len(inputs) < 2 or len(inputs) != len(weights):
        raise ValueError("Provide at least two inputs and one weight per input")
    if any(weight <= 0 for weight in weights):
        raise ValueError("Mixture weights must be positive")
    roots, output = [Path(value).resolve() for value in inputs], Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is not empty; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    manifests = [json.loads((root / "manifest.json").read_text(encoding="utf-8")) for root in roots]
    tokenizer_bytes = [(root / "tokenizer.json").read_bytes() for root in roots]
    if any(value != tokenizer_bytes[0] for value in tokenizer_bytes[1:]):
        raise ValueError("All mixture inputs must use the exact same tokenizer")
    shutil.copyfile(roots[0] / "tokenizer.json", output / "tokenizer.json")
    total_weight = sum(weights)
    weights = [value / total_weight for value in weights]
    records, offsets = [], {split: 0 for split in ("train", "val", "test")}
    token_counts = {split: 0 for split in offsets}
    for split in offsets:
        arrays = []
        for root, manifest, source_weight in zip(roots, manifests, weights, strict=True):
            source_records = [record for record in manifest["records"] if record["split"] == split]
            source_tokens = sum(record["length"] - 1 for record in source_records)
            data = np.fromfile(root / f"{split}.bin", dtype=np.uint16)
            base = offsets[split]
            arrays.append(data)
            for record in source_records:
                merged = dict(record)
                merged["offset"] = base + record["offset"]
                merged["dataset"] = manifest.get("dataset", root.name)
                merged["sampling_weight"] = (
                    source_weight * (record["length"] - 1) / max(1, source_tokens)
                )
                records.append(merged)
            offsets[split] += len(data)
        merged_data = np.concatenate(arrays) if arrays else np.array([], dtype=np.uint16)
        merged_data.tofile(output / f"{split}.bin")
        token_counts[split] = len(merged_data)
    manifest = {
        "format_version": 1,
        "dtype": "uint16",
        "dataset": "source-balanced mixture",
        "sources": [
            {
                "path": str(root),
                "weight": weight,
                "manifest_sha256": hashlib.sha256(
                    (root / "manifest.json").read_bytes()
                ).hexdigest(),
            }
            for root, weight in zip(roots, weights, strict=True)
        ],
        "preprocessing": manifests[0]["preprocessing"],
        "representation": "Source-balanced compatible token corpora",
        "vocab_size": manifests[0]["vocab_size"],
        "tokenizer_sha256": hashlib.sha256(tokenizer_bytes[0]).hexdigest(),
        "records": records,
        "skipped": [],
        "tokens": token_counts,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"sources": manifest["sources"], "tokens": token_counts}, indent=2))
    if not token_counts["train"] or not token_counts["val"]:
        raise ValueError("Need nonempty train and val splits")
    return output
