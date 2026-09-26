"""Align an existing prepared pretraining corpus to the phrase-aware vocabulary."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .midi import load_tokenizer


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_phrase_pretraining(source, output, target_tokenizer):
    source, output = Path(source).resolve(), Path(output).resolve()
    target_tokenizer = Path(target_tokenizer).resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    if digest(source / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError("Source tokenizer does not match its manifest")
    before = load_tokenizer(source / "tokenizer.json")
    after = load_tokenizer(target_tokenizer)
    old_names = [before[i] for i in range(len(before))]
    new_names = {after[i] for i in range(len(after))}
    if "PhraseStart_None" in old_names or new_names != set(old_names) | {"PhraseStart_None"}:
        raise ValueError("Target must add only PhraseStart to the original vocabulary")
    old_config, new_config = before.config.to_dict(), after.config.to_dict()
    old_config.pop("special_tokens")
    new_config.pop("special_tokens")
    if old_config != new_config:
        raise ValueError("Tokenization settings differ beyond the new phrase token")
    provenance = {
        "source_manifest_sha256": digest(source / "manifest.json"),
        "source_token_files": {
            split: digest(source / f"{split}.bin") for split in ("train", "val", "test")
        },
        "target_tokenizer_sha256": digest(target_tokenizer),
    }
    if output.exists():
        if not (output / "manifest.json").is_file():
            raise ValueError(f"{output} exists without a completed manifest; use a new directory")
        existing = json.loads((output / "manifest.json").read_text())
        if existing.get("vocabulary_alignment") != provenance:
            raise ValueError("Existing aligned corpus uses different inputs; use a new directory")
        if digest(output / "tokenizer.json") != provenance["target_tokenizer_sha256"]:
            raise ValueError("Aligned tokenizer has changed")
        for split, expected in existing["token_file_sha256"].items():
            if digest(output / f"{split}.bin") != expected:
                raise ValueError(f"Aligned {split} tokens have changed")
        return output
    remap = np.array([after[name] for name in old_names], dtype=np.uint16)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as directory:
        temporary = Path(directory)
        checksums = {}
        for split in ("train", "val", "test"):
            path = source / f"{split}.bin"
            if path.stat().st_size != manifest["tokens"][split] * 2:
                raise ValueError(f"Source {split} token length differs from manifest")
            ids = np.fromfile(path, dtype=np.uint16)
            if ids.size and int(ids.max()) >= len(before):
                raise ValueError(f"Source {split} contains an unknown token")
            remap[ids].tofile(temporary / f"{split}.bin")
            checksums[split] = digest(temporary / f"{split}.bin")
        shutil.copyfile(target_tokenizer, temporary / "tokenizer.json")
        # Retain every recording, split, offset, source weight, and preprocessing field.
        manifest.update(
            vocab_size=len(after),
            tokenizer_sha256=provenance["target_tokenizer_sha256"],
            vocabulary_alignment=provenance,
            token_file_sha256=checksums,
            phrase_boundaries=False,
            representation=manifest.get("representation", "REMI")
            + "; phrase-compatible vocabulary, without invented phrase annotations",
        )
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2))
        temporary.rename(output)
    return output
