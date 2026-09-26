"""Download and verify the datasets used by the final two-stage experiment."""

import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from jazzbot.download import download, fetch
from jazzbot.folder import prepare_midi_directory
from jazzbot.mix import mix_corpora
from jazzbot.phrase_pretrain import digest
from jazzbot.prepare import prepare
from jazzbot.weimar import prepare_weimar


def manifest_digest(path):
    value = json.loads(Path(path).read_text())
    # Mixture manifests record absolute input paths, which vary between machines.
    for source in value.get("sources", []):
        source.pop("path", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verify_file(path, expected):
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f"{path} differs from the final experiment's data; refusing to train")


def verify_corpus(root, spec):
    for name, expected in spec["files"].items():
        verify_file(root / name, expected)
    if manifest_digest(root / "manifest.json") != spec["manifest_sha256"]:
        raise ValueError(f"{root}: splits or sampling metadata differ from the final experiment")
    print(f"Verified final-training data: {root}", flush=True)


def main():
    spec = json.loads(Path("configs/final-data.json").read_text())
    pijama = Path("data/pijama-chord-compatible")
    parker = Path("data/parker-chord-compatible")
    mixture = Path("data/pretrain-mixture")
    phrases = Path("data/weimar-phrases")
    if not (pijama / "manifest.json").exists():
        download("data/pijama")
        verify_file(Path("data/pijama/pijama.csv"), spec["pijama_metadata_sha256"])
        prepare("data/pijama", pijama, chord_compatible=True)
    verify_corpus(pijama, spec["corpora"][pijama.name])

    if not (parker / "manifest.json").exists():
        snapshot_download(
            "xavriley/CharlieParkerAlignedOmnibook",
            repo_type="dataset",
            revision=spec["parker_revision"],
            local_dir="data/parker",
            allow_patterns=["README.md", "midi_performance_aligned/*.mid"],
        )
        prepare_midi_directory(
            "data/parker/midi_performance_aligned", parker, "parker-aligned-omnibook"
        )
    verify_corpus(parker, spec["corpora"][parker.name])

    if not (mixture / "manifest.json").exists():
        mix_corpora([pijama, parker], [0.9, 0.1], mixture)
    verify_corpus(mixture, spec["corpora"][mixture.name])

    if not (phrases / "manifest.json").exists():
        database = Path("data/weimar/wjazzd.db")
        if not database.exists():
            fetch("https://jazzomat.hfm-weimar.de/download/downloads/wjazzd.db", database)
        verify_file(database, spec["weimar_database_sha256"])
        prepare_weimar(database, phrases, phrase_boundaries=True)
    verify_corpus(phrases, spec["corpora"][phrases.name])


if __name__ == "__main__":
    main()
