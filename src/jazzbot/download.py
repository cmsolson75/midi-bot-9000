"""Download the published Hawthorne MIDI release, checking its Zenodo checksum."""

import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

MIDI_URL = "https://zenodo.org/records/8354955/files/midi_hawthorne.zip?download=1"
MIDI_MD5 = "29b8ef8500f508abde4e5c8d76895f93"
METADATA_URL = "https://raw.githubusercontent.com/almostimplemented/PiJAMA/main/pijama.csv"


def fetch(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "midi-bot-9000/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        with tqdm(
            total=int(response.headers.get("Content-Length", 0)) or None,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as bar:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                bar.update(len(chunk))
    temporary.replace(destination)


def safe_extract(archive, destination, flatten=False):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (destination / member.filename).resolve()
            if (
                not target.is_relative_to(destination)
                or (member.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise ValueError(f"Unsafe archive entry: {member.filename}")
        if not flatten:
            source.extractall(destination)
            return {}
        destination.mkdir(parents=True, exist_ok=True)
        mapping = {}
        for member in source.infolist():
            if member.is_dir() or Path(member.filename).suffix.lower() not in {".mid", ".midi"}:
                continue
            name = hashlib.sha256(member.filename.encode()).hexdigest() + ".mid"
            with source.open(member) as src, (destination / name).open("wb") as dst:
                shutil.copyfileobj(src, dst)
            mapping["data/" + member.filename] = "files/" + name
        return mapping


def download(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "midi_hawthorne.zip"
    if not archive.exists():
        fetch(MIDI_URL, archive)
    with archive.open("rb") as handle:
        digest = hashlib.file_digest(handle, "md5").hexdigest()
    if digest != MIDI_MD5:
        raise ValueError(f"Checksum mismatch for {archive}; remove it and retry")
    mapping = safe_extract(archive, root / "files", flatten=True)
    (root / "paths.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    metadata = root / "pijama.csv"
    if not metadata.exists():
        fetch(METADATA_URL, metadata)
    provenance = {
        "midi_url": MIDI_URL,
        "midi_md5": digest,
        "metadata_url": METADATA_URL,
        "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
    }
    (root / "download.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"PiJAMA downloaded to {root.resolve()}")
