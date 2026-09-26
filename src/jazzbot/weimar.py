"""Prepare beat-normalized, chord-conditioned Weimar Jazz Database solos."""

import bisect
import hashlib
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
from symusic import Note, Score, Tempo, TimeSignature, Track
from tqdm import tqdm

from .midi import CHORD_ROOTS, PreprocessConfig, encode, make_tokenizer, monophonize

_PITCH_CLASS = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}


def normalize_chord(symbol):
    """Collapse WJazzD symbols to a root plus a practical quality class."""
    value = (symbol or "").strip()
    if not value or value.upper() in {"NC", "N.C.", "?"}:
        return "NC"
    match = re.match(r"^([A-G](?:b|#)?)(.*)$", value)
    if not match or match.group(1) not in _PITCH_CLASS:
        return "NC"
    root, suffix = CHORD_ROOTS[_PITCH_CLASS[match.group(1)]], match.group(2).split("/", 1)[0]
    lower = suffix.lower()
    if "m7b5" in lower or "-7b5" in lower or "ø" in lower:
        quality = "hdim"
    elif "dim" in lower or lower.startswith("o"):
        quality = "dim"
    elif "+" in lower or "aug" in lower:
        quality = "aug"
    elif "sus" in lower:
        quality = "sus"
    elif "j7" in lower or "maj7" in lower:
        quality = "maj7"
    elif (lower.startswith("m") and not lower.startswith("maj")) or lower.startswith("-"):
        quality = "min7" if "7" in lower else "min"
    elif "7" in lower:
        quality = "dom7"
    else:
        quality = "maj"
    return f"{root}:{quality}"


def chord_token(tokenizer, chord):
    return tokenizer[f"Chord|{chord}_None"]


def _virtual_beat(onsets, value):
    index = bisect.bisect_right(onsets, value) - 1
    if index < 0:
        width = onsets[1] - onsets[0] if len(onsets) > 1 else 0.5
        return (value - onsets[0]) / max(width, 1e-3)
    if index + 1 < len(onsets):
        width = onsets[index + 1] - onsets[index]
    else:
        width = onsets[-1] - onsets[-2] if len(onsets) > 1 else 0.5
    return index + (value - onsets[index]) / max(width, 1e-3)


def _score_and_chords(connection, melid, cfg, phrase_boundaries=False):
    beats = connection.execute(
        "SELECT onset, bar, chord FROM beats WHERE melid=? ORDER BY onset", (melid,)
    ).fetchall()
    # Some transcriptions number the first complete bar as 0, others as 1.
    # Negative bars are count-ins and are the only rows we intentionally drop.
    start = next((i for i, row in enumerate(beats) if row[1] >= 0), None)
    if start is None or len(beats) - start < 2:
        raise ValueError("missing usable beat grid")
    beats = beats[start:]
    onsets = [float(row[0]) for row in beats]
    active, chords = "NC", []
    for _, _, raw_chord in beats:
        if raw_chord and raw_chord.strip():
            active = normalize_chord(raw_chord)
        chords.append(active)

    rows = connection.execute(
        "SELECT onset, pitch, duration, loud_med FROM melody WHERE melid=? ORDER BY onset, eventid",
        (melid,),
    ).fetchall()
    track = Track(name="Weimar chord-conditioned solo", program=0)
    grid = 480 // cfg.resolution
    retained = {}
    for index, (onset, pitch, duration, loudness) in enumerate(rows):
        pitch = int(round(pitch))
        if not cfg.min_pitch <= pitch <= cfg.max_pitch or duration <= 0:
            continue
        begin = round(_virtual_beat(onsets, float(onset)) * cfg.resolution) * grid
        finish = round(_virtual_beat(onsets, float(onset + duration)) * cfg.resolution) * grid
        if finish <= begin:
            finish = begin + 1
        velocity = max(1, min(127, round(loudness if loudness is not None else 80)))
        track.notes.append(Note(max(0, begin), max(1, finish - begin), pitch, velocity))
        retained[index] = (max(0, begin), pitch)
    if len(track.notes) < cfg.min_notes:
        raise ValueError(f"only {len(track.notes)} usable notes")
    score = Score(480)
    score.tempos = [Tempo(0, 120)]
    score.time_signatures = [TimeSignature(0, 4, 4)]
    score.tracks = [track]
    score = monophonize(score)
    phrase_positions = set()
    if phrase_boundaries:
        sections = connection.execute(
            "SELECT start, end FROM sections WHERE melid=? AND type='PHRASE' ORDER BY start",
            (melid,),
        ).fetchall()
        if not sections:
            raise ValueError("missing phrase annotations")
        surviving = {(note.time, note.pitch) for note in score.tracks[0].notes}
        for first, last in sections:
            # Section indices are zero-based, inclusive, relative to the original
            # melody; they are NOT global event IDs or filtered-note indices.
            if not 0 <= first <= last < len(rows):
                raise ValueError("phrase annotation outside melody bounds")
            for index in range(first, last + 1):
                identity = retained.get(index)
                if identity in surviving:
                    phrase_positions.add(identity[0] // grid)
                    break
        if not phrase_positions:
            raise ValueError("no phrase starts survive preprocessing")
    return score, chords, phrase_positions


def _insert_chords(tokenizer, ids, chords, resolution, phrase_positions=None):
    result, bar = [], -1
    for token_id in ids:
        token = tokenizer[token_id]
        result.append(token_id)
        if token.startswith("Bar_"):
            bar += 1
        elif token.startswith("Position_"):
            position = int(token.split("_", 1)[1])
            beat = max(0, bar * 4 + position // resolution)
            result.append(chord_token(tokenizer, chords[min(beat, len(chords) - 1)]))
            if phrase_positions and bar * 4 * resolution + position in phrase_positions:
                result.append(tokenizer["PhraseStart_None"])
    return result


def prepare_weimar(database, output, seed=42, limit=None, cfg=None, phrase_boundaries=False):
    database, output = Path(database).resolve(), Path(output)
    cfg = cfg or PreprocessConfig(min_pitch=36, max_pitch=100, min_notes=24)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"{output} is not empty; use a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = make_tokenizer(cfg.resolution, use_chords=True, use_phrases=phrase_boundaries)
    tokenizer.save(output / "tokenizer.json")
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    solos = connection.execute(
        "SELECT s.melid, s.compid, s.performer, s.title, s.instrument, s.style "
        "FROM solo_info s JOIN transcription_info t USING(melid) "
        "WHERE t.status='FINAL' ORDER BY s.melid"
    ).fetchall()
    if limit is not None:
        solos = solos[:limit]

    records, skipped = [], []
    buffers = {split: [] for split in ("train", "val", "test")}
    offsets = {split: 0 for split in buffers}
    try:
        for solo in tqdm(solos, desc="Preparing chord-conditioned solos"):
            identity = str(solo["compid"] if solo["compid"] is not None else solo["melid"])
            fraction = (
                int(hashlib.sha256(f"{seed}|{identity}".encode()).hexdigest()[:8], 16) / 2**32
            )
            split = "train" if fraction < 0.8 else "val" if fraction < 0.9 else "test"
            try:
                score, chords, phrase_positions = _score_and_chords(
                    connection, solo["melid"], cfg, phrase_boundaries
                )
                encoded = _insert_chords(
                    tokenizer, encode(tokenizer, score), chords, cfg.resolution, phrase_positions
                )
                if phrase_boundaries and encoded.count(tokenizer["PhraseStart_None"]) != len(
                    phrase_positions
                ):
                    raise ValueError("phrase starts lost during tokenization")
                ids = [tokenizer["BOS_None"], *encoded, tokenizer["EOS_None"]]
                records.append(
                    {
                        "id": solo["melid"],
                        "composition_id": solo["compid"],
                        "performer": solo["performer"],
                        "title": solo["title"],
                        "instrument": solo["instrument"],
                        "style": solo["style"],
                        "split": split,
                        "offset": offsets[split],
                        "length": len(ids),
                        "notes": len(score.tracks[0].notes),
                        **({"phrase_starts": len(phrase_positions)} if phrase_boundaries else {}),
                    }
                )
                buffers[split].append(np.asarray(ids, dtype=np.uint16))
                offsets[split] += len(ids)
            except (ValueError, RuntimeError) as exc:
                skipped.append({"id": solo["melid"], "reason": str(exc)})
    finally:
        connection.close()
    for split, arrays in buffers.items():
        (np.concatenate(arrays) if arrays else np.array([], dtype=np.uint16)).tofile(
            output / f"{split}.bin"
        )
    tokenizer_hash = hashlib.sha256((output / "tokenizer.json").read_bytes()).hexdigest()
    manifest = {
        "format_version": 1,
        "dtype": "uint16",
        "dataset": "Weimar Jazz Database v2.1 / DB v2.2",
        "license": "ODbL",
        "split_mode": "composition",
        "seed": seed,
        "preprocessing": cfg.to_dict(),
        "representation": "Beat-normalized REMI with active chord before every note",
        "vocab_size": len(tokenizer),
        "tokenizer_sha256": tokenizer_hash,
        "database_sha256": hashlib.sha256(database.read_bytes()).hexdigest(),
        "records": records,
        "skipped": skipped,
        "tokens": offsets,
    }
    if phrase_boundaries:
        manifest["phrase_boundaries"] = True
        manifest["representation"] += "; PhraseStart before annotated phrase-initial pitches"
        manifest["phrase_annotation_policy"] = (
            "Zero-based inclusive melody indices; marker moves to first surviving note "
            "inside the same phrase. Empty phrases omitted; colliding starts merged."
        )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"recordings": len(records), "skipped": len(skipped), "tokens": offsets}, indent=2
        )
    )
    if not records or not offsets["train"] or not offsets["val"]:
        raise ValueError("Need nonempty train and val splits")
    return output
