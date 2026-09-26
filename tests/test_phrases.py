import hashlib
import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest
import torch

from jazzbot.augment import NoteAugmenter
from jazzbot.generate import MonophonicGrammar, generate
from jazzbot.midi import PreprocessConfig, decode, load_tokenizer, make_tokenizer
from jazzbot.train import train
from jazzbot.weimar import _score_and_chords, prepare_weimar


def make_weimar(path):
    c = sqlite3.connect(path)
    c.executescript("""
        CREATE TABLE melody(eventid INTEGER, melid INTEGER, onset REAL, pitch REAL,
                            duration REAL, loud_med REAL);
        CREATE TABLE beats(melid INTEGER, onset REAL, bar INTEGER, chord TEXT);
        CREATE TABLE sections(melid INTEGER, type TEXT, start INTEGER, end INTEGER);
        CREATE TABLE solo_info(melid INTEGER, compid INTEGER, performer TEXT, title TEXT,
                               instrument TEXT, style TEXT);
        CREATE TABLE transcription_info(melid INTEGER, status TEXT);
    """)
    selected = {}
    for i in range(1, 200):
        value = int(hashlib.sha256(f"42|{i}".encode()).hexdigest()[:8], 16) / 2**32
        split = "train" if value < 0.8 else "val" if value < 0.9 else "test"
        selected.setdefault(split, i)
    for melid in selected.values():
        c.execute(
            "INSERT INTO solo_info VALUES(?,?,?,?,?,?)",
            (melid, melid, "Player", "Tune", "ts", "BEBOP"),
        )
        c.execute("INSERT INTO transcription_info VALUES(?, 'FINAL')", (melid,))
        for i in range(32):
            c.execute("INSERT INTO beats VALUES(?,?,?,?)", (melid, i * 0.5, i // 4, "C7"))
        for i in range(48):
            # Drop the first note of two phrases and all of the third phrase.
            pitch = 20 if i in (0, 12) or 24 <= i < 36 else 60 + i % 8
            c.execute(
                "INSERT INTO melody VALUES(?,?,?,?,?,?)",
                (melid * 1000 + i, melid, i * 0.25, pitch, 0.2, 80),
            )
        for i in range(4):
            c.execute("INSERT INTO sections VALUES(?,'PHRASE',?,?)", (melid, i * 12, i * 12 + 11))
    c.commit()
    return c, selected


def test_phrase_annotation_mapping_and_unchanged_notes(tmp_path):
    database = tmp_path / "weimar.db"
    c, ids = make_weimar(database)
    cfg = PreprocessConfig(min_pitch=36, max_pitch=100, min_notes=4)
    _, _, positions = _score_and_chords(c, ids["train"], cfg, True)
    assert positions == {12, 156, 432}  # Original indices 1, 13, 36 at half-beat spacing.
    c.execute("UPDATE sections SET end=10000 WHERE melid=? AND start=0", (ids["train"],))
    with pytest.raises(ValueError, match="outside melody bounds"):
        _score_and_chords(c, ids["train"], cfg, True)
    c.rollback()
    c.close()
    plain, phrases = tmp_path / "plain", tmp_path / "phrases"
    prepare_weimar(database, plain, cfg=cfg)
    prepare_weimar(database, phrases, cfg=cfg, phrase_boundaries=True)
    a, b = load_tokenizer(plain / "tokenizer.json"), load_tokenizer(phrases / "tokenizer.json")
    assert len(b) == len(a) + 1
    for split in ("train", "val", "test"):
        original = np.fromfile(plain / f"{split}.bin", dtype=np.uint16).tolist()
        marked = np.fromfile(phrases / f"{split}.bin", dtype=np.uint16).tolist()
        assert [a[i] for i in original] == [b[i] for i in marked if b[i] != "PhraseStart_None"]
        assert marked.count(b["PhraseStart_None"]) == 3
        notes_a, notes_b = decode(a, original).tracks[0].notes, decode(b, marked).tracks[0].notes
        assert [(n.time, n.duration, n.pitch) for n in notes_a] == [
            (n.time, n.duration, n.pitch) for n in notes_b
        ]
        augmented = NoteAugmenter(b, 1, 6, (0.65, 1.35))(
            torch.tensor(marked), torch.Generator().manual_seed(42)
        )
        assert (augmented == b["PhraseStart_None"]).nonzero().tolist() == (
            torch.tensor(marked) == b["PhraseStart_None"]
        ).nonzero().tolist()


def test_phrase_generation_grammar():
    tokenizer = make_tokenizer(use_chords=True, use_phrases=True)
    grammar = MonophonicGrammar(tokenizer)
    for name in ("BOS_None", "Bar_None", "Position_0", "Chord|C:maj_None"):
        grammar.consume(tokenizer[name])
    marker = tokenizer["PhraseStart_None"]
    assert marker in grammar.allowed()
    assert tokenizer["Pitch_60"] in grammar.allowed()
    grammar.consume(marker)
    assert marker not in grammar.allowed()
    assert all(tokenizer[i].startswith("Pitch_") for i in grammar.allowed())
    assert grammar.phrase_starts == [{"note_index": 0, "beat": 0.0}]
    grammar.consume(tokenizer["Pitch_60"])
    assert marker not in grammar.allowed()


def test_phrase_scratch_resume_export_and_generation(tmp_path):
    database = tmp_path / "weimar.db"
    c, _ = make_weimar(database)
    c.close()
    data = tmp_path / "data"
    prepare_weimar(database, data, cfg=PreprocessConfig(min_notes=4), phrase_boundaries=True)
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
                    "dropout": 0.1,
                },
                "batch_size": 1,
                "accumulation_steps": 1,
                "max_steps": 2,
                "eval_interval": 1,
                "eval_batches": 1,
                "eval_train_batches": 1,
                "warmup_steps": 1,
                "device": "cpu",
                "augmentation_probability": 0.9,
                "velocity_shift_bins": 6,
                "duration_scale_min": 0.65,
                "duration_scale_max": 1.35,
            }
        )
    )
    run = tmp_path / "run"
    train(data, run, config, max_steps=1)
    train(data, run, config, resume=run / "last.pt")
    state = torch.load(run / "last.pt", map_location="cpu", weights_only=True)
    assert state["phrase_boundaries"] is True
    assert state["initialized_from"] is None
    portable = tmp_path / "portable"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jazzbot",
            "export",
            "--checkpoint",
            str(run / "last.pt"),
            "--output",
            str(portable),
        ],
        check=True,
    )
    exported = torch.load(portable / "model.pt", map_location="cpu", weights_only=True)
    assert exported["phrase_boundaries"] is True
    assert "optimizer" not in exported
    output = tmp_path / "solo.mid"
    score = generate(
        portable / "model.pt",
        output,
        chords="Dm7,G7,Cmaj7,Cmaj7",
        min_notes=8,
        max_new_tokens=128,
        device="cpu",
        strip_tempo=True,
    )
    notes = score.tracks[0].notes
    assert len(notes) >= 8
    assert all(a.end <= b.time for a, b in zip(notes, notes[1:]))
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert "phrase_starts" in metadata
    assert all(s["note_index"] < len(notes) for s in metadata["phrase_starts"])
