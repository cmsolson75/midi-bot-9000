import json

import mido
import pretty_midi
import torch

from jazzbot.generate import MonophonicGrammar, sample
from jazzbot.inspect import midi_stats
from jazzbot.midi import (
    decode,
    encode,
    load_tokenizer,
    make_tokenizer,
    preprocess_midi,
    strip_tempo_events,
)
from jazzbot.weimar import normalize_chord


def write_midi(path, offset=0):
    midi = pretty_midi.PrettyMIDI(initial_tempo=93)
    track = pretty_midi.Instrument(0)
    for i in range(40):
        onset = i * 0.2
        track.notes += [
            pretty_midi.Note(80 + i % 4, 60 + offset + i % 8, onset, onset + 0.3),
            pretty_midi.Note(70, 48, onset + 0.01, onset + 0.2),
        ]
    midi.instruments.append(track)
    midi.write(str(path))


def test_extraction_roundtrip_and_crop(tmp_path):
    path = tmp_path / "piano.mid"
    write_midi(path)
    # Crop between onsets: the MIDI fixture itself quantizes seconds to ticks.
    score = preprocess_midi(path, start=0.95, end=4.95)
    notes = score.tracks[0].notes
    assert len(notes) == 20
    assert all(note.pitch >= 60 for note in notes)
    assert all(a.end <= b.time for a, b in zip(notes, notes[1:]))
    tokenizer = make_tokenizer()
    restored = decode(tokenizer, encode(tokenizer, score))
    actual = restored.tracks[0].notes
    assert len(actual) == len(notes)
    assert [n.pitch for n in actual] == [n.pitch for n in notes]
    assert all(a.end <= b.time for a, b in zip(actual, actual[1:]))
    assert abs(actual[-1].end / restored.ticks_per_quarter - notes[-1].end / 480) < 0.05
    restored.dump_midi(tmp_path / "roundtrip.mid")
    stats = midi_stats(tmp_path / "roundtrip.mid")
    assert stats["overlapping_note_onsets"] == 0
    json.dumps(stats)
    assert len(pretty_midi.PrettyMIDI(str(tmp_path / "roundtrip.mid")).instruments[0].notes) == 20


def test_strip_tempo_events_preserves_note_ticks(tmp_path):
    path = tmp_path / "tempo.mid"
    write_midi(path)
    before = mido.MidiFile(path)
    before_notes = [
        (message.type, message.note, message.time)
        for track in before.tracks
        for message in track
        if message.type in {"note_on", "note_off"}
    ]

    strip_tempo_events(path)

    after = mido.MidiFile(path)
    assert not any(message.type == "set_tempo" for track in after.tracks for message in track)
    after_notes = [
        (message.type, message.note, message.time)
        for track in after.tracks
        for message in track
        if message.type in {"note_on", "note_off"}
    ]
    assert after_notes == before_notes


def test_grammar_never_creates_polyphony_or_partial_note_tuples():
    tokenizer = make_tokenizer()
    grammar = MonophonicGrammar(tokenizer, min_notes=10000)
    ids = [tokenizer["BOS_None"]]
    generator = torch.Generator().manual_seed(42)
    complete = 1
    for _ in range(400):
        token = sample(torch.zeros(len(tokenizer)), grammar.allowed(), top_k=0, generator=generator)
        grammar.consume(token)
        ids.append(token)
        if grammar.previous == "Duration":
            complete = len(ids)
    score = decode(tokenizer, ids[:complete])
    notes = score.tracks[0].notes
    assert len(notes) > 20
    assert all(a.end <= b.time for a, b in zip(notes, notes[1:]))
    # Inspect decoded raw output too: cleanup must not conceal grammar overlap bugs.
    raw = tokenizer.decode([ids[1:complete]])
    assert len(raw.tracks[0].notes) == len(notes)
    assert all(a.end <= b.time for a, b in zip(raw.tracks[0].notes, raw.tracks[0].notes[1:]))


def test_chord_tokenizer_roundtrip_grammar_and_normalization(tmp_path):
    tokenizer = make_tokenizer(use_chords=True)
    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    restored = load_tokenizer(path)
    assert len(restored) == len(tokenizer)
    assert restored["Chord|C:maj7_None"] == tokenizer["Chord|C:maj7_None"]
    assert normalize_chord("Dbj7") == "C#:maj7"
    assert normalize_chord("G7alt") == "G:dom7"
    assert normalize_chord("C-7") == "C:min7"
    grammar = MonophonicGrammar(restored)
    grammar.consume(restored["BOS_None"])
    grammar.consume(restored["Bar_None"])
    grammar.consume(restored["Position_0"])
    assert restored["Chord|C:maj7_None"] in grammar.allowed()
    grammar.consume(restored["Chord|C:maj7_None"])
    allowed = grammar.allowed()
    assert allowed
    assert all(restored[token].startswith("Pitch_") for token in allowed)
