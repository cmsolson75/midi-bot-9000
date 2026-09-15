import json

import pretty_midi
import torch

from jazzbot.generate import MonophonicGrammar, sample
from jazzbot.inspect import midi_stats
from jazzbot.midi import decode, encode, make_tokenizer, preprocess_midi


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
