"""Small, interpretable MIDI diagnostics for data inspection and listening experiments."""

import math
from collections import Counter

import pretty_midi


def midi_stats(path):
    midi = pretty_midi.PrettyMIDI(str(path))
    notes = sorted(
        (n for t in midi.instruments if not t.is_drum for n in t.notes),
        key=lambda n: (n.start, n.pitch),
    )
    if not notes:
        return {"notes": 0}
    counts = Counter(n.pitch % 12 for n in notes)
    entropy = -sum((n / len(notes)) * math.log2(n / len(notes)) for n in counts.values())
    duration = max(n.end for n in notes) - notes[0].start
    end = -1.0
    overlaps = 0
    for note in notes:
        overlaps += int(note.start < end - 1e-5)
        end = max(end, note.end)
    return {
        "notes": len(notes),
        "duration_seconds": duration,
        "notes_per_second": len(notes) / max(duration, 1e-9),
        "pitch_min": min(n.pitch for n in notes),
        "pitch_max": max(n.pitch for n in notes),
        "velocity_min": min(n.velocity for n in notes),
        "velocity_max": max(n.velocity for n in notes),
        "pitch_class_entropy_bits": entropy,
        "overlapping_note_onsets": overlaps,
    }
