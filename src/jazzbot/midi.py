"""Shared preprocessing for both training MIDI and continuation prompts.

PiJAMA has no verified beat grid. We preserve seconds on a fixed 120-QPM grid;
REMI bars are computational frames, not inferred musical measures.
"""

from dataclasses import asdict, dataclass
from pathlib import Path

import pretty_midi
from miditok import REMI, TokenizerConfig
from symusic import Note, Score, Tempo, TimeSignature, Track


@dataclass
class PreprocessConfig:
    min_pitch: int = 48
    max_pitch: int = 96
    min_velocity: int = 20
    min_duration: float = 0.05
    onset_tolerance: float = 0.03
    min_notes: int = 32
    resolution: int = 24

    def to_dict(self):
        return asdict(self)


def make_tokenizer(resolution=24):
    return REMI(
        TokenizerConfig(
            pitch_range=(21, 109),
            beat_res={(0, 16): resolution},
            num_velocities=32,
            special_tokens=["PAD", "BOS", "EOS"],
            use_chords=False,
            use_rests=False,
            use_tempos=False,
            use_time_signatures=False,
            use_programs=False,
        )
    )


def load_tokenizer(path):
    return REMI(params=Path(path))


def monophonize(score):
    """Choose highest pitch at each onset, then clip overlaps without resuming notes."""
    notes = sorted(
        (n for t in score.tracks if not t.is_drum for n in t.notes),
        key=lambda n: (n.time, -n.pitch, -n.velocity),
    )
    result = []
    for note in notes:
        if result and result[-1].time == note.time:
            continue
        if result and result[-1].end > note.time:
            result[-1].duration = note.time - result[-1].time
        result.append(Note(note.time, max(1, note.duration), note.pitch, note.velocity))
    track = Track(name="Jazz solo", program=0)
    track.notes = result
    score.tracks = [track]
    return score


def preprocess_midi(path, cfg=None, start=0.0, end=None):
    cfg = cfg or PreprocessConfig()
    midi = pretty_midi.PrettyMIDI(str(path))
    end = midi.get_end_time() if end is None else end
    candidates = sorted(
        (
            n
            for instrument in midi.instruments
            if not instrument.is_drum
            for n in instrument.notes
            if start <= n.start < end
            and cfg.min_pitch <= n.pitch <= cfg.max_pitch
            and n.velocity >= cfg.min_velocity
            and n.end - n.start >= cfg.min_duration
        ),
        key=lambda n: n.start,
    )
    selected = []
    index = 0
    while index < len(candidates):
        anchor = candidates[index].start
        group = []
        while index < len(candidates) and candidates[index].start <= anchor + cfg.onset_tolerance:
            group.append(candidates[index])
            index += 1
        selected.append(max(group, key=lambda n: (n.pitch, n.velocity)))
    score = Score(480)
    score.tempos = [Tempo(0, 120)]
    score.time_signatures = [TimeSignature(0, 4, 4)]
    track = Track(name="Extracted upper voice", program=0)
    # Quantize before clipping: quantization must not reintroduce polyphony.
    grid = score.ticks_per_quarter // cfg.resolution
    if grid < 1 or score.ticks_per_quarter % cfg.resolution:
        raise ValueError("resolution must divide 480")
    for note in selected:
        onset = max(0, round((note.start - start) * 960 / grid) * grid)
        offset = round((min(note.end, end) - start) * 960 / grid) * grid
        if offset > onset:
            track.notes.append(Note(onset, offset - onset, note.pitch, note.velocity))
    score.tracks = [track]
    return monophonize(score)


def encode(tokenizer, score):
    sequences = tokenizer.encode(score)
    sequence = sequences[0] if isinstance(sequences, list) else sequences
    return list(sequence.ids)


def decode(tokenizer, ids):
    special = {tokenizer["PAD_None"], tokenizer["BOS_None"], tokenizer["EOS_None"]}
    clean = [int(i) for i in ids if i not in special]
    return monophonize(tokenizer.decode([clean]))
