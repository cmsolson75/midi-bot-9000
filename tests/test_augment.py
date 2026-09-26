import json

import pytest
import torch
from symusic import Note, Score, Track

from jazzbot.augment import NoteAugmenter
from jazzbot.config import TrainConfig
from jazzbot.midi import add_no_chord_tokens, encode, make_tokenizer


def fixture_tokens():
    tokenizer = make_tokenizer(use_chords=True)
    score = Score(480)
    track = Track()
    # Includes tightly packed notes, bar crossings and a long rest.
    track.notes = [
        Note(time, duration, 60 + i, velocity)
        for i, (time, duration, velocity) in enumerate(
            [
                (0, 240, 64),
                (240, 120, 80),
                (480, 400, 72),
                (1800, 240, 64),
                (2040, 120, 88),
                (4800, 480, 72),
            ]
        )
    ]
    score.tracks = [track]
    ids = add_no_chord_tokens(tokenizer, encode(tokenizer, score))
    return tokenizer, torch.tensor(ids)


def raw_notes(tokenizer, ids):
    clean = [i for i in ids.tolist() if not tokenizer[i].startswith("Chord|")]
    return tokenizer.decode([clean]).tracks[0].notes


def test_augmentation_preserves_onsets_harmony_accents_and_monophony():
    tokenizer, ids = fixture_tokens()
    original = raw_notes(tokenizer, ids)
    augmenter = NoteAugmenter(tokenizer, 1, 6, (0.65, 1.35))
    changed_durations, changed_velocities = False, False
    for seed in range(40):
        result = augmenter(ids, torch.Generator().manual_seed(seed))
        actual = raw_notes(tokenizer, result)
        assert len(actual) == len(original)
        assert [(n.time, n.pitch) for n in actual] == [(n.time, n.pitch) for n in original]
        assert all(a.end <= b.time for a, b in zip(actual, actual[1:]))
        assert all(n.duration > 0 for n in actual)
        assert actual[-1].duration <= original[-1].duration
        shifts = {
            augmenter.velocity_indices[int(new)] - augmenter.velocity_indices[int(old)]
            for old, new in zip(ids, result)
            if int(old) in augmenter.velocity_indices
        }
        assert len(shifts) == 1
        for old, new in zip(ids, result):
            if not tokenizer[int(old)].startswith(("Velocity_", "Duration_")):
                assert old == new
        changed_durations |= any(a.duration != b.duration for a, b in zip(actual, original))
        changed_velocities |= any(a.velocity != b.velocity for a, b in zip(actual, original))
    assert changed_durations and changed_velocities
    assert torch.equal(ids, fixture_tokens()[1])


def test_partial_crops_and_reproducibility():
    tokenizer, ids = fixture_tokens()
    augmenter = NoteAugmenter(tokenizer, 1, 6, (1.35, 1.35))
    start = next(i for i, token in enumerate(ids) if tokenizer[int(token)].startswith("Duration_"))
    cropped = ids[start:-2]
    first = augmenter(cropped, torch.Generator().manual_seed(19))
    second = augmenter(cropped, torch.Generator().manual_seed(19))
    assert torch.equal(first, second)
    assert augmenter.duration_lookup[int(first[0])] <= augmenter.duration_lookup[int(cropped[0])]
    generator = torch.Generator().manual_seed(7)
    rng = generator.get_state().clone()
    assert torch.equal(NoteAugmenter(tokenizer)(ids, generator), ids)
    assert torch.equal(generator.get_state(), rng)


@pytest.mark.parametrize(
    "values",
    [
        {"augmentation_probability": 1.1},
        {"velocity_shift_bins": -1},
        {"duration_scale_min": 0},
        {"duration_scale_max": 0.5},
        {"duration_scale_max": float("inf")},
        {"eval_train_batches": -1},
    ],
)
def test_invalid_augmentation_config(tmp_path, values):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError):
        TrainConfig.load(path)
