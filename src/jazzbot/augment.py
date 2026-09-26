"""Training-only REMI dynamics/articulation variations, without moving onsets."""

from bisect import bisect_left, bisect_right

import torch


class NoteAugmenter:
    def __init__(self, tokenizer, probability=0.0, velocity_bins=0, duration_scale=(1.0, 1.0)):
        self.probability = probability
        self.velocity_bins = velocity_bins
        self.duration_scale = duration_scale
        self.tokens = [tokenizer[i] for i in range(len(tokenizer))]
        self.velocities = sorted(
            (int(token.split("_")[1]), i)
            for i, token in enumerate(self.tokens)
            if token.startswith("Velocity_")
        )
        self.velocity_indices = {token_id: i for i, (_, token_id) in enumerate(self.velocities)}
        self.positions = {
            i: int(token.split("_")[1])
            for i, token in enumerate(self.tokens)
            if token.startswith("Position_")
        }
        # Position units per 4-quarter REMI frame. Read from the actual vocabulary.
        self.bar_width = max(self.positions.values()) + 1
        self.resolution = self.bar_width / 4
        self.bar_ids = {i for i, token in enumerate(self.tokens) if token.startswith("Bar_")}
        durations = []
        for i, token in enumerate(self.tokens):
            if token.startswith("Duration_"):
                beats, fraction, denominator = map(int, token.split("_")[1].split("."))
                durations.append(((beats + fraction / denominator) * self.resolution, i))
        self.durations = sorted(durations)
        self.duration_values = [value for value, _ in self.durations]
        self.duration_lookup = dict((token_id, value) for value, token_id in self.durations)

    def __call__(self, ids, generator):
        if self.probability == 0 or float(torch.rand((), generator=generator)) >= self.probability:
            return ids
        # One shift/factor per excerpt preserves relative accents and articulation.
        shift = int(
            torch.randint(-self.velocity_bins, self.velocity_bins + 1, (), generator=generator)
        )
        low, high = self.duration_scale
        scale = low + (high - low) * float(torch.rand((), generator=generator))
        result = ids.clone()
        sequence = ids.tolist()
        onsets, durations = [], []
        bar, onset = 0, None
        for index, token_id in enumerate(sequence):
            if token_id in self.bar_ids:
                bar += self.bar_width
            elif token_id in self.positions:
                onset = bar + self.positions[token_id]
                onsets.append((index, onset))
            elif token_id in self.duration_lookup:
                durations.append((index, onset, self.duration_lookup[token_id]))
        # Avoid clipping accents at the ends of the velocity vocabulary.
        indices = [self.velocity_indices[i] for i in sequence if i in self.velocity_indices]
        if indices:
            shift = max(-min(indices), min(shift, len(self.velocities) - 1 - max(indices)))
        for index, token_id in enumerate(sequence):
            if token_id in self.velocity_indices:
                result[index] = self.velocities[self.velocity_indices[token_id] + shift][1]
        onset_indices = [index for index, _ in onsets]
        for index, start, duration in durations:
            following = bisect_right(onset_indices, index)
            # Crops can start/end inside a note. Without both onsets, never lengthen it.
            cap = duration
            if start is not None and following < len(onsets):
                cap = onsets[following][1] - start
            limit = bisect_right(self.duration_values, cap) - 1
            if limit < 0:
                continue
            target = min(duration * scale, cap)
            right = min(bisect_left(self.duration_values, target), limit)
            candidates = {max(0, right - 1), right}
            nearest = min(candidates, key=lambda i: (abs(self.duration_values[i] - target), i))
            result[index] = self.durations[nearest][1]
        return result
