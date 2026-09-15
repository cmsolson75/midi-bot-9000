"""Memory-mapped recording-aware windows: no attention across unrelated performances."""

import json
from pathlib import Path

import numpy as np
import torch


class TokenCorpus:
    def __init__(self, root, split, context_length, pad_id):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.records = [r for r in self.manifest["records"] if r["split"] == split]
        if not self.records:
            raise ValueError(f"No {split} recordings in {root}")
        self.tokens = np.memmap(self.root / f"{split}.bin", dtype=np.uint16, mode="r")
        self.context, self.pad = context_length, pad_id
        self.weights = torch.tensor(
            [r.get("sampling_weight", r["length"] - 1) for r in self.records],
            dtype=torch.float64,
        )
        self.windows = [
            (i, start)
            for i, r in enumerate(self.records)
            for start in range(0, r["length"] - 1, context_length)
        ]

    def window(self, record_index, start):
        record = self.records[record_index]
        size = min(self.context + 1, record["length"] - start)
        offset = record["offset"] + start
        ids = torch.from_numpy(np.array(self.tokens[offset : offset + size], dtype=np.int64))
        x = torch.full((self.context,), self.pad, dtype=torch.long)
        y = torch.full((self.context,), -100, dtype=torch.long)
        x[: size - 1], y[: size - 1] = ids[:-1], ids[1:]
        return x, y

    def random_batch(
        self, batch_size, generator, pitch_lookup=None, transpose=0, chord_remaps=None
    ):
        choices = torch.multinomial(self.weights, batch_size, replacement=True, generator=generator)
        samples = []
        for index in choices.tolist():
            length = self.records[index]["length"]
            start = int(torch.randint(max(1, length - self.context), (1,), generator=generator))
            x, y = self.window(index, start)
            if transpose and pitch_lookup is not None:
                values = torch.cat((x, y[y >= 0]))
                pitches = pitch_lookup[values]
                pitches = pitches[pitches >= 0]
                if pitches.numel():
                    low = max(-transpose, 21 - int(pitches.min()))
                    high = min(transpose, 108 - int(pitches.max()))
                    shift = int(torch.randint(low, high + 1, (1,), generator=generator))
                    # Pitch IDs are mapped explicitly; no assumption of consecutive vocabulary IDs.
                    reverse = {int(p): i for i, p in enumerate(pitch_lookup.tolist()) if p >= 0}
                    remap = torch.arange(len(pitch_lookup))
                    for token_id, pitch in enumerate(pitch_lookup.tolist()):
                        if pitch >= 0 and pitch + shift in reverse:
                            remap[token_id] = reverse[pitch + shift]
                    x = remap[x]
                    valid = y >= 0
                    y[valid] = remap[y[valid]]
                    if chord_remaps is not None:
                        chord_remap = chord_remaps[shift]
                        x = chord_remap[x]
                        y[valid] = chord_remap[y[valid]]
            samples.append((x, y))
        return torch.stack([s[0] for s in samples]), torch.stack([s[1] for s in samples])

    def eval_batches(self, batch_size, max_batches=None):
        windows = self.windows
        if max_batches and len(windows) > max_batches * batch_size:
            # Fixed, evenly distributed windows, covering artists throughout the corpus.
            indices = np.linspace(0, len(windows) - 1, max_batches * batch_size, dtype=int)
            windows = [windows[i] for i in indices]
        for start in range(0, len(windows), batch_size):
            samples = [self.window(*item) for item in windows[start : start + batch_size]]
            yield torch.stack([s[0] for s in samples]), torch.stack([s[1] for s in samples])
