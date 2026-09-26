"""Read a running or completed training log without loading model weights."""

import argparse
import json
from pathlib import Path


def summarize(run):
    path = Path(run) / "metrics.jsonl"
    if not path.exists():
        print("No metrics yet; wait for the first logging interval.")
        return
    events = {}
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1:
                break  # A live writer may not have finished its final line.
            raise
        if "val_loss" in event:
            events[event["step"]] = event
    rows = sorted(events.values(), key=lambda event: event["step"])
    if not rows:
        print("No validation results yet.")
        return
    best = min(rows, key=lambda event: event["val_loss"])
    last = rows[-1]
    print(" step   augmented-train   clean-train   validation   clean-gap")
    for event in rows:
        clean = event.get("clean_train_loss")
        clean_text = f"{clean:11.4f}" if clean is not None else "        n/a"
        gap_text = f"{event['val_loss'] - clean:9.4f}" if clean is not None else "      n/a"
        print(
            f"{event['step']:5d}   {event['train_loss']:15.4f}   {clean_text}"
            f"   {event['val_loss']:10.4f}   {gap_text}"
        )
    print(f"Best validation: {best['val_loss']:.4f} at step {best['step']}.")
    print(
        f"Latest validation: {last['val_loss']:.4f} at step {last['step']} "
        f"({last['val_loss'] - best['val_loss']:+.4f} versus best)."
    )
    print(
        "Look for sustained validation worsening while clean training loss falls; "
        "one worse evaluation is not enough to establish overfitting."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="Run directory containing metrics.jsonl")
    summarize(parser.parse_args().run)
