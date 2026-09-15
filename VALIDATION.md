# Local validation

Validated on Windows / Python 3.12 / PyTorch 2.14.0 / MidiTok 3.0.6.post1, using CPU.

- Test suite: **8 passed, 2 skipped**. CUDA and MPS backend tests skip when hardware is unavailable; they execute automatically on supported machines. The MIDI diagnostics regression test also passed after its JSON serialization fix.
- Ruff lint and formatting checks pass; `uv lock --check` passes.
- `uv build` produces the source distribution and wheel.
- Official Hawthorne archive downloaded and verified against the Zenodo MD5.
- Full preparation completed: 2,774 retained recordings; three excluded because their extracted lines contained fewer than 32 notes.

| Split | Recordings | Tokens |
|---|---:|---:|
| Train | 2,347 | 12,459,539 |
| Validation | 246 | 1,376,198 |
| Test | 181 | 1,038,494 |

The actual recording proportions differ from 80/10/10 because complete albums and linked recordings are assigned together. The manifest in `data/processed/manifest.json` records the exact assignment and extraction failures.

A small CPU model trained for eight steps on 40 actual PiJAMA recordings and resumed through step ten. Validation loss decreased from 6.4984 at step four to 6.4808 at step ten. This verifies the pipeline, not musical quality.

The production 7,548,928-parameter model completed two CPU optimizer steps with a 1,024-token context and validation. It used a one-sequence batch for this hardware check. These are not sufficient training steps for meaningful music.

Unconditional generation, an actual PiJAMA prompt continuation, inference export, and generation from that exported bundle all completed. Exported output and the real-prompt continuation both had zero overlapping note onsets when reloaded from MIDI. Local artifacts are under `runs/smoke`, `runs/production-check`, and `outputs`; they are intentionally ignored by Git.

CUDA mixed precision, M4 Max performance, long-run stability, and musical quality remain unverified here. Run the backend tests and a short training job on the target hardware before the full run. No listening-quality claim is made for the smoke outputs.
