# MIDI Bot 9000: Collaborator Usage Guide

This bundle generates monophonic jazz MIDI. It contains three 7.6M-parameter
models, the matching tokenizers, and an installable Python wheel. It does not need
the training repository or datasets.

## Setup

1. Install Python 3.11, 3.12, or 3.13.
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
3. Extract the zip and open a terminal in the extracted
   `midi-bot-9000-collaborator` directory.

The first command may download PyTorch and the remaining runtime dependencies.
Later commands reuse uv's cached environment. Apple Silicon uses MPS automatically;
other machines use CUDA when the installed PyTorch build supports it, otherwise CPU.
This model is small enough for CPU inference.

## Choose a model

| Checkpoint | Use it for | Chord chart support |
|---|---|---|
| `models/pretrained-step5000/model.pt` | Freer, denser, surprising lines and MIDI continuation | No |
| `models/finetune-best-step900/model.pt` | Best held-out result and the safest chord-aware default | Yes |
| `models/finetune-last-step2200/model.pt` | More memorized experimental alternative; audition it rather than assuming it is worse | Yes |

The step-900 checkpoint had the best validation loss. Step 2,200 had overfit by
that metric, but it can still produce musically useful phrases. The exact
step-2,000 weights were overwritten during training and are not in this bundle.

Each model must remain beside its own `tokenizer.json`. Do not send or move a
`model.pt` by itself.

## Recommended first generation

Run the best chord-conditioned model over a four-chord progression:

```sh
uvx --from ./midi_bot_9000-0.1.0-py3-none-any.whl jazzbot generate \
  --checkpoint ./models/finetune-best-step900/model.pt \
  --output ./best-take.mid \
  --chords 'Amaj7(13),F#maj/G#,F#min9,C#min9' \
  --beats-per-chord 4 \
  --max-new-tokens 1152 \
  --min-notes 72 \
  --temperature 0.92 \
  --top-k 40 \
  --top-p 0.96 \
  --seed 29 \
  --strip-tempo \
  --device auto
```

The command creates `best-take.mid` and `best-take.json`. The JSON sidecar records
the checkpoint, normalized chords, sampling settings, seed, and generated tokens.

To audition the more memorized model, change only the checkpoint and output:

```sh
uvx --from ./midi_bot_9000-0.1.0-py3-none-any.whl jazzbot generate \
  --checkpoint ./models/finetune-last-step2200/model.pt \
  --output ./memorized-take.mid \
  --chords 'Amaj7(13),F#maj/G#,F#min9,C#min9' \
  --beats-per-chord 4 \
  --max-new-tokens 1152 \
  --min-notes 72 \
  --temperature 0.92 \
  --top-k 40 \
  --top-p 0.96 \
  --seed 29 \
  --strip-tempo \
  --device auto
```

## Use the pretrained model

The pretrained model learned from the mixed PiJAMA/Parker corpus before Weimar
fine-tuning. It often sounds wilder and more energetic. It never learned actual
chord tokens, so do **not** give it `--chords`. Generate freely and place the MIDI
over harmony afterward in a DAW:

```sh
uvx --from ./midi_bot_9000-0.1.0-py3-none-any.whl jazzbot generate \
  --checkpoint ./models/pretrained-step5000/model.pt \
  --output ./pretrained-take.mid \
  --max-new-tokens 1152 \
  --min-notes 72 \
  --temperature 0.92 \
  --top-k 40 \
  --top-p 0.96 \
  --seed 29 \
  --strip-tempo \
  --device auto
```

## Generate several takes

Changing the seed is the best first source of variety. Generate several takes with
the same musical settings, then curate by ear:

```sh
for seed in 11 22 33 44 55 66 77 88; do
  uvx --from ./midi_bot_9000-0.1.0-py3-none-any.whl jazzbot generate \
    --checkpoint ./models/finetune-best-step900/model.pt \
    --output "./take-${seed}.mid" \
    --chords 'Amaj7(13),F#maj/G#,F#min9,C#min9' \
    --beats-per-chord 4 \
    --max-new-tokens 1152 \
    --min-notes 72 \
    --temperature 0.92 \
    --top-k 40 \
    --top-p 0.96 \
    --seed "${seed}" \
    --strip-tempo \
    --device auto
done
```

This loop works in Bash and Zsh. On Windows PowerShell, run the single-generation
command repeatedly with a different `--seed`.

## Continue an existing MIDI phrase

All three models can continue a MIDI prompt. For the pretrained model, omit the
chord arguments. For a conditioned continuation:

```sh
uvx --from ./midi_bot_9000-0.1.0-py3-none-any.whl jazzbot generate \
  --checkpoint ./models/finetune-best-step900/model.pt \
  --prompt ./opening-phrase.mid \
  --prompt-seconds 8 \
  --output ./continued-take.mid \
  --chords 'Dm7,G7,Cmaj7,A7' \
  --beats-per-chord 4 \
  --max-new-tokens 1152 \
  --min-notes 64 \
  --temperature 0.92 \
  --top-k 40 \
  --top-p 0.96 \
  --seed 29 \
  --strip-tempo \
  --device auto
```

The prompt extractor keeps one upper monophonic line, removes accompaniment and
pedal information, and uses only the first `--prompt-seconds`. A clean monophonic
prompt is most predictable.

## What every generation option means

| Option | Meaning | Good default |
|---|---|---|
| `--checkpoint` | Model to use. Its tokenizer must be in the same directory. | Step 900 for chords; step 5,000 for free generation |
| `--output` | Destination MIDI path. A JSON report is written beside it. | A new descriptive filename per take |
| `--chords` | Comma-separated chord chart forced into a conditioned model. | Omit for pretrained; quote the entire chart |
| `--beats-per-chord` | Duration shared by every supplied chord, in quarter-note beats. | `4` |
| `--max-new-tokens` | Hard generation budget. One note consumes several tokens, so this is not a note count. | `1152`; use `2048` for longer takes |
| `--min-notes` | Prevents EOS until at least this many new notes are complete. The token budget still wins. | `72`; use `48` for shorter ideas |
| `--temperature` | Higher values flatten probabilities and increase surprise. Very high values weaken coherence. | `0.92` |
| `--top-k` | Samples only among the K most likely legal tokens at each step. | `40` |
| `--top-p` | Also restricts sampling to the smallest probability mass reaching P. | `0.96` |
| `--seed` | Reproducible sampling stream. Different seeds produce different takes. | Try `11`, `29`, `33`, `55`, `77` |
| `--strip-tempo` | Removes the 120 BPM metadata event so the DAW keeps its project tempo. | Recommended for DAW import |
| `--device` | `auto` selects MPS, CUDA, or CPU. | `auto` |
| `--prompt` | Optional MIDI phrase to continue. | Clean monophonic MIDI |
| `--prompt-seconds` | Uses only this many seconds from the beginning of the prompt. | `8` |

Suggested sampling personalities:

| Character | Temperature | Top K | Top P |
|---|---:|---:|---:|
| Focused / inside | `0.78` | `24` | `0.90` |
| Balanced | `0.92` | `40` | `0.96` |
| Adventurous | `1.02` | `64` | `0.98` |

Generate multiple seeds before raising temperature. Values much above `1.1` tend
to trade phrasing and harmonic clarity for randomness.

## Chord-chart behavior and limitations

- The chord chart repeats if the solo outlasts one pass.
- Every chord currently has the same `--beats-per-chord` duration.
- Supported internal families are major, minor, dominant seventh, major seventh,
  minor seventh, half-diminished, diminished, augmented, and suspended.
- Extensions and slash basses are simplified. For example, `Amaj7(13)` becomes the
  A-major-seventh family and `F#maj/G#` becomes F-sharp major.
- Use an explicit `min7` spelling when you want the minor-seventh family; a symbol
  such as `F#min9` currently simplifies to the minor family.
- MIDI chord-track input and variable chord durations are not implemented yet.
- Output is a monophonic solo, not rendered audio or polyphonic accompaniment.

## Tempo and rhythm

Training uses a normalized 120-quarter-notes-per-minute grid. Without
`--strip-tempo`, that tempo event is embedded in the MIDI and some DAWs offer to
replace the project tempo with it. With `--strip-tempo`, the file follows the DAW's
existing tempo; tick positions are unchanged, so wall-clock playback speed changes
when the project is not 120 BPM.

The pretrained model can produce unusually dense or loose timing. That is part of
this experimental checkpoint, not a MIDI corruption. Quantizing selectively in the
DAW or trying several seeds is currently the fastest way to shape its groove.

## Troubleshooting

- **`uvx` or `uv` is not found:** install uv, restart the terminal, and retry.
- **Wheel or model not found:** run the command from the extracted bundle root.
- **Tokenizer mismatch/missing:** restore the `tokenizer.json` beside that model.
- **First run seems slow:** uv may be downloading and installing PyTorch once.
- **MPS/CUDA problem:** retry with `--device cpu`; inference is still practical.
- **Solo is too short:** increase `--min-notes` and `--max-new-tokens` together.
- **Solo is chaotic:** lower temperature to `0.78`, Top K to `24`, and Top P to
  `0.90`, then try a new seed.
- **Solo is too conservative:** try several seeds, then use the adventurous preset.
- **DAW changes tempo:** confirm the generation command includes `--strip-tempo`.

