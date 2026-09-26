# MIDI Bot 9000

A small Transformer that generates monophonic jazz MIDI. This is the source for
an **AI Song Contest submission**, kept as an archival experiment. No ongoing
maintenance or support is planned.

## Training and evaluation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), download or
clone this repo, and open a terminal in the repository root. Then run:

```sh
uv run bash scripts/train_final.sh
```

The script downloads and prepares the data, then runs our final training recipe:

1. **Pretrain for 5,000 steps** on a source-balanced mix of 90% PiJAMA and 10%
   aligned Charlie Parker MIDI.
2. **Fine-tune for 4,000 steps** on Weimar solos with chord and phrase annotations,
   starting from the best pretraining checkpoint.

Both stages use the same approximately 7.6M-parameter model, a 1,024-token context,
seed 42, and pitch, velocity, and articulation augmentation. The exact settings
are in [the pretraining config](configs/phrase-pretrain-mps.json) and
[the fine-tuning config](configs/phrase-finetune-mps.json).

Allow several hours on a capable GPU; CPU will be much slower. Our run used Apple
Silicon MPS with FP32. The script selects available hardware automatically, or
you can choose it:

```sh
uv run bash scripts/train_final.sh --device mps
```

`--device cuda` and `--device cpu` also work with a compatible PyTorch installation.
The Bash script works on macOS and Linux; on Windows, use WSL. `uv` sets up Python
and dependencies. Downloads and the first setup need internet.

**Rerun the same command to resume an interrupted run.** Completed stages are
skipped. To download and prepare data without starting training:

```sh
uv run bash scripts/train_final.sh --prepare-only
```

The script checks data against the saved final-experiment checksums and stops if
it differs. An incomplete preparation directory must be moved aside before retrying.
The recipe is reproducible, but identical model weights are not guaranteed across
hardware and software environments.

## Where the models are saved

| Model | Checkpoint |
|---|---|
| Best pretraining model | `runs/augmented-pretrain-phrases-mps/best.pt` |
| Final pretraining model | `runs/augmented-pretrain-phrases-mps/last.pt` |
| Best fine-tuned model | `runs/pretrained-weimar-phrases-mps/best.pt` |
| Final fine-tuned model | `runs/pretrained-weimar-phrases-mps/last.pt` |

In our saved run, pretraining's best and final checkpoints were both at step
5,000. Fine-tuning's best validation checkpoint was at step **1,600** (loss 1.6812);
the final checkpoint was at step **4,000** (loss 1.7195). Both are retained for
listening comparisons. These are sampled validation losses, not musical-quality
scores or results you are guaranteed to reproduce exactly.

The original weights are not included in this repo. The script trains new models
using that recipe. To inspect training progress:

```sh
uv run python scripts/summarize_training.py runs/pretrained-weimar-phrases-mps
```

## Generate MIDI

After training, generate a melody over a repeating chord progression:

```sh
uv run jazzbot generate --checkpoint runs/pretrained-weimar-phrases-mps/best.pt --chords "Dm7,G7,Cmaj7,A7" --max-new-tokens 1536 --seed 29 --output outputs/solo.mid
```

Use `last.pt` to try the final model, or change `--seed` for another take. Open the
MIDI in a DAW or MIDI player. Add `--strip-tempo` to follow your DAW's project tempo.
To continue an existing melody, add `--prompt phrase.mid --prompt-seconds 8`.
Output includes the processed prompt; polyphonic prompts are reduced to an upper
voice. Use a new output filename for each take to avoid overwriting it.

Every model needs its exact tokenizer. **New checkpoints include it inside the
`.pt` file.** Older checkpoints need their matching `tokenizer.json` beside them.
To make a smaller, self-contained inference file from either format:

```sh
uv run jazzbot export --checkpoint runs/pretrained-weimar-phrases-mps/best.pt --output checkpoints/model.pt
uv run jazzbot generate --checkpoint checkpoints/model.pt --output outputs/another-solo.mid
```

Exported files cannot resume training. Run `uv run jazzbot generate --help` for
all generation options. Inference needs only the source, dependencies, and a
compatible checkpoint; it does not need training data.

## Data and checks

Training uses [PiJAMA](https://almostimplemented.github.io/PiJAMA/),
[aligned Charlie Parker MIDI](https://huggingface.co/datasets/xavriley/CharlieParkerAlignedOmnibook),
and the [Weimar Jazz Database](https://jazzomat.hfm-weimar.de/download/download.html).
The model uses PyTorch and MidiTok. Data and dependencies retain their upstream
terms; the source-code license has not yet been selected. Data, weights, and
outputs are excluded from Git.

```sh
uv run --locked --extra dev ruff check src tests scripts
uv run --locked --extra dev pytest -q
```

Earlier [experiment notes](docs/research/DATASET_TEST_PLAN.md) and
[validation notes](docs/research/VALIDATION.md) are retained for context.
