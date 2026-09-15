# MIDI Bot 9000

A **7.55M-parameter PyTorch Transformer** for monophonic jazz MIDI generation and prompt continuation. Includes PiJAMA download/preparation, MidiTok REMI, CUDA and Apple Silicon MPS training, checkpoint resume, evaluation, and an inference-only export.

The infrastructure runs end to end. The included local smoke checkpoints are pipeline checks, **not trained jazz models**. Musical quality still needs a real training run and listening evaluation.

See the [dataset and representation test plan](DATASET_TEST_PLAN.md) for Weimar, FiloSax, Parker, and DTL comparisons, experiment order, and a listening scorecard.

## Run it

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from this directory:

```sh
uv sync --frozen --extra dev
uv run --no-sync jazzbot info
uv run --no-sync jazzbot download
uv run --no-sync jazzbot prepare
```

`download` gets the 24 MB Hawthorne transcription archive and PiJAMA metadata, verifies the published archive checksum, and creates short filenames for Windows compatibility. `prepare` processes the full corpus. Outputs are ignored by Git. Preparation refuses to overwrite a nonempty output directory; choose a new directory for a changed extraction recipe.

On this workspace, the downloaded data is already under `data/pijama`. Once prepared, `data/processed` is portable: copy that folder with the project to your training machine. Do not copy the Windows `.venv` to your Mac.

**M4 Max:**

```sh
uv run --no-sync jazzbot train --config configs/m4max.json --run runs/jazz-m4
```

**NVIDIA GPU:** install a PyTorch build matching the machine's driver, then train:

```sh
uv pip install --reinstall torch --torch-backend=auto
uv run --no-sync jazzbot info
uv run --no-sync jazzbot train --config configs/cuda.json --run runs/jazz-cuda
```

Check `cuda_available: true` before training. The GPU install is an intentional platform-specific override of the generic lockfile; `--no-sync` preserves it. For a reproducible production GPU environment, pin the selected torch version/backend after the initial hardware check. See [uv's PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

Both presets use 32 sequences per optimizer update, a 1,024-token context, AdamW, warmup/cosine learning rate, gradient clipping, and training-only transposition up to five semitones. CUDA automatically chooses BF16 when available, otherwise FP16 with loss scaling. MPS uses FP32 as the baseline. Lower `batch_size` and increase `accumulation_steps` proportionally if memory is tight. This is single-device training; distributed training is not implemented.

For a quick local check with a much smaller model:

```sh
uv run --no-sync jazzbot prepare --limit 40 --output data/smoke
uv run --no-sync jazzbot train --data data/smoke --config configs/smoke.json --run runs/smoke
uv run --no-sync jazzbot generate --checkpoint runs/smoke/best.pt --output outputs/smoke.mid
uv run --no-sync pytest -q
```

If these directories already exist, use a new output/run name. CPU runs can benefit from `OMP_NUM_THREADS=4` (PowerShell: `$env:OMP_NUM_THREADS = '4'`).

## Continue a MIDI solo

```sh
uv run --no-sync jazzbot generate --checkpoint runs/jazz-m4/best.pt --output outputs/solo.mid --max-new-tokens 512
uv run --no-sync jazzbot generate --checkpoint runs/jazz-m4/best.pt --prompt my-phrase.mid --prompt-seconds 8 --output outputs/continued.mid --temperature 0.9 --top-p 0.95
uv run --no-sync jazzbot inspect outputs/continued.mid
```

Output includes the preprocessed prompt followed by generated notes. `--prompt-seconds` takes the beginning of the input file; it does not search for its first note. The same extraction, time normalization, and quantization used for training apply to prompts. Accompaniment, original instrument choices, and pedal are removed. Seeded top-k/top-p sampling is reproducible on a fixed software/hardware stack; floating-point differences can change samples across devices.

Generation masks invalid note tuples, backward positions, and overlapping notes. The minimum note count applies to new notes; EOS can end the sample after that minimum. The token budget remains a hard maximum, so a short budget may produce fewer notes. The output JSON records sampled tokens and settings. KV caching keeps incremental generation cheap; when the context fills, generation re-prefills the latest half-context and resumes caching. This bounds cache memory but loses older musical context. MIDI playback requires a DAW or MIDI synthesizer.

## Resume, evaluate, export

```sh
uv run --no-sync jazzbot train --config configs/m4max.json --run runs/jazz-m4 --resume runs/jazz-m4/last.pt
uv run --no-sync jazzbot evaluate --checkpoint runs/jazz-m4/best.pt --split val
uv run --no-sync jazzbot evaluate --checkpoint runs/jazz-m4/best.pt --split test
uv run --no-sync jazzbot export --checkpoint runs/jazz-m4/best.pt --output outputs/jazzbot
uv run --no-sync jazzbot generate --checkpoint outputs/jazzbot/model.pt --output outputs/deployed.mid
```

Training writes `last.pt`, `best.pt`, `tokenizer.json`, the resolved config, and `metrics.jsonl`. Checkpoints are saved every validation interval and at the final step. Resume restores model, optimizer, scaler, sampler and backend RNG states. Model/dataset mismatches are rejected. Changing batch size, learning-rate schedule, or total steps on resume is permitted and changes the training trajectory. Use `--max-steps` to extend a completed run. An interruption can lose work since the last validation checkpoint.

Training validation uses fixed windows distributed across the validation recordings. The standalone evaluator defaults to **all** windows and reports token-weighted NLL, perplexity, and accuracy. Keep test results for the final selected model. Export removes optimizer/RNG state and retains the tokenizer and preprocessing recipe. It is a PyTorch bundle; Core ML, mobile app integration, and quantization are future deployment work.

## Representation and data decisions

PiJAMA is **polyphonic solo piano**, not a dataset of isolated monophonic improvisations. V1 builds a pseudo-melody: filter pitches to 48–96 and very quiet/short notes, group onsets within 30 ms, choose the highest pitch, and truncate overlaps. This can pick chord tops or accompaniment; it is a baseline heuristic, not reliable melody ground truth. Listen to extracted examples before spending a long training run:

```sh
uv run --no-sync jazzbot extract --input some-pijama-file.mid --seconds 30 --output outputs/extracted.mid
```

`data/pijama/paths.json` maps original PiJAMA paths to downloaded files. Prepared manifests retain each recording's source, artist, album, checksum, note count, split, and token offsets; skipped recordings include reasons.

Timing is preserved in seconds on a **fixed 120-QPM virtual grid** with 24 positions per quarter note: approximately 20.8 ms resolution. Velocity uses 32 bins. Notes use REMI `Bar → Position → Pitch → Velocity → Duration` events. Durations have 24 subdivisions per quarter up to 16 virtual quarters. This keeps fine timing without pretending the automatically transcribed files provide verified beats. **The resulting bars are two-second computational frames, not musically aligned measures.** Original rubato is represented through note placement in those frames; microtiming below the grid is lost. Reliable beat/downbeat annotations are needed before claiming beat-aware swing modeling or chord-chart alignment.

The default deterministic split groups albums approximately 80/10/10; identical file content and shared recording identifiers also join groups. Exact duplicate MIDI content is kept once. Splitting happens before windows and augmentation. This reduces leakage across tracks on an album, though alternate takes, shared tunes, and unrecognized duplicates can still cross splits. `--split official` uses PiJAMA's published recording split and rejects recognized duplicate recordings crossing it. Windows never mix recordings. Training windows are randomly cropped; validation/test targets cover each recording exactly once in full evaluation.

## Model

| Setting | V1 |
|---|---:|
| Parameters | 7,548,928 |
| Layers / width | 10 / 256 |
| Query / KV heads | 8 / 4 |
| SwiGLU hidden width | 704 |
| Vocabulary | 667 MidiTok tokens |
| Context | 1,024 tokens |
| FP32 weights | 28.8 MiB |
| Position / normalization | RoPE / RMSNorm |
| Output head | Tied to input embeddings |

Attention uses PyTorch scaled-dot-product attention, selecting the available backend. Grouped KV heads reduce persistent cache size; explicit head expansion keeps the attention path portable. No inter-layer weight sharing, pretrained weights, chord conditioning, audio input, or polyphonic output is implemented.

Start with a shorter run (for example `--max-steps 2000` in a new run directory), inspect validation loss and several fixed-prompt samples, then choose a longer budget. Compare against the extracted validation melodies, not the original full piano recordings. Loss alone does not establish musical quality: listen for phrase shape, repetition, swing feel, and playable intervals. `inspect` reports note density, pitch/velocity range, pitch-class entropy, and overlap counts, which are diagnostics rather than a musical-quality score.

## References

- [PiJAMA dataset and examples](https://almostimplemented.github.io/PiJAMA/) and [versioned MIDI release](https://zenodo.org/records/8354955): source recordings, crop boundaries, and metadata. Dataset material retains its upstream terms; this project does not redistribute audio.
- [REMI / Pop Music Transformer](https://arxiv.org/abs/2002.00212): explicit position and note-attribute sequence formulation. V1 uses [MidiTok's REMI implementation](https://miditok.readthedocs.io/en/latest/tokenizations.html).
- [This Time with Feeling](https://arxiv.org/abs/1808.03715): motivates retaining performance timing and velocity. V1 uses quantized durations rather than note-on/off events.
- [Music Transformer](https://arxiv.org/abs/1809.04281): context and relative position matter for symbolic music. V1 uses RoPE instead of reproducing its relative-attention implementation.
- [TinyStories](https://arxiv.org/abs/2305.07759): motivation for a specialized small-model experiment, not evidence that 7.55M parameters guarantees coherent jazz.
- [Compound Word Transformer](https://ojs.aaai.org/index.php/AAAI/article/view/16091): compound events are a future sequence-length ablation. V1 retains a single REMI token head to keep decoding simple.
- [MobileLLM](https://arxiv.org/abs/2402.14905): informs the deep/narrow shape, tied embeddings, and grouped KV heads. V1 does not copy its full architecture or training setup.

Source lives in `src/jazzbot`; presets in `configs`; tests cover causal attention and KV-cache equivalence, synthetic overfitting, MIDI round trips, constrained generation, recording boundaries, duplicate split protection, archive extraction, and train/resume/continuation. CI is configured for Linux, macOS and Windows; GPU/MPS execution requires hardware validation separately.
