# Jazz Bot: dataset and representation test plan

Build a small model that generates an energetic, convincing monophonic jazz solo: coherent phrases, swing, chromatic movement, useful rests, and musical resolutions. High note density alone is not success.

**Working recommendation:** add Weimar as the main source of clean solo lines, retain PiJAMA for a pretraining comparison, and test TSD against REMI. These are hypotheses to test, not established results for this model.

## 1. Dataset shortlist

| Dataset | What it provides | Role in the experiment | Main limitation |
|---|---|---|---|
| **Weimar Jazz Database / Jazzomat** | 456 transcribed monophonic solos; unquantized MIDI; a SQLite database with note timing, beats, chords, and phrase/form annotations. | First dataset to add. Direct examples of the target behavior. | Small enough that memorization needs close attention. |
| **PiJAMA** | Automatically transcribed solo piano performances. Our prepared corpus retains 2,774 recordings. | Existing baseline and candidate pretraining source. | Our extracted upper voice can contain chord tops and accompaniment rather than the intended solo melody. |
| **FiloSax** | Five saxophonists performing 48 pieces, with performance MIDI and annotations distinguishing heads, written solos, and improvisation. | Add genuine improvisation sections after establishing a Weimar baseline. | Repeated compositions across performers; access requires agreement to noncommercial research terms. |
| **Charlie Parker Omnibook alignment dataset** | 50 pieces with scores, performance-aligned MIDI, and downbeat annotations. | Focused bebop supplement. | Narrow performer/style coverage and possible overlap with other corpora. |
| **Dig That Lick / DTL1000** | 1,736 automatically extracted monophonic solos, approximately 300,000 note events. | Later scale experiment if the clean datasets help. | Automatic transcription quality needs an audition audit. |

Sources and downloads:

- **Weimar:** [downloads, including MIDI and SQLite](https://jazzomat.hfm-weimar.de/download/download.html), [database schema](https://jazzomat.hfm-weimar.de/dbformat/dbformat.html), [solo catalogue](https://jazzomat.hfm-weimar.de/dbformat/dbcontent.html). The download page identifies ODbL as the database license.
- **PiJAMA:** [dataset documentation](https://almostimplemented.github.io/PiJAMA/), [versioned MIDI release](https://zenodo.org/records/8354955). Local preparation results are in [VALIDATION.md](VALIDATION.md).
- **FiloSax:** [dataset, annotations, access, and terms](https://dave-foster.github.io/filosax/). Use section labels to distinguish improvisation from written material.
- **Parker:** [dataset construction and supplementary link](https://arxiv.org/html/2405.16687v1). It includes themes as well as solos; select the intended sections.
- **DTL1000:** [project and corpus descriptions](https://dig-that-lick.eecs.qmul.ac.uk/), [project data repository and OSF pointers](https://github.com/ppquadrat/DigThatLick).

If the target later becomes specifically a piano solo over a rhythm section, also consider the **Jazz Trio Database**. It supplies automatically transcribed piano MIDI, beat information, and solo timestamps, but still needs monophonic extraction. [Database structure](https://huwcheston.github.io/Jazz-Trio-Database/installation/database-structure.html).

## 2. What works now versus what needs implementation

| Capability | Current status |
|---|---|
| PiJAMA download, extraction, REMI preparation, training, generation, evaluation | Implemented |
| CUDA and MPS presets | Implemented; local validation used CPU |
| Weimar, FiloSax, Parker, or DTL import | Not implemented |
| TSD tokenizer and matching generation grammar | Not implemented |
| Fine-tuning on a different dataset | Not implemented; `--resume` requires the original dataset and model configuration |
| Common held-out evaluation across training corpora | Needs an explicit evaluation path; the current evaluator checks the training manifest |
| Chord-conditioned generation | Not implemented |

The existing PiJAMA commands below can run today. The other experiment rows describe work to implement before running them; there are no hypothetical CLI flags to copy.

## 3. Import and audition before training

- [ ] Download Weimar's SQLite database as well as its MIDI archive. Retain beats, chords, phrase boundaries, performer, tune, and recording identity in the intermediate data, even when an experiment does not use all of them.
- [ ] Bypass the piano upper-voice extraction for already transcribed monophonic solos. Audit brief notes and small overlaps; do not blindly apply the current 30 ms onset grouping, 50 ms minimum duration, or piano pitch filter.
- [ ] Verify sounding pitch, timing units, note lengths, and any tempo map through a MIDI round trip.
- [ ] Check whether velocity values are meaningful. Weimar's database includes loudness measurements, which are not automatically equivalent to MIDI velocity. Document any mapping or use an explicit neutral-velocity baseline.
- [ ] Audition at least 10 varied excerpts per source: fast lines, slower phrases, rests, chromatic passages, and upper/lower registers. Compare against the recording or supplied score when available.
- [ ] Reject or flag major missing-note, octave, timing, or extraction errors; keep reasons in the manifest.

Weimar's note and beat fields are documented in its [schema](https://jazzomat.hfm-weimar.de/dbformat/dbformat.html). For an energetic style subset, its [catalogue](https://jazzomat.hfm-weimar.de/dbformat/dbcontent.html) includes Parker, Cannonball Adderley, and Brecker, with style and tempo metadata. Begin with broad clean coverage; test style emphasis after the baseline works.

## 4. Representation candidates

| Candidate | Hypothesis | What to check |
|---|---|---|
| **Current REMI** | Existing reference for all comparisons. | Fine timing survives, but its fixed 120-QPM bars are computational frames rather than verified musical measures. |
| **MidiTok TSD** | Time shifts plus pitch, velocity, and duration may suit expressive single-line continuation with less structural machinery. | Timing error, tokens per note, long-rest handling, generation validity, and listening preference. |
| **Beat-aware representation with chord context** | Actual beat placement and supplied harmony may improve solos over a progression. | Preserve swing/microtiming during beat normalization and provide only the intended accompaniment context. This is a separate conditioned task. |
| **Compound note events** | Packing attributes might reduce sequence length and improve inference speed. | Requires different prediction heads/decoding; defer until profiling shows sequence length is a real bottleneck. |

TSD conceptually describes a note using elapsed time, pitch, velocity, and duration. It does not inherently recover beats or eliminate quantization. MidiTok documents a maximum-time-shift limitation for long pauses, so test that explicitly. [Tokenizer documentation](https://miditok.readthedocs.io/en/latest/tokenizations.html).

For the first REMI-versus-TSD comparison, use the **same cleaned notes, timing resolution, pitch range, and velocity treatment**. Record tokens per note and the musical duration covered by 1,024 tokens. Changing both preprocessing and tokenization would obscure which change helped.

## 5. Experiment order

| ID | Training data and initialization | Representation | Question |
|---|---|---|---|
| P0 | PiJAMA, from scratch | Current REMI | What can the existing pipeline produce? |
| W0 | Weimar, from scratch | Same REMI recipe | Does a cleaner solo corpus improve the target behavior? |
| W1 | PiJAMA pretraining, then Weimar fine-tuning | Same REMI recipe and vocabulary in both stages | Does pretraining help beyond Weimar alone? |
| W2 | Weimar, from scratch | TSD | Does representation improve over W0? |
| W3 | Repeat the winning data recipe with the other representation if useful | REMI or TSD | Does the result persist when data and representation are combined? |
| S0 | Best recipe plus selected FiloSax or Parker training sections | Winning representation | Does one targeted supplement improve held-out solos? |
| S1 | Best recipe plus audited DTL training material | Winning representation | Does extra automatic transcription data help or dilute quality? |

Run **one supplement at a time**. Record source sampling proportions so PiJAMA or DTL cannot silently dominate the cleaner material by volume.

Use the existing 7.55M model initially to isolate data changes. If Weimar training loss falls while held-out quality deteriorates, compare a smaller model as a separate experiment. A pretrained checkpoint can only transfer directly when the representation, vocabulary mapping, and model dimensions agree; a REMI checkpoint is not a drop-in TSD checkpoint.

Fine-tuning needs a new path that loads compatible model weights while creating a fresh optimizer, schedule, run directory, and target-dataset manifest. Do not disable the existing resume checks to simulate it.

## 6. Fair comparison protocol

1. **Split before augmentation.** Group recordings and all excerpts from the same performance. For a tune-generalization test, keep all versions of a composition in one split. FiloSax performances of the same piece should stay together. Check shared recordings/tunes across datasets before pretraining.
2. **Reserve common Weimar validation and test material.** Evaluate the candidates on the same target solos and prompts. Keep final test examples out of model and sampling decisions. Do not compare PiJAMA validation loss with Weimar validation loss as if they measure the same task.
3. **Keep the model and training setup fixed within each comparison.** Log optimizer updates, tokens processed, effective batch, passes through the corpus, and wall time. Include pretraining cost when reporting W1. The small Weimar corpus should not automatically inherit PiJAMA's 20,000-step budget.
4. **Choose checkpoints by validation.** Use frequent checks and stop extending a run when validation and listening stop improving. Repeat promising results with multiple training seeds before calling a winner.
5. **Compare likelihood only where comparable.** Raw token perplexity is not comparable across REMI and TSD. Use same-tokenizer held-out loss within an experiment, and listening, timing preservation, validity, and runtime across representations.
6. **Keep sampling fixed initially.** Start with temperature 0.9, top-k 32, top-p 0.95; use generation seeds 11, 22, and 33. Tune sampling separately afterward.

## 7. Listening test and scorecard

Choose eight held-out prompts, including short motifs, fast passages, slower phrases, and different registers. Generate three continuations per prompt: **24 samples per candidate**. Use the same prompt performances, synth patch, rendering settings, and playback gain. Compare a consistent listening duration, such as the first 20 seconds after each prompt; record early endings rather than hiding them. Equal token budgets do not imply equal musical duration across tokenizers.

Hide model names and randomize playback order. Rate each criterion from **1 (poor) to 5 (strong)**:

| Criterion | Listen for |
|---|---|
| Phrase development | Motifs grow or receive an answer; the line has direction. |
| Rhythm and swing | Convincing placement, articulation, varied rhythm, and intentional rests. |
| Jazz vocabulary | Chromatic approaches, enclosures, interval choices, and convincing resolutions. |
| Energy | Momentum and excitement with contrast; not merely more notes. |
| Prompt continuity | The continuation belongs with the phrase that preceded it. |
| Overall musical preference | Would you keep, play, or develop this solo? |

Only score fit to a **specific chord progression** when the model was given that progression. The current model cannot be expected to follow an unseen backing track.

Also record malformed outputs, overlaps, excessive silence, repeated loops, note density, pitch range, and generation speed. Compare suspiciously familiar phrases against training examples; shared short jazz licks alone are not proof of memorization. Existing `inspect` metrics are diagnostics, not a musical-quality score.

## 8. Run the existing baseline now

The full PiJAMA corpus is already prepared in this workspace at `data/processed`. On a new training machine, copy it with the project and install that machine's environment using the [README](README.md).

```sh
uv sync --frozen --extra dev
uv run --no-sync jazzbot train --data data/processed --config configs/m4max.json --run runs/p0-pijama-remi --max-steps 2000
uv run --no-sync jazzbot evaluate --data data/processed --checkpoint runs/p0-pijama-remi/best.pt --split val
```

For NVIDIA, follow the README's CUDA installation instructions and substitute `configs/cuda.json`. Use a fresh run directory. This 2,000-step command is an exploratory PiJAMA budget, not a recommended Weimar budget or a guarantee of convergence.

Choose an actual held-out MIDI file and replace `path/to/held-out.mid` below:

```sh
uv run --no-sync jazzbot generate --checkpoint runs/p0-pijama-remi/best.pt --prompt path/to/held-out.mid --prompt-seconds 8 --output outputs/p0-prompt01-seed11.mid --max-new-tokens 512 --min-notes 32 --temperature 0.9 --top-k 32 --top-p 0.95 --seed 11
uv run --no-sync jazzbot inspect outputs/p0-prompt01-seed11.mid
```

Repeat with seeds 22 and 33 and unique output filenames. Current output includes the preprocessed prompt; judge the newly generated portion separately. The token cap remains a hard limit and can end a sample before the requested minimum notes.

## 9. Experiment log

Copy this block for each run:

```text
Experiment ID / date:
Code revision or source snapshot:
Dataset versions, file checksums, and split manifest:
Source mixture / filtering / augmentation:
Tokenizer configuration and hash:
Model parameters / context tokens / typical context duration:
Hardware / precision / software versions:
Training seed / generation seeds:
Pretraining budget / fine-tuning budget / total wall time:
Selected checkpoint and selection reason:
Validation loss (only compare compatible tokenizations and evaluation data):
Malformed outputs / overlaps / timing errors:
Generation speed / memory:
Blind listening scores / pairwise preferences:
Memorization or repeated-loop observations:
Decision: retain / revise / reject
Next single variable to change:
```

First milestone: a fair **W0 versus W1 versus W2** comparison, with listening results and a clean held-out split. That will tell us whether more piano data, cleaner solo data, or a different event representation actually improves this bot.
