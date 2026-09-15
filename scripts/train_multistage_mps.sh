#!/usr/bin/env bash
set -euo pipefail

MIDI_BOT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIDI_BOT_ROOT="$(cd "${MIDI_BOT_SCRIPT_DIR}/.." && pwd)"
cd "${MIDI_BOT_ROOT}"

MIDI_PRETRAIN_STEPS="${MIDI_PRETRAIN_STEPS:-5000}"
MIDI_FINETUNE_STEPS="${MIDI_FINETUNE_STEPS:-1000}"
MIDI_PRETRAIN_RUN="${MIDI_PRETRAIN_RUN:-runs/mixed-pretrain-mps}"
MIDI_FINETUNE_RUN="${MIDI_FINETUNE_RUN:-runs/weimar-finetune-mps}"
MIDI_PIJAMA_DATA="${MIDI_PIJAMA_DATA:-data/pijama-chord-compatible}"
MIDI_PARKER_ROOT="${MIDI_PARKER_ROOT:-data/parker}"
MIDI_PARKER_DATA="${MIDI_PARKER_DATA:-data/parker-chord-compatible}"
MIDI_PRETRAIN_DATA="${MIDI_PRETRAIN_DATA:-data/pretrain-mixture}"
MIDI_WEIMAR_DATA="${MIDI_WEIMAR_DATA:-data/weimar-conditioned}"

if [[ ! "${MIDI_PRETRAIN_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MIDI_PRETRAIN_STEPS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${MIDI_FINETUNE_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "MIDI_FINETUNE_STEPS must be a positive integer" >&2
  exit 2
fi

uv sync --frozen --extra dev
uv run --no-sync jazzbot info \
  --config configs/pijama-pretrain-mps.json \
  --chord-compatible

if [[ ! -f data/pijama/pijama.csv ]]; then
  uv run --no-sync jazzbot download --root data/pijama
fi
if [[ ! -f "${MIDI_PIJAMA_DATA}/manifest.json" ]]; then
  uv run --no-sync jazzbot prepare \
    --root data/pijama \
    --output "${MIDI_PIJAMA_DATA}" \
    --chord-compatible
fi
if [[ ! -d "${MIDI_PARKER_ROOT}/midi_performance_aligned" ]]; then
  uv run --no-sync python -c \
    'from huggingface_hub import snapshot_download; import sys; snapshot_download("xavriley/CharlieParkerAlignedOmnibook", repo_type="dataset", local_dir=sys.argv[1], allow_patterns=["README.md", "midi_performance_aligned/*.mid"])' \
    "${MIDI_PARKER_ROOT}"
fi
if [[ ! -f "${MIDI_PARKER_DATA}/manifest.json" ]]; then
  uv run --no-sync jazzbot prepare-midi-dir \
    --input "${MIDI_PARKER_ROOT}/midi_performance_aligned" \
    --output "${MIDI_PARKER_DATA}" \
    --dataset parker-aligned-omnibook
fi
if [[ ! -f "${MIDI_PRETRAIN_DATA}/manifest.json" ]]; then
  if [[ -n "${MIDI_DTL_MIDI_DIR:-}" ]]; then
    if [[ ! -d "${MIDI_DTL_MIDI_DIR}" ]]; then
      echo "MIDI_DTL_MIDI_DIR is not a directory: ${MIDI_DTL_MIDI_DIR}" >&2
      exit 2
    fi
    MIDI_DTL_DATA="${MIDI_DTL_DATA:-data/dtl-chord-compatible}"
    if [[ ! -f "${MIDI_DTL_DATA}/manifest.json" ]]; then
      uv run --no-sync jazzbot prepare-midi-dir \
        --input "${MIDI_DTL_MIDI_DIR}" \
        --output "${MIDI_DTL_DATA}" \
        --dataset dtl1000
    fi
    uv run --no-sync jazzbot mix \
      --input "${MIDI_PIJAMA_DATA}" --weight 0.35 \
      --input "${MIDI_DTL_DATA}" --weight 0.55 \
      --input "${MIDI_PARKER_DATA}" --weight 0.10 \
      --output "${MIDI_PRETRAIN_DATA}"
  else
    uv run --no-sync jazzbot mix \
      --input "${MIDI_PIJAMA_DATA}" --weight 0.90 \
      --input "${MIDI_PARKER_DATA}" --weight 0.10 \
      --output "${MIDI_PRETRAIN_DATA}"
  fi
fi
if [[ ! -f data/weimar/wjazzd.db ]]; then
  mkdir -p data/weimar
  curl -fL --retry 3 \
    -o data/weimar/wjazzd.db \
    https://jazzomat.hfm-weimar.de/download/downloads/wjazzd.db
fi
if [[ ! -f "${MIDI_WEIMAR_DATA}/manifest.json" ]]; then
  uv run --no-sync jazzbot prepare-weimar \
    --database data/weimar/wjazzd.db \
    --output "${MIDI_WEIMAR_DATA}"
fi

MIDI_KEEP_AWAKE=()
if command -v caffeinate >/dev/null 2>&1; then
  MIDI_KEEP_AWAKE=(caffeinate -dimsu)
fi

if [[ -f "${MIDI_PRETRAIN_RUN}/last.pt" ]]; then
  MIDI_PRETRAIN_CURRENT="$(uv run --no-sync python -c \
    'import sys, torch; print(torch.load(sys.argv[1], map_location="cpu", weights_only=True)["step"])' \
    "${MIDI_PRETRAIN_RUN}/last.pt")"
  if (( MIDI_PRETRAIN_CURRENT < MIDI_PRETRAIN_STEPS )); then
    "${MIDI_KEEP_AWAKE[@]}" uv run --no-sync jazzbot train \
      --data "${MIDI_PRETRAIN_DATA}" \
      --config configs/pijama-pretrain-mps.json \
      --run "${MIDI_PRETRAIN_RUN}" \
      --resume "${MIDI_PRETRAIN_RUN}/last.pt" \
      --max-steps "${MIDI_PRETRAIN_STEPS}"
  else
    echo "Mixed pretraining already reached step ${MIDI_PRETRAIN_CURRENT}; skipping"
  fi
else
  "${MIDI_KEEP_AWAKE[@]}" uv run --no-sync jazzbot train \
    --data "${MIDI_PRETRAIN_DATA}" \
    --config configs/pijama-pretrain-mps.json \
    --run "${MIDI_PRETRAIN_RUN}" \
    --max-steps "${MIDI_PRETRAIN_STEPS}"
fi

if [[ -f "${MIDI_FINETUNE_RUN}/last.pt" ]]; then
  MIDI_FINETUNE_CURRENT="$(uv run --no-sync python -c \
    'import sys, torch; print(torch.load(sys.argv[1], map_location="cpu", weights_only=True)["step"])' \
    "${MIDI_FINETUNE_RUN}/last.pt")"
  if (( MIDI_FINETUNE_CURRENT < MIDI_FINETUNE_STEPS )); then
    "${MIDI_KEEP_AWAKE[@]}" uv run --no-sync jazzbot train \
      --data "${MIDI_WEIMAR_DATA}" \
      --config configs/weimar-finetune-mps.json \
      --run "${MIDI_FINETUNE_RUN}" \
      --resume "${MIDI_FINETUNE_RUN}/last.pt" \
      --max-steps "${MIDI_FINETUNE_STEPS}"
  else
    echo "Weimar fine-tuning already reached step ${MIDI_FINETUNE_CURRENT}; skipping"
  fi
else
  "${MIDI_KEEP_AWAKE[@]}" uv run --no-sync jazzbot train \
    --data "${MIDI_WEIMAR_DATA}" \
    --config configs/weimar-finetune-mps.json \
    --run "${MIDI_FINETUNE_RUN}" \
    --init-from "${MIDI_PRETRAIN_RUN}/best.pt" \
    --max-steps "${MIDI_FINETUNE_STEPS}"
fi

uv run --no-sync jazzbot generate \
  --checkpoint "${MIDI_FINETUNE_RUN}/best.pt" \
  --output outputs/mixed-pretrain-then-weimar-conditioned-demo.mid \
  --chords 'Amaj7(13),F#maj/G#,F#min9,C#min9' \
  --beats-per-chord 4 \
  --max-new-tokens 1024 \
  --min-notes 48 \
  --temperature 0.9 \
  --top-k 32 \
  --top-p 0.95 \
  --seed 33 \
  --device mps

uv run --no-sync jazzbot inspect outputs/mixed-pretrain-then-weimar-conditioned-demo.mid
echo "Finished: outputs/mixed-pretrain-then-weimar-conditioned-demo.mid"
