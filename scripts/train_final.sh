#!/usr/bin/env bash
# Reproduce the final PiJAMA/Parker -> Weimar phrase-training recipe.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

device=auto
prepare_only=false
while (( $# )); do
  case "$1" in
    --device)
      if (( $# < 2 )); then
        echo "--device requires auto, cpu, cuda, or mps" >&2
        exit 2
      fi
      device="$2"
      shift 2
      ;;
    --prepare-only) prepare_only=true; shift ;;
    -h|--help)
      echo "Usage: uv run bash scripts/train_final.sh [--device auto|cpu|cuda|mps] [--prepare-only]"
      echo "Downloads/prepares data, pretrains for 5000 steps, then fine-tunes for 4000."
      echo "Rerun the same command to resume. --prepare-only stops before training."
      exit 0
      ;;
    *) echo "Unknown option: $1; use --help" >&2; exit 2 ;;
  esac
done
case "$device" in
  auto|cpu|cuda|mps) ;;
  *) echo "Invalid device: $device" >&2; exit 2 ;;
esac

if [[ "$prepare_only" == false ]]; then
  uv run --locked python -c \
    'import sys; from jazzbot.train import choose_device; print("Training device:", choose_device(sys.argv[1]))' \
    "$device"
fi
uv run --locked python scripts/prepare_final_training.py
training_args=(--device "$device")
if [[ "$prepare_only" == true ]]; then
  training_args+=(--prepare-only)
fi
exec uv run --locked python scripts/train_pretrained_phrases.py \
  "${training_args[@]}"
