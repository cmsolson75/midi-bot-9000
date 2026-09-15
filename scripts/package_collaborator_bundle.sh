#!/usr/bin/env bash
set -euo pipefail

MIDI_BOT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIDI_BOT_ROOT="$(cd "${MIDI_BOT_SCRIPT_DIR}/.." && pwd)"
cd "${MIDI_BOT_ROOT}"

if (( $# > 1 )); then
  echo "Usage: $0 [OUTPUT_DIRECTORY]" >&2
  exit 2
fi

MIDI_BOT_BUNDLE="${1:-release/midi-bot-9000-collaborator}"
MIDI_BOT_ARCHIVE="${MIDI_BOT_BUNDLE%/}.zip"
MIDI_BOT_PRETRAIN="runs/mixed-pretrain-mps/best.pt"
MIDI_BOT_SMOKE="runs/chord-mps-30m/best.pt"
MIDI_BOT_BEST="runs/weimar-finetune-mps/best.pt"
MIDI_BOT_LAST="runs/weimar-finetune-mps/last.pt"

for checkpoint in \
  "${MIDI_BOT_PRETRAIN}" \
  "${MIDI_BOT_SMOKE}" \
  "${MIDI_BOT_BEST}" \
  "${MIDI_BOT_LAST}"; do
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Checkpoint does not exist: ${checkpoint}" >&2
    exit 2
  fi
done
if [[ -e "${MIDI_BOT_BUNDLE}" || -e "${MIDI_BOT_ARCHIVE}" ]]; then
  echo "Bundle or archive already exists; choose a new output directory" >&2
  exit 2
fi

MIDI_BOT_WHEEL_TMP="$(mktemp -d)"
trap 'rm -rf "${MIDI_BOT_WHEEL_TMP}"' EXIT

mkdir -p "${MIDI_BOT_BUNDLE}/models"
uv run --no-sync jazzbot export \
  --checkpoint "${MIDI_BOT_PRETRAIN}" \
  --output "${MIDI_BOT_BUNDLE}/models/pretrained-step5000"
uv run --no-sync jazzbot export \
  --checkpoint "${MIDI_BOT_SMOKE}" \
  --output "${MIDI_BOT_BUNDLE}/models/smoke-original-step1400"
uv run --no-sync jazzbot export \
  --checkpoint "${MIDI_BOT_BEST}" \
  --output "${MIDI_BOT_BUNDLE}/models/finetune-best-step900"
uv run --no-sync jazzbot export \
  --checkpoint "${MIDI_BOT_LAST}" \
  --output "${MIDI_BOT_BUNDLE}/models/finetune-last-step2200"
uv build --wheel --out-dir "${MIDI_BOT_WHEEL_TMP}"
cp "${MIDI_BOT_WHEEL_TMP}"/*.whl "${MIDI_BOT_BUNDLE}/"
cp USAGE_GUIDE.md "${MIDI_BOT_BUNDLE}/README.md"

if command -v ditto >/dev/null 2>&1; then
  ditto -c -k --keepParent "${MIDI_BOT_BUNDLE}" "${MIDI_BOT_ARCHIVE}"
  echo "Email this archive: ${MIDI_BOT_ARCHIVE}"
else
  echo "Share this directory: ${MIDI_BOT_BUNDLE}"
fi
