#!/usr/bin/env bash
set -euo pipefail

MIDI_BOT_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIDI_BOT_ROOT="$(cd "${MIDI_BOT_SCRIPT_DIR}/.." && pwd)"
cd "${MIDI_BOT_ROOT}"

if (( $# < 1 || $# > 2 )); then
  echo "Usage: $0 CHECKPOINT [OUTPUT_DIRECTORY]" >&2
  exit 2
fi

MIDI_BOT_CHECKPOINT="$1"
MIDI_BOT_BUNDLE="${2:-release/midi-bot-9000}"

if [[ ! -f "${MIDI_BOT_CHECKPOINT}" ]]; then
  echo "Checkpoint does not exist: ${MIDI_BOT_CHECKPOINT}" >&2
  exit 2
fi
if [[ -e "${MIDI_BOT_BUNDLE}" ]]; then
  echo "Output already exists; choose a new directory: ${MIDI_BOT_BUNDLE}" >&2
  exit 2
fi
MIDI_BOT_ARCHIVE="${MIDI_BOT_BUNDLE%/}.zip"
if [[ -e "${MIDI_BOT_ARCHIVE}" ]]; then
  echo "Archive already exists; choose a new output directory: ${MIDI_BOT_ARCHIVE}" >&2
  exit 2
fi

MIDI_BOT_WHEEL_TMP="$(mktemp -d)"
trap 'rm -rf "${MIDI_BOT_WHEEL_TMP}"' EXIT

uv run --no-sync jazzbot export \
  --checkpoint "${MIDI_BOT_CHECKPOINT}" \
  --output "${MIDI_BOT_BUNDLE}"
uv build --wheel --out-dir "${MIDI_BOT_WHEEL_TMP}"
cp "${MIDI_BOT_WHEEL_TMP}"/*.whl "${MIDI_BOT_BUNDLE}/"

echo "Share this entire directory: ${MIDI_BOT_BUNDLE}"
echo "It contains model.pt, tokenizer.json, and the installable project wheel."
if command -v ditto >/dev/null 2>&1; then
  ditto -c -k --keepParent "${MIDI_BOT_BUNDLE}" "${MIDI_BOT_ARCHIVE}"
  echo "Shareable archive: ${MIDI_BOT_ARCHIVE}"
fi
