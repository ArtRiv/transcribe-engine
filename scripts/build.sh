#!/usr/bin/env bash
# Build transcribe-engine into a single PyInstaller onefile binary.
# Spike-validated: see .claude/skills/spike-findings-transcribe (R-03 + R-05).
#
# Per-OS overrides (CI matrix sets these):
#   PYSTRAY_BACKEND   _xorg (Linux) / _darwin (macOS) / _win32 (Windows)
#   WHISPER_CLI       Path to the per-OS whisper-cli binary built upstream
#
# The R-03 invocation includes THREE explicit --add-data version.info lines.
# DO NOT remove any of them — bundle dies on first launch with FileNotFoundError.
# IMPORTANT: --copy-metadata for lightning_fabric must NOT be added — it crashes the build (R-03).
set -euo pipefail

SP="$(uv run python -c 'import site; print(site.getsitepackages()[0])')"
PYSTRAY_BACKEND="${PYSTRAY_BACKEND:-_xorg}"
WHISPER_CLI="${WHISPER_CLI:-./whisper-cli}"

if [[ ! -f "$WHISPER_CLI" ]]; then
  echo "ERROR: WHISPER_CLI not found at $WHISPER_CLI" >&2
  echo "Run scripts/build_whisper_cpp_<OS>.sh first (or set WHISPER_CLI env var)." >&2
  exit 1
fi

echo "Building transcribe-engine (PYSTRAY_BACKEND=$PYSTRAY_BACKEND, WHISPER_CLI=$WHISPER_CLI)..."

# The R-03 invocation. DO NOT remove any --add-data version.info line — bundle dies on first launch.
# IMPORTANT: --copy-metadata must NOT include lightning_fabric (it crashes the build; R-03 anti-pattern).
uv run pyinstaller --onefile --name transcribe-engine \
  --collect-all pyannote.audio \
  --collect-all pyannote \
  --collect-all torch \
  --collect-all torchaudio \
  --collect-all lightning \
  --collect-all pytorch_lightning \
  --collect-all lightning_fabric \
  --collect-all asteroid_filterbanks \
  --collect-all speechbrain \
  --collect-all transformers \
  --collect-all huggingface_hub \
  --add-data "$SP/lightning_fabric/version.info:lightning_fabric" \
  --add-data "$SP/lightning/version.info:lightning" \
  --add-data "$SP/pytorch_lightning/version.info:pytorch_lightning" \
  --add-data "transcribe_engine/picker_assets:picker_assets" \
  --add-data "models.toml:." \
  --copy-metadata lightning \
  --copy-metadata pytorch_lightning \
  --copy-metadata torch \
  --copy-metadata pyannote.audio \
  --add-binary "$WHISPER_CLI:." \
  --hidden-import "pystray.${PYSTRAY_BACKEND}" \
  transcribe_engine/__main__.py

echo "Built: dist/transcribe-engine$([ "${PYSTRAY_BACKEND}" = "_win32" ] && echo .exe || true)"
