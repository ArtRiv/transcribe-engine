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

# sysconfig.get_paths()['purelib'] is reliable on all platforms.
# site.getsitepackages()[0] returns the venv prefix (NOT site-packages) on
# Windows uv venvs, which causes PyInstaller --add-data to fail with
# "Unable to find <venv>\lightning_fabric\version.info".
SP="$(uv run python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
# PyInstaller --add-data uses os.pathsep as the SRC<sep>DEST separator:
# ':' on POSIX, ';' on Windows. Hardcoding ':' breaks on Windows because
# PyInstaller treats the path as containing a drive letter separator.
SEP="$(uv run python -c 'import os; print(os.pathsep)')"
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
  --add-data "$SP/lightning_fabric/version.info${SEP}lightning_fabric" \
  --add-data "$SP/lightning/version.info${SEP}lightning" \
  --add-data "$SP/pytorch_lightning/version.info${SEP}pytorch_lightning" \
  --add-data "transcribe_engine/picker_assets${SEP}picker_assets" \
  --add-data "models.toml${SEP}." \
  --copy-metadata lightning \
  --copy-metadata pytorch_lightning \
  --copy-metadata torch \
  --copy-metadata pyannote.audio \
  --add-binary "$WHISPER_CLI${SEP}." \
  --hidden-import "pystray.${PYSTRAY_BACKEND}" \
  transcribe_engine/__main__.py

echo "Built: dist/transcribe-engine$([ "${PYSTRAY_BACKEND}" = "_win32" ] && echo .exe || true)"
