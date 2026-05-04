#!/usr/bin/env bash
# scripts/build_whisper_cpp_linux.sh
#
# Idempotent builder for whisper.cpp v1.8.4 with the Vulkan backend enabled.
# Produces $BUILD_DIR/build/bin/whisper-cli — the binary consumed via
# --add-binary in scripts/build.sh (PyInstaller step).
#
# Re-runs are no-ops once the binary exists at the pinned tag.
#
# Ported from v1 backend/scripts/build_whisper_cpp.sh with these engine-side
# touch-ups:
#   - BUILD_DIR uses $RUNNER_TEMP (CI) or /tmp as default (not $HOME/.transcribe)
#   - Removed docs/DEPENDENCIES.md append (engine has its own README)
#   - Prints whisper-cli path to stdout on success so build.sh can pick it up

set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Pre-flight: required tools must be on $PATH.
# ---------------------------------------------------------------------------
for cmd in cmake git glslc; do
  if ! command -v "$cmd" >/dev/null; then
    echo "ERROR: $cmd not installed." >&2
    echo "       Install: sudo apt-get install cmake git glslc (Ubuntu/Debian)" >&2
    echo "       Or: sudo apt-get install mesa-vulkan-drivers libvulkan-dev vulkan-tools" >&2
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# 1. Resolve paths + pin tag.
# ---------------------------------------------------------------------------
BUILD_DIR="${WHISPER_CPP_BUILD_DIR:-${RUNNER_TEMP:-/tmp}/whisper.cpp}"
PIN_TAG="v1.8.4"
WHISPER_CLI="$BUILD_DIR/build/bin/whisper-cli"

# ---------------------------------------------------------------------------
# 2. Idempotency: if the binary exists AND HEAD is already at the pinned tag,
#    short-circuit. This makes re-runs (and CI smoke checks) cheap.
# ---------------------------------------------------------------------------
if [ -x "$WHISPER_CLI" ] && [ -d "$BUILD_DIR/.git" ]; then
  HEAD_SHA="$(git -C "$BUILD_DIR" rev-parse HEAD 2>/dev/null || echo none)"
  TAG_SHA="$(git -C "$BUILD_DIR" rev-parse "$PIN_TAG" 2>/dev/null || echo missing)"
  if [ "$HEAD_SHA" = "$TAG_SHA" ] && [ "$HEAD_SHA" != "none" ]; then
    echo "[build_whisper_cpp_linux.sh] already built at $PIN_TAG ($HEAD_SHA); skipping" >&2
    echo "$WHISPER_CLI"
    exit 0
  fi
fi

# ---------------------------------------------------------------------------
# 3. Clone (if absent) + checkout the pinned tag.
# ---------------------------------------------------------------------------
mkdir -p "$(dirname "$BUILD_DIR")"
if [ ! -d "$BUILD_DIR/.git" ]; then
  echo "[build_whisper_cpp_linux.sh] cloning whisper.cpp into $BUILD_DIR" >&2
  git clone https://github.com/ggml-org/whisper.cpp.git "$BUILD_DIR"
fi
git -C "$BUILD_DIR" fetch --tags --quiet
git -C "$BUILD_DIR" checkout "$PIN_TAG"

# ---------------------------------------------------------------------------
# 4. Configure + build with the Vulkan backend (-DGGML_VULKAN=1).
# ---------------------------------------------------------------------------
echo "[build_whisper_cpp_linux.sh] configuring cmake (-DGGML_VULKAN=1)" >&2
cmake -S "$BUILD_DIR" -B "$BUILD_DIR/build" \
  -DGGML_VULKAN=1 \
  -DCMAKE_BUILD_TYPE=Release

echo "[build_whisper_cpp_linux.sh] building (this can take 5-10 min on first run)" >&2
cmake --build "$BUILD_DIR/build" --config Release -j

# ---------------------------------------------------------------------------
# 5. Post-build verification.
# ---------------------------------------------------------------------------
if [ ! -x "$WHISPER_CLI" ]; then
  echo "ERROR: build completed but $WHISPER_CLI not found or not executable" >&2
  exit 1
fi

"$WHISPER_CLI" --help >/dev/null 2>&1 || {
  echo "ERROR: $WHISPER_CLI --help failed" >&2
  exit 1
}

# Binary-content probe: verify GGML Vulkan symbol is present.
if ! grep -q "ggml_vulkan" "$WHISPER_CLI" 2>/dev/null; then
  echo "WARN: 'ggml_vulkan' symbol not visible via grep; deferring Vulkan check to runtime probe" >&2
fi

BUILD_SHA="$(git -C "$BUILD_DIR" rev-parse HEAD)"
echo "[build_whisper_cpp_linux.sh] whisper-cli built: $WHISPER_CLI ($PIN_TAG @ $BUILD_SHA)" >&2

# Print the binary path to stdout so build.sh + CI can capture it.
echo "$WHISPER_CLI"
