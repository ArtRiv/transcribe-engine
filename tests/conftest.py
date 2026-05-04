"""Engine test fixtures — MOCK_ENGINE pattern from v1, plus engine-specific helpers.

The MOCK_ENGINE / SKIP_VULKAN_PROBE pair lets tests exercise pipeline + state code
on machines without a GPU. Only @pytest.mark.gpu tests touch real hardware.

Carry from backend/tests/conftest.py (lines 26-51):
  - _default_mock_engine autouse fixture (MOCK_ENGINE + SKIP_VULKAN_PROBE)

Dropped (ASGI/HTTP-framework specific — engine has no HTTP server in Phase 7):
  - lifespan fixture (LifespanManager + app)
  - client fixture (httpx + ASGI transport)
  - _mock_jwt_verify fixture (engine has no JWKS)

Engine-side replacements:
  - mock_gpu_info: stub GpuInfo for CPU branch
  - temp_xdg_dirs: monkeypatches XDG dirs so paths.* don't pollute ~/.config
  - mock_whisper_cli: fake whisper-cli binary emitting stub transcript.json
"""
import os
from pathlib import Path

import pytest

from transcribe_engine.platform.gpu import GpuInfo


@pytest.fixture(autouse=True)
def _default_mock_engine(monkeypatch):
    """Carry from v1: default MOCK_ENGINE=1 + SKIP_VULKAN_PROBE=1 unless overridden.

    MOCK_ENGINE=1  — replaces real pipeline with stubs (matches v1 behaviour).
    SKIP_VULKAN_PROBE=1 — bypasses the Vulkan/Metal probe in GPU detection so
        tests don't need an actual GPU or vulkaninfo/metal CLI on the test host.
    """
    if os.environ.get("MOCK_ENGINE") is None:
        monkeypatch.setenv("MOCK_ENGINE", "1")
    if os.environ.get("SKIP_VULKAN_PROBE") is None:
        monkeypatch.setenv("SKIP_VULKAN_PROBE", "1")
    yield


@pytest.fixture
def temp_xdg_dirs(monkeypatch, tmp_path):
    """Override XDG dirs so paths.* helpers don't pollute the dev's real ~/.config.

    platformdirs honours XDG env vars on Linux. macOS/Windows tests that exercise
    paths.* should either use this fixture or be marked linux-only.
    """
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    return tmp_path


@pytest.fixture
def mock_gpu_info() -> GpuInfo:
    """Stub GpuInfo in CPU fallback mode.

    Tests that need GPU detection logic can use this to force the CPU branch
    without running the real vulkaninfo/metal probe.
    """
    return GpuInfo(backend="cpu", device_name="CPU", label="CPU (no GPU detected)")


@pytest.fixture
def mock_whisper_cli(tmp_path: Path) -> Path:
    """Fake whisper-cli that emits a stub transcript.json.

    Used by pipeline tests that need to exercise transcribe_subprocess without
    a real whisper.cpp build. The script captures --output-file and writes the
    expected JSON shape alongside it (whisper.cpp adds .json suffix).
    """
    cli = tmp_path / "whisper-cli"
    cli.write_text(
        '#!/usr/bin/env bash\n'
        '# Mock whisper-cli — used by tests under MOCK_ENGINE=1.\n'
        '# Real wrapper is transcribe_engine/pipeline/transcribe.py.\n'
        'OUT=""\n'
        'while [[ $# -gt 0 ]]; do\n'
        '  case "$1" in\n'
        '    --output-file) OUT="$2"; shift 2;;\n'
        '    *) shift;;\n'
        '  esac\n'
        'done\n'
        'if [[ -n "$OUT" ]]; then\n'
        "  cat > \"${OUT}.json\" <<'EOF'\n"
        '{"systeminfo":"mock","model":{"type":"mock"},"params":{},"result":{"language":"en"},"transcription":[]}\n'
        'EOF\n'
        'fi\n'
    )
    cli.chmod(0o755)
    return cli
