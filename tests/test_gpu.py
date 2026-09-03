"""test_gpu.py — Mock-based per-OS GPU probe tests (D-18).

Tests verify each branch of detect_gpu() via mocked subprocess.
All tests run in <1s (no real subprocess execution).
"""
import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from transcribe_engine.platform.gpu import GpuInfo, detect_gpu


def _mock_proc(stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
    proc = MagicMock()
    proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.returncode = returncode
    return proc


@pytest.mark.asyncio
async def test_detect_gpu_returns_cpu_when_no_probe_succeeds(monkeypatch):
    async def _raise(*a, **kw):
        raise FileNotFoundError("not on PATH")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _raise)
    result = await detect_gpu()
    assert result.backend == "cpu"
    assert "CPU" in result.label


@pytest.mark.asyncio
async def test_detect_gpu_linux_vulkan_success(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    proc = _mock_proc(stdout=b"deviceName = AMD Radeon RX 6600\n")
    async def _exec(*a, **kw):
        return proc
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)
    result = await detect_gpu()
    assert result.backend == "vulkan"
    assert "AMD" in result.device_name
    assert "Vulkan" in result.label


@pytest.mark.asyncio
async def test_detect_gpu_windows_vulkaninfo_missing(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    async def _raise(*a, **kw):
        raise FileNotFoundError("vulkaninfo.exe")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _raise)
    result = await detect_gpu()
    assert result.backend == "cpu"  # Pitfall 7 mitigation


@pytest.mark.asyncio
async def test_detect_gpu_macos_metal_success(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "platform", "darwin")
    # Create a placeholder whisper-cli so the existence check passes
    cli = tmp_path / "whisper-cli"
    cli.write_text("")
    cli.chmod(0o755)
    proc = _mock_proc(stderr=b"ggml-metal: GPU name: Apple M2\n")
    async def _exec(*a, **kw):
        return proc
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _exec)
    result = await detect_gpu(whisper_cli=cli)
    assert result.backend == "metal"
    assert "Apple M2" in result.device_name


def test_gpuinfo_label_format():
    info = GpuInfo(backend="vulkan", device_name="AMD Radeon RX 6600", label="AMD Radeon RX 6600 (Vulkan)")
    assert "(Vulkan)" in info.label
