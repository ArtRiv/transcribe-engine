"""Cross-platform GPU detection (D-18).

Vulkan-first on Linux/Windows, Metal-first on macOS, CPU fallback always.
NEVER raises — engine must boot on any user's machine. This is different
from v1's `probe_vulkan_or_die` (single-host AMD; engine targets any user).
"""

import asyncio
import logging
import re
import struct
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

Backend = Literal["vulkan", "metal", "cpu"]


@dataclass
class GpuInfo:
    backend: Backend
    device_name: str
    label: str  # tray status line suffix


_DEVICE_NAME_RE = re.compile(r"deviceName\s*=\s*(.+)")
_METAL_BANNER_RE = re.compile(r"ggml-metal:\s*GPU name:\s*(.+)")


def _make_silence_wav(path: Path, *, seconds: float = 0.3, sample_rate: int = 16000) -> None:
    """Write a brief silence WAV for whisper-cli probe (carries from v1 vulkan.py)."""
    n_frames = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))


async def _probe_vulkan() -> GpuInfo | None:
    """Probe vulkaninfo. Returns None on any failure (CPU fallback).

    Handles vulkaninfo.exe on Windows (Pitfall 7 — not always on PATH).
    """
    binary = "vulkaninfo.exe" if sys.platform == "win32" else "vulkaninfo"
    try:
        proc = await asyncio.create_subprocess_exec(
            binary,
            "--summary",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout_b, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
        if proc.returncode != 0:
            return None
        stdout = stdout_b.decode(errors="replace")
        m = _DEVICE_NAME_RE.search(stdout)
        if not m:
            return None
        device_name = m.group(1).strip()
        return GpuInfo(backend="vulkan", device_name=device_name, label=f"{device_name} (Vulkan)")
    except (TimeoutError, FileNotFoundError, OSError) as e:
        log.info("Vulkan probe failed (%s); falling back to CPU", type(e).__name__)
        return None


async def _probe_metal(whisper_cli: Path | None = None) -> GpuInfo | None:
    """Probe Metal via whisper-cli silence-WAV trick. Returns None on failure.

    Carries the whisper-cli stderr-banner pattern from v1 vulkan.py (lines 113-181)
    adapted for the ggml-metal banner instead of ggml_vulkan.
    """
    if whisper_cli is None or not whisper_cli.exists():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        silence = Path(tmp) / "silence.wav"
        _make_silence_wav(silence)
        try:
            proc = await asyncio.create_subprocess_exec(
                str(whisper_cli),
                "-f",
                str(silence),
                "-nt",
                "-otxt",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            _, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=15)
            stderr = stderr_b.decode(errors="replace")
            m = _METAL_BANNER_RE.search(stderr)
            if not m:
                return None
            device_name = m.group(1).strip()
            return GpuInfo(backend="metal", device_name=device_name, label=f"{device_name} (Metal)")
        except (TimeoutError, FileNotFoundError, OSError) as e:
            log.info("Metal probe failed (%s); falling back to CPU", type(e).__name__)
            return None


async def detect_gpu(*, whisper_cli: Path | None = None) -> GpuInfo:
    """Cross-platform GPU detection (D-18). Always returns valid GpuInfo.

    - Linux/Windows: Vulkan first → CPU fallback
    - macOS: Metal first → CPU fallback

    NEVER raises. Engine boots on any user's machine.
    """
    if sys.platform == "darwin":
        info = await _probe_metal(whisper_cli)
        if info is not None:
            return info
    else:  # linux / win32
        info = await _probe_vulkan()
        if info is not None:
            return info
    return GpuInfo(backend="cpu", device_name="CPU", label="CPU (no GPU detected)")
