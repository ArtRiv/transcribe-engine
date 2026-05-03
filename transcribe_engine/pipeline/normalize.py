"""ffmpeg subprocess wrapper: any container → 16 kHz mono PCM WAV (CORE-05).

Subprocess invariants (RESEARCH.md §1473): args as a list, never shell=True,
never f-string interpolation. Path arg is a UUID-derived filename.

Engine port of v1 backend/app/pipeline/normalize.py — carried verbatim.
No v1-specific imports; the function signature is cross-platform.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path


async def normalize_to_wav(src: Path, dst: Path) -> None:
    """Decode any container to 16 kHz mono PCM WAV. Raises on ffmpeg failure."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(dst),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}): {stderr.decode()[:500]}")
