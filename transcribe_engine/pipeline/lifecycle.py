"""Pipeline lifecycle — disk cleanup + per-backend GPU memory release.

Lifted from v1 backend/app/pipeline/lifecycle.py. Engine port:
- keeps normalized_wav_path helper (single source of truth for .norm.wav sibling)
- keeps cleanup_job_files (idempotent log-and-swallow pattern)
- drops sweep_orphans (TUS-specific; engine has no uploads dir)
- adds release_models() (per-backend cache release between jobs)
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def normalized_wav_path(work_path: Path) -> Path:
    """Path of the normalized 16 kHz mono PCM WAV derived from `work_path`.

    Single source of truth shared by the job runner (where ffmpeg writes
    it) and cleanup_job_files (where the worker's outer finally deletes
    it). The path is ALWAYS distinct from `work_path`, even when the upload
    is itself a .wav — `with_name(f"{stem}.norm.wav")` produces a sibling
    file whose name differs from any plausible upload extension.

    The `.norm.wav` suffix is load-bearing: passing the same path to
    ffmpeg as both -i input and output triggers
    "Output X same as Input #0 - exiting / FFmpeg cannot edit existing
    files in-place", which previously crashed normalize_to_wav for every
    .wav upload.
    """
    return work_path.with_name(f"{work_path.stem}.norm.wav")


async def cleanup_job_files(job: Any) -> None:
    """Idempotent cleanup. Safe to call multiple times. Never raises.

    Deletes both the original upload (`job.work_path`) and the intermediate
    16 kHz WAV (`normalized_wav_path(job.work_path)`). Missing files are
    silently skipped via `unlink(missing_ok=True)`; OS errors are logged +
    swallowed so the worker's `finally` cleanup never escalates a
    partial-failure into a crash.

    `job` is a per-job dataclass with at least `work_path: Path`.
    Engine's exact Job shape is defined in plan 07-06 (__main__) when the
    integration wires the actual job message.
    """
    for p in (job.work_path, normalized_wav_path(job.work_path)):
        try:
            p.unlink(missing_ok=True)
        except OSError as e:
            log.warning("cleanup failed for %s: %s", p, e)


def release_models(backend: str) -> None:
    """Per-backend GPU memory release between jobs.

    Vulkan: no torch backend control — best-effort GC only.
    Metal: torch.mps.empty_cache() if available.
    CPU: GC only.

    Wrapped in try/except per backend, log-but-don't-raise (PATTERNS idiom).
    Cross-backend equivalent of v1's `torch.cuda.empty_cache()` between jobs
    (CLAUDE.md §Architecture).
    """
    try:
        if backend == "metal":
            # torch.mps.empty_cache() exists on torch>=1.12 with MPS support
            import torch
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        # Vulkan: no torch interface; rely on whisper.cpp's own cleanup between subprocess calls.
        # CPU: nothing to release; GC handles it.
    except Exception as e:
        log.warning("release_models(%s) failed (non-fatal): %s", backend, e)
    finally:
        gc.collect()
