"""Whisper.cpp subprocess wrapper — engine port of v1 backend/app/pipeline/transcribe.py.

Cross-platform refit: whisper-cli resolved from PyInstaller _MEIPASS
(R-04, Pitfall 6). VAD-missing error points to tray Manage models...

Spawns the locally-built `whisper-cli` binary with `--output-json-full`,
streams its stdout for `[HH:MM:SS.mmm --> HH:MM:SS.mmm]` segment lines (intra-
stage progress at ≥5 s cadence — D3), and parses the side-car JSON file using
the schema locked Wave 0 in `docs/DEPENDENCIES.md`.

Cooperative cancel (D4): when ``cancel_event`` is set, we DO NOT SIGKILL the
subprocess mid-inference — that's the documented cause of Vulkan VRAM leaks
on this stack (Pitfall 1). The streaming loop just stops reading intra-
segment progress; the subprocess runs to natural completion and the parent
worker honours the cancel between stages. The lone allowed kill
lives in the ``finally`` block as a last-resort cleanup if this function
exits abnormally with the subprocess still alive (terminate → 10 s wait →
SIGKILL).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from transcribe_engine.platform.paths import bundle_root

# Stdout pattern for intra-stage progress: "[00:01:23.450 --> 00:01:27.890]"
_TS_RE = re.compile(
    r"\[(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})\]"
)


def _whisper_cli_path() -> Path:
    """Resolve whisper-cli from the PyInstaller bundle root (R-04, Pitfall 6).

    Bundled mode: `_MEIPASS / whisper-cli` (Linux/Mac) or `_MEIPASS / whisper-cli.exe` (Win).
    Source mode: bundle_root() falls back to engine package parent — caller passes a path.

    The .exe suffix on Windows is mandatory: subprocess.exec() without it fails
    with FileNotFoundError on Windows even when the binary exists (Pitfall 6).
    """
    name = "whisper-cli.exe" if sys.platform == "win32" else "whisper-cli"
    return bundle_root() / name


def _ts_to_seconds(h: str | int, m: str | int, s: str | int, ms: str | int) -> float:
    """Convert (HH, MM, SS, mmm) → float seconds."""
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def _threads() -> int:
    """Return the thread count for whisper-cli's --threads flag.

    `min(8, ncpu // 2)` — Whisper.cpp scales sublinearly past ~8 threads on
    the Vulkan backend (the ASR is GPU-bound), and leaving half the host CPU
    free keeps pyannote-on-CPU diarization responsive on the same machine.
    """
    return min(8, (os.cpu_count() or 4) // 2)


_TAIL_BYTES = 8 * 1024  # bytes of subprocess output retained for error context


def _build_whisper_args(
    bin_path: str,
    model_path: str,
    wav_path: Path,
    json_out: Path,
    *,
    language: str | None,
    vad_model_path: str,
) -> list[str]:
    """Build the whisper-cli argv list. Pure function — extracted for test
    coverage of the language-flag default behaviour.

    whisper-cli's `--language` defaults to "en" when omitted (verified via
    `whisper-cli --help`). Skipping the flag for non-English audio causes
    Portuguese (and other) inputs to be transcribed as English. Always pass
    the flag, defaulting to "auto" so detection runs.
    """
    return [
        bin_path,
        "--model",
        model_path,
        "--file",
        str(wav_path),
        "--output-json-full",
        "--output-file",
        str(json_out.with_suffix("")),
        "--print-progress",
        "--vad",
        "--vad-model",
        vad_model_path,
        "--threads",
        str(_threads()),
        "--language",
        language if language else "auto",
    ]


async def transcribe_subprocess(
    bin_path: str,
    model_path: str,
    wav_path: Path,
    *,
    language: str | None = None,
    on_progress: Callable[[float], Awaitable[None]] | None = None,
    cancel_event: asyncio.Event | None = None,
    vad_model_path: str = "",
) -> dict:
    """Run whisper-cli on a 16 kHz mono WAV; return the normalized payload.

    See module docstring for cancel semantics. Returns the dict produced by
    `_normalize_payload` (keys: `language`, `segments`).

    ``vad_model_path`` is required (Pitfall 5 — silence-hallucination
    mitigation): whisper.cpp 1.8+'s ``--vad`` flag does NOT bring its own
    model; passing ``--vad`` without ``--vad-model FNAME`` fails with
    ``failed to compute VAD`` and a non-zero exit. The caller resolves the
    path; an empty value here is a configuration error and we fail fast
    before spawning the subprocess.

    ``bin_path`` should be resolved by the call site via ``_whisper_cli_path()``
    for bundle-mode operation. Accepting it as a parameter keeps the function
    pure and testable (callers can inject a mock whisper-cli binary).
    """
    if not vad_model_path:
        raise RuntimeError(
            "transcribe_subprocess: vad_model_path is empty. whisper.cpp 1.8+ "
            "requires a Silero VAD model when --vad is enabled. "
            "Open the engine tray menu → Manage models... → "
            "download a tier (Fast / Average / Best) before transcribing."
        )

    json_out = wav_path.with_suffix(".json")
    args = _build_whisper_args(
        bin_path,
        model_path,
        wav_path,
        json_out,
        language=language,
        vad_model_path=vad_model_path,
    )

    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    last_progress_write = 0.0
    output_tail: bytearray = bytearray()
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                # D4: cooperative cancel — do NOT kill mid-inference. Stop
                # reading progress and let the subprocess exit naturally; the
                # parent worker checks `cancel_event` between stages.
                pass
            line = await proc.stdout.readline()  # type: ignore[union-attr]
            if not line:
                break
            # Retain a rolling tail of stdout+stderr (combined via
            # stderr=STDOUT) so RuntimeError on non-zero exit can show what
            # whisper-cli actually said instead of just the bare exit code.
            output_tail.extend(line)
            if len(output_tail) > _TAIL_BYTES:
                del output_tail[: len(output_tail) - _TAIL_BYTES]
            text = line.decode(errors="ignore")
            m = _TS_RE.search(text)
            if m and on_progress is not None:
                # Group indices 4-7 are the 'to' timestamp (HH, MM, SS, mmm).
                seconds_processed = _ts_to_seconds(*m.groups()[4:])
                now = time.monotonic()
                if now - last_progress_write >= 5.0:  # D3 cadence
                    await on_progress(seconds_processed)
                    last_progress_write = now
        await proc.wait()
        if proc.returncode != 0:
            tail = output_tail.decode(errors="ignore").strip()
            raise RuntimeError(
                f"whisper-cli exit {proc.returncode}\n"
                f"--- last {len(output_tail)} bytes of combined stdout+stderr ---\n"
                f"{tail or '<no output captured>'}"
            )
    finally:
        # Last-resort cleanup if we exited abnormally (exception in the
        # streaming loop, parent task cancelled, etc.) and the subprocess is
        # still running. Terminate gracefully, then wait up to 10 s, then
        # SIGKILL. This is the ONLY allowed last-resort kill in this module
        # (T-7-pipeline-01 mitigation — Vulkan VRAM leak avoidance).
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=10)
            except (asyncio.TimeoutError, ProcessLookupError):
                proc.kill()
                await proc.wait()

    payload = json.loads(json_out.read_text())
    json_out.unlink(missing_ok=True)  # clean intermediate
    return _normalize_payload(payload)


def _normalize_payload(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert whisper.cpp's --output-json-full into our internal Segment list.

    The locked schema: top-level ``transcription[i]`` carries ``offsets`` (ms)
    and ``tokens[j]`` carries ``id`` + ``offsets`` + ``p`` + ``text``
    (special-token markers like ``[_BEG_]`` carry ``id >= 50257`` and are
    filtered out).
    """
    segs: list[dict[str, Any]] = []
    for entry in raw.get("transcription", []):
        words: list[dict[str, Any]] = []
        for tok in entry.get("tokens", []):
            tid = tok.get("id", -1)
            # Filter special tokens (BOS, EOS, language, timestamps);
            # whisper vocab id range for content tokens is [0, 50257).
            if tid < 0 or tid >= 50257:
                continue
            text = tok.get("text", "").strip()
            if not text or text.startswith("[_"):
                continue
            words.append(
                {
                    "w": text,
                    "s": tok["offsets"]["from"] / 1000.0,
                    "e": tok["offsets"]["to"] / 1000.0,
                    "p": tok.get("p", 1.0),
                }
            )
        segs.append(
            {
                "start": entry["offsets"]["from"] / 1000.0,
                "end": entry["offsets"]["to"] / 1000.0,
                "text": entry["text"].strip(),
                "words": words,
            }
        )
    return {
        "language": raw.get("result", {}).get("language", "en"),
        "segments": segs,
    }
