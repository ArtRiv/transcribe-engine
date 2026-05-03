"""pyannote-CPU diarization (CORE-06 diar half; L2 + Pitfall 2).

Pipeline.from_pretrained → pipeline.to(torch.device("cpu")) → assert
device.type == "cpu". Both load_pyannote and the inference call are blocking
(PyTorch C++ under the hood); always wrap in `loop.run_in_executor`.

Engine port of v1 backend/app/pipeline/diarize.py.
Delta from v1: settings: Any parameter replaced with hf_token: str | None.
The CPU pin + assertion is carried verbatim (Pitfall 2 mitigation — NON-NEGOTIABLE).
"""

from __future__ import annotations

import asyncio
from typing import Any

import torch
from pyannote.audio import Pipeline


async def load_pyannote(hf_token: str | None) -> Any:
    """Load `pyannote/speaker-diarization-3.1` and pin it to CPU.

    Wrapped in `run_in_executor` because `Pipeline.from_pretrained` does
    blocking I/O (HF cache lookup + torch tensor materialization). The CPU
    pin is asserted via `pipe.device.type == "cpu"` after `pipe.to(...)`
    sets `self.device` (Pitfall 2 — guards against an accidental CUDA torch
    wheel). pyannote.audio 3.x's `Pipeline` is NOT a `torch.nn.Module`; its
    `.parameters()` returns a dict of hyperparameters, not torch params, so
    the canonical device introspection on the high-level Pipeline is the
    `.device` attribute (set inside `Pipeline.to()` after walking every
    sub-pipeline, model, and inference — see pyannote.audio core/pipeline.py).

    Engine port: hf_token is passed directly by the caller (the engine's
    tray/picker flow stores the user-provided HF token; __main__ reads it
    from on-disk state and passes here). Engine has no Settings BaseModel.
    """

    def _load() -> Any:
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token or None,
        )
        # L2: explicit CPU pin
        pipe.to(torch.device("cpu"))
        # Pitfall 2: assert pin took effect via the public Pipeline.device.
        dev = getattr(getattr(pipe, "device", None), "type", None)
        assert dev == "cpu", f"pyannote pipeline not on CPU after .to(cpu): {dev!r}"
        return pipe

    return await asyncio.get_running_loop().run_in_executor(None, _load)


async def diarize(
    pipeline: Any,
    wav_path: str,
    num_speakers: int | None = None,
) -> list[dict[str, Any]]:
    """Run the diarization pipeline; return turns sorted by `start`.

    Returns ``[{start, end, speaker}, ...]`` (start/end as floats in seconds,
    speaker as the raw pyannote label, e.g., ``"SPEAKER_00"``). Wrapped in
    ``run_in_executor`` so the engine event loop is not blocked by the
    PyTorch call (which can run for tens of seconds on CPU).
    """

    def _run() -> list[dict[str, Any]]:
        kwargs: dict[str, Any] = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        diarization = pipeline(wav_path, **kwargs)
        turns = [
            {
                "start": float(turn.start),
                "end": float(turn.end),
                "speaker": str(speaker),
            }
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]
        return sorted(turns, key=lambda t: t["start"])

    return await asyncio.get_running_loop().run_in_executor(None, _run)
