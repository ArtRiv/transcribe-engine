"""webrtc/handler.py — InboundJobHandler (Plan 08-07 Task 2).

Manages the full inbound job lifecycle on one data channel:
    1. Accumulate binary audio bytes → .partial temp file (via PartialFileWriter)
    2. On audio_eof: finalize .partial → kick pipeline
    3. During pipeline: emit progress messages over data channel
    4. On completion: send result JSON (or error) over data channel
    5. Finally: emit state=idle via signaling; cleanup temp file

Resume (RTC-06):
    - On resume_query → reply resume_state(byte_offset=current_offset)
    - Verified by test_resume_protocol.py

Threat mitigations:
    T-08-07-01: assert len(message) <= 65536 for each binary chunk (drop + log)
    T-08-07-02: try/finally cleanup in _run_pipeline (deletes partial/final audio)
    T-08-07-03: inbound chunk queue cap via _MAX_INBOUND_CHUNKS (defense-in-depth)
    T-08-07-06: pipeline stage exceptions caught; ErrorMsg sent back; state(idle)

SEC-08: no supabase import.  No aiortc top-level import (lazy in peer.py).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from transcribe_engine.protocol.messages import (
    CheckpointMsg,
    ErrorMsg,
    ProgressMsg,
    ResumeStateMsg,
    ResultMsg,
    TranscriptPayload,
)
from transcribe_engine.webrtc.chunker import PartialFileWriter

log = logging.getLogger(__name__)

# T-08-07-03 defense-in-depth: drop binary chunks above this cap.
# The frontend's backpressure should make this unreachable in practice.
_MAX_INBOUND_CHUNKS: int = 16

# Maximum binary frame size (T-08-07-01 — SCTP message-size invariant).
_MAX_CHUNK_BYTES: int = 65_536


class InboundJobHandler:
    """Handle one transcription job over a single RTCDataChannel.

    Constructor:
        channel:           aiortc (or fake) RTCDataChannel — `channel.send(str)` is used.
        signaling_client:  EngineSignalingClient (or mock); `emit_state(value)` is awaited.
        job_id:            Unique identifier for this job (used for .partial filename).
        pipeline_module:   Module-like object with run_pipeline() — injected for testability.
                           Defaults to a module-level shim that calls the real pipeline stages.
        temp_dir:          Base temp directory for .partial files.

    Usage:
        handler = InboundJobHandler(channel, signaling_client, job_id="uuid-123")
        handler.bind()   # registers @channel.on("message")
    """

    def __init__(
        self,
        channel: Any,
        signaling_client: Any,
        job_id: str,
        *,
        pipeline_module: Any = None,
        temp_dir: Path | None = None,
    ) -> None:
        self._channel = channel
        self._signaling = signaling_client
        self._job_id = job_id
        self._temp_dir = temp_dir or Path(tempfile.gettempdir())
        self._pipeline_module = pipeline_module or _DefaultPipelineModule()
        self._writer: PartialFileWriter | None = None
        self._pending_chunks: int = 0  # T-08-07-03 counter

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def bind(self) -> None:
        """Register @channel.on("message") to route inbound messages here."""
        channel = self._channel

        @channel.on("message")
        def _on_message(message: Any) -> None:
            # Dispatch synchronously; schedule coroutines via create_task.
            if isinstance(message, bytes):
                asyncio.ensure_future(self._handle_binary(message))
            else:
                asyncio.ensure_future(self._handle_json(message))

    # -------------------------------------------------------------------------
    # Internal handlers
    # -------------------------------------------------------------------------

    async def _handle_binary(self, data: bytes) -> None:
        """Append a binary audio chunk to the .partial file."""
        # T-08-07-01: drop oversized chunks
        if len(data) > _MAX_CHUNK_BYTES:
            log.warning(
                "handler %s: dropping oversized chunk (%d bytes > %d) — T-08-07-01",
                self._job_id, len(data), _MAX_CHUNK_BYTES,
            )
            return

        # T-08-07-03: drop if too many pending chunks (defense-in-depth)
        if self._pending_chunks >= _MAX_INBOUND_CHUNKS:
            log.warning(
                "handler %s: dropping chunk — inbound queue cap (%d) reached (T-08-07-03)",
                self._job_id, _MAX_INBOUND_CHUNKS,
            )
            return

        if self._writer is None:
            self._writer = PartialFileWriter(self._temp_dir, self._job_id)

        self._pending_chunks += 1
        try:
            checkpoint_offset = self._writer.write(data)
        finally:
            self._pending_chunks -= 1

        if checkpoint_offset is not None:
            msg: CheckpointMsg = {"type": "checkpoint", "byte_offset": checkpoint_offset}
            self._safe_send(json.dumps(msg))

    async def _handle_json(self, raw: str) -> None:
        """Handle a JSON control message from the frontend."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("handler %s: received non-JSON text message", self._job_id)
            return

        msg_type = msg.get("type")

        if msg_type == "audio_eof":
            await self._handle_audio_eof()
        elif msg_type == "resume_query":
            self._handle_resume_query()
        elif msg_type == "ping":
            self._safe_send(json.dumps({"type": "pong"}))
        else:
            log.warning("handler %s: unknown message type %r", self._job_id, msg_type)

    async def _handle_audio_eof(self) -> None:
        """Finalize the .partial file and kick the pipeline."""
        if self._writer is None:
            log.error(
                "handler %s: audio_eof received but no writer (no bytes received?)",
                self._job_id,
            )
            err: ErrorMsg = {
                "type": "error",
                "code": "no_audio_received",
                "message": "No audio bytes received before end-of-file.",
            }
            self._safe_send(json.dumps(err))
            return

        try:
            final_path = self._writer.finalize()
            self._writer = None  # finalized; writer is done
        except RuntimeError as exc:
            log.error("handler %s: finalize failed: %s", self._job_id, exc)
            err = {
                "type": "error",
                "code": "finalize_failed",
                "message": "Failed to finalize audio file.",
            }
            self._safe_send(json.dumps(err))
            return

        await self._signaling.emit_state("transcribing")
        asyncio.ensure_future(self._run_pipeline(final_path))

    def _handle_resume_query(self) -> None:
        """Reply with the current confirmed byte offset (RTC-06 / T-08-07-08)."""
        offset = self._writer.current_offset() if self._writer else 0

        # Also check for an existing .partial on disk for a previous writer
        if self._writer is None:
            partial = self._temp_dir / f"{self._job_id}.partial"
            if partial.exists():
                offset = partial.stat().st_size

        reply: ResumeStateMsg = {"type": "resume_state", "byte_offset": offset}
        self._safe_send(json.dumps(reply))

    # -------------------------------------------------------------------------
    # Pipeline runner
    # -------------------------------------------------------------------------

    async def _run_pipeline(self, audio_path: Path) -> None:
        """Run the full pipeline and send result/error back over the channel.

        T-08-07-02: try/finally ensures cleanup() runs even if send() fails.
        """
        try:
            await self._signaling.emit_state("loading_model")

            def _on_progress(stage: str, fraction: float) -> None:
                msg: ProgressMsg = {
                    "type": "progress",
                    "stage": stage,  # type: ignore[arg-type]
                    "fraction": fraction,
                }
                self._safe_send(json.dumps(msg))

            result = await self._pipeline_module.run_pipeline(
                audio_path,
                job_id=self._job_id,
                on_progress=_on_progress,
            )

            result_msg: ResultMsg = {"type": "result", "transcript": result}
            self._safe_send(json.dumps(result_msg))

        except Exception as exc:
            log.error(
                "handler %s: pipeline failed: %s", self._job_id, exc, exc_info=True
            )
            err: ErrorMsg = {
                "type": "error",
                "code": "pipeline_failed",
                # ASVS V7 — sanitized message, no stack trace
                "message": "Transcription failed. Please try again.",
            }
            # Swallow send errors here (browser-tab-closed case — T-08-07-07)
            try:
                self._safe_send(json.dumps(err))
            except Exception as send_err:
                log.debug(
                    "handler %s: error send failed (channel closed?): %s",
                    self._job_id, send_err,
                )

        finally:
            # D-08: always emit idle + clean up audio file
            try:
                await self._signaling.emit_state("idle")
            except Exception as se:
                log.debug("handler %s: emit_state(idle) failed: %s", self._job_id, se)

            self._cleanup(audio_path)

    def _cleanup(self, audio_path: Path | None = None) -> None:
        """Delete temp audio files (T-08-07-02 — SEC-08 D-08 hygiene).

        Deletes:
          - audio_path (the finalized audio file, pipeline input)
          - the .partial sidecar if still present (resume-fallback cleanup)
        """
        if audio_path is not None:
            try:
                audio_path.unlink(missing_ok=True)
            except OSError as e:
                log.warning("handler %s: cleanup audio_path failed: %s", self._job_id, e)

        # Defensive: also remove any lingering .partial (e.g. finalize() raised)
        if self._writer is not None:
            self._writer.cleanup()
            self._writer = None
        else:
            partial = self._temp_dir / f"{self._job_id}.partial"
            partial.unlink(missing_ok=True)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _safe_send(self, data: str) -> None:
        """send() over the data channel, swallowing closed-channel errors.

        The browser-tab-closed case (T-08-07-07) raises when the channel is
        already closed; we log + continue so the finally cleanup still runs.
        """
        try:
            self._channel.send(data)
        except Exception as exc:
            log.debug(
                "handler %s: channel.send() failed (channel closed?): %s",
                self._job_id, exc,
            )


# ---------------------------------------------------------------------------
# Default pipeline module — wraps the real Phase 7 pipeline stages.
# Under MOCK_ENGINE=1 (test env) returns a stub payload without GPU.
# ---------------------------------------------------------------------------

class _DefaultPipelineModule:
    """Thin shim that wraps the real pipeline under normal conditions.

    Tests inject a mock object instead of using this class.

    Under MOCK_ENGINE=1 (set by conftest.py autouse fixture) the real
    pipeline stages are skipped and a stub TranscriptPayload is returned.
    This lets handler tests run without a GPU.
    """

    async def run_pipeline(
        self,
        audio_path: Path,
        *,
        job_id: str,
        on_progress: Callable[[str, float], None],
    ) -> TranscriptPayload:
        if os.environ.get("MOCK_ENGINE") == "1":
            return self._mock_result(audio_path)

        # Real pipeline — Phase 7 modules.
        # Import lazily (heavy deps: torch, pyannote, whisper.cpp subprocess).
        from transcribe_engine.pipeline.normalize import normalize_to_wav
        from transcribe_engine.pipeline.lifecycle import normalized_wav_path
        from transcribe_engine.pipeline.transcribe import transcribe_subprocess
        from transcribe_engine.pipeline.diarize import load_pyannote, run_diarize
        from transcribe_engine.pipeline.merge import merge_to_payload
        from transcribe_engine.platform.paths import bundle_root, cache_dir
        import sys

        on_progress("normalize", 0.0)
        wav = normalized_wav_path(audio_path)
        await normalize_to_wav(audio_path, wav)
        on_progress("normalize", 1.0)

        on_progress("transcribe", 0.0)
        whisper_bin = bundle_root() / (
            "whisper-cli.exe" if sys.platform == "win32" else "whisper-cli"
        )
        # Load model path from cache_dir — use fast preset as default
        model_path = str(cache_dir() / "ggml-small.bin")
        asr = await transcribe_subprocess(
            str(whisper_bin),
            model_path,
            wav,
            on_progress=lambda frac: on_progress("transcribe", frac),
        )
        on_progress("transcribe", 1.0)

        on_progress("diarize", 0.0)
        pipe = await load_pyannote(hf_token=None)
        diar = await run_diarize(pipe, wav)
        on_progress("diarize", 1.0)

        on_progress("merge", 0.0)
        payload = merge_to_payload(asr, diar)
        on_progress("merge", 1.0)

        # Cleanup intermediate wav
        wav.unlink(missing_ok=True)

        return payload  # type: ignore[return-value]

    @staticmethod
    def _mock_result(audio_path: Path) -> TranscriptPayload:
        """Stub TranscriptPayload for MOCK_ENGINE=1 mode."""
        size = audio_path.stat().st_size if audio_path.exists() else 0
        return {
            "version": 1,
            "language": "en",
            "duration_sec": float(size) / 16000 / 2,  # rough estimate: 16kHz 16-bit
            "speakers": [{"id": "S0", "label": "Speaker 1"}],
            "segments": [
                {
                    "id": "seg_0000",
                    "start": 0.0,
                    "end": 1.0,
                    "speaker": "S0",
                    "text": "[Mock transcript — MOCK_ENGINE=1]",
                }
            ],
        }
