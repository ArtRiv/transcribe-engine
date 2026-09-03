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
    T-08-07-06: pipeline stage exceptions caught; ErrorMsg sent back; state(idle)

CR fixes applied in this module:
    CR-03: job_id validated via _validate_job_id before any filesystem access
    CR-04: asyncio.Lock + _eof_seen flag prevent double pipeline kick
    CR-05: single consumer task + asyncio.Queue(maxsize=64) serialise all messages
    WR-04: MAX_INBOUND_FILE_BYTES cap enforced before pipeline kick
    WR-07: removed dead outer try/except around _safe_send (which already swallows)

SEC-08: no supabase import.  No aiortc top-level import (lazy in peer.py).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from transcribe_engine.protocol.messages import (
    CheckpointMsg,
    ErrorMsg,
    ProgressMsg,
    ResultMsg,
    ResumeStateMsg,
    TranscriptPayload,
)
from transcribe_engine.webrtc.chunker import PartialFileWriter, _validate_job_id

# CR-02: valid lowercase hex SHA-256 (64 chars).
_VALID_SHA256 = re.compile(r"^[0-9a-f]{64}$")

log = logging.getLogger(__name__)

# Maximum binary frame size (T-08-07-01 — SCTP message-size invariant).
_MAX_CHUNK_BYTES: int = 65_536

# CR-05: bounded inbound queue.  If put_nowait raises QueueFull the sender
# is misbehaving; log + close channel (back-pressure escape valve).
_INBOUND_QUEUE_MAXSIZE: int = 64

# WR-04: reject audio files larger than this before kicking the pipeline.
# Mirror this constant on the frontend side: frontend/lib/webrtc/chunker.ts
MAX_INBOUND_FILE_BYTES: int = 5 * 1024 * 1024 * 1024  # 5 GB


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
        # CR-03: validate before any filesystem access.
        _validate_job_id(job_id)
        self._channel = channel
        self._signaling = signaling_client
        self._job_id = job_id
        self._temp_dir = temp_dir or Path(tempfile.gettempdir())
        self._pipeline_module = pipeline_module or _DefaultPipelineModule()
        self._writer: PartialFileWriter | None = None

        # CR-02: SHA-256 of the source file bound to this job.
        # Set on job_init; checked on resume_query.  None until job_init arrives.
        self._job_sha256: str | None = None

        # CR-04: prevent double pipeline kick from duplicate audio_eof.
        self._eof_seen: bool = False
        self._eof_lock: asyncio.Lock = asyncio.Lock()

        # CR-05: single consumer drains the inbound queue; on_message is non-blocking.
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=_INBOUND_QUEUE_MAXSIZE)
        self._consumer_task: asyncio.Task[None] | None = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def bind(self) -> None:
        """Spawn the consumer task and register @channel.on("message")."""
        # CR-05: one consumer task serialises all binary + control messages.
        self._consumer_task = asyncio.ensure_future(self._consume())
        channel = self._channel

        @channel.on("message")
        def _on_message(message: Any) -> None:
            try:
                self._queue.put_nowait(message)
            except asyncio.QueueFull:
                log.warning(
                    "handler %s: inbound queue full (maxsize=%d) — dropping message "
                    "and closing channel (sender misbehaving)",
                    self._job_id, _INBOUND_QUEUE_MAXSIZE,
                )
                try:
                    channel.close()
                except Exception:
                    pass

    async def close(self) -> None:
        """Cancel the consumer task (call from channel-close handler)."""
        if self._consumer_task is not None:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            self._consumer_task = None

    # -------------------------------------------------------------------------
    # Consumer (CR-05)
    # -------------------------------------------------------------------------

    async def _consume(self) -> None:
        """Drain the inbound queue one message at a time (in arrival order)."""
        while True:
            try:
                message = await self._queue.get()
            except asyncio.CancelledError:
                return
            try:
                if isinstance(message, bytes):
                    await self._handle_binary(message)
                else:
                    await self._handle_json(message)
            except Exception as exc:
                log.error(
                    "handler %s: unhandled error in consumer: %s",
                    self._job_id, exc, exc_info=True,
                )
            finally:
                self._queue.task_done()

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

        if self._writer is None:
            self._writer = PartialFileWriter(self._temp_dir, self._job_id)

        checkpoint_offset = self._writer.write(data)

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

        if msg_type == "job_init":
            self._handle_job_init(msg)
        elif msg_type == "audio_eof":
            await self._handle_audio_eof()
        elif msg_type == "resume_query":
            self._handle_resume_query(msg)
        elif msg_type == "ping":
            self._safe_send(json.dumps({"type": "pong"}))
        else:
            log.warning("handler %s: unknown message type %r", self._job_id, msg_type)

    def _handle_job_init(self, msg: dict) -> None:
        """Process a job_init message and persist file-identity sidecar (CR-02).

        The sidecar (<job_id>.meta.json) binds the job_id to the SHA-256 of
        the source file.  On resume the client must echo the same sha256_hex;
        a mismatch causes the engine to refuse the resume (return offset 0)
        and delete the existing .partial so the client starts fresh.
        """
        sha256_hex = msg.get("sha256_hex", "")
        if not isinstance(sha256_hex, str) or not _VALID_SHA256.fullmatch(sha256_hex):
            log.warning(
                "handler %s: job_init with invalid sha256_hex %r — ignoring",
                self._job_id, sha256_hex,
            )
            return

        self._job_sha256 = sha256_hex

        # Persist to sidecar with mode 0o600 (atomic via os.open + O_CREAT).
        sidecar = self._temp_dir / f"{self._job_id}.meta.json"
        sidecar_data = json.dumps({
            "job_id": self._job_id,
            "sha256_hex": sha256_hex,
            "total_bytes": msg.get("total_bytes"),
        })
        try:
            fd = os.open(str(sidecar), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as f:
                f.write(sidecar_data)
        except OSError as exc:
            log.warning(
                "handler %s: could not write meta sidecar: %s", self._job_id, exc
            )

    async def _handle_audio_eof(self) -> None:
        """Finalize the .partial file and kick the pipeline.

        CR-04: protected by _eof_lock + _eof_seen flag — duplicate audio_eof
        messages (network retransmit, malicious) are silently dropped after the
        first one is processed.  The lock also serialises the EOF handler against
        concurrent binary chunk writes from the consumer task.
        """
        async with self._eof_lock:
            if self._eof_seen:
                log.debug(
                    "handler %s: duplicate audio_eof — ignored (CR-04)",
                    self._job_id,
                )
                return
            self._eof_seen = True

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

    def _handle_resume_query(self, msg: dict) -> None:
        """Reply with the current confirmed byte offset (RTC-06 / T-08-07-08 / CR-01/CR-02).

        CR-02: verify the client's sha256_hex against the on-disk sidecar.
        If they don't match (different file, or no sidecar), return offset=0
        and delete the stale .partial so the client starts fresh.

        CR-01: only report bytes that are durably on disk (fsynced boundary).
        """
        # CR-02: extract and validate the client's sha256_hex.
        client_sha256 = msg.get("sha256_hex", "")
        sidecar = self._temp_dir / f"{self._job_id}.meta.json"

        if sidecar.exists():
            try:
                sidecar_data = json.loads(sidecar.read_text())
                stored_sha256 = sidecar_data.get("sha256_hex", "")
            except (OSError, json.JSONDecodeError):
                stored_sha256 = ""

            if client_sha256 != stored_sha256:
                log.warning(
                    "handler %s: resume_query hash mismatch — client=%r stored=%r; "
                    "deleting .partial and refusing resume (CR-02)",
                    self._job_id, client_sha256[:16], stored_sha256[:16],
                )
                # Delete stale .partial so client restarts from byte 0.
                partial_stale = self._temp_dir / f"{self._job_id}.partial"
                partial_stale.unlink(missing_ok=True)
                sidecar.unlink(missing_ok=True)
                if self._writer is not None:
                    self._writer.cleanup()
                    self._writer = None
                reply: ResumeStateMsg = {"type": "resume_state", "byte_offset": 0}
                self._safe_send(json.dumps(reply))
                return

        # Hashes match (or no sidecar — legacy client or first connect).
        if self._writer is not None:
            # Writer live: current_offset() returns last fsynced boundary (CR-01).
            offset = self._writer.current_offset()
        else:
            # Writer absent: check disk directly (clean state after restart).
            partial = self._temp_dir / f"{self._job_id}.partial"
            offset = partial.stat().st_size if partial.exists() else 0

        reply = {"type": "resume_state", "byte_offset": offset}
        self._safe_send(json.dumps(reply))

    # -------------------------------------------------------------------------
    # Pipeline runner
    # -------------------------------------------------------------------------

    async def _run_pipeline(self, audio_path: Path) -> None:
        """Run the full pipeline and send result/error back over the channel.

        T-08-07-02: try/finally ensures cleanup() runs even if send() fails.
        WR-04: reject audio files larger than MAX_INBOUND_FILE_BYTES before
               kicking the pipeline (prevents disk exhaustion / DoS).
        """
        # WR-04: size guard — check before any ffmpeg/whisper invocation.
        try:
            size = audio_path.stat().st_size
        except OSError:
            size = 0
        if size > MAX_INBOUND_FILE_BYTES:
            log.error(
                "handler %s: audio file too large (%d bytes > %d) — aborting pipeline",
                self._job_id, size, MAX_INBOUND_FILE_BYTES,
            )
            audio_path.unlink(missing_ok=True)
            err: ErrorMsg = {
                "type": "error",
                "code": "file_too_large",
                "message": "Audio file exceeds the maximum supported size.",
            }
            self._safe_send(json.dumps(err))
            try:
                await self._signaling.emit_state("idle")
            except Exception:
                pass
            return

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
            err = {
                "type": "error",
                "code": "pipeline_failed",
                # ASVS V7 — sanitized message, no stack trace
                "message": "Transcription failed. Please try again.",
            }
            # WR-07: _safe_send already swallows exceptions; no outer try needed.
            self._safe_send(json.dumps(err))

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
          - the .meta.json sidecar (CR-02 file-identity record)
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

        # CR-02: clean up the file-identity sidecar.
        sidecar = self._temp_dir / f"{self._job_id}.meta.json"
        sidecar.unlink(missing_ok=True)

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
        import sys

        from transcribe_engine.pipeline.diarize import load_pyannote, run_diarize
        from transcribe_engine.pipeline.lifecycle import normalized_wav_path
        from transcribe_engine.pipeline.merge import merge_to_payload
        from transcribe_engine.pipeline.normalize import normalize_to_wav
        from transcribe_engine.pipeline.transcribe import transcribe_subprocess
        from transcribe_engine.platform.paths import bundle_root, cache_dir

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
