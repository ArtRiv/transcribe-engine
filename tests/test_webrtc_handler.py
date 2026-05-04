"""Tests for webrtc/handler.py — InboundJobHandler (Plan 08-07 Task 2).

All tests run under MOCK_ENGINE=1 (autouse fixture from conftest.py) — no GPU required.

Uses a FakeDataChannel (synchronous send() recorder + message injection) and a
FakeSignalingClient to capture emit_state() calls without a real Realtime client.

Coverage:
    - Binary chunks are written to .partial via PartialFileWriter
    - Checkpoint message sent when 10 MB boundary crossed
    - audio_eof triggers pipeline + result returned over channel
    - Pipeline failure sends ErrorMsg
    - pipeline error does NOT prevent state(idle) + cleanup (D-08)
    - resume_query returns current byte offset
    - Oversized chunk is dropped (T-08-07-01)
    - Browser-tab-closed: result send failure is swallowed (T-08-07-07)
"""
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pytest

from transcribe_engine.webrtc.handler import InboundJobHandler

# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

_1KB = 1024
_10MB = 10 * 1024 * 1024


class FakeDataChannel:
    """Minimal in-memory RTCDataChannel stand-in for handler tests."""

    def __init__(self) -> None:
        self.sent: list[str] = []
        self._handlers: dict[str, list[Any]] = {}

    def on(self, event: str) -> Any:
        """Decorator registering a handler for *event*."""
        def decorator(fn: Any) -> Any:
            self._handlers.setdefault(event, []).append(fn)
            return fn
        return decorator

    def send(self, data: str) -> None:
        self.sent.append(data)

    def inject(self, message: Any) -> None:
        """Inject a message as if received from the remote peer."""
        for h in self._handlers.get("message", []):
            if asyncio.iscoroutinefunction(h):
                asyncio.get_event_loop().run_until_complete(h(message))
            else:
                h(message)

    def sent_types(self) -> list[str]:
        return [json.loads(m)["type"] for m in self.sent if isinstance(m, str)]


class FakeSignalingClient:
    """Captures emit_state() calls without a real Realtime client."""

    def __init__(self) -> None:
        self.states: list[str] = []

    async def emit_state(self, value: str) -> None:
        self.states.append(value)


class MockPipeline:
    """Configurable fake pipeline — records invocations; returns stub payload."""

    def __init__(
        self,
        *,
        fail: bool = False,
        recorded_size: int | None = None,
    ) -> None:
        self.calls: list[dict] = []
        self._fail = fail
        self._recorded_size: int | None = None

    async def run_pipeline(
        self,
        audio_path: Path,
        *,
        job_id: str,
        on_progress: Any,
    ) -> dict:
        self.calls.append({"audio_path": audio_path, "job_id": job_id})
        self._recorded_size = audio_path.stat().st_size if audio_path.exists() else None
        if self._fail:
            raise RuntimeError("mock pipeline failure")
        on_progress("normalize", 1.0)
        on_progress("merge", 1.0)
        return {
            "version": 1,
            "language": "en",
            "duration_sec": 1.0,
            "speakers": [{"id": "S0", "label": "Speaker 1"}],
            "segments": [],
        }


# ---------------------------------------------------------------------------
# Async helper to flush create_task() coroutines
# ---------------------------------------------------------------------------

async def _drain() -> None:
    """Yield control so all pending asyncio tasks run."""
    for _ in range(10):
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_binary_chunks_written_to_partial(tmp_path: Path) -> None:
    """Binary messages are written to .partial; file grows accordingly."""
    ch = FakeDataChannel()
    sig = FakeSignalingClient()
    h = InboundJobHandler(ch, sig, "job-001", temp_dir=tmp_path)
    h.bind()

    ch.inject(b"A" * 1024)
    await _drain()

    partial = tmp_path / "job-001.partial"
    assert partial.exists(), ".partial must be created after first binary chunk"
    assert partial.stat().st_size == 1024


@pytest.mark.asyncio
async def test_checkpoint_emitted_at_10mb_boundary(tmp_path: Path) -> None:
    """Sending 10 MB of data results in a checkpoint message on the channel."""
    ch = FakeDataChannel()
    sig = FakeSignalingClient()
    h = InboundJobHandler(ch, sig, "job-chkpt", temp_dir=tmp_path)
    h.bind()

    chunk = b"B" * 65536  # 64 KB
    # 10 MB / 64 KB = 160 chunks
    for _ in range(160):
        ch.inject(chunk)
        await _drain()

    checkpoints = [m for m in ch.sent if json.loads(m)["type"] == "checkpoint"]
    assert len(checkpoints) >= 1, "Must emit at least 1 checkpoint at 10 MB"
    offsets = [json.loads(m)["byte_offset"] for m in checkpoints]
    assert any(o >= _10MB for o in offsets)


@pytest.mark.asyncio
async def test_audio_eof_triggers_pipeline_and_result(tmp_path: Path) -> None:
    """audio_eof finalizes writer, kicks pipeline, channel receives result."""
    ch = FakeDataChannel()
    sig = FakeSignalingClient()
    mock_pl = MockPipeline()
    h = InboundJobHandler(ch, sig, "job-eof", pipeline_module=mock_pl, temp_dir=tmp_path)
    h.bind()

    # Send some audio
    ch.inject(b"C" * 4096)
    await _drain()

    # Send EOF
    ch.inject(json.dumps({"type": "audio_eof", "total_bytes": 4096}))
    await _drain()

    assert len(mock_pl.calls) == 1, "Pipeline must be called exactly once"
    types = ch.sent_types()
    assert "result" in types, f"result message must be sent; got {types}"
    assert "idle" in sig.states, "state(idle) must be emitted after pipeline"


@pytest.mark.asyncio
async def test_pipeline_failure_sends_error_and_idle(tmp_path: Path) -> None:
    """Pipeline failure → ErrorMsg sent; state(idle) emitted; cleanup runs."""
    ch = FakeDataChannel()
    sig = FakeSignalingClient()
    mock_pl = MockPipeline(fail=True)
    h = InboundJobHandler(ch, sig, "job-fail", pipeline_module=mock_pl, temp_dir=tmp_path)
    h.bind()

    ch.inject(b"D" * 1024)
    await _drain()
    ch.inject(json.dumps({"type": "audio_eof", "total_bytes": 1024}))
    await _drain()

    types = ch.sent_types()
    assert "error" in types, f"error message must be sent on pipeline failure; got {types}"
    # state(idle) must fire even on failure (D-08)
    assert "idle" in sig.states, "state(idle) must be emitted even on failure"
    # Audio file must be cleaned up (T-08-07-02)
    final = tmp_path / "job-fail"
    assert not final.exists(), "Final audio file must be deleted after failure"


@pytest.mark.asyncio
async def test_pipeline_failure_swallows_send_error(tmp_path: Path) -> None:
    """If channel.send() raises during error reply, cleanup still runs."""
    class FailOnSendChannel(FakeDataChannel):
        _fail_count = 0
        def send(self, data: str) -> None:
            self._fail_count += 1
            if self._fail_count > 1:
                # First send is progress; second (result/error) fails
                raise RuntimeError("channel closed")
            super().send(data)

    ch = FailOnSendChannel()
    sig = FakeSignalingClient()
    mock_pl = MockPipeline(fail=True)
    h = InboundJobHandler(ch, sig, "job-closed", pipeline_module=mock_pl, temp_dir=tmp_path)
    h.bind()

    ch.inject(b"E" * 1024)
    await _drain()
    ch.inject(json.dumps({"type": "audio_eof", "total_bytes": 1024}))
    await _drain()

    # state(idle) must still fire even if send() raised (T-08-07-07)
    assert "idle" in sig.states, "state(idle) must fire even when send() raised"


@pytest.mark.asyncio
async def test_resume_query_returns_current_offset(tmp_path: Path) -> None:
    """resume_query replies with the current .partial byte offset."""
    ch = FakeDataChannel()
    sig = FakeSignalingClient()
    h = InboundJobHandler(ch, sig, "job-resume", temp_dir=tmp_path)
    h.bind()

    # Write 5 MB
    chunk = b"F" * 65536
    for _ in range(80):  # 80 * 64 KB = 5 MB
        ch.inject(chunk)
        await _drain()

    ch.inject(json.dumps({"type": "resume_query", "job_id": "job-resume"}))
    await _drain()

    resume_replies = [
        json.loads(m) for m in ch.sent if json.loads(m)["type"] == "resume_state"
    ]
    assert len(resume_replies) == 1
    assert resume_replies[0]["byte_offset"] == 80 * 65536


@pytest.mark.asyncio
async def test_oversized_chunk_is_dropped(tmp_path: Path) -> None:
    """Binary chunks > 64 KB are dropped without writing (T-08-07-01)."""
    ch = FakeDataChannel()
    sig = FakeSignalingClient()
    h = InboundJobHandler(ch, sig, "job-oversized", temp_dir=tmp_path)
    h.bind()

    oversized = b"G" * (65537)  # 65537 bytes — 1 byte over the 64 KB cap
    ch.inject(oversized)
    await _drain()

    partial = tmp_path / "job-oversized.partial"
    assert not partial.exists(), ".partial must NOT be created for oversized chunks"


@pytest.mark.asyncio
async def test_ping_replies_with_pong(tmp_path: Path) -> None:
    """Sending ping results in a pong response."""
    ch = FakeDataChannel()
    sig = FakeSignalingClient()
    h = InboundJobHandler(ch, sig, "job-ping", temp_dir=tmp_path)
    h.bind()

    ch.inject(json.dumps({"type": "ping"}))
    await _drain()

    assert "pong" in ch.sent_types(), "pong must be sent in response to ping"
