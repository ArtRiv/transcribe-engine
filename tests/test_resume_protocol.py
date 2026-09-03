"""End-to-end resume protocol test (Plan 08-07 Task 2 — RTC-06).

Scenario:
  1. Open data channel; send 5 MB of audio bytes.
  2. Close channel mid-transfer (no audio_eof).
  3. Open new data channel for same job_id; send resume_query.
  4. Engine responds resume_state(byte_offset=5 MB).
  5. Send remaining 5 MB + audio_eof.
  6. Pipeline is kicked with a 10 MB audio file.

Verifies:
  - Engine's .partial survives channel close (it's a file, not in-memory).
  - resume_state byte_offset matches the durably-written partial size.
  - Pipeline receives the full 10 MB reassembled file.
  - (T-08-07-08) Engine byte_offset is the source of truth; test asserts exact match.

No GPU required — MockPipeline records the final file size.
"""
import asyncio
import json
from pathlib import Path

import pytest

from transcribe_engine.webrtc.handler import InboundJobHandler

# ---------------------------------------------------------------------------
# Fakes (same pattern as test_webrtc_handler.py)
# ---------------------------------------------------------------------------

_1MB = 1 * 1024 * 1024
_5MB = 5 * 1024 * 1024
_10MB = 10 * 1024 * 1024


class FakeDataChannel:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self._handlers: dict = {}

    def on(self, event: str):
        def decorator(fn):
            self._handlers.setdefault(event, []).append(fn)
            return fn
        return decorator

    def send(self, data: str) -> None:
        self.sent.append(data)

    def inject(self, message) -> None:
        for h in self._handlers.get("message", []):
            if asyncio.iscoroutinefunction(h):
                asyncio.get_event_loop().run_until_complete(h(message))
            else:
                h(message)

    def sent_types(self) -> list[str]:
        return [json.loads(m)["type"] for m in self.sent if isinstance(m, str)]


class FakeSignalingClient:
    def __init__(self) -> None:
        self.states: list[str] = []

    async def emit_state(self, value: str) -> None:
        self.states.append(value)


class RecordingPipeline:
    """Pipeline that records file size — no GPU needed."""

    def __init__(self) -> None:
        self.file_size: int | None = None

    async def run_pipeline(self, audio_path: Path, *, job_id: str, on_progress) -> dict:
        self.file_size = audio_path.stat().st_size if audio_path.exists() else 0
        on_progress("merge", 1.0)
        return {
            "version": 1,
            "language": "en",
            "duration_sec": 1.0,
            "speakers": [{"id": "S0", "label": "Speaker 1"}],
            "segments": [],
        }


async def _drain() -> None:
    for _ in range(10):
        await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Resume protocol end-to-end test
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_protocol_end_to_end(tmp_path: Path) -> None:
    """Full resume flow: 5 MB sent, channel dropped, 5 MB resumed, pipeline gets 10 MB."""
    job_id = "job-resume-e2e"
    sig = FakeSignalingClient()
    pl = RecordingPipeline()

    # --- Phase 1: Send first 5 MB on channel 1 (no audio_eof) ---
    ch1 = FakeDataChannel()
    h1 = InboundJobHandler(ch1, sig, job_id, pipeline_module=pl, temp_dir=tmp_path)
    h1.bind()

    chunk = b"A" * 65536  # 64 KB chunks
    sent_phase1 = 0
    while sent_phase1 < _5MB:
        ch1.inject(chunk)
        await _drain()
        sent_phase1 += len(chunk)

    # Confirm .partial exists with 5 MB
    partial = tmp_path / f"{job_id}.partial"
    assert partial.exists(), ".partial must survive channel close"
    assert partial.stat().st_size == _5MB, (
        f"expected 5 MB partial, got {partial.stat().st_size}"
    )

    # "Close" channel 1 — no audio_eof sent; channel object just abandoned
    # (the handler holds a reference but the .partial file is on disk)

    # --- Phase 2: New channel 2 for resume ---
    ch2 = FakeDataChannel()
    h2 = InboundJobHandler(ch2, sig, job_id, pipeline_module=pl, temp_dir=tmp_path)
    h2.bind()

    # Send resume_query
    ch2.inject(json.dumps({"type": "resume_query", "job_id": job_id}))
    await _drain()

    # Engine must reply with resume_state(byte_offset=5 MB)
    resume_replies = [
        json.loads(m) for m in ch2.sent
        if json.loads(m)["type"] == "resume_state"
    ]
    assert len(resume_replies) == 1, f"Expected 1 resume_state, got {resume_replies}"

    reported_offset = resume_replies[0]["byte_offset"]
    assert reported_offset == _5MB, (
        f"T-08-07-08: engine must report exact .partial size={_5MB}, "
        f"got {reported_offset}"
    )

    # --- Phase 3: Resume from confirmed offset — send remaining 5 MB ---
    sent_phase2 = 0
    while sent_phase2 < _5MB:
        ch2.inject(chunk)
        await _drain()
        sent_phase2 += len(chunk)

    # Verify partial is now 10 MB
    assert partial.stat().st_size == _10MB, (
        f"After resume, partial should be {_10MB}, got {partial.stat().st_size}"
    )

    # Send EOF
    ch2.inject(json.dumps({"type": "audio_eof", "total_bytes": _10MB}))
    await _drain()

    # --- Assertions ---
    assert pl.file_size == _10MB, (
        f"Pipeline must receive 10 MB file; got {pl.file_size}"
    )
    assert "result" in ch2.sent_types(), "Result must be sent after pipeline completes"
    assert "idle" in sig.states, "state(idle) must be emitted after completion"

    # T-08-07-02: audio file must be cleaned up
    final = tmp_path / job_id
    assert not final.exists(), "Final audio file must be deleted after job completion"


@pytest.mark.asyncio
async def test_resume_query_with_no_writer_reads_disk(tmp_path: Path) -> None:
    """resume_query on a fresh handler with existing .partial reads disk size."""
    job_id = "job-disk-resume"

    # Pre-create a .partial file simulating a prior run
    partial = tmp_path / f"{job_id}.partial"
    partial.write_bytes(b"Z" * _5MB)

    ch = FakeDataChannel()
    sig = FakeSignalingClient()
    # Fresh handler — no writer instantiated yet
    h = InboundJobHandler(ch, sig, job_id, temp_dir=tmp_path)
    h.bind()

    ch.inject(json.dumps({"type": "resume_query", "job_id": job_id}))
    await _drain()

    resume_replies = [
        json.loads(m) for m in ch.sent
        if json.loads(m)["type"] == "resume_state"
    ]
    assert len(resume_replies) == 1
    assert resume_replies[0]["byte_offset"] == _5MB, (
        f"T-08-07-08: must read .partial size from disk; expected {_5MB}, "
        f"got {resume_replies[0]['byte_offset']}"
    )
