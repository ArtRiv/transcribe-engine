"""Tests for webrtc/chunker.py — PartialFileWriter (Plan 08-07 Task 1).

Tests are pure file-ops: no GPU, no aiortc, no network.

Coverage:
    - First write creates .partial file
    - Resume case: constructor reads existing .partial size as starting offset
    - Checkpoint emission at interval boundary
    - Atomic rename on finalize
    - Cleanup removes .partial on failure
"""
import os
from pathlib import Path

import pytest

from transcribe_engine.webrtc.chunker import PartialFileWriter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_1MB = 1 * 1024 * 1024
_10MB = 10 * 1024 * 1024


def _make_chunk(size: int) -> bytes:
    return b"A" * size


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_writer_first_write_creates_partial(tmp_path: Path) -> None:
    """Creating a PartialFileWriter and writing data produces a .partial file."""
    writer = PartialFileWriter(tmp_path, "job-001")
    assert not writer.path_final.exists(), "Final path should not exist yet"

    writer.write(_make_chunk(1024))
    assert writer.path.exists(), ".partial file must exist after first write"
    assert writer.path.stat().st_size == 1024

    writer.cleanup()


def test_writer_resume_appends_to_existing_partial(tmp_path: Path) -> None:
    """If .partial already exists, the writer resumes from its current size."""
    job_id = "job-resume-001"
    partial_path = tmp_path / f"{job_id}.partial"

    # Pre-create a 1 MB partial (simulates prior incomplete transfer)
    partial_path.write_bytes(_make_chunk(_1MB))

    writer = PartialFileWriter(tmp_path, job_id)
    # current_offset() should reflect the pre-existing bytes
    assert writer.current_offset() == _1MB, (
        f"expected starting offset {_1MB}, got {writer.current_offset()}"
    )

    # Write another 512 KB
    writer.write(_make_chunk(512 * 1024))
    assert writer.current_offset() == _1MB + 512 * 1024

    writer.cleanup()


def test_writer_emits_checkpoint_at_10MB(tmp_path: Path) -> None:
    """write() returns the byte offset exactly once at the 10 MB boundary."""
    writer = PartialFileWriter(tmp_path, "job-chkpt", checkpoint_interval=_10MB)

    checkpoints: list[int] = []
    for _ in range(10):
        result = writer.write(_make_chunk(_1MB))
        if result is not None:
            checkpoints.append(result)

    # Exactly one checkpoint at 10 MB
    assert len(checkpoints) == 1, f"expected 1 checkpoint, got {checkpoints}"
    assert checkpoints[0] == _10MB, f"expected checkpoint at {_10MB}, got {checkpoints[0]}"

    writer.cleanup()


def test_writer_finalize_atomic_rename(tmp_path: Path) -> None:
    """finalize() atomically renames .partial → final; .partial no longer exists."""
    writer = PartialFileWriter(tmp_path, "job-final")
    writer.write(_make_chunk(1024))

    partial_path = writer.path
    final_path = writer.path_final

    result = writer.finalize()

    assert result == final_path
    assert final_path.exists(), "Final file must exist after finalize"
    assert not partial_path.exists(), ".partial must be gone after finalize"
    assert final_path.stat().st_size == 1024


def test_writer_cleanup_after_failure(tmp_path: Path) -> None:
    """cleanup() removes the .partial file (T-08-07-02 — no orphan audio files)."""
    writer = PartialFileWriter(tmp_path, "job-cleanup")
    writer.write(_make_chunk(5 * 1024))  # write 5 KB

    partial_path = writer.path
    assert partial_path.exists()

    # Simulate a failure scenario — call cleanup directly
    writer.cleanup()

    assert not partial_path.exists(), ".partial must be removed by cleanup()"


def test_writer_context_manager_cleans_up_on_exception(tmp_path: Path) -> None:
    """Using the context manager with an exception removes .partial (T-08-07-02)."""
    job_id = "job-ctxmgr"

    with pytest.raises(RuntimeError):
        with PartialFileWriter(tmp_path, job_id) as w:
            w.write(_make_chunk(2048))
            raise RuntimeError("simulated pipeline failure")

    partial_path = tmp_path / f"{job_id}.partial"
    assert not partial_path.exists(), ".partial must be cleaned up via context manager"


def test_writer_finalize_raises_on_empty_file(tmp_path: Path) -> None:
    """finalize() raises RuntimeError if no bytes were written."""
    writer = PartialFileWriter(tmp_path, "job-empty")
    with pytest.raises(RuntimeError, match="zero bytes"):
        writer.finalize()


def test_writer_peer_module_lazy_import() -> None:
    """aiortc must NOT be imported at module top-level in peer.py (startup speed)."""
    import ast
    from pathlib import Path as P
    peer_path = P(__file__).parent.parent / "transcribe_engine" / "webrtc" / "peer.py"
    tree = ast.parse(peer_path.read_text(), filename=str(peer_path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Only check top-level imports (not inside functions)
            # We detect "top-level" by checking that the node is a direct child
            # of the module body.
            pass
    # Check that no top-level import of aiortc exists
    module_body_imports = [
        n for n in tree.body
        if isinstance(n, (ast.Import, ast.ImportFrom))
    ]
    for node in module_body_imports:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("aiortc"), (
                    "aiortc must be imported lazily inside functions, not at module top-level"
                )
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("aiortc"), (
                "aiortc must be imported lazily inside functions, not at module top-level"
            )
