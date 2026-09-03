"""Tests for webrtc/chunker.py — PartialFileWriter (Plan 08-07 Task 1).

Tests are pure file-ops: no GPU, no aiortc, no network.

Coverage:
    - First write creates .partial file
    - Resume case: constructor reads existing .partial size as starting offset
    - Checkpoint emission at interval boundary
    - CR-01: fsync() called at checkpoint boundaries, not per-chunk
    - CR-03: invalid job_id raises ValueError before any file operation
    - WR-08: .partial file created with mode 0o600 (atomic O_CREAT)
    - Atomic rename on finalize
    - Cleanup removes .partial on failure
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from transcribe_engine.webrtc.chunker import PartialFileWriter, _validate_job_id

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
    """If .partial already exists, the writer resumes from its current size.

    CR-01: current_offset() returns the last durably-fsynced offset.  On
    resume the initial offset (start_offset) is the on-disk size — treated
    as already durable.  After writing 512 KB without crossing a checkpoint
    boundary, current_offset() still returns start_offset (not flushed/
    fsync'd yet), but the file on disk has grown.
    """
    job_id = "job-resume-001"
    partial_path = tmp_path / f"{job_id}.partial"

    # Pre-create a 1 MB partial (simulates prior incomplete transfer)
    partial_path.write_bytes(_make_chunk(_1MB))

    writer = PartialFileWriter(tmp_path, job_id)
    # current_offset() reports durable (on-disk) bytes — 1 MB on resume.
    assert writer.current_offset() == _1MB, (
        f"expected starting offset {_1MB}, got {writer.current_offset()}"
    )

    # Write another 512 KB (does NOT cross 10 MB checkpoint → no fsync)
    writer.write(_make_chunk(512 * 1024))

    # File on disk has grown to 1.5 MB
    assert partial_path.stat().st_size == _1MB + 512 * 1024

    # current_offset() still reports last durable (fsynced) boundary = 1 MB
    # (the 512 KB written since resume has not been fsynced yet)
    assert writer.current_offset() == _1MB, (
        "current_offset() must return last fsynced offset, not tell()"
    )

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


def test_writer_file_mode_is_0o600(tmp_path: Path) -> None:
    """WR-08: .partial file must be created with mode 0o600 (user-only rw)."""
    writer = PartialFileWriter(tmp_path, "job-mode")
    writer.write(_make_chunk(1024))
    partial = writer.path
    assert partial.exists(), ".partial must exist after write"
    mode = os.stat(str(partial)).st_mode & 0o777
    assert mode == 0o600, f"expected mode 0o600, got 0o{mode:o}"
    writer.cleanup()


def test_validate_job_id_rejects_traversal() -> None:
    """CR-03: _validate_job_id rejects job_ids that could be path traversal."""
    bad_ids = [
        "../etc/cron.d/evil",
        "../../root",
        "foo/bar",
        "foo\\bar",
        "foo\x00bar",
        "a" * 65,  # > 64 chars
        "",  # empty
        "foo bar",  # space
        "foo%2ebar",  # URL encoding
    ]
    for bad in bad_ids:
        with pytest.raises(ValueError, match="invalid job_id"):
            _validate_job_id(bad)


def test_validate_job_id_accepts_valid() -> None:
    """CR-03: _validate_job_id accepts well-formed job_ids."""
    good_ids = [
        "abc123",
        "user-abc123def456",
        "A_B-C",
        "a" * 64,  # exactly 64 chars
        "a",  # 1 char
    ]
    for good in good_ids:
        _validate_job_id(good)  # must not raise


def test_writer_rejects_traversal_job_id(tmp_path: Path) -> None:
    """CR-03: PartialFileWriter raises ValueError on bad job_id before touching disk."""
    with pytest.raises(ValueError, match="invalid job_id"):
        PartialFileWriter(tmp_path, "../../etc/evil")

    # No file created
    assert list(tmp_path.iterdir()) == []


def test_writer_fsync_called_at_checkpoint_not_per_chunk(tmp_path: Path) -> None:
    """CR-01: os.fsync() is called exactly once per checkpoint boundary, NOT per chunk.

    Durability semantics: the reported checkpoint offset is what is on disk.
    Per-chunk fsync would tank throughput; we amortise over checkpoint_interval.
    """
    checkpoint_interval = _10MB
    writer = PartialFileWriter(tmp_path, "job-fsync", checkpoint_interval=checkpoint_interval)

    fsync_calls = []
    real_fsync = os.fsync

    def counting_fsync(fd: int) -> None:
        fsync_calls.append(fd)
        real_fsync(fd)

    with patch("transcribe_engine.webrtc.chunker.os.fsync", side_effect=counting_fsync):
        # Write 9 chunks of 1 MB each (9 MB total — no checkpoint yet)
        for _ in range(9):
            writer.write(_make_chunk(_1MB))

        assert len(fsync_calls) == 0, (
            f"fsync must NOT be called before crossing 10 MB; got {len(fsync_calls)} calls"
        )

        # Write 1 more MB to cross the 10 MB boundary
        result = writer.write(_make_chunk(_1MB))

        assert result is not None, "must return checkpoint offset at 10 MB boundary"
        assert len(fsync_calls) == 1, (
            f"fsync must be called exactly once at checkpoint boundary; got {len(fsync_calls)}"
        )

        # current_offset() now reflects durable boundary
        assert writer.current_offset() == _10MB

    writer.cleanup()


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
    module_body_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
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
