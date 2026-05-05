"""webrtc/chunker.py — PartialFileWriter: append-write + checkpoint emission.

Structural analog: download/resumable.py (.partial + atomic rename idiom).
Kept as a separate implementation — the audio-reassembly path has different
cleanup and hash-verify semantics vs the model-download path.

SEC-08: no supabase import; T-08-07-02 mitigation: cleanup() always runs
even on failure (caller uses context manager or explicit finally).
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Emit a checkpoint roughly every 10 MB (D-06 / RTC-06).
_DEFAULT_CHECKPOINT_INTERVAL: int = 10 * 1024 * 1024

# SEC: job_id is user-controlled (comes from the wire). Only alphanumeric,
# hyphen, and underscore are accepted. Max 64 chars.  Any deviation is a
# ValueError raised before any file operation.
_VALID_JOB_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _validate_job_id(job_id: str) -> None:
    """Raise ValueError if *job_id* could be used for path traversal.

    Called at every entry-point that converts job_id → filesystem path.
    Belt-and-suspenders: PartialFileWriter.path also checks resolve().
    """
    if not _VALID_JOB_ID.fullmatch(job_id):
        raise ValueError(
            f"invalid job_id {job_id!r}: must match ^[A-Za-z0-9_-]{{1,64}}$"
        )


class PartialFileWriter:
    """Write incoming audio bytes to <temp_dir>/<job_id>.partial.

    Supports resume: if the .partial already exists on construction, the
    current size becomes the starting offset (the engine is the truth for
    resume_state — T-08-07-08).

    Checkpoint interval: emit a checkpoint message when cumulative bytes
    written cross a multiple of `checkpoint_interval`.

    Thread-safety: NOT thread-safe. Designed for use from a single asyncio
    task (the data-channel message handler). All writes happen on the event
    loop; no locking needed.

    Context manager usage (recommended):

        async with PartialFileWriter(tmp, job_id) as w:
            ...
            final_path = w.finalize()

    The __exit__ calls cleanup() on exception so no .partial file is
    orphaned. On normal exit the caller calls finalize() before leaving the
    with-block; cleanup() after finalize() is a safe no-op (final file
    already renamed).
    """

    def __init__(
        self,
        temp_dir: Path,
        job_id: str,
        *,
        checkpoint_interval: int = _DEFAULT_CHECKPOINT_INTERVAL,
    ) -> None:
        # SEC: validate before any filesystem access (CR-03 / path-traversal).
        _validate_job_id(job_id)
        self._temp_dir = temp_dir
        self._job_id = job_id
        self._checkpoint_interval = checkpoint_interval
        self._finalized = False

        partial = self.path

        # Belt-and-suspenders path-traversal check (CR-03).
        resolved = partial.resolve()
        temp_resolved = temp_dir.resolve()
        if not resolved.is_relative_to(temp_resolved):
            raise ValueError(
                f"job_id {job_id!r} resolves outside temp_dir: {resolved}"
            )

        # WR-08: use O_WRONLY|O_CREAT|O_APPEND (atomic) instead of exists()+open().
        # O_CREAT is atomic — avoids the TOCTOU race between exists() and open().
        # Mode 0o600: user audio is sensitive material (project SEC guidelines).
        fd = os.open(str(partial), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        self._fp = os.fdopen(fd, "ab")

        start_offset = self._fp.seek(0, 2)  # seek to end; returns byte position
        log.debug(
            "PartialFileWriter: job_id=%s start_offset=%d",
            job_id,
            start_offset,
        )

        # Track last checkpoint boundary so we emit at most once per interval.
        self._last_checkpoint_offset: int = (
            (start_offset // checkpoint_interval) * checkpoint_interval
        )
        # Track the last durably-fsynced byte offset (CR-01).
        self._last_fsync_offset: int = start_offset

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def path(self) -> Path:
        """Path to the in-flight .partial file."""
        return self._temp_dir / f"{self._job_id}.partial"

    @property
    def path_final(self) -> Path:
        """Destination path after atomic rename (no suffix)."""
        return self._temp_dir / self._job_id

    # -------------------------------------------------------------------------
    # Core write + checkpoint
    # -------------------------------------------------------------------------

    def write(self, chunk: bytes) -> Optional[int]:
        """Append *chunk* to the .partial file.

        Returns the DURABLE byte offset AFTER the write if a checkpoint
        boundary was crossed, otherwise returns None.  The caller sends a
        CheckpointMsg when the return value is not None.

        T-08-07-01 mitigation: caller must assert len(chunk) <= 65536 before
        calling write(); this writer does not enforce it (separation of
        concerns).

        CR-01 mitigation: fsync() is called at EACH checkpoint boundary so the
        reported offset is what is durably on disk.  Per-chunk fsync would tank
        throughput; we amortize the cost over checkpoint_interval (default 10 MB).
        After fsync() returns, _last_fsync_offset is updated.  current_offset()
        returns _last_fsync_offset — the authoritative resume value.
        """
        if self._finalized:
            raise RuntimeError("write() called after finalize()")
        self._fp.write(chunk)
        self._fp.flush()
        offset = self._fp.tell()

        # Emit checkpoint when we cross a new multiple of checkpoint_interval.
        new_boundary = (offset // self._checkpoint_interval) * self._checkpoint_interval
        if new_boundary > self._last_checkpoint_offset and new_boundary > 0:
            # CR-01: fsync BEFORE reporting offset — makes checkpoint durable.
            os.fsync(self._fp.fileno())
            self._last_fsync_offset = offset
            self._last_checkpoint_offset = new_boundary
            return offset
        return None

    def current_offset(self) -> int:
        """Return the last durably-fsynced byte offset.

        This is the authoritative value for resume_query (T-08-07-08 / CR-01).
        We only report what is on disk, not what is in the OS page cache.
        """
        return self._last_fsync_offset

    # -------------------------------------------------------------------------
    # Finalize + cleanup
    # -------------------------------------------------------------------------

    def finalize(self) -> Path:
        """Close the partial file and atomically rename it to path_final.

        Returns path_final.  Raises RuntimeError if the file is empty
        (indicates something went wrong before any bytes arrived).

        Atomic on POSIX; atomic on Windows since Python 3.3 (os.replace).
        Mirror: download/resumable.py line 66.
        """
        if self._finalized:
            return self.path_final

        offset = self._fp.tell()
        self._fp.close()

        if offset == 0:
            # Empty file — don't produce a zero-byte audio input for the pipeline.
            self.path.unlink(missing_ok=True)
            raise RuntimeError(
                f"PartialFileWriter.finalize(): job_id={self._job_id!r} "
                "produced zero bytes — pipeline kickoff aborted."
            )

        os.replace(self.path, self.path_final)
        self._finalized = True
        log.debug(
            "PartialFileWriter: finalized job_id=%s path=%s",
            self._job_id,
            self.path_final,
        )
        return self.path_final

    def cleanup(self) -> None:
        """Close fp if open; delete .partial if it exists.

        Safe to call after finalize() — path_final is left untouched.
        Safe to call multiple times (idempotent).

        T-08-07-02: ensures no orphaned user-audio file in OS temp dir.
        """
        if not self._fp.closed:
            self._fp.close()
        self.path.unlink(missing_ok=True)

    # -------------------------------------------------------------------------
    # Context manager
    # -------------------------------------------------------------------------

    def __enter__(self) -> "PartialFileWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            # Exception path — clean up partial file (T-08-07-02)
            self.cleanup()
        # Normal path: caller has called finalize(); cleanup() is a no-op.
        return None
