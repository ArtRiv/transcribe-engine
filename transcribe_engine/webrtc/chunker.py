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
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Emit a checkpoint roughly every 10 MB (D-06 / RTC-06).
_DEFAULT_CHECKPOINT_INTERVAL: int = 10 * 1024 * 1024


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
        self._temp_dir = temp_dir
        self._job_id = job_id
        self._checkpoint_interval = checkpoint_interval
        self._finalized = False

        partial = self.path
        if partial.exists():
            start_offset = partial.stat().st_size
            self._fp = partial.open("ab")
            # Seek to end to ensure tell() returns the correct resuming offset.
            self._fp.seek(0, 2)
            log.debug(
                "PartialFileWriter: resuming job_id=%s from offset=%d",
                job_id,
                start_offset,
            )
        else:
            self._fp = partial.open("wb")

        # Track last checkpoint boundary so we emit at most once per interval.
        self._last_checkpoint_offset: int = (
            (self.current_offset() // checkpoint_interval) * checkpoint_interval
        )

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

        Returns the byte offset AFTER the write if a checkpoint boundary was
        crossed, otherwise returns None.  The caller sends a CheckpointMsg
        when the return value is not None.

        T-08-07-01 mitigation: caller must assert len(chunk) <= 65536 before
        calling write(); this writer does not enforce it (separation of concerns).
        T-08-07-02: bytes are flushed to OS buffer (not fsync'd) before the
        checkpoint offset is reported.  The offset is the tell() after write —
        consistent with what the OS has durably buffered under normal operation.
        """
        if self._finalized:
            raise RuntimeError("write() called after finalize()")
        self._fp.write(chunk)
        self._fp.flush()  # flush OS buffer before reporting checkpoint
        offset = self._fp.tell()

        # Emit checkpoint when we cross a new multiple of checkpoint_interval.
        new_boundary = (offset // self._checkpoint_interval) * self._checkpoint_interval
        if new_boundary > self._last_checkpoint_offset and new_boundary > 0:
            self._last_checkpoint_offset = new_boundary
            return offset
        return None

    def current_offset(self) -> int:
        """Return current byte position (single source of truth for resume_query)."""
        if self._fp.closed:
            return 0
        return self._fp.tell()

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
