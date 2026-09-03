"""Stdlib resumable HTTP download (D-16).

urllib.request + Range header + .partial sidecar + atomic rename + SHA-256 verify.
Stdlib-only — avoids adding httpx weight to the bundle (RESEARCH "Anti-Patterns").
"""
import hashlib
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path


def download_resumable(
    url: str,
    dest: Path,
    expected_sha256: str,
    *,
    chunk_size: int = 1024 * 1024,
    on_progress: Callable[[int, int], None] | None = None,
) -> None:
    """Resumable HTTP download. Re-entrant on disconnect.

    Writes to <dest>.partial; renames atomically on SHA-256 verify.
    Raises RuntimeError on hash mismatch (deletes .partial first).

    T-7-06 mitigation: SHA-256 verified post-download against value pinned in
    models.toml; mismatch deletes .partial and raises RuntimeError. Atomic
    os.replace() on success. URLs HTTPS-only (verified in models.toml at plan 07-02).
    """
    partial = dest.with_suffix(dest.suffix + ".partial")
    resume_from = partial.stat().st_size if partial.exists() else 0

    req = urllib.request.Request(url)
    if resume_from > 0:
        req.add_header("Range", f"bytes={resume_from}-")

    with urllib.request.urlopen(req) as resp:
        # 206 Partial Content (resume) or 200 OK (fresh download)
        content_length = int(resp.headers.get("Content-Length", "0"))
        total = content_length + resume_from if resume_from > 0 else content_length
        mode = "ab" if resume_from > 0 else "wb"
        with open(partial, mode) as f:
            done = resume_from
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if on_progress:
                    on_progress(done, total)

    # Verify SHA-256 of the FULL file (not just newly downloaded bytes)
    h = hashlib.sha256()
    with open(partial, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected_sha256:
        partial.unlink()  # corrupt — force re-download next attempt
        raise RuntimeError(
            f"SHA-256 mismatch for {dest.name}: expected {expected_sha256}, got {actual}"
        )

    # Atomic rename — POSIX atomic; Windows atomic since Python 3.3
    os.replace(partial, dest)
