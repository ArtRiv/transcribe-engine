"""test_resumable.py — Stdlib resumable download tests with fake HTTP server fixture.

Tests cover: fresh download, SHA-256 mismatch, resume-after-disconnect, progress callback.
All tests use a real ThreadingHTTPServer with Range support (no mocking of urllib internals).
"""

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from transcribe_engine.download.resumable import download_resumable

# 64KB fixture content; deterministic per byte for hash verification
FIXTURE_BYTES = bytes((i % 256) for i in range(65536))
FIXTURE_SHA256 = hashlib.sha256(FIXTURE_BYTES).hexdigest()


class _RangeHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence test logs
        return

    def do_GET(self):
        range_hdr = self.headers.get("Range")
        if range_hdr:
            # Parse "bytes=N-"
            start = int(range_hdr.split("=")[1].split("-")[0])
            content = FIXTURE_BYTES[start:]
            self.send_response(206)
            self.send_header("Content-Length", str(len(content)))
            self.send_header(
                "Content-Range",
                f"bytes {start}-{len(FIXTURE_BYTES) - 1}/{len(FIXTURE_BYTES)}",
            )
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(len(FIXTURE_BYTES)))
            self.end_headers()
            self.wfile.write(FIXTURE_BYTES)


@pytest.fixture
def http_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}/fixture"
    server.shutdown()


def test_fresh_download_succeeds(http_server, tmp_path):
    dest = tmp_path / "model.bin"
    download_resumable(http_server, dest, FIXTURE_SHA256)
    assert dest.exists()
    assert dest.read_bytes() == FIXTURE_BYTES
    assert not dest.with_suffix(".bin.partial").exists()


def test_sha256_mismatch_raises_and_deletes_partial(http_server, tmp_path):
    dest = tmp_path / "model.bin"
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        download_resumable(http_server, dest, "0" * 64)  # wrong hash
    assert not dest.with_suffix(".bin.partial").exists()
    assert not dest.exists()


def test_resume_after_simulated_disconnect(http_server, tmp_path):
    dest = tmp_path / "model.bin"
    partial = dest.with_suffix(".bin.partial")
    # Simulate a previous interrupted download — write first half to .partial
    partial.write_bytes(FIXTURE_BYTES[:32768])
    # Now resume — server's Range handler returns the second half
    download_resumable(http_server, dest, FIXTURE_SHA256)
    assert dest.exists()
    assert dest.read_bytes() == FIXTURE_BYTES
    assert not partial.exists()


def test_progress_callback_invoked(http_server, tmp_path):
    dest = tmp_path / "model.bin"
    progress_events = []

    def on_progress(done, total):
        progress_events.append((done, total))

    download_resumable(http_server, dest, FIXTURE_SHA256, chunk_size=8192, on_progress=on_progress)
    assert len(progress_events) >= 1
    # Final event reaches the total
    assert progress_events[-1][0] == len(FIXTURE_BYTES)
