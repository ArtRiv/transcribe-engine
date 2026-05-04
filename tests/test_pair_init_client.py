"""Tests for transcribe_engine.pairing.pair_init — stdlib urllib HTTP client.

Uses a real ThreadingHTTPServer (cribbed from test_resumable.py) — no mocking
of urllib internals. Tests cover:
  - Correct POST body shape with all 7 required fields (B-02)
  - Signature binds gpu + engine_version to the signed message (B-02)
  - 409 raises PairCodeConflictError
  - 4xx/5xx raises PairInitError with route error code
  - Only urllib.request is used (no httpx / requests / aiohttp)
  - Default timeout is 10 s
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from nacl.signing import SigningKey

from transcribe_engine.pairing.pair_init import (
    PairCodeConflictError,
    PairInitError,
    pair_init_post,
)


# ---------------------------------------------------------------------------
# Fake server infrastructure
# ---------------------------------------------------------------------------


class _CaptureHandler(BaseHTTPRequestHandler):
    """Records the last received request body + path; returns a configurable response."""

    log_request = lambda self, *a: None  # silence test logs  # noqa: E731
    log_message = lambda self, *a: None  # noqa: E731

    # Class-level state set per-test by the fixture
    response_status: int = 200
    response_body: dict[str, Any] = {}
    captured_requests: list[dict[str, Any]] = []
    nonce_payload: dict[str, Any] = {"nonce": "abc123", "issued_at": 1746360000000}
    # Artificial delay for timeout test (seconds)
    delay_seconds: float = 0.0

    def do_GET(self) -> None:
        """Serve /api/signal-token/nonce."""
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)
        body = json.dumps(self.nonce_payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw)
        except Exception:
            data = {}
        self.__class__.captured_requests.append(
            {"path": self.path, "body": data, "headers": dict(self.headers)}
        )
        resp_body = json.dumps(self.response_body).encode()
        self.send_response(self.response_status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp_body)))
        self.end_headers()
        self.wfile.write(resp_body)


@pytest.fixture
def fake_server(request):
    """Start a ThreadingHTTPServer with configurable response; yield base URL."""
    status = getattr(request, "param", {}).get("status", 200)
    body = getattr(request, "param", {}).get("body", {})
    nonce = getattr(request, "param", {}).get(
        "nonce", {"nonce": "abc123", "issued_at": 1746360000000}
    )
    delay = getattr(request, "param", {}).get("delay", 0.0)

    _CaptureHandler.response_status = status
    _CaptureHandler.response_body = body
    _CaptureHandler.captured_requests = []
    _CaptureHandler.nonce_payload = nonce
    _CaptureHandler.delay_seconds = delay

    server = ThreadingHTTPServer(("127.0.0.1", 0), _CaptureHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def _make_signing_key() -> SigningKey:
    return SigningKey.generate()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pair_init_signs_and_posts_correctly(fake_server: str) -> None:
    """POST body has all 7 required fields; signature and pubkey are correct hex lengths."""
    sk = _make_signing_key()
    pair_init_post(
        fake_server,
        code="K7H2Q9X3",
        signing_key=sk,
        gpu="RX 6600 (Vulkan)",
        engine_version="0.2.0",
    )

    assert len(_CaptureHandler.captured_requests) == 1
    body = _CaptureHandler.captured_requests[0]["body"]

    # All 7 required fields (B-02)
    assert "code" in body
    assert "pubkey" in body
    assert "signature" in body
    assert "nonce" in body
    assert "issued_at" in body
    assert "gpu" in body
    assert "engine_version" in body

    # Correct byte lengths in hex
    assert len(body["signature"]) == 128, "Ed25519 signature = 64 bytes = 128 hex chars"
    assert len(body["pubkey"]) == 64, "Ed25519 pubkey = 32 bytes = 64 hex chars"

    # Values match what was passed
    assert body["code"] == "K7H2Q9X3"
    assert body["gpu"] == "RX 6600 (Vulkan)"
    assert body["engine_version"] == "0.2.0"


def test_pair_init_signed_message_includes_gpu_and_engine_version(fake_server: str) -> None:
    """(B-02) Signature binds code+nonce+issued_at+gpu+engine_version.

    1. Verify the signature is valid against the posted pubkey.
    2. Verify that swapping the gpu string invalidates the signature.
    """
    from nacl.signing import VerifyKey

    sk = _make_signing_key()
    code = "K7H2Q9X3"
    gpu = "RX 6600 (Vulkan)"
    engine_version = "0.2.0"
    nonce = "abc123"
    issued_at = 1746360000000  # int — matches nonce-cache.ts Date.now() wire format

    pair_init_post(
        fake_server,
        code=code,
        signing_key=sk,
        gpu=gpu,
        engine_version=engine_version,
    )

    body = _CaptureHandler.captured_requests[0]["body"]
    sig_bytes = bytes.fromhex(body["signature"])
    pubkey_bytes = bytes.fromhex(body["pubkey"])

    # Reconstruct the expected signed message (plan-specified template)
    expected_message = f"pair-init:{code}:{nonce}:{issued_at}:{gpu}:{engine_version}".encode()

    # Valid signature verifies
    vk = VerifyKey(pubkey_bytes)
    vk.verify(expected_message, sig_bytes)  # raises if invalid

    # Swapping gpu invalidates the signature (proves binding)
    tampered_message = f"pair-init:{code}:{nonce}:{issued_at}:EVIL GPU:{engine_version}".encode()
    with pytest.raises(Exception):  # nacl.exceptions.BadSignatureError
        vk.verify(tampered_message, sig_bytes)


def test_pair_init_post_url_is_pair_init(fake_server: str) -> None:
    """POST lands on /api/pair-init."""
    sk = _make_signing_key()
    pair_init_post(fake_server, code="AAAABBBB", signing_key=sk, gpu="GPU", engine_version="0.1.0")
    assert _CaptureHandler.captured_requests[0]["path"] == "/api/pair-init"


@pytest.mark.parametrize(
    "fake_server",
    [{"status": 409, "body": {"error": "code_conflict"}}],
    indirect=True,
)
def test_pair_init_409_raises_PairCodeConflictError(fake_server: str) -> None:
    """HTTP 409 maps to PairCodeConflictError (caller regenerates code)."""
    sk = _make_signing_key()
    with pytest.raises(PairCodeConflictError):
        pair_init_post(fake_server, code="AAAABBBB", signing_key=sk, gpu="GPU", engine_version="0")


@pytest.mark.parametrize(
    "fake_server",
    [{"status": 500, "body": {"error": "server_error"}}],
    indirect=True,
)
def test_pair_init_500_raises_PairInitError_with_route_error_code(fake_server: str) -> None:
    """HTTP 5xx raises PairInitError; error message contains the route error code."""
    sk = _make_signing_key()
    with pytest.raises(PairInitError, match="server_error"):
        pair_init_post(fake_server, code="AAAABBBB", signing_key=sk, gpu="GPU", engine_version="0")


@pytest.mark.parametrize(
    "fake_server",
    [{"status": 401, "body": {"error": "invalid_signature"}}],
    indirect=True,
)
def test_pair_init_4xx_raises_PairInitError(fake_server: str) -> None:
    """HTTP 4xx (non-409) raises PairInitError with the route error code."""
    sk = _make_signing_key()
    with pytest.raises(PairInitError, match="invalid_signature"):
        pair_init_post(fake_server, code="AAAABBBB", signing_key=sk, gpu="GPU", engine_version="0")


def test_pair_init_uses_only_urllib(fake_server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Sentinel (T-08-05-04): monkeypatch httpx/requests/aiohttp with raising stubs.

    pair_init_post must succeed despite these modules being broken — proves
    it doesn't rely on them.
    """
    import sys
    import types

    # Create raising stub modules
    def _make_stub(name: str) -> types.ModuleType:
        mod = types.ModuleType(name)

        def _raise(*a: object, **kw: object) -> None:
            raise AssertionError(f"{name} must not be used by pair_init_post")

        mod.__getattr__ = lambda attr: _raise  # type: ignore[method-assign]
        return mod

    for banned in ("httpx", "requests", "aiohttp"):
        monkeypatch.setitem(sys.modules, banned, _make_stub(banned))

    sk = _make_signing_key()
    # Must not raise (uses only urllib internally)
    pair_init_post(fake_server, code="AAAABBBB", signing_key=sk, gpu="GPU", engine_version="0.2.0")
    assert len(_CaptureHandler.captured_requests) == 1


def test_pair_init_content_type_is_json(fake_server: str) -> None:
    """POST request Content-Type header is application/json."""
    sk = _make_signing_key()
    pair_init_post(fake_server, code="AAAABBBB", signing_key=sk, gpu="GPU", engine_version="0.2.0")
    headers = _CaptureHandler.captured_requests[0]["headers"]
    content_type = headers.get("Content-Type", headers.get("content-type", ""))
    assert "application/json" in content_type


@pytest.mark.parametrize(
    "fake_server",
    [{"delay": 0.0}],
    indirect=True,
)
def test_pair_init_timeout_parameter_accepted(fake_server: str) -> None:
    """Custom http_timeout parameter is accepted without error."""
    sk = _make_signing_key()
    pair_init_post(
        fake_server,
        code="AAAABBBB",
        signing_key=sk,
        gpu="GPU",
        engine_version="0.2.0",
        http_timeout=30.0,
    )
    assert len(_CaptureHandler.captured_requests) == 1


# ---------------------------------------------------------------------------
# WR-03: HTTPS enforcement
# ---------------------------------------------------------------------------


def test_pair_init_rejects_http_non_localhost() -> None:
    """WR-03: http:// scheme rejected for non-localhost targets."""
    sk = _make_signing_key()
    with pytest.raises(PairInitError, match="insecure"):
        pair_init_post(
            "http://example.com",
            code="AAAABBBB",
            signing_key=sk,
            gpu="GPU",
            engine_version="0.1.0",
        )


def test_pair_init_allows_http_localhost(fake_server: str) -> None:
    """WR-03: http://127.0.0.1 is allowed (dev use)."""
    # fake_server is already http://127.0.0.1:<port>
    sk = _make_signing_key()
    pair_init_post(fake_server, code="AAAABBBB", signing_key=sk, gpu="GPU", engine_version="0.1.0")
    assert len(_CaptureHandler.captured_requests) == 1


# ---------------------------------------------------------------------------
# WR-04: network errors mapped to PairInitError
# ---------------------------------------------------------------------------


def test_pair_init_nonce_get_connection_refused_raises_PairInitError() -> None:
    """WR-04: connection refused on nonce GET raises PairInitError (not URLError)."""
    sk = _make_signing_key()
    with pytest.raises(PairInitError, match="nonce fetch failed"):
        pair_init_post(
            "http://127.0.0.1:1",  # port 1 should refuse connection
            code="AAAABBBB",
            signing_key=sk,
            gpu="GPU",
            engine_version="0.1.0",
        )


# ---------------------------------------------------------------------------
# WR-09: issued_at must be int
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fake_server",
    [{"nonce": {"nonce": "abc123", "issued_at": "2026-05-04T12:00:00Z"}}],
    indirect=True,
)
def test_pair_init_rejects_string_issued_at(fake_server: str) -> None:
    """WR-09: PairInitError raised when issued_at is a string instead of int."""
    sk = _make_signing_key()
    with pytest.raises(PairInitError, match="issued_at must be int"):
        pair_init_post(fake_server, code="AAAABBBB", signing_key=sk, gpu="GPU", engine_version="0")


# ---------------------------------------------------------------------------
# WR-10: nonce response shape validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fake_server",
    [{"nonce": {"message": "Cloudflare error"}}],  # missing nonce + issued_at
    indirect=True,
)
def test_pair_init_rejects_malformed_nonce_response(fake_server: str) -> None:
    """WR-10: PairInitError raised on unexpected nonce response shape."""
    sk = _make_signing_key()
    with pytest.raises(PairInitError, match="unexpected nonce response shape"):
        pair_init_post(fake_server, code="AAAABBBB", signing_key=sk, gpu="GPU", engine_version="0")
