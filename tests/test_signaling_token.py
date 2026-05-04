"""Tests for signaling/token.py — Ed25519 challenge-response token fetch.

Uses a ThreadingHTTPServer fixture (same pattern as test_resumable.py) to
simulate the Vercel /api/signal-token nonce + POST endpoints.
"""
import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from nacl.signing import SigningKey

from transcribe_engine.signaling.token import (
    EngineUnpairedError,
    SignalingToken,
    SignalingTokenError,
    fetch_signaling_token,
)


# ---------------------------------------------------------------------------
# Fake signal-token server
# ---------------------------------------------------------------------------

_FAKE_TOKEN = "eyJhbGciOiJIUzI1NiJ9.fake"
_FAKE_CHANNEL = "pair:user-uuid-1234"
_FAKE_SUPABASE_URL = "https://example.supabase.co"
_FAKE_NONCE = "abc123def456"
_FAKE_ISSUED_AT = 1700000000000


class _SignalTokenHandler(BaseHTTPRequestHandler):
    """Simple fake for /api/signal-token/nonce (GET) and /api/signal-token (POST)."""

    # Class-level controls for error injection
    nonce_status: int = 200
    post_status: int = 200
    # Written by POST handler so tests can inspect the received body
    last_received_body: dict = {}
    last_received_sig_msg: str = ""

    def log_message(self, fmt, *args):  # silence test logs
        return

    def do_GET(self):
        if self.path == "/api/signal-token/nonce":
            if self.nonce_status != 200:
                self.send_response(self.nonce_status)
                self.end_headers()
                return
            body = json.dumps(
                {"nonce": _FAKE_NONCE, "issued_at": _FAKE_ISSUED_AT}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/signal-token":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            _SignalTokenHandler.last_received_body = json.loads(raw.decode())

            if self.post_status == 404:
                body = json.dumps({"error": "device_not_found"}).encode()
                self.send_response(404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.post_status != 200:
                body = json.dumps({"error": "internal"}).encode()
                self.send_response(self.post_status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            body = json.dumps(
                {
                    "token": _FAKE_TOKEN,
                    "channel": _FAKE_CHANNEL,
                    "supabase_url": _FAKE_SUPABASE_URL,
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture
def signal_token_server():
    """ThreadingHTTPServer fixture for /api/signal-token routes."""
    # Reset class-level controls
    _SignalTokenHandler.nonce_status = 200
    _SignalTokenHandler.post_status = 200
    _SignalTokenHandler.last_received_body = {}

    server = ThreadingHTTPServer(("127.0.0.1", 0), _SignalTokenHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fetch_signaling_token_success(signal_token_server):
    """Happy path: returns a populated SignalingToken dataclass."""
    sk = SigningKey.generate()
    result = fetch_signaling_token(signal_token_server, sk)

    assert isinstance(result, SignalingToken)
    assert result.token == _FAKE_TOKEN
    assert result.channel == _FAKE_CHANNEL
    assert result.supabase_url == _FAKE_SUPABASE_URL


def test_fetch_signaling_token_sends_signed_payload(signal_token_server):
    """POST body contains pubkey, nonce, issued_at, and a valid Ed25519 signature."""
    sk = SigningKey.generate()
    fetch_signaling_token(signal_token_server, sk)

    body = _SignalTokenHandler.last_received_body
    assert "pubkey" in body
    assert "nonce" in body
    assert "issued_at" in body
    assert "signature" in body

    # Verify the signature covers the correct canonical message
    pubkey_bytes = bytes.fromhex(body["pubkey"])
    sig_bytes = bytes.fromhex(body["signature"])
    nonce = body["nonce"]
    issued_at = body["issued_at"]

    expected_msg = f"signal-token:{nonce}:{issued_at}".encode()

    from nacl.signing import VerifyKey
    vk = VerifyKey(pubkey_bytes)
    # Should not raise — valid signature over the expected message template
    vk.verify(expected_msg, sig_bytes)


def test_fetch_signaling_token_signature_template(signal_token_server):
    """The signed message template is exactly 'signal-token:{nonce}:{issued_at}'."""
    sk = SigningKey.generate()
    fetch_signaling_token(signal_token_server, sk)

    body = _SignalTokenHandler.last_received_body
    nonce = body["nonce"]
    issued_at = body["issued_at"]

    # The signed message string — validates the template spec from the plan
    signed_str = f"signal-token:{nonce}:{issued_at}"
    assert signed_str == f"signal-token:{_FAKE_NONCE}:{_FAKE_ISSUED_AT}"


def test_fetch_signaling_token_404_raises_engine_unpaired(signal_token_server):
    """404 from POST /api/signal-token raises EngineUnpairedError."""
    _SignalTokenHandler.post_status = 404
    sk = SigningKey.generate()

    with pytest.raises(EngineUnpairedError):
        fetch_signaling_token(signal_token_server, sk)


def test_fetch_signaling_token_401_raises_signaling_token_error(signal_token_server):
    """401 from POST /api/signal-token raises SignalingTokenError."""
    _SignalTokenHandler.post_status = 401
    sk = SigningKey.generate()

    with pytest.raises(SignalingTokenError):
        fetch_signaling_token(signal_token_server, sk)


def test_fetch_signaling_token_500_raises_signaling_token_error(signal_token_server):
    """500 from POST /api/signal-token raises SignalingTokenError."""
    _SignalTokenHandler.post_status = 500
    sk = SigningKey.generate()

    with pytest.raises(SignalingTokenError):
        fetch_signaling_token(signal_token_server, sk)
