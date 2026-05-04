"""signaling/token.py — Ed25519 challenge-response Realtime JWT fetch (Plan 08-06).

Performs a two-step handshake against the Vercel /api/signal-token route:
  1. GET  /api/signal-token/nonce  — receives a random nonce + issued_at
  2. POST /api/signal-token        — sends pubkey + signature + nonce + issued_at
                                     receives short-lived Realtime JWT + channel name

Signed message template (canonical — Vercel verifier MUST use the same string):
    message = f"signal-token:{nonce}:{issued_at}".encode()

On 404 → raises EngineUnpairedError (caller clears state.paired_user_id).
On 401/429/5xx → raises SignalingTokenError (caller retries per backoff).

Implementation constraints (CLAUDE.md / SEC-08 sentinel):
  - stdlib urllib.request + json ONLY — no httpx/requests/aiohttp
  - No retry inside this function; the engine state machine drives retries
"""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Final
from urllib.parse import urlparse

from nacl.signing import SigningKey

from transcribe_engine.pairing.ed25519 import sign

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_NONCE_PATH: Final = "/api/signal-token/nonce"
_TOKEN_PATH: Final = "/api/signal-token"

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignalingToken:
    """Parsed response from a successful /api/signal-token POST."""

    token: str  # short-lived Realtime JWT (~5 min TTL)
    channel: str  # e.g. 'pair:<user_id>'
    supabase_url: str  # Supabase project URL (from the Vercel route)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class SignalingTokenError(Exception):
    """Raised on recoverable token fetch failure (4xx other than 404, 5xx, network).

    Caller sleeps the next backoff value and retries via reconnect_delays().
    """


class EngineUnpairedError(Exception):
    """Raised when /api/signal-token returns HTTP 404 (device_not_found).

    Indicates the engine's pairing has been revoked. Caller MUST clear
    state.paired_user_id and state.signaling, save state, and surface
    an 'unpaired' tray status — reconnect attempts are pointless.
    """


# ---------------------------------------------------------------------------
# TLS enforcement
# ---------------------------------------------------------------------------


def _require_tls(url: str) -> None:
    """Raise SignalingTokenError if URL is http:// and host is not localhost."""
    parsed = urlparse(url)
    if parsed.scheme != "https" and parsed.hostname not in ("localhost", "127.0.0.1"):
        raise SignalingTokenError(
            f"insecure base_url scheme: {parsed.scheme!r} "
            "(only https:// allowed for non-localhost targets)"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_signaling_token(
    base_url: str,
    signing_key: SigningKey,
    *,
    http_timeout: float = 10.0,
) -> SignalingToken:
    """Perform the Ed25519 challenge-response and return a SignalingToken.

    Steps:
      1. GET  {base_url}/api/signal-token/nonce — receive nonce + issued_at
      2. Build canonical message; sign with engine's Ed25519 key
      3. POST {base_url}/api/signal-token — receive token + channel + supabase_url

    Raises:
      SignalingTokenError:  base_url is http:// for a non-localhost host.
      SignalingTokenError:  Nonce GET fails (network error / non-200).
      SignalingTokenError:  Nonce response has unexpected shape.
      EngineUnpairedError:  POST returns 404 (device not found — engine unpaired).
      SignalingTokenError:  POST returns 401/429/5xx or network error.
    """
    _require_tls(base_url)
    base = base_url.rstrip("/")

    # Step 1: fetch nonce
    nonce_url = base + _NONCE_PATH
    nonce_req = urllib.request.Request(nonce_url, method="GET")
    try:
        with urllib.request.urlopen(nonce_req, timeout=http_timeout) as resp:
            nonce_data: dict = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise SignalingTokenError(f"nonce fetch failed (HTTP {exc.code})") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SignalingTokenError(f"nonce fetch failed: {exc}") from exc

    if (
        not isinstance(nonce_data, dict)
        or "nonce" not in nonce_data
        or "issued_at" not in nonce_data
    ):
        raise SignalingTokenError(f"unexpected nonce response shape: {nonce_data!r}")

    nonce: str = nonce_data["nonce"]
    issued_at_raw = nonce_data["issued_at"]
    if not isinstance(issued_at_raw, int):
        raise SignalingTokenError(
            f"nonce issued_at must be int, got {type(issued_at_raw).__name__}"
        )
    issued_at: int = issued_at_raw

    # Step 2: build canonical message and sign
    # Template (PLAN §D-01): "signal-token:{nonce}:{issued_at}"
    # Vercel /api/signal-token verifier MUST use the same template.
    message = f"signal-token:{nonce}:{issued_at}".encode()
    signature_hex = sign(signing_key, message).hex()
    pubkey_hex = signing_key.verify_key.encode().hex()

    # Step 3: POST
    payload = {
        "pubkey": pubkey_hex,
        "nonce": nonce,
        "issued_at": issued_at,
        "signature": signature_hex,
    }
    body = json.dumps(payload).encode()
    post_req = urllib.request.Request(
        base + _TOKEN_PATH,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(post_req, timeout=http_timeout) as resp:
            resp_data: dict = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        _handle_post_error(exc)
        return  # unreachable — _handle_post_error always raises
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise SignalingTokenError(f"signal-token POST failed: {exc}") from exc

    return SignalingToken(
        token=resp_data["token"],
        channel=resp_data["channel"],
        supabase_url=resp_data["supabase_url"],
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _handle_post_error(exc: urllib.error.HTTPError) -> None:
    """Parse error body and raise the appropriate exception. Never returns."""
    try:
        err_body: dict = json.loads(exc.read().decode())
        error_code: str = err_body.get("error", str(exc.code))
    except Exception:
        error_code = str(exc.code)

    if exc.code == 404:
        raise EngineUnpairedError(
            f"Engine unpaired — device not found (HTTP 404): {error_code}"
        ) from exc

    raise SignalingTokenError(f"signal-token failed (HTTP {exc.code}): {error_code}") from exc
