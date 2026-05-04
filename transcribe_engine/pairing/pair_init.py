"""Engine-side pair-init HTTP client (Plan 08-05, D-02).

POSTs `(code, pubkey, signature, nonce, issued_at, gpu, engine_version)` to
`${BASE_URL}/api/pair-init` BEFORE the engine opens the user's browser to the
pair page — synchronous handshake guarantees the server has code+pubkey on
file before the user sees the pair card.

**Wire message template (B-02 — locked for Plan 04b cross-reference):**
    message = f"pair-init:{code}:{nonce}:{issued_at}:{gpu}:{engine_version}".encode()

The signature binds the displayed fingerprint values (gpu, engine_version) to
the engine's identity — a MITM cannot swap those labels without invalidating
the sig (T-08-05-03 mitigation).

Implementation:
- stdlib urllib.request ONLY — no httpx/requests/aiohttp (T-08-05-04)
- JSON encode/decode via json.dumps/json.loads only
- No retry inside this function — the engine state machine drives retries
- Connect + read timeout both default to 10.0 s
"""
import json
import urllib.request
from typing import Final

from nacl.signing import SigningKey

from transcribe_engine.pairing.ed25519 import sign

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PairInitError(Exception):
    """Raised when /api/pair-init returns an unexpected 4xx or 5xx status."""


class PairCodeConflictError(PairInitError):
    """Raised when /api/pair-init returns HTTP 409 (code already claimed).

    Caller (engine state machine) regenerates a fresh pairing code and retries.
    """


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

_NONCE_PATH: Final = "/api/signal-token/nonce"
_PAIR_INIT_PATH: Final = "/api/pair-init"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def pair_init_post(
    base_url: str,
    code: str,
    signing_key: SigningKey,
    *,
    gpu: str,
    engine_version: str,
    http_timeout: float = 10.0,
) -> None:
    """POST pair-init payload to the frontend API.

    Steps:
      1. GET  ``{base_url}/api/signal-token/nonce`` — receive ``nonce`` + ``issued_at``.
      2. Build and sign the canonical message (B-02 template).
      3. POST ``{base_url}/api/pair-init`` with the 7-field JSON body.

    Raises:
      PairCodeConflictError: HTTP 409 — caller should regenerate code.
      PairInitError:         Other 4xx/5xx — contains route ``error`` code.
    """
    # --- Step 1: fetch nonce ---
    nonce_url = base_url.rstrip("/") + _NONCE_PATH
    nonce_req = urllib.request.Request(nonce_url, method="GET")
    with urllib.request.urlopen(nonce_req, timeout=http_timeout) as resp:
        nonce_data: dict = json.loads(resp.read().decode())
    nonce: str = nonce_data["nonce"]
    issued_at: str = nonce_data["issued_at"]

    # --- Step 2: build signed message (B-02 locked template) ---
    # Plan 04b verifier MUST use this exact template string.
    message = f"pair-init:{code}:{nonce}:{issued_at}:{gpu}:{engine_version}".encode()
    signature_hex = sign(signing_key, message).hex()
    pubkey_hex = signing_key.verify_key.encode().hex()

    # --- Step 3: POST ---
    payload = {
        "code": code,
        "pubkey": pubkey_hex,
        "signature": signature_hex,
        "nonce": nonce,
        "issued_at": issued_at,
        "gpu": gpu,
        "engine_version": engine_version,
    }
    body = json.dumps(payload).encode()
    pair_url = base_url.rstrip("/") + _PAIR_INIT_PATH
    post_req = urllib.request.Request(
        pair_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(post_req, timeout=http_timeout) as resp:
            resp.read()  # consume body; 2xx is success
    except urllib.error.HTTPError as exc:
        _handle_http_error(exc)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _handle_http_error(exc: urllib.error.HTTPError) -> None:
    """Parse error body and raise the appropriate exception."""
    try:
        err_body: dict = json.loads(exc.read().decode())
        error_code: str = err_body.get("error", str(exc.code))
    except Exception:
        error_code = str(exc.code)

    if exc.code == 409:
        raise PairCodeConflictError(
            f"Pairing code conflict (HTTP 409): {error_code}"
        ) from exc
    raise PairInitError(
        f"pair-init failed (HTTP {exc.code}): {error_code}"
    ) from exc
