"""Ed25519 key persistence + signing for the engine pairing flow (Plan 08-03).

PyNaCl-only — no `cryptography.hazmat.*`, no hand-rolled crypto.

Key file format: 32 raw bytes on disk (NOT PEM). Filename `keypair.bin` under
`<config_dir>/`. POSIX permissions are tightened to 0o600 immediately after
creation; on Windows the chmod is best-effort and per-user `%APPDATA%` ACLs
provide the equivalent guarantee (see `picker_server.save_hf_token` for the
same idiom).

This module signs with plain Ed25519 (RFC 8032 §5.1) — never Ed25519ph or
Ed25519ctx (ASVS V6 / T-08-03-05 mitigation). PyNaCl's `SigningKey.sign()`
defaults to plain Ed25519 and exposes no parameter to switch variants, which
is the property we want.

The State object is mutated in place (sets `state.keypair_path` and
`state.pubkey`). The CALLER is responsible for `state.save(state_path)`
afterward — Plan 05 owns the schema bump and pairing-flow lifecycle.
"""

import os
from pathlib import Path

from nacl.signing import SigningKey

KEYPAIR_FILENAME = "keypair.bin"


def load_or_generate_signing_key(state, config_dir: Path) -> SigningKey:
    """Return a PyNaCl SigningKey, loading from `state.keypair_path` if set and
    the file exists, otherwise generating a new one and persisting it.

    On generation:
      - Writes 32 raw bytes to `<config_dir>/keypair.bin`.
      - Tightens permissions to 0o600 (POSIX) — best-effort on Windows.
      - Sets `state.keypair_path` and `state.pubkey` (lowercase 64-char hex).

    The State dataclass is imported lazily-via-duck-typing (we only touch the
    `keypair_path` and `pubkey` attributes) so this module doesn't pull
    `transcribe_engine.state` into its dependency surface.
    """
    if state.keypair_path:
        existing = Path(state.keypair_path)
        if existing.exists():
            seed = existing.read_bytes()
            if len(seed) != 32:
                raise ValueError(
                    f"keypair file {existing} has wrong size: {len(seed)} bytes "
                    f"(expected 32 raw Ed25519 seed bytes)"
                )
            sk = SigningKey(seed)
            # Refresh pubkey on State in case it drifted (defensive, no-op normally).
            state.pubkey = sk.verify_key.encode().hex()
            return sk

    # Generate fresh keypair.
    sk = SigningKey.generate()
    config_dir.mkdir(parents=True, exist_ok=True)
    keypair_path = config_dir / KEYPAIR_FILENAME
    keypair_path.write_bytes(sk.encode())
    try:
        os.chmod(keypair_path, 0o600)
    except OSError:
        # Windows: chmod is best-effort. %APPDATA% is per-user by default.
        pass

    state.keypair_path = str(keypair_path)
    state.pubkey = sk.verify_key.encode().hex()
    return sk


def sign(signing_key: SigningKey, message: bytes) -> bytes:
    """Return the 64-byte detached Ed25519 signature for `message`."""
    return signing_key.sign(message).signature


def public_key_hex(signing_key: SigningKey) -> str:
    """Return the 32-byte raw VerifyKey as a 64-char lowercase hex string."""
    return signing_key.verify_key.encode().hex()
