"""Tests for transcribe_engine.pairing.ed25519 — PyNaCl SigningKey wrappers.

Plan 08-03: load-or-generate a 32-byte Ed25519 private key, persist it with
mode 0o600, sign arbitrary messages, expose the pubkey as 64-char lowercase hex.

Negative tests are mandatory (T-08-03-03 mitigation):
  - sign-then-verify with wrong VerifyKey must fail.
"""
import os
from pathlib import Path

import nacl.signing
import pytest

from transcribe_engine.pairing.ed25519 import (
    load_or_generate_signing_key,
    public_key_hex,
    sign,
)
from transcribe_engine.state import State


def _fresh_state() -> State:
    return State()


def test_generate_creates_file_when_missing(tmp_path: Path) -> None:
    state = _fresh_state()
    sk = load_or_generate_signing_key(state, tmp_path)
    assert isinstance(sk, nacl.signing.SigningKey)
    assert state.keypair_path is not None
    keypair_file = Path(state.keypair_path)
    assert keypair_file.exists()
    # Raw 32-byte private key on disk (not PEM).
    assert keypair_file.stat().st_size == 32


@pytest.mark.skipif(os.name == "nt", reason="chmod is best-effort on Windows; ACLs differ")
def test_generate_persists_with_mode_0600(tmp_path: Path) -> None:
    state = _fresh_state()
    load_or_generate_signing_key(state, tmp_path)
    keypair_file = Path(state.keypair_path)  # type: ignore[arg-type]
    mode = keypair_file.stat().st_mode & 0o777
    assert mode == 0o600, f"expected 0o600, got 0o{mode:03o}"


def test_load_after_generate_returns_same_pubkey(tmp_path: Path) -> None:
    """Round-trip: generate, then re-instantiating from the same state.keypair_path
    must yield a SigningKey with the identical public key."""
    state = _fresh_state()
    sk1 = load_or_generate_signing_key(state, tmp_path)
    pub1 = public_key_hex(sk1)

    # Second call with state.keypair_path already set must reload, not regenerate.
    sk2 = load_or_generate_signing_key(state, tmp_path)
    pub2 = public_key_hex(sk2)
    assert pub1 == pub2


def test_state_keypair_path_set_after_generate(tmp_path: Path) -> None:
    state = _fresh_state()
    assert state.keypair_path is None
    load_or_generate_signing_key(state, tmp_path)
    assert state.keypair_path is not None
    assert state.keypair_path.endswith("keypair.bin")


def test_state_pubkey_set_after_generate(tmp_path: Path) -> None:
    """state.pubkey must hold the 64-char lowercase hex pubkey for downstream callers
    (Plan 05 device-row registration)."""
    state = _fresh_state()
    sk = load_or_generate_signing_key(state, tmp_path)
    assert state.pubkey == public_key_hex(sk)
    assert len(state.pubkey) == 64
    assert state.pubkey == state.pubkey.lower()
    int(state.pubkey, 16)  # must be valid hex


def test_sign_produces_64_bytes(tmp_path: Path) -> None:
    state = _fresh_state()
    sk = load_or_generate_signing_key(state, tmp_path)
    sig = sign(sk, b"challenge-bytes-here")
    assert isinstance(sig, bytes)
    assert len(sig) == 64  # Ed25519 detached signature length


def test_signature_verifies_against_pynacl_verify_key(tmp_path: Path) -> None:
    state = _fresh_state()
    sk = load_or_generate_signing_key(state, tmp_path)
    msg = b"hello pairing world"
    sig = sign(sk, msg)
    vk = sk.verify_key
    # raises BadSignatureError if invalid
    assert vk.verify(msg, sig) == msg


def test_sign_then_verify_with_wrong_key_fails(tmp_path: Path) -> None:
    """Negative test (T-08-03-03): a fresh VerifyKey must reject our signature."""
    state = _fresh_state()
    sk = load_or_generate_signing_key(state, tmp_path)
    msg = b"hello pairing world"
    sig = sign(sk, msg)
    other_vk = nacl.signing.SigningKey.generate().verify_key
    with pytest.raises(nacl.exceptions.BadSignatureError):
        other_vk.verify(msg, sig)


def test_public_key_hex_is_lowercase_64_char(tmp_path: Path) -> None:
    state = _fresh_state()
    sk = load_or_generate_signing_key(state, tmp_path)
    hex_pub = public_key_hex(sk)
    assert len(hex_pub) == 64
    assert hex_pub == hex_pub.lower()
    # Must be the raw VerifyKey bytes hex-encoded.
    assert bytes.fromhex(hex_pub) == sk.verify_key.encode()


def test_existing_keypair_outside_config_dir_is_loaded(tmp_path: Path) -> None:
    """If state.keypair_path points elsewhere (e.g. legacy install moved it), loader
    must honour it and NOT clobber by writing to <config_dir>/keypair.bin."""
    seed = b"\x01" * 32
    elsewhere = tmp_path / "subdir"
    elsewhere.mkdir()
    custom_path = elsewhere / "custom-keypair.bin"
    custom_path.write_bytes(seed)
    try:
        os.chmod(custom_path, 0o600)
    except OSError:
        pass

    state = State(keypair_path=str(custom_path))
    sk = load_or_generate_signing_key(state, tmp_path)
    # Loader must have honoured the existing path; SigningKey rebuilt from the seed.
    assert state.keypair_path == str(custom_path)
    assert sk.encode() == seed
