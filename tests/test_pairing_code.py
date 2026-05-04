"""Tests for transcribe_engine.pairing.code — Crockford-base32 pairing-code RNG.

Plan 08-03 D-03: 8-char pairing code with 40 bits of entropy, formatted as XXXX-XXXX,
sampled from CROCKFORD_BASE32 (omits I, L, O, U) using `secrets.choice`.
"""
import re

from transcribe_engine.pairing.code import CROCKFORD_BASE32, generate_pairing_code

# Allowed alphabet — strict regex matches the public contract.
_CODE_RE = re.compile(r"^[0-9A-HJKMNPQRSTVWXYZ]{4}-[0-9A-HJKMNPQRSTVWXYZ]{4}$")


def test_format_is_4_dash_4() -> None:
    code = generate_pairing_code()
    assert len(code) == 9
    assert code[4] == "-"
    assert _CODE_RE.match(code), f"unexpected format: {code!r}"


def test_uses_only_crockford_alphabet() -> None:
    """No char outside CROCKFORD_BASE32 may appear (excludes the hyphen separator)."""
    for _ in range(200):
        code = generate_pairing_code()
        for ch in code.replace("-", ""):
            assert ch in CROCKFORD_BASE32, f"char {ch!r} not in Crockford alphabet"


def test_excludes_iluo() -> None:
    """The whole point of Crockford-base32 is that I, L, O, U never appear."""
    forbidden = set("ILOU")
    for _ in range(500):
        code = generate_pairing_code()
        assert not (set(code) & forbidden), f"forbidden char in {code!r}"


def test_distinct_calls_diverge() -> None:
    """1000 calls should produce >= 999 unique codes (40 bits of entropy → collision
    probability is ~1000²/2^40 ≈ 9e-7; effectively zero)."""
    codes = {generate_pairing_code() for _ in range(1000)}
    assert len(codes) >= 999, f"only {len(codes)} unique codes in 1000 calls"


def test_alphabet_constant_has_32_chars_no_iluo() -> None:
    """Lock the alphabet definition itself — guards against an editor accidentally
    inserting I/L/O/U or trimming a character."""
    assert len(CROCKFORD_BASE32) == 32
    assert CROCKFORD_BASE32 == "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    for ch in "ILOU":
        assert ch not in CROCKFORD_BASE32
