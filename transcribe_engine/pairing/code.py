"""Pairing-code generator — Crockford-base32, 8 chars, 40 bits of entropy.

Format: XXXX-XXXX (4 chars, hyphen, 4 chars).
Alphabet: 0-9 + A-Z minus I, L, O, U (those four are visually ambiguous with
0/1) — `CROCKFORD_BASE32` (32 symbols → 5 bits each → 40 bits total).

Entropy budget validated by Plan 08-03 RESEARCH for a single-use, 10-min code:
brute-forcing 2^40 = 1.1e12 attempts is infeasible inside the TTL window even
without rate limiting; with `pair_confirm` rate limiting (Plan 08-02) it is
beyond infeasible.

RNG: `secrets.choice` (cryptographic) — NOT `random.choices`.
"""

import secrets

CROCKFORD_BASE32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"  # omits I, L, O, U


def generate_pairing_code() -> str:
    """Return a fresh 8-char Crockford-base32 pairing code formatted XXXX-XXXX."""
    chars = "".join(secrets.choice(CROCKFORD_BASE32) for _ in range(8))
    return f"{chars[:4]}-{chars[4:]}"
