"""Pairing primitives — pairing-code RNG + Ed25519 sign (Plan 08-03).

Pure-function modules used by the Phase 8 pairing flow:
  - code.py — Crockford-base32 8-char pairing code (40 bits of entropy)
  - ed25519.py — PyNaCl SigningKey load/generate/persist + sign

Crypto here is delegated to the stdlib `secrets` module (RNG) and PyNaCl
(Ed25519). No hand-rolled crypto. RFC 8032 §5.1 plain Ed25519 only — no
Ed25519ph / Ed25519ctx variants.
"""
