"""signaling/reconnect.py — Exponential backoff generator with jitter (Plan 08-06).

Implements D-09: 1 → 2 → 4 → 8 → 16 → 30 → 30 → 30 ... seconds,
each multiplied by (1 + uniform(-0.2, 0.2)) for ±20% jitter.

Usage:
    delays = reconnect_delays()
    for delay in delays:
        await asyncio.sleep(delay)
        try:
            await connect()
            break       # success — create a new iterator to reset the curve
        except SomeError:
            continue    # next loop iteration applies the next backoff value

To reset (on successful connect), discard the iterator and call reconnect_delays()
again. The sequence is stateless — each call produces a fresh iterator from step 1.
"""

import random
from collections.abc import Iterator

# Backoff curve base values (seconds). After 16 → 30, stays at 30 indefinitely.
_CURVE = [1, 2, 4, 8, 16, 30]


def reconnect_delays() -> Iterator[float]:
    """Yield backoff delays: 1, 2, 4, 8, 16, 30, 30, ... with ±20% jitter.

    Each yielded value = base × (1 + uniform(-0.2, 0.2)).
    Jitter is non-cryptographic (random.uniform — not secrets) because its
    purpose is thundering-herd prevention, not security.

    The iterator never exhausts — engine reconnect is infinite (D-09).
    To reset after a successful connection, create a new iterator instance.
    """
    for base in _CURVE:
        yield base * (1.0 + random.uniform(-0.2, 0.2))  # noqa: S311 — non-crypto jitter
    while True:
        yield 30.0 * (1.0 + random.uniform(-0.2, 0.2))  # noqa: S311
