"""Tests for signaling/reconnect.py — exponential backoff generator with jitter.

Validates the reconnect curve: 1, 2, 4, 8, 16, 30, 30, 30, ...
with ±20% jitter applied to each value.
"""
import pytest

from transcribe_engine.signaling.reconnect import reconnect_delays


def _snap_to_curve(v: float) -> int:
    """Snap a jittered delay to the nearest expected base value."""
    curve = [1, 2, 4, 8, 16, 30]
    return min(curve, key=lambda c: abs(v - c))


def test_first_sample_mean_within_jitter_band():
    """Mean of 100 first-sample draws should be ∈ [0.8, 1.2] (base=1 ±20%)."""
    samples = []
    for _ in range(100):
        gen = reconnect_delays()
        samples.append(next(gen))

    mean = sum(samples) / len(samples)
    assert 0.8 <= mean <= 1.2, f"First-sample mean {mean:.3f} outside [0.8, 1.2]"


def test_all_samples_within_jitter_band():
    """Every jittered value for a known base must stay within ±20% of that base."""
    gen = reconnect_delays()
    expected_bases = [1, 2, 4, 8, 16, 30, 30, 30]
    for base in expected_bases:
        value = next(gen)
        lo = base * 0.8
        hi = base * 1.2
        assert lo <= value <= hi, (
            f"Value {value:.3f} for base {base} is outside [{lo:.3f}, {hi:.3f}]"
        )


def test_curve_snaps_correctly():
    """Ignoring jitter, the first 8 values snap to 1,2,4,8,16,30,30,30."""
    gen = reconnect_delays()
    expected = [1, 2, 4, 8, 16, 30, 30, 30]
    snapped = [_snap_to_curve(next(gen)) for _ in expected]
    assert snapped == expected, f"Snapped curve {snapped} != {expected}"


def test_generator_continues_indefinitely():
    """Generator yields values forever without StopIteration."""
    gen = reconnect_delays()
    for _ in range(50):
        val = next(gen)
        # Cap is 30 ±20% = max 36
        assert val <= 36, f"Value {val:.3f} exceeds max cap of 36"


def test_max_value_capped_at_36():
    """No value exceeds 30 × 1.2 = 36 (cap + max jitter)."""
    gen = reconnect_delays()
    for _ in range(100):
        val = next(gen)
        assert val <= 36, f"Value {val:.3f} exceeds 36"


def test_fresh_iterator_starts_from_one():
    """Creating a new iterator always starts from the base-1 step."""
    gen1 = reconnect_delays()
    # exhaust the first few steps
    for _ in range(10):
        next(gen1)

    # New iterator starts fresh
    gen2 = reconnect_delays()
    first = next(gen2)
    assert 0.8 <= first <= 1.2, f"Fresh iterator first value {first:.3f} not near 1.0"
