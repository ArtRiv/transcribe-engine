"""Tests for signaling/keepalive.py — ping/pong keepalive manager.

Uses asyncio + monkeypatched time.monotonic to drive timing without real sleeps.
"""
import asyncio

import pytest

from transcribe_engine.signaling.keepalive import Keepalive


@pytest.mark.asyncio
async def test_send_ping_called_multiple_times():
    """send_ping is called once per ping_interval elapsed."""
    ping_count = 0

    async def send_ping():
        nonlocal ping_count
        ping_count += 1

    async def on_dead():
        pass

    # Use a very short interval for testing
    ka = Keepalive(
        send_ping=send_ping,
        on_dead=on_dead,
        ping_interval=0.02,
        dead_after=1.0,
    )
    await ka.start()

    # Wait for 5 ping intervals (5 × 0.02s = 0.1s), with some margin
    await asyncio.sleep(0.12)
    await ka.stop()

    assert ping_count >= 4, f"Expected at least 4 pings, got {ping_count}"


@pytest.mark.asyncio
async def test_on_dead_called_when_pong_not_received():
    """on_dead fires exactly once when dead_after elapses without pong."""
    dead_called = 0

    async def send_ping():
        pass

    async def on_dead():
        nonlocal dead_called
        dead_called += 1

    ka = Keepalive(
        send_ping=send_ping,
        on_dead=on_dead,
        ping_interval=0.01,
        dead_after=0.05,
    )
    await ka.start()

    # Wait longer than dead_after to ensure on_dead fires
    await asyncio.sleep(0.15)
    await ka.stop()

    assert dead_called == 1, f"on_dead should be called exactly once, got {dead_called}"


@pytest.mark.asyncio
async def test_record_pong_prevents_dead_call():
    """If pong arrives regularly, on_dead is never called."""
    dead_called = 0

    async def send_ping():
        pass

    async def on_dead():
        nonlocal dead_called
        dead_called += 1

    ka = Keepalive(
        send_ping=send_ping,
        on_dead=on_dead,
        ping_interval=0.02,
        dead_after=0.08,
    )
    await ka.start()

    # Record pong every 0.03s — faster than dead_after
    for _ in range(5):
        await asyncio.sleep(0.03)
        ka.record_pong()

    await ka.stop()

    assert dead_called == 0, "on_dead should not fire when pong is received regularly"


@pytest.mark.asyncio
async def test_stop_cancels_task():
    """stop() terminates the background task cleanly."""
    async def send_ping():
        pass

    async def on_dead():
        pass

    ka = Keepalive(
        send_ping=send_ping,
        on_dead=on_dead,
        ping_interval=0.02,
        dead_after=1.0,
    )
    await ka.start()
    # Task should be running
    assert ka._task is not None
    assert not ka._task.done()

    await ka.stop()
    # Task should be done after stop
    assert ka._task.done()


@pytest.mark.asyncio
async def test_on_dead_called_at_most_once():
    """on_dead is not called more than once even if dead_after elapses repeatedly."""
    dead_count = 0

    async def send_ping():
        pass

    async def on_dead():
        nonlocal dead_count
        dead_count += 1

    ka = Keepalive(
        send_ping=send_ping,
        on_dead=on_dead,
        ping_interval=0.01,
        dead_after=0.03,
    )
    await ka.start()

    # Wait well beyond dead_after
    await asyncio.sleep(0.2)
    await ka.stop()

    assert dead_count == 1, f"on_dead called {dead_count} times, expected exactly 1"
