"""signaling/keepalive.py — Application-level ping/pong heartbeat (Plan 08-06).

Implements D-10: engine sends `ping` every 25s; if no `pong` is received within
50s, the connection is considered dead and on_dead() is called.

Usage:
    ka = Keepalive(
        send_ping=lambda: channel.send_broadcast("ping", {}),
        on_dead=lambda: asyncio.create_task(reconnect()),
    )
    await ka.start()
    ...
    # When a pong broadcast arrives:
    ka.record_pong()
    ...
    await ka.stop()
"""

import asyncio
import time
from collections.abc import Awaitable, Callable


class Keepalive:
    """Application-level keepalive with ping/pong monitoring.

    Args:
        send_ping:      Awaitable called every ping_interval seconds.
        on_dead:        Awaitable called once when no pong is received within dead_after.
        ping_interval:  How often to send a ping (seconds). Default: 25.0.
        dead_after:     How long without a pong before the connection is declared dead.
                        Default: 50.0 (must be > ping_interval for useful operation).
    """

    def __init__(
        self,
        send_ping: Callable[[], Awaitable[None]],
        on_dead: Callable[[], Awaitable[None]],
        *,
        ping_interval: float = 25.0,
        dead_after: float = 50.0,
    ) -> None:
        self._send_ping = send_ping
        self._on_dead = on_dead
        self._ping_interval = ping_interval
        self._dead_after = dead_after
        self._last_pong: float = time.monotonic()  # seed so we don't fire immediately
        self._task: asyncio.Task | None = None
        self._dead_fired: bool = False

    async def start(self) -> None:
        """Spawn the background keepalive task."""
        self._last_pong = time.monotonic()  # reset on (re)start
        self._dead_fired = False
        self._task = asyncio.create_task(self._run(), name="keepalive")

    def record_pong(self) -> None:
        """Call when a pong message arrives; resets the dead-connection timer."""
        self._last_pong = time.monotonic()

    async def stop(self) -> None:
        """Cancel the background task and wait for cleanup."""
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        """Background loop: ping every interval; declare dead if no pong arrives."""
        try:
            while True:
                await asyncio.sleep(self._ping_interval)
                await self._send_ping()
                elapsed_since_pong = time.monotonic() - self._last_pong
                if elapsed_since_pong > self._dead_after:
                    if not self._dead_fired:
                        self._dead_fired = True
                        await self._on_dead()
                    return  # stop the loop after declaring dead
        except asyncio.CancelledError:
            raise
