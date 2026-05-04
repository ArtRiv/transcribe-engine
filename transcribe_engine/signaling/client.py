"""signaling/client.py — EngineSignalingClient (Plan 08-06).

Manages the engine's persistent Supabase Realtime broadcast channel:
  - Ed25519 challenge-response token fetch on every connect (D-01)
  - Exponential backoff reconnect with jitter (D-09)
  - Application-level ping/pong keepalive (D-10)
  - State-event broadcasting (D-11)
  - Detects unpair via 404 on /api/signal-token (PAIR-06)

After this plan, Plan 07 calls emit_state() on pipeline transitions;
Plan 09 wraps run() in asyncio.create_task() and passes set_connection_state
from tray/status.py as on_state_change.

Inbound broadcast routing is factored into signaling/dispatcher.py.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Literal

from realtime import AsyncRealtimeClient, RealtimeSubscribeStates

from transcribe_engine import __version__
from transcribe_engine.protocol.messages import (
    PROTOCOL_VERSION,
    CandidateMsg,
    OfferMsg,
    ResumeQueryMsg,
)
from transcribe_engine.signaling.dispatcher import BroadcastDispatcher
from transcribe_engine.signaling.keepalive import Keepalive
from transcribe_engine.signaling.reconnect import reconnect_delays
from transcribe_engine.signaling.token import EngineUnpairedError, fetch_signaling_token
from transcribe_engine.state import State

logger = logging.getLogger(__name__)

# Maximum token-fetch failures before treating as definitively unpaired.
_MAX_TOKEN_FAILURES = 5

ConnectionState = Literal["connected", "reconnecting", "offline"]
EngineState = Literal["idle", "transcribing", "loading_model", "gpu_warming"]


class UnpairedExit(Exception):
    """Raised by run() when the engine is confirmed unpaired.

    Caller (__main__.py / tray) should surface 'unpaired' state and stop
    attempting reconnects — the engine has no valid pairing to reconnect to.
    """


class EngineSignalingClient:
    """Persistent Supabase Realtime client for engine↔frontend signaling.

    Args:
        state:            Engine State object (will be mutated on unpair).
        base_url:         Vercel frontend URL (e.g. https://app.example.com).
        signing_key:      Engine Ed25519 SigningKey (from state.keypair_path).
        on_state_change:  Callback for connection state transitions.
        on_offer:         Awaitable handler for inbound SDP offer.
        on_candidate:     Awaitable handler for inbound ICE candidate.
        on_resume_query:  Awaitable handler for inbound resume_query message.
        gpu_label:        GPU label to include in the hello broadcast.
    """

    def __init__(
        self,
        *,
        state: State,
        base_url: str,
        signing_key,  # nacl.signing.SigningKey — avoid importing nacl at type-check time
        on_state_change: Callable[[ConnectionState], None],
        on_offer: Callable[[OfferMsg], Awaitable[None]],
        on_candidate: Callable[[CandidateMsg], Awaitable[None]],
        on_resume_query: Callable[[ResumeQueryMsg], Awaitable[None]],
        gpu_label: str,
    ) -> None:
        self._state = state
        self._base_url = base_url
        self._signing_key = signing_key
        self._on_state_change = on_state_change
        self._gpu_label = gpu_label
        self._channel = None  # AsyncRealtimeChannel while connected
        self._rt_client = None  # AsyncRealtimeClient while connected
        self._keepalive: Keepalive | None = None

        self._dispatcher = BroadcastDispatcher(
            on_offer=on_offer,
            on_candidate=on_candidate,
            on_resume_query=on_resume_query,
            on_pong=self._record_pong,
            on_ping_reply=self._send_pong,
        )

    # -------------------------------------------------------------------------
    # Public run loop
    # -------------------------------------------------------------------------

    async def run(self) -> None:
        """Connect to Realtime, subscribe to the channel, loop forever.

        Reconnects on any error using exponential backoff (D-09).
        Raises UnpairedExit when the engine is confirmed unpaired (PAIR-06).
        """
        delays = reconnect_delays()
        consecutive_token_failures = 0

        while True:
            self._on_state_change("reconnecting")

            # --- Token fetch (stdlib urllib — sync; handle async mocks too) ---
            try:
                _result = fetch_signaling_token(self._base_url, self._signing_key)
                sig_token = await _result if inspect.iscoroutine(_result) else _result
                consecutive_token_failures = 0
            except EngineUnpairedError as exc:
                logger.warning("Engine unpaired (device_not_found): %s", exc)
                self._on_state_change("offline")
                self._clear_pairing()
                raise UnpairedExit("Engine has been unpaired") from exc
            except Exception as exc:
                consecutive_token_failures += 1
                logger.error(
                    "Token fetch failed (%d/%d): %s",
                    consecutive_token_failures,
                    _MAX_TOKEN_FAILURES,
                    exc,
                )
                if consecutive_token_failures >= _MAX_TOKEN_FAILURES:
                    self._on_state_change("offline")
                    self._clear_pairing()
                    raise UnpairedExit("Token fetch failed too many times") from exc
                await asyncio.sleep(next(delays))
                continue

            # --- Realtime connection ---
            rt = AsyncRealtimeClient(
                sig_token.supabase_url,
                sig_token.token,
                auto_reconnect=False,  # we own the reconnect loop
                max_retries=1,
            )
            self._rt_client = rt
            channel_name = sig_token.channel

            try:
                await rt.connect()
            except Exception as exc:
                logger.warning("Realtime connect failed: %s", exc)
                await _close_quietly(rt)
                await asyncio.sleep(next(delays))
                continue

            # --- Channel subscribe ---
            ch = rt.channel(channel_name)
            self._channel = ch

            _ch_name = channel_name  # capture for closure

            def _on_subscribe(status: str, err=None, _n=_ch_name) -> None:
                if status in (RealtimeSubscribeStates.SUBSCRIBED, "SUBSCRIBED"):
                    pass  # subscribed_event not needed — success path below
                elif status in (
                    RealtimeSubscribeStates.CHANNEL_ERROR,
                    "CHANNEL_ERROR",
                    RealtimeSubscribeStates.CLOSED,
                    "CLOSED",
                ):
                    logger.warning("Channel %s status: %s (err=%s)", _n, status, err)

            for event in ("description", "candidate", "resume_query", "ping", "pong"):
                ch.on_broadcast(event, self._dispatcher.make_handler(event))

            try:
                await ch.subscribe(_on_subscribe)
            except Exception as exc:
                logger.warning("Channel subscribe failed: %s", exc)
                await _close_quietly(rt)
                await asyncio.sleep(next(delays))
                continue

            delays = reconnect_delays()  # reset on successful connect
            self._on_state_change("connected")

            # Send hello
            hello: dict = {
                "type": "hello",
                "version": __version__,
                "protocol_version": PROTOCOL_VERSION,
                "gpu": self._gpu_label,
            }
            try:
                await ch.send_broadcast("hello", hello)
            except Exception as exc:
                logger.warning("Failed to send hello: %s", exc)

            # Start keepalive
            _ch_cap = ch  # bind loop variable for lambda

            ka = Keepalive(
                send_ping=lambda _c=_ch_cap: _c.send_broadcast("ping", {"type": "ping"}),
                on_dead=self._on_keepalive_dead,
                ping_interval=25.0,
                dead_after=50.0,
            )
            self._keepalive = ka
            await ka.start()

            # Park until cancelled or error
            try:
                while True:
                    await asyncio.sleep(5.0)
            except asyncio.CancelledError:
                await ka.stop()
                await _close_quietly(rt)
                raise
            except Exception as exc:
                logger.warning("Realtime session error: %s — reconnecting", exc)

            await ka.stop()
            await _close_quietly(rt)
            await asyncio.sleep(next(delays))

    # -------------------------------------------------------------------------
    # Emit helpers (called by Plan 07 WebRTC handler)
    # -------------------------------------------------------------------------

    async def emit_state(self, value: EngineState) -> None:
        """Broadcast a StateMsg on the signaling channel (D-11)."""
        await self.emit({"type": "state", "value": value})

    async def emit(self, msg: dict) -> None:
        """Forward an arbitrary outbound message through the signaling channel.

        No-op if channel is not currently subscribed.
        """
        if self._channel is None:
            logger.debug("emit() called with no active channel — dropping %r", msg.get("type"))
            return
        event = msg.get("type", "unknown")
        try:
            await self._channel.send_broadcast(event, msg)
        except Exception as exc:
            logger.warning("emit(%s) failed: %s", event, exc)

    # -------------------------------------------------------------------------
    # Internal callbacks
    # -------------------------------------------------------------------------

    def _record_pong(self) -> None:
        if self._keepalive:
            self._keepalive.record_pong()

    def _send_pong(self) -> None:
        asyncio.create_task(self.emit({"type": "pong"}))

    def _clear_pairing(self) -> None:
        """Clear signaling state from State (PAIR-06 unpair)."""
        self._state.paired_user_id = None
        self._state.signaling.channel = None
        self._state.signaling.supabase_url = None
        self._channel = None
        self._rt_client = None

    async def _on_keepalive_dead(self) -> None:
        logger.warning("Keepalive: no pong received — declaring connection dead")
        self._on_state_change("reconnecting")
        if self._rt_client is not None:
            await _close_quietly(self._rt_client)


async def _close_quietly(rt) -> None:
    """Close an AsyncRealtimeClient without raising."""
    try:
        await rt.close()
    except Exception as exc:
        logger.debug("rt.close() raised: %s", exc)
