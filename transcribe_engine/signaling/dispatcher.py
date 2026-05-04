"""signaling/dispatcher.py — on_broadcast routing for EngineSignalingClient.

Factored from client.py per the plan's "if blowing past 300 LOC" guidance.
Handles parse_wire dispatch for all inbound broadcast events.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from transcribe_engine.protocol.messages import (
    CandidateMsg,
    OfferMsg,
    ResumeQueryMsg,
    parse_wire,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class BroadcastDispatcher:
    """Routes inbound Realtime broadcast payloads to typed handler callbacks.

    Usage:
        dispatcher = BroadcastDispatcher(on_offer=..., on_candidate=..., ...)
        channel.on_broadcast("description", dispatcher.handle)
        channel.on_broadcast("candidate", dispatcher.handle)
        ...
    """

    def __init__(
        self,
        *,
        on_offer: Callable[[OfferMsg], Awaitable[None]],
        on_candidate: Callable[[CandidateMsg], Awaitable[None]],
        on_resume_query: Callable[[ResumeQueryMsg], Awaitable[None]],
        on_pong: Callable[[], None],
        on_ping_reply: Callable[[], None],
    ) -> None:
        self._on_offer = on_offer
        self._on_candidate = on_candidate
        self._on_resume_query = on_resume_query
        self._on_pong = on_pong
        self._on_ping_reply = on_ping_reply

    def make_handler(self, event_name: str):
        """Return an on_broadcast callback for the given event name."""

        def _handler(payload: dict) -> None:
            raw_payload = payload.get("payload", {})
            if not isinstance(raw_payload, dict):
                logger.warning("Non-dict payload for event %r — dropping", event_name)
                return
            try:
                msg = parse_wire(json.dumps(raw_payload))
            except (ValueError, Exception) as exc:
                logger.warning("parse_wire failed for event %r: %s — dropping", event_name, exc)
                return

            msg_type = msg.get("type")  # type: ignore[union-attr]

            if msg_type == "pong":
                self._on_pong()
                return

            if msg_type == "ping":
                self._on_ping_reply()
                return

            if msg_type == "description":
                asyncio.create_task(self._safe_call(self._on_offer, msg))  # type: ignore[arg-type]
                return

            if msg_type == "candidate":
                asyncio.create_task(self._safe_call(self._on_candidate, msg))  # type: ignore[arg-type]
                return

            if msg_type == "resume_query":
                asyncio.create_task(self._safe_call(self._on_resume_query, msg))  # type: ignore[arg-type]
                return

            logger.debug("Unhandled event %r (msg_type=%r)", event_name, msg_type)

        return _handler

    async def _safe_call(self, fn: Callable, msg) -> None:
        """Call fn(msg) and log any exception without propagating."""
        try:
            await fn(msg)
        except Exception as exc:
            logger.exception("Broadcast handler %s raised: %s", fn.__name__, exc)
