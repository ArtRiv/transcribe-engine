"""signaling — Supabase Realtime signaling channel for the engine (Plan 08-06).

Modules:
  token.py      — Ed25519 challenge-response; fetches short-lived Realtime JWT
  reconnect.py  — Exponential backoff generator with ±20% jitter
  keepalive.py  — Application-level ping/pong heartbeat manager
  client.py     — EngineSignalingClient wrapping AsyncRealtimeClient
"""

from transcribe_engine.signaling.keepalive import Keepalive
from transcribe_engine.signaling.reconnect import reconnect_delays
from transcribe_engine.signaling.token import (
    EngineUnpairedError,
    SignalingToken,
    SignalingTokenError,
    fetch_signaling_token,
)

__all__ = [
    "EngineUnpairedError",
    "SignalingToken",
    "SignalingTokenError",
    "fetch_signaling_token",
    "reconnect_delays",
    "Keepalive",
]
