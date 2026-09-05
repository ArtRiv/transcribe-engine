"""Tests for signaling/client.py — EngineSignalingClient.

Uses an in-process fake Realtime broker (FakeRealtimeClient + FakeChannel)
to exercise subscribe/broadcast/reconnect/unpair logic without a real
Supabase Realtime server.

The fake broker is wired as a pytest fixture in this file (not conftest.py,
since it is only needed here).
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from nacl.signing import SigningKey

from transcribe_engine.protocol.messages import (
    PROTOCOL_VERSION,
)
from transcribe_engine.signaling.client import EngineSignalingClient, UnpairedExit
from transcribe_engine.signaling.token import EngineUnpairedError, SignalingToken
from transcribe_engine.state import Signaling, State

# ---------------------------------------------------------------------------
# Fake Realtime broker
# ---------------------------------------------------------------------------


class FakeChannel:
    """Minimal fake of AsyncRealtimeChannel for unit tests.

    Supports:
      - on_broadcast(event, cb): registers a handler
      - subscribe(cb=None): calls cb with 'SUBSCRIBED' state; marks subscribed
      - send_broadcast(event, data): records the broadcast; fires local listeners
      - inject_broadcast(event, payload): inject an inbound broadcast from the 'server'
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list] = {}
        self.broadcasts_sent: list[tuple[str, Any]] = []
        self._subscribe_cb = None
        self.subscribed = False

    def on_broadcast(self, event: str, callback) -> "FakeChannel":
        self._handlers.setdefault(event, []).append(callback)
        return self

    async def subscribe(self, callback=None) -> "FakeChannel":
        self._subscribe_cb = callback
        self.subscribed = True
        if callback:
            callback("SUBSCRIBED", None)
        return self

    async def send_broadcast(self, event: str, data: Any) -> None:
        self.broadcasts_sent.append((event, data))
        # Also fire any local listeners for loopback tests
        for cb in self._handlers.get(event, []):
            payload = {"event": event, "payload": data}
            cb(payload)

    def inject_broadcast(self, event: str, payload: Any) -> None:
        """Simulate an inbound broadcast from the remote side."""
        for cb in self._handlers.get(event, []):
            msg = {"event": event, "payload": payload}
            cb(msg)

    async def unsubscribe(self) -> None:
        self.subscribed = False


class FakeRealtimeClient:
    """Minimal fake of AsyncRealtimeClient for unit tests.

    Wraps FakeChannel instances keyed by channel name. Set
    .raise_on_connect to an exception to simulate connection errors.
    """

    def __init__(self, url: str, token: str) -> None:
        self.url = url
        self.token = token
        self.channels: dict[str, FakeChannel] = {}
        self.connected = False
        self.raise_on_connect: Exception | None = None
        self.connect_call_count = 0

    async def connect(self) -> None:
        self.connect_call_count += 1
        if self.raise_on_connect is not None:
            raise self.raise_on_connect
        self.connected = True

    def channel(self, name: str, params=None) -> FakeChannel:
        if name not in self.channels:
            self.channels[name] = FakeChannel()
        return self.channels[name]

    async def close(self) -> None:
        self.connected = False

    async def remove_channel(self, channel: FakeChannel) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(channel: str = "pair:user-abc") -> State:
    return State(
        paired_user_id="user-abc",
        signaling=Signaling(
            supabase_url="https://fake.supabase.co",
            channel=channel,
        ),
    )


def _make_signing_key() -> SigningKey:
    return SigningKey.generate()


def _fake_token(channel: str = "pair:user-abc") -> SignalingToken:
    return SignalingToken(
        token="fake.jwt.token",
        channel=channel,
        supabase_url="https://fake.supabase.co",
    )


# ---------------------------------------------------------------------------
# Fake token fetch
# ---------------------------------------------------------------------------


async def _good_fetch(base_url: str, signing_key: SigningKey, **kwargs) -> SignalingToken:
    return _fake_token()


async def _unpaired_fetch(base_url: str, signing_key: SigningKey, **kwargs) -> SignalingToken:
    raise EngineUnpairedError("device_not_found")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_sends_hello_on_subscribed():
    """After subscribe, client broadcasts a hello with correct fields."""
    state = _make_state()
    sk = _make_signing_key()
    on_state_change = MagicMock()
    fake_rt = FakeRealtimeClient("https://fake.supabase.co", "fake.jwt")

    client = EngineSignalingClient(
        state=state,
        base_url="https://fake.vercel.app",
        signing_key=sk,
        on_state_change=on_state_change,
        on_offer=AsyncMock(),
        on_candidate=AsyncMock(),
        on_resume_query=AsyncMock(),
        gpu_label="TestGPU",
    )

    # Patch fetch_signaling_token and AsyncRealtimeClient
    with (
        patch(
            "transcribe_engine.signaling.client.fetch_signaling_token",
            side_effect=_good_fetch,
        ),
        patch(
            "transcribe_engine.signaling.client.AsyncRealtimeClient",
            return_value=fake_rt,
        ),
    ):
        # Run for just long enough to connect + subscribe, then cancel
        task = asyncio.create_task(client.run())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, UnpairedExit):
            pass

    channel = fake_rt.channels.get("pair:user-abc")
    assert channel is not None, "Channel was never created"

    hello_broadcasts = [b for b in channel.broadcasts_sent if b[0] == "hello"]
    assert hello_broadcasts, "No hello broadcast sent"

    _, payload = hello_broadcasts[0]
    assert payload.get("type") == "hello"
    assert payload.get("protocol_version") == PROTOCOL_VERSION
    assert payload.get("gpu") == "TestGPU"
    assert "version" in payload


@pytest.mark.asyncio
async def test_client_calls_on_state_change_connected_after_subscribe():
    """on_state_change('connected') is called after successful subscribe."""
    state = _make_state()
    sk = _make_signing_key()
    on_state_change = MagicMock()
    fake_rt = FakeRealtimeClient("https://fake.supabase.co", "fake.jwt")

    client = EngineSignalingClient(
        state=state,
        base_url="https://fake.vercel.app",
        signing_key=sk,
        on_state_change=on_state_change,
        on_offer=AsyncMock(),
        on_candidate=AsyncMock(),
        on_resume_query=AsyncMock(),
        gpu_label="TestGPU",
    )

    with (
        patch(
            "transcribe_engine.signaling.client.fetch_signaling_token",
            side_effect=_good_fetch,
        ),
        patch(
            "transcribe_engine.signaling.client.AsyncRealtimeClient",
            return_value=fake_rt,
        ),
    ):
        task = asyncio.create_task(client.run())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, UnpairedExit):
            pass

    state_calls = [str(c.args[0]) for c in on_state_change.call_args_list]
    assert "connected" in state_calls, f"Expected 'connected' in {state_calls}"


@pytest.mark.asyncio
async def test_client_handles_unpaired_token_404_by_exiting_with_state_clear():
    """EngineUnpairedError clears state and raises UnpairedExit."""
    state = _make_state()
    sk = _make_signing_key()
    on_state_change = MagicMock()

    client = EngineSignalingClient(
        state=state,
        base_url="https://fake.vercel.app",
        signing_key=sk,
        on_state_change=on_state_change,
        on_offer=AsyncMock(),
        on_candidate=AsyncMock(),
        on_resume_query=AsyncMock(),
        gpu_label="TestGPU",
    )

    with patch(
        "transcribe_engine.signaling.client.fetch_signaling_token",
        side_effect=_unpaired_fetch,
    ):
        with pytest.raises(UnpairedExit):
            await client.run()

    # State should be cleared
    assert state.paired_user_id is None
    assert state.signaling.channel is None
    on_state_change.assert_any_call("offline")


@pytest.mark.asyncio
async def test_client_dispatches_offer_to_handler():
    """Inbound 'description' broadcast with sdp.type='offer' calls on_offer."""
    state = _make_state()
    sk = _make_signing_key()
    on_offer = AsyncMock()
    fake_rt = FakeRealtimeClient("https://fake.supabase.co", "fake.jwt")

    client = EngineSignalingClient(
        state=state,
        base_url="https://fake.vercel.app",
        signing_key=sk,
        on_state_change=MagicMock(),
        on_offer=on_offer,
        on_candidate=AsyncMock(),
        on_resume_query=AsyncMock(),
        gpu_label="TestGPU",
    )

    with (
        patch(
            "transcribe_engine.signaling.client.fetch_signaling_token",
            side_effect=_good_fetch,
        ),
        patch(
            "transcribe_engine.signaling.client.AsyncRealtimeClient",
            return_value=fake_rt,
        ),
    ):
        task = asyncio.create_task(client.run())
        # Wait for subscribe
        await asyncio.sleep(0.05)

        channel = fake_rt.channels.get("pair:user-abc")
        assert channel is not None

        # Inject an inbound offer broadcast
        offer_payload = {
            "type": "description",
            "sdp": {"type": "offer", "sdp": "v=0\r\no=..."},
        }
        channel.inject_broadcast("description", offer_payload)
        await asyncio.sleep(0.05)

        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, UnpairedExit):
            pass

    on_offer.assert_called_once()


@pytest.mark.asyncio
async def test_emit_state_sends_state_broadcast():
    """emit_state() sends a StateMsg broadcast on the channel."""
    state = _make_state()
    sk = _make_signing_key()
    fake_rt = FakeRealtimeClient("https://fake.supabase.co", "fake.jwt")

    client = EngineSignalingClient(
        state=state,
        base_url="https://fake.vercel.app",
        signing_key=sk,
        on_state_change=MagicMock(),
        on_offer=AsyncMock(),
        on_candidate=AsyncMock(),
        on_resume_query=AsyncMock(),
        gpu_label="TestGPU",
    )

    with (
        patch(
            "transcribe_engine.signaling.client.fetch_signaling_token",
            side_effect=_good_fetch,
        ),
        patch(
            "transcribe_engine.signaling.client.AsyncRealtimeClient",
            return_value=fake_rt,
        ),
    ):
        task = asyncio.create_task(client.run())
        await asyncio.sleep(0.05)

        await client.emit_state("transcribing")
        await asyncio.sleep(0.02)

        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, UnpairedExit):
            pass

    channel = fake_rt.channels.get("pair:user-abc")
    state_broadcasts = [b for b in channel.broadcasts_sent if b[0] == "state"]
    assert state_broadcasts, "No state broadcast sent"
    _, payload = state_broadcasts[0]
    assert payload.get("type") == "state"
    assert payload.get("value") == "transcribing"


@pytest.mark.asyncio
async def test_client_reconnects_after_socket_error_with_backoff():
    """After a connection error, the client retries with increasing delays."""
    state = _make_state()
    sk = _make_signing_key()

    connect_times: list[float] = []
    connect_attempt = 0

    class _ErrorThenConnectRealtime(FakeRealtimeClient):
        async def connect(self) -> None:
            nonlocal connect_attempt
            import time

            connect_times.append(time.monotonic())
            connect_attempt += 1
            if connect_attempt <= 2:
                raise ConnectionError("fake socket error")
            await super().connect()

    fake_rt_factory_calls = []

    def fake_rt_factory(url, token, **kwargs):
        rt = _ErrorThenConnectRealtime(url, token)
        fake_rt_factory_calls.append(rt)
        return rt

    client = EngineSignalingClient(
        state=state,
        base_url="https://fake.vercel.app",
        signing_key=sk,
        on_state_change=MagicMock(),
        on_offer=AsyncMock(),
        on_candidate=AsyncMock(),
        on_resume_query=AsyncMock(),
        gpu_label="TestGPU",
    )

    # Patch reconnect_delays to return tiny delays for the test
    async def _fast_fetch(base_url, signing_key, **kwargs):
        return _fake_token()

    with (
        patch(
            "transcribe_engine.signaling.client.fetch_signaling_token",
            side_effect=_fast_fetch,
        ),
        patch(
            "transcribe_engine.signaling.client.AsyncRealtimeClient",
            side_effect=fake_rt_factory,
        ),
        patch(
            "transcribe_engine.signaling.client.reconnect_delays",
            return_value=iter([0.01, 0.02, 0.04]),
        ),
    ):
        task = asyncio.create_task(client.run())
        await asyncio.sleep(0.3)
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, UnpairedExit):
            pass

    # Should have attempted at least 2 connections (errors) + 1 successful
    assert connect_attempt >= 2, f"Expected >= 2 connect calls, got {connect_attempt}"
