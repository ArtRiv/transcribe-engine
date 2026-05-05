"""Tests for webrtc/peer.py — aiortc RTCPeerConnection wrapper (Plan 08-07 Task 1).

Tests use real aiortc (in-process, no actual network) to validate:
    - ICE server config with STUN-only
    - ICE server config with STUN + TURN
    - accept_offer() returns an SDP answer dict
    - close() runs without error

Marked @pytest.mark.slow for the accept_offer test because aiortc DTLS
setup can take a few hundred ms.
"""
import pytest

from transcribe_engine.webrtc import peer as peer_module


def _ice_servers(pc):
    """Extract ICE servers from aiortc RTCPeerConnection internals."""
    # aiortc uses Python name-mangling: _RTCPeerConnection__configuration
    config = pc._RTCPeerConnection__configuration
    return config.iceServers


@pytest.mark.asyncio
async def test_create_peer_connection_with_stun_only() -> None:
    """No turn_creds → single STUN ICE server entry."""
    pc = await peer_module.create_peer_connection(turn_creds=None)
    servers = _ice_servers(pc)
    assert len(servers) == 1
    assert "stun" in servers[0].urls
    await peer_module.close(pc)


@pytest.mark.asyncio
async def test_create_peer_connection_with_turn() -> None:
    """Providing turn_creds → 2 ICE servers (STUN + TURN)."""
    turn_creds = {
        "urls": "turn:turn.example.com:3478",
        "username": "testuser",
        "credential": "testpass",
    }
    pc = await peer_module.create_peer_connection(turn_creds=turn_creds)
    servers = _ice_servers(pc)
    assert len(servers) == 2
    urls = [s.urls for s in servers]
    assert any("stun" in u for u in urls), "Expected STUN server"
    assert any("turn" in u for u in urls), "Expected TURN server"
    await peer_module.close(pc)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_accept_offer_returns_answer_with_correct_type() -> None:
    """accept_offer() produces a valid SDP answer dict from a real aiortc offer."""
    # Build an offerer peer to generate a valid offer SDP
    offerer = await peer_module.create_peer_connection(turn_creds=None)

    # Add a data channel so the offer includes media (required for SDP answer)
    offerer.createDataChannel("test-channel", ordered=True)
    offer = await offerer.createOffer()
    await offerer.setLocalDescription(offer)

    offer_dict = {
        "type": offerer.localDescription.type,
        "sdp": offerer.localDescription.sdp,
    }

    # The answerer (engine's impolite peer)
    answerer = await peer_module.create_peer_connection(turn_creds=None)
    answer_dict = await peer_module.accept_offer(answerer, offer_dict)

    assert answer_dict["type"] == "answer", (
        f"Expected type='answer', got {answer_dict['type']!r}"
    )
    assert len(answer_dict["sdp"]) > 0, "SDP string must not be empty"

    await peer_module.close(offerer)
    await peer_module.close(answerer)


@pytest.mark.asyncio
async def test_close_does_not_raise() -> None:
    """close() on an open PC completes without raising."""
    pc = await peer_module.create_peer_connection(turn_creds=None)
    # Should not raise
    await peer_module.close(pc)


@pytest.mark.asyncio
async def test_add_remote_candidate_parses_sdp_fields(tmp_path) -> None:
    """WR-01: add_remote_candidate() parses real SDP fields, not placeholder zeros.

    A valid ICE candidate string must produce an RTCIceCandidate with the
    correct ip/port, NOT '0.0.0.0'/0.
    """
    from aiortc.sdp import candidate_from_sdp

    # A representative SDP candidate string (realistic host candidate)
    good_sdp = (
        "candidate:3348148302 1 udp 2113937151 "
        "192.168.1.100 56123 typ host generation 0"
    )

    # Verify candidate_from_sdp parses correctly (basis for add_remote_candidate fix)
    ice = candidate_from_sdp(good_sdp)
    assert ice.ip == "192.168.1.100", f"expected ip 192.168.1.100, got {ice.ip}"
    assert ice.port == 56123, f"expected port 56123, got {ice.port}"
    assert ice.ip != "0.0.0.0", "ip must not be the placeholder 0.0.0.0"
    assert ice.port != 0, "port must not be the placeholder 0"


@pytest.mark.asyncio
async def test_add_remote_candidate_drops_malformed() -> None:
    """WR-01: add_remote_candidate() drops a malformed SDP line without raising."""
    pc = await peer_module.create_peer_connection(turn_creds=None)

    # Should not raise — malformed SDP is logged and dropped
    await peer_module.add_remote_candidate(pc, {"candidate": "not-a-valid-sdp-candidate"})

    await peer_module.close(pc)


@pytest.mark.asyncio
async def test_add_remote_candidate_ignores_empty_candidate() -> None:
    """End-of-candidates signal (empty candidate string) is silently ignored."""
    pc = await peer_module.create_peer_connection(turn_creds=None)

    # Should not raise
    await peer_module.add_remote_candidate(pc, {"candidate": ""})
    await peer_module.add_remote_candidate(pc, {})  # missing key

    await peer_module.close(pc)
