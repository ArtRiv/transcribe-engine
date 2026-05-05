"""webrtc/peer.py — aiortc RTCPeerConnection wrapper (Plan 08-07).

aiortc is imported LAZILY inside create_peer_connection() to keep engine
startup fast — aiortc loads pylibsrtp + av on import (~500 ms per Plan 01
spike measurements).  Callers should not import aiortc directly.

Responsibilities of this module:
    create_peer_connection — build RTCPeerConnection with ICE config
    accept_offer           — impolite-peer answer exchange
    add_remote_candidate   — addIceCandidate wrapper
    close                  — clean shutdown

NOT responsible for pipeline orchestration — that lives in handler.py.

SEC-08: no supabase import; no banned HTTP frameworks.
T-08-07-04: DTLS fingerprint exchange happens inside aiortc SDP; this
module does not need to handle it manually.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# ICE server defaults (overridden by turn_creds when available)
_STUN_URL = "stun:stun.l.google.com:19302"


async def create_peer_connection(
    turn_creds: dict | None = None,
):
    """Build and return an RTCPeerConnection.

    Args:
        turn_creds: Optional dict with keys ``urls``, ``username``,
                    ``credential`` from the /api/turn-credentials route.
                    When None, STUN-only is used (direct path / same LAN).

    Returns:
        aiortc.RTCPeerConnection

    Lazy import: aiortc is imported here, not at module top-level, to
    avoid the ~500 ms pylibsrtp + av import cost at engine startup.
    """
    from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection  # lazy

    ice_servers = [RTCIceServer(urls=_STUN_URL)]

    if turn_creds:
        ice_servers.append(
            RTCIceServer(
                urls=turn_creds["urls"],
                username=turn_creds.get("username"),
                credential=turn_creds.get("credential"),
            )
        )
        log.debug("create_peer_connection: STUN + TURN (%s)", turn_creds["urls"])
    else:
        log.debug("create_peer_connection: STUN-only")

    return RTCPeerConnection(RTCConfiguration(iceServers=ice_servers))


async def accept_offer(pc, offer_sdp: dict) -> dict:
    """Impolite-peer side: set remote description (offer), create and set answer.

    Args:
        pc:        RTCPeerConnection (from create_peer_connection)
        offer_sdp: Dict with keys ``type`` == 'offer' and ``sdp`` (SDP string).

    Returns:
        Answer SDP dict with keys ``type`` == 'answer' and ``sdp`` (SDP string).
    """
    from aiortc import RTCSessionDescription  # lazy

    remote_desc = RTCSessionDescription(
        sdp=offer_sdp["sdp"],
        type=offer_sdp["type"],
    )
    await pc.setRemoteDescription(remote_desc)

    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    log.debug("accept_offer: answer SDP generated")
    return {
        "type": pc.localDescription.type,
        "sdp": pc.localDescription.sdp,
    }


async def add_remote_candidate(pc, candidate: dict) -> None:
    """Add a remote ICE candidate received via the signaling channel.

    Args:
        pc:        RTCPeerConnection
        candidate: Dict with key ``candidate`` (ICE candidate string) and
                   optional ``sdpMid``, ``sdpMLineIndex``.

    WR-01 fix: parse the SDP candidate string via candidate_from_sdp() so all
    fields (ip, port, foundation, priority, …) are populated from the actual
    wire value.  Constructing RTCIceCandidate with placeholder zeros/empty
    strings produces a candidate that points nowhere and silently fails ICE.
    """
    # aiortc's candidate_from_sdp parses the SDP candidate line into a
    # fully-populated RTCIceCandidate dataclass.
    # The dict shape matches CandidateMsg.candidate from protocol/messages.py.
    raw = candidate.get("candidate", "")
    if not raw:
        # End-of-candidates signal — safe to ignore in aiortc.
        return

    try:
        from aiortc.sdp import candidate_from_sdp  # lazy; stable aiortc internal
        ice = candidate_from_sdp(raw)
        ice.sdpMid = candidate.get("sdpMid")
        ice.sdpMLineIndex = candidate.get("sdpMLineIndex")
    except Exception as exc:
        log.warning(
            "add_remote_candidate: malformed candidate SDP %r — dropped (%s)",
            raw[:80], exc,
        )
        return

    await pc.addIceCandidate(ice)
    log.debug("add_remote_candidate: added %s", raw[:60])


async def close(pc) -> None:
    """Clean shutdown — close the RTCPeerConnection and release aiortc resources."""
    try:
        await pc.close()
        log.debug("close: RTCPeerConnection closed")
    except Exception as exc:
        log.debug("close: pc.close() raised (non-fatal): %s", exc)
