"""WebRTC peer connection + chunked audio transport (Plan 08-07).

Modules:
    chunker — PartialFileWriter: append-write + checkpoint emission + atomic rename
    peer    — aiortc RTCPeerConnection wrapper (lazy-imported for startup speed)
    handler — InboundJobHandler: audio bytes → .partial file → pipeline kickoff
"""
