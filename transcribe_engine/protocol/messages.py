"""Wire-format TypedDicts for the WebRTC data channel (Plan 08-05).

Defines the 14 message TypedDicts (13 wire-type strings — OfferMsg + AnswerMsg
share the 'description' type discriminator and are told apart by sdp.type).

Plans 06 and 07 import from this module:
  Plan 06 (signaling client): HelloMsg, StateMsg
  Plan 07 (WebRTC handler):   OfferMsg, AnswerMsg, CandidateMsg, AudioEofMsg,
                               CheckpointMsg, ProgressMsg, ResultMsg

The frontend mirror is frontend/lib/webrtc/protocol.ts — both files export an
identical KNOWN_MESSAGE_TYPES list. A cross-repo set-equality test in
frontend/tests/lib/webrtc/protocol.test.ts catches drift (T-08-05-02).
"""
import json
from typing import Literal, Optional, Union

from typing_extensions import TypedDict

# Wire protocol version — bumped when the message shape changes in a
# backwards-incompatible way. Both engine and frontend must agree on this.
PROTOCOL_VERSION: str = "1"

# ---------------------------------------------------------------------------
# Nested payload types
# ---------------------------------------------------------------------------


class SdpDescription(TypedDict):
    type: Literal["offer", "answer"]
    sdp: str


class IceCandidate(TypedDict, total=False):
    candidate: str
    sdpMid: Optional[str]
    sdpMLineIndex: Optional[int]


class TranscriptWord(TypedDict, total=False):
    w: str       # word text
    s: float     # start seconds
    e: float     # end seconds
    p: float     # probability (optional)


class TranscriptSegment(TypedDict, total=False):
    id: str      # "seg_NNNN" zero-padded
    start: float
    end: float
    speaker: str  # references Speaker.id
    text: str
    words: list[TranscriptWord]


class TranscriptSpeaker(TypedDict):
    id: str     # e.g. "S0", "S1"
    label: str  # human-renamed; "Speaker 1" default


class TranscriptPayload(TypedDict):
    """v1 transcript schema — matches frontend/lib/mock/data.ts exactly (D-22)."""
    version: Literal[1]
    language: str           # BCP-47 short, e.g. "en"
    duration_sec: float
    speakers: list[TranscriptSpeaker]
    segments: list[TranscriptSegment]


# ---------------------------------------------------------------------------
# Message types (14 TypedDicts, 13 wire-type strings)
# ---------------------------------------------------------------------------


class HelloMsg(TypedDict):
    type: Literal["hello"]
    version: str            # engine semver e.g. "0.2.0"
    protocol_version: str   # wire protocol version e.g. "1"
    gpu: str                # GPU label e.g. "RX 6600 (Vulkan)"


class StateMsg(TypedDict):
    type: Literal["state"]
    value: Literal["idle", "transcribing", "loading_model", "gpu_warming", "offline"]


class OfferMsg(TypedDict):
    """SDP offer — disambiguated from AnswerMsg by sdp.type == 'offer'."""
    type: Literal["description"]
    sdp: SdpDescription


class AnswerMsg(TypedDict):
    """SDP answer — disambiguated from OfferMsg by sdp.type == 'answer'."""
    type: Literal["description"]
    sdp: SdpDescription


class CandidateMsg(TypedDict):
    type: Literal["candidate"]
    candidate: IceCandidate


class AudioEofMsg(TypedDict):
    type: Literal["audio_eof"]
    total_bytes: int


class CheckpointMsg(TypedDict):
    type: Literal["checkpoint"]
    byte_offset: int


class ProgressMsg(TypedDict):
    type: Literal["progress"]
    stage: Literal["normalize", "transcribe", "diarize", "merge"]
    fraction: float


class ResultMsg(TypedDict):
    type: Literal["result"]
    transcript: TranscriptPayload


class ResumeQueryMsg(TypedDict):
    type: Literal["resume_query"]
    job_id: str


class ResumeStateMsg(TypedDict):
    type: Literal["resume_state"]
    byte_offset: int


class PingMsg(TypedDict):
    type: Literal["ping"]


class PongMsg(TypedDict):
    type: Literal["pong"]


class ErrorMsg(TypedDict):
    type: Literal["error"]
    code: str
    message: str


# ---------------------------------------------------------------------------
# Discriminating union
# ---------------------------------------------------------------------------

WireMessage = Union[
    HelloMsg,
    StateMsg,
    OfferMsg,      # AnswerMsg is structurally identical; differentiate by sdp.type
    CandidateMsg,
    AudioEofMsg,
    CheckpointMsg,
    ProgressMsg,
    ResultMsg,
    ResumeQueryMsg,
    ResumeStateMsg,
    PingMsg,
    PongMsg,
    ErrorMsg,
]

# 13 distinct `type` discriminator strings; OfferMsg and AnswerMsg share `type: 'description'`
# and are disambiguated by `sdp.type` ('offer' vs 'answer'). The 14 message TypedDicts
# collapse to 13 wire-format type strings — the test_known_message_types_count assertion
# below pins this exact count.
KNOWN_MESSAGE_TYPES: list[str] = [
    "hello",
    "state",
    "description",
    "candidate",
    "audio_eof",
    "checkpoint",
    "progress",
    "result",
    "resume_query",
    "resume_state",
    "ping",
    "pong",
    "error",
]


# ---------------------------------------------------------------------------
# parse_wire
# ---------------------------------------------------------------------------


def parse_wire(raw: str | bytes) -> WireMessage:
    """Decode a JSON wire message and validate its `type` discriminator.

    Raises ValueError with the offending type value on unknown types, or when
    `protocol_version` on a 'hello' message does not match PROTOCOL_VERSION.
    Plans 06 and 07 wrap this with try/except + ErrorMsg emission.
    """
    text = raw if isinstance(raw, str) else raw.decode()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Wire message must be a JSON object, got {type(data).__name__}")
    if "type" not in data:
        raise ValueError("Wire message missing 'type' field")
    msg_type = data["type"]
    if msg_type not in KNOWN_MESSAGE_TYPES:
        raise ValueError(
            f"Unknown wire message type {msg_type!r}. "
            f"Known types: {KNOWN_MESSAGE_TYPES}"
        )
    if msg_type == "hello":
        pv = data.get("protocol_version")
        if pv != PROTOCOL_VERSION:
            raise ValueError(
                f"Protocol version mismatch: peer={pv!r} local={PROTOCOL_VERSION!r}"
            )
    return data  # type: ignore[return-value]  # structural TypedDict — runtime check via 'type'
