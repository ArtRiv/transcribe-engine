"""Tests for transcribe_engine.protocol.messages — wire-format TypedDicts.

Verifies:
- PROTOCOL_VERSION constant is "1"
- KNOWN_MESSAGE_TYPES has exactly 14 entries (CR-02 added job_init; OfferMsg + AnswerMsg
  share 'description')
- parse_wire round-trips each type
- parse_wire rejects unknown types
- Module uses only stdlib (json + typing)
"""

import json

import pytest

from transcribe_engine.protocol.messages import (
    KNOWN_MESSAGE_TYPES,
    PROTOCOL_VERSION,
    parse_wire,
)

# ---------------------------------------------------------------------------
# Basic constants
# ---------------------------------------------------------------------------


def test_protocol_version_is_1() -> None:
    """Wire version is pinned at '1' for v0.2.0."""
    assert PROTOCOL_VERSION == "1"


def test_known_message_types_count() -> None:
    """Exactly 14 wire-type strings (CR-02 added job_init; OfferMsg + AnswerMsg share 'description')."""
    assert len(KNOWN_MESSAGE_TYPES) == 14, (
        f"Expected 14 type strings, got {len(KNOWN_MESSAGE_TYPES)}: {KNOWN_MESSAGE_TYPES}"
    )


def test_known_message_types_contains_expected_values() -> None:
    expected = {
        "hello",
        "state",
        "description",
        "candidate",
        "job_init",
        "audio_eof",
        "checkpoint",
        "progress",
        "result",
        "resume_query",
        "resume_state",
        "ping",
        "pong",
        "error",
    }
    assert set(KNOWN_MESSAGE_TYPES) == expected


# ---------------------------------------------------------------------------
# parse_wire round-trips
# ---------------------------------------------------------------------------

SYNTHETIC_PAYLOADS = [
    {"type": "hello", "version": "0.2.0", "protocol_version": "1", "gpu": "RX 6600 (Vulkan)"},
    {"type": "state", "value": "idle"},
    {
        "type": "description",
        "sdp": {"type": "offer", "sdp": "v=0\r\no=..."},
    },
    {
        "type": "candidate",
        "candidate": {"candidate": "candidate:0 ...", "sdpMid": "0", "sdpMLineIndex": 0},
    },
    {"type": "audio_eof", "total_bytes": 1048576},
    {"type": "checkpoint", "byte_offset": 524288},
    {"type": "progress", "stage": "transcribe", "fraction": 0.42},
    {
        "type": "result",
        "transcript": {
            "version": 1,
            "language": "en",
            "duration_sec": 60.0,
            "speakers": [{"id": "S0", "label": "Speaker 1"}],
            "segments": [
                {
                    "id": "seg_0001",
                    "start": 0.0,
                    "end": 5.0,
                    "speaker": "S0",
                    "text": "Hello.",
                    "words": [{"w": "Hello.", "s": 0.0, "e": 0.5}],
                }
            ],
        },
    },
    {
        "type": "job_init",
        "job_id": "user-abc123def456ab",
        "sha256_hex": "a" * 64,
        "total_bytes": 1048576,
    },
    {
        "type": "resume_query",
        "job_id": "user-abc123def456ab",
        "sha256_hex": "a" * 64,
    },
    {"type": "resume_state", "byte_offset": 262144},
    {"type": "ping"},
    {"type": "pong"},
    {"type": "error", "code": "no_model", "message": "Model not loaded"},
]


@pytest.mark.parametrize("payload", SYNTHETIC_PAYLOADS, ids=lambda p: p["type"])
def test_parse_wire_round_trip_each_type(payload: dict) -> None:
    """parse_wire returns the original dict unchanged for each valid type."""
    raw = json.dumps(payload)
    result = parse_wire(raw)
    assert result == payload


def test_parse_wire_accepts_bytes_input() -> None:
    """parse_wire also accepts bytes input (websocket frames are bytes)."""
    msg = {"type": "ping"}
    result = parse_wire(json.dumps(msg).encode())
    assert result == msg


def test_parse_wire_rejects_unknown_type() -> None:
    """parse_wire raises ValueError with the offending type for unknown types."""
    with pytest.raises(ValueError, match="bogus"):
        parse_wire('{"type":"bogus"}')


def test_parse_wire_rejects_missing_type_key() -> None:
    """parse_wire raises ValueError when type key is absent."""
    with pytest.raises(ValueError, match="missing 'type'"):
        parse_wire('{"data": "no type key"}')


def test_parse_wire_rejects_non_object_json() -> None:
    """parse_wire raises ValueError on non-object JSON (null, list, string, number)."""
    with pytest.raises(ValueError, match="JSON object"):
        parse_wire("null")
    with pytest.raises(ValueError, match="JSON object"):
        parse_wire("[1, 2, 3]")
    with pytest.raises(ValueError, match="JSON object"):
        parse_wire('"a string"')


def test_parse_wire_rejects_wrong_protocol_version() -> None:
    """parse_wire raises ValueError when hello msg carries wrong protocol_version."""
    msg = {"type": "hello", "version": "0.2.0", "protocol_version": "2", "gpu": "RX 6600"}
    with pytest.raises(ValueError, match="Protocol version mismatch"):
        parse_wire(json.dumps(msg))


def test_parse_wire_rejects_missing_protocol_version_in_hello() -> None:
    """parse_wire raises ValueError when hello msg is missing protocol_version."""
    msg = {"type": "hello", "version": "0.2.0", "gpu": "RX 6600"}
    with pytest.raises(ValueError, match="Protocol version mismatch"):
        parse_wire(json.dumps(msg))


# ---------------------------------------------------------------------------
# Import hygiene sentinel (T-08-05-04)
# ---------------------------------------------------------------------------


def test_no_banned_imports() -> None:
    """protocol.messages uses only json + typing — no httpx, requests, etc."""
    import ast
    import pathlib

    src = pathlib.Path(__file__).parent.parent / "transcribe_engine" / "protocol" / "messages.py"
    tree = ast.parse(src.read_text())
    banned = {"httpx", "requests", "aiohttp", "supabase", "fastapi", "flask", "starlette"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in banned, f"messages.py imports banned module {alias.name!r}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            assert root not in banned, f"messages.py imports from banned module {node.module!r}"
