"""Tests for state.py — Phase 8 v2 State dataclass.

Updated in Plan 08-05 to match the v2 schema: `last_signaling_url` is replaced
by `signaling: Signaling` + `protocol_version`. The v1 carry-forward fields
(paired_user_id, pubkey, keypair_path, installed_models) are unchanged.
"""

from pathlib import Path

from transcribe_engine.state import SCHEMA_VERSION, Signaling, State


def test_load_missing_file_returns_default_state(tmp_path: Path) -> None:
    result = State.load(tmp_path / "nonexistent.json")
    assert result.schema_version == SCHEMA_VERSION
    assert result.paired_user_id is None
    assert result.pubkey is None
    assert result.keypair_path is None
    assert isinstance(result.signaling, Signaling)
    assert result.signaling.supabase_url is None
    assert result.signaling.channel is None
    assert result.installed_models == []


def test_round_trip_preserves_all_fields(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    original = State(
        paired_user_id="user-123",
        pubkey="ed25519-pub",
        keypair_path="/home/user/.config/transcribe-engine/keypair.pem",
        signaling=Signaling(supabase_url="https://realtime.supabase.co", channel="pair:user-123"),
        installed_models=["fast", "average"],
    )
    original.save(path)
    loaded = State.load(path)
    assert loaded == original


def test_has_models_false_when_empty() -> None:
    assert State().has_models() is False


def test_has_models_true_when_populated() -> None:
    s = State(installed_models=["fast"])
    assert s.has_models() is True


def test_save_writes_human_readable_indented_json(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    State().save(path)
    content = path.read_text()
    assert "\n" in content  # multi-line (indent applied)
    assert '"schema_version": 2' in content
