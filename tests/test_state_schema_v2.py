"""Tests for state.py v2 schema — lazy v1→v2 migration, Signaling subdict,
protocol_version field.

Phase 8 Plan 05 bumps SCHEMA_VERSION from 1 → 2, adds a `Signaling` dataclass
(supabase_url, channel), drops `last_signaling_url`, and adds `protocol_version`.
Migration runs lazily in `State.load()` — existing Phase 7 v1 installs upgrade
silently on first boot.
"""
import json
from pathlib import Path

import pytest

from transcribe_engine.state import SCHEMA_VERSION, Signaling, State, migrate_in_place


def test_schema_version_is_2() -> None:
    """SCHEMA_VERSION constant must equal 2 after the bump."""
    assert SCHEMA_VERSION == 2


def test_load_v1_file_returns_v2_object(tmp_path: Path) -> None:
    """Loading a v1 state.json (with last_signaling_url) returns a fully-migrated v2 State."""
    v1_path = tmp_path / "state.json"
    v1_data = {
        "schema_version": 1,
        "paired_user_id": "abc",
        "pubkey": "deadbeef",
        "keypair_path": "/home/user/.config/keypair.bin",
        "last_signaling_url": "https://old.vercel.app",
        "installed_models": ["fast"],
    }
    v1_path.write_text(json.dumps(v1_data))

    state = State.load(v1_path)

    assert state.schema_version == 2
    assert state.paired_user_id == "abc"
    assert state.pubkey == "deadbeef"
    assert state.keypair_path == "/home/user/.config/keypair.bin"
    assert state.installed_models == ["fast"]
    assert state.signaling.supabase_url is None
    assert state.signaling.channel is None
    assert state.protocol_version == "1"
    # last_signaling_url must NOT be an attribute on v2 State
    assert not hasattr(state, "last_signaling_url")


def test_load_v1_file_without_last_signaling_url(tmp_path: Path) -> None:
    """v1 file without last_signaling_url (it was optional) also migrates cleanly."""
    v1_path = tmp_path / "state.json"
    v1_data = {
        "schema_version": 1,
        "paired_user_id": None,
        "pubkey": None,
        "keypair_path": None,
        "installed_models": [],
    }
    v1_path.write_text(json.dumps(v1_data))

    state = State.load(v1_path)

    assert state.schema_version == 2
    assert state.signaling.supabase_url is None
    assert state.protocol_version == "1"


def test_save_emits_v2_format(tmp_path: Path) -> None:
    """Saving a v2 State writes the new fields and does NOT include last_signaling_url."""
    path = tmp_path / "state.json"
    state = State(
        paired_user_id="x",
        signaling=Signaling(supabase_url="https://s.co", channel="pair:x"),
    )
    state.save(path)

    raw = json.loads(path.read_text())
    assert raw["schema_version"] == 2
    assert "last_signaling_url" not in raw
    assert raw["signaling"]["supabase_url"] == "https://s.co"
    assert raw["signaling"]["channel"] == "pair:x"
    assert raw["protocol_version"] == "1"


def test_round_trip_v2(tmp_path: Path) -> None:
    """Save then load preserves all v2 fields verbatim."""
    path = tmp_path / "state.json"
    original = State(
        paired_user_id="user-42",
        pubkey="abc123",
        keypair_path="/etc/keypair.bin",
        signaling=Signaling(supabase_url="https://xyz.supabase.co", channel="pair:user-42"),
        installed_models=["medium", "large"],
        protocol_version="1",
    )
    original.save(path)
    loaded = State.load(path)
    assert loaded == original


def test_load_unknown_future_version_raises_clear_error(tmp_path: Path) -> None:
    """Loading a state.json with schema_version=99 raises ValueError with a clear message."""
    future_path = tmp_path / "state.json"
    future_path.write_text(json.dumps({"schema_version": 99}))

    with pytest.raises(ValueError, match="unsupported schema version"):
        State.load(future_path)


def test_old_v1_fields_carry_forward() -> None:
    """Verify that paired_user_id, pubkey, keypair_path, installed_models
    are STILL accessible on the v2 State (field names did NOT change).
    """
    state = State(
        paired_user_id="u1",
        pubkey="pk",
        keypair_path="/tmp/k.bin",
        installed_models=["fast"],
    )
    assert state.paired_user_id == "u1"
    assert state.pubkey == "pk"
    assert state.keypair_path == "/tmp/k.bin"
    assert state.installed_models == ["fast"]


def test_migrate_in_place_upgrades_v1_file(tmp_path: Path) -> None:
    """migrate_in_place(path) loads v1, saves v2, returns — on-disk file is now v2."""
    v1_path = tmp_path / "state.json"
    v1_data = {
        "schema_version": 1,
        "paired_user_id": "usr",
        "pubkey": None,
        "keypair_path": None,
        "last_signaling_url": "https://example.vercel.app",
        "installed_models": [],
    }
    v1_path.write_text(json.dumps(v1_data))

    migrate_in_place(v1_path)

    on_disk = json.loads(v1_path.read_text())
    assert on_disk["schema_version"] == 2
    assert "last_signaling_url" not in on_disk
    assert "signaling" in on_disk
    assert on_disk["protocol_version"] == "1"


def test_default_state_is_v2(tmp_path: Path) -> None:
    """A brand-new State() has schema_version=2 and proper signaling defaults."""
    state = State()
    assert state.schema_version == 2
    assert state.signaling.supabase_url is None
    assert state.signaling.channel is None
    assert state.protocol_version == "1"


def test_load_missing_file_returns_default_v2_state(tmp_path: Path) -> None:
    """State.load on a missing file returns a default v2 State (not v1)."""
    result = State.load(tmp_path / "nonexistent.json")
    assert result.schema_version == 2
    assert isinstance(result.signaling, Signaling)
    assert result.protocol_version == "1"


def test_load_ignores_unknown_fields_in_v2_file(tmp_path: Path) -> None:
    """WR-06: State.load silently ignores unknown keys rather than crashing."""
    v2_path = tmp_path / "state.json"
    data = {
        "schema_version": 2,
        "paired_user_id": "u1",
        "pubkey": None,
        "keypair_path": None,
        "signaling": {"supabase_url": None, "channel": None},
        "installed_models": [],
        "protocol_version": "1",
        "deprecated_field": "some_value",   # unknown field
        "future_key": 42,                   # unknown field
    }
    v2_path.write_text(json.dumps(data))
    state = State.load(v2_path)  # must not raise TypeError
    assert state.paired_user_id == "u1"
    assert not hasattr(state, "deprecated_field")
    assert not hasattr(state, "future_key")


def test_migrate_in_place_is_noop_for_v2_file(tmp_path: Path) -> None:
    """WR-05: migrate_in_place does NOT rewrite a file already at SCHEMA_VERSION."""
    v2_path = tmp_path / "state.json"
    state = State(
        paired_user_id="user-99",
        signaling=Signaling(supabase_url="https://s.co", channel="pair:user-99"),
    )
    state.save(v2_path)
    original_mtime = v2_path.stat().st_mtime

    migrate_in_place(v2_path)

    # File must not have been touched (mtime unchanged)
    assert v2_path.stat().st_mtime == original_mtime
