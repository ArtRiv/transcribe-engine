"""state.json — local source of truth for paired-or-not (D-11).

Phase 7 ships the schema with reservation slots (paired_user_id, pubkey,
keypair_path, last_signaling_url) all None. Phase 8 (Plan 05) bumps to v2:
- Replaces `last_signaling_url` with a `Signaling` subdict
  (supabase_url + channel), matching the Supabase Realtime pivot (D-17).
- Adds `protocol_version` for BYO-5 wire-format negotiation.
- Adds lazy v1→v2 migration in `State.load()` per the Phase 7 D-11 hook.

v0.1.0 (Phase 7) installs upgrade silently on first boot after v0.2.0.
"""
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

SCHEMA_VERSION = 2


@dataclass
class Signaling:
    """Supabase Realtime signaling channel config (Phase 8 D-17 pivot)."""

    supabase_url: str | None = None
    channel: str | None = None  # 'pair:<user_id>'


@dataclass
class State:
    schema_version: int = SCHEMA_VERSION
    paired_user_id: str | None = None       # Phase 8 fills
    pubkey: str | None = None               # Phase 8 fills
    keypair_path: str | None = None         # Phase 8 fills
    signaling: Signaling = field(default_factory=Signaling)
    installed_models: list[str] = field(default_factory=list)
    protocol_version: str = "1"                # BYO-5 wire-format negotiation

    @classmethod
    def load(cls, path: Path) -> "State":
        if not path.exists():
            return cls()
        with open(path) as f:
            data = json.load(f)

        version = data.get("schema_version", 1)

        if version > SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema version {version} "
                f"(this engine supports up to v{SCHEMA_VERSION}); "
                "upgrade transcribe-engine to read this state.json"
            )

        if version < 2:
            # v1 → v2 migration:
            # - Drop last_signaling_url (pointed to Vercel; v2 uses Supabase Realtime)
            # - Add signaling subdict (both fields None)
            # - Add protocol_version
            data.pop("last_signaling_url", None)
            data["schema_version"] = 2
            data.setdefault("signaling", {"supabase_url": None, "channel": None})
            data.setdefault("protocol_version", "1")

        # Convert signaling dict → Signaling dataclass
        signaling_raw = data.pop("signaling", {})
        if isinstance(signaling_raw, dict):
            signaling = Signaling(**signaling_raw)
        else:
            signaling = signaling_raw  # already a Signaling instance

        # WR-06: filter data to only known fields to avoid TypeError on
        # unexpected keys (hand-edited files, downgrade from a newer version).
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in data.items() if k in known}

        return cls(signaling=signaling, **data)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    def has_models(self) -> bool:
        return bool(self.installed_models)


def migrate_in_place(path: Path) -> None:
    """Load `path`, upgrade schema if needed, save back to disk, and return.

    Called by `__main__.py` at boot to silently upgrade Phase 7 v0.1.0 installs
    from schema_version=1 to schema_version=2 before any other code runs.
    No-op if the file does not exist or is already at SCHEMA_VERSION.
    """
    if not path.exists():
        return
    with open(path) as f:
        data = json.load(f)
    if data.get("schema_version", 1) >= SCHEMA_VERSION:
        return  # already at current version — no-op (idempotent)
    State.load(path).save(path)
