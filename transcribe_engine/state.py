"""state.json — local source of truth for paired-or-not (D-11).

Phase 7 ships the schema with reservation slots (paired_user_id, pubkey,
keypair_path, last_signaling_url) all None. Phase 8 fills them when the
user pairs the engine with their account.
"""
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

SCHEMA_VERSION = 1


@dataclass
class State:
    schema_version: int = SCHEMA_VERSION
    paired_user_id: Optional[str] = None       # Phase 8 fills
    pubkey: Optional[str] = None               # Phase 8 fills
    keypair_path: Optional[str] = None         # Phase 8 fills
    last_signaling_url: Optional[str] = None   # Phase 8 fills
    installed_models: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "State":
        if not path.exists():
            return cls()
        with open(path) as f:
            data = json.load(f)
        # Schema migration hook: if data.get("schema_version", 0) < SCHEMA_VERSION, migrate.
        # v0.1.0 ships schema_version=1; no migrations yet.
        return cls(**data)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    def has_models(self) -> bool:
        return bool(self.installed_models)
