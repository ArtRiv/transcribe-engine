"""models.toml registry reader (D-22).

Stdlib tomllib only — no third-party TOML lib needed (Python 3.11+).
T-7-pickle-01 mitigation: stdlib tomllib is read-only; no eval/pickle on registry data.
models.toml is shipped inside the read-only bundle root (R-04 — _MEIPASS).
"""
import tomllib
from pathlib import Path
from typing import Any

from transcribe_engine.platform.paths import bundle_root


def _models_toml_path() -> Path:
    """Locate models.toml relative to bundle root (R-04 path discipline)."""
    return bundle_root() / "models.toml"


def load_models() -> dict[str, Any]:
    """Read models.toml shipped inside the bundle. Returns parsed TOML data.

    Raises FileNotFoundError if models.toml is missing (bundling failure).
    """
    with open(_models_toml_path(), "rb") as f:
        return tomllib.load(f)


def get_tier(tier: str) -> dict[str, Any]:
    """Return single whisper.<tier> entry (raises KeyError if not found).

    Valid tiers per D-14: 'fast', 'average', 'best'.
    """
    data = load_models()
    whisper = data.get("whisper", {})
    if tier not in whisper:
        raise KeyError(f"Unknown tier {tier!r}; valid tiers: {sorted(whisper.keys())}")
    return whisper[tier]


def list_tiers() -> list[dict[str, Any]]:
    """Return all whisper tier entries in display order (fast, average, best).

    Used by picker page (plan 07-05) to render tier cards in the deterministic
    order the user expects: cheapest → most expensive on the left → right axis.
    """
    data = load_models()
    whisper = data.get("whisper", {})
    order = ["fast", "average", "best"]
    return [whisper[t] for t in order if t in whisper]
