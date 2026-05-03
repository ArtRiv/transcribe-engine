"""R-04 path discipline — three roots, never confused.

1. bundle_root: read-only, ephemeral in --onefile mode (sys._MEIPASS)
2. config_dir: persistent state (state.json, keypair.pem in Phase 8)
3. cache_dir: persistent but OS may wipe (model weights)
4. log_dir: persistent log files (rotating, opened by tray "View logs")

NEVER derive state paths from __file__ (R-04 anti-pattern). In --onefile
mode __file__ resolves to /tmp/_MEI*, vanishes on exit.
"""
import sys
from pathlib import Path

from platformdirs import user_cache_dir, user_config_dir, user_log_dir

APP = "transcribe-engine"


def bundle_root() -> Path:
    """Read-only assets shipped inside the binary (HTML/CSS, models.toml,
    whisper-cli). In --onefile mode this is the PyInstaller temp-extracted
    dir (`/tmp/_MEI*`); in source mode it is the engine package parent."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass is not None:
        return Path(meipass)
    # Source-mode fallback ONLY for bundle_root. config/cache/log NEVER fall back to __file__.
    # paths.py is at transcribe_engine/platform/paths.py — go up three levels to reach repo root.
    return Path(__file__).parent.parent.parent


def config_dir() -> Path:
    """Persistent state. Linux: ~/.config/transcribe-engine/."""
    p = Path(user_config_dir(APP, appauthor=False))
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    """Model weights cache. Linux: ~/.cache/transcribe-engine/models/."""
    p = Path(user_cache_dir(APP, appauthor=False)) / "models"
    p.mkdir(parents=True, exist_ok=True)
    return p


def log_dir() -> Path:
    """Rotating logs. Linux: ~/.local/state/transcribe-engine/log/."""
    p = Path(user_log_dir(APP, appauthor=False))
    p.mkdir(parents=True, exist_ok=True)
    return p
