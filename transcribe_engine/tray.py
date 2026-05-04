"""System tray icon — D-09 locked 5-item menu surface.

Cross-platform via pystray (auto-selects _xorg / _darwin / _win32 backend).
PyInstaller --hidden-import is set per-OS in build script (R-05).
"""
import os
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw
from pystray import Icon, Menu, MenuItem

from transcribe_engine import __version__

HOSTED_FRONTEND_URL = "https://transcribe.fel.tec.br"  # placeholder; align with v1 production URL


def make_icon_image() -> Image.Image:
    """Programmatic icon (no asset file to ship). 64x64 dark square + TR text."""
    img = Image.new("RGB", (64, 64), "#111")
    d = ImageDraw.Draw(img)
    d.rectangle((10, 24, 54, 40), fill="#fff")
    d.text((22, 26), "TR", fill="#111")
    return img


def open_logs_in_default_viewer(log_path: Path) -> None:
    """OS-dispatch: open the rotating log file in the user's default text viewer.

    D-21 ('non-developer can scan'); RESEARCH §Pattern 6.
    """
    try:
        if sys.platform == "win32":
            os.startfile(str(log_path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", str(log_path)], check=False)
        else:  # linux / other unix
            subprocess.run(["xdg-open", str(log_path)], check=False)
    except (OSError, FileNotFoundError):
        pass  # fail silently — tray menu callback must not crash the daemon


def build_tray(
    *,
    gpu_label: str,
    on_open_website: Callable[[], None],
    on_open_logs: Callable[[], None],
    on_open_picker: Callable[[], None],
    on_quit: Callable[[], None],
) -> Icon:
    """Build the locked D-09 tray menu (5 items, exact order, no additions).

    - Status line (non-clickable) — `transcribe-engine v{X} — {GPU label}`
    - Open Transcribe website
    - View logs
    - Manage models...
    - Quit
    """
    title = f"transcribe-engine v{__version__} — {gpu_label}"
    return Icon(
        "transcribe-engine",
        make_icon_image(),
        title,
        menu=Menu(
            MenuItem(title, None, enabled=False),
            MenuItem("Open Transcribe website", lambda icon, item: on_open_website()),
            MenuItem("View logs", lambda icon, item: on_open_logs()),
            MenuItem("Manage models...", lambda icon, item: on_open_picker()),
            MenuItem("Quit", lambda icon, item: on_quit()),
        ),
    )


def open_hosted_frontend() -> None:
    """Default 'Open Transcribe website' callback — opens hosted webapp in default browser."""
    webbrowser.open(HOSTED_FRONTEND_URL)
