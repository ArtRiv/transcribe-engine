"""transcribe-engine entry point — D-08 single binary, no subcommands.

State machine (D-08 + D-11):
  Run binary → load state.json → detect GPU →
    if no models: run picker first-launch flow →
    enter tray daemon mode.

Phase 7 stubs the keypair / signaling slots in state.json; Phase 8 fills them.
"""

import argparse
import asyncio
import logging
import sys
import time

from transcribe_engine import __version__
from transcribe_engine.logging_setup import setup_logging
from transcribe_engine.picker_server import start_picker
from transcribe_engine.platform.gpu import detect_gpu
from transcribe_engine.platform.paths import config_dir
from transcribe_engine.state import State

# NOTE: `tray` is imported lazily inside main() — its `pystray` dependency
# initializes the platform GUI backend at import time (Xlib.Display() on Linux),
# which fails on headless systems (CI, SSH without X-forwarding) and would
# break --version / --logs-path / --reset-models.

log = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    """D-08 acceptable undocumented escape hatches: --no-tray / --reset-models / --logs-path."""
    p = argparse.ArgumentParser(prog="transcribe-engine", add_help=True)
    p.add_argument("--version", action="version", version=f"transcribe-engine {__version__}")
    p.add_argument("--no-tray", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--reset-models", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--logs-path", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    log_path = setup_logging()
    log.info("transcribe-engine v%s starting", __version__)

    if args.logs_path:
        print(log_path)
        return 0

    state_path = config_dir() / "state.json"
    state = State.load(state_path)

    if args.reset_models:
        state.installed_models = []
        state.save(state_path)
        log.info("models reset; will re-prompt on next launch")
        return 0

    # GPU detection (D-18 — never raises, CPU fallback always)
    gpu = asyncio.run(detect_gpu())
    log.info("GPU detected: %s (%s)", gpu.device_name, gpu.backend)

    # First-launch flow (D-11): no models installed → picker page
    hf_token_path = config_dir() / "hf_token"
    if not state.has_models():
        log.info("no models installed; starting first-launch picker flow")
        server, port, picker_state = start_picker(
            gpu_label=gpu.label,
            gpu_backend=gpu.backend,  # D-19 — drives per-GPU time-estimate scaling in picker
            app_state=state,
            state_path=state_path,
            hf_token_path=hf_token_path,
        )
        # Block until user completes picker → download → handoff
        try:
            while picker_state.state not in ("done", "error"):
                time.sleep(0.5)
            if picker_state.state == "error":
                log.error("picker download failed: %s", picker_state.error)
                return 1
            # Reload state — picker writes installed_models after successful download
            state = State.load(state_path)
        finally:
            server.shutdown()

    # Phase 8 will gen Ed25519 keypair + signaling-WebSocket connect here.
    # Phase 7: state.keypair_path is None; just sit in the tray.

    if args.no_tray:
        log.info("--no-tray flag set; idle until SIGTERM")
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            return 0

    # Tray daemon mode — runs OS event loop until Quit
    def _on_open_picker():
        """D-15: tray Manage models... re-opens the picker page."""
        log.info("Manage models... -> re-opening picker server")
        start_picker(
            gpu_label=gpu.label,
            gpu_backend=gpu.backend,  # D-19 — drives per-GPU time-estimate scaling in picker
            app_state=state,
            state_path=state_path,
            hf_token_path=hf_token_path,
        )

    def _on_quit():
        log.info("user clicked Quit; shutting down")
        icon.stop()  # type: ignore[has-type]

    # Lazy import — see module-level NOTE.
    from transcribe_engine.tray import build_tray, open_hosted_frontend, open_logs_in_default_viewer

    icon = build_tray(
        gpu_label=gpu.label,
        on_open_website=open_hosted_frontend,
        on_open_logs=lambda: open_logs_in_default_viewer(log_path),
        on_open_picker=_on_open_picker,
        on_quit=_on_quit,
    )
    log.info("tray ready; entering pystray event loop")
    icon.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
