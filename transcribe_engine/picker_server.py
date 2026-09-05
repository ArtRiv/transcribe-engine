"""Localhost picker server (D-12 / D-13 / D-15) — ephemeral first-launch UI.

Stdlib ThreadingHTTPServer only — no FastAPI/Flask (D-08 single binary discipline).
Bound to 127.0.0.1 (T-7-04); random free port (D-23).
"""

import json
import logging
import os
import re
import socket
import threading
import webbrowser
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Literal

from transcribe_engine.download.registry import get_tier, list_tiers
from transcribe_engine.download.resumable import download_resumable
from transcribe_engine.platform.paths import bundle_root, cache_dir
from transcribe_engine.state import State

log = logging.getLogger(__name__)

_HF_TOKEN_RE = re.compile(r"^hf_[A-Za-z0-9]{20,}$")  # V5 input validation


@dataclass
class PickerState:
    """Polled by progress.html every 500ms (RESEARCH §Pattern 4)."""

    state: Literal["idle", "downloading", "verifying", "done", "error"] = "idle"
    bytes_done: int = 0
    bytes_total: int = 0
    eta_seconds: int | None = None
    error: str | None = None
    selected_tier: str | None = None


def find_free_port() -> int:
    """Random free port via OS assignment (D-23)."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))  # 127.0.0.1 ONLY (T-7-04 mitigation)
        return s.getsockname()[1]


def save_hf_token(token: str, path: Path) -> None:
    """Persist HF token with file mode 0600 on Unix (T-7-03 mitigation).

    Token format pre-validated by caller (V5 input validation).
    """
    path.write_text(token)
    try:
        os.chmod(path, 0o600)
    except OSError:
        # Windows: chmod is best-effort; ACL would be ideal but adds complexity.
        # User's %APPDATA% is per-user by default on modern Windows.
        pass


# D-19: rough scaling factor vs reference Vulkan GPU (AMD RX 6600 class).
# Picker tier card displays "~Xmin for a 30-min recording (your <gpu_label>)".
# Numbers are deliberately rough — header copy uses "~" prefix.
_GPU_CLASS_MULTIPLIER = {
    "vulkan": 1.0,  # reference class
    "metal": 1.2,  # ~20% slower than reference Vulkan on whisper.cpp (rough)
    "cpu": 8.0,  # CPU fallback is ~8x slower than reference GPU class
}


def _format_time_estimate(realtime_factor: float, gpu_backend: str, gpu_label: str) -> str:
    """D-19 — render a per-tier-per-GPU time-estimate string for the picker card.

    Args:
        realtime_factor: tier.realtime_factor from models.toml (RTF on reference GPU)
        gpu_backend: GpuInfo.backend ("vulkan" | "metal" | "cpu" | unknown)
        gpu_label: GpuInfo.label, e.g. "AMD Radeon RX 6600 (Vulkan)" or "CPU (no GPU detected)"

    Returns: "~X min for a 30-min recording (your <gpu_label>)" — X rounded to nearest int.
    """
    mult = _GPU_CLASS_MULTIPLIER.get(gpu_backend, 8.0)  # unknown -> conservative
    minutes_for_30min = max(1, round(30.0 * realtime_factor * mult))
    return f"~{minutes_for_30min} min for a 30-min recording (your {gpu_label})"


def _render_tier_cards(gpu_backend: str, gpu_label: str) -> str:
    """Server-render tier cards from models.toml (D-22 -> D-14 verbatim copy + D-19 time estimate)."""
    cards = []
    for tier in list_tiers():
        recommended = ' data-recommended="true"' if tier.get("recommended") else ""
        size_gb = tier["size_bytes"] / 1_000_000_000
        time_est = _format_time_estimate(tier["realtime_factor"], gpu_backend, gpu_label)
        cards.append(
            f'<div class="tier-card" data-tier="{tier["tier"]}"{recommended}>\n'
            f'  <div class="tier-name">{tier["display_name"]}</div>\n'
            f'  <div class="tier-meta">~{size_gb:.1f} GB · {tier["filename"]}</div>\n'
            f'  <div class="tier-time">{time_est}</div>\n'
            f'  <div class="tier-desc">{tier["description"]}</div>\n'
            f"</div>"
        )
    return "\n".join(cards)


def _make_handler(
    *,
    gpu_label: str,
    gpu_backend: str,
    hf_token_path: Path,
    picker_state: PickerState,
    app_state: State,
    state_path: Path,
    server_port_holder: dict,
):
    """Closure-based handler factory — captures runtime context the request handler needs."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # silence default access log
            return

        def _send_html(self, body: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body.encode("utf-8"))))
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))

        def _send_json(self, payload: dict, status: int = 200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _check_origin(self) -> bool:
            """T-7-csrf-01 mitigation — POST endpoints reject non-localhost Origin."""
            origin = self.headers.get("Origin", "")
            expected_port = server_port_holder.get("port")
            if not origin:
                return True  # browser may omit Origin on same-origin POST; server-rendered page is same-origin
            return origin == f"http://127.0.0.1:{expected_port}"

        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                template = (bundle_root() / "picker_assets" / "index.html").read_text()
                body = template.replace("{{ gpu_label }}", gpu_label).replace(
                    "{{ tier_cards }}", _render_tier_cards(gpu_backend, gpu_label)
                )
                self._send_html(body)
            elif self.path == "/style.css":
                body = (bundle_root() / "picker_assets" / "style.css").read_text()
                self.send_response(200)
                self.send_header("Content-Type", "text/css; charset=utf-8")
                self.send_header("Content-Length", str(len(body.encode("utf-8"))))
                self.end_headers()
                self.wfile.write(body.encode("utf-8"))
            elif self.path == "/progress":
                body = (bundle_root() / "picker_assets" / "progress.html").read_text()
                self._send_html(body)
            elif self.path == "/status":
                self._send_json(asdict(picker_state))
            elif self.path == "/done":
                # D-13 handoff: user clicks "Pair this engine" -> deep-links to /pair?code=...
                # Phase 7 stubs the code; Phase 8 wires real Ed25519 keypair + pairing code.
                handoff = (
                    '<!doctype html><html><head><meta charset="utf-8">'
                    '<title>Engine ready</title><link rel="stylesheet" href="/style.css"></head>'
                    "<body><h1>Engine ready</h1>"
                    '<p class="subtitle">Models downloaded. The engine is sitting in your menu bar.</p>'
                    "<p>Next: head to the Transcribe website and pair this engine with your account. "
                    "(Pairing UI ships in Phase 8.)</p>"
                    '<button onclick="window.close()">Close this window</button>'
                    "</body></html>"
                )
                self._send_html(handoff)
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            if not self._check_origin():
                self._send_json({"error": "origin not allowed"}, status=403)
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            try:
                payload = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                payload = {}

            if self.path == "/save-hf-token":
                token = (payload.get("token") or "").strip()
                if not _HF_TOKEN_RE.match(token):
                    self._send_json({"error": "invalid HF token format"}, status=400)
                    return
                save_hf_token(token, hf_token_path)
                self._send_json({"ok": True})
            elif self.path == "/start-download":
                tier_name = (payload.get("tier") or "").strip()
                if tier_name not in {"fast", "average", "best"}:
                    self._send_json({"error": "invalid tier"}, status=400)
                    return
                picker_state.selected_tier = tier_name
                threading.Thread(
                    target=_run_download,
                    args=(tier_name, picker_state, app_state, state_path),
                    daemon=True,
                ).start()
                self._send_json({"ok": True})
            elif self.path == "/shutdown":
                threading.Thread(
                    target=lambda: server_port_holder["server"].shutdown(),
                    daemon=True,
                ).start()
                self._send_json({"ok": True})
            else:
                self.send_response(404)
                self.end_headers()

    return Handler


def _run_download(tier_name: str, picker_state: PickerState, app_state: State, state_path: Path):
    """Background daemon thread — downloads model via download_resumable, updates picker_state."""
    try:
        tier = get_tier(tier_name)
        dest = cache_dir() / tier["filename"]
        picker_state.state = "downloading"
        picker_state.bytes_done = 0
        picker_state.bytes_total = tier["size_bytes"]

        def _on_progress(done: int, total: int):
            picker_state.bytes_done = done
            picker_state.bytes_total = total

        picker_state.state = "verifying" if dest.exists() else "downloading"
        download_resumable(tier["url"], dest, tier["sha256"], on_progress=_on_progress)

        # Record installed model
        if tier_name not in app_state.installed_models:
            app_state.installed_models.append(tier_name)
            app_state.save(state_path)

        picker_state.state = "done"
    except Exception as e:
        log.exception("download failed for tier=%s", tier_name)
        picker_state.state = "error"
        picker_state.error = str(e)


def start_picker(
    *, gpu_label: str, gpu_backend: str, app_state: State, state_path: Path, hf_token_path: Path
) -> tuple[ThreadingHTTPServer, int, PickerState]:
    """Start picker server in daemon thread; opens default browser; returns (server, port, picker_state).

    Args:
        gpu_label: GpuInfo.label (D-18) — shown in tier-time line + page header
        gpu_backend: GpuInfo.backend ("vulkan" | "metal" | "cpu") — D-19 time-estimate scaling

    Caller can poll picker_state.state == 'done' to detect handoff completion,
    then call server.shutdown() to release the port.
    """
    port = find_free_port()
    picker_state = PickerState()
    port_holder: dict = {}
    Handler = _make_handler(
        gpu_label=gpu_label,
        gpu_backend=gpu_backend,
        hf_token_path=hf_token_path,
        picker_state=picker_state,
        app_state=app_state,
        state_path=state_path,
        server_port_holder=port_holder,
    )
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)  # 127.0.0.1 ONLY (T-7-04)
    port_holder["server"] = server
    port_holder["port"] = port
    threading.Thread(target=server.serve_forever, daemon=True).start()
    webbrowser.open(f"http://127.0.0.1:{port}/")
    log.info("picker server listening on http://127.0.0.1:%d", port)
    return server, port, picker_state
