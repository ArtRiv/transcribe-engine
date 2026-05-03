"""SEC-08 sentinel — engine perimeter is milestone-locked.

Engine NEVER holds Supabase credentials and NEVER talks to Supabase.
This test AST-scans every module under transcribe_engine/ to enforce
the perimeter. Phase 8 will add WebRTC + Ed25519 keypair, but the
Supabase boundary holds permanently.

Inherits from v1's Phase 02 SEC-08 sentinel pattern (engine version
is broader: also bans HTTP frameworks per RESEARCH "Anti-Patterns").

Banned import names rationale:
- supabase: engine never holds Supabase credentials (milestone-locked)
- httpx, requests, aiohttp: engine uses stdlib urllib.request only
  (RESEARCH "Anti-Patterns to Avoid" + bundle weight discipline)
- fastapi, flask, starlette: engine uses stdlib ThreadingHTTPServer
  for picker (D-23); no web framework in bundle

Parameterized per file — failures point to the exact module + banned
import name (no opaque "somewhere in engine" message).
"""

import ast
from pathlib import Path

import pytest

ENGINE_ROOT = Path(__file__).parent.parent / "transcribe_engine"
BANNED_IMPORT_NAMES = {
    "supabase",
    "httpx",
    "requests",
    "aiohttp",
    "fastapi",
    "flask",
    "starlette",
}
BANNED_STRING_SUBSTRINGS = ("supabase.co",)


def _python_files() -> list[Path]:
    return sorted(ENGINE_ROOT.rglob("*.py"))


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(ENGINE_ROOT)))
def test_module_does_not_import_banned_libraries(path: Path) -> None:
    """No engine module imports Supabase or HTTP frameworks (SEC-08 + bundle hygiene)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in BANNED_IMPORT_NAMES, (
                    f"{path.relative_to(ENGINE_ROOT)} imports banned module {alias.name!r}"
                )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".")[0]
                assert root not in BANNED_IMPORT_NAMES, (
                    f"{path.relative_to(ENGINE_ROOT)} imports from banned module {node.module!r}"
                )


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(ENGINE_ROOT)))
def test_module_has_no_supabase_url_string_literals(path: Path) -> None:
    """No string literal in engine code references Supabase URLs."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            for needle in BANNED_STRING_SUBSTRINGS:
                assert needle not in node.value, (
                    f"{path.relative_to(ENGINE_ROOT)} contains banned string {needle!r}"
                )
