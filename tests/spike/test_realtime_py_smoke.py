"""Wave-0 spike: Supabase realtime-py pure-Python PyInstaller cleanliness.

Validates assumption A3 from RESEARCH.md:
  'realtime-py is pure-Python — no native extensions; PyInstaller-clean.'

Tests:
  1. AsyncRealtimeClient imports without raising.
  2. The package directory contains no .so/.dylib/.pyd files.
"""

import importlib.util
from pathlib import Path


def test_realtime_client_imports_in_source() -> None:
    """AsyncRealtimeClient is importable from the installed `realtime` package.

    Uses the public API path documented in github.com/supabase/realtime-py.
    Does NOT import internal/private modules (underscore-prefixed submodules).
    """
    from realtime import AsyncRealtimeClient  # noqa: F401

    # Verify the class is the async variant expected for the engine's event loop
    assert AsyncRealtimeClient is not None
    assert callable(AsyncRealtimeClient)


def test_realtime_py_no_native_extensions() -> None:
    """Validates assumption A3: realtime is pure-Python with no native extensions.

    Walks the installed package directory and asserts no .so, .dylib, or .pyd
    files exist. Pure-Python packages are PyInstaller-clean by default — no
    --collect-binaries or --add-binary flags needed in the engine spec.

    If this test ever fails (a future realtime release adds a C extension),
    the transcribe-engine.spec must be updated with explicit collect_binaries
    calls before the next PyInstaller build.
    """
    spec = importlib.util.find_spec("realtime")
    assert spec is not None, "realtime package not installed"
    assert spec.origin is not None, "realtime package origin not found"

    pkg_dir = Path(spec.origin).parent
    native_exts = (
        list(pkg_dir.rglob("*.so")) + list(pkg_dir.rglob("*.dylib")) + list(pkg_dir.rglob("*.pyd"))
    )
    assert native_exts == [], (
        f"realtime package contains native extensions (assumption A3 violated): {native_exts}\n"
        "If this is intentional, update transcribe-engine.spec with collect_binaries('realtime')."
    )
