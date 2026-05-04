"""Wave-0 spike: aiortc PyInstaller bundling — source-mode and frozen-binary tests.

Validates assumptions A1/A2 from RESEARCH.md §"Pitfall #2":
  A1: aiortc bundles cleanly inside a PyInstaller --onefile binary.
  A2: pylibsrtp and av native deps are included in the frozen binary.

The entire Phase 8 protocol layer (Plans 06-09) builds on top of aiortc;
if this spike fails it is cheaper to replan now than after those plans ship.

SPIKE FAILURE HANDLING (per 08-01-PLAN.md):
  If the frozen binary test fails, capture:
    1. Exact error (dlopen failure, ImportError, missing .so paths)
    2. PyInstaller flags tried (AIORTC_PYINSTALLER_FLAGS constant below)
    3. Whether source-mode test still passes
  Record in SUMMARY.md under "Spike Findings" and mark plan BLOCKED.
  Do NOT mark xfail or skip to paper over the failure.
"""

import asyncio
import subprocess
import textwrap
from pathlib import Path

import pytest

# ── PyInstaller flags needed to bundle aiortc ──────────────────────────────────
#
# These flags are the authoritative source for aiortc bundling; they are
# referenced by the main transcribe-engine.spec in Plan 09 (final binary build).
#
# Phase 8 Wave-0 spike results (2026-05-04, Linux x86_64 Ubuntu 26.04):
#   - av (PyAV / FFmpeg wrapper): requires collect_all('av') — has data files
#     and .so binaries in the package dir
#   - pylibsrtp: requires collect_binaries('pylibsrtp') — has libsrtp2 .so files
#   - aiortc itself: collect_all('aiortc') covers the rest
#
# If these flags are empty, the frozen binary will fail with an ImportError on
# first launch (cannot find libsrtp2.so or av/*.so).
#
# Citation: RESEARCH.md §"Pitfall #2 — aiortc native deps" (lines 553-557).
AIORTC_PYINSTALLER_FLAGS: list[str] = [
    "--collect-all=av",
    "--collect-binaries=pylibsrtp",
    "--collect-all=aiortc",
]


def test_aiortc_imports_in_source() -> None:
    """aiortc.RTCPeerConnection can be imported and instantiated in source mode.

    Validates assumption A1 (source-mode half). Smoke test: open a peer
    connection with no ICE servers and immediately close it.
    """
    from aiortc import RTCConfiguration, RTCPeerConnection

    async def _smoke() -> None:
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        assert pc is not None
        await pc.close()

    asyncio.run(_smoke())


@pytest.mark.slow
def test_aiortc_imports_in_frozen_binary(tmp_path: Path) -> None:
    """aiortc.RTCPeerConnection can be imported inside a PyInstaller --onefile binary.

    This test takes ~2-5 minutes (PyInstaller build time). It is gated behind
    the `slow` marker and the RUN_SPIKE=1 env var to avoid running in fast CI.

    SPIKE FAILURE: If this test fails, do NOT mark xfail. Instead:
      1. Capture the exact error from the frozen binary stdout/stderr.
      2. Record the PyInstaller flags that were tried (AIORTC_PYINSTALLER_FLAGS).
      3. Document in SUMMARY.md "Spike Findings" and mark plan BLOCKED.
      4. Check whether source-mode test_aiortc_imports_in_source still passes.

    The test binary is minimal: it only imports RTCPeerConnection and prints
    "aiortc OK" — it does NOT open a real peer connection (which would need
    a network interface and ICE stack).
    """
    import os

    # Gate: only run when explicitly opted in (avoids 2-minute CI overhead)
    if os.environ.get("RUN_SPIKE") != "1":
        pytest.skip("Set RUN_SPIKE=1 to run the PyInstaller frozen-binary spike")

    # Write a minimal entrypoint that imports aiortc and exits
    entry = tmp_path / "aiortc_smoke_entry.py"
    entry.write_text(
        textwrap.dedent("""\
        from aiortc import RTCPeerConnection, RTCConfiguration
        # Instantiate without opening ICE (no network needed for the import test)
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        print("aiortc OK")
        # Close is async; in this minimal probe we skip the event loop
        import asyncio
        asyncio.run(pc.close())
        print("aiortc close OK")
        """)
    )

    dist_dir = tmp_path / "dist"
    work_dir = tmp_path / "build"
    binary_name = "spike_aiortc"

    # Build the frozen binary with the same flags that the main spec will use
    pyinstaller_cmd = [
        "uv",
        "run",
        "pyinstaller",
        "--onefile",
        "--name",
        binary_name,
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir),
        "--specpath",
        str(tmp_path),
        *AIORTC_PYINSTALLER_FLAGS,
        str(entry),
    ]

    build_result = subprocess.run(
        pyinstaller_cmd,
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
    )

    assert build_result.returncode == 0, (
        f"PyInstaller build failed (returncode={build_result.returncode}).\n"
        f"STDOUT:\n{build_result.stdout[-3000:]}\n"
        f"STDERR:\n{build_result.stderr[-3000:]}"
    )

    frozen_binary = dist_dir / binary_name
    assert frozen_binary.exists(), f"Frozen binary not found at {frozen_binary}"

    # Record binary size for SUMMARY (Phase 7 baseline: 331 MB)
    binary_size_mb = frozen_binary.stat().st_size / (1024 * 1024)
    print(f"\nFrozen binary size: {binary_size_mb:.1f} MB (Phase 7 baseline: 331 MB)")

    # Run the frozen binary and check output
    run_result = subprocess.run(
        [str(frozen_binary)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert run_result.returncode == 0, (
        f"Frozen binary exited non-zero (returncode={run_result.returncode}).\n"
        f"This indicates a dlopen failure (libsrtp2, av/*.so) or import error.\n"
        f"Flags tried: {AIORTC_PYINSTALLER_FLAGS}\n"
        f"STDOUT:\n{run_result.stdout}\n"
        f"STDERR:\n{run_result.stderr}"
    )

    assert "aiortc OK" in run_result.stdout, (
        f"Frozen binary ran but did not print 'aiortc OK'.\n"
        f"STDOUT: {run_result.stdout!r}\n"
        f"STDERR: {run_result.stderr!r}"
    )
