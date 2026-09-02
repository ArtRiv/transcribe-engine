# transcribe-engine

Self-hostable transcription engine for the [Transcribe](https://github.com/artriv/transcribe) webapp.
Bundles `whisper.cpp` (Vulkan / Metal / CPU) and `pyannote.audio` (CPU diarization)
into a single tray-icon binary per OS — no Docker, no Python install, no GPU drivers
to wrangle beyond what's already on your machine for games.

Audio never leaves your machine. Pairing with the hosted webapp over WebRTC is implemented
on `main` and ships in the next release; the published binaries run standalone (run binary
→ first-launch picker downloads model weights → engine sits in your menu bar).

## Privacy model

| Holds                                         | Talks to                                            |
|-----------------------------------------------|-----------------------------------------------------|
| Local model weights (cached on disk)          | huggingface.co (one-time model download)            |
| A local device keypair                        | Your local browser (loopback HTTP, picker UI)       |
| Optional HuggingFace token (stored mode 0600) | The Transcribe signaling function                   |

The engine **never** holds:
- Your webapp credentials (the hosted frontend owns those, not the engine)
- Service-role keys, JWTs, or any token that authenticates against a cloud backend
- Audio data after a transcription completes (raw audio is processed in-place and discarded)

The engine is open-source MIT — the trust mechanism is "you can read the source and verify
the SHA-256 of the binary you downloaded matches the one CI built."

## Install

1. Download the binary for your OS from the [latest Release](https://github.com/artriv/transcribe-engine/releases/latest):
   - macOS Apple Silicon: `transcribe-engine-darwin-arm64`
   - Windows 64-bit: `transcribe-engine-windows-amd64.exe`
   - Linux 64-bit: `transcribe-engine-linux-amd64`
2. Download the matching `.sha256` file and verify integrity (see "Verifying SHA-256" below).
3. Run the binary. Your OS may show a "Apple could not verify" (macOS) or
   "Windows protected your PC" (Windows) prompt the first time — these are expected
   for unsigned open-source binaries; click "Open" / "Run anyway" after verifying SHA-256.
4. The first launch opens your default browser to a localhost picker page where you
   choose a quality tier (Fast / Average / Best) and paste a HuggingFace token. After the
   model download completes, the engine sits in your menu bar / system tray.

## Verifying SHA-256

The release publishes a `.sha256` sidecar for every binary. Verify before running:

**macOS / Linux**
```bash
shasum -a 256 -c transcribe-engine-darwin-arm64.sha256
# or
sha256sum -c transcribe-engine-linux-amd64.sha256
```

**Windows (PowerShell)**
```powershell
$expected = (Get-Content transcribe-engine-windows-amd64.exe.sha256 -Raw).Split()[0]
$actual = (Get-FileHash transcribe-engine-windows-amd64.exe -Algorithm SHA256).Hash.ToLower()
if ($expected -eq $actual) { "OK" } else { "MISMATCH" }
```

A mismatch means the file was corrupted or tampered with — re-download from the Release page.

## Build from source

Requires:
- Python 3.11
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- `cmake` >= 3.20 (whisper.cpp build)
- **Linux/Windows:** Vulkan SDK + drivers (Mesa `libvulkan-dev` on Linux; LunarG SDK on Windows)
- **macOS:** Xcode Command Line Tools (Metal backend built automatically)

```bash
git clone https://github.com/artriv/transcribe-engine.git
cd transcribe-engine
uv sync --extra dev

# Build whisper-cli for your OS (pick one):
bash scripts/build_whisper_cpp_linux.sh
bash scripts/build_whisper_cpp_macos.sh
pwsh  scripts/build_whisper_cpp_windows.ps1

# Build the engine binary (single PyInstaller --onefile output in dist/)
# Linux:
WHISPER_CLI=$(realpath whisper.cpp/build/bin/whisper-cli) \
  PYSTRAY_BACKEND=_xorg bash scripts/build.sh

# macOS:
WHISPER_CLI=$(realpath whisper.cpp/build/bin/whisper-cli) \
  PYSTRAY_BACKEND=_darwin bash scripts/build.sh

# Windows (PowerShell):
$env:WHISPER_CLI = (Resolve-Path "whisper.cpp\build\bin\whisper-cli.exe").Path
$env:PYSTRAY_BACKEND = "_win32"
bash scripts/build.sh

# Smoke-test the binary:
./dist/transcribe-engine --version
# Expect: transcribe-engine 0.1.0
```

Expected:
- Bundle size: ~330 MB per OS (most of it is `torch` + `pyannote`)
- Cold-start: ~6-7s on Linux; ~8-15s on macOS (Gatekeeper re-scans the `--onefile` extraction on
  first run — a daemon launched once per session amortizes the cost over hours of use)

## Architecture (high level)

```
Hosted Frontend (transcribe.fel.tec.br)
    |
    |  WebRTC P2P — engine is paired once, then sits in tray
    |
transcribe-engine  (this repo)
    |
    +-- subprocess --> whisper-cli (Vulkan / Metal / CPU)
    |                      |
    |                      +--> transcript JSON
    |
    +-- in-process --> pyannote.audio on CPU
                           |
                           +--> speaker labels
                                (merged with transcript, returned over WebRTC to frontend)
```

The engine is intentionally minimal: a tray daemon that bundles whisper.cpp + pyannote,
runs a localhost picker on first launch, and accepts inbound transcription jobs over
WebRTC. It does not run an HTTP API. It does not hold any cloud credentials. It does not
auto-update (manual download-and-replace for v0.1.x).

## License

MIT — see [LICENSE](./LICENSE).

## Status

`v0.1.1` is the latest published release: standalone tray daemon, first-launch model picker
(Fast / Average / Best quality tiers), CPU-fallback GPU detection, and a single PyInstaller
`--onefile` binary per OS.

Pairing and the WebRTC transport are implemented on `main` — device keypair, signaling
client, chunked transport, and the resume path — and ship in the next release. They are not
in the published binaries yet.
