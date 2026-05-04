# scripts/build_whisper_cpp_windows.ps1
# whisper.cpp Windows Vulkan build. Pinned to v1.8.4 (matches v1 backend).
#
# Idempotent: skips clone+build if pinned tag already on disk with binary present.
# Outputs path to whisper-cli.exe on stdout so build.sh can pick it up.
$ErrorActionPreference = "Stop"

$WhisperTag = "v1.8.4"
$BuildDir = if ($env:WHISPER_CPP_BUILD_DIR) { $env:WHISPER_CPP_BUILD_DIR }
            elseif ($env:RUNNER_TEMP) { Join-Path $env:RUNNER_TEMP "whisper.cpp" }
            else { Join-Path $env:TEMP "whisper.cpp" }

# Pre-flight tools
Get-Command cmake -ErrorAction Stop | Out-Null
Get-Command git -ErrorAction Stop | Out-Null

# Idempotency: skip clone+build if pinned tag already on disk with binary present
$WhisperCli = Join-Path $BuildDir "build\bin\Release\whisper-cli.exe"
if (-not (Test-Path $WhisperCli)) {
    $WhisperCli = Join-Path $BuildDir "build\bin\whisper-cli.exe"
}

if ((Test-Path "$BuildDir\.git") -and (Test-Path $WhisperCli)) {
    $HeadTag = (git -C $BuildDir describe --tags --exact-match 2>$null)
    if ($HeadTag -eq $WhisperTag) {
        Write-Host "[build_whisper_cpp_windows.ps1] already built at $WhisperTag; skipping" -ForegroundColor DarkGray
        Write-Output $WhisperCli
        exit 0
    }
}

# Clone (if absent) + checkout pinned tag
if (-not (Test-Path "$BuildDir\.git")) {
    Write-Host "[build_whisper_cpp_windows.ps1] cloning whisper.cpp into $BuildDir" -ForegroundColor DarkGray
    git clone https://github.com/ggml-org/whisper.cpp.git $BuildDir
} else {
    git -C $BuildDir fetch --tags --quiet
}
git -C $BuildDir checkout $WhisperTag

# Configure + build with Vulkan
Write-Host "[build_whisper_cpp_windows.ps1] configuring cmake (-DGGML_VULKAN=1)" -ForegroundColor DarkGray
cmake -S $BuildDir -B "$BuildDir\build" -DGGML_VULKAN=1 -DCMAKE_BUILD_TYPE=Release

Write-Host "[build_whisper_cpp_windows.ps1] building (this can take 5-10 min on first run)" -ForegroundColor DarkGray
cmake --build "$BuildDir\build" --config Release --parallel

# Resolve output path (MSVC puts it under Release/ subdirectory)
$WhisperCli = Join-Path $BuildDir "build\bin\Release\whisper-cli.exe"
if (-not (Test-Path $WhisperCli)) {
    $WhisperCli = Join-Path $BuildDir "build\bin\whisper-cli.exe"
}
if (-not (Test-Path $WhisperCli)) {
    Write-Error "[build_whisper_cpp_windows.ps1] whisper-cli.exe not found after build"
    exit 1
}

$BuildSha = (git -C $BuildDir rev-parse HEAD)
Write-Host "[build_whisper_cpp_windows.ps1] whisper-cli.exe built: $WhisperCli ($WhisperTag @ $BuildSha)" -ForegroundColor DarkGray

# Print the binary path to stdout so build.sh + CI can capture it.
Write-Output $WhisperCli
