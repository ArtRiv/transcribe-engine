"""Quality preset registry — Fast / Average / Best (D-14).

Source-of-truth for model files is models.toml (D-22); this module
composes the runtime view (which presets fit current VRAM, which
have weights downloaded). Plan 07-04 ships the registry reader.

Engine port of v1 backend/app/pipeline/presets.py. Deltas applied:
1. Renamed "slow" → "best" everywhere (D-14)
2. Dropped "average_turbo" (D-14 collapses to three named tiers: fast/average/best)
3. Replaced settings.enable_slow_preset gate with True constant (D-14: Best is always offered)
4. Replaced Path(settings.models_dir) with cache_dir() from transcribe_engine.platform.paths
5. Added import: from transcribe_engine.platform.paths import cache_dir
"""

from __future__ import annotations

from typing import Any

from transcribe_engine.platform.paths import cache_dir

HEADROOM_MB = 1024  # leave 1 GB for diarization+OS+activations

# Engine's runtime preset spec — model_filename + estimated_vram_mb
# match models.toml (plan 07-02 / D-22). For v0.1.0 we hardcode here;
# plan 07-04's download/registry.py wires this from models.toml.
PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "model_filename": "ggml-small.bin",
        "estimated_vram_mb": 1500,
    },
    "average": {
        "model_filename": "ggml-medium.bin",
        "estimated_vram_mb": 3500,
    },
    "best": {
        "model_filename": "ggml-large-v3.bin",
        "estimated_vram_mb": 5500,
    },
}


def available_presets(vram_total_mb: int) -> dict[str, dict[str, Any]]:
    """Filter PRESETS by which fit the user's VRAM and have weights downloaded.

    CPU-fallback (vram_total_mb == 0) skips the VRAM filter — CPU has no
    VRAM constraint, but per D-19 the picker page warns about realtime
    factor.

    A preset is included only if (a) its estimated VRAM fits under
    (vram_total_mb - HEADROOM_MB) [skipped for CPU/vram_total_mb==0], AND
    (b) the model file exists at cache_dir() / model_filename.

    D-14: "Best" is always offered when available — the settings gate
    from v1 (enable_slow_preset) is dropped in the engine.
    """
    out: dict[str, dict[str, Any]] = {}
    budget = vram_total_mb - HEADROOM_MB if vram_total_mb > 0 else 0
    for name, spec in PRESETS.items():
        if vram_total_mb > 0 and spec["estimated_vram_mb"] > budget:
            continue
        model_path = cache_dir() / spec["model_filename"]
        if not model_path.exists():
            continue
        out[name] = {**spec, "model_path": str(model_path), "tier": name}
    return out
