"""ASR + diarization → canonical jsonb payload (Phase 2→3 contract; RESEARCH.md §1329-1356).

Engine port of v1 backend/app/pipeline/merge.py — carried verbatim.
Pure function, no side effects, no v1-specific imports.
Output schema is the locked Phase 2→3 contract (unchanged in the engine).
"""

from __future__ import annotations

from typing import Any


def merge_to_payload(asr: dict, diar: list[dict]) -> dict[str, Any]:
    """Combine ASR segments with diarization turns into the canonical payload shape.

    asr: {'language', 'segments': [{start, end, text, words}]}
    diar: [{start, end, speaker}]

    Returns the canonical jsonb shape (Phase 2→3 contract): renumbered speakers
    (S0, S1, ...), human-readable labels, and segments tagged with the
    max-overlap speaker.
    """
    speaker_map: dict[str, str] = {}
    next_id = 0
    out_segs: list[dict[str, Any]] = []
    for i, seg in enumerate(asr.get("segments", [])):
        speaker_raw = _max_overlap(seg["start"], seg["end"], diar)
        if speaker_raw not in speaker_map:
            speaker_map[speaker_raw] = f"S{next_id}"
            next_id += 1
        spk = speaker_map[speaker_raw]
        out_segs.append(
            {
                "id": f"seg_{i:04d}",
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "speaker": spk,
                "text": seg.get("text", ""),
                "words": seg.get("words", []),
            }
        )
    speakers = [
        {"id": sid, "label": f"Speaker {int(sid[1:]) + 1}"}
        for sid in sorted(speaker_map.values(), key=lambda s: int(s[1:]))
    ]
    duration_sec = out_segs[-1]["end"] if out_segs else 0.0
    return {
        "version": 1,
        "language": asr.get("language", "en"),
        "duration_sec": duration_sec,
        "speakers": speakers,
        "segments": out_segs,
    }


def _max_overlap(start: float, end: float, diar: list[dict]) -> str:
    """Return the diarization speaker label that maximally overlaps [start, end].

    If no diarization turn overlaps the interval, returns the sentinel
    `"SPEAKER_UNKNOWN"` (the merge will renumber it to whatever S{n} slot it
    lands in).
    """
    best, best_overlap = "SPEAKER_UNKNOWN", 0.0
    for turn in diar:
        overlap = max(0.0, min(end, turn["end"]) - max(start, turn["start"]))
        if overlap > best_overlap:
            best, best_overlap = turn["speaker"], overlap
    return best
