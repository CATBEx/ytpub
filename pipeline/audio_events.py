"""
Explainer audio-event detection — "silent action/horror" blind-spot fix (2026-08-26).

Whisper's VAD filter (stage2_transcribe.py) drops non-speech entirely, so a genuinely
wordless action or horror beat (gunfire, a chase, a jump-scare sting) has ZERO entries
in the transcript that Explainer Stage 3 reads. It isn't classified badly — it's
literally invisible to the model, since Stage 3's prompt is built entirely from the
transcript's numbered lines.

This module scans the FULL audio track for dialogue-free gaps with sustained high
energy, reusing the same librosa RMS approach already built for highlight-reel mode's
audio-energy signal (pipeline/stage4_signals.py's `_audio_energy_for_range`, generalized
here to scan continuously in fixed windows rather than per pre-detected shot). A gap
that clears the threshold gets a synthetic placeholder entry (AUDIO_EVENT_PLACEHOLDER_TEXT)
merged into the same segment list Stage 3/4 already build their prompts from — so Gemini
can actually see that span of the movie exists and choose to write + cite a beat for it.

Explicit user choice (2026-08-26): sustained-energy detection only for this first build.
Distinguishing energy SHAPE (a sudden spike-after-quiet, more characteristic of a horror
jump-scare, vs. sustained loud, more characteristic of action) and folding in visual
motion (OpenCV, already used in highlight-reel mode) were both explicitly deferred.

CRITICAL: Stage 3 and Stage 4 MUST both call `augment_segments()` (not build their own
merge, and not load the raw transcript and use it directly) — Stage 4's citation indices
are only meaningful if built against the EXACT same list Stage 3's prompt used to number
its lines. Results are cached to disk per video (`{stem}_audio_events.json`) so this is
deterministic across repeat calls (e.g. a Stage-4-only retry) without redoing the librosa
analysis, which also guarantees Stage 3 and Stage 4 see identical indices.

Does NOT touch the real Whisper transcript file (`data/transcripts/{stem}.json`) — the
synthetic entries exist only in the merged list this module builds. Highlight-reel mode,
which reuses that same transcript file for its own (unrelated) scoring, is unaffected.

CHANGED 2026-08-27 (multi-language redesign): detection stays LANGUAGE-INDEPENDENT by
design and deliberate choice — it measures real audio energy in the source track, which
doesn't change no matter which language's narration/subtitle a given job is running
against. Rather than recompute (and re-run real librosa RMS analysis over the full audio
track) once per language for identical results, the cache now lives at the per-movie root
(`data/explainer/{movie}/audio_events.json`), SIBLING to the per-language `en/`, `hi/`,
`bn/` subfolders rather than inside any one of them — every language's Stage 3/4 call
reads/writes the same shared file.
"""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings

AUDIO_EVENT_PLACEHOLDER_TEXT = "[no dialogue — sustained high audio energy, possible action/horror moment]"


def _energy_profile(audio_path: Path, window_seconds: float) -> list:
    """RMS energy per fixed window across the whole track (same RMS approach as
    stage4_signals.py's _audio_energy_for_range, generalized to scan continuously
    rather than per pre-detected shot)."""
    import librosa
    import numpy as np

    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    hop = int(window_seconds * sr)
    if hop <= 0 or len(y) == 0:
        return []

    windows = []
    for i0 in range(0, len(y), hop):
        i1 = min(len(y), i0 + hop)
        clip = y[i0:i1]
        if len(clip) == 0:
            continue
        rms = librosa.feature.rms(y=clip)[0]
        windows.append({
            "start": round(i0 / sr, 2),
            "end": round(i1 / sr, 2),
            "energy": float(np.mean(rms)),
        })
    return windows


def _dialogue_free_gaps(segments: list, total_duration: float, min_gap_seconds: float) -> list:
    """Spans of [start, end) NOT covered by any transcript segment, long enough to
    plausibly be a real scene rather than an ordinary pause between lines of dialogue.
    Pure function — no file I/O — so it's directly unit-testable."""
    covered = sorted((s["start"], s["end"]) for s in segments)
    gaps = []
    cursor = 0.0
    for start, end in covered:
        if start - cursor >= min_gap_seconds:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if total_duration - cursor >= min_gap_seconds:
        gaps.append((cursor, total_duration))
    return gaps


def _flag_loud_gaps(gaps: list, windows: list, energy_ratio: float, min_loud_fraction: float) -> list:
    """Given dialogue-free gaps and a track's energy-window profile, decide which
    gaps are sustained-loud enough to flag as audio events. Pure function — no file
    I/O — so it's directly unit-testable with a synthetic windows list.

    "Sustained" means a real fraction of the gap is loud, not one stray loud window
    in an otherwise quiet gap (e.g. a single line of off-screen dialogue that Whisper
    happened to miss shouldn't get flagged as an action/horror beat)."""
    if not windows:
        return []

    baseline = statistics.median(w["energy"] for w in windows)
    threshold = baseline * energy_ratio

    events = []
    for gap_start, gap_end in gaps:
        gap_windows = [w for w in windows if w["start"] < gap_end and w["end"] > gap_start]
        if not gap_windows:
            continue
        loud = [w for w in gap_windows if w["energy"] >= threshold]
        if len(loud) / len(gap_windows) < min_loud_fraction:
            continue
        events.append({
            "start": round(gap_start, 2),
            "end": round(gap_end, 2),
            "text": AUDIO_EVENT_PLACEHOLDER_TEXT,
            "is_audio_event": True,
        })
    return events


def detect_audio_events(video_stem: str, segments: list, audio_path: Path = None) -> list:
    """Find dialogue-free gaps with sustained high audio energy and return them as
    synthetic segment-shaped dicts (start/end/text/is_audio_event=True). Cached to
    disk per video so repeat calls (Stage 3 AND Stage 4 both call this) don't redo
    the librosa analysis and, more importantly, stay identical to each other.

    audio_path defaults to the production convention (settings.AUDIO_DIR/{stem}.wav,
    written by stage1_ingest.extract_audio) but can be overridden — mainly so tests
    can point at a synthetic audio file without needing a full Stage 1 run."""
    if not settings.AUDIO_EVENT_DETECTION_ENABLED:
        return []

    cache_path = settings.EXPLAINER_DIR / video_stem / "audio_events.json"  # shared at the
                                          # movie root -- NOT under a language subfolder, see
                                          # module docstring
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))["events"]

    resolved_audio_path = audio_path or (settings.AUDIO_DIR / f"{video_stem}.wav")
    if not resolved_audio_path.exists():
        print(f"  [warn] no extracted audio at {resolved_audio_path} — skipping audio-event detection")
        return []

    import librosa
    total_duration = librosa.get_duration(path=str(resolved_audio_path))

    gaps = _dialogue_free_gaps(segments, total_duration, settings.AUDIO_EVENT_MIN_GAP_SECONDS)
    events = []
    if gaps:
        windows = _energy_profile(resolved_audio_path, settings.AUDIO_EVENT_WINDOW_SECONDS)
        events = _flag_loud_gaps(gaps, windows, settings.AUDIO_EVENT_ENERGY_RATIO,
                                  settings.AUDIO_EVENT_MIN_LOUD_FRACTION)

    cache_path.write_text(json.dumps({"events": events}, indent=2, ensure_ascii=False), encoding="utf-8")
    if events:
        print(f"  [audio-events] {len(events)} dialogue-free, sustained-high-energy window(s) "
              f"detected -> flagged for Stage 3/4")
    return events


def merge_segments_with_audio_events(segments: list, events: list) -> list:
    """Combine real transcript segments with synthetic audio-event entries into ONE
    time-ordered list, each tagged is_audio_event True/False."""
    combined = [dict(s, is_audio_event=False) for s in segments] + list(events)
    combined.sort(key=lambda s: s["start"])
    return combined


def augment_segments(video_stem: str, segments: list, audio_path: Path = None) -> list:
    """Shared entry point Stage 3 AND Stage 4 must both call (instead of using the
    raw transcript segments directly) so their line-index numbering stays identical —
    Stage 4's citation indices are only meaningful if built against the same list
    Stage 3's prompt used to number its lines."""
    events = detect_audio_events(video_stem, segments, audio_path=audio_path)
    return merge_segments_with_audio_events(segments, events)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python audio_events.py <video_stem>")
        sys.exit(1)
    stem = sys.argv[1]
    transcript_path = settings.TRANSCRIPT_DIR / f"{stem}.json"
    segs = json.loads(transcript_path.read_text(encoding="utf-8"))["segments"]
    found = detect_audio_events(stem, segs)
    print(json.dumps(found, indent=2))
