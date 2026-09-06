"""
Shared output-framing helper — 16:9 crop-to-fill + burned-in caption removal
(added 2026-08-26, used by BOTH modes' clip-cutting stages).

Two separate real problems are solved with ONE ffmpeg crop, computed
per-source-video before any clips are cut:

  1. Burned-in CC removal: some source rips have captions baked directly
     into the picture (confirmed real example: "The Book of Boba Fett_S1E1"
     — plain white speaker-labeled text, bottom-center, measured via a
     bright-pixel-row scan at y=677-772 out of 804px height — see
     check_source_captions.py). There is no reliable way to erase those
     pixels; the only honest fix is to crop the band they sit in off the
     bottom of the frame, via a per-source SOURCE_CROP_OVERRIDES entry in
     config/settings.py (crop_bottom_pct). Not every source has this — the
     other real source checked (A Man in Full S01E01) showed no burned-in
     captions across multiple sampled dialogue frames, so it has no entry
     and gets crop_bottom_pct=0 (DEFAULT_CROP_BOTTOM_PCT).
  2. 16:9 aspect ratio: source videos aren't guaranteed to already be 16:9
     (Boba Fett's real frame is 1920x804, a ~2.39:1 cinematic crop, not
     16:9) — crop-to-fill (not letterbox/pad) to 16:9, by explicit user
     choice.

The bottom-band trim happens FIRST (removing burned-in caption pixels from
consideration entirely), then the 16:9 crop-to-fill is computed from
whatever's left — so on a source with both problems, one crop solves both,
rather than stacking two separate lossy operations.
"""
import subprocess
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings


def probe_dimensions(video_path: Path) -> tuple:
    """Returns (width, height) of the video's first video stream."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(video_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"ffprobe failed to read dimensions for {video_path}:\n{result.stderr}")
    w_str, h_str = result.stdout.strip().split("x")
    return int(w_str), int(h_str)


def _crop_bottom_pct_for(video_stem: str) -> float:
    override = settings.SOURCE_CROP_OVERRIDES.get(video_stem)
    if override and "crop_bottom_pct" in override:
        return override["crop_bottom_pct"]
    return settings.DEFAULT_CROP_BOTTOM_PCT


def compute_crop_filter(width: int, height: int, video_stem: str,
                         target_ar: tuple = (16, 9)) -> str:
    """Returns an ffmpeg `crop=w:h:x:y` filter fragment (no leading/trailing
    commas) that trims the configured burned-in-caption band off the bottom
    (if any, via SOURCE_CROP_OVERRIDES) and then crops the remainder to
    exactly `target_ar`, centered. Returns None if the source is already
    a clean, correctly-cropped 16:9 frame with no caption band configured
    (i.e. the crop would be a total no-op) — callers should skip adding a
    -vf entry in that case rather than pay for a pointless filter pass.
    """
    crop_bottom_pct = _crop_bottom_pct_for(video_stem)
    kept_height = height * (1.0 - crop_bottom_pct)

    target_w, target_h = target_ar
    wanted_ar = target_w / target_h
    region_ar = width / kept_height

    if region_ar >= wanted_ar:
        # region (post caption-trim) is wider than 16:9 -> crop width, keep full kept_height
        out_h = kept_height
        out_w = out_h * wanted_ar
        y = 0.0
    else:
        # region is taller than 16:9 -> crop height further, centered in the kept region
        out_w = float(width)
        out_h = out_w / wanted_ar
        y = (kept_height - out_h) / 2.0

    # round to even ints (required by yuv420p and most codecs), floor to stay in-bounds
    out_w = int(out_w) - (int(out_w) % 2)
    out_h = int(out_h) - (int(out_h) % 2)
    x = int((width - out_w) / 2)
    x -= x % 2
    y = int(y)
    y -= y % 2
    x, y = max(0, x), max(0, y)
    out_w = min(out_w, width - x)
    out_h = min(out_h, height - y)

    if out_w == width and out_h == height and x == 0 and y == 0:
        return None
    return f"crop={out_w}:{out_h}:{x}:{y}"
