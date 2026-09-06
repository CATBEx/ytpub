"""
One-off probe tool — NOT part of the pipeline.

Checks a source video for burned-in (baked into the picture) captions, and
suggests a SOURCE_CROP_OVERRIDES entry for config/settings.py if it finds
one. Formalizes the manual check done for the first two real source videos
(2026-08-26): confirm via ffprobe that there's no soft-subtitle stream (if
there IS one, this whole approach is moot — just don't mux it), then sample
several dialogue-heavy frames and scan for a horizontal band of bright
(near-white) pixels that shows up in the SAME row range across multiple
independent samples — a real caption band is positionally consistent
frame-to-frame; a bright lamp/window/prop in one shot isn't.

No new dependencies: frame sampling uses ffmpeg's rawvideo/gray output
piped straight into numpy (already pinned in requirements.txt) rather than
Pillow, and the human-checkable JPG samples are extracted by ffmpeg itself.

Run:
    python check_source_captions.py "data/input/Some Movie.mp4"
Optionally set how many frames to sample (default 8):
    python check_source_captions.py "data/input/Some Movie.mp4" 12

Output:
  - data/_cc_samples/{stem}/sample_*.jpg — for you to eyeball directly
  - a printed verdict + (if a band is found) a ready-to-paste
    SOURCE_CROP_OVERRIDES entry
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings

BRIGHT_THRESHOLD = 225      # grayscale value considered "near-white text"
MIN_BRIGHT_PIXELS_IN_ROW = 15   # a row needs at least this many bright pixels to count as "text"
MIN_SAMPLE_AGREEMENT = 0.5  # a row must be flagged in at least this fraction of samples
BAND_MARGIN_PCT = 0.03      # extra safety margin added above the measured band top


def _has_subtitle_stream(video_path: Path) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "s", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def _probe_dimensions(video_path: Path) -> tuple:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(video_path)],
        capture_output=True, text=True,
    )
    w, h = result.stdout.strip().split("x")
    return int(w), int(h)


def _probe_duration(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def _sample_timestamps(video_path: Path, video_stem: str, n: int) -> list:
    """Prefer real dialogue timestamps from an existing transcript (more
    likely to actually show captions, if any exist) — falls back to evenly
    spaced points across the runtime if no transcript has been generated
    for this source yet."""
    transcript_path = settings.TRANSCRIPT_DIR / f"{video_stem}.json"
    if transcript_path.exists():
        import json
        segments = json.loads(transcript_path.read_text(encoding="utf-8"))["segments"]
        if len(segments) >= n:
            step = len(segments) / n
            return [segments[int(i * step)]["start"] + 0.5 for i in range(n)]
    duration = _probe_duration(video_path)
    return [duration * (i + 1) / (n + 1) for i in range(n)]


def _extract_gray_frame(video_path: Path, timestamp: float, width: int, height: int):
    cmd = ["ffmpeg", "-y", "-ss", str(timestamp), "-i", str(video_path),
           "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-vsync", "0", "-"]
    result = subprocess.run(cmd, capture_output=True)
    expected = width * height
    if result.returncode != 0 or len(result.stdout) < expected:
        return None
    return np.frombuffer(result.stdout[:expected], dtype=np.uint8).reshape(height, width)


def main():
    if len(sys.argv) < 2:
        print("Usage: python check_source_captions.py <video_path> [n_samples]")
        sys.exit(1)
    video_path = Path(sys.argv[1])
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    video_stem = video_path.stem

    print(f"[cc-check] {video_path.name}")

    if _has_subtitle_stream(video_path):
        print("[cc-check] Has a SOFT subtitle stream in the container — that's the easy case, "
              "just don't include that stream (no crop needed for this). Checking for burned-in "
              "captions too, in case there are BOTH:")
    else:
        print("[cc-check] No soft-subtitle stream found — if this video has visible captions, "
              "they must be burned into the picture. Checking...")

    width, height = _probe_dimensions(video_path)
    timestamps = _sample_timestamps(video_path, video_stem, n_samples)
    print(f"[cc-check] sampling {len(timestamps)} frames at: "
          f"{', '.join(f'{t:.0f}s' for t in timestamps)}")

    sample_dir = settings.DATA_DIR / "_cc_samples" / video_stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    row_votes = np.zeros(height, dtype=int)
    n_valid = 0
    for i, t in enumerate(timestamps):
        gray = _extract_gray_frame(video_path, t, width, height)
        jpg_path = sample_dir / f"sample_{i:02d}_{t:.0f}s.jpg"
        subprocess.run(["ffmpeg", "-y", "-ss", str(t), "-i", str(video_path),
                         "-frames:v", "1", str(jpg_path)],
                        capture_output=True)
        if gray is None:
            print(f"[cc-check]   {t:.0f}s: frame extraction failed, skipping")
            continue
        n_valid += 1
        bright_count = (gray > BRIGHT_THRESHOLD).sum(axis=1)
        is_text_row = bright_count > MIN_BRIGHT_PIXELS_IN_ROW
        row_votes += is_text_row.astype(int)

    if n_valid == 0:
        print("[cc-check] ERROR: couldn't extract any frames — check the video path/duration.")
        sys.exit(1)

    agreement = row_votes / n_valid
    candidate_rows = np.where(agreement >= MIN_SAMPLE_AGREEMENT)[0]

    n_saved = len(list(sample_dir.glob("*.jpg")))
    print(f"\n[cc-check] {n_saved} sample frame(s) saved to {sample_dir} — eyeball them "
          f"yourself too, this heuristic can false-positive on a bright static object.")

    if len(candidate_rows) == 0:
        print("[cc-check] VERDICT: no consistent bright-text band found across samples — "
              "looks like this source has no burned-in captions. No SOURCE_CROP_OVERRIDES "
              "entry needed.")
        return

    band_top, band_bottom = candidate_rows.min(), candidate_rows.max()
    band_top_pct = band_top / height
    crop_bottom_pct = round(1.0 - band_top_pct + BAND_MARGIN_PCT, 2)
    print(f"[cc-check] VERDICT: consistent bright-text band found at y={band_top}-{band_bottom} "
          f"(out of {height}px height) across >= {int(MIN_SAMPLE_AGREEMENT*100)}% of samples — "
          f"likely burned-in captions.")
    print(f"\n[cc-check] Suggested config/settings.py entry (check the sample JPGs before "
          f"trusting this):\n")
    print(f'    "{video_stem}": {{')
    print(f'        "crop_bottom_pct": {crop_bottom_pct},   # band measured at '
          f'{band_top_pct*100:.1f}% down, +{BAND_MARGIN_PCT*100:.0f}% margin')
    print(f'    }},')


if __name__ == "__main__":
    main()
