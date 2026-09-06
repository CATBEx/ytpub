"""
One-off preview concat — NOT part of the pipeline.

Concatenates all of test_merge_fenrir_preview.py's per-beat Preview-N.mp4
files (Fenrir voice + matching clip, already muxed) into ONE single MP4, in
story order, so you can watch the whole episode's narration straight
through instead of clicking through 8 separate files.

Fast path: every Preview-N.mp4 was encoded with identical params (same
VIDEO_ENCODE_ARGS/AUDIO_ENCODE_ARGS for all of them), so this is a
stream-copy concat (ffmpeg -c copy, via the same _assemble_concat() helper
the real highlight-reel pipeline uses) — seconds, not minutes, no
re-encoding.

Run:
    python test_concat_fenrir_preview.py
Optionally pass a different video stem (defaults to the Boba Fett episode):
    python test_concat_fenrir_preview.py "Some Other Movie"

Requires test_merge_fenrir_preview.py to have already been run for this
stem (reads its preview_fenrir/Preview-N.mp4 output).

Output: data/explainer/{stem}/Full_Preview_Fenrir.mp4 — one file, all
beats back to back in story order.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings
from pipeline import stage7_assemble as hl  # reuse _assemble_concat(), _probe_duration

DEFAULT_STEM = "The Book of Boba Fett_S1E1"


def main():
    video_stem = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_STEM

    preview_dir = settings.EXPLAINER_DIR / video_stem / "preview_fenrir"
    if not preview_dir.exists():
        print(f"[concat-fenrir] ERROR: {preview_dir} not found — run "
              f"test_merge_fenrir_preview.py first.")
        sys.exit(1)

    numbered = []
    for p in preview_dir.glob("Preview-*.mp4"):
        m = re.match(r"Preview-(\d+)\.mp4$", p.name)
        if m:
            numbered.append((int(m.group(1)), p))
    if not numbered:
        print(f"[concat-fenrir] ERROR: no Preview-N.mp4 files in {preview_dir}.")
        sys.exit(1)

    numbered.sort(key=lambda t: t[0])  # story order
    clip_paths = [p for _, p in numbered]
    print(f"[concat-fenrir] {len(clip_paths)} beat(s), in order: "
          f"{', '.join(p.name for p in clip_paths)}")

    out_path = settings.EXPLAINER_DIR / video_stem / "Full_Preview_Fenrir.mp4"
    hl._assemble_concat(clip_paths, out_path)

    total_duration = hl._probe_duration(out_path)
    print(f"\n[concat-fenrir] done -> {out_path} ({total_duration:.1f}s total)")


if __name__ == "__main__":
    main()
