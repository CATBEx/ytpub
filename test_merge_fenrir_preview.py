"""
One-off preview merge — NOT part of the pipeline.

Muxes each beat's Fenrir voice-N.mp3 (from test_voice_fenrir_full_episode.py)
together with its video-only Clip-N.mp4 (from the real pipeline run), so
you can actually watch+listen to each beat instead of two separate files.
Purely a preview/QA tool — the real pipeline's per-beat Clip-N.mp4/
voice-N.mp3-with-no-merge design is unchanged; this doesn't touch or
replace it, and writes to its own folder.

Duration mismatch handling: Clip-N.mp4 was originally retimed (Stage 6) to
match the DIFFERENT (Charon) voice-N.mp3's spoken duration for that beat's
text — Fenrir reading the same text will almost certainly run a different
length. To avoid ever cutting off narration mid-sentence:
  - if Fenrir's audio is LONGER than the clip, the clip's last frame is
    frozen (held) to cover the difference
  - if the clip is LONGER than Fenrir's audio, silence is padded onto the
    end of the audio so the clip isn't cut short
Either way, output duration = max(clip, voice) for every beat — nothing
gets cut off, whichever side is longer.

Run:
    python test_merge_fenrir_preview.py
Optionally pass a different video stem (defaults to the Boba Fett episode):
    python test_merge_fenrir_preview.py "Some Other Movie"

Output: data/explainer/{stem}/preview_fenrir/Preview-N.mp4 — one merged
file per beat that has both a Clip-N.mp4 and a voice-N.mp3.
"""
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings
from pipeline import stage7_assemble as hl  # reuse VIDEO_ENCODE_ARGS/AUDIO_ENCODE_ARGS, duration probing

DEFAULT_STEM = "The Book of Boba Fett_S1E1"


def _find_beat_pairs(clips_dir: Path, voices_dir: Path):
    """Return sorted [(n, clip_path, voice_path), ...] for every N present
    in BOTH folders — skips (with a warning) any beat missing from either
    side."""
    clip_ns = {}
    for p in clips_dir.glob("Clip-*.mp4"):
        m = re.match(r"Clip-(\d+)\.mp4$", p.name)
        if m:
            clip_ns[int(m.group(1))] = p

    voice_ns = {}
    for p in voices_dir.glob("voice-*.mp3"):
        m = re.match(r"voice-(\d+)\.mp3$", p.name)
        if m:
            voice_ns[int(m.group(1))] = p

    pairs = []
    for n in sorted(set(clip_ns) | set(voice_ns)):
        clip_path, voice_path = clip_ns.get(n), voice_ns.get(n)
        if clip_path is None:
            print(f"  [warn] beat {n}: voice-{n}.mp3 exists but no Clip-{n}.mp4 — skipping")
            continue
        if voice_path is None:
            print(f"  [warn] beat {n}: Clip-{n}.mp4 exists but no voice-{n}.mp3 — skipping")
            continue
        pairs.append((n, clip_path, voice_path))
    return pairs


def _merge_one(clip_path: Path, voice_path: Path, out_path: Path):
    clip_duration = hl._probe_duration(clip_path)
    voice_duration = hl._probe_duration(voice_path)
    target_duration = max(clip_duration, voice_duration)
    video_pad = max(0.0, target_duration - clip_duration)

    filter_parts = []
    if video_pad > 0.02:
        filter_parts.append(f"[0:v]tpad=stop_mode=clone:stop_duration={video_pad}[v]")
        video_map = "[v]"
    else:
        video_map = "0:v"
    filter_parts.append(f"[1:a]apad=whole_dur={target_duration}[a]")

    cmd = ["ffmpeg", "-y", "-i", str(clip_path), "-i", str(voice_path),
           "-filter_complex", ";".join(filter_parts),
           "-map", video_map, "-map", "[a]",
           *hl.VIDEO_ENCODE_ARGS, *hl.AUDIO_ENCODE_ARGS,
           "-t", str(target_duration), str(out_path)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg merge failed for {out_path.name}:\n{result.stderr}")
    return clip_duration, voice_duration, target_duration


def main():
    video_stem = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_STEM

    clips_dir = settings.EXPLAINER_DIR / video_stem / "clips"
    voices_dir = settings.EXPLAINER_DIR / video_stem / "voices_fenrir"
    if not clips_dir.exists():
        print(f"[merge-fenrir] ERROR: {clips_dir} not found.")
        sys.exit(1)
    if not voices_dir.exists():
        print(f"[merge-fenrir] ERROR: {voices_dir} not found — run "
              f"test_voice_fenrir_full_episode.py first.")
        sys.exit(1)

    pairs = _find_beat_pairs(clips_dir, voices_dir)
    if not pairs:
        print("[merge-fenrir] ERROR: no matching Clip-N.mp4 / voice-N.mp3 pairs found.")
        sys.exit(1)

    out_dir = settings.EXPLAINER_DIR / video_stem / "preview_fenrir"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[merge-fenrir] {len(pairs)} beat(s) to merge -> {out_dir}")

    for n, clip_path, voice_path in pairs:
        out_path = out_dir / f"Preview-{n}.mp4"
        print(f"\n[merge-fenrir] beat {n}: {clip_path.name} + {voice_path.name}")
        clip_dur, voice_dur, target_dur = _merge_one(clip_path, voice_path, out_path)
        note = ""
        if abs(clip_dur - voice_dur) > 0.5:
            longer = "voice" if voice_dur > clip_dur else "clip"
            note = f" (mismatch {abs(clip_dur - voice_dur):.1f}s, {longer} was longer — padded, nothing cut off)"
        print(f"[merge-fenrir]   clip={clip_dur:.1f}s voice={voice_dur:.1f}s -> {target_dur:.1f}s{note}")
        print(f"[merge-fenrir]   -> {out_path}")

    print(f"\n[merge-fenrir] done — {len(pairs)} merged preview(s) in {out_dir}")


if __name__ == "__main__":
    main()
