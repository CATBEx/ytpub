"""
Stage 8 — Cut clips + assemble the final review cut

    "cut"  (default) — every clip is cut with IDENTICAL encode settings
        (codec/resolution/pix_fmt/audio params), so assembly is a fast
        stream-copy concat (ffmpeg -f concat -c copy). Hard cuts, punchy
        pacing, and this is also what fixes the old ~31-minute assembly
        bottleneck from crossfading every single cut.
    "fade" — the previous xfade/acrossfade chain, still available if you
        want soft transitions back (slower, one full filter_complex pass).

If CAPTIONS_ENABLED, an SRT is generated from each clip's own `text` field
using the clips' REAL (ffprobed) durations — not nominal/assumed durations
— so captions can't drift across a long edit, then burned in with one final
ffmpeg pass. All text I/O here is explicit UTF-8 so Bengali/Hindi/etc.
dialogue survives on Windows regardless of the machine's local codepage.
Free, CPU-only.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from pipeline import framing  # 16:9 crop-to-fill + burned-in CC removal, 2026-08-26

VIDEO_ENCODE_ARGS = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
AUDIO_ENCODE_ARGS = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]


def _build_vf(crop_filter):
    """Combines the (optional) output-framing crop with the (optional)
    OUTPUT_RESOLUTION scale into ONE -vf chain — ffmpeg only honors the
    last -vf if given twice, so these can't be two separate flags."""
    parts = []
    if crop_filter:
        parts.append(crop_filter)
    if settings.OUTPUT_RESOLUTION:
        w, h = settings.OUTPUT_RESOLUTION.lower().split("x")
        parts.append(f"scale={w}:{h}")
    if not parts:
        return []
    return ["-vf", ",".join(parts)]


def _probe_duration(path: Path) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    return float(probe.stdout.strip())


def _cut_clip(video_path: Path, start: float, end: float, out_path: Path, crop_filter=None):
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", str(video_path),
        "-t", str(duration),
        *VIDEO_ENCODE_ARGS,
        *AUDIO_ENCODE_ARGS,
        *_build_vf(crop_filter),
        "-avoid_negative_ts", "make_zero",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg cut failed for {out_path.name}:\n{result.stderr}")


def _escape_concat_path(p: Path) -> str:
    # concat demuxer takes single-quoted paths; escape any embedded single quote.
    s = str(p.resolve()).replace("\\", "/")
    return s.replace("'", "'\\''")


def _assemble_concat(clip_paths, out_path: Path):
    """Fast path: all clips share identical encode params, so this is a
    stream-copy concat — seconds, not minutes, regardless of clip count."""
    if len(clip_paths) == 1:
        subprocess.run(["ffmpeg", "-y", "-i", str(clip_paths[0]), "-c", "copy", str(out_path)],
                        check=True, capture_output=True)
        return

    filelist_path = out_path.parent / f"{out_path.stem}_concat_list.txt"
    lines = [f"file '{_escape_concat_path(p)}'" for p in clip_paths]
    filelist_path.write_text("\n".join(lines), encoding="utf-8")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(filelist_path),
        "-c", "copy",
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat assembly failed:\n{result.stderr}")


def _assemble_with_fades(clip_paths, out_path: Path):
    """Legacy path: crossfade (xfade/acrossfade) between every pair. Slower —
    one big filter_complex re-encode — but softer transitions if you want them."""
    if len(clip_paths) == 1:
        subprocess.run(["ffmpeg", "-y", "-i", str(clip_paths[0]), "-c", "copy", str(out_path)],
                        check=True, capture_output=True)
        return

    inputs = []
    for p in clip_paths:
        inputs += ["-i", str(p)]

    td = settings.TRANSITION_DURATION
    filter_parts = []
    durations = [_probe_duration(p) for p in clip_paths]

    prev_label = "0:v"
    prev_audio = "0:a"
    running_offset = durations[0] - td
    for i in range(1, len(clip_paths)):
        v_out = f"v{i}"
        a_out = f"a{i}"
        filter_parts.append(
            f"[{prev_label}][{i}:v]xfade=transition=fade:duration={td}:offset={running_offset}[{v_out}]"
        )
        filter_parts.append(f"[{prev_audio}][{i}:a]acrossfade=d={td}[{a_out}]")
        prev_label, prev_audio = v_out, a_out
        running_offset += durations[i] - td

    filter_complex = ";".join(filter_parts)
    cmd = [
        "ffmpeg", "-y", *inputs,
        "-filter_complex", filter_complex,
        "-map", f"[{prev_label}]", "-map", f"[{prev_audio}]",
        *VIDEO_ENCODE_ARGS,
        *AUDIO_ENCODE_ARGS,
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg fade assembly failed:\n{result.stderr}")


def _srt_timestamp(seconds: float) -> str:
    ms_total = round(seconds * 1000)
    hh, rem = divmod(ms_total, 3_600_000)
    mm, rem = divmod(rem, 60_000)
    ss, ms = divmod(rem, 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


def _build_srt(clip_paths, segments, srt_path: Path):
    """Cumulative offsets from each clip's REAL (ffprobed) duration, not the
    nominal start/end from selection — keeps captions in sync across many
    clips even if individual cuts land a few frames off nominal length."""
    entries = []
    cursor = 0.0
    idx = 1
    for clip_path, seg in zip(clip_paths, segments):
        real_duration = _probe_duration(clip_path)
        text = (seg.get("text") or "").strip()
        if text:
            entries.append(
                f"{idx}\n{_srt_timestamp(cursor)} --> {_srt_timestamp(cursor + real_duration)}\n{text}\n"
            )
            idx += 1
        cursor += real_duration
    srt_path.write_text("\n".join(entries), encoding="utf-8")
    return srt_path


def _burn_in_captions(video_path: Path, srt_path: Path, out_path: Path):
    # ffmpeg's subtitles filter needs escaped path syntax on Windows (colon in
    # drive letters, backslashes) — normalize to forward slashes and escape colons.
    srt_arg = str(srt_path.resolve()).replace("\\", "/").replace(":", "\\:")
    force_style = f"FontName={settings.CAPTION_FONT},FontSize={settings.CAPTION_FONT_SIZE}"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"subtitles='{srt_arg}':force_style='{force_style}'",
        *VIDEO_ENCODE_ARGS,
        *AUDIO_ENCODE_ARGS,
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg caption burn-in failed:\n{result.stderr}")


def assemble(video_stem: str, source_video: Path) -> Path:
    selected_path = settings.ANALYSIS_DIR / f"{video_stem}_selected.json"
    data = json.loads(selected_path.read_text(encoding="utf-8"))
    segments = data["segments"]

    if not segments:
        raise RuntimeError("No segments were selected — nothing to assemble. "
                            "Check MIN_SCORE_FLOOR in config/settings.py.")

    clip_dir = settings.CLIPS_DIR / video_stem
    clip_dir.mkdir(parents=True, exist_ok=True)

    src_width, src_height = framing.probe_dimensions(source_video)
    crop_filter = framing.compute_crop_filter(src_width, src_height, video_stem)
    if crop_filter:
        has_cc_override = video_stem in settings.SOURCE_CROP_OVERRIDES
        print(f"[stage8] output framing: {crop_filter} (16:9 crop-to-fill"
              + (", includes burned-in CC removal" if has_cc_override else "") + ")")

    clip_paths = []
    for i, seg in enumerate(segments):
        clip_path = clip_dir / f"clip_{i:03d}_{seg['start']:.1f}s.mp4"
        print(f"[stage8] cutting clip {i+1}/{len(segments)}: "
              f"{seg['start']:.1f}s-{seg['end']:.1f}s (score {seg.get('overall_score', '?')})")
        _cut_clip(source_video, seg["start"], seg["end"], clip_path, crop_filter)
        clip_paths.append(clip_path)

    assembled_path = settings.OUTPUT_DIR / f"{video_stem}_assembled.mp4"
    print(f"[stage8] assembling {len(clip_paths)} clips "
          f"(transition={settings.TRANSITION_TYPE})...")
    if settings.TRANSITION_TYPE == "fade":
        _assemble_with_fades(clip_paths, assembled_path)
    else:
        _assemble_concat(clip_paths, assembled_path)

    final_path = settings.OUTPUT_DIR / f"{video_stem}_review_cut.mp4"
    if settings.CAPTIONS_ENABLED:
        srt_path = settings.OUTPUT_DIR / f"{video_stem}.srt"
        print("[stage8] building captions from selected segments' dialogue...")
        _build_srt(clip_paths, segments, srt_path)
        print(f"[stage8] burning in captions ({settings.CAPTION_FONT}, "
              f"{settings.CAPTION_FONT_SIZE}pt)...")
        _burn_in_captions(assembled_path, srt_path, final_path)
    else:
        assembled_path.replace(final_path)

    print(f"[stage8] done -> {final_path} (~{data['total_duration_sec']/60:.1f} min)")
    return final_path


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python stage7_assemble.py <video_stem> <source_video.mp4>")
        sys.exit(1)
    assemble(sys.argv[1], Path(sys.argv[2]))
