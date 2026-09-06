"""
Sandbox test for output framing (16:9 crop-to-fill + burned-in CC removal +
Explainer "Summary CC" captions), added 2026-08-26.

Two layers:
  1. Pure math unit tests for framing.compute_crop_filter() against the
     REAL measured numbers from the two real source videos (Boba Fett's
     1920x804 frame + its confirmed 0.19 crop_bottom_pct override, and "A
     Man in Full"'s already-16:9 1920x1080 no-op case) plus synthetic edge
     cases (a 4:3 source, a portrait-ish source).
  2. Real ffmpeg round-trips: cut a synthetic clip through BOTH
     stage7_assemble._cut_clip (highlight-reel) and
     explainer_stage6_assemble._cut_stretch_clip (explainer, plus its new
     Summary CC caption burn-in) and ffprobe the real output to confirm the
     dimensions actually landed on 16:9 — not just that compute_crop_filter
     returns the right numbers on paper. The caption burn-in is verified by
     re-using the same bright-pixel-row scan check_source_captions.py uses
     to detect burned-in text, confirming real text pixels actually landed
     in the output frame (not just that ffmpeg didn't error).

Run: python test_output_framing.py
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings
from pipeline import framing
from pipeline import stage7_assemble as hl
from pipeline import explainer_stage6_assemble as stage6

TEST_DIR = settings.DATA_DIR / "framing_sandbox_test"


def _run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{result.stderr}")


def make_synthetic_video(path: Path, width=320, height=240, duration=6):
    _run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=duration={duration}:size={width}x{height}:rate=15",
        "-f", "lavfi", "-i", f"sine=frequency=220:duration={duration}",
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
        str(path),
    ])


def make_plain_video(path: Path, width=320, height=240, duration=6):
    """A flat mid-gray source with NO inherent bright content — testsrc's own
    gradient/color-bar pattern is bright enough in places to false-positive
    the bright-pixel-row caption heuristic, so the caption on/off checks
    need a clean baseline instead."""
    _run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=gray:size={width}x{height}:duration={duration}:rate=15",
        "-c:v", "libx264", "-preset", "ultrafast",
        str(path),
    ])


def _ffprobe_dims(path: Path) -> tuple:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        capture_output=True, text=True,
    )
    w, h = result.stdout.strip().split("x")
    return int(w), int(h)


def _has_audio_stream(path: Path) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def _has_bright_text_band(path: Path) -> bool:
    """Same heuristic as check_source_captions.py: does this frame have a
    row with a lot of near-white pixels (i.e. burned-in text)?"""
    w, h = _ffprobe_dims(path)
    cmd = ["ffmpeg", "-y", "-i", str(path), "-frames:v", "1",
           "-f", "rawvideo", "-pix_fmt", "gray", "-vsync", "0", "-"]
    result = subprocess.run(cmd, capture_output=True)
    expected = w * h
    if result.returncode != 0 or len(result.stdout) < expected:
        return False
    gray = np.frombuffer(result.stdout[:expected], dtype=np.uint8).reshape(h, w)
    bright_count = (gray > 225).sum(axis=1)
    return bool((bright_count > 15).any())


def test_compute_crop_filter_math():
    print("\n[test] === framing.compute_crop_filter — pure math ===")

    # Real case #1: Boba Fett, 1920x804, crop_bottom_pct=0.19 (from SOURCE_CROP_OVERRIDES)
    assert "The Book of Boba Fett_S1E1" in settings.SOURCE_CROP_OVERRIDES, (
        "this test assumes the real Boba Fett override is still in settings.py"
    )
    f = framing.compute_crop_filter(1920, 804, "The Book of Boba Fett_S1E1")
    assert f == "crop=1156:650:382:0", f"unexpected crop filter: {f}"
    # sanity: the kept region's top row (650) must be well above the measured
    # caption band's top row (677/804 = 84.2%) — i.e. captions are excluded
    assert 650 < 677, "crop must exclude the measured caption band"
    print(f"[test]   Boba Fett (1920x804, crop_bottom_pct=0.19) -> {f} OK (excludes y>=677 caption band)")

    # Real case #2: A Man in Full, already exactly 1920x1080 (16:9), no override -> no-op
    assert "A Man in Full - S01E01 - Saddlebags WEBRip-1080p" not in settings.SOURCE_CROP_OVERRIDES
    f2 = framing.compute_crop_filter(1920, 1080, "A Man in Full - S01E01 - Saddlebags WEBRip-1080p")
    assert f2 is None, f"already-16:9 source with no override should need no crop, got {f2}"
    print("[test]   A Man in Full (1920x1080, no override) -> None (no-op) OK")

    # Synthetic: 4:3 source (960x720) — narrower than 16:9, so crop-to-fill must
    # crop the HEIGHT (top/bottom) and keep the full width, not the other way round.
    f3 = framing.compute_crop_filter(960, 720, "unknown_4_3_source")
    assert f3 is not None
    w, h, x, y = (int(v) for v in f3.replace("crop=", "").replace(":", " ").split())
    assert w == 960 and h == 540 and x == 0 and y == 90 and abs(w / h - 16 / 9) < 0.01
    print(f"[test]   4:3 source (960x720) -> {f3} OK (crops top/bottom, keeps full width)")

    # Synthetic: portrait-ish source (720x1280) — even narrower, same direction (crop height)
    f4 = framing.compute_crop_filter(720, 1280, "unknown_portrait_source")
    w4, h4, x4, y4 = (int(v) for v in f4.replace("crop=", "").replace(":", " ").split())
    assert w4 == 720 and abs(w4 / h4 - 16 / 9) < 0.01 and x4 == 0 and y4 > 0
    print(f"[test]   portrait source (720x1280) -> {f4} OK (crops top/bottom, keeps full width)")

    print("[test]   compute_crop_filter OK across real + synthetic cases")


def test_probe_dimensions(video_path: Path, expected_w: int, expected_h: int):
    print("\n[test] === framing.probe_dimensions ===")
    w, h = framing.probe_dimensions(video_path)
    assert (w, h) == (expected_w, expected_h), f"expected {(expected_w, expected_h)}, got {(w, h)}"
    print(f"[test]   probe_dimensions({video_path.name}) -> {(w, h)} OK")


def test_highlight_reel_crop_real_ffmpeg(video_path: Path):
    print("\n[test] === stage7_assemble._cut_clip real ffmpeg crop ===")
    out_path = TEST_DIR / "hl_cropped.mp4"
    # 320x240 (4:3) source -> synthesize the same crop-to-16:9 math with no CC override
    crop_filter = framing.compute_crop_filter(320, 240, "unknown_stem_no_override")
    assert crop_filter is not None
    hl._cut_clip(video_path, 0.0, 2.0, out_path, crop_filter)
    w, h = _ffprobe_dims(out_path)
    assert abs(w / h - 16 / 9) < 0.02, f"output not 16:9: {w}x{h}"
    print(f"[test]   real ffmpeg crop -> {w}x{h} (16:9) OK")


def test_explainer_crop_and_caption_real_ffmpeg(video_path: Path):
    print("\n[test] === explainer_stage6_assemble._cut_stretch_clip real ffmpeg crop+caption ===")
    # use a flat-gray source (not the testsrc pattern) for the caption on/off
    # checks -- testsrc's own gradient bar is bright enough to false-positive
    # the bright-pixel-row heuristic even with no burned-in text at all.
    plain_video = video_path.parent / "plain_320x240.mp4"
    make_plain_video(plain_video, width=320, height=240, duration=3)
    video_path = plain_video
    crop_filter = framing.compute_crop_filter(320, 240, "unknown_stem_no_override")

    # without caption
    out_no_caption = TEST_DIR / "beat_no_caption.mp4"
    stage6._cut_stretch_clip(video_path, 0.0, 2.0, 1.0, 0.0, 2.0, out_no_caption,
                              crop_filter=crop_filter, caption_text="")
    w, h = _ffprobe_dims(out_no_caption)
    assert abs(w / h - 16 / 9) < 0.02, f"output not 16:9: {w}x{h}"
    assert not _has_audio_stream(out_no_caption), "explainer clips must stay video-only"
    assert not _has_bright_text_band(out_no_caption), "no caption_text was given -- should be no burned-in text"
    print(f"[test]   no caption_text -> {w}x{h}, video-only, no burned-in text OK")

    # with caption ("Summary CC") -- feature defaults OFF since 2026-08-26 (reversed by
    # explicit request: Explainer clips are silent/caption-free again), but the on-path
    # is still real code (settings.EXPLAINER_CAPTIONS_ENABLED can be flipped back on) so
    # it stays covered here by forcing it on for just this assertion.
    original_captions_enabled = settings.EXPLAINER_CAPTIONS_ENABLED
    settings.EXPLAINER_CAPTIONS_ENABLED = True
    try:
        out_with_caption = TEST_DIR / "beat_with_caption.mp4"
        stage6._cut_stretch_clip(video_path, 0.0, 2.0, 1.0, 0.0, 2.0, out_with_caption,
                                  crop_filter=crop_filter,
                                  caption_text="This is the beat's own narration text.")
        w2, h2 = _ffprobe_dims(out_with_caption)
        assert abs(w2 / h2 - 16 / 9) < 0.02, f"output not 16:9: {w2}x{h2}"
        assert not _has_audio_stream(out_with_caption)
        assert _has_bright_text_band(out_with_caption), (
            "caption_text was given and EXPLAINER_CAPTIONS_ENABLED=True -- expected burned-in text pixels"
        )
        print(f"[test]   with caption_text -> {w2}x{h2}, video-only, burned-in text confirmed OK")
    finally:
        settings.EXPLAINER_CAPTIONS_ENABLED = original_captions_enabled

    # respects EXPLAINER_CAPTIONS_ENABLED=False
    original = settings.EXPLAINER_CAPTIONS_ENABLED
    settings.EXPLAINER_CAPTIONS_ENABLED = False
    try:
        out_disabled = TEST_DIR / "beat_captions_disabled.mp4"
        stage6._cut_stretch_clip(video_path, 0.0, 2.0, 1.0, 0.0, 2.0, out_disabled,
                                  crop_filter=crop_filter,
                                  caption_text="This should NOT appear.")
        assert not _has_bright_text_band(out_disabled), (
            "EXPLAINER_CAPTIONS_ENABLED=False must suppress the caption burn-in"
        )
        print("[test]   EXPLAINER_CAPTIONS_ENABLED=False correctly suppresses burn-in OK")
    finally:
        settings.EXPLAINER_CAPTIONS_ENABLED = original

    # scratch SRT files must not be left behind
    leftover_srts = list(TEST_DIR.glob("*.srt"))
    assert not leftover_srts, f"scratch SRT files leaked: {leftover_srts}"
    print("[test]   no scratch .srt files left behind OK")


def main():
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    video_path = TEST_DIR / "synthetic_320x240.mp4"
    print("[test] building synthetic 320x240 (4:3) video...")
    make_synthetic_video(video_path, width=320, height=240, duration=6)

    test_compute_crop_filter_math()
    test_probe_dimensions(video_path, 320, 240)
    test_highlight_reel_crop_real_ffmpeg(video_path)
    test_explainer_crop_and_caption_real_ffmpeg(video_path)

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
