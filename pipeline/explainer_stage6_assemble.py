"""
Explainer Stage 6 — Per-beat clip retiming
Cuts the exact cited footage for each beat, then retimes it (a speed change
via ffmpeg `setpts`, or extend/crop outside a tasteful range) so its
duration matches that beat's narration (voice-N.mp3, from Stage 5) real
spoken length. Also applies output framing (16:9 crop-to-fill + burned-in
CC removal, pipeline/framing.py) and burns in "Summary CC" — captions built
from the beat's own narration text, added 2026-08-26 (reverses the
original redesign's "no captions" decision, by explicit user request; the
Explainer-specific `EXPLAINER_CAPTIONS_ENABLED` setting is the revert
switch if that's wanted again).

Output is VIDEO-ONLY (no audio track) — the narration audio still lives in
its own voice-N.mp3, there is NO mixing or concatenation here. You assemble
the final cut yourself from the Clip-N.mp4 / voice-N.mp3 pairs this stage
produces, one pair per beat.

Per beat, in order of preference (closest to "the exact cited moment" first):
  1. Tasteful speed change (STRETCH_MIN_RATIO..STRETCH_MAX_RATIO) on the
     exact cited footage — the common case.
  2. If narration needs much more time than a tasteful slow-down can cover,
     extend real surrounding footage first (up to CLIP_EXTENSION_MAX_SECONDS),
     then cap the slow-down and freeze-hold any still-missing remainder.
  3. If the cited footage is much longer than narration needs even at max
     speed-up, center-crop it down before speeding up.

Reuses VIDEO_ENCODE_ARGS/_probe_duration/_srt_timestamp from
stage7_assemble.py and the shared beat_envelope() helper from
explainer_common.py.

CHANGED same day (later still): each beat's `scene_type` and `beat_role`
(the hook/twist/highlight/setup/resolution/body retention design — see
explainer_stage3_summary.py) are carried through into `_final_beats.json`.
Both are set upstream (Stage 3) and already survive Stage 4/5 automatically
(those stages mutate beat dicts in place); this stage rebuilds its output
dict from scratch, so both fields need this explicit pass-through.

FIXED 2026-08-27 (real latent bug, found via code review before it caused
data loss, multi-language redesign): `clip_dir` had no language component
(`data/explainer/{movie}/clips/`) — running a second language for the same
movie would silently ffmpeg-`-y`-overwrite the first language's Clip-N.mp4
files, since retiming (and therefore clip content/length) is genuinely
per-language (target_duration is derived directly from that language's own
spoken voice_duration). Clips, voices and final_beats.json are now all
under this language's own subfolder — data/explainer/{movie}/{language}/.
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from pipeline import stage7_assemble as hl  # reuse VIDEO_ENCODE_ARGS, duration probing, _srt_timestamp
from pipeline import framing  # 16:9 crop-to-fill + burned-in CC removal, 2026-08-26
from pipeline.explainer_common import beat_envelope


def _widen_to_min_snippet(env_start: float, env_end: float, video_duration: float):
    """An uncited (zero-width) or very short beat still needs a real snippet
    of footage to cut and retime, not a knife-edge point."""
    if env_end - env_start >= settings.MIN_CLIP_SNIPPET_SECONDS:
        return env_start, env_end
    mid = (env_start + env_end) / 2
    half = settings.MIN_CLIP_SNIPPET_SECONDS / 2
    start = max(0.0, mid - half)
    end = min(video_duration, mid + half)
    if end - start < settings.MIN_CLIP_SNIPPET_SECONDS:
        if start <= 0.0:
            end = min(video_duration, settings.MIN_CLIP_SNIPPET_SECONDS)
        else:
            start = max(0.0, end - settings.MIN_CLIP_SNIPPET_SECONDS)
    return start, end


def _plan_clip(env_start: float, env_end: float, target_duration: float, video_duration: float):
    """Decide how to turn footage window [env_start, env_end] into exactly
    target_duration seconds of output.

    Returns (clip_start, clip_duration, stretch_ratio, freeze_needed):
      - cut [clip_start, clip_start + clip_duration] from the source
      - retime it by stretch_ratio (setpts={ratio}*PTS)
      - then hold the last frame for freeze_needed more seconds if still short
    """
    env_start, env_end = _widen_to_min_snippet(env_start, env_end, video_duration)
    base_duration = max(0.1, env_end - env_start)
    ratio = target_duration / base_duration

    if settings.STRETCH_MIN_RATIO <= ratio <= settings.STRETCH_MAX_RATIO:
        return env_start, base_duration, ratio, 0.0

    if ratio > settings.STRETCH_MAX_RATIO:
        extension_needed = (target_duration / settings.STRETCH_MAX_RATIO) - base_duration
        extend = min(max(0.0, extension_needed), settings.CLIP_EXTENSION_MAX_SECONDS)
        new_start = max(0.0, env_start - extend / 2)
        new_end = min(video_duration, env_end + extend / 2)
        new_duration = max(0.1, new_end - new_start)
        new_ratio = target_duration / new_duration
        stretch_ratio = min(new_ratio, settings.STRETCH_MAX_RATIO)
        stretched_duration = new_duration * stretch_ratio
        freeze_needed = max(0.0, target_duration - stretched_duration)
        return new_start, new_duration, stretch_ratio, freeze_needed

    # ratio < STRETCH_MIN_RATIO: footage is much longer than needed — center-crop
    # down before speeding up, rather than speeding up past a tasteful limit
    max_usable = target_duration / settings.STRETCH_MIN_RATIO
    extra = base_duration - max_usable
    clip_start = env_start + extra / 2
    return max(0.0, clip_start), max_usable, settings.STRETCH_MIN_RATIO, 0.0


def _cut_stretch_clip(source_video: Path, clip_start: float, clip_duration: float,
                       stretch_ratio: float, freeze_needed: float, target_duration: float,
                       out_path: Path, crop_filter=None, caption_text: str = None):
    """Video-only output — no audio track, no mixing (Stage 6 redesign).
    Applies output framing (crop_filter — 16:9 crop-to-fill + burned-in CC
    removal) and, if caption_text is given and EXPLAINER_CAPTIONS_ENABLED,
    burns in a "Summary CC" caption (the beat's own narration text, one cue
    spanning the whole clip) — all in a single ffmpeg pass."""
    vf_parts = []
    if crop_filter:
        vf_parts.append(crop_filter)
    if abs(stretch_ratio - 1.0) > 1e-3:
        vf_parts.append(f"setpts={stretch_ratio}*PTS")
    if freeze_needed > 0.02:
        vf_parts.append(f"tpad=stop_mode=clone:stop_duration={freeze_needed}")

    srt_path = None
    if caption_text and caption_text.strip() and settings.EXPLAINER_CAPTIONS_ENABLED:
        srt_path = out_path.with_suffix(".srt")
        ts_end = hl._srt_timestamp(target_duration)
        srt_path.write_text(f"1\n00:00:00,000 --> {ts_end}\n{caption_text.strip()}\n", encoding="utf-8")
        srt_arg = str(srt_path.resolve()).replace("\\", "/").replace(":", "\\:")
        force_style = f"FontName={settings.CAPTION_FONT},FontSize={settings.CAPTION_FONT_SIZE}"
        vf_parts.append(f"subtitles='{srt_arg}':force_style='{force_style}'")

    cmd = ["ffmpeg", "-y", "-ss", str(clip_start), "-t", str(clip_duration), "-i", str(source_video)]
    if vf_parts:
        cmd += ["-vf", ",".join(vf_parts)]
    cmd += ["-an", *hl.VIDEO_ENCODE_ARGS, "-t", str(target_duration), str(out_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg beat cut failed for {out_path.name}:\n{result.stderr}")
    finally:
        if srt_path is not None:
            srt_path.unlink(missing_ok=True)  # scratch file for the burn-in pass only


def retime_clips(video_stem: str, source_video: Path, language: str = "en") -> list:
    if language not in settings.EXPLAINER_LANGUAGE_NAMES:
        raise RuntimeError(f"Unsupported language {language!r} — must be one of "
                            f"{list(settings.EXPLAINER_LANGUAGE_NAMES)}.")

    voiced_path = settings.EXPLAINER_DIR / video_stem / language / "voiced.json"
    beats = json.loads(voiced_path.read_text(encoding="utf-8"))["beats"]
    if not beats:
        raise RuntimeError("No beats — nothing to retime.")

    video_duration = hl._probe_duration(source_video)
    # Per-language clip folder — data/explainer/{movie}/{language}/clips/. See the FIXED
    # 2026-08-27 note in the module docstring for why this must NOT be shared across languages.
    clip_dir = settings.EXPLAINER_DIR / video_stem / language / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)

    src_width, src_height = framing.probe_dimensions(source_video)
    crop_filter = framing.compute_crop_filter(src_width, src_height, video_stem)
    if crop_filter:
        has_cc_override = video_stem in settings.SOURCE_CROP_OVERRIDES
        print(f"[explainer-retime] output framing: {crop_filter} (16:9 crop-to-fill"
              + (", includes burned-in CC removal" if has_cc_override else "") + ")")
    if settings.EXPLAINER_CAPTIONS_ENABLED:
        print("[explainer-retime] burning in Summary CC (narration text) per beat")

    results = []
    for i, beat in enumerate(beats):
        voice_duration = beat.get("voice_duration", 0.0)
        n = beat["index"] + 1  # 1-indexed
        target_duration = max(1.0, voice_duration + 2 * settings.BEAT_PAD_SECONDS) \
            if voice_duration else 2.0  # no narration for this beat: brief hold

        env_start, env_end = beat_envelope(beats, i)
        clip_start, clip_duration, stretch_ratio, freeze_needed = _plan_clip(
            env_start, env_end, target_duration, video_duration
        )

        clip_path = clip_dir / f"Clip-{n}.mp4"
        print(f"[explainer-retime] beat {i + 1}/{len(beats)}: "
              f"{clip_start:.1f}s +{clip_duration:.1f}s x{stretch_ratio:.2f} "
              f"(freeze={freeze_needed:.1f}s) -> {target_duration:.1f}s -> {clip_path.name}")
        _cut_stretch_clip(source_video, clip_start, clip_duration, stretch_ratio,
                           freeze_needed, target_duration, clip_path,
                           crop_filter=crop_filter, caption_text=beat.get("narration", ""))

        results.append({
            "index": beat["index"],
            "narration": beat.get("narration", ""),
            "scene_type": beat.get("scene_type"),
            "beat_role": beat.get("beat_role"),
            "clip_path": str(clip_path),
            "voice_path": beat.get("voice_path"),
        })

    out_dir = settings.EXPLAINER_DIR / video_stem / language
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "final_beats.json"
    out_path.write_text(json.dumps({"beats": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[explainer-retime] {len(results)} clip/voice pairs ready -> {out_path}")
    return results


def retime_intro_clip(video_stem: str, source_video: Path, language: str = "en") -> dict:
    """Stage-6 half of the intro feature (see pipeline/explainer_intro.py for the design
    rationale). Deliberately NOT folded into retime_clips() above: the intro is a single,
    un-numbered output, not one more entry in the beats list, and it has no neighboring
    beats to widen its citation window against — beat_envelope() doesn't apply here, so
    the intro's own [start, end] (already resolved by explainer_intro.resolve_intro_citation())
    is passed straight into _plan_clip(), which handles the zero-width/uncited case itself
    via _widen_to_min_snippet(). Same target_duration formula every other beat uses, so the
    intro's real length is driven by its own real synthesized voice_duration, same as always.

    Output: Intro.mp4, written into the SAME per-language clips/ folder retime_clips() uses
    (per the user's explicit choice — no separate intro/ subfolder), not Clip-N.mp4-numbered.
    Rewrites intro.json in place with a new clip_path field, completing that file's
    progressively-enriched narration -> voice_duration/voice_path -> clip_path lifecycle."""
    if language not in settings.EXPLAINER_LANGUAGE_NAMES:
        raise RuntimeError(f"Unsupported language {language!r} — must be one of "
                            f"{list(settings.EXPLAINER_LANGUAGE_NAMES)}.")

    intro_path = settings.EXPLAINER_DIR / video_stem / language / "intro.json"
    intro = json.loads(intro_path.read_text(encoding="utf-8"))

    video_duration = hl._probe_duration(source_video)
    clip_dir = settings.EXPLAINER_DIR / video_stem / language / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)

    src_width, src_height = framing.probe_dimensions(source_video)
    crop_filter = framing.compute_crop_filter(src_width, src_height, video_stem)

    voice_duration = intro.get("voice_duration", 0.0)
    target_duration = max(1.0, voice_duration + 2 * settings.BEAT_PAD_SECONDS) \
        if voice_duration else 2.0

    env_start = intro.get("start", 0.0) or 0.0
    env_end = intro.get("end", 0.0) or 0.0
    clip_start, clip_duration, stretch_ratio, freeze_needed = _plan_clip(
        env_start, env_end, target_duration, video_duration
    )

    clip_path = clip_dir / "Intro.mp4"
    print(f"[explainer-retime] intro: {clip_start:.1f}s +{clip_duration:.1f}s x{stretch_ratio:.2f} "
          f"(freeze={freeze_needed:.1f}s) -> {target_duration:.1f}s -> {clip_path.name}")
    _cut_stretch_clip(source_video, clip_start, clip_duration, stretch_ratio,
                       freeze_needed, target_duration, clip_path,
                       crop_filter=crop_filter, caption_text=intro.get("narration", ""))

    intro["clip_path"] = str(clip_path)
    intro_path.write_text(json.dumps(intro, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[explainer-retime] intro clip/voice pair ready -> {intro_path}")
    return intro


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: python explainer_stage6_assemble.py <video_stem> <source_video.mp4> [language]")
        sys.exit(1)
    results = retime_clips(sys.argv[1], Path(sys.argv[2]), sys.argv[3] if len(sys.argv) == 4 else "en")
    for r in results:
        print(f"beat {r['index']}: {r['clip_path']} / {r['voice_path']}")
