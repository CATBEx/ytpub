"""
Sandbox integration test for the redesigned Explainer pipeline (Stages 3-6,
2026-08-26 redesign): summary generation, clip matching, per-beat TTS, and
per-beat clip retiming (NO merged final video, video-only clips).

No real GCP/Gemini/TTS network calls (this sandbox has no ADC credentials
anyway) — Gemini and TTS clients are mocked/monkeypatched. Everything else
(ffmpeg, librosa, OpenCV signal extraction, real MP3/MP4 rendering) is real.

Two layers of testing:
  1. Direct unit tests of each stage's own logic (prompt building, schema
     config construction, index resolution, intensity computation, retiming
     math) — fast, precise.
  2. A full pipeline run through the actual top-level functions
     (generate_summary -> match_clips -> synthesize_narration -> retime_clips)
     with google.genai.Client / texttospeech.TextToSpeechClient patched via
     unittest.mock, verifying the real file-based handoff between stages
     (the *_summary.json -> *_matched.json -> *_voiced.json -> *_final_beats.json
     chain) actually works end to end.

Run: python test_explainer_pipeline.py
"""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings
import google.cloud.texttospeech  # noqa: F401 — import eagerly so `patch("google.cloud.texttospeech...")`
                                   # can resolve the attribute chain (namespace packages don't
                                   # register submodules on the parent until something imports them)
from pipeline import stage7_assemble as hl
from pipeline.explainer_common import beat_envelope
from pipeline import explainer_stage3_summary as stage3
from pipeline import explainer_stage4_match as stage4
from pipeline import explainer_intro as intro_mod
from pipeline import explainer_stage5_tts as stage5
from pipeline import explainer_stage6_assemble as stage6
from pipeline import audio_events
from pipeline import subtitles

TEST_DIR = settings.DATA_DIR / "explainer_pipeline_sandbox_test"
STEM = "sandbox_test_movie"


def _run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(cmd)}\n{result.stderr}")


def make_synthetic_video(path: Path, duration=40):
    _run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=320x240:rate=15",
        "-f", "lavfi", "-i", f"sine=frequency=220:duration={duration // 2}",
        "-f", "lavfi", "-i", f"sine=frequency=880:duration={duration - duration // 2}",
        "-filter_complex", "[1:a][2:a]concat=n=2:v=0:a=1[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
        str(path),
    ])


def make_silence_mp3(path: Path, duration: float):
    _run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
        "-t", str(duration), "-c:a", "libmp3lame", str(path),
    ])


def make_quiet_loud_quiet_audio(path: Path):
    """0-2s quiet, 2-12s LOUD, 12-14s quiet, 14-24s quiet -- exactly one sustained-loud
    dialogue-free gap (2-12s) and one quiet dialogue-free gap (14-24s), for
    audio_events.detect_audio_events() testing against real ffmpeg audio + real librosa."""
    _run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "sine=frequency=100:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
        "-f", "lavfi", "-i", "sine=frequency=100:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=100:duration=10",
        "-filter_complex",
        "[0:a]volume=0.02[a0];[1:a]volume=1.0[a1];[2:a]volume=0.02[a2];[3:a]volume=0.02[a3];"
        "[a0][a1][a2][a3]concat=n=4:v=0:a=1[aout]",
        "-map", "[aout]", "-ar", "16000", "-ac", "1",
        str(path),
    ])


def has_audio_stream(path: Path) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


# ---------------------------------------------------------------------------
# Stage 3 — summary generation: prompt building + mocked schema round-trip
# ---------------------------------------------------------------------------
def test_stage3_unit():
    print("\n[test] === Stage 3 unit: summary generation ===")

    segments = [
        {"start": 0.0, "end": 5.0, "text": "Rahul, where have you been?"},
        {"start": 5.5, "end": 9.0, "text": "I was at the old house."},
    ]
    prompt = stage3._build_prompt(segments, "en")
    assert "[0] 0.0s-5.0s: Rahul, where have you been?" in prompt
    assert "[1] 5.5s-9.0s: I was at the old house." in prompt
    assert "Write every \"narration\" string in English" in prompt
    print("[test]   _build_prompt OK")

    prompt_hi = stage3._build_prompt(segments, "hi")
    assert "Write every \"narration\" string in Hindi" in prompt_hi
    print("[test]   _build_prompt: language instruction changes with `language` param OK")

    # As of the 2026-08-26 hotfix, .parsed is a plain dict (json.loads of the
    # response text) — response_schema is now a native dict schema
    # (stage3.SUMMARY_SCHEMA), not a nested Pydantic model, specifically so
    # there's no $ref/$defs for google-genai's dereferencing to mishandle.
    # These fakes mirror that real shape rather than a pydantic instance.
    class FakeResponse:
        def __init__(self, parsed):
            self.parsed, self.text = parsed, "{}"

    class FakeModels:
        def __init__(self, parsed):
            self.parsed, self.calls = parsed, []

        def generate_content(self, model, contents, config):
            self.calls.append({"model": model, "contents": contents, "config": config})
            return FakeResponse(self.parsed)

    class FakeClient:
        def __init__(self, parsed):
            self.models = FakeModels(parsed)

    fake_parsed = {"beats": [
        {"index": 1, "narration": "second", "scene_type": "mystery", "beat_role": "body"},
        {"index": 0, "narration": "first", "scene_type": "action", "beat_role": "hook"},
    ]}
    client = FakeClient(fake_parsed)
    result = stage3._call_with_retry(client, "prompt text", stage3.SUMMARY_SCHEMA)
    assert result is fake_parsed
    assert client.models.calls[0]["config"].response_mime_type == "application/json"
    assert client.models.calls[0]["config"].response_schema == stage3.SUMMARY_SCHEMA
    print("[test]   _call_with_retry passes schema-constrained config OK")

    beats = sorted(
        [{"index": b["index"], "narration": b["narration"], "scene_type": b["scene_type"],
          "beat_role": b["beat_role"]} for b in fake_parsed["beats"]],
        key=lambda b: b["index"],
    )
    assert beats == [
        {"index": 0, "narration": "first", "scene_type": "action", "beat_role": "hook"},
        {"index": 1, "narration": "second", "scene_type": "mystery", "beat_role": "body"},
    ]
    print("[test]   out-of-order response re-sorted by index OK")

    # Regression test for the live 2026-08-26 bug: Gemini 1-indexed beats on
    # its own (no Clip-1.mp4 downstream, first file was Clip-2.mp4) despite no
    # instruction either way. _reindex_beats() must re-derive a clean 0..N-1
    # sequence in Python regardless of what indices Gemini actually returns.
    # Also covers the same-day scene_type/beat_role additions -- both must
    # survive re-indexing.
    raw = [
        {"index": 2, "narration": "second", "scene_type": "horror", "beat_role": "body"},
        {"index": 1, "narration": "first", "scene_type": "emotional", "beat_role": "body"},
    ]
    reindexed = stage3._reindex_beats(raw)
    assert reindexed == [
        {"index": 0, "narration": "first", "scene_type": "emotional", "beat_role": "body"},
        {"index": 1, "narration": "second", "scene_type": "horror", "beat_role": "body"},
    ], reindexed
    print("[test]   1-indexed Gemini output re-derived to clean 0-based sequence, "
          "scene_type/beat_role preserved OK")

    # Hook/dopamine redesign (2026-08-26): the beat tagged "hook" must always land
    # first, regardless of its own raw index -- narrative position matters more
    # than the literal index value, same defensive spirit as the 0-indexing fix.
    raw_with_hook = [
        {"index": 0, "narration": "chronological start", "scene_type": "plot", "beat_role": "body"},
        {"index": 9, "narration": "flash-forward teaser", "scene_type": "mystery", "beat_role": "hook"},
    ]
    reindexed_hook = stage3._reindex_beats(raw_with_hook)
    assert reindexed_hook[0] == {
        "index": 0, "narration": "flash-forward teaser", "scene_type": "mystery", "beat_role": "hook",
    }, reindexed_hook
    assert reindexed_hook[1]["beat_role"] == "body"
    print("[test]   beat tagged 'hook' forced to index 0 regardless of its own raw index OK")

    # Multiple hooks -> keep the first (by raw index), demote the rest to 'highlight'
    # rather than erroring or fabricating/discarding content.
    raw_multi_hook = [
        {"index": 0, "narration": "hook A", "scene_type": "action", "beat_role": "hook"},
        {"index": 1, "narration": "body", "scene_type": "plot", "beat_role": "body"},
        {"index": 2, "narration": "hook B", "scene_type": "horror", "beat_role": "hook"},
    ]
    reindexed_multi = stage3._reindex_beats(raw_multi_hook)
    assert reindexed_multi[0]["narration"] == "hook A" and reindexed_multi[0]["beat_role"] == "hook"
    assert [b["beat_role"] for b in reindexed_multi[1:]] == ["body", "highlight"], reindexed_multi
    print("[test]   multiple 'hook' beats: first kept as hook, extras demoted to 'highlight' OK")

    # Zero hooks -> proceeds without one, no crash, no fabricated hook.
    raw_no_hook = [{"index": 0, "narration": "only beat", "scene_type": "plot", "beat_role": "body"}]
    reindexed_none = stage3._reindex_beats(raw_no_hook)
    assert reindexed_none == [{"index": 0, "narration": "only beat", "scene_type": "plot", "beat_role": "body"}]
    print("[test]   zero 'hook' beats: proceeds without fabricating one OK")

    # Real regression check: hand the actual SUMMARY_SCHEMA through google-genai's
    # own schema conversion (t_schema/process_schema) exactly as the live SDK does
    # for a real API call — no mocking. This is the step the old pydantic-model
    # schema silently skipped in this test suite, which is how the live $ref/$defs
    # bug shipped without being caught here.
    import copy
    from google.genai import _transformers as _genai_transformers

    class FakeApiClient:
        vertexai = True

    converted = _genai_transformers.t_schema(FakeApiClient(), copy.deepcopy(stage3.SUMMARY_SCHEMA))
    assert converted is not None
    print("[test]   SUMMARY_SCHEMA passes google-genai's real schema conversion (no $ref/$defs) OK")

    # Real check that scene_type's `enum` constraint actually survives conversion
    # (not just that conversion doesn't error) -- this is the field this session
    # added, so it's the one most likely to silently lose its constraint.
    converted_scene_type = converted.properties["beats"].items.properties["scene_type"]
    assert converted_scene_type.enum == list(settings.EXPLAINER_SCENE_TYPES.keys()), (
        f"scene_type enum did not survive schema conversion intact: {converted_scene_type.enum}"
    )
    print(f"[test]   scene_type enum survives conversion intact: {converted_scene_type.enum}")

    # Same check for beat_role (the hook/twist/highlight/setup/resolution/body field,
    # added same day as the retention/"dopamine" redesign).
    converted_beat_role = converted.properties["beats"].items.properties["beat_role"]
    assert converted_beat_role.enum == list(settings.EXPLAINER_BEAT_ROLES.keys()), (
        f"beat_role enum did not survive schema conversion intact: {converted_beat_role.enum}"
    )
    print(f"[test]   beat_role enum survives conversion intact: {converted_beat_role.enum}")


# ---------------------------------------------------------------------------
# Stage 4 — clip matching: prompt building + resolve_matches (real logic)
# ---------------------------------------------------------------------------
def test_stage4_unit():
    print("\n[test] === Stage 4 unit: clip matching ===")

    beats = [{"index": 0, "narration": "Rahul finds the letter."},
             {"index": 1, "narration": "He confronts his mother."},
             {"index": 2, "narration": "Meanwhile, elsewhere in the city..."}]
    segments = [
        {"start": 10.0, "end": 15.0, "text": "..."},
        {"start": 15.5, "end": 20.0, "text": "..."},
        {"start": 30.0, "end": 35.0, "text": "..."},
    ]
    prompt = stage4._build_prompt(beats, segments)
    assert "[0] Rahul finds the letter." in prompt
    assert "[2] 30.0s-35.0s" in prompt
    print("[test]   _build_prompt OK")

    matches_by_beat = {0: [0, 1], 1: [2]}
    resolved = stage4.resolve_matches([dict(b) for b in beats], segments, matches_by_beat)
    assert resolved[0]["cited_segments"] == [0, 1]
    assert resolved[0]["start"] == 10.0 and resolved[0]["end"] == 20.0
    assert resolved[1]["cited_segments"] == [2]
    assert resolved[1]["start"] == 30.0 and resolved[1]["end"] == 35.0
    assert resolved[2]["cited_segments"] == [] and resolved[2]["start"] is None
    print("[test]   resolve_matches: multi-segment span + missing-beat fallback OK")

    matches_by_beat2 = {0: [0, 99], 1: [], 2: []}
    resolved2 = stage4.resolve_matches([dict(b) for b in beats], segments, matches_by_beat2)
    assert resolved2[0]["cited_segments"] == [0], "unrecoverable out-of-range index 99 should be dropped"
    assert resolved2[0]["start"] == 10.0 and resolved2[0]["end"] == 15.0
    print("[test]   resolve_matches: unrecoverable out-of-range index dropped OK")

    matches_by_beat3 = {0: [], 1: [0], 2: []}
    resolved3 = stage4.resolve_matches([dict(b) for b in beats], segments, matches_by_beat3)
    assert resolved3[0]["start"] is None and resolved3[0]["end"] is None
    print("[test]   resolve_matches: explicit empty citation -> uncited beat OK")

    # Real bug, 2026-08-27: on a long transcript Gemini sometimes cites a segment's
    # TIMESTAMP instead of its bracket INDEX (confirmed on a real ~1787-line movie
    # run — cited "9209" meaning index 1710, whose real start is 9209.2s). Segment 2's
    # start here is 30.0 -> Gemini citing "30" (an out-of-range index for this 3-segment
    # list) should recover to index 2, not get dropped.
    matches_by_beat4 = {0: [], 1: [], 2: [30]}
    resolved4 = stage4.resolve_matches([dict(b) for b in beats], segments, matches_by_beat4)
    assert resolved4[2]["cited_segments"] == [2], "timestamp-shaped citation should recover to the real index"
    assert resolved4[2]["start"] == 30.0 and resolved4[2]["end"] == 35.0
    print("[test]   resolve_matches: timestamp-cited-instead-of-index recovered OK")

    # _recover_timestamp_index directly: within tolerance recovers, far away doesn't,
    # empty list doesn't crash.
    starts = [10.0, 15.5, 30.0, 9209.2, 9629.48, 9914.5]
    assert stage4._recover_timestamp_index(9209, starts) == 3
    assert stage4._recover_timestamp_index(9629, starts) == 4
    assert stage4._recover_timestamp_index(9914, starts) == 5
    assert stage4._recover_timestamp_index(50000, starts) is None, "far-away value should not recover"
    assert stage4._recover_timestamp_index(9209, []) is None, "empty segment list should not crash"
    print("[test]   _recover_timestamp_index: real-world values (from the actual Odyssey run) OK")

    # Real regression check, same reasoning as Stage 3's — run MATCH_SCHEMA through
    # google-genai's actual schema conversion, no mocking.
    import copy
    from google.genai import _transformers as _genai_transformers

    class FakeApiClient:
        vertexai = True

    converted = _genai_transformers.t_schema(FakeApiClient(), copy.deepcopy(stage4.MATCH_SCHEMA))
    assert converted is not None
    print("[test]   MATCH_SCHEMA passes google-genai's real schema conversion (no $ref/$defs) OK")


# ---------------------------------------------------------------------------
# Intro feature (2026-08-27, new): prompt building + resolve_intro_citation()'s
# reuse of Stage 4's resolve_matches() (including its timestamp-recovery fix,
# Bug log #11 — this is the whole point of reusing rather than reimplementing).
# ---------------------------------------------------------------------------
def test_intro_unit():
    print("\n[test] === Intro unit: prompt building + resolve_intro_citation ===")

    summary_beats = [
        {"narration": "Rahul finds a hidden letter."},
        {"narration": "He confronts his mother."},
    ]
    segments = [
        {"start": 0.0, "end": 5.0, "text": "Rahul, where have you been?"},
        {"start": 5.5, "end": 9.0, "text": "I was at the old house."},
    ]
    prompt = intro_mod._build_prompt(summary_beats, segments, "en")
    assert "- Rahul finds a hidden letter." in prompt
    assert "[0] 0.0s-5.0s: Rahul, where have you been?" in prompt
    assert "Write in English" in prompt
    assert "Do NOT state the movie's exact title" in prompt
    print("[test]   _build_prompt: summary context + numbered transcript + no-title instruction OK")

    prompt_hi = intro_mod._build_prompt(summary_beats, segments, "hi")
    assert "Write in Hindi" in prompt_hi
    print("[test]   _build_prompt: language instruction changes with `language` param OK")

    # resolve_intro_citation reuses stage4.resolve_matches() directly -- prove the ordinary
    # multi-segment-span case resolves correctly...
    resolved = intro_mod.resolve_intro_citation(
        "Today, we're diving into a mystery.", segments, [0, 1],
    )
    assert resolved["narration"] == "Today, we're diving into a mystery."
    assert resolved["cited_segments"] == [0, 1]
    assert resolved["start"] == 0.0 and resolved["end"] == 9.0
    print("[test]   resolve_intro_citation: multi-segment span resolves via resolve_matches() OK")

    # ...and, crucially, that it gets Stage 4's timestamp-cited-instead-of-index recovery
    # (Bug log #11) for free, not a second, drift-prone reimplementation.
    ts_segments = [
        {"start": 0.0, "end": 5.0, "text": "a"},
        {"start": 5.5, "end": 9.0, "text": "b"},
        {"start": 200.0, "end": 205.0, "text": "c"},
    ]
    resolved_ts = intro_mod.resolve_intro_citation("Intro line.", ts_segments, [200])
    assert resolved_ts["cited_segments"] == [2], (
        "a timestamp-shaped citation (200, matching segment 2's real start) should recover "
        "to the real index via the SAME logic Stage 4 uses, not be dropped as out-of-range"
    )
    assert resolved_ts["start"] == 200.0 and resolved_ts["end"] == 205.0
    print("[test]   resolve_intro_citation: reuses Stage 4's timestamp-recovery fix (Bug log #11) OK")

    # empty citation -> uncited intro, no crash
    resolved_empty = intro_mod.resolve_intro_citation("No footage cited.", segments, [])
    assert resolved_empty["cited_segments"] == [] and resolved_empty["start"] is None
    print("[test]   resolve_intro_citation: empty citation -> uncited intro OK")

    # Real regression check, same reasoning as Stage 3/4's -- run INTRO_SCHEMA through
    # google-genai's actual schema conversion, no mocking.
    import copy
    from google.genai import _transformers as _genai_transformers

    class FakeApiClient:
        vertexai = True

    converted = _genai_transformers.t_schema(FakeApiClient(), copy.deepcopy(intro_mod.INTRO_SCHEMA))
    assert converted is not None
    print("[test]   INTRO_SCHEMA passes google-genai's real schema conversion (no $ref/$defs) OK")


# ---------------------------------------------------------------------------
# Subtitle (.srt / SDH) support — real-transcript alternative to Whisper (2026-08-27)
# ---------------------------------------------------------------------------
def test_subtitles_unit():
    print("\n[test] === Subtitle (.srt) support unit: parse_srt + load_or_transcribe ===")

    srt_content = (
        "1\n"
        "00:00:40,000 --> 00:00:42,000\n"
        "<i>Ah, apologies.</i>\n"
        "\n"
        "2\n"
        "00:00:42,500 --> 00:00:45,500\n"
        "<i>Sorry, the thing is, uh, public speaking</i>\n"
        "<i>has never been my, uh</i>\n"
        "\n"
        "3\n"
        "00:00:10,000 --> 00:00:16,000\n"
        "[Growling]\n"
        "\n"
        "4\n"
        "this is a malformed block with no timestamp at all\n"
        "should be skipped entirely\n"
        "\n"
        "5\n"
        "00:00:47,000 --> 00:00:48,000\n"
        "<i></i>\n"
    )
    srt_path = TEST_DIR / "sample.srt"
    srt_path.write_text(srt_content, encoding="utf-8")

    segments = subtitles.parse_srt(srt_path)
    assert segments == [
        {"start": 10.0, "end": 16.0, "text": "[Growling]"},
        {"start": 40.0, "end": 42.0, "text": "Ah, apologies."},
        {"start": 42.5, "end": 45.5,
         "text": "Sorry, the thing is, uh, public speaking has never been my, uh"},
    ], segments
    print("[test]   parse_srt: <i> tags stripped, multi-line cues joined, malformed/empty "
          "cues skipped, SDH bracketed cue preserved, sorted by start OK")

    # find_subtitle_file: matches a same-stem .srt next to the video (English backward-compat
    # bare-file fallback), None when absent
    video_with_srt = TEST_DIR / "sample.mp4"
    assert subtitles.find_subtitle_file(video_with_srt, "en") == srt_path
    video_without_srt = TEST_DIR / "no_subs_here.mp4"
    assert subtitles.find_subtitle_file(video_without_srt, "en") is None
    print("[test]   find_subtitle_file: matches same-stem .srt (English bare-file fallback), "
          "None when absent OK")

    # find_subtitle_file: per-movie folder convention (2026-09-05 folder-structure reorg) --
    # data/Subtitle/{movie}/{language}_subtitle.srt. No bare-file fallback for non-English.
    video_with_hi_srt = TEST_DIR / "sample_hi.mp4"
    hi_srt_path = subtitles.subtitle_path_for(video_with_hi_srt, "hi")
    hi_srt_path.write_text(srt_content, encoding="utf-8")
    assert subtitles.find_subtitle_file(video_with_hi_srt, "hi") == hi_srt_path
    assert subtitles.find_subtitle_file(video_with_hi_srt, "bn") is None, (
        "a hi subtitle must not be picked up when asking for bn"
    )
    assert subtitles.find_subtitle_file(video_with_srt, "hi") is None, (
        "the bare-file fallback is English-only -- hi/bn must not fall back to a bare .srt"
    )
    print("[test]   find_subtitle_file: data/Subtitle/{movie}/{language}_subtitle.srt convention, "
          "no cross-language or non-English bare-file fallback OK")

    # migrate_legacy_subtitles(): a subtitle still sitting at the OLD {stem}.{language}.srt-
    # next-to-the-video location (2026-08-27 through 2026-09-05 convention) must be moved into
    # the new per-movie folder, and be findable there afterward -- the real upgrade path for
    # whatever the user already generated before this reorg.
    legacy_video = TEST_DIR / "sample_legacy.mp4"
    legacy_old_path = TEST_DIR / "sample_legacy.bn.srt"
    legacy_old_path.write_text(srt_content, encoding="utf-8")
    moved = subtitles.migrate_legacy_subtitles(input_dir=TEST_DIR)
    assert (str(legacy_old_path), str(subtitles.subtitle_path_for(legacy_video, "bn"))) in moved
    assert not legacy_old_path.exists(), "the old-location file must be MOVED, not copied"
    assert subtitles.find_subtitle_file(legacy_video, "bn") == subtitles.subtitle_path_for(legacy_video, "bn")
    assert subtitles.migrate_legacy_subtitles(input_dir=TEST_DIR) == [], (
        "must be a no-op the second time -- nothing old-style left to move"
    )
    print("[test]   migrate_legacy_subtitles: moves an old-location subtitle into the new "
          "per-movie folder, idempotent (no-op) on a second call OK")

    # load_or_transcribe: .srt present -> Whisper must NOT be called at all (that's the whole
    # point -- skip the ~20-30 minute transcription step, not just prefer the .srt afterward)
    real_video = TEST_DIR / "subtitles_probe.mp4"
    make_synthetic_video(real_video, duration=3)
    real_srt = real_video.with_suffix(".srt")
    real_srt.write_text(srt_content, encoding="utf-8")
    transcript_path = settings.TRANSCRIPT_DIR / f"{real_video.stem}.json"
    transcript_path.unlink(missing_ok=True)

    def _boom(*a, **kw):
        raise AssertionError("stage2_transcribe.transcribe() must NOT be called when a .srt is present")

    with patch("pipeline.stage2_transcribe.transcribe", _boom):
        out_path = subtitles.load_or_transcribe(real_video, TEST_DIR / "unused.wav", "en")
    assert out_path == transcript_path
    written = json.loads(transcript_path.read_text(encoding="utf-8"))
    assert written["source"] == "srt"
    assert written["segments"] == segments
    assert written["language"] == "en"
    assert written["duration"] > 0, "duration should be a real ffprobed value, not a zero fallback"
    print("[test]   load_or_transcribe: .srt present -> Whisper skipped entirely, transcript.json "
          "written in the same shape Stage 2 writes OK")

    # load_or_transcribe: no .srt -> falls back to Whisper (mocked here, not actually run) for
    # English, which allows the fallback
    no_srt_video = TEST_DIR / "subtitles_no_srt.mp4"
    fake_transcribe_calls = []

    def _fake_transcribe(audio_path):
        fake_transcribe_calls.append(audio_path)
        return Path("fake_transcript.json")

    with patch("pipeline.stage2_transcribe.transcribe", _fake_transcribe):
        result = subtitles.load_or_transcribe(no_srt_video, TEST_DIR / "some_audio.wav", "en")
    assert fake_transcribe_calls == [TEST_DIR / "some_audio.wav"], (
        "Whisper fallback must be called with the real audio_path when no .srt is found"
    )
    assert result == Path("fake_transcript.json")
    print("[test]   load_or_transcribe: no .srt, language=en -> falls back to Whisper (called "
          "with the right audio_path) OK")

    # load_or_transcribe: Hindi/Bengali have NO Whisper fallback -- a missing subtitle must
    # raise, not silently transcribe the (likely wrong-language) audio.
    no_srt_video_hi = TEST_DIR / "subtitles_no_srt_hi.mp4"

    def _boom_hi(*a, **kw):
        raise AssertionError("stage2_transcribe.transcribe() must NOT be called for hi/bn -- no "
                              "Whisper fallback exists for those languages")

    with patch("pipeline.stage2_transcribe.transcribe", _boom_hi):
        try:
            subtitles.load_or_transcribe(no_srt_video_hi, TEST_DIR / "some_audio.wav", "hi")
            raise AssertionError("expected RuntimeError for a missing required hi subtitle")
        except RuntimeError:
            pass
    print("[test]   load_or_transcribe: language=hi with no subtitle -> raises (no Whisper "
          "fallback attempted) OK")


# ---------------------------------------------------------------------------
# Subtitle generate/translate utilities (2026-09-05) — solves "movie has no
# subtitle in the language I need" without hunting one down online. See
# pipeline/subtitles.py's module docstring for the full "translate-first"
# architecture reasoning.
# ---------------------------------------------------------------------------
def test_subtitle_generate_translate_unit():
    print("\n[test] === Subtitle generate/translate unit ===")

    # --- _segments_to_srt / _seconds_to_srt_timestamp: pure round-trip via parse_srt ---
    assert subtitles._seconds_to_srt_timestamp(0.0) == "00:00:00,000"
    assert subtitles._seconds_to_srt_timestamp(3725.4) == "01:02:05,400"
    print("[test]   _seconds_to_srt_timestamp: formatting OK")

    round_trip_segments = [
        {"start": 1.0, "end": 4.0, "text": "Hello there."},
        {"start": 4.5, "end": 9.25, "text": "[Music]"},
    ]
    srt_text = subtitles._segments_to_srt(round_trip_segments)
    round_trip_path = TEST_DIR / "round_trip.srt"
    round_trip_path.write_text(srt_text, encoding="utf-8")
    reparsed = subtitles.parse_srt(round_trip_path)
    assert reparsed == round_trip_segments, reparsed
    print("[test]   _segments_to_srt -> parse_srt round-trips exactly OK")

    # --- detect_language / generate_subtitle: mocked WhisperModel ---
    # faster-whisper isn't installed in this sandbox (it's a real-machine-only dependency,
    # same as google-cloud-texttospeech would be if it weren't already imported above for
    # attribute-chain resolution) -- inject a fake module into sys.modules directly rather
    # than patching an attribute on a package that doesn't exist here. subtitles.py does
    # `from faster_whisper import WhisperModel` INSIDE each function, so this is resolved
    # fresh on every call and picks up whatever's in sys.modules at call time.
    import types

    class FakeWhisperModel:
        def __init__(self, *a, **kw):
            pass

        def transcribe(self, path, language=None, beam_size=5, vad_filter=None):
            used_lang = language or "hi"
            info = SimpleNamespace(language=used_lang, language_probability=0.97)
            fake_segments = [
                SimpleNamespace(start=0.0, end=2.5, text=" Namaste duniya "),
                SimpleNamespace(start=3.0, end=6.0, text=" Yeh ek kahani hai "),
            ]
            return iter(fake_segments), info

    fake_faster_whisper = types.ModuleType("faster_whisper")
    fake_faster_whisper.WhisperModel = FakeWhisperModel
    sys.modules["faster_whisper"] = fake_faster_whisper

    # generate_subtitle() now needs a real, ffmpeg-decodable file too (2026-09-05 folder
    # reorg: it calls ensure_movie_audio() to extract/cache data/Subtitle/{movie}/audio.mp3
    # before ever calling Whisper, rather than handing Whisper the raw video). Reusing ONE
    # real synthetic video for both calls below also exercises the audio.mp3 CACHE itself:
    # the second generate_subtitle() call must reuse the audio.mp3 the first one created,
    # not re-extract it.
    probe_path = TEST_DIR / "generate_probe.mp4"
    make_synthetic_video(probe_path, duration=10)
    probe_audio_path = settings.SUBTITLE_DIR / probe_path.stem / "audio.mp3"

    # detect_language() ALSO needs a real, ffmpeg-decodable file (2026-09-05 fix: it extracts
    # a short real audio sample via stage1_ingest.extract_audio_sample() before ever calling
    # Whisper, rather than handing Whisper the whole file — see that function's docstring for
    # the real-machine ArrayMemoryError this fixes).
    detect_probe_path = TEST_DIR / "detect_probe.mp4"
    make_synthetic_video(detect_probe_path, duration=10)
    detect_audio_path = settings.SUBTITLE_DIR / detect_probe_path.stem / "audio.mp3"
    sample_path = settings.AUDIO_DIR / f"{detect_probe_path.stem}.lang_sample.wav"

    lang, prob = subtitles.detect_language(detect_probe_path)
    assert lang == "hi" and prob == 0.97
    assert not sample_path.exists(), "the throwaway audio sample must be deleted after use"
    assert detect_audio_path.exists(), "detect_language() must cache the movie's audio.mp3 for reuse"
    print("[test]   detect_language: extracts+caches audio.mp3, samples a real short clip from it "
          "(cleaned up after use), auto-detects via info.language/language_probability OK")

    srt_path, used_language, probability = subtitles.generate_subtitle(probe_path, language=None)
    assert used_language == "hi" and probability == 0.97
    assert srt_path == subtitles.subtitle_path_for(probe_path, "hi")
    assert probe_audio_path.exists(), "generate_subtitle() must cache the movie's audio.mp3 too"
    first_audio_mtime = probe_audio_path.stat().st_mtime
    generated_segments = subtitles.parse_srt(srt_path)
    assert generated_segments == [
        {"start": 0.0, "end": 2.5, "text": "Namaste duniya"},
        {"start": 3.0, "end": 6.0, "text": "Yeh ek kahani hai"},
    ], generated_segments
    print("[test]   generate_subtitle: auto-detect -> real .srt written under the per-movie "
          "data/Subtitle/{movie}/ folder, parses back to the same cues OK")

    srt_path_en, used_language_en, probability_en = subtitles.generate_subtitle(probe_path, language="en")
    assert used_language_en == "en" and probability_en is None, (
        "probability must be None when language was pinned explicitly -- nothing was detected"
    )
    assert srt_path_en == subtitles.subtitle_path_for(probe_path, "en")
    assert probe_audio_path.stat().st_mtime == first_audio_mtime, (
        "this second call must REUSE the cached audio.mp3, not re-extract it"
    )
    print("[test]   generate_subtitle: language pinned explicitly -> probability=None, "
          "en_subtitle.srt written, cached audio.mp3 reused (not re-extracted) OK")

    # --- translate_subtitle: mocked google.genai.Client ---
    translate_video = TEST_DIR / "translate_probe.mp4"
    en_srt = subtitles.subtitle_path_for(translate_video, "en")
    source_segments = [
        {"start": 0.0, "end": 2.0, "text": f"Line number {i}"} for i in range(5)
    ]
    en_srt.write_text(subtitles._segments_to_srt(source_segments), encoding="utf-8")

    original_batch_size = settings.TRANSLATE_BATCH_SIZE
    settings.TRANSLATE_BATCH_SIZE = 2  # force multiple batches (5 cues / 2 per batch = 3 calls)
    settings.GCP_PROJECT_ID = settings.GCP_PROJECT_ID or "fake-test-project"

    translate_call_log = []

    class FakeTranslateResponse:
        def __init__(self, parsed):
            self.parsed, self.text = parsed, "{}"

    class FakeTranslateModels:
        def generate_content(self, model, contents, config):
            translate_call_log.append(contents)
            # Simulate: translate every line EXCEPT deliberately drop index 1 of the FIRST
            # batch, to exercise the "no translation returned for this cue" fallback path.
            n_lines = contents.count("\n[")
            lines = []
            for i in range(n_lines):
                if len(translate_call_log) == 1 and i == 1:
                    continue  # drop this one on purpose
                lines.append({"index": i, "text": f"[HI] line {i}"})
            return FakeTranslateResponse({"lines": lines})

    class FakeTranslateClient:
        def __init__(self, *a, **kw):
            self.models = FakeTranslateModels()

    try:
        with patch("google.genai.Client", FakeTranslateClient):
            out_path = subtitles.translate_subtitle(translate_video, "en", "hi")
    finally:
        settings.TRANSLATE_BATCH_SIZE = original_batch_size

    assert len(translate_call_log) == 3, f"expected 3 batches of <=2 cues, got {len(translate_call_log)}"
    assert out_path == subtitles.subtitle_path_for(translate_video, "hi")
    translated_segments = subtitles.parse_srt(out_path)
    assert len(translated_segments) == 5
    for i, (orig, trans) in enumerate(zip(source_segments, translated_segments)):
        assert trans["start"] == orig["start"] and trans["end"] == orig["end"], (
            "translated cues must carry the EXACT original timestamps, never re-derived"
        )
    assert translated_segments[1]["text"] == "Line number 1", (
        "the deliberately-dropped index in batch 1's response must fall back to the "
        "ORIGINAL untranslated text, not crash or silently disappear"
    )
    assert translated_segments[0]["text"] == "[HI] line 0"
    assert translated_segments[2]["text"] == "[HI] line 0", (
        "batch 2 (cues 2-3) should restart its own local indexing at 0"
    )
    print("[test]   translate_subtitle: batches correctly, preserves original timestamps exactly, "
          "index-mapped response resolves to the right cue, missing-index falls back to the "
          "original untranslated text OK")

    # --- error paths ---
    try:
        subtitles.translate_subtitle(translate_video, "en", "en")
        raise AssertionError("expected RuntimeError for source == target language")
    except RuntimeError:
        pass
    try:
        subtitles.translate_subtitle(TEST_DIR / "no_such_movie.mp4", "en", "hi")
        raise AssertionError("expected RuntimeError when no source-language subtitle exists")
    except RuntimeError:
        pass
    try:
        subtitles.translate_subtitle(translate_video, "fr", "hi")
        raise AssertionError("expected RuntimeError for an unsupported source language")
    except RuntimeError:
        pass
    print("[test]   translate_subtitle: same-language, missing-source-subtitle, and "
          "unsupported-language all raise cleanly before any Gemini call OK")

    # --- Real regression check, same reasoning as every other schema in this file --
    # run TRANSLATE_SCHEMA through google-genai's actual schema conversion, no mocking.
    import copy
    from google.genai import _transformers as _genai_transformers

    class FakeApiClient:
        vertexai = True

    converted = _genai_transformers.t_schema(FakeApiClient(), copy.deepcopy(subtitles.TRANSLATE_SCHEMA))
    assert converted is not None
    print("[test]   TRANSLATE_SCHEMA passes google-genai's real schema conversion (no $ref/$defs) OK")


# ---------------------------------------------------------------------------
# Audio-event detection — "silent action/horror" blind-spot fix (2026-08-26)
# ---------------------------------------------------------------------------
def test_audio_events_unit():
    print("\n[test] === Audio-event detection unit: dialogue-free gap + energy logic ===")

    # _dialogue_free_gaps: pure function, no I/O
    segs = [{"start": 0.0, "end": 2.0}, {"start": 12.0, "end": 14.0}]
    gaps = audio_events._dialogue_free_gaps(segs, total_duration=24.0, min_gap_seconds=8.0)
    assert gaps == [(2.0, 12.0), (14.0, 24.0)], gaps
    print("[test]   _dialogue_free_gaps: correct gaps computed OK")

    # a gap shorter than min_gap_seconds must NOT be reported (ordinary pause between lines)
    segs2 = [{"start": 0.0, "end": 2.0}, {"start": 5.0, "end": 6.0}]  # 3s gap, below 8.0 threshold
    gaps2 = audio_events._dialogue_free_gaps(segs2, total_duration=6.0, min_gap_seconds=8.0)
    assert gaps2 == [], gaps2
    print("[test]   _dialogue_free_gaps: gap shorter than min_gap_seconds correctly excluded OK")

    # _flag_loud_gaps: pure function, synthetic energy windows
    windows = (
        [{"start": float(i), "end": float(i + 1), "energy": 1.0} for i in range(0, 8)]      # quiet baseline
        + [{"start": float(i), "end": float(i + 1), "energy": 10.0} for i in range(8, 12)]  # loud gap
    )
    loud_events = audio_events._flag_loud_gaps(
        gaps=[(8.0, 12.0)], windows=windows, energy_ratio=1.5, min_loud_fraction=0.6,
    )
    assert len(loud_events) == 1 and loud_events[0]["start"] == 8.0 and loud_events[0]["end"] == 12.0
    assert loud_events[0]["is_audio_event"] is True
    assert loud_events[0]["text"] == audio_events.AUDIO_EVENT_PLACEHOLDER_TEXT
    print("[test]   _flag_loud_gaps: sustained-loud gap correctly flagged OK")

    quiet_events = audio_events._flag_loud_gaps(
        gaps=[(0.0, 4.0)], windows=windows, energy_ratio=1.5, min_loud_fraction=0.6,
    )
    assert quiet_events == [], quiet_events
    print("[test]   _flag_loud_gaps: quiet gap correctly NOT flagged OK")

    # a single stray loud window in an otherwise quiet gap (below min_loud_fraction) must NOT
    # be flagged -- e.g. a loud sound effect during a mostly-quiet pause isn't a real scene
    mixed_windows = (
        [{"start": float(i), "end": float(i + 1), "energy": 1.0} for i in range(0, 4)]
        + [{"start": 4.0, "end": 5.0, "energy": 10.0}]  # 1 of 5 windows loud = 20%, below 60%
    )
    mixed_events = audio_events._flag_loud_gaps(
        gaps=[(0.0, 5.0)], windows=mixed_windows, energy_ratio=1.5, min_loud_fraction=0.6,
    )
    assert mixed_events == [], mixed_events
    print("[test]   _flag_loud_gaps: single stray loud window (below min_loud_fraction) "
          "correctly NOT flagged OK")

    # merge_segments_with_audio_events: correct time-ordered interleaving
    merged = audio_events.merge_segments_with_audio_events(segs, [
        {"start": 2.0, "end": 12.0, "text": audio_events.AUDIO_EVENT_PLACEHOLDER_TEXT, "is_audio_event": True},
    ])
    assert [(m["start"], m["is_audio_event"]) for m in merged] == [
        (0.0, False), (2.0, True), (12.0, False),
    ], merged
    print("[test]   merge_segments_with_audio_events: correct time-ordered interleaving OK")

    # Real end-to-end: real ffmpeg-generated quiet/loud/quiet audio, real librosa analysis --
    # not just the pure math above. This is the actual blind-spot fix being exercised for real.
    audio_stem = "audio_events_sandbox_test"
    cache_path = settings.EXPLAINER_DIR / audio_stem / "audio_events.json"  # per-movie root,
                                          # shared across languages -- see audio_events.py docstring
    cache_path.unlink(missing_ok=True)
    wav_path = TEST_DIR / "quiet_loud_quiet.wav"
    make_quiet_loud_quiet_audio(wav_path)

    real_segments = [
        {"start": 0.0, "end": 2.0, "text": "intro line"},
        {"start": 12.0, "end": 14.0, "text": "mid line"},
    ]
    real_events = audio_events.detect_audio_events(audio_stem, real_segments, audio_path=wav_path)
    assert len(real_events) == 1, real_events
    assert real_events[0]["start"] == 2.0 and real_events[0]["end"] == 12.0, real_events
    assert real_events[0]["is_audio_event"] is True
    print(f"[test]   detect_audio_events (real ffmpeg audio + real librosa): the loud gap "
          f"(2.0s-12.0s) is correctly detected and the quiet gap (14.0s-24.0s) is correctly "
          f"ignored -> {real_events}")

    # Cached on the second call -- prove it by removing the audio file and re-calling with the
    # SAME stem; an uncached call would hit "no extracted audio" and return [] instead.
    wav_path.unlink()
    cached_events = audio_events.detect_audio_events(audio_stem, real_segments, audio_path=wav_path)
    assert cached_events == real_events, cached_events
    print("[test]   detect_audio_events: second call served from cache (audio file removed, "
          "still returns the identical result) OK")
    cache_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Multi-language redesign (2026-08-27): NARRATOR_VOICES lookup + clear-error
# behavior for a language with no voice configured yet (Bengali, deferred by
# explicit user choice). Must raise BEFORE touching the network or reading
# matched.json — no real matched.json fixture needed for this check.
# ---------------------------------------------------------------------------
def test_multilang_voice_unit():
    print("\n[test] === Multi-language unit: NARRATOR_VOICES lookup / missing-voice error ===")

    assert settings.NARRATOR_VOICES.get("bn") is None, (
        "this test assumes Bengali has no voice configured yet -- if that's changed, "
        "update this test's expectations"
    )
    try:
        stage5.synthesize_narration("nonexistent_stem", Path("nonexistent.mp4"),
                                     Path("nonexistent.wav"), "bn")
        raise AssertionError("expected RuntimeError for a language with no configured voice")
    except RuntimeError as e:
        assert "bn" in str(e) or "Bengali" in str(e), f"error should name the language: {e}"
    print("[test]   synthesize_narration: language with NARRATOR_VOICES[lang]=None raises a "
          "clear error before any file I/O or network call OK")

    try:
        stage5.synthesize_narration("nonexistent_stem", Path("nonexistent.mp4"),
                                     Path("nonexistent.wav"), "fr")
        raise AssertionError("expected RuntimeError for an unsupported language")
    except RuntimeError:
        pass
    print("[test]   synthesize_narration: unsupported language raises OK")


# ---------------------------------------------------------------------------
# Stage 6 — _plan_clip branch coverage (pure math)
# ---------------------------------------------------------------------------
def test_stage6_plan_clip(video_duration):
    print("\n[test] === Stage 6 unit: _plan_clip branches ===")
    cs, cd, ratio, freeze = stage6._plan_clip(2.0, 5.0, 3.65, video_duration)
    assert settings.STRETCH_MIN_RATIO <= ratio <= settings.STRETCH_MAX_RATIO and freeze == 0.0
    cs, cd, ratio, freeze = stage6._plan_clip(19.5, 20.0, 25.0, video_duration)
    assert abs(ratio - settings.STRETCH_MAX_RATIO) < 1e-6 and freeze > 0
    cs, cd, ratio, freeze = stage6._plan_clip(5.0, 35.0, 1.0, video_duration)
    assert abs(ratio - settings.STRETCH_MIN_RATIO) < 1e-6 and freeze == 0.0
    print("[test]   stretch / extend+freeze / crop branches all OK")


# ---------------------------------------------------------------------------
# Full pipeline: generate_summary -> match_clips -> synthesize_narration ->
# retime_clips, through the REAL top-level functions, with only the Gemini/
# TTS network clients mocked out.
# ---------------------------------------------------------------------------
def test_full_pipeline(video_path: Path, audio_path: Path):
    print("\n[test] === Full pipeline (real functions, mocked network clients) ===")
    settings.GCP_PROJECT_ID = settings.GCP_PROJECT_ID or "fake-test-project"

    # Guard against a stale audio_events cache from a previous run — this test's own audio
    # (TEST_DIR/synthetic.wav) is never written to settings.AUDIO_DIR/{STEM}.wav, so
    # detect_audio_events() should find no extracted audio and gracefully return no events;
    # a leftover cache file would silently defeat that and inject unexpected synthetic beats.
    (settings.EXPLAINER_DIR / STEM / "audio_events.json").unlink(missing_ok=True)

    # --- fixture transcript, written where Stage 3 expects it ---
    segments = [
        {"start": 2.0, "end": 5.0, "text": "Rahul finds a letter in the attic."},
        {"start": 20.0, "end": 24.0, "text": "He confronts his mother about it."},
        {"start": 34.5, "end": 35.5, "text": "She refuses to answer."},
    ]
    transcript_path = settings.TRANSCRIPT_DIR / f"{STEM}.json"
    transcript_path.write_text(
        json.dumps({"language": "en", "duration": 40.0, "segments": segments}, indent=2),
        encoding="utf-8",
    )

    # --- Stage 3: mock genai.Client to return a fixed 2-beat summary ---
    # Plain dicts, matching what response.parsed actually is now (json.loads
    # of the response text, since response_schema is a native dict schema —
    # see the note atop explainer_stage3_summary.py for why).
    # Deliberately 1-indexed (2, 1 instead of 0, 1), matching the real
    # 2026-08-26 bug — generate_summary() must re-derive clean 0-based
    # indices regardless, so Stage 4's beat-index labels below use the
    # POST-reindex values (0, 1), not what this fake returns.
    # beat_role: the second beat (Gemini's raw index 5, deliberately HIGHER than the
    # first beat's raw index 1) is tagged "hook" -- _reindex_beats() must force it to
    # final index 0 anyway, proving hook placement doesn't just follow index order.
    fake_summary_parsed = {"beats": [
        {"index": 5, "narration": "He confronts his mother, who refuses to explain.",
         "scene_type": "emotional", "beat_role": "hook"},
        {"index": 1, "narration": "Rahul finds a hidden letter.", "scene_type": "mystery",
         "beat_role": "body"},
    ]}
    fake_match_parsed = {"matches": [
        {"beat_index": 0, "cited_segments": [0]},
        {"beat_index": 1, "cited_segments": [1, 2]},
    ]}
    # Intro feature (2026-08-27, new): a separate one-call fake result, distinguished from
    # the summary/match fakes below by the intro prompt's own marker text ("channel intro").
    fake_intro_parsed = {
        "narration": "Today, we're diving into a tense family drama full of secrets.",
        "cited_segments": [0],
    }

    class FakeGenaiResponse:
        def __init__(self, parsed):
            self.parsed, self.text = parsed, "{}"

    call_log = []

    class FakeGenaiModels:
        def generate_content(self, model, contents, config):
            call_log.append(contents)
            if "beat" in contents.lower() and "match" not in contents.lower()[:60]:
                pass
            # distinguish Stage 3 vs Stage 4 vs the intro call by each prompt's own marker text
            if "channel intro" in contents.lower():
                return FakeGenaiResponse(fake_intro_parsed)
            if "matching narration beats" in contents.lower():
                return FakeGenaiResponse(fake_match_parsed)
            return FakeGenaiResponse(fake_summary_parsed)

    class FakeGenaiClient:
        def __init__(self, *a, **kw):
            self.models = FakeGenaiModels()

    with patch("google.genai.Client", FakeGenaiClient):
        stage3.generate_summary(STEM, "en")
        intro_mod.generate_intro(STEM, "en")
        stage4.match_clips(STEM, "en")

    intro_after_generate = json.loads(
        (settings.EXPLAINER_DIR / STEM / "en" / "intro.json").read_text(encoding="utf-8")
    )
    assert intro_after_generate["narration"] == fake_intro_parsed["narration"]
    assert intro_after_generate["cited_segments"] == [0]
    assert intro_after_generate["start"] == 2.0 and intro_after_generate["end"] == 5.0
    print("[test]   generate_intro() writes intro.json with narration + resolved real "
          "timestamps from its cited transcript segment OK")

    summary = json.loads(
        (settings.EXPLAINER_DIR / STEM / "en" / "summary.json").read_text(encoding="utf-8")
    )
    assert len(summary["beats"]) == 2
    assert [b["index"] for b in summary["beats"]] == [0, 1], (
        f"generate_summary() should re-derive 0-based indices even though the fake Gemini "
        f"response was 1-indexed; got {[b['index'] for b in summary['beats']]}"
    )
    assert summary["beats"][0]["narration"] == "He confronts his mother, who refuses to explain.", (
        "the beat tagged 'hook' must be forced to index 0 regardless of its own raw index "
        "(it had the HIGHER raw index of the two, 5 vs 1)"
    )
    assert [b["scene_type"] for b in summary["beats"]] == ["emotional", "mystery"], (
        f"scene_type must survive the 0-based re-indexing; got "
        f"{[b.get('scene_type') for b in summary['beats']]}"
    )
    assert [b["beat_role"] for b in summary["beats"]] == ["hook", "body"], (
        f"beat_role must survive the 0-based re-indexing, hook forced first; got "
        f"{[b.get('beat_role') for b in summary['beats']]}"
    )
    print("[test]   generate_summary() re-derives 0-based indices from 1-indexed Gemini output, "
          "hook forced to index 0, scene_type/beat_role preserved OK")
    matched = json.loads(
        (settings.EXPLAINER_DIR / STEM / "en" / "matched.json").read_text(encoding="utf-8")
    )
    assert matched["beats"][0]["start"] == 2.0 and matched["beats"][0]["end"] == 5.0
    assert matched["beats"][1]["start"] == 20.0 and matched["beats"][1]["end"] == 35.5
    assert [b["scene_type"] for b in matched["beats"]] == ["emotional", "mystery"], (
        "scene_type must survive Stage 4's resolve_matches() untouched"
    )
    assert [b["beat_role"] for b in matched["beats"]] == ["hook", "body"], (
        "beat_role must survive Stage 4's resolve_matches() untouched"
    )
    print("[test]   Stage 3 -> Stage 4 file handoff correct (summary.json -> matched.json), "
          "scene_type/beat_role carried through")

    # --- Stage 5: mock texttospeech.TextToSpeechClient ---
    # Chirp3-HD names are bare VoiceSelectionParams(name=..., language_code=...)
    # with no model_name and no per-call style — assert the input carries no
    # `prompt` attribute (SynthesisInput(text=...) only), matching the removal
    # of the per-beat intensity/style-prompt system in the 2026-08-26 switch
    # to Chirp3-HD/Fenrir.
    class FakeTTSClient:
        def synthesize_speech(self, input, voice, audio_config):
            assert voice.name == settings.NARRATOR_VOICES["en"]
            assert not getattr(voice, "model_name", None), (
                "Chirp3-HD is not a Gemini model — model_name must not be set"
            )
            tmp = TEST_DIR / "_fake_tts_out.mp3"
            make_silence_mp3(tmp, 2.5)
            return SimpleNamespace(audio_content=tmp.read_bytes())

    with patch("google.cloud.texttospeech.TextToSpeechClient", lambda: FakeTTSClient()):
        stage5.synthesize_narration(STEM, video_path, audio_path, "en")
        stage5.synthesize_intro_narration(STEM, "en")

    intro_after_tts = json.loads(
        (settings.EXPLAINER_DIR / STEM / "en" / "intro.json").read_text(encoding="utf-8")
    )
    assert intro_after_tts["voice_duration"] > 0
    intro_voice_path = Path(intro_after_tts["voice_path"])
    assert intro_voice_path.name == "intro-voice.mp3", f"expected intro-voice.mp3, got {intro_voice_path.name}"
    assert intro_voice_path.exists()
    print("[test]   synthesize_intro_narration() writes intro-voice.mp3 into the SAME voices/ "
          "folder as voice-N.mp3, rewrites intro.json in place with voice_duration/voice_path")

    voiced = json.loads(
        (settings.EXPLAINER_DIR / STEM / "en" / "voiced.json").read_text(encoding="utf-8")
    )
    assert all(b.get("voice_path") for b in voiced["beats"])
    assert all("intensity" not in b for b in voiced["beats"]), (
        "per-beat intensity was removed with the Chirp3-HD/Fenrir switch — no beat should "
        "carry an 'intensity' field anymore"
    )
    assert [b["scene_type"] for b in voiced["beats"]] == ["emotional", "mystery"], (
        "scene_type must survive Stage 5's TTS synthesis untouched"
    )
    assert [b["beat_role"] for b in voiced["beats"]] == ["hook", "body"], (
        "beat_role must survive Stage 5's TTS synthesis untouched"
    )
    voice1 = Path(voiced["beats"][0]["voice_path"])
    assert voice1.name == "voice-1.mp3", f"expected voice-1.mp3, got {voice1.name}"
    print("[test]   Stage 4 -> Stage 5 file handoff correct (matched.json -> voiced.json), "
          "voice-N.mp3 naming confirmed")

    # --- Stage 6: real ffmpeg, no mocking needed ---
    results = stage6.retime_clips(STEM, video_path, "en")
    assert len(results) == 2
    assert [r["scene_type"] for r in results] == ["emotional", "mystery"], (
        "scene_type must survive all the way to _final_beats.json"
    )
    assert [r["beat_role"] for r in results] == ["hook", "body"], (
        "beat_role must survive all the way to _final_beats.json"
    )
    for r in results:
        n = r["index"] + 1
        clip_path = Path(r["clip_path"])
        assert clip_path.name == f"Clip-{n}.mp4"
        assert clip_path.exists()
        assert not has_audio_stream(clip_path), f"{clip_path.name} should be video-only"
        assert r["voice_path"] and Path(r["voice_path"]).exists()
    print("[test]   Stage 5 -> Stage 6 file handoff correct (voiced.json -> final_beats.json), "
          "Clip-N.mp4 naming confirmed, video-only confirmed, scene_type carried through to the end")

    final_path = settings.EXPLAINER_DIR / STEM / "en" / "final_beats.json"
    assert final_path.exists()
    print(f"[test]   full chain complete -> {final_path}")

    # --- Intro Stage 6 (2026-08-27, new): real ffmpeg, no mocking needed ---
    intro_final = stage6.retime_intro_clip(STEM, video_path, "en")
    intro_clip_path = Path(intro_final["clip_path"])
    assert intro_clip_path.name == "Intro.mp4", f"expected Intro.mp4, got {intro_clip_path.name}"
    assert intro_clip_path.exists()
    assert intro_clip_path.parent.name == "clips", (
        "Intro.mp4 must live in the SAME clips/ folder as Clip-N.mp4, not a separate subfolder "
        "(explicit user choice)"
    )
    assert not has_audio_stream(intro_clip_path), "Intro.mp4 should be video-only, same as Clip-N.mp4"
    assert intro_voice_path.parent.name == "voices", (
        "intro-voice.mp3 must live in the SAME voices/ folder as voice-N.mp3, not a separate subfolder"
    )
    print("[test]   retime_intro_clip() writes Intro.mp4 into the SAME clips/ folder as Clip-N.mp4, "
          "video-only, rewrites intro.json in place with clip_path -- full intro chain complete")


def main():
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    video_path = TEST_DIR / "synthetic.mp4"
    audio_path = TEST_DIR / "synthetic.wav"
    print("[test] building synthetic 40s video...")
    make_synthetic_video(video_path, duration=40)
    _run(["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ar", "16000", "-ac", "1", str(audio_path)])
    video_duration = hl._probe_duration(video_path)

    test_stage3_unit()
    test_stage4_unit()
    test_intro_unit()
    test_subtitles_unit()
    test_subtitle_generate_translate_unit()
    test_audio_events_unit()
    test_multilang_voice_unit()
    test_stage6_plan_clip(video_duration)
    test_full_pipeline(video_path, audio_path)

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
