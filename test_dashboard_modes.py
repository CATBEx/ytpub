"""
Sandbox integration test for dashboard/API mode support, updated for the
2026-08-26 Explainer redesign (per-beat Clip-N.mp4/voice-N.mp3 outputs, no
merged final video). Monkeypatches the actual pipeline stage functions with
fast stand-ins (no real ffmpeg/whisper/Gemini/TTS calls) so this exercises
the job-manager/API wiring itself: mode validation, per-mode job dict shape,
background-thread execution, the /beat-file routing, the /download mode
guard, and the reselect/scores mode guards.

Run: python test_dashboard_modes.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings

# --- monkeypatch every pipeline stage function BEFORE importing api.jobs,
# so api.jobs binds to the same (now-patched) module objects. ---
from pipeline import stage1_ingest, stage2_transcribe, stage3_shots
from pipeline import stage4_signals, stage5_score, stage6_select, stage7_assemble
from pipeline import explainer_stage3_summary, explainer_stage4_match, explainer_intro
from pipeline import explainer_stage5_tts, explainer_stage6_assemble
from pipeline import subtitles

CALLS = []


def fake_extract_audio(video_path):
    CALLS.append("extract_audio")
    return Path("/tmp/fake_audio.wav")


def fake_transcribe(audio_path):
    CALLS.append("transcribe")
    return Path("/tmp/fake_transcript.json")


def fake_detect_shots(video_path):
    CALLS.append("detect_shots")
    return Path("/tmp/fake_shots.json")


def fake_extract_signals(video_path, audio_path, shots_path):
    CALLS.append("extract_signals")
    return Path("/tmp/fake_signals.json")


def fake_score_segments(stem):
    CALLS.append("score_importance")
    return Path("/tmp/fake_scored.json")


def fake_select_segments(stem):
    CALLS.append("select_segments")
    return Path("/tmp/fake_selected.json")


def fake_assemble(stem, video_path):
    CALLS.append("assemble")
    out = settings.OUTPUT_DIR / f"{stem}_highlight_FAKE.mp4"
    out.write_bytes(b"fake highlight mp4 bytes")
    return out


def fake_generate_summary(stem, language="en"):
    CALLS.append("generate_summary")
    return Path("/tmp/fake_summary.json")


def fake_generate_intro(stem, language="en"):
    CALLS.append("generate_intro")
    return Path("/tmp/fake_intro.json")


def fake_match_clips(stem, language="en"):
    CALLS.append("match_clips")
    return Path("/tmp/fake_matched.json")


def fake_synthesize_narration(stem, video_path, audio_path, language="en"):
    CALLS.append("synthesize_narration")
    return Path("/tmp/fake_voiced.json")


def fake_synthesize_intro_narration(stem, language="en"):
    CALLS.append("synthesize_intro_narration")
    return Path("/tmp/fake_intro_voiced.json")


def fake_retime_clips(stem, video_path, language="en"):
    CALLS.append("retime_clips")
    results = []
    for i in range(2):
        clip_path = settings.OUTPUT_DIR / f"{stem}_Clip-{i + 1}_FAKE.mp4"
        voice_path = settings.OUTPUT_DIR / f"{stem}_voice-{i + 1}_FAKE.mp3"
        clip_path.write_bytes(f"fake clip {i} bytes".encode())
        voice_path.write_bytes(f"fake voice {i} bytes".encode())
        results.append({
            "index": i,
            "narration": f"fake narration for beat {i}",
            "clip_path": str(clip_path),
            "voice_path": str(voice_path),
        })
    return results


def fake_retime_intro_clip(stem, video_path, language="en"):
    CALLS.append("retime_intro_clip")
    clip_path = settings.OUTPUT_DIR / f"{stem}_Intro_FAKE.mp4"
    voice_path = settings.OUTPUT_DIR / f"{stem}_intro-voice_FAKE.mp3"
    clip_path.write_bytes(b"fake intro clip bytes")
    voice_path.write_bytes(b"fake intro voice bytes")
    return {
        "narration": "fake intro narration, describing the premise without the title",
        "cited_segments": [0],
        "start": 0.0,
        "end": 5.0,
        "voice_duration": 5.0,
        "voice_path": str(voice_path),
        "clip_path": str(clip_path),
    }


def fake_detect_language(media_path):
    CALLS.append("detect_language")
    return ("hi", 0.95)


def fake_generate_subtitle(video_path, language=None):
    CALLS.append("generate_subtitle")
    used_language = language or "hi"
    # Uses the REAL (unpatched) subtitle_path_for() -- only detect/generate/translate
    # themselves are faked here, so this fake writes to the same per-movie folder structure
    # the real generate_subtitle() would (2026-09-05 folder reorg).
    srt_path = subtitles.subtitle_path_for(video_path, used_language)
    srt_path.write_text("1\n00:00:00,000 --> 00:00:02,000\nfake generated line\n", encoding="utf-8")
    return srt_path, used_language, (None if language else 0.95)


def fake_translate_subtitle(video_path, source_language, target_language):
    CALLS.append("translate_subtitle")
    out_path = subtitles.subtitle_path_for(video_path, target_language)
    out_path.write_text("1\n00:00:00,000 --> 00:00:02,000\nfake translated line\n", encoding="utf-8")
    return out_path


stage1_ingest.extract_audio = fake_extract_audio
stage2_transcribe.transcribe = fake_transcribe
stage3_shots.detect_shots = fake_detect_shots
stage4_signals.extract_signals = fake_extract_signals
stage5_score.score_segments = fake_score_segments
stage6_select.select_segments = fake_select_segments
stage7_assemble.assemble = fake_assemble
explainer_stage3_summary.generate_summary = fake_generate_summary
explainer_intro.generate_intro = fake_generate_intro
explainer_stage4_match.match_clips = fake_match_clips
explainer_stage5_tts.synthesize_narration = fake_synthesize_narration
explainer_stage5_tts.synthesize_intro_narration = fake_synthesize_intro_narration
explainer_stage6_assemble.retime_clips = fake_retime_clips
explainer_stage6_assemble.retime_intro_clip = fake_retime_intro_clip
subtitles.detect_language = fake_detect_language
subtitles.generate_subtitle = fake_generate_subtitle
subtitles.translate_subtitle = fake_translate_subtitle

from fastapi.testclient import TestClient
from api.server import app

client = TestClient(app)

TEST_DIR = settings.DATA_DIR / "dashboard_sandbox_test"
TEST_DIR.mkdir(parents=True, exist_ok=True)


def make_test_upload(name: str) -> Path:
    p = TEST_DIR / name
    p.write_bytes(b"not a real mp4, just needs a .mp4 extension for the endpoint check")
    return p


def wait_for_terminal(job_id: str, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/api/v1/pipeline/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            return job
        time.sleep(0.1)
    raise TimeoutError(f"job {job_id} did not reach a terminal state in {timeout}s")


def main():
    print("[test] 1. Invalid mode is rejected...")
    upload = make_test_upload("bad_mode.mp4")
    with upload.open("rb") as f:
        res = client.post("/api/v1/pipeline/submit", files={"file": (upload.name, f, "video/mp4")},
                           data={"mode": "not_a_real_mode"})
    assert res.status_code == 400, f"expected 400, got {res.status_code}: {res.text}"
    print("[test]    OK -> 400 as expected")

    print("\n[test] 2. Submit with mode=explainer runs the explainer pipeline...")
    CALLS.clear()
    upload = make_test_upload("explainer_test.mp4")
    with upload.open("rb") as f:
        res = client.post("/api/v1/pipeline/submit", files={"file": (upload.name, f, "video/mp4")},
                           data={"mode": "explainer"})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["mode"] == "explainer"
    job_id = data["job_id"]
    job = wait_for_terminal(job_id)
    assert job["status"] == "completed", f"job failed: {job.get('error')}"
    assert job["mode"] == "explainer"
    assert job["outputs"], "expected a populated outputs dict"
    assert job["language"] == "en", "no language was specified on submit -> should default to en"
    lang = job["language"]
    assert lang in job["outputs"]
    beats = job["outputs"][lang]
    assert len(beats) == 2
    assert beats[0]["clip_path"] and beats[0]["voice_path"]
    assert job["output_path"] is None, "highlight-only field should stay empty for explainer jobs"
    # Intro feature (2026-08-27, new): a separate job["intro"] = {language: intro_dict} field,
    # parallel to but distinct from job["outputs"] (a list of beats) since there's exactly one
    # intro object per language.
    assert job["intro"], "expected a populated intro dict"
    assert lang in job["intro"]
    intro = job["intro"][lang]
    assert intro["clip_path"] and intro["voice_path"]
    assert CALLS == ["extract_audio", "transcribe", "generate_summary", "generate_intro",
                      "match_clips", "synthesize_narration", "synthesize_intro_narration",
                      "retime_clips", "retime_intro_clip"], f"unexpected call order: {CALLS}"
    print(f"[test]    OK -> {len(beats)} beat pairs + 1 intro pair for language {lang!r}")

    print("\n[test] 2b. Explainer submit with an unsupported language is rejected...")
    upload = make_test_upload("bad_lang.mp4")
    with upload.open("rb") as f:
        res = client.post("/api/v1/pipeline/submit", files={"file": (upload.name, f, "video/mp4")},
                           data={"mode": "explainer", "language": "fr"})
    assert res.status_code == 400, f"expected 400, got {res.status_code}: {res.text}"
    print("[test]     OK -> 400 as expected")

    print("\n[test] 2c. Explainer submit for Hindi with no subtitle is rejected (required)...")
    upload = make_test_upload("hi_no_srt.mp4")
    with upload.open("rb") as f:
        res = client.post("/api/v1/pipeline/submit", files={"file": (upload.name, f, "video/mp4")},
                           data={"mode": "explainer", "language": "hi"})
    assert res.status_code == 400, f"expected 400, got {res.status_code}: {res.text}"
    print(f"[test]     OK -> 400: {res.json()['detail']}")

    print("\n[test] 2d. Explainer submit for Hindi WITH a subtitle succeeds, runs natively in hi...")
    CALLS.clear()
    upload = make_test_upload("hi_with_srt.mp4")
    srt_content = "1\n00:00:00,000 --> 00:00:02,000\nनमस्ते\n"
    srt_path = TEST_DIR / "hi_srt_upload.srt"
    srt_path.write_text(srt_content, encoding="utf-8")
    with upload.open("rb") as f_video, srt_path.open("rb") as f_srt:
        res = client.post(
            "/api/v1/pipeline/submit",
            files={"file": (upload.name, f_video, "video/mp4"),
                   "srt_file": (srt_path.name, f_srt, "text/plain")},
            data={"mode": "explainer", "language": "hi"},
        )
    assert res.status_code == 200, res.text
    data_hi = res.json()
    assert data_hi["language"] == "hi"
    assert data_hi["srt_file"] == "hi_subtitle.srt", (
        f"subtitle should be saved under the per-movie folder as hi_subtitle.srt, "
        f"got {data_hi['srt_file']!r}"
    )
    job_hi = wait_for_terminal(data_hi["job_id"])
    assert job_hi["status"] == "completed", f"job failed: {job_hi.get('error')}"
    assert job_hi["language"] == "hi"
    assert "hi" in job_hi["outputs"]
    assert "hi" in job_hi["intro"], "intro dict should also be keyed by 'hi' for a Hindi job"
    # transcribe (Whisper) must NOT have run -- a real .srt was supplied
    assert "transcribe" not in CALLS, f"Whisper should be skipped when a subtitle is provided: {CALLS}"
    print(f"[test]     OK -> Hindi job completed natively, outputs+intro keyed by 'hi', Whisper skipped")

    print("\n[test] 3. beat-file download (clip) works for both beats...")
    for beat in beats:
        res = client.get(f"/api/v1/pipeline/jobs/{job_id}/beat-file",
                          params={"index": beat["index"], "kind": "clip"})
        assert res.status_code == 200, res.text
        assert res.content == Path(beat["clip_path"]).read_bytes()
    print("[test]    OK")

    print("\n[test] 4. beat-file download (voice) works for both beats...")
    for beat in beats:
        res = client.get(f"/api/v1/pipeline/jobs/{job_id}/beat-file",
                          params={"index": beat["index"], "kind": "voice"})
        assert res.status_code == 200, res.text
        assert res.content == Path(beat["voice_path"]).read_bytes()
    print("[test]    OK")

    print("\n[test] 5. beat-file with unknown kind is rejected...")
    res = client.get(f"/api/v1/pipeline/jobs/{job_id}/beat-file", params={"index": 0, "kind": "bogus"})
    assert res.status_code == 400, res.text
    print("[test]    OK")

    print("\n[test] 6. beat-file with unknown beat index 404s...")
    res = client.get(f"/api/v1/pipeline/jobs/{job_id}/beat-file", params={"index": 999, "kind": "clip"})
    assert res.status_code == 404, res.text
    print("[test]    OK")

    print("\n[test] 7. beat-file with unknown language 404s...")
    res = client.get(f"/api/v1/pipeline/jobs/{job_id}/beat-file",
                      params={"index": 0, "kind": "clip", "language": "xx"})
    assert res.status_code == 404, res.text
    print("[test]    OK")

    print("\n[test] 7b. intro-file download (clip) works...")
    res = client.get(f"/api/v1/pipeline/jobs/{job_id}/intro-file", params={"kind": "clip"})
    assert res.status_code == 200, res.text
    assert res.content == Path(intro["clip_path"]).read_bytes()
    print("[test]     OK")

    print("\n[test] 7c. intro-file download (voice) works...")
    res = client.get(f"/api/v1/pipeline/jobs/{job_id}/intro-file", params={"kind": "voice"})
    assert res.status_code == 200, res.text
    assert res.content == Path(intro["voice_path"]).read_bytes()
    print("[test]     OK")

    print("\n[test] 7d. intro-file with unknown kind is rejected...")
    res = client.get(f"/api/v1/pipeline/jobs/{job_id}/intro-file", params={"kind": "bogus"})
    assert res.status_code == 400, res.text
    print("[test]     OK")

    print("\n[test] 7e. intro-file with unknown language 404s...")
    res = client.get(f"/api/v1/pipeline/jobs/{job_id}/intro-file", params={"kind": "clip", "language": "xx"})
    assert res.status_code == 404, res.text
    print("[test]     OK")

    print("\n[test] 8. Plain /download on an explainer job is rejected (no merged output)...")
    res = client.get(f"/api/v1/pipeline/jobs/{job_id}/download")
    assert res.status_code == 400, f"expected 400, got {res.status_code}: {res.text}"
    print(f"[test]    OK -> 400: {res.json()['detail']}")

    print("\n[test] 9. beat-file on a highlight-mode job is rejected...")
    # (submitted below, but check the guard logic exists — verified in step 11's job too)

    print("\n[test] 10. Reselect on an explainer job is rejected (mode guard)...")
    res = client.post(f"/api/v1/pipeline/jobs/{job_id}/reselect", params={"target_duration_sec": 300})
    assert res.status_code == 400, f"expected 400, got {res.status_code}: {res.text}"
    print(f"[test]    OK -> 400: {res.json()['detail']}")

    print("\n[test] 11. Scores endpoint on an explainer job is rejected...")
    res = client.get(f"/api/v1/pipeline/jobs/{job_id}/scores")
    assert res.status_code == 409, f"expected 409, got {res.status_code}: {res.text}"
    print("[test]    OK")

    print("\n[test] 12. Highlight-reel mode (default, no mode field) still works end-to-end...")
    CALLS.clear()
    upload = make_test_upload("highlight_test.mp4")
    with upload.open("rb") as f:
        res = client.post("/api/v1/pipeline/submit", files={"file": (upload.name, f, "video/mp4")})
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["mode"] == "highlight"
    job_id2 = data["job_id"]
    job2 = wait_for_terminal(job_id2)
    assert job2["status"] == "completed", f"job failed: {job2.get('error')}"
    assert job2["output_path"], "expected output_path to be set for highlight mode"
    assert job2["outputs"] is None, "explainer-only field should stay empty for highlight jobs"
    assert job2["intro"] is None, "explainer-only intro field should stay empty for highlight jobs"
    assert CALLS == ["extract_audio", "transcribe", "detect_shots", "extract_signals",
                      "score_importance", "select_segments", "assemble"], f"unexpected call order: {CALLS}"
    print(f"[test]    OK -> output_path={job2['output_path']}")

    print("\n[test] 13. Highlight-reel download still works...")
    res = client.get(f"/api/v1/pipeline/jobs/{job_id2}/download")
    assert res.status_code == 200, res.text
    assert res.content == Path(job2["output_path"]).read_bytes()
    print("[test]    OK")

    print("\n[test] 14. beat-file on a highlight-mode job is rejected...")
    res = client.get(f"/api/v1/pipeline/jobs/{job_id2}/beat-file", params={"index": 0, "kind": "clip"})
    assert res.status_code == 400, res.text
    print("[test]    OK")

    print("\n[test] 14b. intro-file on a highlight-mode job is rejected...")
    res = client.get(f"/api/v1/pipeline/jobs/{job_id2}/intro-file", params={"kind": "clip"})
    assert res.status_code == 400, res.text
    print("[test]     OK")

    print("\n[test] 15. Reselect on a highlight job still works (regression check)...")
    res = client.post(f"/api/v1/pipeline/jobs/{job_id2}/reselect", params={"target_duration_sec": 200})
    assert res.status_code == 200, res.text
    print("[test]    OK")

    print("\n[test] 16. Job list includes both jobs with correct mode fields...")
    jobs = client.get("/api/v1/pipeline/jobs").json()
    by_id = {j["id"]: j for j in jobs}
    assert by_id[job_id]["mode"] == "explainer"
    assert by_id[job_id2]["mode"] == "highlight"
    print("[test]    OK")

    # --- Subtitle generate/translate utilities (2026-09-05) --------------------------------

    def wait_for_subtitle_job(job_id: str, timeout=10):
        deadline = time.time() + timeout
        while time.time() < deadline:
            job = client.get(f"/api/v1/pipeline/subtitle-jobs/{job_id}").json()
            if job["status"] in ("completed", "failed"):
                return job
            time.sleep(0.1)
        raise TimeoutError(f"subtitle job {job_id} did not reach a terminal state in {timeout}s")

    print("\n[test] 17. /detect-language returns the (faked) detected language...")
    subtitle_upload = make_test_upload("subtitle_utils.mp4")
    with subtitle_upload.open("rb") as f:
        res = client.post("/api/v1/pipeline/detect-language",
                           files={"file": (subtitle_upload.name, f, "video/mp4")})
    assert res.status_code == 200, res.text
    detect_data = res.json()
    assert detect_data["filename"] == "subtitle_utils.mp4"
    assert detect_data["language"] == "hi" and detect_data["probability"] == 0.95
    assert "detect_language" in CALLS
    print(f"[test]     OK -> {detect_data}")

    print("\n[test] 18. /subtitle-exists is false before anything is generated...")
    res = client.get("/api/v1/pipeline/subtitle-exists",
                      params={"filename": "subtitle_utils.mp4", "language": "hi"})
    assert res.status_code == 200, res.text
    assert res.json()["exists"] is False
    print("[test]     OK")

    print("\n[test] 19. /subtitle/generate runs in the background and produces a real .srt...")
    res = client.post("/api/v1/pipeline/subtitle/generate",
                       data={"filename": "subtitle_utils.mp4", "language": "hi"})
    assert res.status_code == 200, res.text
    gen_job = wait_for_subtitle_job(res.json()["job_id"])
    assert gen_job["status"] == "completed", gen_job
    assert gen_job["result"]["language"] == "hi"
    res = client.get("/api/v1/pipeline/subtitle-exists",
                      params={"filename": "subtitle_utils.mp4", "language": "hi"})
    assert res.json()["exists"] is True
    print(f"[test]     OK -> {gen_job['result']}")

    print("\n[test] 20. /subtitle/translate runs in the background and produces a real .srt...")
    res = client.post("/api/v1/pipeline/subtitle/translate",
                       data={"filename": "subtitle_utils.mp4", "source_language": "hi", "target_language": "en"})
    assert res.status_code == 200, res.text
    trans_job = wait_for_subtitle_job(res.json()["job_id"])
    assert trans_job["status"] == "completed", trans_job
    res = client.get("/api/v1/pipeline/subtitle-exists",
                      params={"filename": "subtitle_utils.mp4", "language": "en"})
    assert res.json()["exists"] is True
    print(f"[test]     OK -> {trans_job['result']}")

    print("\n[test] 21. /subtitle-jobs/{unknown} 404s...")
    res = client.get("/api/v1/pipeline/subtitle-jobs/doesnotexist")
    assert res.status_code == 404, res.text
    print("[test]     OK")

    print("\n[test] 22. /submit accepts a subtitle already on disk (from /subtitle/generate) -- "
          "no fresh srt_file upload required for THIS request...")
    CALLS.clear()
    with subtitle_upload.open("rb") as f:
        res = client.post("/api/v1/pipeline/submit", files={"file": (subtitle_upload.name, f, "video/mp4")},
                           data={"mode": "explainer", "language": "hi"})
    assert res.status_code == 200, res.text
    submit_data = res.json()
    assert submit_data["srt_file"] == "hi_subtitle.srt", (
        f"expected the already-generated hi_subtitle.srt to be picked up automatically, got "
        f"{submit_data['srt_file']!r}"
    )
    job22 = wait_for_terminal(submit_data["job_id"])
    assert job22["status"] == "completed", job22
    print(f"[test]     OK -> submit succeeded using the pre-generated subtitle: {submit_data['srt_file']}")

    # --- Subtitle Library page backend (2026-09-05) -----------------------------------------
    # The standalone /dashboard/subtitle page (the user's own proposal, replacing the in-line
    # per-job detect/generate/translate flow above, which hung in practice): upload/pick a
    # movie once, confirm its language, one button generates all 3 languages' subtitles.

    print("\n[test] 23. /upload-movie saves the file without starting any job...")
    CALLS.clear()
    library_upload = make_test_upload("library_movie.mp4")
    with library_upload.open("rb") as f:
        res = client.post("/api/v1/pipeline/upload-movie",
                           files={"file": (library_upload.name, f, "video/mp4")})
    assert res.status_code == 200, res.text
    assert res.json() == {"filename": "library_movie.mp4"}
    assert CALLS == [], f"upload-movie should not touch detect/generate/translate, got {CALLS}"
    print("[test]     OK")

    print("\n[test] 24. /movies lists it with no subtitles yet, alongside earlier test movies...")
    res = client.get("/api/v1/pipeline/movies")
    assert res.status_code == 200, res.text
    movies = {m["filename"]: m["subtitles"] for m in res.json()["movies"]}
    assert "library_movie.mp4" in movies, movies
    assert movies["library_movie.mp4"] == {"en": None, "hi": None, "bn": None}
    # subtitle_utils.mp4 (test 19/20) already has .hi.srt and .en.srt on disk by now
    assert movies["subtitle_utils.mp4"]["hi"] and movies["subtitle_utils.mp4"]["en"]
    print(f"[test]     OK -> {movies['library_movie.mp4']}")

    print("\n[test] 25. /subtitle/detect is job-based (not synchronous) and reuses detect_language()...")
    res = client.post("/api/v1/pipeline/subtitle/detect", data={"filename": "library_movie.mp4"})
    assert res.status_code == 200, res.text
    detect_job = wait_for_subtitle_job(res.json()["job_id"])
    assert detect_job["status"] == "completed", detect_job
    assert detect_job["result"] == {"language": "hi", "probability": 0.95}
    print(f"[test]     OK -> {detect_job['result']}")

    print("\n[test] 26. /subtitle/generate-all with a confirmed language generates + translates "
          "to the other two, all in one job, with live per-step progress...")
    CALLS.clear()
    res = client.post("/api/v1/pipeline/subtitle/generate-all",
                       data={"filename": "library_movie.mp4", "language": "hi"})
    assert res.status_code == 200, res.text
    all_job = wait_for_subtitle_job(res.json()["job_id"], timeout=15)
    assert all_job["status"] == "completed", all_job
    assert all_job["result"]["source_language"] == "hi"
    assert set(all_job["result"]["subtitles"]) == {"hi", "en", "bn"}, all_job["result"]
    # confirmed language -> no detect_language() call, straight to generate + 2 translates
    assert CALLS == ["generate_subtitle", "translate_subtitle", "translate_subtitle"], CALLS
    step_labels = [s["label"] for s in all_job["steps"]]
    assert any("Generating" in l for l in step_labels) and sum("Translating" in l for l in step_labels) == 2, step_labels
    assert all(s["status"] == "done" for s in all_job["steps"]), all_job["steps"]
    res = client.get("/api/v1/pipeline/movies")
    movies = {m["filename"]: m["subtitles"] for m in res.json()["movies"]}
    assert all(movies["library_movie.mp4"][l] for l in ("en", "hi", "bn")), movies["library_movie.mp4"]
    print(f"[test]     OK -> {all_job['result']}")

    print("\n[test] 27. /subtitle/generate-all with no language auto-detects first (extra step)...")
    CALLS.clear()
    auto_upload = make_test_upload("library_movie_auto.mp4")
    with auto_upload.open("rb") as f:
        client.post("/api/v1/pipeline/upload-movie", files={"file": (auto_upload.name, f, "video/mp4")})
    res = client.post("/api/v1/pipeline/subtitle/generate-all", data={"filename": "library_movie_auto.mp4"})
    assert res.status_code == 200, res.text
    auto_job = wait_for_subtitle_job(res.json()["job_id"], timeout=15)
    assert auto_job["status"] == "completed", auto_job
    assert CALLS == ["detect_language", "generate_subtitle", "translate_subtitle", "translate_subtitle"], CALLS
    print(f"[test]     OK -> {auto_job['result']}")

    print("\n[test] 28. /dashboard/subtitle serves the standalone Subtitle Library page "
          "(not intercepted by the /dashboard StaticFiles mount)...")
    res = client.get("/api/v1/pipeline/movies")  # sanity: API still fine after page checks below
    assert res.status_code == 200
    res = client.get("/dashboard/subtitle")
    assert res.status_code == 200, res.text
    assert "Subtitle Library" in res.text
    print("[test]     OK")

    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
