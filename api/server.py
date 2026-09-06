"""
Movie Review Pipeline — FastAPI server + dashboard.

Run:
    uvicorn api.server:app --host 0.0.0.0 --port 9999 --reload

Dashboard: http://localhost:9999/dashboard
API base:  http://localhost:9999/api/v1/pipeline

Two pipeline modes, selected on submit:
  "highlight" — Director-cut highlight reel. One output video; supports a
                free re-cut to a new target duration via /reselect.
  "explainer" — narrated recap. NO merged final video (2026-08-26 redesign)
                — per-beat Clip-N.mp4 (video-only) / voice-N.mp3 pairs,
                fetched individually via /beat-file, plus one separate
                Intro.mp4/intro-voice.mp3 pair (2026-08-27 intro feature)
                fetched via /intro-file.
"""
import asyncio
import shutil
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from api.jobs import job_manager, VALID_MODES
from api.subtitle_jobs import subtitle_job_manager
from pipeline import subtitles

app = FastAPI(title="Movie Review Pipeline")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

# One-time, idempotent folder-structure migration (2026-09-05) — moves any subtitle still
# sitting at the old {stem}.{language}.srt-next-to-the-movie location into the new
# data/Subtitle/{movie}/{language}_subtitle.srt structure. Runs once per server startup;
# a no-op once nothing old-style is left. See pipeline/subtitles.py's
# migrate_legacy_subtitles() docstring for exactly what does and doesn't get moved.
subtitles.migrate_legacy_subtitles()


@app.get("/dashboard/subtitle")
def subtitle_library_page():
    """Standalone "Subtitle Library" page (2026-09-05, user's own proposal, replacing the
    in-line per-job detect/generate/translate flow that turned out to hang in practice).
    Registered as an explicit route BEFORE the /dashboard StaticFiles mount below — Starlette
    matches routes in registration order, and the mount would otherwise intercept this path
    first (it matches any /dashboard/* prefix) and 404 looking for a literal file named
    "subtitle" inside dashboard/."""
    return FileResponse(str(DASHBOARD_DIR / "subtitle.html"))


app.mount("/dashboard", StaticFiles(directory=str(DASHBOARD_DIR), html=True), name="dashboard")


@app.get("/")
def root():
    return {"message": "Movie Review Pipeline API. Dashboard at /dashboard, API at /api/v1/pipeline"}


@app.post("/api/v1/pipeline/submit")
async def submit_movie(file: UploadFile = File(...), mode: str = Form("highlight"),
                        language: str = Form("en"),
                        srt_file: Optional[UploadFile] = File(None)):
    """Upload an MP4, saves it to data/input/, kicks off the chosen pipeline
    (mode="highlight" or "explainer") in the background.

    language (2026-08-27, multi-language redesign): only meaningful for
    Explainer mode — the ONE language the whole job runs in (dashboard
    ID/EXPLAINER_LANGUAGE_NAMES for what's offered). Ignored for highlight
    mode, which is English-only.

    srt_file (optional, 2026-08-27): a real subtitle file — plain or SDH, same
    .srt format either way — saved next to the video under the VIDEO's stem
    PLUS the language (e.g. Movie.hi.srt), matching the convention
    pipeline/subtitles.py's find_subtitle_file() looks for. Only Explainer
    mode actually uses it (skips Whisper transcription when present).
    REQUIRED for Hindi/Bengali (settings.EXPLAINER_SUBTITLE_REQUIRED) —
    Whisper cannot substitute for a missing translated subtitle, since it
    transcribes whatever language is actually spoken in the source audio,
    not the requested output language. CHANGED 2026-09-05: this requirement
    can now ALSO be satisfied by a subtitle already sitting on disk under
    the matching {stem}.{language}.srt name — i.e. one produced by
    /subtitle/generate or /subtitle/translate before this submit — not just
    a file freshly attached to this exact request."""
    if not file.filename.lower().endswith((".mp4", ".mov", ".mkv")):
        raise HTTPException(400, "Only .mp4/.mov/.mkv files are supported")
    if mode not in VALID_MODES:
        raise HTTPException(400, f"Unknown mode: {mode!r} (expected one of {VALID_MODES})")

    if mode == "explainer":
        if language not in settings.EXPLAINER_LANGUAGE_NAMES:
            raise HTTPException(400, f"Unsupported language: {language!r} (expected one of "
                                      f"{list(settings.EXPLAINER_LANGUAGE_NAMES)})")
        if not settings.NARRATOR_VOICES.get(language):
            raise HTTPException(400, f"No narrator voice configured yet for "
                                      f"{settings.EXPLAINER_LANGUAGE_NAMES[language]} — pick one "
                                      f"in config/settings.py's NARRATOR_VOICES before submitting "
                                      f"a job in this language.")

    dest = settings.INPUT_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    srt_saved = None
    if srt_file is not None and srt_file.filename:
        if not srt_file.filename.lower().endswith(".srt"):
            raise HTTPException(400, "Subtitle file must be a .srt")
        # CHANGED 2026-09-05 (folder-structure reorg): saved into the same per-movie
        # data/Subtitle/{movie}/ folder as generated/translated subtitles, not next to the
        # video — one convention for a subtitle regardless of how it was produced.
        srt_dest = subtitles.subtitle_path_for(dest, language)
        with srt_dest.open("wb") as f:
            shutil.copyfileobj(srt_file.file, f)
        srt_saved = srt_dest.name

    if mode == "explainer" and settings.EXPLAINER_SUBTITLE_REQUIRED.get(language, False):
        existing = subtitles.find_subtitle_file(dest, language)
        if existing is None:
            raise HTTPException(400, f"A {settings.EXPLAINER_LANGUAGE_NAMES[language]} subtitle "
                                      f"(.srt) is required for this language — Whisper cannot "
                                      f"substitute for a missing translated subtitle. Attach one, "
                                      f"or generate one on the Subtitle Library page (/dashboard/subtitle).")
        srt_saved = srt_saved or existing.name

    job_id = job_manager.create_job(dest, mode=mode, language=language)
    return {"job_id": job_id, "filename": file.filename, "mode": mode, "language": language,
            "status": "queued", "srt_file": srt_saved}


@app.post("/api/v1/pipeline/detect-language")
async def detect_language(file: UploadFile = File(...)):
    """New (2026-09-05): quick language-ID pass on an uploaded movie, BEFORE
    the user commits to a full submit — lets the dashboard show "Detected:
    English" (with a manual override) as soon as a file is picked. Saves the
    file to data/input/ under its real filename (same destination /submit
    itself uses) so a later /submit for the same file just overwrites it
    with identical bytes — harmless, avoids needing a separate temp-upload
    area for a single-user local tool. Does NOT transcribe — see
    pipeline/subtitles.py's detect_language() for why this is fast."""
    if not file.filename.lower().endswith((".mp4", ".mov", ".mkv")):
        raise HTTPException(400, "Only .mp4/.mov/.mkv files are supported")

    dest = settings.INPUT_DIR / file.filename
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    language, probability = subtitles.detect_language(dest)
    return {"filename": file.filename, "language": language, "probability": probability}


def _save_upload_sync(file_obj, dest: Path):
    with dest.open("wb") as f:
        shutil.copyfileobj(file_obj, f)


@app.post("/api/v1/pipeline/upload-movie")
async def upload_movie(file: UploadFile = File(...)):
    """New (2026-09-05), for the Subtitle Library page: save a movie to data/input/ WITHOUT
    doing anything else (no language detection, no job). Runs the file copy in a worker
    thread via asyncio.to_thread — unlike /detect-language and /submit above (both still
    do a synchronous shutil.copyfileobj() directly inside their async def handler, which
    blocks the whole event loop for the entire copy), so a large movie upload here doesn't
    stall any other request this single-process server is handling at the same time. Once
    saved, the file is usable by /subtitle/detect, /subtitle/generate-all, and /submit."""
    if not file.filename.lower().endswith((".mp4", ".mov", ".mkv")):
        raise HTTPException(400, "Only .mp4/.mov/.mkv files are supported")
    dest = settings.INPUT_DIR / file.filename
    await asyncio.to_thread(_save_upload_sync, file.file, dest)
    return {"filename": file.filename}


@app.get("/api/v1/pipeline/movies")
def list_movies():
    """New (2026-09-05): every movie already sitting in data/input/ (from a past /submit,
    /detect-language, or /upload-movie), plus which of the 3 supported languages already
    have a subtitle saved for it — lets the Subtitle Library page offer "pick an existing
    movie" as an alternative to uploading one again, and show what's already covered."""
    movies = []
    for p in sorted(settings.INPUT_DIR.iterdir()):
        if not p.is_file() or not p.name.lower().endswith((".mp4", ".mov", ".mkv")):
            continue
        by_language = {}
        for lang in settings.EXPLAINER_LANGUAGES:
            found = subtitles.find_subtitle_file(p, lang)
            by_language[lang] = found.name if found else None
        movies.append({"filename": p.name, "subtitles": by_language})
    return {"movies": movies}


@app.post("/api/v1/pipeline/subtitle/detect")
async def start_subtitle_detect(filename: str = Form(...)):
    """New (2026-09-05), job-based replacement of /detect-language for the Subtitle Library
    page: runs subtitles.detect_language() as a background job (poll /subtitle-jobs/{id})
    instead of blocking the request until it finishes. Used to pre-fill a guessed movie
    language for the user to confirm/override before generating."""
    video_path = settings.INPUT_DIR / filename
    if not video_path.exists():
        raise HTTPException(404, f"No such file in data/input/: {filename!r} — upload it first "
                                  f"(e.g. via /upload-movie).")
    job_id = subtitle_job_manager.start_detect(video_path)
    return {"job_id": job_id}


@app.post("/api/v1/pipeline/subtitle/generate-all")
async def start_subtitle_generate_all(filename: str = Form(...), language: Optional[str] = Form(None)):
    """New (2026-09-05): the Subtitle Library page's single "Generate Subtitle" button —
    produces a real, reusable .srt for ALL THREE supported languages in one job (generate in
    the movie's own language, then translate to the other two). `language` empty/omitted lets
    the job auto-detect; a confirmed code (from /subtitle/detect, or the user overriding it)
    pins Whisper to it directly for better accuracy. Poll /subtitle-jobs/{id} — its `steps`
    list reports live per-stage progress (detect/generate/translate x2)."""
    video_path = settings.INPUT_DIR / filename
    if not video_path.exists():
        raise HTTPException(404, f"No such file in data/input/: {filename!r} — upload it first "
                                  f"(e.g. via /upload-movie).")
    if language and language not in settings.EXPLAINER_LANGUAGES:
        raise HTTPException(400, f"Unsupported language: {language!r} (expected one of "
                                  f"{settings.EXPLAINER_LANGUAGES})")
    job_id = subtitle_job_manager.start_generate_all(video_path, language or None)
    return {"job_id": job_id}


@app.get("/api/v1/pipeline/subtitle-exists")
def subtitle_exists(filename: str, language: str):
    """New (2026-09-05): does a {filename-stem}.{language}.srt already exist
    next to this video (uploaded, generated, or translated)? Lets the
    dashboard decide whether to offer "Generate" / "Translate" prompts
    without guessing from job state."""
    video_path = settings.INPUT_DIR / filename
    found = subtitles.find_subtitle_file(video_path, language)
    return {"exists": found is not None, "srt_filename": found.name if found else None}


@app.post("/api/v1/pipeline/subtitle/generate")
async def start_subtitle_generate(filename: str = Form(...), language: Optional[str] = Form(None)):
    """New (2026-09-05): kick off pipeline/subtitles.py's generate_subtitle()
    as a background job — Whisper transcription of a movie already sitting
    in data/input/ (via a prior /detect-language or /submit call), written
    out as a real .srt. `language` empty/omitted means auto-detect. Poll
    /subtitle-jobs/{id} for progress; this takes as long as a normal
    transcription (same "can take a while on CPU" cost as English mode's
    existing silent fallback), so it's a background job, not synchronous."""
    video_path = settings.INPUT_DIR / filename
    if not video_path.exists():
        raise HTTPException(404, f"No such file in data/input/: {filename!r} — upload it first "
                                  f"(e.g. via /detect-language or /submit).")
    job_id = subtitle_job_manager.start_generate(video_path, language or None)
    return {"job_id": job_id}


@app.post("/api/v1/pipeline/subtitle/translate")
async def start_subtitle_translate(filename: str = Form(...), source_language: str = Form(...),
                                    target_language: str = Form(...)):
    """New (2026-09-05): kick off pipeline/subtitles.py's translate_subtitle()
    as a background job — translates an existing {filename-stem}.
    {source_language}.srt into {target_language}, same timestamps. Poll
    /subtitle-jobs/{id} for progress."""
    video_path = settings.INPUT_DIR / filename
    if not video_path.exists():
        raise HTTPException(404, f"No such file in data/input/: {filename!r} — upload it first "
                                  f"(e.g. via /detect-language or /submit).")
    job_id = subtitle_job_manager.start_translate(video_path, source_language, target_language)
    return {"job_id": job_id}


@app.get("/api/v1/pipeline/subtitle-jobs/{job_id}")
def get_subtitle_job(job_id: str):
    job = subtitle_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"No such subtitle job: {job_id}")
    return job


@app.get("/api/v1/pipeline/jobs")
def list_jobs():
    return job_manager.list_jobs()


@app.get("/api/v1/pipeline/jobs/{job_id}")
def get_job(job_id: str):
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"No such job: {job_id}")
    return job


@app.post("/api/v1/pipeline/jobs/{job_id}/reselect")
def reselect(job_id: str, target_duration_sec: float = None, min_score_floor: float = None):
    """Re-run selection + assembly only, with a new target duration and/or
    quality floor. Reuses the already-computed transcript/signals/scores —
    no re-transcription, no new paid Vertex calls. Fast (concat, not re-encode
    crossfades). Highlight-reel mode only."""
    try:
        return job_manager.reselect(job_id, target_duration_sec, min_score_floor)
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/v1/pipeline/jobs/{job_id}/download")
def download(job_id: str):
    """Highlight-reel mode only — the single finished rough-cut. Explainer
    mode has no merged output; use /beat-file for each Clip-N.mp4/voice-N.mp3."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"No such job: {job_id}")

    if job.get("mode") == "explainer":
        raise HTTPException(400, "Explainer jobs have no single merged output — use "
                                  "/jobs/{id}/beat-file?index=N&kind=clip|voice for each beat")

    if not job.get("output_path"):
        raise HTTPException(409, "Job has no output yet")
    return FileResponse(job["output_path"], media_type="video/mp4",
                         filename=Path(job["output_path"]).name)


@app.get("/api/v1/pipeline/jobs/{job_id}/beat-file")
def download_beat_file(job_id: str, index: int, kind: str, language: str = None):
    """Explainer mode only: fetch one beat's Clip-N.mp4 (kind='clip', video-only)
    or voice-N.mp3 (kind='voice'). `index` is the beat's 0-based index (as in
    job['outputs'][language]), not the 1-based N in the filename."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"No such job: {job_id}")
    if job.get("mode") != "explainer":
        raise HTTPException(400, "Beat-file download is only available for explainer jobs")
    if kind not in ("clip", "voice"):
        raise HTTPException(400, f"Unknown kind {kind!r} (expected 'clip' or 'voice')")

    outputs = job.get("outputs") or {}
    if not outputs:
        raise HTTPException(409, "Job has no output yet")
    lang = language or next(iter(outputs))
    beats = outputs.get(lang)
    if beats is None:
        raise HTTPException(404, f"No output for language {lang!r} (have: {list(outputs)})")
    beat = next((b for b in beats if b["index"] == index), None)
    if not beat:
        raise HTTPException(404, f"No beat with index {index}")

    path = beat.get(f"{kind}_path")
    if not path:
        raise HTTPException(409, f"No {kind} file for beat {index}")
    media_type = "video/mp4" if kind == "clip" else "audio/mpeg"
    return FileResponse(path, media_type=media_type, filename=Path(path).name)


@app.get("/api/v1/pipeline/jobs/{job_id}/intro-file")
def download_intro_file(job_id: str, kind: str, language: str = None):
    """Explainer mode only, new 2026-08-27 (intro feature): fetch the intro's
    Intro.mp4 (kind='clip', video-only) or intro-voice.mp3 (kind='voice').
    Mirrors /beat-file above but with no `index` param — there is exactly one
    intro object per language (job['intro'][language]), not a list of beats."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"No such job: {job_id}")
    if job.get("mode") != "explainer":
        raise HTTPException(400, "Intro-file download is only available for explainer jobs")
    if kind not in ("clip", "voice"):
        raise HTTPException(400, f"Unknown kind {kind!r} (expected 'clip' or 'voice')")

    intro_by_language = job.get("intro") or {}
    if not intro_by_language:
        raise HTTPException(409, "Job has no intro output yet")
    lang = language or next(iter(intro_by_language))
    intro = intro_by_language.get(lang)
    if intro is None:
        raise HTTPException(404, f"No intro for language {lang!r} (have: {list(intro_by_language)})")

    path = intro.get(f"{kind}_path")
    if not path:
        raise HTTPException(409, f"No {kind} file for the intro")
    media_type = "video/mp4" if kind == "clip" else "audio/mpeg"
    return FileResponse(path, media_type=media_type, filename=Path(path).name)


@app.get("/api/v1/pipeline/jobs/{job_id}/scores")
def get_scores(job_id: str):
    """Return the per-segment scored data for this job, for the dashboard's
    score list. Highlight-reel mode only — explainer jobs never produce a
    scored.json (no Stage 6 scoring in that pipeline)."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, f"No such job: {job_id}")
    if job.get("mode") == "explainer":
        raise HTTPException(409, "Scores are only available for highlight-reel jobs")
    scored_path = settings.ANALYSIS_DIR / f"{job['video_stem']}_scored.json"
    if not scored_path.exists():
        raise HTTPException(409, "Scores not available yet")
    import json
    return JSONResponse(json.loads(scored_path.read_text(encoding="utf-8")))
