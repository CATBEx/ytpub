"""
In-memory job store + background pipeline runner.

Single-user, local, run-on-your-own-machine tool — no Redis/Celery, just a
dict guarded by a lock and one thread per job. Good enough for "submit a
movie, watch it process, grab the result" on one machine.

Two job modes, same job-store/threading machinery:
  "highlight" — the original Director-cut pipeline. Single output file
                (job["output_path"]), supports re-selecting to a new target
                duration without re-running the paid stages.
  "explainer" — the narrated recap pipeline (redesigned 2026-08-26 — NO
                merged final video by explicit design). Produces per-beat
                Clip-N.mp4 (video-only) / voice-N.mp3 pairs, in story order.
                CHANGED 2026-08-27 (multi-language redesign): each job now
                picks ONE language up front (job["language"]) and runs
                every stage natively in it, from a subtitle supplied in
                that language (required for hi/bn — see
                pipeline/subtitles.py). job["outputs"] stays shaped as
                {language: [beat_dict, ...]} for compatibility with the
                dashboard's existing rendering, just with exactly one key
                per job now instead of one key per
                settings.EXPLAINER_LANGUAGES entry. No re-select
                equivalent: every stage is either cheap-and-deterministic
                (retiming) or paid-and-not-worth-recomputing-for-a-tweak
                (summary/match/TTS), so there's no free re-cut like
                highlight mode has.

Limitation: jobs are keyed by video filename stem, matching the CLI/data/
layout from main.py. Uploading two different files with the same name will
overwrite each other's intermediate files. Fine for personal use; would need
per-job subfolders to support real concurrent multi-user use.
"""
import threading
import time
import traceback
import uuid
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from pipeline import stage1_ingest, stage2_transcribe, stage3_shots
from pipeline import stage4_signals, stage5_score, stage6_select, stage7_assemble
from pipeline import explainer_stage3_summary, explainer_stage4_match, explainer_intro
from pipeline import explainer_stage5_tts, explainer_stage6_assemble
from pipeline import subtitles

HIGHLIGHT_STAGE_NAMES = [
    "extract_audio",
    "transcribe",
    "detect_shots",
    "extract_signals",
    "score_importance",
    "select_segments",
    "assemble",
]
EXPLAINER_STAGE_NAMES = [
    "extract_audio",
    "transcribe",
    "generate_summary",
    "generate_intro",       # new 2026-08-27 (intro feature) — its own stage, since it's a
                             # separate paid Gemini call; the TTS/retiming for it are folded
                             # into the existing synthesize_narration/retime_clips stages
                             # below rather than getting their own stage entries.
    "match_clips",
    "synthesize_narration",
    "retime_clips",
]

VALID_MODES = ("highlight", "explainer")


class JobManager:
    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()

    def _update(self, job_id, **fields):
        with self._lock:
            self._jobs[job_id].update(fields)
            self._jobs[job_id]["updated_at"] = time.time()

    def _log(self, job_id, message):
        with self._lock:
            self._jobs[job_id]["logs"].append(f"[{time.strftime('%H:%M:%S')}] {message}")
            self._jobs[job_id]["updated_at"] = time.time()

    def create_job(self, video_path: Path, mode: str = "highlight", language: str = "en") -> str:
        if mode not in VALID_MODES:
            raise ValueError(f"Unknown mode: {mode!r} (expected one of {VALID_MODES})")
        if mode == "explainer" and language not in settings.EXPLAINER_LANGUAGE_NAMES:
            raise ValueError(f"Unsupported language: {language!r} (expected one of "
                              f"{list(settings.EXPLAINER_LANGUAGE_NAMES)})")

        job_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "mode": mode,
                "filename": video_path.name,
                "video_stem": video_path.stem,
                "video_path": str(video_path),
                "status": "queued",
                "current_stage": None,
                "stages_completed": [],
                "error": None,
                "output_path": None,   # highlight mode result
                "outputs": None,       # explainer mode result: {language: [beat_dict, ...]}
                "intro": None,         # explainer mode result: {language: intro_dict} — new
                                        # 2026-08-27 (intro feature); separate from "outputs"
                                        # since there's exactly one intro object per language,
                                        # not a list of beats.
                "language": language if mode == "explainer" else None,  # single language this
                                          # job runs in (CHANGED 2026-08-27 -- was a list of
                                          # every supported language; now every job is one language)
                "target_duration_used": settings.TARGET_OUTPUT_DURATION if mode == "highlight" else None,
                "min_score_floor_used": settings.MIN_SCORE_FLOOR if mode == "highlight" else None,
                "logs": [],
                "created_at": time.time(),
                "updated_at": time.time(),
            }
        if mode == "explainer":
            thread = threading.Thread(target=self._run_explainer_job, args=(job_id, video_path, language), daemon=True)
        else:
            thread = threading.Thread(target=self._run_job, args=(job_id, video_path), daemon=True)
        thread.start()
        return job_id

    def _run_job(self, job_id: str, video_path: Path):
        """Highlight-reel ("Director" cut) pipeline — Stages 1-8."""
        self._update(job_id, status="running")
        try:
            self._log(job_id, "Stage 1/8: extracting audio (free, CPU)")
            self._update(job_id, current_stage="extract_audio")
            audio_path = stage1_ingest.extract_audio(video_path)
            self._mark_done(job_id, "extract_audio")

            self._log(job_id, "Stage 2/8: transcribing (free, CPU — can take a while)")
            self._update(job_id, current_stage="transcribe")
            stage2_transcribe.transcribe(audio_path)
            self._mark_done(job_id, "transcribe")

            self._log(job_id, "Stage 3/8: detecting shots (free, CPU)")
            self._update(job_id, current_stage="detect_shots")
            shots_path = stage3_shots.detect_shots(video_path)
            self._mark_done(job_id, "detect_shots")

            self._log(job_id, "Stage 4/5: extracting audio/visual signals (free, CPU)")
            self._update(job_id, current_stage="extract_signals")
            stage4_signals.extract_signals(video_path, audio_path, shots_path)
            self._mark_done(job_id, "extract_signals")

            self._log(job_id, "Stage 6: scoring importance (PAID — Vertex ADC)")
            self._update(job_id, current_stage="score_importance")
            stage5_score.score_segments(video_path.stem)
            self._mark_done(job_id, "score_importance")

            self._log(job_id, f"Stage 7: selecting segments "
                               f"(target={settings.TARGET_OUTPUT_DURATION}s, "
                               f"floor={settings.MIN_SCORE_FLOOR})")
            self._update(job_id, current_stage="select_segments")
            stage6_select.select_segments(video_path.stem)
            self._mark_done(job_id, "select_segments")

            self._log(job_id, "Stage 8: cutting + assembling rough-cut (free, CPU)")
            self._update(job_id, current_stage="assemble")
            output_path = stage7_assemble.assemble(video_path.stem, video_path)
            self._mark_done(job_id, "assemble")

            self._log(job_id, f"Done -> {output_path}")
            self._update(job_id, status="completed", current_stage=None, output_path=str(output_path))
        except Exception as e:
            self._log(job_id, f"FAILED: {e}")
            self._update(job_id, status="failed", error=str(e))
            traceback.print_exc()

    def _run_explainer_job(self, job_id: str, video_path: Path, language: str = "en"):
        """Explainer (narrated recap) pipeline — Stages 1-7 (2026-08-26 redesign,
        +intro feature 2026-08-27). Produces per-beat Clip-N.mp4/voice-N.mp3 pairs plus a
        separate Intro.mp4/intro-voice.mp3 pair — NO merged final video, by explicit design;
        you assemble the final cut yourself.

        CHANGED 2026-08-27 (multi-language redesign): `language` is threaded through
        every stage from here on — subtitle lookup, summary, matching, TTS, and
        retiming all run natively in this one language, reading/writing that
        language's own subfolder (data/explainer/{movie}/{language}/).

        CHANGED 2026-08-27 (intro feature, by explicit user request): a new
        "generate_intro" stage runs right after generate_summary (it only needs
        summary.json + the transcript, same as generate_summary itself does — no
        dependency on match_clips). Its TTS and retiming are folded into the existing
        synthesize_narration/retime_clips stages rather than getting stage entries of
        their own, since from the user's/dashboard's point of view they're the same
        conceptual step ("make the audio", "make the clips") just with one extra
        un-numbered item. See pipeline/explainer_intro.py for the full design."""
        self._update(job_id, status="running")
        try:
            self._log(job_id, "Stage 1/7: extracting audio (free, CPU)")
            self._update(job_id, current_stage="extract_audio")
            audio_path = stage1_ingest.extract_audio(video_path)
            self._mark_done(job_id, "extract_audio")

            srt_path = subtitles.find_subtitle_file(video_path, language)
            if srt_path:
                self._log(job_id, f"Stage 2/7: using {srt_path.name} (Whisper transcription skipped)")
            else:
                self._log(job_id, f"Stage 2/7: transcribing (free, CPU — can take a while; "
                                   f"language={language!r})")
            self._update(job_id, current_stage="transcribe")
            subtitles.load_or_transcribe(video_path, audio_path, language)
            self._mark_done(job_id, "transcribe")

            self._log(job_id, f"Stage 3/7: generating summary in "
                               f"{settings.EXPLAINER_LANGUAGE_NAMES[language]} (PAID — Gemini "
                               f"via Vertex ADC, schema-constrained)")
            self._update(job_id, current_stage="generate_summary")
            explainer_stage3_summary.generate_summary(video_path.stem, language)
            self._mark_done(job_id, "generate_summary")

            self._log(job_id, f"Stage 4/7: generating ~{settings.INTRO_TARGET_SECONDS}s channel "
                               f"intro (PAID — Gemini via Vertex ADC, schema-constrained)")
            self._update(job_id, current_stage="generate_intro")
            explainer_intro.generate_intro(video_path.stem, language)
            self._mark_done(job_id, "generate_intro")

            self._log(job_id, "Stage 5/7: matching beats to transcript (PAID — Gemini via "
                               "Vertex ADC, schema-constrained)")
            self._update(job_id, current_stage="match_clips")
            explainer_stage4_match.match_clips(video_path.stem, language)
            self._mark_done(job_id, "match_clips")

            self._log(job_id, f"Stage 6/7: synthesizing narration + intro voice (PAID — "
                               f"Chirp3-HD via ADC; voice={settings.NARRATOR_VOICES.get(language)}; "
                               f"language={language!r})")
            self._update(job_id, current_stage="synthesize_narration")
            explainer_stage5_tts.synthesize_narration(video_path.stem, video_path, audio_path, language)
            explainer_stage5_tts.synthesize_intro_narration(video_path.stem, language)
            self._mark_done(job_id, "synthesize_narration")

            self._log(job_id, "Stage 7/7: retiming per-beat clips + intro clip (free, CPU) — "
                               "Clip-N.mp4/voice-N.mp3 pairs plus Intro.mp4/intro-voice.mp3, no merge")
            self._update(job_id, current_stage="retime_clips")
            beats = explainer_stage6_assemble.retime_clips(video_path.stem, video_path, language)
            intro = explainer_stage6_assemble.retime_intro_clip(video_path.stem, video_path, language)
            self._mark_done(job_id, "retime_clips")

            outputs = {language: beats}
            intro_by_language = {language: intro}
            self._log(job_id, f"Done -> {len(beats)} beat clip/voice pairs + 1 intro clip/voice pair")
            self._update(job_id, status="completed", current_stage=None,
                         outputs=outputs, intro=intro_by_language)
        except Exception as e:
            self._log(job_id, f"FAILED: {e}")
            self._update(job_id, status="failed", error=str(e))
            traceback.print_exc()

    def _mark_done(self, job_id, stage_name):
        with self._lock:
            self._jobs[job_id]["stages_completed"].append(stage_name)

    def reselect(self, job_id: str, target_duration_sec: float = None, min_score_floor: float = None):
        """Re-run Stage 7 (select) + Stage 8 (assemble) with a new target duration
        and/or quality floor, reusing the already-computed transcript/signals/scores
        — no re-transcription, no new Vertex API calls. Runs synchronously since
        it's fast (concat assembly, not crossfade re-encode). Highlight-reel mode
        only — explainer mode has no equivalent cheap re-cut."""
        with self._lock:
            job = self._jobs.get(job_id)
        if not job:
            raise KeyError(f"No such job: {job_id}")
        if job.get("mode", "highlight") != "highlight":
            raise ValueError("Re-select/re-cut is only available for highlight-reel jobs")
        if job["status"] != "completed":
            raise ValueError("Can only re-select on a completed job")

        original_target = settings.TARGET_OUTPUT_DURATION
        original_floor = settings.MIN_SCORE_FLOOR
        if target_duration_sec is not None:
            settings.TARGET_OUTPUT_DURATION = target_duration_sec
        if min_score_floor is not None:
            settings.MIN_SCORE_FLOOR = min_score_floor
        try:
            self._log(job_id, f"Re-selecting (target={settings.TARGET_OUTPUT_DURATION}s, "
                               f"floor={settings.MIN_SCORE_FLOOR})")
            stage6_select.select_segments(job["video_stem"])
            output_path = stage7_assemble.assemble(job["video_stem"], Path(job["video_path"]))
            self._update(job_id, output_path=str(output_path),
                         target_duration_used=settings.TARGET_OUTPUT_DURATION,
                         min_score_floor_used=settings.MIN_SCORE_FLOOR)
            self._log(job_id, f"Re-assembled -> {output_path}")
        finally:
            settings.TARGET_OUTPUT_DURATION = original_target
            settings.MIN_SCORE_FLOOR = original_floor
        return self.get_job(job_id)

    def get_job(self, job_id: str):
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def list_jobs(self):
        with self._lock:
            return [dict(j) for j in sorted(self._jobs.values(), key=lambda j: j["created_at"], reverse=True)]


job_manager = JobManager()
