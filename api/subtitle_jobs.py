"""
Background job tracking for the two new subtitle utilities (2026-09-05):
generating a subtitle via Whisper when none exists, and translating an
existing subtitle into a different language.

Kept as its OWN small job store, separate from api/jobs.py's JobManager, on
purpose — these are quick, single-shot utility operations with a different
shape from the multi-stage explainer/highlight pipelines (no "outputs"/
"intro"/"stages_completed" fields, no mode branching). Reusing JobManager's
machinery for something this different would mean threading a lot of
not-applicable fields through it for no real benefit; this mirrors its
lock+thread pattern at a much smaller scale instead.

Same "single-user, local" scope note as api/jobs.py: an in-memory dict
guarded by a lock, one daemon thread per job. Fine for one person watching
a dashboard on their own machine.
"""
import threading
import time
import traceback
import uuid
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from pipeline import subtitles


class SubtitleJobManager:
    def __init__(self):
        self._jobs = {}
        self._lock = threading.Lock()
        # New (2026-09-05, real-machine crash fix): every job kind here starts its own daemon
        # thread with NO concurrency limit, so picking a second movie on the Subtitle Library
        # page while a Generate job is still running for a first one loads a SECOND Whisper
        # model and runs a SECOND transcription/detection pass at the same time — exactly the
        # kind of double memory pressure that crashed with an ArrayMemoryError on the user's
        # real machine (modest RAM). translate_subtitle() only calls Gemini (no local model,
        # no meaningful memory cost) so it's left OUT of this lock and can still run alongside
        # anything else; detect_language()/generate_subtitle() (both load a WhisperModel) share
        # this one lock so only one of them ever runs at a time, regardless of how many jobs
        # get queued from the UI. Doesn't cover a Library job racing a live pipeline job's own
        # Whisper transcription (api/jobs.py) — those are separate systems; if that combination
        # ever crashes the same way, the fix is the same idea at a higher level.
        self._whisper_lock = threading.Lock()

    def _update(self, job_id, **fields):
        with self._lock:
            self._jobs[job_id].update(fields)
            self._jobs[job_id]["updated_at"] = time.time()

    def start_generate(self, video_path: Path, language: str = None) -> str:
        """language=None lets generate_subtitle() auto-detect."""
        job_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id, "kind": "generate", "filename": video_path.name,
                "requested_language": language, "status": "running", "error": None,
                "result": None, "created_at": time.time(), "updated_at": time.time(),
            }
        thread = threading.Thread(target=self._run_generate, args=(job_id, video_path, language), daemon=True)
        thread.start()
        return job_id

    def _run_generate(self, job_id, video_path, language):
        try:
            with self._whisper_lock:
                srt_path, used_language, probability = subtitles.generate_subtitle(video_path, language)
            self._update(job_id, status="completed", result={
                "srt_filename": srt_path.name, "language": used_language, "probability": probability,
            })
        except Exception as e:
            self._update(job_id, status="failed", error=str(e))
            traceback.print_exc()

    def start_translate(self, video_path: Path, source_language: str, target_language: str) -> str:
        job_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id, "kind": "translate", "filename": video_path.name,
                "source_language": source_language, "target_language": target_language,
                "status": "running", "error": None, "result": None,
                "created_at": time.time(), "updated_at": time.time(),
            }
        thread = threading.Thread(
            target=self._run_translate, args=(job_id, video_path, source_language, target_language), daemon=True,
        )
        thread.start()
        return job_id

    def _run_translate(self, job_id, video_path, source_language, target_language):
        try:
            srt_path = subtitles.translate_subtitle(video_path, source_language, target_language)
            self._update(job_id, status="completed", result={"srt_filename": srt_path.name})
        except Exception as e:
            self._update(job_id, status="failed", error=str(e))
            traceback.print_exc()

    def get_job(self, job_id: str):
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    # --- Standalone "Subtitle Library" page support (2026-09-05) -------------------------
    # Two more job kinds, added when the user proposed a dedicated /dashboard/subtitle page
    # instead of the in-line per-job detect/generate/translate flow above (which turned out
    # to hang indefinitely in practice — see detect_language()'s call site in server.py for
    # the actual fix). Both reuse the exact same subtitles.py functions; nothing about the
    # underlying generate/translate logic changes, only how/when they're invoked.

    def start_detect(self, video_path: Path) -> str:
        """Language-ID only (no generation) — lets the Subtitle Library page pre-fill a
        guessed movie language for the user to confirm/override, without blocking on it."""
        job_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id, "kind": "detect", "filename": video_path.name,
                "status": "running", "error": None, "result": None,
                "created_at": time.time(), "updated_at": time.time(),
            }
        thread = threading.Thread(target=self._run_detect, args=(job_id, video_path), daemon=True)
        thread.start()
        return job_id

    def _run_detect(self, job_id, video_path):
        try:
            with self._whisper_lock:
                language, probability = subtitles.detect_language(video_path)
            self._update(job_id, status="completed", result={"language": language, "probability": probability})
        except Exception as e:
            self._update(job_id, status="failed", error=str(e))
            traceback.print_exc()

    def start_generate_all(self, video_path: Path, language: str = None) -> str:
        """The "one Generate Subtitle button" the user asked for: produce a REAL, reusable
        .srt for ALL THREE supported languages (settings.EXPLAINER_LANGUAGES) in one job —
        generate_subtitle() in the movie's own (confirmed or auto-detected) language, then
        translate_subtitle() into each of the other two. `language=None` auto-detects first;
        a confirmed code skips detection and pins Whisper to it directly (more accurate, per
        generate_subtitle()'s own docstring). Exposes a `steps` list on the job (appended to
        live, not just a single message) so the dashboard can show real per-stage progress
        instead of one opaque spinner."""
        job_id = str(uuid.uuid4())[:8]
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id, "kind": "generate_all", "filename": video_path.name,
                "requested_language": language, "status": "running", "error": None,
                "result": None, "steps": [], "created_at": time.time(), "updated_at": time.time(),
            }
        thread = threading.Thread(target=self._run_generate_all, args=(job_id, video_path, language), daemon=True)
        thread.start()
        return job_id

    def _add_step(self, job_id, label, status, detail=None):
        """Upserts by label — a step goes "running" then later "done"/"failed" IN PLACE, one
        row per step, rather than appending a second entry for the same step's later status.
        (Get this wrong and the dashboard's step list would show every step twice: once stuck
        forever on "running", once "done" right under it.)"""
        with self._lock:
            steps = self._jobs[job_id]["steps"]
            existing = next((s for s in steps if s["label"] == label), None)
            if existing is not None:
                existing["status"] = status
                existing["detail"] = detail
            else:
                steps.append({"label": label, "status": status, "detail": detail})
            self._jobs[job_id]["updated_at"] = time.time()

    def _run_generate_all(self, job_id, video_path, language):
        try:
            source_language = language
            if not source_language:
                self._add_step(job_id, "Detecting movie language", "running")
                with self._whisper_lock:
                    source_language, probability = subtitles.detect_language(video_path)
                self._add_step(job_id, "Detecting movie language", "done",
                                f"detected {source_language!r} ({probability:.0%} confidence)")
            else:
                self._add_step(job_id, "Movie language confirmed", "done", source_language)

            by_language = {}
            gen_name = settings.EXPLAINER_LANGUAGE_NAMES.get(source_language, source_language)
            gen_label = f"Generating {gen_name} subtitle from the movie's own audio"
            self._add_step(job_id, gen_label, "running")
            with self._whisper_lock:
                srt_path, used_language, _probability = subtitles.generate_subtitle(video_path, source_language)
            source_language = used_language  # only differs if source_language was None (auto-detect)
            by_language[source_language] = srt_path.name
            self._add_step(job_id, gen_label, "done", srt_path.name)

            for target in settings.EXPLAINER_LANGUAGES:
                if target == source_language:
                    continue
                target_name = settings.EXPLAINER_LANGUAGE_NAMES.get(target, target)
                label = f"Translating to {target_name}"
                self._add_step(job_id, label, "running")
                try:
                    out_path = subtitles.translate_subtitle(video_path, source_language, target)
                    by_language[target] = out_path.name
                    self._add_step(job_id, label, "done", out_path.name)
                except Exception as e:
                    # One language failing (e.g. a transient Gemini error) shouldn't lose the
                    # others, which are already safely written to disk by this point — report
                    # it as a failed step and keep going, rather than aborting the whole job.
                    self._add_step(job_id, label, "failed", str(e))
                    traceback.print_exc()

            self._update(job_id, status="completed", result={
                "source_language": source_language, "subtitles": by_language,
            })
        except Exception as e:
            self._update(job_id, status="failed", error=str(e))
            traceback.print_exc()


subtitle_job_manager = SubtitleJobManager()
