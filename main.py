"""
Movie Review Video Pipeline — entrypoint.

Two content modes:

  HIGHLIGHT-REEL mode ("Director" cut) — fast-paced hype trailer, best
  moments out of order, 5-6 min target.
    stage1_ingest.py    -> Stage 1: extract audio
    stage2_transcribe.py-> Stage 2: transcribe (faster-whisper, free)
    stage3_shots.py     -> Stage 3: shot detection (PySceneDetect, free)
    stage4_signals.py   -> Stage 4+5: audio/visual signal extraction (free)
    stage5_score.py     -> Stage 6: importance scoring (Gemini via Vertex ADC, PAID)
    stage6_select.py    -> Stage 7: curated duration-target selection (free)
    stage7_assemble.py  -> Stage 8: cut + assemble rough-cut (ffmpeg, free)

  EXPLAINER mode — narrated movie/series recap. Produces per-beat
  Clip-N.mp4 (video-only) / voice-N.mp3 pairs, in story order — NOT a
  merged final video; you assemble the final cut yourself (2026-08-26
  redesign). CHANGED 2026-08-27 (multi-language redesign): each run now
  picks ONE language (English/Hindi/Bengali) and runs every stage natively
  in it, reading that language's own subtitle where required — see
  config/settings.py's EXPLAINER_LANGUAGES/EXPLAINER_SUBTITLE_REQUIRED/
  NARRATOR_VOICES for what's supported and pipeline/subtitles.py for the
  per-language subtitle-file convention.
    stage1_ingest.py, subtitles.py -> reused as-is / language-aware transcript
    explainer_stage3_summary.py -> Stage 3: narration summary, schema-constrained (Gemini, PAID)
    explainer_stage4_match.py   -> Stage 4: match beats to transcript, schema-constrained (Gemini, PAID)
    explainer_stage5_tts.py     -> Stage 5: per-beat narration voice synthesis (Chirp3-HD, PAID)
    explainer_stage6_assemble.py -> Stage 6: per-beat clip retiming, video-only, no merge (free)

Run:

    python main.py
        Starts the FastAPI server + dashboard, fixed on port 9999, with
        auto-reload on — edit any pipeline/api file and the server picks it
        up on the next request, no manual restart needed.
        Dashboard: http://localhost:9999/dashboard

    python main.py "data/input/my_movie.mp4"
        Runs the highlight-reel pipeline once via CLI, no server involved.

    python main.py explain "data/input/my_movie.mp4" [language]
        Runs the explainer pipeline once via CLI in the given language
        (en/hi/bn, default en — see config/settings.py), no server. Hindi/
        Bengali require a matching subtitle file (Movie.hi.srt / Movie.bn.srt)
        next to the video.
"""
import sys
import multiprocessing
from pathlib import Path

# Windows' console defaults to a legacy ANSI codepage (the "charmap" codec),
# not UTF-8 — any print() of generated non-English text (Hindi/Bengali
# narration, transcript segments, etc.) crashes with UnicodeEncodeError the
# moment it hits a non-ASCII character. Reconfigure stdout/stderr to UTF-8
# up front so no print() anywhere in the pipeline can ever crash a job over
# this again, regardless of language or which stage does the printing.
# (uvicorn's reload=True re-executes this module fresh in each worker
# subprocess, so this applies there too, not just in the parent process.)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# On some Windows/venv combinations, multiprocessing's "spawn" start method
# (which uvicorn's reload=True relies on to launch its real worker process)
# silently resolves to a DIFFERENT Python interpreter than the one actually
# running this process — e.g. the machine's global Python install instead of
# this project's own venv. When that global install doesn't have every
# package this project needs, a job can crash deep into a run (e.g.
# `ImportError: cannot import name 'texttospeech' from 'google.cloud'` at the
# TTS stage) even though earlier stages using packages the global install
# happens to also have (Gemini, faster-whisper) worked fine — a confusing,
# stage-dependent failure that looks unrelated to environment setup. Pin the
# executable explicitly so every spawned worker always uses the SAME
# interpreter as whatever started this process (the venv's, if that's how
# it was launched) — no divergence possible regardless of the underlying
# cause. Must be set before uvicorn (or anything else) spawns a subprocess.
multiprocessing.set_executable(sys.executable)

sys.path.insert(0, str(Path(__file__).resolve().parent))

SERVER_PORT = 9999


def run_server():
    """Start the FastAPI server + dashboard with auto-reload, fixed on port 9999.
    reload=True is set here in code (not as a CLI flag) specifically so a pasted/
    mistyped --reload flag can't break the command the way it did before."""
    import uvicorn
    print(f"Starting Movie Review Pipeline server on http://0.0.0.0:{SERVER_PORT} (auto-reload on)")
    print(f"Dashboard: http://localhost:{SERVER_PORT}/dashboard")
    uvicorn.run("api.server:app", host="0.0.0.0", port=SERVER_PORT, reload=True)


def run_pipeline(video_path: Path):
    """Run the full pipeline once via CLI. Imports are local to this function
    so `python main.py` (server mode) starts fast without loading whisper/
    scenedetect/etc. into the server's parent process for no reason."""
    from pipeline import stage1_ingest, stage2_transcribe, stage3_shots
    from pipeline import stage4_signals, stage5_score, stage6_select, stage7_assemble

    stem = video_path.stem

    print("=" * 70)
    print(f"Movie Review Pipeline — {video_path.name}")
    print("=" * 70)

    print("\n--- Stage 1: extract audio ---")
    audio_path = stage1_ingest.extract_audio(video_path)

    print("\n--- Stage 2: transcribe (free, CPU) ---")
    stage2_transcribe.transcribe(audio_path)

    print("\n--- Stage 3: shot detection (free, CPU) ---")
    shots_path = stage3_shots.detect_shots(video_path)

    print("\n--- Stage 4/5: signal extraction (free, CPU) ---")
    stage4_signals.extract_signals(video_path, audio_path, shots_path)

    print("\n--- Stage 6: importance scoring (PAID — Gemini via Vertex ADC) ---")
    stage5_score.score_segments(stem)

    print("\n--- Stage 7: curated selection to target duration (free) ---")
    stage6_select.select_segments(stem)

    print("\n--- Stage 8: cut + assemble rough-cut (free, CPU) ---")
    output_path = stage7_assemble.assemble(stem, video_path)

    print("\n" + "=" * 70)
    print(f"Done. Rough-cut saved to: {output_path}")
    print("=" * 70)
    return output_path


def run_explainer_pipeline(video_path: Path, language: str = "en"):
    """Run the explainer pipeline once via CLI, natively in `language`.
    Produces per-beat Clip-N.mp4/voice-N.mp3 pairs — no merged final video,
    by design."""
    from pipeline import stage1_ingest, subtitles
    from pipeline import explainer_stage3_summary, explainer_stage4_match, explainer_intro
    from pipeline import explainer_stage5_tts, explainer_stage6_assemble
    from config import settings

    if language not in settings.EXPLAINER_LANGUAGE_NAMES:
        raise SystemExit(f"Unsupported language: {language!r} (expected one of "
                          f"{list(settings.EXPLAINER_LANGUAGE_NAMES)})")

    stem = video_path.stem

    print("=" * 70)
    print(f"Movie Explainer Pipeline — {video_path.name}")
    print(f"Language: {settings.EXPLAINER_LANGUAGE_NAMES[language]} ({language})")
    print("=" * 70)

    print("\n--- Stage 1: extract audio ---")
    audio_path = stage1_ingest.extract_audio(video_path)

    srt_path = subtitles.find_subtitle_file(video_path, language)
    if srt_path:
        print(f"\n--- Stage 2: using {srt_path.name} (Whisper transcription skipped) ---")
    else:
        print("\n--- Stage 2: transcribe (free, CPU) ---")
    subtitles.load_or_transcribe(video_path, audio_path, language)

    print("\n--- Stage 3: summary generation (PAID — Gemini via Vertex ADC, schema-constrained) ---")
    explainer_stage3_summary.generate_summary(stem, language)

    print(f"\n--- Stage 3.5: ~{settings.INTRO_TARGET_SECONDS}s intro generation "
          f"(PAID — Gemini via Vertex ADC, schema-constrained) ---")
    explainer_intro.generate_intro(stem, language)

    print("\n--- Stage 4: clip matching (PAID — Gemini via Vertex ADC, schema-constrained) ---")
    explainer_stage4_match.match_clips(stem, language)

    print("\n--- Stage 5: narration synthesis (PAID — Chirp3-HD via ADC) ---")
    explainer_stage5_tts.synthesize_narration(stem, video_path, audio_path, language)
    explainer_stage5_tts.synthesize_intro_narration(stem, language)

    print("\n--- Stage 6: per-beat + intro clip retiming (free, CPU) — no merge ---")
    beats = explainer_stage6_assemble.retime_clips(stem, video_path, language)
    intro = explainer_stage6_assemble.retime_intro_clip(stem, video_path, language)

    print("\n" + "=" * 70)
    print(f"Done. Intro.mp4 / intro-voice.mp3: {intro['clip_path']}")
    print(f"Plus {len(beats)} beat clip/voice pairs saved:")
    for b in beats:
        print(f"  Clip-{b['index'] + 1}.mp4 / voice-{b['index'] + 1}.mp3: {b['clip_path']}")
    print("=" * 70)
    return {"intro": intro, "beats": beats}


if __name__ == "__main__":
    if len(sys.argv) == 1:
        run_server()
    elif len(sys.argv) == 2:
        run_pipeline(Path(sys.argv[1]))
    elif len(sys.argv) in (3, 4) and sys.argv[1] == "explain":
        run_explainer_pipeline(Path(sys.argv[2]), sys.argv[3] if len(sys.argv) == 4 else "en")
    else:
        print("Usage:")
        print("  python main.py                                 # start server + dashboard on :9999 (auto-reload)")
        print("  python main.py <movie.mp4>                      # run the highlight-reel pipeline once via CLI")
        print("  python main.py explain <movie.mp4> [language]   # run the explainer pipeline once via CLI")
        print("                                                   #   language: en (default) / hi / bn")
        sys.exit(1)
