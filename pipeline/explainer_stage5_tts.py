"""
Explainer Stage 5 — Narration voice synthesis (per beat)
PAID — Cloud Text-to-Speech API (ADC). CHANGED 2026-08-26: switched from
Gemini TTS ("Charon") to Chirp 3: HD ("Fenrir") after the user A/B-tested
voices side by side — see config/settings.py's Stage 5 comment for the
full rationale. Chirp3-HD has no style-prompt field, so every beat gets one
consistent plain read; there is no more per-beat intensity computation.

CHANGED 2026-08-27 (multi-language redesign): now takes a `language` param
and looks up that language's voice from settings.NARRATOR_VOICES (a dict,
so adding/changing a voice later is a one-line settings.py edit, not a code
change) and locale from settings.LOCALE_BY_LANGUAGE. Raises a clear error
if that language has no voice configured yet (e.g. Bengali, deferred by
explicit user choice) rather than silently falling back to some other
language's voice/locale.

Output: voice-{n}.mp3 per beat (1-indexed, matching Clip-{n}.mp4 from
Stage 6), plus its real spoken duration — Stage 6 uses that duration to
decide how to retime the visual clip. Both now live under this language's
own subfolder — data/explainer/{movie}/{language}/voices/.

CHANGED 2026-08-27 (intro feature): also exposes synthesize_intro_narration(),
the Stage-5 half of the new 10s-intro feature (see pipeline/explainer_intro.py
for the full design rationale). It reads/rewrites the SAME progressively-
enriched data/explainer/{movie}/{language}/intro.json file explainer_intro.py
wrote, reusing this file's existing _synthesize_with_retry()/_probe_duration()
helpers unchanged. Output is intro-voice.mp3, written into the SAME voices/
folder synthesize_narration() already uses (per the user's explicit choice —
no separate intro/ subfolder), not part of the numbered voice-{n}.mp3 sequence.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings


def _synthesize_with_retry(client, texttospeech, text: str, locale: str, voice_name: str):
    last_error = None
    for attempt in range(settings.TTS_MAX_RETRIES + 1):
        try:
            response = client.synthesize_speech(
                input=texttospeech.SynthesisInput(text=text),  # no `prompt` — Chirp3-HD doesn't support one
                voice=texttospeech.VoiceSelectionParams(
                    language_code=locale,
                    name=voice_name,
                ),
                audio_config=texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3),
            )
            return response.audio_content
        except Exception as e:
            last_error = e
            is_rate_limit = "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e) or "quota" in str(e).lower()
            if attempt < settings.TTS_MAX_RETRIES:
                backoff = settings.TTS_RETRY_BACKOFF_BASE * (2 ** attempt)
                reason = "rate limited" if is_rate_limit else "error"
                print(f"    [retry] TTS {reason}, attempt {attempt + 1}/{settings.TTS_MAX_RETRIES}, "
                      f"waiting {backoff}s: {e}")
                time.sleep(backoff)
    raise last_error


def _probe_duration(path: Path) -> float:
    import subprocess
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    return float(probe.stdout.strip())


def synthesize_narration(video_stem: str, source_video: Path, audio_path: Path, language: str = "en") -> Path:
    from google.cloud import texttospeech

    if language not in settings.EXPLAINER_LANGUAGE_NAMES:
        raise RuntimeError(f"Unsupported language {language!r} — must be one of "
                            f"{list(settings.EXPLAINER_LANGUAGE_NAMES)}.")

    voice_name = settings.NARRATOR_VOICES.get(language)
    if not voice_name:
        raise RuntimeError(
            f"No narrator voice configured for language={language!r} in "
            f"settings.NARRATOR_VOICES — pick one (see test_voice_hindi_chirp3hd.py for the "
            f"kind of A/B sampling script used to choose the others) and set it there before "
            f"running TTS for this language."
        )
    locale = settings.LOCALE_BY_LANGUAGE[language]

    matched_path = settings.EXPLAINER_DIR / video_stem / language / "matched.json"
    beats = json.loads(matched_path.read_text(encoding="utf-8"))["beats"]
    if not beats:
        raise RuntimeError("No beats to synthesize narration for.")

    if not settings.GCP_PROJECT_ID:
        raise RuntimeError(
            "GCP_PROJECT_ID is not set. Set it in config/settings.py before running TTS "
            "(this stage bills your GCP credit)."
        )

    client = texttospeech.TextToSpeechClient()  # ADC, same as gcloud auth application-default login

    voice_dir = settings.EXPLAINER_DIR / video_stem / language / "voices"
    voice_dir.mkdir(parents=True, exist_ok=True)

    print(f"[explainer-tts] language={language!r}, synthesizing {len(beats)} beats "
          f"(voice={voice_name}, locale={locale}, no per-beat style variation)...")
    for beat in beats:
        text = beat.get("narration", "").strip()
        n = beat["index"] + 1  # 1-indexed, matches Clip-N.mp4 from Stage 6
        out_path = voice_dir / f"voice-{n}.mp3"
        if not text:
            beat["voice_duration"] = 0.0
            beat["voice_path"] = None
            continue
        audio_bytes = _synthesize_with_retry(client, texttospeech, text, locale, voice_name)
        out_path.write_bytes(audio_bytes)
        duration = _probe_duration(out_path)
        beat["voice_duration"] = round(duration, 2)
        beat["voice_path"] = str(out_path)
    print("[explainer-tts] done")

    out_dir = settings.EXPLAINER_DIR / video_stem / language
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "voiced.json"
    out_path.write_text(json.dumps({"beats": beats}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[explainer-tts] narration synthesis complete -> {out_path}")
    return out_path


def synthesize_intro_narration(video_stem: str, language: str = "en") -> Path:
    from google.cloud import texttospeech

    if language not in settings.EXPLAINER_LANGUAGE_NAMES:
        raise RuntimeError(f"Unsupported language {language!r} — must be one of "
                            f"{list(settings.EXPLAINER_LANGUAGE_NAMES)}.")

    voice_name = settings.NARRATOR_VOICES.get(language)
    if not voice_name:
        raise RuntimeError(
            f"No narrator voice configured for language={language!r} in "
            f"settings.NARRATOR_VOICES — pick one and set it there before running TTS "
            f"for this language."
        )
    locale = settings.LOCALE_BY_LANGUAGE[language]

    intro_path = settings.EXPLAINER_DIR / video_stem / language / "intro.json"
    intro = json.loads(intro_path.read_text(encoding="utf-8"))

    if not settings.GCP_PROJECT_ID:
        raise RuntimeError(
            "GCP_PROJECT_ID is not set. Set it in config/settings.py before running TTS "
            "(this stage bills your GCP credit)."
        )

    client = texttospeech.TextToSpeechClient()

    voice_dir = settings.EXPLAINER_DIR / video_stem / language / "voices"
    voice_dir.mkdir(parents=True, exist_ok=True)

    text = intro.get("narration", "").strip()
    out_path = voice_dir / "intro-voice.mp3"
    if not text:
        intro["voice_duration"] = 0.0
        intro["voice_path"] = None
    else:
        print(f"[explainer-tts] language={language!r}, synthesizing intro narration "
              f"(voice={voice_name}, locale={locale})...")
        audio_bytes = _synthesize_with_retry(client, texttospeech, text, locale, voice_name)
        out_path.write_bytes(audio_bytes)
        duration = _probe_duration(out_path)
        intro["voice_duration"] = round(duration, 2)
        intro["voice_path"] = str(out_path)

    intro_path.write_text(json.dumps(intro, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[explainer-tts] intro narration synthesis complete -> {intro_path}")
    return intro_path


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        print("Usage: python explainer_stage5_tts.py <video_stem> <source_video.mp4> <audio.wav> [language]")
        sys.exit(1)
    synthesize_narration(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]),
                          sys.argv[4] if len(sys.argv) == 5 else "en")
