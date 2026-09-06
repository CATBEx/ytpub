"""
One-off voice regeneration — NOT part of the pipeline.

Re-synthesizes ALL beats' narration for an already-completed Explainer job,
using Chirp3-HD's "Fenrir" voice instead of the pipeline's current Gemini
TTS "Charon". Reads the narration text straight from the existing
{stem}_voiced.json (produced by the real Stage 5 run) — no new Gemini
summary/match calls, just re-voicing text that's already final.

By explicit request: NO per-beat intensity/style variation — Chirp3-HD
doesn't support a style prompt anyway, so every beat gets the same plain,
consistent read. This deliberately ignores the "intensity" field already
present in {stem}_voiced.json from the real pipeline run.

Output goes to a SEPARATE folder from the real pipeline's voices/ output,
so nothing from the actual completed job gets overwritten:
    data/explainer/{stem}/voices_fenrir/voice-{n}.mp3

Run (needs your GCP ADC creds — same auth as the real pipeline):
    python test_voice_fenrir_full_episode.py
Optionally pass a different video stem (defaults to the Boba Fett episode):
    python test_voice_fenrir_full_episode.py "Some Other Movie"
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings

DEFAULT_STEM = "The Book of Boba Fett_S1E1"
VOICE_NAME = "en-US-Chirp3-HD-Fenrir"


def _synthesize_with_retry(client, texttospeech, text: str, locale: str):
    last_error = None
    for attempt in range(settings.TTS_MAX_RETRIES + 1):
        try:
            response = client.synthesize_speech(
                input=texttospeech.SynthesisInput(text=text),  # no `prompt` — Chirp3-HD doesn't support one
                voice=texttospeech.VoiceSelectionParams(
                    language_code=locale,
                    name=VOICE_NAME,
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


def main():
    from google.cloud import texttospeech

    video_stem = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_STEM
    locale = settings.LOCALE_BY_LANGUAGE["en"]

    voiced_path = settings.EXPLAINER_DIR / f"{video_stem}_voiced.json"
    if not voiced_path.exists():
        print(f"[voice-fenrir] ERROR: {voiced_path} not found. Pass the right video stem, "
              f"or run the real pipeline for this movie first.")
        sys.exit(1)

    beats = json.loads(voiced_path.read_text(encoding="utf-8"))["beats"]
    if not beats:
        print("[voice-fenrir] ERROR: no beats in that file.")
        sys.exit(1)

    out_dir = settings.EXPLAINER_DIR / video_stem / "voices_fenrir"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[voice-fenrir] {len(beats)} beats from {voiced_path.name}")
    print(f"[voice-fenrir] voice={VOICE_NAME}, locale={locale}, no per-beat style variation")
    print(f"[voice-fenrir] output -> {out_dir}")

    client = texttospeech.TextToSpeechClient()  # ADC, same as the real pipeline

    for beat in beats:
        n = beat["index"] + 1
        text = beat.get("narration", "").strip()
        if not text:
            print(f"[voice-fenrir] beat {n}: no narration text, skipping")
            continue
        print(f"\n[voice-fenrir] beat {n}/{len(beats)}: {text[:70]}{'...' if len(text) > 70 else ''}")
        audio_bytes = _synthesize_with_retry(client, texttospeech, text, locale)
        out_path = out_dir / f"voice-{n}.mp3"
        out_path.write_bytes(audio_bytes)
        print(f"[voice-fenrir]   -> {out_path}")

    print(f"\n[voice-fenrir] done — {len(beats)} beats voiced with Fenrir -> {out_dir}")


if __name__ == "__main__":
    main()
