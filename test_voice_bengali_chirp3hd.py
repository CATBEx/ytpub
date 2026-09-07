"""
One-off diagnostic — NOT part of the pipeline (same category as
list_available_voices.py / test_voice_hindi_chirp3hd.py already in this
folder). Queries Google's real ListVoices API for every bn-IN (Bengali,
India) Chirp3-HD voice on your project, then synthesizes the SAME short
Bengali sample line with each one so you can A/B them side by side — same
approach used to pick Fenrir (English) and Achird (Hindi).

Auth: Application Default Credentials only (same as the rest of this
project) — do NOT set GEMINI_API_KEY. Run `gcloud auth application-default
login` first if you haven't already on this machine.

Output: voice_samples_bengali/<voice-name>_<gender>.mp3 — one file per real
bn-IN Chirp3-HD voice found on your project. Also prints a plain-text
summary at the end (name/gender/status) so you can see what succeeded
without opening every file.

Run:
    venv\\Scripts\\activate
    python test_voice_bengali_chirp3hd.py

Cost: Chirp 3: HD is $30 per million characters (see config/settings.py).
The sample line below is ~90 characters — roughly 30 voices x 90 chars is
under 3,000 characters total, a small fraction of a cent.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings

# A short, recap-narrator-style Bengali line — the same sentence used for the
# Hindi A/B test, translated — representative of what Stage 3 would actually
# write, not just a generic "hello world" test phrase.
SAMPLE_TEXT_BN = (
    "একটি রহস্যময় দ্বীপে, আমাদের নায়ক নিজেকে সম্পূর্ণ একা আবিষ্কার করে। "
    "কিন্তু শীঘ্রই, সে বুঝতে পারে যে সে একা নয়।"
)

OUT_DIR = Path(__file__).resolve().parent / "voice_samples_bengali"
RETRY_BACKOFF = 5  # seconds, doubled per retry — same pattern as the pipeline's own TTS stage
MAX_RETRIES = 3


def _synthesize_with_retry(client, texttospeech, text, locale, voice_name):
    last_error = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.synthesize_speech(
                input=texttospeech.SynthesisInput(text=text),
                voice=texttospeech.VoiceSelectionParams(language_code=locale, name=voice_name),
                audio_config=texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3),
            )
            return response.audio_content
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                backoff = RETRY_BACKOFF * (2 ** attempt)
                print(f"    [retry] {voice_name} failed, attempt {attempt + 1}/{MAX_RETRIES}, "
                      f"waiting {backoff}s: {e}")
                time.sleep(backoff)
    raise last_error


def main():
    from google.cloud import texttospeech

    if not settings.GCP_PROJECT_ID:
        raise RuntimeError("GCP_PROJECT_ID is not set in config/settings.py — needed before this can run.")

    locale = settings.LOCALE_BY_LANGUAGE["bn"]  # "bn-IN"
    client = texttospeech.TextToSpeechClient()  # ADC — same auth as the rest of this project

    print(f"[voices] querying real Chirp3-HD voices for locale={locale} on project={settings.GCP_PROJECT_ID}...")
    response = client.list_voices(language_code=locale)
    voices = sorted(
        ((v.name, texttospeech.SsmlVoiceGender(v.ssml_gender).name) for v in response.voices if "Chirp3-HD" in v.name),
        key=lambda t: t[0],
    )

    if not voices:
        print(f"[voices] No Chirp3-HD voices found for {locale} on this project. "
              f"Either Chirp3-HD isn't enabled for this locale on your account yet, "
              f"or the Cloud Text-to-Speech API needs enabling in the GCP console.")
        return

    print(f"[voices] found {len(voices)} real bn-IN Chirp3-HD voices\n")

    OUT_DIR.mkdir(exist_ok=True)
    results = []
    for name, gender in voices:
        out_path = OUT_DIR / f"{name}_{gender}.mp3"
        print(f"  synthesizing {name} ({gender})...")
        try:
            audio = _synthesize_with_retry(client, texttospeech, SAMPLE_TEXT_BN, locale, name)
            out_path.write_bytes(audio)
            results.append((name, gender, "OK", str(out_path)))
        except Exception as e:
            results.append((name, gender, f"FAILED: {e}", None))
        time.sleep(0.5)  # gentle pacing, avoid tripping any per-second quota

    print(f"\n[voices] done -> {OUT_DIR}\n")
    print(f"{'VOICE':<30} {'GENDER':<8} STATUS")
    for name, gender, status, path in results:
        print(f"{name:<30} {gender:<8} {status}")

    ok_count = sum(1 for _, _, status, _ in results if status == "OK")
    print(f"\n{ok_count}/{len(results)} samples synthesized successfully -> {OUT_DIR}")
    print("Play them and tell me which name you like best — that becomes "
          "settings.NARRATOR_VOICES['bn'].")


if __name__ == "__main__":
    main()
