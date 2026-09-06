"""
Gemini TTS — deep male voice comparison, EN / HI / BN
Round 2 of voice testing. Round 1 confirmed the mechanics (bare voice names,
locale quirks — bn-IN is rejected, bn-BD works) but only exercised female
voices (Kore, Achernar). This round compares two MALE voices with deeper/
more authoritative characteristics, per Google's official voice table:

    Charon   — "Informative"  (calm, steady, documentary-narrator style)
    Algenib  — "Gravelly"     (the deepest/roughest of the 30 Gemini voices)

Locales confirmed working from round 1: en-US, hi-IN, bn-BD (NOT bn-IN).

Usage:
    venv\\Scripts\\activate
    python test_gemini_tts.py

Output: MP3 files in data/tts_test/ named <locale>_<voice>_<intensity>.mp3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings

MODEL_NAME = "gemini-2.5-flash-tts"

SAMPLE_LINES = {
    "en-US": "After years of silence, Maya finally opens the letter her mother left behind.",
    "hi-IN": "वर्षों की खामोशी के बाद, माया आख़िरकार अपनी माँ की छोड़ी हुई चिट्ठी खोलती है।",
    "bn-BD": "বছরের নীরবতার পর, মায়া অবশেষে তার মায়ের রেখে যাওয়া চিঠিটি খোলে।",
}

MALE_VOICE_CANDIDATES = ["Charon", "Algenib"]

STYLE_PROMPTS = {
    "calm": "Narrate this calmly and warmly, at a slow, measured pace, like a gentle bedtime story.",
    "intense": "Narrate this with urgency and rising tension, a faster pace, and dramatic emphasis, "
               "like the climax of a thriller.",
}

OUT_DIR = settings.DATA_DIR / "tts_test"


def try_synthesize(client, texttospeech, locale: str, voice_name: str, text: str, prompt: str, out_path: Path):
    try:
        response = client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=text, prompt=prompt),
            voice=texttospeech.VoiceSelectionParams(
                language_code=locale, name=voice_name, model_name=MODEL_NAME,
            ),
            audio_config=texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3),
        )
        out_path.write_bytes(response.audio_content)
        print(f"  [OK] {out_path.name} ({len(response.audio_content)} bytes)")
        return True
    except Exception as e:
        print(f"  [FAIL] {locale} / {voice_name}: {type(e).__name__}: {e}")
        return False


def main():
    from google.cloud import texttospeech

    if not settings.GCP_PROJECT_ID:
        print("[FAIL] GCP_PROJECT_ID is empty. Set it in config/settings.py first.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = texttospeech.TextToSpeechClient()  # ADC

    print(f"Project: {settings.GCP_PROJECT_ID}  Model: {MODEL_NAME}")
    print(f"Output folder: {OUT_DIR}\n")

    any_ok = False
    for locale, text in SAMPLE_LINES.items():
        for voice_name in MALE_VOICE_CANDIDATES:
            print(f"\n{locale} / {voice_name}:")
            for intensity, prompt in STYLE_PROMPTS.items():
                out_path = OUT_DIR / f"{locale}_{voice_name}_{intensity}.mp3"
                ok = try_synthesize(client, texttospeech, locale, voice_name, text, prompt, out_path)
                any_ok = any_ok or ok

    print("\n" + "=" * 70)
    if any_ok:
        print(f"Done. Files are in {OUT_DIR} — 2 voices x 3 languages x 2 intensities.")
        print("Listen and tell me: which voice (Charon or Algenib) do you want as the default "
              "narrator, and does it hold up across all three languages or only some?")
    else:
        print("Nothing synthesized successfully — see [FAIL] lines above.")
    print("=" * 70)


if __name__ == "__main__":
    main()
