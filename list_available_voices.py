"""
One-off diagnostic — NOT part of the pipeline.

Queries Google's own ListVoices API directly (the authoritative source,
unlike scraped docs) to print every real en-US voice name available on
your account/project, split into Chirp3-HD, Gemini TTS-usable, and
everything else. Use this to get exact, correct voice names before running
test_voice_compare_chirp3hd.py or test_voice_compare_gemini.py.

Run:
    python list_available_voices.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings


def main():
    from google.cloud import texttospeech

    locale = settings.LOCALE_BY_LANGUAGE["en"]  # "en-US"
    client = texttospeech.TextToSpeechClient()

    response = client.list_voices(language_code=locale)
    names = sorted(v.name for v in response.voices)

    chirp3hd = [n for n in names if "Chirp3-HD" in n]
    chirp_other = [n for n in names if "Chirp" in n and "Chirp3-HD" not in n]
    other = [n for n in names if "Chirp" not in n]

    print(f"[list-voices] {len(names)} total voices for locale={locale}\n")

    print(f"--- Chirp3-HD ({len(chirp3hd)}) ---")
    for n in chirp3hd:
        print(" ", n)

    if chirp_other:
        print(f"\n--- Other Chirp tiers ({len(chirp_other)}) ---")
        for n in chirp_other:
            print(" ", n)

    print(f"\n--- Everything else, incl. Standard/Neural2/Wavenet/Studio ({len(other)}) ---")
    for n in other:
        print(" ", n)

    print("\n[list-voices] Note: bare Gemini TTS voice names (e.g. 'Charon', not "
          "'en-US-Chirp3-HD-Charon') are a SEPARATE API surface and typically do NOT "
          "show up in this locale-scoped list — they're selected by name + "
          "model_name=settings.GEMINI_TTS_MODEL directly, as the pipeline already does. "
          "This list is mainly to get exact Chirp3-HD / Standard / Neural2 / Studio names.")


if __name__ == "__main__":
    main()
