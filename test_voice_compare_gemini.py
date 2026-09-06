"""
One-off voice sample generator — NOT part of the pipeline.

Samples SEVERAL Gemini TTS male voices (not just Charon) reading the SAME
line, all with the same "moderate" style prompt, so you can compare voice
character apples-to-apples before picking a narrator. This is a companion
to test_voice_samples.py (which compares Charon across intensity levels,
not across different voices).

Candidates below are Google's own published male voices for Gemini TTS,
narrowed to ones whose stated character (per Google's docs) leans deep/
authoritative/informative rather than upbeat/casual/youthful — Charon and
Algenib were already tried; these are new ones worth a listen.

Run for real (needs your GCP ADC creds — same auth as the real pipeline):
    python test_voice_compare_gemini.py
Optionally pass your own comparison line:
    python test_voice_compare_gemini.py "Your own sample line here."

Output: data/voice_samples/gemini_<voice>.mp3 — one file per voice.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings

DEFAULT_SAMPLE_TEXT = (
    "Boba Fett has claimed Jabba's throne, but ruling with respect instead of "
    "fear puts him on a collision course with everyone who expects a tyrant."
)

# name -> Google's own one-word character label for this voice
CANDIDATE_VOICES = {
    "Charon": "Informative",     # already the current NARRATOR_VOICE, included as the baseline
    "Orus": "Firm",
    "Iapetus": "Clear",
    "Alnilam": "Firm",
    "Rasalgethi": "Informative",
    "Sadaltager": "Knowledgeable",
}

STYLE_PROMPT = settings.INTENSITY_STYLE_PROMPTS["moderate"]  # one fixed style, for apples-to-apples


def main():
    from google.cloud import texttospeech

    sample_text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SAMPLE_TEXT
    locale = settings.LOCALE_BY_LANGUAGE["en"]

    out_dir = settings.DATA_DIR / "voice_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[voice-compare-gemini] text: {sample_text!r}")
    print(f"[voice-compare-gemini] model={settings.GEMINI_TTS_MODEL}, locale={locale}, "
          f"style='{STYLE_PROMPT}'")

    client = texttospeech.TextToSpeechClient()  # ADC, same as the real pipeline

    for voice_name, character in CANDIDATE_VOICES.items():
        print(f"\n[voice-compare-gemini] synthesizing '{voice_name}' ({character})...")
        response = client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=sample_text, prompt=STYLE_PROMPT),
            voice=texttospeech.VoiceSelectionParams(
                language_code=locale,
                name=voice_name,  # bare name — Gemini voices require this, not locale-prefixed
                model_name=settings.GEMINI_TTS_MODEL,
            ),
            audio_config=texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3),
        )
        out_path = out_dir / f"gemini_{voice_name.lower()}.mp3"
        out_path.write_bytes(response.audio_content)
        print(f"[voice-compare-gemini]   -> {out_path}")

    print(f"\n[voice-compare-gemini] done — {len(CANDIDATE_VOICES)} voice samples in {out_dir}")
    print("[voice-compare-gemini] all use the SAME text and SAME style prompt — only the voice differs.")


if __name__ == "__main__":
    main()
