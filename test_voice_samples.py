"""
One-off voice sample generator — NOT part of the pipeline.

Synthesizes the SAME narration line at each intensity style
(settings.INTENSITY_STYLE_PROMPTS: calm / moderate / intense) using the
current narrator voice (settings.NARRATOR_VOICE, "Charon"), so you can
listen to all three back to back and decide whether to keep the per-beat
adaptive intensity design or lock in a single style for every beat.

Run for real (needs your GCP ADC creds — same auth as the real pipeline):
    python test_voice_samples.py
Optionally pass your own comparison line:
    python test_voice_samples.py "Your own sample line here."

Output: data/voice_samples/charon_<intensity>.mp3 — one file per intensity,
same text in every file, only the style prompt differs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings

DEFAULT_SAMPLE_TEXT = (
    "Boba Fett has claimed Jabba's throne, but ruling with respect instead of "
    "fear puts him on a collision course with everyone who expects a tyrant."
)


def main():
    from google.cloud import texttospeech

    sample_text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SAMPLE_TEXT
    locale = settings.LOCALE_BY_LANGUAGE["en"]

    out_dir = settings.DATA_DIR / "voice_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[voice-samples] text: {sample_text!r}")
    print(f"[voice-samples] voice={settings.NARRATOR_VOICE}, model={settings.GEMINI_TTS_MODEL}, "
          f"locale={locale}")

    client = texttospeech.TextToSpeechClient()  # ADC, same as the real pipeline

    for intensity, style_prompt in settings.INTENSITY_STYLE_PROMPTS.items():
        print(f"\n[voice-samples] synthesizing '{intensity}': {style_prompt}")
        response = client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=sample_text, prompt=style_prompt),
            voice=texttospeech.VoiceSelectionParams(
                language_code=locale,
                name=settings.NARRATOR_VOICE,
                model_name=settings.GEMINI_TTS_MODEL,
            ),
            audio_config=texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3),
        )
        out_path = out_dir / f"charon_{intensity}.mp3"
        out_path.write_bytes(response.audio_content)
        print(f"[voice-samples]   -> {out_path}")

    print(f"\n[voice-samples] done — {len(settings.INTENSITY_STYLE_PROMPTS)} samples in {out_dir}")
    print("[voice-samples] all three use the SAME text — only the style/intensity prompt differs.")


if __name__ == "__main__":
    main()
