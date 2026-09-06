"""
One-off voice sample generator — NOT part of the pipeline.

Samples Google Cloud's Chirp 3: HD voices (a different, more established
TTS tier than Gemini TTS, described by Google as delivering "realism and
emotional resonance") PLUS the Studio tier (Google's premium voices built
specifically for long-form narration/audiobook work — exactly this use
case). Still your existing GCP project/ADC auth, no new vendor or billing.
Same voice NAME "Charon"/"Orus"/etc. exist in Chirp3-HD too, so this is a
direct comparison against what you already heard from Gemini TTS.

Voice names below are confirmed exact via list_available_voices.py against
your real account (2026-08-26) — the earlier "D" name was bad data from a
scraped doc, not a real voice; this list is the authoritative one.

Key difference from Gemini TTS: neither tier supports the free-text style
`prompt` field Gemini TTS has — these are plain reads with no style
instruction (their realism is meant to come from the model itself). If you
like one of these better, the per-beat adaptive-intensity design in
Stage 5 would need to change to something else (a preset style, if the
tier offers one, or just a single consistent read).

Cost note: Studio voices are Google's priciest TTS tier (per Google's own
pricing page — check before heavy use). Fine for a couple of short samples
here, just don't build the full pipeline on it without checking cost per
movie first.

Run for real (needs your GCP ADC creds — same auth as the real pipeline):
    python test_voice_compare_chirp3hd.py
Optionally pass your own comparison line:
    python test_voice_compare_chirp3hd.py "Your own sample line here."

Output: data/voice_samples/<label>.mp3 — one file per voice.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings

DEFAULT_SAMPLE_TEXT = (
    "Boba Fett has claimed Jabba's throne, but ruling with respect instead of "
    "fear puts him on a collision course with everyone who expects a tyrant."
)

# label -> exact full voice name, confirmed via list_available_voices.py
# against the real account (not guessed/scraped).
CANDIDATE_VOICES = {
    "chirp3hd_charon": "en-US-Chirp3-HD-Charon",
    "chirp3hd_orus": "en-US-Chirp3-HD-Orus",
    "chirp3hd_fenrir": "en-US-Chirp3-HD-Fenrir",
    "chirp3hd_enceladus": "en-US-Chirp3-HD-Enceladus",
    "chirp3hd_puck": "en-US-Chirp3-HD-Puck",
    "chirp3hd_rasalgethi": "en-US-Chirp3-HD-Rasalgethi",
    "studio_o": "en-US-Studio-O",
    "studio_q": "en-US-Studio-Q",
}


def main():
    from google.cloud import texttospeech

    sample_text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SAMPLE_TEXT
    locale = settings.LOCALE_BY_LANGUAGE["en"]  # "en-US"

    out_dir = settings.DATA_DIR / "voice_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[voice-compare-chirp3hd] text: {sample_text!r}")
    print(f"[voice-compare-chirp3hd] locale={locale} (no style prompt — neither tier supports one)")

    client = texttospeech.TextToSpeechClient()  # ADC, same as the real pipeline

    for label, full_voice_name in CANDIDATE_VOICES.items():
        print(f"\n[voice-compare-chirp3hd] synthesizing '{full_voice_name}'...")
        response = client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=sample_text),  # no `prompt` — unsupported here
            voice=texttospeech.VoiceSelectionParams(
                language_code=locale,
                name=full_voice_name,
            ),
            audio_config=texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3),
        )
        out_path = out_dir / f"{label}.mp3"
        out_path.write_bytes(response.audio_content)
        print(f"[voice-compare-chirp3hd]   -> {out_path}")

    print(f"\n[voice-compare-chirp3hd] done — {len(CANDIDATE_VOICES)} voice samples in {out_dir}")


if __name__ == "__main__":
    main()
