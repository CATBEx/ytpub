"""
One-off voice sample generator — NOT part of the pipeline.

Samples ElevenLabs voices — a separate vendor from Google Cloud, widely
considered the current top tier for natural-sounding narration. This is a
NEW integration: its own account, its own API key, its own billing,
entirely outside your GCP credit. Only run this after you've done the
one-time setup below.

--- One-time setup (you have to do this part yourself) ---
1. Create an account at https://elevenlabs.io if you don't have one.
   Free tier: 10,000 credits/month. Paid tiers start at $5/month (Starter,
   30,000 credits) and $22/month (Creator, 100,000 credits) if you want to
   keep testing/using it beyond the free quota.
2. Get an API key: https://elevenlabs.io/app/settings/api-keys
3. In this project folder, create a file named ".env" (if it doesn't
   already exist) and add this line:
       ELEVENLABS_API_KEY=your_key_here
4. Pick 2-3 candidate voices for a deep/documentary/narrator style:
   go to https://elevenlabs.io/app/voice-library, search "narrator" or
   "documentary", open a voice you like, and copy its Voice ID (shown on
   the voice's page). Paste those into CANDIDATE_VOICES below, replacing
   the placeholder entries.
5. Install the SDK:
       pip install elevenlabs python-dotenv

--- Then run ---
    python test_voice_compare_elevenlabs.py
Optionally pass your own comparison line:
    python test_voice_compare_elevenlabs.py "Your own sample line here."

Output: data/voice_samples/elevenlabs_<label>.mp3 — one file per voice.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import settings

DEFAULT_SAMPLE_TEXT = (
    "Boba Fett has claimed Jabba's throne, but ruling with respect instead of "
    "fear puts him on a collision course with everyone who expects a tyrant."
)

# EDIT THIS: replace with real Voice IDs from your own ElevenLabs Voice
# Library (step 4 above) — these placeholders will not work as-is.
CANDIDATE_VOICES = {
    # "label-for-filename": "voice_id_from_elevenlabs_dashboard",
    "candidate-1": "REPLACE_WITH_REAL_VOICE_ID",
    "candidate-2": "REPLACE_WITH_REAL_VOICE_ID",
}

MODEL_ID = "eleven_v3"


def main():
    import os
    from dotenv import load_dotenv
    from elevenlabs.client import ElevenLabs

    load_dotenv()
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        print("[voice-compare-elevenlabs] ERROR: ELEVENLABS_API_KEY not set.")
        print("  Add it to a .env file in this project folder — see the setup steps "
              "in this script's docstring.")
        sys.exit(1)

    if any(v == "REPLACE_WITH_REAL_VOICE_ID" for v in CANDIDATE_VOICES.values()):
        print("[voice-compare-elevenlabs] ERROR: CANDIDATE_VOICES still has placeholder "
              "voice IDs.")
        print("  Edit this script and paste real Voice IDs from "
              "https://elevenlabs.io/app/voice-library — see the setup steps in this "
              "script's docstring.")
        sys.exit(1)

    sample_text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SAMPLE_TEXT
    out_dir = settings.DATA_DIR / "voice_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[voice-compare-elevenlabs] text: {sample_text!r}")
    print(f"[voice-compare-elevenlabs] model={MODEL_ID}")

    client = ElevenLabs(api_key=api_key)

    for label, voice_id in CANDIDATE_VOICES.items():
        print(f"\n[voice-compare-elevenlabs] synthesizing '{label}' ({voice_id})...")
        audio = client.text_to_speech.convert(
            text=sample_text,
            voice_id=voice_id,
            model_id=MODEL_ID,
            output_format="mp3_44100_128",
        )
        out_path = out_dir / f"elevenlabs_{label}.mp3"
        # convert() returns a generator of audio chunks
        with out_path.open("wb") as f:
            for chunk in audio:
                f.write(chunk)
        print(f"[voice-compare-elevenlabs]   -> {out_path}")

    print(f"\n[voice-compare-elevenlabs] done — {len(CANDIDATE_VOICES)} voice samples in {out_dir}")


if __name__ == "__main__":
    main()
