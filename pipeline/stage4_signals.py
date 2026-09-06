"""
Stage 4 & 5 — Signal extraction
Audio energy (librosa) + visual motion (OpenCV frame-diff), computed per
shot from Stage 3, so every candidate segment carries objective signal
data into the Stage 6 scoring prompt (not just raw transcript text).
Free, CPU-only.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings


def _audio_energy_for_range(y, sr, start, end):
    """RMS loudness for [start, end] seconds of an already-loaded waveform."""
    i0, i1 = int(start * sr), int(end * sr)
    if i1 <= i0 or i1 > len(y):
        i1 = min(len(y), max(i1, i0 + 1))
    clip = y[i0:i1]
    if len(clip) == 0:
        return 0.0
    import librosa
    rms = librosa.feature.rms(y=clip)[0]
    return float(np.mean(rms))


def _motion_for_range(video_path, start, end, sample_fps=2):
    """Average frame-to-frame pixel difference over [start, end] as a motion proxy."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frame_interval = max(1, int(fps / sample_fps))

    start_frame = int(start * fps)
    end_frame = int(end * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    prev_gray = None
    diffs = []
    frame_idx = start_frame
    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx - start_frame) % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (160, 90))  # downscale for speed
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                diffs.append(float(np.mean(diff)))
            prev_gray = gray
        frame_idx += 1
    cap.release()
    return float(np.mean(diffs)) if diffs else 0.0


def extract_signals(video_path: Path, audio_path: Path, shots_path: Path) -> Path:
    import librosa

    shots = json.loads(shots_path.read_text(encoding="utf-8"))["shots"]

    print(f"[stage4/5] loading audio waveform for energy analysis...")
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)

    signals = []
    for shot in shots:
        start, end = shot["start"], shot["end"]
        duration = end - start
        if duration < settings.SEGMENT_MIN_DURATION:
            continue  # too short to be a usable clip on its own

        audio_energy = _audio_energy_for_range(y, sr, start, end)
        motion = _motion_for_range(video_path, start, end)

        signals.append({
            "start": start,
            "end": end,
            "duration": round(duration, 2),
            "audio_energy": round(audio_energy, 5),
            "motion": round(motion, 3),
        })

    # Normalize audio_energy and motion to 0-1 so they're comparable/combinable later
    if signals:
        energies = [s["audio_energy"] for s in signals]
        motions = [s["motion"] for s in signals]
        e_min, e_max = min(energies), max(energies)
        m_min, m_max = min(motions), max(motions)
        for s in signals:
            s["audio_energy_norm"] = round((s["audio_energy"] - e_min) / (e_max - e_min + 1e-9), 3)
            s["motion_norm"] = round((s["motion"] - m_min) / (m_max - m_min + 1e-9), 3)

    out_path = settings.ANALYSIS_DIR / f"{video_path.stem}_signals.json"
    out_path.write_text(json.dumps({"segments": signals}, indent=2), encoding="utf-8")
    print(f"[stage4/5] signals computed for {len(signals)} segments -> {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python stage4_signals.py <input_video.mp4> <audio.wav> <shots.json>")
        sys.exit(1)
    extract_signals(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
