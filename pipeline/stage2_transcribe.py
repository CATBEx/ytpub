"""
Stage 2 — Transcribe
Run faster-whisper (CPU, free, local model) on the extracted audio to get
timestamped dialogue segments. Saves a JSON transcript.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings


def transcribe(audio_path: Path) -> Path:
    from faster_whisper import WhisperModel  # imported lazily so stage1-only runs don't need it

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"[stage2] loading whisper model '{settings.WHISPER_MODEL_SIZE}' "
          f"({settings.WHISPER_DEVICE}/{settings.WHISPER_COMPUTE_TYPE}, "
          f"cpu_threads={settings.WHISPER_CPU_THREADS})...")
    model = WhisperModel(
        settings.WHISPER_MODEL_SIZE,
        device=settings.WHISPER_DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE,
        cpu_threads=settings.WHISPER_CPU_THREADS,
    )

    print(f"[stage2] transcribing {audio_path.name} (this can take a while on CPU)...")
    segments, info = model.transcribe(str(audio_path), beam_size=5, vad_filter=True)

    transcript = {
        "language": info.language,
        "duration": info.duration,
        "segments": [],
    }
    for seg in segments:
        transcript["segments"].append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        })
        print(f"  [{seg.start:7.1f}s - {seg.end:7.1f}s] {seg.text.strip()}")

    out_path = settings.TRANSCRIPT_DIR / f"{audio_path.stem}.json"
    out_path.write_text(json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[stage2] transcript saved -> {out_path} ({len(transcript['segments'])} segments)")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python stage2_transcribe.py <audio.wav>")
        sys.exit(1)
    transcribe(Path(sys.argv[1]))
