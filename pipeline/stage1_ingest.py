"""
Stage 1 — Ingest
Extract a mono 16kHz WAV audio track from the source MP4 using ffmpeg.
Free, CPU-only, no external deps beyond the ffmpeg binary being on PATH.
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings


def extract_audio(input_video: Path) -> Path:
    """Extract audio from input_video into data/audio/<stem>.wav. Returns the wav path."""
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    out_wav = settings.AUDIO_DIR / f"{input_video.stem}.wav"

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-vn",                 # no video
        "-acodec", "pcm_s16le",
        "-ar", "16000",        # 16kHz — what whisper expects
        "-ac", "1",            # mono
        str(out_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed:\n{result.stderr}")

    print(f"[stage1] extracted audio -> {out_wav}")
    return out_wav


def extract_audio_mp3(input_media: Path, out_path: Path) -> Path:
    """New (2026-09-05, folder-structure reorg): extract audio as a compressed MP3 to an
    ARBITRARY destination path (unlike extract_audio() above, which always writes WAV to the
    fixed data/audio/{stem}.wav location for the main pipeline's own Stage 1/2). Used by
    pipeline/subtitles.py to cache ONE full-audio copy per movie (data/Subtitle/{movie}/
    audio.mp3) that detect_language() and generate_subtitle() both read from, so a movie's
    (often large) video container is only ever audio-decoded once, not once per operation.
    MP3 chosen over WAV here specifically because it's what the user asked to see in the
    per-movie folder, and it's a fraction of the disk space of an uncompressed WAV for a full
    movie's audio — Whisper reads either format equally well.

    `input_media` may itself be a video OR an existing audio file (ffmpeg doesn't care)."""
    if not input_media.exists():
        raise FileNotFoundError(f"Input media not found: {input_media}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_media),
        "-vn",
        "-acodec", "libmp3lame",
        "-ar", "16000",        # 16kHz mono is plenty for Whisper -- keeps the file small
        "-ac", "1",
        "-q:a", "4",            # reasonable-quality VBR
        str(out_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg mp3 audio extraction failed:\n{result.stderr}")

    print(f"[stage1] extracted audio -> {out_path}")
    return out_path


def extract_audio_sample(input_video: Path, start_seconds: float = 60.0, duration_seconds: float = 45.0) -> Path:
    """New (2026-09-05, real-machine bug fix): extract a SHORT audio clip rather than the
    whole track — for quick language-ID only (pipeline/subtitles.py's detect_language()),
    where faster-whisper computing a mel-spectrogram feature array for an ENTIRE long movie
    just to guess its language turned out to be a real memory problem on a real run: a
    ~105-minute audio track needed a ~484 MiB single array allocation and crashed with
    numpy.core._exceptions._ArrayMemoryError on the user's machine (i5 8th gen, modest RAM —
    see config/settings.py's WHISPER_CPU_THREADS comment). detect_language()'s own docstring
    always claimed language-ID only costs "roughly the first 30s of audio" — true of
    faster-whisper's actual DECODING step, but not of feeding it a full-length file to begin
    with, which still pays for the full feature-extraction pass up front regardless. This
    function makes that claim actually true by handing Whisper a genuinely short clip.

    Starts 60s in rather than at 0:00 — skips the silent/logo-only/quiet-score opening most
    movies have, landing more reliably on real spoken dialogue than sampling from the very
    start would. Falls back to sampling from 0:00 if the movie itself is shorter than that.

    Returns a throwaway temp wav path under data/audio/ — caller is responsible for deleting
    it once done (it is NOT the movie's real extracted-audio file and must not collide with
    extract_audio()'s own output for the same movie)."""
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")

    duration = get_duration_seconds(input_video)
    start = start_seconds if duration > start_seconds + 5 else 0.0

    out_wav = settings.AUDIO_DIR / f"{input_video.stem}.lang_sample.wav"
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),     # BEFORE -i: fast (keyframe-ish) seek -- fine for a language
                                # guess, doesn't need frame-accurate positioning
        "-i", str(input_video),
        "-t", str(duration_seconds),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(out_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg audio-sample extraction failed:\n{result.stderr}")

    return out_wav


def get_duration_seconds(input_video: Path) -> float:
    """Return video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_video),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{result.stderr}")
    return float(result.stdout.strip())


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python stage1_ingest.py <input_video.mp4>")
        sys.exit(1)
    video_path = Path(sys.argv[1])
    extract_audio(video_path)
