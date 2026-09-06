"""
Stage 3 — Shot detection
Use PySceneDetect (free, CPU) to find shot/scene boundary timestamps.
These boundaries are later merged with the transcript + signal data to
form "candidate segments" for scoring.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings


def _to_seconds(timecode) -> float:
    """PySceneDetect's FrameTimecode API changed across versions: older
    releases (e.g. 0.6.x, what requirements.txt pins) only have
    get_seconds(); newer releases (0.7.x+) added a `seconds` property and
    deprecated get_seconds(). Support both so this doesn't break again on
    an upgrade or a different installed version."""
    try:
        return timecode.seconds
    except AttributeError:
        return timecode.get_seconds()


def detect_shots(video_path: Path) -> Path:
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"[stage3] detecting shot boundaries in {video_path.name}...")
    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=settings.SCENE_DETECT_THRESHOLD))
    scene_manager.detect_scenes(video, show_progress=True)
    # start_in_scene=True ensures we still get one full-length "shot" when the
    # detector finds zero cuts (e.g. a short/static clip), instead of an empty list
    scene_list = scene_manager.get_scene_list(start_in_scene=True)

    shots = [{
        "start": round(_to_seconds(start), 2),
        "end": round(_to_seconds(end), 2),
    } for start, end in scene_list]

    out_path = settings.ANALYSIS_DIR / f"{video_path.stem}_shots.json"
    out_path.write_text(json.dumps({"shots": shots}, indent=2), encoding="utf-8")
    print(f"[stage3] {len(shots)} shots detected -> {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python stage3_shots.py <input_video.mp4>")
        sys.exit(1)
    detect_shots(Path(sys.argv[1]))
