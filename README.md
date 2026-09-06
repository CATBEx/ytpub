# Movie Review Video Pipeline

Feed in a movie MP4 → detects important scenes → outputs one assembled
rough-cut review video. Everything runs locally/CPU and is free, **except**
Stage 6 (importance scoring), which calls Gemini Flash via Vertex AI and is
billed to your GCP credit (typically ~$0.01–0.05 per movie).

## Setup

```powershell
cd "Video Production"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

You'll also need `ffmpeg` on PATH (https://ffmpeg.org/download.html — grab a
Windows build, add its `bin` folder to PATH).

### Vertex ADC auth (Stage 6 only)
Do NOT use a `GEMINI_API_KEY`. Authenticate with Application Default
Credentials instead (same pattern as the Cartoon Production pipeline):

```powershell
gcloud auth application-default login
```

Then set your project in `config/settings.py` (or as env vars):

```python
GCP_PROJECT_ID = "your-gcp-project-id"
GCP_LOCATION = "us-central1"
```

## Usage

### Option A — CLI (one-off runs)

Drop a movie file into `data/input/`, then:

```powershell
python main.py "data/input/my_movie.mp4"
```

### Option B — Dashboard + API (recommended for repeated/production use)

```powershell
uvicorn api.server:app --host 0.0.0.0 --port 9999 --reload
```

Then open **http://localhost:9999/dashboard** — upload a movie there, watch
it move through the 8 stages, preview/download the finished rough-cut, and
re-tune `IMPORTANCE_THRESHOLD` with a slider (re-cutting is fast — it reuses
the already-computed transcript/scores, no re-transcription, no new paid
Vertex calls).

API endpoints (used by the dashboard, callable directly too):

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/v1/pipeline/submit` | upload a movie (multipart `file`), starts the pipeline in the background |
| GET | `/api/v1/pipeline/jobs` | list all jobs |
| GET | `/api/v1/pipeline/jobs/{id}` | job status, current stage, logs |
| GET | `/api/v1/pipeline/jobs/{id}/scores` | per-segment scored data |
| POST | `/api/v1/pipeline/jobs/{id}/reselect?threshold=N` | re-cut with a new threshold (fast, no re-scoring) |
| GET | `/api/v1/pipeline/jobs/{id}/download` | download/stream the finished rough-cut |

Jobs run one background thread each, tracked in memory (server restart loses
job history, but files already written to `data/` are untouched). Fine for
single-user local use; not built for concurrent multi-user production.

Note: jobs are keyed by filename, matching the CLI's `data/` layout —
uploading two different files with the same name will overwrite each
other's intermediate files.

Either way, the final rough-cut lands in `data/output/<name>_review_cut.mp4`.

Each stage also writes its intermediate output to `data/` so you can inspect
or re-run a single stage without redoing everything:

| Stage | Script | Output |
|---|---|---|
| 1. Extract audio | `pipeline/stage1_ingest.py` | `data/audio/<name>.wav` |
| 2. Transcribe | `pipeline/stage2_transcribe.py` | `data/transcripts/<name>.json` |
| 3. Shot detection | `pipeline/stage3_shots.py` | `data/analysis/<name>_shots.json` |
| 4/5. Signal extraction | `pipeline/stage4_signals.py` | `data/analysis/<name>_signals.json` |
| 6. Importance scoring (PAID) | `pipeline/stage5_score.py` | `data/analysis/<name>_scored.json` |
| 7. Selection | `pipeline/stage6_select.py` | `data/analysis/<name>_selected.json` |
| 8. Cut + assemble | `pipeline/stage7_assemble.py` | `data/output/<name>_review_cut.mp4` |

## Tuning

All the knobs live in `config/settings.py`:

- `IMPORTANCE_THRESHOLD` (default 70) — the main dial. Output length is NOT
  fixed; it's however many segments score at/above this. Lower it for a
  longer rough-cut, raise it for a tighter one. Start here after your first
  real run — look at `data/analysis/<name>_scored.json` to see the actual
  score distribution for that movie before deciding.
- `WHISPER_MODEL_SIZE` — `small` by default (CPU-friendly). Bump to `medium`
  for better transcript accuracy at the cost of slower runs.
- `SCENE_DETECT_THRESHOLD` — lower = more (shorter) shots detected.
- `MAX_OUTPUT_DURATION` — optional hard cap in seconds, in case a very
  eventful movie produces an overlong cut. `None` by default (no cap).
- `TRANSITION_TYPE` / `TRANSITION_DURATION` — crossfade style between clips.

## Known first-run gotchas

- First `faster-whisper` run downloads the model weights (one-time, a few
  hundred MB depending on size) — needs internet once, then works offline.
- Stage 6 requires `GCP_PROJECT_ID` to be set — it raises a clear error if
  you try to run it without one, rather than silently failing.
- A 2GB local GPU is intentionally unused — Whisper runs on CPU
  (`WHISPER_DEVICE = "cpu"` in settings) since the card is too small to help
  meaningfully. No code path here touches the GPU.

## Status

Scaffolded 2026-08-26, dashboard/API added same day. Every stage verified
individually on synthetic test clips (audio extraction, shot detection,
signal extraction, selection, multi-clip crossfade assembly, and the API's
submit/status/error-handling flow) — not yet run end-to-end on a real
movie. Next step is a real run to calibrate `IMPORTANCE_THRESHOLD` and check
transcript/shot quality on actual content.
