"""
Stage 6 — Score importance (Gemini via Vertex ADC, batched, multi-dimensional)
PAID stage — billed to your GCP credit, but cheap. Batching segments into one
call each (instead of one call per segment) cuts both runtime and exposure to
rate limits dramatically.

Auth: Application Default Credentials only. Do NOT set GEMINI_API_KEY.
Run once beforehand:  gcloud auth application-default login
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings

BATCH_PROMPT_TEMPLATE = """You are scoring segments of a movie for a YouTube "movie review" \
highlight reel. For EACH segment below, rate it on five 0-100 scales:

- plot_score: how plot-critical / narratively important this moment is
- emotion_score: emotional weight (drama, stakes, relationships, grief, tension)
- humor_score: comedic value
- action_score: visual/physical intensity (fights, chases, high-energy movement)
- quotability_score: how striking, poetic, or quotable the DIALOGUE PHRASING itself is —
  score this high for a great line even if the moment is low-stakes for the plot

Segments (index, time range, dialogue, and objective audio/visual signal readings 0-1 \
where higher = louder/more intense or more visual motion):

{segments_block}

Respond with ONLY a JSON array, no other text, one object per segment in the SAME ORDER, \
each shaped exactly like:
{{"index": <int>, "plot_score": <0-100>, "emotion_score": <0-100>, "humor_score": <0-100>, \
"action_score": <0-100>, "quotability_score": <0-100>, "reason": "<one short phrase>"}}
"""

SEGMENT_LINE_TEMPLATE = (
    "[{index}] {start:.1f}s-{end:.1f}s | audio_energy={audio_energy_norm} motion={motion_norm} "
    "| dialogue: \"{text}\""
)


def _merge_transcript_into_segments(transcript_path: Path, signals_path: Path):
    transcript = json.loads(transcript_path.read_text(encoding="utf-8"))["segments"]
    signals = json.loads(signals_path.read_text(encoding="utf-8"))["segments"]

    merged = []
    for sig in signals:
        text_parts = [
            t["text"] for t in transcript
            if sig["start"] <= (t["start"] + t["end"]) / 2 <= sig["end"]
        ]
        merged.append({
            **sig,
            "text": " ".join(text_parts).strip(),
        })
    return merged


def _heuristic_score(seg: dict) -> dict:
    """Signal-only fallback for segments with no dialogue — skips the LLM call entirely."""
    energy = seg.get("audio_energy_norm", 0)
    motion = seg.get("motion_norm", 0)
    return {
        "plot_score": 0,
        "emotion_score": 0,
        "humor_score": 0,
        "action_score": round(100 * max(energy, motion)),
        "quotability_score": 0,
        "reason": "no dialogue — signal-only heuristic score",
    }


def _build_batch_prompt(batch: list) -> str:
    lines = []
    for i, seg in enumerate(batch):
        lines.append(SEGMENT_LINE_TEMPLATE.format(
            index=i,
            start=seg["start"],
            end=seg["end"],
            audio_energy_norm=seg.get("audio_energy_norm", 0),
            motion_norm=seg.get("motion_norm", 0),
            text=seg["text"],
        ))
    return BATCH_PROMPT_TEMPLATE.format(segments_block="\n".join(lines))


def _parse_batch_response(raw_text: str, expected_count: int) -> list:
    raw = raw_text.strip().strip("`")
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError(f"expected a JSON array, got {type(parsed).__name__}")
    by_index = {int(item["index"]): item for item in parsed}
    results = []
    for i in range(expected_count):
        item = by_index.get(i)
        if item is None:
            raise ValueError(f"batch response missing index {i}")
        results.append(item)
    return results


def _call_batch_with_retry(client, batch: list):
    """Returns a list of score dicts (len == len(batch)), or raises after exhausting retries."""
    prompt = _build_batch_prompt(batch)
    last_error = None
    for attempt in range(settings.SCORE_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
            )
            parsed = _parse_batch_response(response.text, len(batch))
            return parsed
        except Exception as e:
            last_error = e
            is_rate_limit = "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)
            if attempt < settings.SCORE_MAX_RETRIES:
                backoff = settings.SCORE_RETRY_BACKOFF_BASE * (2 ** attempt)
                reason = "rate limited" if is_rate_limit else "error"
                print(f"  [retry] batch {reason}, attempt {attempt + 1}/{settings.SCORE_MAX_RETRIES}, "
                      f"waiting {backoff}s: {e}")
                time.sleep(backoff)
            else:
                print(f"  [warn] batch failed after {settings.SCORE_MAX_RETRIES} retries: {e}")
    raise last_error


def score_segments(video_stem: str) -> Path:
    from google import genai

    transcript_path = settings.TRANSCRIPT_DIR / f"{video_stem}.json"
    signals_path = settings.ANALYSIS_DIR / f"{video_stem}_signals.json"
    segments = _merge_transcript_into_segments(transcript_path, signals_path)

    # Split into segments needing an LLM call (have dialogue) vs. heuristic-only (silent shots)
    llm_indices = [i for i, s in enumerate(segments) if s["text"]]
    heuristic_indices = [i for i, s in enumerate(segments) if not s["text"]]

    for i in heuristic_indices:
        segments[i].update(_heuristic_score(segments[i]))

    if llm_indices:
        if not settings.GCP_PROJECT_ID:
            raise RuntimeError(
                "GCP_PROJECT_ID is not set. Set it in config/settings.py or as an env var "
                "before running Stage 6 (this stage bills your GCP credit)."
            )
        client = genai.Client(
            vertexai=True,
            project=settings.GCP_PROJECT_ID,
            location=settings.GCP_LOCATION,
        )

        batch_size = settings.SCORE_BATCH_SIZE
        n_batches = (len(llm_indices) + batch_size - 1) // batch_size
        print(f"[stage6] scoring {len(llm_indices)} dialogue segments in {n_batches} batch(es) "
              f"of up to {batch_size} via {settings.GEMINI_MODEL} "
              f"(Vertex ADC, project={settings.GCP_PROJECT_ID})...")

        for b in range(n_batches):
            batch_indices = llm_indices[b * batch_size:(b + 1) * batch_size]
            batch = [segments[i] for i in batch_indices]
            try:
                results = _call_batch_with_retry(client, batch)
                for idx, result in zip(batch_indices, results):
                    for cat in settings.SCORE_CATEGORIES:
                        segments[idx][cat] = int(result.get(cat, 0))
                    segments[idx]["reason"] = result.get("reason", "")
            except Exception as e:
                # whole batch failed even after retries — fall back to signal-only scores
                # rather than losing these segments to a score of 0 across the board
                print(f"  [warn] batch {b + 1}/{n_batches} failed entirely, using heuristic "
                      f"fallback for its {len(batch)} segments: {e}")
                for idx in batch_indices:
                    segments[idx].update(_heuristic_score(segments[idx]))
                    segments[idx]["reason"] = f"batch scoring failed: {e}"

            done = min((b + 1) * batch_size, len(llm_indices))
            print(f"  [stage6] {done}/{len(llm_indices)} dialogue segments scored")

    # overall_score = the single number Stage 7's quality floor and legacy tooling can use
    for seg in segments:
        seg["overall_score"] = max(seg[cat] for cat in settings.SCORE_CATEGORIES)

    out_path = settings.ANALYSIS_DIR / f"{video_stem}_scored.json"
    out_path.write_text(json.dumps({"segments": segments}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[stage6] scoring complete -> {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python stage5_score.py <video_stem>  (e.g. 'my_movie', no extension)")
        sys.exit(1)
    score_segments(sys.argv[1])
