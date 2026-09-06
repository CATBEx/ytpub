"""
Explainer Stage 4 — Clip matching
PAID — a SECOND Gemini call, schema-constrained, separate from Stage 3's
summary generation by explicit design choice (2026-08-26 redesign — the old
model did both jobs in one call). Reads Stage 3's beats plus the FULL
transcript, and for each beat returns which transcript segment(s) it's
actually describing — BY INDEX ONLY, never a timestamp value. Python then
resolves the real start/end from Whisper's own transcript (Stage 2), so
Gemini can never hallucinate a timestamp — it can only point at a transcript
line that genuinely exists.

The schema below is a plain dict (Gemini's native OBJECT/ARRAY/STRING/
INTEGER format), not derived from a nested Pydantic model — see the longer
note in explainer_stage3_summary.py for why: a nested-Pydantic-model schema
produces `$ref`/`$defs` that google-genai's dereferencing turned out to be
sensitive to the runtime pydantic version for, causing a real failure on
the first live run. A plain dict has no `$ref` to begin with, so this can't
recur regardless of pydantic version. `.parsed` is a plain dict here.

CHANGED same day (later still, the "silent action/horror" fix): the
transcript segments used here are passed through
`pipeline/audio_events.py`'s `augment_segments()` — the SAME function Stage 3
calls — rather than being read directly from Stage 2's Whisper output. This
is required, not optional: Stage 3's prompt numbered its lines against the
audio-event-augmented list, so Stage 4's citation indices are only valid if
built against that identical list. `augment_segments()` is deterministic and
cached per video, so both stages land on the same numbering without needing
to pass anything between them.

FIXED 2026-08-27 (real bug, found on a real ~2h47m/1787-line movie run —
*The Odyssey*): each transcript line in the prompt is shown as
`[index] start-end: text` — a small bracket INDEX sitting right next to a
much larger TIMESTAMP. On a long transcript, Gemini sometimes cites the
TIMESTAMP instead of the INDEX (confirmed on the real run: cited "9209"
where the correct index was 1710, whose real start time is 9209.2s — not a
coincidence, every dropped number that session matched some real segment's
start time almost exactly). Silently dropping these left 24 of 31 beats
with zero real footage — and because `explainer_common.beat_envelope()`'s
uncited-beat fallback interpolates between the nearest CITED neighbors, a
long run of consecutively-uncited beats (as this bug reliably produces,
since it hits every later beat in a long movie) collapsed onto one single
repeated anchor point instead of spreading across the movie — the literal
"Clip 11 to 31 are the same scene" symptom.
`resolve_matches()` now tries to RECOVER an out-of-range citation by
checking whether it's numerically close to some real segment's start time
before giving up on it — see `_recover_timestamp_index()` below.

Auth: Application Default Credentials only. Do NOT set GEMINI_API_KEY.
"""
import bisect
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from pipeline import audio_events

MATCH_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "matches": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "beat_index": {"type": "INTEGER"},
                    "cited_segments": {
                        "type": "ARRAY",
                        "items": {"type": "INTEGER"},
                    },
                },
                "required": ["beat_index", "cited_segments"],
            },
        },
    },
    "required": ["matches"],
}

MATCH_PROMPT_TEMPLATE = """You are matching narration beats from a movie/series recap script \
back to the exact transcript lines they're describing. Below are the beats (in story order) \
and the FULL numbered transcript.

For EACH beat, list which transcript line index(es) it's actually describing — the footage \
that will play under that beat's narration comes from those exact lines' timestamps, so be \
precise; include every line a beat covers if it spans more than one. Use an empty array ONLY \
for a beat with no specific footage to point to (a rare pure-transition/connective beat, e.g. \
"Meanwhile, back at the house...").

Some numbered lines have no real dialogue — they mark a real, dialogue-free span of the movie \
with sustained high audio energy instead. These are valid, citable lines: if a beat's narration \
describes a wordless action/horror moment, cite the line(s) covering that span like any other.

Beats ({n_beats} total, in order):
{beats_block}

Full transcript ({n_lines} numbered lines):
{transcript_block}
"""

BEAT_LINE_TEMPLATE = "[{index}] {narration}"
LINE_TEMPLATE = "[{index}] {start:.1f}s-{end:.1f}s: {text}"


def _build_prompt(beats: list, segments: list) -> str:
    beats_block = "\n".join(
        BEAT_LINE_TEMPLATE.format(index=b["index"], narration=b["narration"]) for b in beats
    )
    transcript_block = "\n".join(
        LINE_TEMPLATE.format(index=i, start=s["start"], end=s["end"], text=s["text"])
        for i, s in enumerate(segments)
    )
    return MATCH_PROMPT_TEMPLATE.format(
        n_beats=len(beats), beats_block=beats_block,
        n_lines=len(segments), transcript_block=transcript_block,
    )


def _call_with_retry(client, prompt: str, schema):
    from google.genai import types

    last_error = None
    for attempt in range(settings.MATCH_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            parsed = response.parsed
            if parsed is None:
                raise ValueError(f"Gemini returned no parsed structured output: {response.text!r}")
            return parsed
        except Exception as e:
            last_error = e
            is_rate_limit = "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)
            if attempt < settings.MATCH_MAX_RETRIES:
                backoff = settings.MATCH_RETRY_BACKOFF_BASE * (2 ** attempt)
                reason = "rate limited" if is_rate_limit else "error"
                print(f"  [retry] matching {reason}, attempt {attempt + 1}/"
                      f"{settings.MATCH_MAX_RETRIES}, waiting {backoff}s: {e}")
                time.sleep(backoff)
    raise last_error


def _recover_timestamp_index(value, segment_starts: list, tolerance: float = 2.0):
    """An out-of-range cited 'index' is sometimes actually a TIMESTAMP Gemini
    meant to cite — the prompt shows both numbers right next to each other
    (`[index] start-end: text`), and on a long transcript the model can
    confuse the two (confirmed on a real ~1787-line movie run — see the
    module docstring). If `value` is within `tolerance` seconds of some real
    segment's start time, return that segment's real index; otherwise None.

    `segment_starts` must be sorted ascending (true of the transcript/
    audio-event-augmented segment list, which is always time-ordered).
    Pure function — no file I/O — so it's directly unit-testable."""
    if not segment_starts:
        return None
    i = bisect.bisect_left(segment_starts, value)
    candidates = [j for j in (i - 1, i) if 0 <= j < len(segment_starts)]
    if not candidates:
        return None
    best = min(candidates, key=lambda j: abs(segment_starts[j] - value))
    return best if abs(segment_starts[best] - value) <= tolerance else None


def resolve_matches(beats: list, segments: list, matches_by_beat: dict) -> list:
    """Attach cited_segments/start/end onto each beat from the (Gemini-provided)
    segment indices, looking up the REAL timestamps from the transcript. Pulled
    out as its own function so it's testable without a live Gemini call.

    Out-of-range citations get one recovery attempt (see
    `_recover_timestamp_index()`) before being dropped — this is what fixes
    the real "Clip 11 to 31 are the same scene" bug: without recovery, a
    long transcript reliably loses most of its later beats' citations
    entirely, and the uncited-beat fallback then collapses them all onto one
    repeated anchor point."""
    segment_starts = [s["start"] for s in segments]

    missing = [b["index"] for b in beats if b["index"] not in matches_by_beat]
    if missing:
        print(f"  [warn] {len(missing)} beat(s) had no match entry in the response "
              f"(indices {missing}) — treating as uncited/transition beats: {missing}")

    for beat in beats:
        cited = matches_by_beat.get(beat["index"], [])
        in_range = []
        recovered = []
        dropped = []
        for i in cited:
            if 0 <= i < len(segments):
                in_range.append(i)
                continue
            r = _recover_timestamp_index(i, segment_starts)
            if r is not None:
                recovered.append((i, r))
            else:
                dropped.append(i)
        if recovered:
            print(f"  [warn] beat {beat['index']}: recovered {len(recovered)} cited segment(s) "
                  f"Gemini cited by timestamp instead of index: "
                  f"{[f'{orig}->{idx}' for orig, idx in recovered]}")
        if dropped:
            print(f"  [warn] beat {beat['index']}: dropping unrecoverable out-of-range cited "
                  f"segment(s) {dropped} (transcript has {len(segments)} lines)")
        all_indices = sorted(set(in_range) | {idx for _, idx in recovered})
        beat["cited_segments"] = all_indices
        if all_indices:
            beat["start"] = round(min(segments[i]["start"] for i in all_indices), 2)
            beat["end"] = round(max(segments[i]["end"] for i in all_indices), 2)
        else:
            beat["start"] = None
            beat["end"] = None
    return beats


def match_clips(video_stem: str, language: str = "en") -> Path:
    from google import genai

    if language not in settings.EXPLAINER_LANGUAGE_NAMES:
        raise RuntimeError(f"Unsupported language {language!r} — must be one of "
                            f"{list(settings.EXPLAINER_LANGUAGE_NAMES)}.")

    # Per-language summary (Stage 3's output for THIS language) and per-language transcript —
    # same path conventions as explainer_stage3_summary.py / pipeline/subtitles.py.
    summary_path = settings.EXPLAINER_DIR / video_stem / language / "summary.json"
    beats = json.loads(summary_path.read_text(encoding="utf-8"))["beats"]
    transcript_path = settings.TRANSCRIPT_DIR / (
        f"{video_stem}.json" if language == "en" else f"{video_stem}.{language}.json"
    )
    segments = json.loads(transcript_path.read_text(encoding="utf-8"))["segments"]
    if not beats:
        raise RuntimeError("No beats in the summary — nothing to match.")
    if not segments:
        raise RuntimeError("Transcript has no segments — nothing to match against.")

    # Shared, language-independent (see pipeline/audio_events.py) — no language param here.
    segments = audio_events.augment_segments(video_stem, segments)

    if not settings.GCP_PROJECT_ID:
        raise RuntimeError(
            "GCP_PROJECT_ID is not set. Set it in config/settings.py or as an env var "
            "before running clip matching (this stage bills your GCP credit)."
        )

    print(f"[explainer-match] language={language!r}, matching {len(beats)} beats against "
          f"{len(segments)} transcript lines -> one Gemini call ({settings.GEMINI_MODEL}, "
          f"Vertex ADC, project={settings.GCP_PROJECT_ID}, schema-constrained)")

    client = genai.Client(vertexai=True, project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)
    prompt = _build_prompt(beats, segments)
    result = _call_with_retry(client, prompt, MATCH_SCHEMA)

    matches_by_beat = {m["beat_index"]: m["cited_segments"] for m in result["matches"]}
    beats = resolve_matches(beats, segments, matches_by_beat)

    out_dir = settings.EXPLAINER_DIR / video_stem / language
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "matched.json"
    out_path.write_text(json.dumps({"beats": beats}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[explainer-match] {len(beats)} beats matched -> {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python explainer_stage4_match.py <video_stem> [language]")
        sys.exit(1)
    match_clips(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else "en")
