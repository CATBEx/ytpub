"""
Explainer Stage 3.5 — Intro generation (new, 2026-08-27)
PAID — one Gemini call, schema-constrained, SEPARATE from Stage 3's beat summary. By explicit
user request ("add a 10 seconds intro... which topic today we will explain"): a short, ~10s
channel-style intro line spoken before the story itself begins.

NOT a plot-teaser — that's already what the "hook" beat_role beat does (a flash-forward tease of
the story's biggest moment, forced to Clip-1 by explainer_stage3_summary.py's _reindex_beats()).
This is more like a host briefly setting up what KIND of video this is ("Today, we're diving
into a tense heist thriller where a crew of strangers...") before the recap proper starts.

By explicit user-confirmed design choice, does NOT state the movie's exact title. This project's
real filenames are messy release-scene names (e.g. "Balls Up (2026) 720p AMZN-WEB x264
MSubs..."), so having Gemini name a title risks either garbage or a wrong guess; describing the
premise/genre/tone instead is safer and still hooks interest.

Unlike Stage 3/4 (which are split into two calls specifically because matching MANY paraphrased
beats back to literal dialogue is a hard semantic-matching problem deserving its own focused
call), a single intro line doesn't need that split — this ONE call both writes the narration AND
cites which real transcript line(s) best represent a striking, illustrative visual moment for it.
Citation resolution reuses `explainer_stage4_match.resolve_matches()` directly (wrapping the
intro as a single-item "beat" list) rather than reimplementing it, which gets Stage 4's
out-of-range-timestamp recovery (see Bug log #11 in the project doc) for free.

Reads the SAME audio-event-augmented transcript segment list Stage 3/4 already build their
prompts from (`audio_events.augment_segments()`) — required, not optional: a citation is only
meaningful if it points at the exact list this call was shown, and that list must have the same
numbering Stage 3/4 use so `augment_segments()`'s own cache is reused rather than recomputed.

Output: `data/explainer/{movie}/{language}/intro.json` — {"narration", "cited_segments", "start",
"end"} after this stage; explainer_stage5_tts.py's `synthesize_intro_narration()` and
explainer_stage6_assemble.py's `retime_intro_clip()` each read this SAME file and add their own
fields to it (voice_duration/voice_path, then clip_path) rather than chaining through separate
files the way the numbered-beat pipeline does — there's only ever one intro object, so one
progressively-enriched file is simpler than four.

Auth: Application Default Credentials only. Do NOT set GEMINI_API_KEY.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from pipeline import audio_events
from pipeline.explainer_stage4_match import resolve_matches

INTRO_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "narration": {"type": "STRING"},
        "cited_segments": {
            "type": "ARRAY",
            "items": {"type": "INTEGER"},
        },
    },
    "required": ["narration", "cited_segments"],
}

INTRO_PROMPT_TEMPLATE = """You are writing a short CHANNEL INTRO line for a YouTube movie/series \
explainer (recap) video — the very first thing viewers hear, before the story itself begins. \
This is NOT a plot-teaser (a separate "hook" beat elsewhere in the script already does that) — \
it's a host briefly setting up what KIND of video this is, e.g. "Today, we're diving into a \
tense heist thriller where a crew of strangers gets pulled into one last job." Write in \
{language_name}.

Do NOT state the movie's exact title, and do NOT name specific characters the audience hasn't \
met yet — describe the premise, genre, and tone in a way that hooks interest without needing to \
know the title.

Keep it to one or two sentences of natural spoken narration, roughly {target_seconds} seconds \
when read aloud (around {target_words} words) — a rough target, not a hard limit.

For context, here is the beat-by-beat story summary already written for this same video (do not \
repeat these lines verbatim, this is just so you understand what the story is about):
{summary_block}

Then, from the FULL numbered transcript below, cite which transcript line(s) show a striking, \
illustrative moment to play as footage under this intro narration — something that gives a \
taste of the film's tone without spoiling specifics. Use an empty array only if truly nothing \
in the transcript fits.

Full transcript ({n_lines} numbered lines):
{transcript_block}
"""

LINE_TEMPLATE = "[{index}] {start:.1f}s-{end:.1f}s: {text}"


def _build_prompt(summary_beats: list, segments: list, language: str) -> str:
    summary_block = "\n".join(f"- {b['narration']}" for b in summary_beats)
    transcript_block = "\n".join(
        LINE_TEMPLATE.format(index=i, start=s["start"], end=s["end"], text=s["text"])
        for i, s in enumerate(segments)
    )
    return INTRO_PROMPT_TEMPLATE.format(
        language_name=settings.EXPLAINER_LANGUAGE_NAMES[language],
        target_seconds=settings.INTRO_TARGET_SECONDS,
        target_words=settings.INTRO_TARGET_WORDS,
        summary_block=summary_block,
        n_lines=len(segments), transcript_block=transcript_block,
    )


def _call_with_retry(client, prompt: str, schema):
    from google.genai import types

    last_error = None
    for attempt in range(settings.INTRO_MAX_RETRIES + 1):
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
            if attempt < settings.INTRO_MAX_RETRIES:
                backoff = settings.INTRO_RETRY_BACKOFF_BASE * (2 ** attempt)
                reason = "rate limited" if is_rate_limit else "error"
                print(f"  [retry] intro generation {reason}, attempt {attempt + 1}/"
                      f"{settings.INTRO_MAX_RETRIES}, waiting {backoff}s: {e}")
                time.sleep(backoff)
    raise last_error


def resolve_intro_citation(narration: str, segments: list, cited_segments: list) -> dict:
    """Resolve a raw {narration, cited_segments} result into {narration, cited_segments, start,
    end} by reusing explainer_stage4_match.resolve_matches() directly — wraps the single intro
    object as a one-item "beat" list so the exact same real-timestamp-lookup and out-of-range/
    timestamp-cited-instead-of-index recovery logic (see Bug log #11) applies here too, rather
    than a second, drift-prone reimplementation. Pure function — no file I/O — directly
    unit-testable."""
    intro_beat = [{"index": 0, "narration": narration}]
    matches_by_beat = {0: cited_segments}
    resolved = resolve_matches(intro_beat, segments, matches_by_beat)[0]
    return {
        "narration": resolved["narration"],
        "cited_segments": resolved["cited_segments"],
        "start": resolved["start"],
        "end": resolved["end"],
    }


def generate_intro(video_stem: str, language: str = "en") -> Path:
    from google import genai

    if language not in settings.EXPLAINER_LANGUAGE_NAMES:
        raise RuntimeError(f"Unsupported language {language!r} — must be one of "
                            f"{list(settings.EXPLAINER_LANGUAGE_NAMES)}.")

    summary_path = settings.EXPLAINER_DIR / video_stem / language / "summary.json"
    summary_beats = json.loads(summary_path.read_text(encoding="utf-8"))["beats"]
    if not summary_beats:
        raise RuntimeError("No beats in the summary — nothing to base an intro on.")

    transcript_path = settings.TRANSCRIPT_DIR / (
        f"{video_stem}.json" if language == "en" else f"{video_stem}.{language}.json"
    )
    segments = json.loads(transcript_path.read_text(encoding="utf-8"))["segments"]
    if not segments:
        raise RuntimeError("Transcript has no segments — nothing to cite footage from.")

    # Shared, language-independent (see pipeline/audio_events.py) — no language param here. Same
    # augmented list Stage 3/4 already numbered their prompts from, so a cited index here means
    # the same thing it would mean in those stages' own citations.
    segments = audio_events.augment_segments(video_stem, segments)

    if not settings.GCP_PROJECT_ID:
        raise RuntimeError(
            "GCP_PROJECT_ID is not set. Set it in config/settings.py or as an env var "
            "before running intro generation (this stage bills your GCP credit)."
        )

    print(f"[explainer-intro] language={language!r}, generating a ~{settings.INTRO_TARGET_SECONDS}s "
          f"intro from {len(summary_beats)} summary beats / {len(segments)} transcript lines -> "
          f"one Gemini call ({settings.GEMINI_MODEL}, Vertex ADC, project={settings.GCP_PROJECT_ID}, "
          f"schema-constrained)")

    client = genai.Client(vertexai=True, project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)
    prompt = _build_prompt(summary_beats, segments, language)
    result = _call_with_retry(client, prompt, INTRO_SCHEMA)

    intro = resolve_intro_citation(result["narration"], segments, result.get("cited_segments", []))

    out_dir = settings.EXPLAINER_DIR / video_stem / language
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "intro.json"
    out_path.write_text(json.dumps(intro, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[explainer-intro] intro written -> {out_path}: {intro['narration']!r}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python explainer_intro.py <video_stem> [language]")
        sys.exit(1)
    generate_intro(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else "en")
