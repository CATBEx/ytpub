"""
Explainer Stage 3 — Summary generation
PAID — one Gemini call, SCHEMA-CONSTRAINED structured output (response_schema
on GenerateContentConfig, not prompt-requested JSON with fence-stripping).
Reads the FULL transcript and writes a beat-by-beat narration summary, in
story order.

CHANGED 2026-08-27 (multi-language redesign): this now takes a `language`
param and writes the narration NATIVELY in that language — Gemini is told
which language to write in, and reads that language's OWN transcript (a
real Hindi/Bengali subtitle, or English Whisper/subtitle output). This is a
deliberate choice over generating once in English and machine-translating:
it lets Gemini reason about pacing/emphasis/idiom directly in the target
language, at the real cost that two languages of the same movie are
independently-generated recaps — they can pick different moments as the
hook/twist/highlight, since each is its own from-scratch reasoning pass over
that language's transcript. Every artifact this stage reads or writes is now
per-language (see generate_summary()).

By explicit design (2026-08-26 redesign): this call does NOT cite transcript
line numbers or resolve timestamps — that's Stage 4's job, as a separate
call. This call can focus purely on writing a good narration.

CHANGED same day (later): each beat also gets a "scene_type" field,
constrained via the schema's STRING `enum` to settings.EXPLAINER_SCENE_TYPES'
keys (action/mystery/horror/emotional/comedy/plot) — both a label AND a
coverage requirement (the prompt tells Gemini not to skip a genuine action/
mystery/horror moment just because the story is plot-heavy overall).

CHANGED same day (later still, the "hook/dopamine" redesign): each beat also
gets a "beat_role" field (settings.EXPLAINER_BEAT_ROLES —
hook/twist/highlight/setup/resolution/body), classifying its narrative JOB
in the recap rather than what it depicts. Exactly one beat must be "hook" —
a flash-forward teaser of the story's biggest moment, shown before context —
and `_reindex_beats()` below forces it to index 0 regardless of whatever
index Gemini itself assigned it, demoting extra hooks to "highlight" and
tolerating zero hooks rather than fabricating one (same defensive spirit as
bug #7's 0-indexing fix). twist/highlight beats get their own coverage
requirement and role-aware writing-style guidance in the prompt, since
Chirp3-HD (Stage 5) reads every beat in the same flat delivery — the words
have to carry the suspense/punch that a voice actor's delivery normally
would.

CHANGED same day (later still, the "silent action/horror" fix): the transcript
segments fed into the prompt are no longer read directly from Stage 2's
Whisper output — they're passed through `pipeline/audio_events.py`'s
`augment_segments()` first, which merges in synthetic placeholder entries for
dialogue-free spans of the movie with sustained high audio energy (a wordless
fight, chase, or dread-filled moment Whisper's VAD filter would otherwise drop
entirely, making it invisible to this call). Stage 4 calls the exact same
function so both stages see identical line numbering.

The schema below is handed to Gemini as a plain dict (Gemini's own native
schema format: OBJECT/ARRAY/STRING/INTEGER), NOT derived from a nested
Pydantic model. A first real-world run (2026-08-26) hit a live bug: nested
Pydantic classes (`beats: list[SummaryBeat]`) make `model_json_schema()`
emit `$ref`/`$defs`, and google-genai's dereferencing of those turned out to
be sensitive to the exact pydantic version resolved at runtime — it worked
in one environment and failed with `Extra inputs are not permitted
[$ref/$defs]` in another, on the identical google-genai version. A plain
dict schema has no `$ref` to dereference in the first place, so this whole
class of bug can't happen regardless of what pydantic version is active.
`.parsed` comes back as a plain dict (`json.loads`) rather than a validated
model instance — fine here, since we just read keys off it below.

Auth: Application Default Credentials only. Do NOT set GEMINI_API_KEY.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings
from pipeline import audio_events

SUMMARY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "beats": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "index": {"type": "INTEGER"},
                    "narration": {"type": "STRING"},
                    "scene_type": {
                        "type": "STRING",
                        "enum": list(settings.EXPLAINER_SCENE_TYPES.keys()),
                    },
                    "beat_role": {
                        "type": "STRING",
                        "enum": list(settings.EXPLAINER_BEAT_ROLES.keys()),
                    },
                },
                "required": ["index", "narration", "scene_type", "beat_role"],
            },
        },
    },
    "required": ["beats"],
}

SUMMARY_PROMPT_TEMPLATE = """You are writing the narration script for a YouTube movie/series \
explainer (recap) video. Below is the FULL dialogue transcript of the movie/episode, as \
numbered lines with timestamps. Read the whole thing, understand the plot, then write a \
beat-by-beat narration summary that explains the story in order, the way a recap channel \
would — clear, engaging, spoiler-accurate, no fluff.

Write every "narration" string in {language_name} — natural, fluent {language_name} a native \
speaker would actually say out loud, not a stiff or literal translation. The transcript lines \
below may themselves be in a different language; read and understand them in whatever language \
they're in, but ALWAYS write the narration itself in {language_name}.

Keep each beat to 1-3 sentences of spoken narration — natural, like something a narrator would \
say out loud, not a dry plot-point bullet. Do NOT reference transcript line numbers or \
timestamps in the narration text itself — just tell the story.

Number the beats' "index" field starting at 0 (the first beat is 0, the second is 1, and so on) \
— zero-based, not one-based.

Classify each beat's "scene_type" as exactly ONE of the following (use the definition, not just \
the label, to decide):
{scene_type_block}

Coverage requirement (scene_type): if the movie/episode genuinely contains action, mystery, or \
horror moments, make sure at least one beat captures each such moment that's actually present in \
the transcript — don't summarize only plot/dialogue and let a real action, mystery, or horror \
scene go unmentioned just because the story overall leans plot-heavy.

Also classify each beat's narrative "beat_role" as exactly ONE of the following:
{beat_role_block}

Beat-role rules:
- EXACTLY ONE beat must be "hook", and it MUST be index 0 — a flash-forward teaser of the \
single most shocking, exciting, or irresistible moment in the ENTIRE story, shown BEFORE any \
context. Withhold the "why" and the setup, but stay concrete about WHAT happens — specific \
enough to still be groundable in real dialogue, not vague teasing that names nothing real. Do \
NOT use the hook beat to describe the actual opening of the story — that's what index 1 is for.
- After the hook, tell the story in true chronological order, starting from the real beginning.
- Tag a beat "twist" only for a genuine reveal, reversal, or betrayal that recontextualizes what \
came before — write it to build suspense and hold the payoff back a beat longer where you can.
- Tag a beat "highlight" for the story's other most memorable moments (biggest action, funniest \
line, hardest emotional gut-punch) that aren't a twist — write these with extra punch and energy.
- Use "setup" for beats that build toward an upcoming highlight/twist, "resolution" for \
wrap-up/aftermath beats (often near the end), and "body" for everything else — most beats will \
be "body".

Coverage requirement (beat_role and scene_type together): classify honestly based on what's \
actually in the transcript. Don't invent a scene type, a twist, or a highlight that isn't really \
there — but don't default everything to "plot"/"body" either if the story genuinely has a real \
action, mystery, horror, twist, or standout moment; make sure it gets tagged and gets its own beat.

Some numbered lines below have no real dialogue — text reading exactly "{audio_event_marker}" \
marks a real span of the movie where the audio track had no spoken lines but genuinely sustained \
high energy, detected directly from the audio, not invented. Treat these as real, citable \
moments: if a span like this plausibly fits the surrounding story as a wordless action or horror \
beat, write a narration beat for it and cite it — don't skip it just because there's no dialogue \
to quote. Use judgment from context about what such a moment actually is (a wordless fight or \
chase reads differently than, say, a musical number) — don't assume every one of these is action \
or horror if the surrounding story suggests otherwise.

Full transcript ({n_lines} numbered lines):

{transcript_block}
"""

LINE_TEMPLATE = "[{index}] {start:.1f}s-{end:.1f}s: {text}"


def _build_prompt(segments: list, language: str) -> str:
    lines = [
        LINE_TEMPLATE.format(index=i, start=s["start"], end=s["end"], text=s["text"])
        for i, s in enumerate(segments)
    ]
    scene_type_block = "\n".join(
        f"- {name}: {desc}" for name, desc in settings.EXPLAINER_SCENE_TYPES.items()
    )
    beat_role_block = "\n".join(
        f"- {name}: {desc}" for name, desc in settings.EXPLAINER_BEAT_ROLES.items()
    )
    language_name = settings.EXPLAINER_LANGUAGE_NAMES[language]
    return SUMMARY_PROMPT_TEMPLATE.format(
        n_lines=len(segments), transcript_block="\n".join(lines),
        scene_type_block=scene_type_block, beat_role_block=beat_role_block,
        audio_event_marker=audio_events.AUDIO_EVENT_PLACEHOLDER_TEXT,
        language_name=language_name,
    )


def _call_with_retry(client, prompt: str, schema):
    from google.genai import types

    last_error = None
    for attempt in range(settings.SUMMARY_MAX_RETRIES + 1):
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
            if attempt < settings.SUMMARY_MAX_RETRIES:
                backoff = settings.SUMMARY_RETRY_BACKOFF_BASE * (2 ** attempt)
                reason = "rate limited" if is_rate_limit else "error"
                print(f"  [retry] summary generation {reason}, attempt {attempt + 1}/"
                      f"{settings.SUMMARY_MAX_RETRIES}, waiting {backoff}s: {e}")
                time.sleep(backoff)
    raise last_error


def _reindex_beats(raw_beats: list) -> list:
    """Turn Gemini's raw beat list into the final ordered, 0-indexed beat list.

    Two responsibilities, both defensive against the model not doing exactly
    what the prompt asked — same "don't trust the model's own count" lesson
    as bug #7 (Gemini 1-indexing beats despite the prompt saying 0-based):
      1. Re-derive a clean 0..N-1 index sequence from whatever indices Gemini
         actually returned, preserving the story order those indices imply.
      2. Enforce exactly one "hook" beat, forced to the front regardless of
         its own index value — the hook's narrative POSITION (always first)
         matters more than any literal index Gemini assigned it. If Gemini
         tagged zero or more than one beat "hook", handle it gracefully
         rather than erroring: keep the first (by its own raw index) and
         demote any extras to "highlight" (still a valid standout-moment
         role) rather than fabricating or discarding content.
    """
    enriched = [
        {
            "narration": b["narration"],
            "scene_type": b.get("scene_type") or "plot",
            "beat_role": b.get("beat_role") or "body",
            "_gemini_index": b["index"],
        }
        for b in raw_beats
    ]

    hooks = [b for b in enriched if b["beat_role"] == "hook"]
    if len(hooks) > 1:
        print(f"  [warn] Gemini tagged {len(hooks)} beats as 'hook' — keeping the first "
              f"(by its own index), demoting the rest to 'highlight'")
        keep = min(hooks, key=lambda b: b["_gemini_index"])
        for h in hooks:
            if h is not keep:
                h["beat_role"] = "highlight"
        hooks = [keep]
    elif not hooks:
        print("  [warn] Gemini did not tag any beat as 'hook' — proceeding without one")

    non_hooks = sorted([b for b in enriched if b["beat_role"] != "hook"], key=lambda b: b["_gemini_index"])
    ordered = hooks + non_hooks  # hook (if any) always first; story body in chronological order after

    return [
        {"index": i, "narration": b["narration"], "scene_type": b["scene_type"], "beat_role": b["beat_role"]}
        for i, b in enumerate(ordered)
    ]


def generate_summary(video_stem: str, language: str = "en") -> Path:
    from google import genai

    if language not in settings.EXPLAINER_LANGUAGE_NAMES:
        raise RuntimeError(f"Unsupported language {language!r} — must be one of "
                            f"{list(settings.EXPLAINER_LANGUAGE_NAMES)}.")

    # Per-language transcript path — matches pipeline/subtitles.py's load_or_transcribe()
    # convention: bare {stem}.json for English (backward compat), {stem}.{language}.json
    # for hi/bn (each language's own real transcript, not a shared/relabeled one).
    transcript_path = settings.TRANSCRIPT_DIR / (
        f"{video_stem}.json" if language == "en" else f"{video_stem}.{language}.json"
    )
    segments = json.loads(transcript_path.read_text(encoding="utf-8"))["segments"]
    if not segments:
        raise RuntimeError("Transcript has no segments — nothing to summarize.")

    # Audio-event detection is language-independent (real audio energy, not narration) and
    # SHARED across languages — see pipeline/audio_events.py's module docstring. No language
    # param here on purpose.
    segments = audio_events.augment_segments(video_stem, segments)

    if not settings.GCP_PROJECT_ID:
        raise RuntimeError(
            "GCP_PROJECT_ID is not set. Set it in config/settings.py or as an env var "
            "before running summary generation (this stage bills your GCP credit)."
        )

    approx_chars = sum(len(s["text"]) for s in segments)
    print(f"[explainer-summary] language={language!r}, {len(segments)} transcript lines, "
          f"~{approx_chars} chars of dialogue -> one Gemini call ({settings.GEMINI_MODEL}, "
          f"Vertex ADC, project={settings.GCP_PROJECT_ID}, schema-constrained)")
    if approx_chars > 150_000:
        print("  [warn] this is a very long transcript — single-call summary generation is "
              "untested at this scale. Watch for a truncated/incomplete response; splitting "
              "into acts and stitching is the fallback if this fails, not yet built.")

    client = genai.Client(vertexai=True, project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)
    prompt = _build_prompt(segments, language)
    result = _call_with_retry(client, prompt, SUMMARY_SCHEMA)
    beats = _reindex_beats(result["beats"])

    # Per-language output: data/explainer/{movie}/{language}/summary.json — SIBLING to the
    # movie-root audio_events.json, not sharing a folder with any other language's files.
    out_dir = settings.EXPLAINER_DIR / video_stem / language
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps({"beats": beats}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[explainer-summary] {len(beats)} beats written -> {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python explainer_stage3_summary.py <video_stem> [language]")
        sys.exit(1)
    generate_summary(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else "en")
