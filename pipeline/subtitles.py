"""
Explainer Stage 2 alternative — real subtitle (.srt) file support (2026-08-27).

If a `.srt` file exists next to the source video (same stem, `.srt`
extension), it's parsed directly into the same segment shape faster-whisper's
`transcribe()` produces (`{"start": float, "end": float, "text": str}`) and
written to the SAME `data/transcripts/{stem}.json` file Stage 2 writes — so
every downstream consumer (explainer_stage3_summary.py's transcript read,
explainer_stage4_match.py's, `audio_events.py`'s dialogue-free-gap detection,
highlight-reel mode's own reuse of the transcript file) needs ZERO changes.
They don't know or care whether the file came from Whisper or a real
subtitle — it's the same shape either way.

Why this is worth having, from real discussion + a real run this session:
- Real subtitles are more accurate than Whisper's ASR guesses — no
  mishearing/hallucination (see bug #10: a real ~9-minute Whisper segment
  attributed a whole silent stretch to one short, wrong line of dialogue).
- Skips the single biggest time cost in either pipeline. A real Boba Fett
  run showed Stage 2 (transcription) taking ~26 of the run's ~31 minutes
  (12:38:37 -> 13:04:19). A real 18-beat run showed the same pattern.
- SDH ("Subtitles for the Deaf and Hard-of-hearing") files additionally
  include bracketed non-dialogue sound cues (`[Growling]`, `[Music]`,
  `[Gunshot]`) — strictly better ground truth than `audio_events.py`'s
  sustained-energy heuristic for describing WHAT a wordless moment actually
  is, not just THAT something loud happened. These cues need no special
  handling anywhere downstream: they just become ordinary citable transcript
  lines, and Stage 3's prompt already tells Gemini to use context/judgment on
  unusual bracketed lines (it was written for the audio-event placeholder
  marker, but the instruction generalizes fine to a real SDH cue too).

Sourcing (explicit user choice, 2026-08-27): the user supplies the `.srt`
themselves — this module does NOT fetch/download subtitles from anywhere.
Whisper stays the automatic fallback (explicit user choice, same discussion)
when no matching `.srt` is found next to the source video — nothing breaks
for a movie you haven't sourced subtitles for yet. `load_or_transcribe()` is
a drop-in replacement for calling `stage2_transcribe.transcribe()` directly,
so callers (main.py, api/jobs.py) don't need their own branching logic.

Known limitation, not handled: ASS-style `{\an8}`-type position/style tags
occasionally found in low-quality rips are NOT stripped (only HTML-ish
`<i>`/`<b>`/`<font ...>` tags are) — if one shows up in a real file, it'll
reach Gemini as literal text. Not fixed proactively since it wasn't in any
real sample seen this session; fix if/when a real file actually has one.

--- Subtitle generate/translate utilities (2026-09-05) ---
Added after a real user hit exactly the gap these solve: uploaded a movie in
Hindi mode with no Hindi subtitle available anywhere online. Two new,
independent pieces, both designed to produce a REAL .srt on disk using the
SAME naming convention as everything above (`{stem}.{language}.srt`) — so
find_subtitle_file()/load_or_transcribe() pick up their output automatically,
with zero special-casing anywhere downstream:

- detect_language() / generate_subtitle() — an explicit, user-confirmed
  alternative to the silent Whisper fallback above. English mode already
  ran Whisper automatically when no .srt was found; Hindi/Bengali never do
  (and load_or_transcribe() below still never will, silently) — but the user
  can now explicitly ask "generate one from the movie's own audio" for ANY
  language, and get a real, reusable .srt out of it instead of a hard
  refusal every time no subtitle exists.

- translate_subtitle() — the "translate-first" architecture chosen after
  discussion (2026-09-05): rather than teaching Gemini to write narration
  directly from a different-language transcript (which would force Stage
  4's already-fragile citation-matching, see the project doc's Bug log #11,
  into a harder cross-language semantic match, never tested that way), this
  translates an existing subtitle cue-by-cue into a new language, carrying
  the ORIGINAL timestamps over unchanged. The output is then just an
  ordinary same-language subtitle as far as Stage 3/4/5/6 are concerned —
  none of that code needs to change at all.
"""
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings

_TIMESTAMP_RE = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
)
_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")  # strips <i>, </i>, <b>, <font color="...">, etc.


def subtitle_dir_for(video_path: Path) -> Path:
    """CHANGED 2026-09-05 (folder-structure reorg, user's own proposal): every movie gets its
    own folder under data/Subtitle/, named after the movie's filename stem — e.g.
    data/Subtitle/Movie Name (2026) 1080p.../ — holding that movie's audio.mp3 plus all 3
    languages' subtitles together, instead of scattering `{stem}.{language}.srt` files loose
    next to the movie in data/input/ (which got cluttered fast once you had more than a
    couple of movies). Creates the folder if it doesn't exist yet."""
    d = settings.SUBTITLE_DIR / video_path.stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def subtitle_path_for(video_path: Path, language: str) -> Path:
    """Where a given language's subtitle for this movie lives (or will be written) under the
    new per-movie folder structure — e.g. data/Subtitle/Movie Name/hi_subtitle.srt. Used by
    both this module (generate_subtitle()/translate_subtitle() write here) and
    api/server.py (a manually-attached .srt at /submit is saved here too, so every subtitle
    ends up in the same place regardless of how it was produced)."""
    return subtitle_dir_for(video_path) / f"{language}_subtitle.srt"


def find_subtitle_file(video_path: Path, language: str = "en"):
    """A subtitle file for a specific LANGUAGE. CHANGED 2026-09-05 (folder-structure reorg):
    now looks in the new per-movie folder (subtitle_path_for()) first — e.g.
    data/Subtitle/Movie Name/hi_subtitle.srt — rather than a `{stem}.{language}.srt` file
    sitting loose next to the movie (that in-between convention, 2026-08-27 through
    2026-09-05, is migrated into the new structure by migrate_legacy_subtitles() at server
    startup, so this function doesn't need to know about it at all).

    For English ONLY, still falls back to the much older bare `{stem}.srt` convention next to
    the video (pre-multi-language, and still what highlight-reel mode expects/writes on its
    own) — that's a SEPARATE, still-live convention this reorg deliberately leaves alone.
    Hindi/Bengali have no bare-file fallback — see load_or_transcribe()'s comment on why
    Whisper can't substitute for a missing translated subtitle either, so there's nothing
    sensible to fall back to.

    Returns None if nothing is found."""
    structured = subtitle_path_for(video_path, language)
    if structured.exists():
        return structured
    if language == "en":
        bare = video_path.with_suffix(".srt")
        if bare.exists():
            return bare
    return None


def migrate_legacy_subtitles(input_dir: Path = None) -> list:
    """One-time, idempotent migration (2026-09-05 folder-structure reorg): moves subtitle
    files written under the brief in-between `{stem}.{language}.srt`-next-to-the-video
    convention (2026-08-27 through 2026-09-05) into the new data/Subtitle/{movie}/
    {language}_subtitle.srt structure. Safe to call on every server startup — a no-op once
    nothing old-style is left. Does NOT touch the much older bare `{stem}.srt` fallback (a
    separate, still-live convention highlight-reel mode expects — see find_subtitle_file()'s
    docstring); only the per-language files THIS feature itself created get moved.

    Returns a list of (old_path, new_path) string pairs actually moved, for logging."""
    input_dir = input_dir or settings.INPUT_DIR
    moved = []
    for language in settings.EXPLAINER_LANGUAGES:
        suffix = f".{language}.srt"
        for old_path in input_dir.glob(f"*{suffix}"):
            video_stem = old_path.name[: -len(suffix)]
            new_path = settings.SUBTITLE_DIR / video_stem / f"{language}_subtitle.srt"
            new_path.parent.mkdir(parents=True, exist_ok=True)
            if new_path.exists():
                # Already migrated (or separately generated under the new structure) --
                # the new one wins, just clear out the stale duplicate rather than overwrite it.
                old_path.unlink()
                continue
            old_path.rename(new_path)
            moved.append((str(old_path), str(new_path)))
    if moved:
        print(f"[subtitles] migrated {len(moved)} subtitle(s) to the new data/Subtitle/ folder structure")
    return moved


def _timestamp_to_seconds(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000


def parse_srt(srt_path: Path) -> list:
    """Parse a .srt file (regular OR SDH — both use the same file format;
    SDH just has bracketed sound-cue text like other cues have dialogue)
    into the same segment shape Whisper's transcribe() produces: a list of
    {"start": float, "end": float, "text": str} dicts, sorted by start time.

    Pure function — no other I/O — so it's directly unit-testable. Handles
    real-world quirks seen in actual downloaded files this session:
    - Multi-line cue text (joined into one line with a space).
    - HTML-ish formatting tags (<i>, </i>, <b>, <font ...> etc.) — stripped
      entirely, since Gemini has no use for subtitle markup.
    - A missing/non-numeric cue-index line (some tools omit it) — detected
      by scanning each line in a block for the timestamp pattern rather than
      assuming a fixed line position.
    - Cues with empty/whitespace-only (or markup-only) text — skipped.
    - A single cue spanning many seconds (a real SDH `[Music]` cue can
      legitimately run 10-20+ seconds) — no length assumption is made.
    - Malformed blocks with no timestamp at all — skipped rather than
      raising, so one bad block in a large file doesn't kill the whole parse.
    """
    raw = srt_path.read_text(encoding="utf-8-sig")  # -sig strips a BOM if present
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    blocks = re.split(r"\n{2,}", raw.strip())

    segments = []
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip() != ""]
        if not lines:
            continue

        ts_line_idx = next((i for i, ln in enumerate(lines) if _TIMESTAMP_RE.search(ln)), None)
        if ts_line_idx is None:
            continue  # no timestamp anywhere in this block -- malformed, skip rather than crash

        match = _TIMESTAMP_RE.search(lines[ts_line_idx])
        g = match.groups()
        start = _timestamp_to_seconds(*g[0:4])
        end = _timestamp_to_seconds(*g[4:8])

        text = " ".join(lines[ts_line_idx + 1:]).strip()
        text = _TAG_RE.sub("", text).strip()
        if not text:
            continue  # cue had no real text (formatting-only or blank)

        segments.append({"start": round(start, 2), "end": round(end, 2), "text": text})

    segments.sort(key=lambda s: s["start"])
    return segments


def load_or_transcribe(video_path: Path, audio_path: Path, language: str = "en") -> Path:
    """Drop-in replacement for calling stage2_transcribe.transcribe(audio_path)
    directly, now per-LANGUAGE (2026-08-27 multi-language redesign). If a
    matching `.{language}.srt` (or, for English only, a bare `.srt`) exists
    next to video_path, parses it and writes it to that language's own
    transcript path, WITHOUT running Whisper at all.

    Hindi and Bengali have NO Whisper fallback and never will: Whisper
    transcribes whatever language is actually SPOKEN in the source audio
    track, it does not translate, so running it against (most likely)
    English/original-language audio and labeling the result "Hindi" would
    silently produce a wrong-language transcript instead of failing loudly.
    settings.EXPLAINER_SUBTITLE_REQUIRED enforces this: for those languages,
    a missing or unparseable subtitle raises rather than falling back.

    English keeps the original (pre-multi-language) behavior: Whisper runs
    automatically when no .srt is found, exactly as before this feature
    existed. Either way, returns the transcript path."""
    from pipeline import stage1_ingest, stage2_transcribe

    # English keeps writing to the original bare path (no language suffix) so it stays a
    # drop-in match for whatever stage2_transcribe.transcribe()/highlight-reel mode already
    # produce there; hi/bn get their own language-suffixed transcript file, since their text
    # is genuinely different content, not just a relabeling of the English one.
    out_path = settings.TRANSCRIPT_DIR / (
        f"{video_path.stem}.json" if language == "en" else f"{video_path.stem}.{language}.json"
    )

    srt_path = find_subtitle_file(video_path, language)
    if srt_path is None:
        if settings.EXPLAINER_SUBTITLE_REQUIRED.get(language, False):
            raise RuntimeError(
                f"No subtitle found for language={language!r} — expected "
                f"{subtitle_path_for(video_path, language)}. "
                f"Whisper cannot substitute for a missing {language} subtitle (it transcribes "
                f"whatever language is spoken in the audio, not the requested output language). "
                f"Supply a real {language} subtitle file and resubmit."
            )
        print(f"[subtitles] no .srt found for language={language!r} -> falling back to Whisper")
        return stage2_transcribe.transcribe(audio_path)

    print(f"[subtitles] found {srt_path.name} -> parsing it instead of running Whisper (Stage 2 skipped)")
    segments = parse_srt(srt_path)
    if not segments:
        if settings.EXPLAINER_SUBTITLE_REQUIRED.get(language, False):
            raise RuntimeError(
                f"{srt_path.name} parsed to zero usable segments, and language={language!r} "
                f"requires a real subtitle (no Whisper fallback available for this language)."
            )
        print(f"  [warn] {srt_path.name} parsed to zero usable segments -- falling back to Whisper")
        return stage2_transcribe.transcribe(audio_path)

    try:
        duration = stage1_ingest.get_duration_seconds(video_path)
    except Exception:
        duration = segments[-1]["end"]  # best-effort fallback if ffprobe fails for some reason

    transcript = {
        "language": language,   # now known exactly -- this is the language the caller asked
                                  # for and the subtitle we found for it, not a guess
        "duration": duration,
        "segments": segments,
        "source": "srt",  # distinguishes from Whisper output when inspecting the saved file
    }
    out_path.write_text(json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[subtitles] parsed {len(segments)} cues from {srt_path.name} -> {out_path}")
    return out_path


def ensure_movie_audio(video_path: Path) -> Path:
    """New (2026-09-05, folder-structure reorg): the movie's own audio, extracted ONCE to
    data/Subtitle/{movie}/audio.mp3 and reused from then on — by detect_language() (which
    samples a short clip out of it) AND generate_subtitle() (which transcribes it directly),
    so a movie's video container only ever gets audio-decoded once per movie, not once per
    Subtitle Library operation. Whichever of the two runs first creates it; the other finds
    it already there and skips straight to using it. NOT the same file as
    stage1_ingest.extract_audio()'s own data/audio/{stem}.wav — that's a separate extraction
    for the main pipeline's own Stage 1/2, untouched by this."""
    from pipeline import stage1_ingest

    audio_path = subtitle_dir_for(video_path) / "audio.mp3"
    if audio_path.exists():
        return audio_path
    print(f"[subtitles] extracting audio for {video_path.name} -> {audio_path} (first use, cached from here on)")
    return stage1_ingest.extract_audio_mp3(video_path, audio_path)


def detect_language(media_path: Path) -> tuple:
    """Quick language-ID pass ONLY — does not transcribe. faster-whisper
    decodes actual segments from roughly the first 30s of audio before it
    ever starts decoding the rest (the returned `segments` is a lazy
    generator), so not iterating that generator costs only the short
    language-ID DECODING pass, not a full transcription.

    FIXED 2026-09-05 (real-machine crash): that's only true of the decoding
    step — feeding model.transcribe() a full-length media file still makes
    faster-whisper's feature extractor compute a mel-spectrogram array for
    the ENTIRE track up front, regardless of language detection needing
    only a fraction of it. On a real ~105-minute movie this needed a single
    ~484 MiB array allocation and crashed with
    numpy.core._exceptions._ArrayMemoryError on the user's machine (modest
    RAM — see config/settings.py's WHISPER_CPU_THREADS comment). Now
    extracts a genuinely short sample clip first (stage1_ingest.
    extract_audio_sample(), 45s starting at the 1-minute mark, skipping the
    silent/logo-only opening most movies have) and runs language-ID against
    just that — bounded memory/time regardless of the source movie's length,
    which is what this function's docstring always claimed to do. The sample
    is now cut from the cached ensure_movie_audio() file rather than the raw
    video, so this also benefits from that cache.

    Returns (language_code, probability) — language_code is whatever
    faster-whisper's underlying Whisper model uses (ISO 639-1-ish, e.g.
    "en"/"hi"/"bn"), probability is its own confidence, 0-1."""
    from faster_whisper import WhisperModel
    from pipeline import stage1_ingest

    if not media_path.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    print(f"[subtitles] detecting language for {media_path.name}...")
    audio_path = ensure_movie_audio(media_path)
    sample_path = stage1_ingest.extract_audio_sample(audio_path)
    try:
        model = WhisperModel(
            settings.WHISPER_MODEL_SIZE, device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE, cpu_threads=settings.WHISPER_CPU_THREADS,
        )
        _segments, info = model.transcribe(str(sample_path), beam_size=1)
        print(f"[subtitles] detected language={info.language!r} (probability={info.language_probability:.2f})")
        return info.language, round(info.language_probability, 3)
    finally:
        sample_path.unlink(missing_ok=True)  # throwaway clip -- not the cached movie audio itself


def _seconds_to_srt_timestamp(seconds: float) -> str:
    total_ms = round(seconds * 1000)
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _segments_to_srt(segments: list) -> str:
    """Pure function, the reverse of parse_srt() above — serializes our
    internal {"start", "end", "text"} segment shape back into real .srt file
    text (1-based cue numbers, HH:MM:SS,mmm timestamps). Used by both
    generate_subtitle() and translate_subtitle() below so their output is a
    real, ordinary .srt file — not a special internal format — and round-
    trips cleanly back through parse_srt() the next time it's read."""
    blocks = []
    for i, seg in enumerate(segments, start=1):
        blocks.append(
            f"{i}\n{_seconds_to_srt_timestamp(seg['start'])} --> {_seconds_to_srt_timestamp(seg['end'])}\n"
            f"{seg['text']}"
        )
    return "\n\n".join(blocks) + "\n"


def generate_subtitle(video_path: Path, language: str = None) -> tuple:
    """New (2026-09-05): generate a real, reusable .srt via Whisper when none
    exists — the explicit, user-confirmed alternative to load_or_transcribe()'s
    silent English-only fallback (see this module's docstring above). Works
    for ANY language, including Hindi/Bengali — load_or_transcribe() itself
    still never silently substitutes Whisper for those; this is a separate,
    deliberate action the user asks for.

    language=None auto-detects (same short pass as detect_language(), except
    this time the full transcription that follows naturally confirms/uses
    whatever it detects). language=<code> pins Whisper to that language for
    better accuracy/speed, same tradeoff as config/settings.py's
    WHISPER_LANGUAGE.

    Writes a REAL .srt file (not just an internal transcript.json) under this
    module's per-movie folder structure (subtitle_path_for(), CHANGED
    2026-09-05 folder reorg) — the load-bearing choice that makes reuse/
    caching free: find_subtitle_file()/load_or_transcribe() pick this file up
    next time exactly like any uploaded or translated subtitle, no special-
    casing needed anywhere else. Transcribes from ensure_movie_audio()'s
    cached audio.mp3 (CHANGED 2026-09-05) rather than the raw video file, so
    a movie already processed by detect_language() doesn't get its video
    container decoded a second time.

    Returns (srt_path, language_used, probability) — probability is None
    when `language` was pinned explicitly (nothing was actually detected)."""
    from faster_whisper import WhisperModel

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    audio_path = ensure_movie_audio(video_path)
    print(f"[subtitles] loading whisper model '{settings.WHISPER_MODEL_SIZE}' to generate a "
          f"subtitle for {video_path.name} (language={language or 'auto-detect'})...")
    model = WhisperModel(
        settings.WHISPER_MODEL_SIZE, device=settings.WHISPER_DEVICE,
        compute_type=settings.WHISPER_COMPUTE_TYPE, cpu_threads=settings.WHISPER_CPU_THREADS,
    )
    print(f"[subtitles] transcribing {video_path.name} (this can take a while on CPU)...")
    segments_iter, info = model.transcribe(str(audio_path), language=language, beam_size=5, vad_filter=True)

    used_language = language or info.language
    probability = None if language else round(info.language_probability, 3)
    if used_language not in settings.EXPLAINER_LANGUAGE_NAMES:
        print(f"  [warn] detected/used language {used_language!r} is not one of "
              f"{list(settings.EXPLAINER_LANGUAGE_NAMES)} — saving the .srt anyway under that "
              f"code, but Explainer mode only offers those three as an output language.")

    segments = []
    for seg in segments_iter:
        segments.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": seg.text.strip()})
        print(f"  [{seg.start:7.1f}s - {seg.end:7.1f}s] {seg.text.strip()}")
    if not segments:
        raise RuntimeError(f"Whisper produced zero segments for {video_path.name} — nothing to save.")

    out_path = subtitle_path_for(video_path, used_language)
    out_path.write_text(_segments_to_srt(segments), encoding="utf-8")
    print(f"[subtitles] generated {len(segments)} cues (language={used_language!r}) -> {out_path}")
    return out_path, used_language, probability


TRANSLATE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "lines": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "index": {"type": "INTEGER"},
                    "text": {"type": "STRING"},
                },
                "required": ["index", "text"],
            },
        },
    },
    "required": ["lines"],
}

TRANSLATE_PROMPT_TEMPLATE = """You are translating subtitle cues for a movie/show from {source_name} \
to {target_name}. Below is a numbered list of cue texts — timestamps are NOT shown and don't matter \
here; only translate the text, strictly one-to-one. Do not merge, split, add, reorder, or drop lines.

Translate each line naturally and fluently into {target_name}, the way a professional subtitler \
would — not a stiff, literal, word-for-word translation. Some lines may be bracketed non-dialogue \
sound cues (e.g. "[Music]", "[Gunshot]", "[Growling]") rather than spoken dialogue — translate the \
bracketed word/phrase into {target_name} too, keeping the brackets themselves in place.

Return EXACTLY one translated line per input line below, using the SAME "index" value as the input \
so the results can be matched back up — do not renumber, reorder, merge, or skip any line, even if a \
line is very short or seems unimportant.

Input lines ({n_lines} total):
{lines_block}
"""


def _translate_batch_with_retry(client, texts: list, source_name: str, target_name: str) -> dict:
    """One Gemini call translating a batch of cue texts, index-mapped in both
    directions so a reordered or partial response still resolves to the
    right cue — same defensive pattern as stage5_score.py's batched scoring
    (an explicit `index` field the response is matched back against, not
    positional array order). Returns {index: translated_text}, only for
    indices Gemini actually returned — missing ones are the caller's problem
    to fall back on."""
    from google.genai import types

    lines_block = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
    prompt = TRANSLATE_PROMPT_TEMPLATE.format(
        source_name=source_name, target_name=target_name, n_lines=len(texts), lines_block=lines_block,
    )
    last_error = None
    for attempt in range(settings.TRANSLATE_MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=settings.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=TRANSLATE_SCHEMA,
                ),
            )
            parsed = response.parsed
            if parsed is None:
                raise ValueError(f"Gemini returned no parsed structured output: {response.text!r}")
            return {line["index"]: line["text"] for line in parsed["lines"]}
        except Exception as e:
            last_error = e
            is_rate_limit = "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)
            if attempt < settings.TRANSLATE_MAX_RETRIES:
                backoff = settings.TRANSLATE_RETRY_BACKOFF_BASE * (2 ** attempt)
                reason = "rate limited" if is_rate_limit else "error"
                print(f"  [retry] translate batch {reason}, attempt {attempt + 1}/"
                      f"{settings.TRANSLATE_MAX_RETRIES}, waiting {backoff}s: {e}")
                time.sleep(backoff)
    raise last_error


def translate_subtitle(video_path: Path, source_language: str, target_language: str) -> Path:
    """New (2026-09-05): translate an existing subtitle from source_language
    into target_language, cue-by-cue, SAME timestamps — the "translate-first"
    piece of the architecture (see this module's docstring above). This is
    what turns "movie has an English subtitle, I want Hindi output" into a
    real Hindi subtitle without needing to find one online: translate the
    English one, then everything downstream (find_subtitle_file(),
    load_or_transcribe(), Stage 3/4/5/6) treats the result exactly like any
    uploaded or Whisper-generated same-language subtitle.

    Deliberately does NOT ask Gemini to touch timestamps at all — that
    sidesteps the exact class of bug the project's Bug log #11 hit
    (Gemini mis-citing a transcript TIMESTAMP instead of an INDEX): a
    translation is a strict 1:1 text remap over an already-real, already-
    timestamped cue list, so timestamps are just carried over unchanged, no
    resolution/matching involved at all."""
    from google import genai

    if source_language not in settings.EXPLAINER_LANGUAGE_NAMES:
        raise RuntimeError(f"Unsupported source language {source_language!r} — must be one of "
                            f"{list(settings.EXPLAINER_LANGUAGE_NAMES)}.")
    if target_language not in settings.EXPLAINER_LANGUAGE_NAMES:
        raise RuntimeError(f"Unsupported target language {target_language!r} — must be one of "
                            f"{list(settings.EXPLAINER_LANGUAGE_NAMES)}.")
    if source_language == target_language:
        raise RuntimeError("Source and target language are the same — nothing to translate.")

    src_path = find_subtitle_file(video_path, source_language)
    if src_path is None:
        raise RuntimeError(
            f"No {source_language!r} subtitle found for {video_path.name} to translate from "
            f"(expected {subtitle_path_for(video_path, source_language)})."
        )
    segments = parse_srt(src_path)
    if not segments:
        raise RuntimeError(f"{src_path.name} parsed to zero usable cues — nothing to translate.")

    if not settings.GCP_PROJECT_ID:
        raise RuntimeError("GCP_PROJECT_ID is not set — translation uses Gemini and needs it.")

    source_name = settings.EXPLAINER_LANGUAGE_NAMES[source_language]
    target_name = settings.EXPLAINER_LANGUAGE_NAMES[target_language]
    client = genai.Client(vertexai=True, project=settings.GCP_PROJECT_ID, location=settings.GCP_LOCATION)

    print(f"[subtitles] translating {len(segments)} cues from {source_name} -> {target_name} "
          f"(source: {src_path.name}, batches of {settings.TRANSLATE_BATCH_SIZE})")
    translated_texts = [None] * len(segments)
    for batch_start in range(0, len(segments), settings.TRANSLATE_BATCH_SIZE):
        batch = segments[batch_start:batch_start + settings.TRANSLATE_BATCH_SIZE]
        texts = [s["text"] for s in batch]
        by_index = _translate_batch_with_retry(client, texts, source_name, target_name)
        for local_i in range(len(batch)):
            global_i = batch_start + local_i
            if local_i in by_index:
                translated_texts[global_i] = by_index[local_i]
            else:
                print(f"  [warn] batch starting at cue {batch_start}: no translation returned for "
                      f"cue {global_i} — keeping original text untranslated")
                translated_texts[global_i] = segments[global_i]["text"]

    translated_segments = [
        {"start": seg["start"], "end": seg["end"], "text": translated_texts[i]}
        for i, seg in enumerate(segments)
    ]

    out_path = subtitle_path_for(video_path, target_language)
    out_path.write_text(_segments_to_srt(translated_segments), encoding="utf-8")
    print(f"[subtitles] wrote {len(translated_segments)} translated cues -> {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python subtitles.py <video_path> [language]")
        sys.exit(1)
    vp = Path(sys.argv[1])
    lang = sys.argv[2] if len(sys.argv) == 3 else "en"
    found = find_subtitle_file(vp, lang)
    if found is None:
        print(f"No .srt found for {vp} language={lang!r} (expected {vp.with_suffix(f'.{lang}.srt')})")
    else:
        print(json.dumps(parse_srt(found), indent=2, ensure_ascii=False))
