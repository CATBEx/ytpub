"""
Central config for the Movie Review Video pipeline.
Edit values here rather than hardcoding paths/thresholds inside stage scripts.
"""
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"          # drop source MP4s here
AUDIO_DIR = DATA_DIR / "audio"          # extracted .wav audio tracks
TRANSCRIPT_DIR = DATA_DIR / "transcripts"  # whisper transcripts (json)
ANALYSIS_DIR = DATA_DIR / "analysis"    # shot boundaries + signal + scores (json)
CLIPS_DIR = DATA_DIR / "clips"          # individual cut clips
OUTPUT_DIR = DATA_DIR / "output"        # final assembled rough-cut
SUBTITLE_DIR = DATA_DIR / "Subtitle"    # NEW 2026-09-05 (folder-structure reorg, user's own
                                          # proposal): one subfolder per movie (named after its
                                          # filename stem), holding that movie's audio.mp3 plus
                                          # en_subtitle.srt/hi_subtitle.srt/bn_subtitle.srt --
                                          # replaces the old {stem}.{language}.srt-loose-next-
                                          # to-the-movie convention, which got cluttered fast.
                                          # See pipeline/subtitles.py's subtitle_path_for()/
                                          # migrate_legacy_subtitles().

for d in (INPUT_DIR, AUDIO_DIR, TRANSCRIPT_DIR, ANALYSIS_DIR, CLIPS_DIR, OUTPUT_DIR, SUBTITLE_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Stage 2 — Transcription (faster-whisper, CPU, free)
# ---------------------------------------------------------------------------
WHISPER_MODEL_SIZE = "small"     # tiny / base / small / medium / large-v3
WHISPER_DEVICE = "cpu"           # keep CPU — local GPU (2GB) is too small to help here
WHISPER_COMPUTE_TYPE = "int8"    # fastest CPU option with acceptable accuracy
WHISPER_CPU_THREADS = 4          # matches the user's real core count (i5 8th gen, 4 cores).
                                  # Lets the ONE transcription job use all cores internally via
                                  # faster-whisper/CTranslate2's own intra-op threading — chosen
                                  # over a manual multi-worker/audio-slot split, which would just
                                  # oversubscribe 4 cores (2 workers x 2 threads ~= 1 job x 4
                                  # threads, but with real chunk-boundary/timestamp-merge risk
                                  # added for no real gain). Adjust if this ever runs on different
                                  # hardware.
WHISPER_LANGUAGE = None          # None = auto-detect. Set explicitly for better accuracy/speed:
                                  # "bn" for Bengali, "hi" for Hindi, "en" for English, etc.
                                  # (ISO 639-1 codes; auto-detect uses the first 30s of audio,
                                  # which is usually fine but an explicit hint is more reliable
                                  # and skips the detection step)

# ---------------------------------------------------------------------------
# Stage 3 — Shot detection (PySceneDetect, free)
# ---------------------------------------------------------------------------
SCENE_DETECT_THRESHOLD = 27.0    # ContentDetector default-ish; lower = more sensitive

# ---------------------------------------------------------------------------
# Stage 4/5 — Signal extraction (free)
# ---------------------------------------------------------------------------
SEGMENT_MIN_DURATION = 3.0       # seconds — ignore candidate segments shorter than this
SEGMENT_MAX_DURATION = 45.0      # seconds — cap a single candidate segment's length

# ---------------------------------------------------------------------------
# Stage 6 — Scoring (Gemini via Vertex ADC — PAID, billed to GCP credit)
# ---------------------------------------------------------------------------
# Do NOT use GEMINI_API_KEY. Auth is via Application Default Credentials:
#   gcloud auth application-default login
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "video-production-506709")
GCP_LOCATION = os.environ.get("GCP_LOCATION", "us-central1")
GEMINI_MODEL = "gemini-2.5-flash"   # GA, stable Flash tier — cheap, fast, good enough for scoring
                                     # (gemini-2.0-flash-001 was retired from Vertex; 3.x Flash
                                     # tiers exist but some are preview/restricted — try after
                                     # this is confirmed working if you want to compare quality)

SCORE_BATCH_SIZE = 30            # segments scored per API call (was 1 — this is the big fix for
                                  # both the 38-minute Stage 6 runtime and the 429 rate-limit hits)
SCORE_MAX_RETRIES = 4            # retry a failed/rate-limited batch this many times
SCORE_RETRY_BACKOFF_BASE = 8     # seconds — exponential: 8s, 16s, 32s, 64s between retries

# Each segment now gets 5 sub-scores (0-100) instead of one holistic number, so curation in
# Stage 7 can pick a genuine mix instead of just "whatever scored highest overall":
#   plot_score         — plot-critical / narrative importance
#   emotion_score       — emotional weight (drama, stakes, relationships)
#   humor_score         — comedic value
#   action_score         — visual/physical intensity
#   quotability_score  — striking, poetic, or otherwise quotable phrasing — a line can score
#                         high here even if it's low-stakes for the plot
SCORE_CATEGORIES = ["plot_score", "emotion_score", "humor_score", "action_score", "quotability_score"]

# ---------------------------------------------------------------------------
# Stage 7 — Selection (curated for a target length, not a fixed score threshold)
# ---------------------------------------------------------------------------
TARGET_OUTPUT_DURATION = 330     # seconds (5.5 min) — the real goal now; overrides pure
                                  # threshold-based selection from before
TARGET_DURATION_TOLERANCE = 30   # seconds — acceptable +/- around the target
MIN_SCORE_FLOOR = 40             # 0-100 — segments below this never get included even if the
                                  # duration budget isn't full; a quality floor, not a target
HOOK_MIN_SCORE = 60              # a candidate must clear this on quotability OR overall to be
                                  # eligible as the cold-open hook
PAD_SECONDS = 0.25               # padding before/after each kept segment (was 0.5 — tighter for
                                  # punchier pacing)

# ---------------------------------------------------------------------------
# Stage 8 — Assembly
# ---------------------------------------------------------------------------
TRANSITION_TYPE = "cut"          # "cut" = hard cuts via fast concat (punchy pacing, and MUCH
                                  # faster to render — this is also what fixes the 31-minute
                                  # Stage 8 bottleneck from the crossfade-everything version).
                                  # "fade" = the old crossfade-every-cut behavior, still available.
TRANSITION_DURATION = 0.15       # only used when TRANSITION_TYPE = "fade"
OUTPUT_RESOLUTION = None         # e.g. "1920x1080" to force a resolution; None = keep source

CAPTIONS_ENABLED = True          # burn in captions from the transcript
CAPTION_FONT = "Noto Sans"       # NOTE: for Bengali/Hindi captions to render correctly, this
                                  # needs to be a font that actually covers that script, e.g.
                                  # "Noto Sans Bengali" / "Noto Sans Devanagari" — if the font
                                  # isn't installed on the machine or doesn't cover the script,
                                  # captions will show as boxes/garbled text instead of failing
                                  # loudly. Check the first real Bengali test carefully.
CAPTION_FONT_SIZE = 28

# ---------------------------------------------------------------------------
# Explainer mode — narrated movie/series recap, EN/BN/HI (separate from the
# highlight-reel mode above; reuses Stage 1/2, replaces Stages 6-8)
# ---------------------------------------------------------------------------
EXPLAINER_DIR = DATA_DIR / "explainer"   # scripts, per-language TTS audio, beat clips
EXPLAINER_DIR.mkdir(parents=True, exist_ok=True)


# CHANGED 2026-08-27 (multi-language redesign): each explainer job now picks ONE language
# up front (dashboard dropdown / API `language` param) and runs the ENTIRE pipeline —
# Stage 3's narration, Stage 4's matching, Stage 5's voice, Stage 6's retiming — natively in
# that language, from a subtitle file supplied IN that language. This is a deliberate design
# choice over machine-translating a single English edit into other languages: it lets Gemini
# reason and write narration natively per language, at the cost that two languages of the
# same movie are independently-generated recaps (their own hook/beat choices), not
# guaranteed to be the identical edit just re-voiced. EXPLAINER_LANGUAGES is now the list of
# SUPPORTED/OFFERED languages (what populates the dashboard dropdown), not "languages to
# generate in one job" — every job carries its own single `language`.
EXPLAINER_LANGUAGES = ["en", "hi", "bn"]
EXPLAINER_LANGUAGE_NAMES = {"en": "English", "hi": "Hindi", "bn": "Bengali"}  # for Stage 3's
                                   # prompt ("write the narration in {name}") and dashboard labels

# English can still fall back to Whisper when no subtitle is supplied (unchanged prior
# behavior). Hindi and Bengali CANNOT — Whisper transcribes whatever language is actually
# SPOKEN in the source video's audio track, it does not translate, so running Whisper against
# (most likely) English/original-language audio and labeling the result "Hindi" would silently
# produce the wrong-language transcript. A real subtitle in that language is required instead
# (pipeline/subtitles.py enforces this — see find_subtitle_file()/load_or_transcribe()).
EXPLAINER_SUBTITLE_REQUIRED = {"en": False, "hi": True, "bn": True}

LOCALE_BY_LANGUAGE = {"en": "en-US", "hi": "hi-IN", "bn": "bn-IN"}  # Chirp3-HD locale codes —
                                   # CORRECTED 2026-08-27: the old "bn-BD" here was specific to
                                   # the since-removed bare-name Gemini TTS model (Charon-era);
                                   # Chirp3-HD's real supported Bengali locale is "Bengali
                                   # (India)" = bn-IN, confirmed directly against Google's
                                   # current Chirp3-HD voice list (bn-IN-Chirp3-HD-<name>, same
                                   # 30-name set as every other locale).

# Stage 3 — Summary generation (redesigned 2026-08-26). ONE Gemini call, SCHEMA-CONSTRAINED
# structured output (response_schema — Gemini is forced into the exact shape, not just asked
# for JSON in the prompt). Reads the full transcript, writes ONLY the beat-by-beat narration
# text — no citations/timestamps here, that's Stage 4's separate job. Reuses
# GEMINI_MODEL/GCP_PROJECT_ID/GCP_LOCATION from above.
SUMMARY_MAX_RETRIES = 4
SUMMARY_RETRY_BACKOFF_BASE = 8

# Scene-type classification (2026-08-26) — each beat also gets a "scene_type" field,
# constrained via the schema's STRING `enum` (Gemini can only return one of these keys —
# no drift, no free text). Used two ways in explainer_stage3_summary.py: (1) fed into the
# SUMMARY_SCHEMA's enum list, and (2) the values (short definitions) are written into the
# prompt so Gemini classifies against a real definition, not just guesses from the label.
# By explicit user request, this is also a COVERAGE requirement, not just a label: the
# prompt tells Gemini not to skip a genuine action/mystery/horror moment that's actually in
# the transcript just because it's summarizing a plot-heavy story overall. Internal-only for
# now (not surfaced in the dashboard) — written into every beat dict from Stage 3 onward
# (_summary.json -> _matched.json -> _voiced.json -> _final_beats.json).
EXPLAINER_SCENE_TYPES = {
    "action": "physical action — combat, chases, stunts, or other high-intensity physical stakes",
    "mystery": "investigation, secrets, withheld information, or a question the audience is meant to wonder about",
    "horror": "dread, threat, the uncanny, or an unsettling/scary moment",
    "emotional": "drama, grief, love, relationships, or another emotionally weighty character moment",
    "comedy": "humor, comedic beats, or banter",
    "plot": "plot-advancing exposition or dialogue that doesn't fit the other categories",
}

# Hook / retention design (2026-08-26, same day as scene_type, in response to the user asking
# how the pipeline plans to keep viewers from skipping/dropping off). Each beat also gets a
# "beat_role" field, same enum-constrained-schema mechanism as scene_type. Unlike scene_type
# (which classifies WHAT a beat is), beat_role classifies its NARRATIVE JOB in the recap:
#   hook       — mandatory, exactly one, ALWAYS forced to index 0 by _reindex_beats() regardless
#                of whatever index Gemini assigns it: a flash-forward teaser of the single most
#                shocking/exciting moment in the whole story, shown before any context
#   twist      — a genuine reveal/reversal/betrayal; written to build suspense, payoff delayed
#   highlight  — another standout, most-memorable moment that isn't a twist (biggest action beat,
#                funniest line, hardest emotional gut-punch); written with extra punch/energy
#   setup      — table-setting beat building toward an upcoming highlight/twist
#   resolution — wrap-up/aftermath beat, typically near the end
#   body       — the default: a plain chronological narration beat carrying the story forward
# Same coverage-requirement principle as scene_type: don't classify everything "body" just
# because the story is mostly straightforward — a genuine twist/highlight should be tagged.
# _reindex_beats() (explainer_stage3_summary.py) enforces exactly one "hook" defensively (same
# "don't trust the model's own count" lesson as bug #7's 0-indexing fix): if Gemini tags zero,
# it proceeds without fabricating one; if it tags more than one, the first (by its own raw
# index) is kept and the rest are demoted to "highlight" rather than erroring.
EXPLAINER_BEAT_ROLES = {
    "hook": "a flash-forward teaser of the single most shocking/exciting moment in the whole "
            "story, shown before any context — always exactly one, always the first beat",
    "twist": "a reveal, reversal, or betrayal that recontextualizes what came before",
    "highlight": "another standout, most-memorable moment (biggest action, funniest line, "
                 "hardest emotional gut-punch) that isn't a twist",
    "setup": "a table-setting beat that builds toward an upcoming highlight or twist",
    "resolution": "a wrap-up or aftermath beat, typically near the end of the story",
    "body": "a plain chronological narration beat carrying the story forward — the default",
}

# Audio-event detection (2026-08-26, "silent action/horror" blind-spot fix). Whisper's VAD
# filter drops non-speech entirely, so a genuinely wordless action/horror beat (gunfire, a
# chase, a jump-scare sting) has ZERO entries in the transcript Stage 3 reads — it isn't
# classified badly, it's literally invisible to the model. pipeline/audio_events.py scans
# the FULL audio track (reusing the same librosa RMS approach as stage4_signals.py,
# highlight-reel mode's audio-energy signal) for dialogue-free gaps with sustained high
# energy, and surfaces them to Stage 3/4 as synthetic placeholder transcript-line entries
# merged into the same segment list both stages already build their prompts from — so
# Gemini can actually see, and choose to write a beat for, a scene it currently can't see
# at all. Explicit user choice (2026-08-26): sustained-energy detection only for the first
# build — distinguishing energy SHAPE (a sudden spike-after-quiet, more characteristic of a
# horror jump-scare, vs. sustained loud, more characteristic of action) was deferred to a
# later iteration, as was folding in visual motion (OpenCV, already used in highlight-reel
# mode) alongside audio.
AUDIO_EVENT_DETECTION_ENABLED = True
AUDIO_EVENT_WINDOW_SECONDS = 2.0        # energy is measured in windows this long
AUDIO_EVENT_MIN_GAP_SECONDS = 8.0       # only consider dialogue-free spans at least this
                                          # long — skips ordinary pauses between lines of
                                          # dialogue, which are not what this is for
AUDIO_EVENT_ENERGY_RATIO = 1.5          # a window counts as "loud" at >= this many times
                                          # the track's own median energy (relative to this
                                          # movie's own audio, not an absolute loudness value)
AUDIO_EVENT_MIN_LOUD_FRACTION = 0.6     # >= this fraction of a gap's windows must be "loud"
                                          # for the whole gap to count as sustained — rejects
                                          # a single stray loud moment in an otherwise quiet gap

# Stage 4 — Clip matching (new, 2026-08-26). A SEPARATE Gemini call (schema-constrained) from
# Stage 3, by explicit design choice: reads Stage 3's beats + the full transcript, and for each
# beat returns which transcript segment(s) it's describing — BY INDEX ONLY. Python then resolves
# the real start/end from Whisper's own transcript (Stage 2), so Gemini can never hallucinate a
# timestamp, only point at a transcript line that genuinely exists.
MATCH_MAX_RETRIES = 4
MATCH_RETRY_BACKOFF_BASE = 8

# Stage 3.5 — Intro generation (new, 2026-08-27, by explicit user request: "add a 10 seconds
# intro... which topic today we will explain"). A SEPARATE, one-off Gemini call from Stage 3's
# beat summary — one short channel-style intro line ("Today, we're diving into a heist thriller
# where..."), NOT a plot-teaser (that's what the "hook" beat_role already is) and, by explicit
# design choice, does NOT state the movie's exact title (this project's filenames are messy —
# "Balls Up (2026) 720p AMZN-WEB x264 MSubs..." — so naming a title risks either garbage or a
# wrong guess; describing the premise/genre/tone is the safer, still-hooky choice). Cites a real
# transcript line for its footage using the SAME anti-hallucination index-citation mechanism as
# Stage 4 (reuses explainer_stage4_match.resolve_matches() directly rather than reimplementing
# it — including its out-of-range-timestamp recovery, see Bug log #11). Output is a SEPARATE
# Intro.mp4/intro-voice.mp3 pair living directly inside the existing clips/ and voices/ folders
# (not a new subfolder, not folded into the numbered Clip-N/voice-N beat sequence, not
# renumbering "hook" — explicit user choice after discussion) — data/explainer/{movie}/
# {language}/clips/Intro.mp4 and .../voices/intro-voice.mp3.
INTRO_MAX_RETRIES = 4
INTRO_RETRY_BACKOFF_BASE = 8
INTRO_TARGET_SECONDS = 10        # rough target, NOT enforced/validated in code — only a hint in
                                  # the prompt. Actual clip length comes from the REAL synthesized
                                  # TTS audio duration (same target_duration = max(1.0,
                                  # voice_duration + 2*BEAT_PAD_SECONDS) formula every other beat
                                  # already uses in Stage 6), so it naturally lands in roughly the
                                  # 8-12s range rather than being force-locked to exactly 10.00s —
                                  # consistent with how every other beat in this pipeline is timed.
INTRO_TARGET_WORDS = 25          # ~150 wpm average narration pace -> roughly INTRO_TARGET_SECONDS
                                  # worth of speech; also just a prompt hint, not validated.

# Stage 5 — TTS narration (Cloud Text-to-Speech API, PAID — same Vertex/GCP ADC auth, no API
# key). CHANGED 2026-08-26: switched from Gemini TTS ("Charon") to Chirp 3: HD ("Fenrir") after
# the user A/B-tested Gemini TTS voices, Chirp3-HD voices, and Studio voices side by side and
# picked Chirp3-HD/Fenrir as more natural-sounding. This is a real provider switch within the
# same GCP project (no new vendor/billing) but two real consequences:
#   1. Chirp3-HD voice names are LOCALE-PREFIXED (e.g. "en-US-Chirp3-HD-Fenrir"), not bare names
#      like Gemini TTS used — and it is NOT a Gemini model, so no model_name field is passed.
#   2. Chirp3-HD has NO free-text style `prompt` field — so the per-beat adaptive-intensity
#      design (INTENSITY_THRESHOLDS/INTENSITY_STYLE_PROMPTS, audio-energy/visual-motion signal
#      computation) is REMOVED, by explicit user request ("dont change intensity"). Every beat
#      now gets one consistent plain read. If per-beat style variation is wanted again later,
#      Gemini TTS is still available and this is a straightforward revert — see the project doc's
#      bug/decision log for exactly what changed.
# Pricing (confirmed via Google's own pricing page, 2026-08-26): Chirp 3: HD is $30 per million
# characters — a movie recap's narration script (a few thousand characters) costs a few cents.
# CHANGED 2026-08-27 (multi-language redesign): NARRATOR_VOICE (singular) is now
# NARRATOR_VOICES, a per-language dict — the whole point being that changing a voice in the
# future is a one-line edit here, not a code change. "en" is the already-in-production pick
# (Fenrir, chosen via A/B test, see above). "hi" and "bn" were chosen the same way, by the
# user running test_voice_hindi_chirp3hd.py / test_voice_bengali_chirp3hd.py and listening to
# every real Chirp3-HD voice on the project for that locale; explainer_stage5_tts.py must
# still raise a clear error
# if a bn job is submitted before this is set, not silently fall back to some other language's
# voice or a wrong-locale name.
NARRATOR_VOICES = {
    "en": "en-US-Chirp3-HD-Fenrir",
    "hi": "hi-IN-Chirp3-HD-Achird",
    "bn": "bn-IN-Chirp3-HD-Algenib",
}
TTS_MAX_RETRIES = 4
TTS_RETRY_BACKOFF_BASE = 5

# Stage 6 — Per-beat clip retiming (redesigned 2026-08-26 — this is NOT a full assembly
# anymore, by explicit user choice). Exact cited timestamp, retimed with a speed change
# (setpts) to match the narration's real spoken duration, as long as that only needs a
# tasteful speed adjustment. Outside that range, extend real surrounding footage (or
# freeze-hold) instead of an ugly extreme stretch. Output is VIDEO-ONLY (no audio track) —
# no mixing, no concatenation. Final assembly is done by hand from the Clip-N.mp4 /
# voice-N.mp3 pairs this stage produces. CHANGED same day: now also applies output framing
# (16:9 crop-to-fill + burned-in CC removal, see below) and burns in "Summary CC" captions
# from each beat's own narration text — reverses the original "no captions" call, by
# explicit user request.
STRETCH_MIN_RATIO = 0.7          # fastest allowed speed-up (target/source ratio floor)
STRETCH_MAX_RATIO = 1.4          # slowest allowed slow-down (target/source ratio ceiling)
MIN_CLIP_SNIPPET_SECONDS = 1.0   # an uncited (point-in-time) beat still grabs at least this
                                  # much real footage around its resolved timestamp
CLIP_EXTENSION_MAX_SECONDS = 6.0 # how much extra source footage we'll pull around a cited
                                  # moment when even max stretch isn't enough, before
                                  # freeze-framing the remainder
BEAT_PAD_SECONDS = 0.3           # a little breathing room added to each beat's target clip
                                  # length beyond its narration's exact spoken duration
EXPLAINER_CAPTIONS_ENABLED = False  # "Summary CC" (2026-08-26) — burn each beat's own
                                     # narration text into its Clip-N.mp4, one cue spanning
                                     # the whole clip. Uses the same CAPTION_FONT/
                                     # CAPTION_FONT_SIZE as highlight-reel mode.
                                     # REVERSED 2026-08-26: turned back off by explicit
                                     # request — Explainer clips go back to silent,
                                     # caption-free video-only footage (no burned-in CC
                                     # of any kind). Highlight-reel mode's own verbatim
                                     # captions are a separate flag/path and unaffected.

# ---------------------------------------------------------------------------
# Output framing (2026-08-26) — 16:9 crop-to-fill + burned-in CC removal,
# applied to every clip cut in BOTH modes (pipeline/framing.py). Explicit
# user choice: crop-to-fill, not letterbox/pad (loses some picture at the
# edges rather than adding black bars).
# ---------------------------------------------------------------------------
OUTPUT_ASPECT_RATIO = (16, 9)

# Not every source video has burned-in (baked into the picture) captions —
# confirmed by checking two real sources: "The Book of Boba Fett_S1E1" DOES
# (plain white speaker-labeled text, bottom-center — measured by
# check_source_captions.py's bright-pixel-row scan across 8 real dialogue
# frames, consistent band at y=677-772 out of 804px height, i.e. starting
# ~84.2% down the frame); "A Man in Full - S01E01 - Saddlebags WEBRip-1080p"
# does NOT (same tool, 8 samples, no consistent band found) and has no
# entry below, so it gets no extra crop. Neither source has a soft-subtitle
# stream (confirmed via ffprobe), so this can't be solved by just not
# muxing a subtitle track — the only reliable removal is cropping the band
# off before it ever reaches a clip. Use check_source_captions.py on any
# NEW source video to measure its own caption band (if any) and get a
# suggested entry to paste in here — don't guess a value by eye.
SOURCE_CROP_OVERRIDES = {
    "The Book of Boba Fett_S1E1": {
        "crop_bottom_pct": 0.19,   # tool-measured band starts at 84.2% down, +3% margin
                                    # for occasional 3-line captions -> 19% (check_source_captions.py output)
    },
}
DEFAULT_CROP_BOTTOM_PCT = 0.0   # sources not listed above get no extra bottom trim

# ---------------------------------------------------------------------------
# Subtitle generate/translate utilities (2026-09-05) — solves "movie has no
# subtitle in the language I need" without hunting one down online.
#
# generate_subtitle() (pipeline/subtitles.py) extends what English mode
# already did SILENTLY (a Whisper fallback) into an explicit, user-confirmed
# action available for ANY language, including Hindi/Bengali — those still
# NEVER silently fall back to Whisper inside load_or_transcribe() (see
# EXPLAINER_SUBTITLE_REQUIRED above; that rule is unchanged), but the user
# can now explicitly ask "generate one from the movie's own audio" instead
# of needing to find a real subtitle file.
#
# translate_subtitle() takes an existing .srt in one language and produces a
# new one in another, SAME timestamps, translated text only — this is the
# "translate-first" architecture: rather than teaching Gemini to write
# narration directly from a different-language transcript (which would make
# Stage 4's already-fragile citation-matching, see Bug log #11, do harder
# cross-language matching), a translated subtitle just becomes a normal
# same-language subtitle as far as the rest of the pipeline is concerned —
# Stage 3/4/5/6 need ZERO changes.
# ---------------------------------------------------------------------------
TRANSLATE_BATCH_SIZE = 60         # subtitle cues translated per Gemini call — text-only and much
                                    # shorter per-item than Stage 3's summary prompt, so a larger
                                    # batch than SCORE_BATCH_SIZE (30) is safe; still batched (not
                                    # one giant call) so a very long movie's cue count doesn't risk
                                    # truncation, and so one bad batch doesn't sink the whole file.
TRANSLATE_MAX_RETRIES = 4
TRANSLATE_RETRY_BACKOFF_BASE = 8  # seconds — same exponential backoff shape as SUMMARY/MATCH above
