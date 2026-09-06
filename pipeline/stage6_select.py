"""
Stage 7 — Select (curated for a target length, not just a score threshold)

Builds a ~5-6 minute (configurable) edit that:
  1. Opens with a cold-open hook — the single most quotable/striking moment,
     out of chronological order, dropped first with no setup.
  2. Fills the rest of the runtime with a DIVERSE mix across five categories
     (plot / emotion / humor / action / quotability) rather than just
     whatever scored highest overall — avoids an edit that's all action and
     no character moments, or vice versa.
  3. Tries to close on a strong plot/emotional beat from late in the movie,
     for a cliffhanger rather than an arbitrary fade-out.
Everything below MIN_SCORE_FLOOR never gets in, regardless of duration budget.
Free, CPU-only, plain Python.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import settings


def _padded_bounds(seg):
    return max(0.0, seg["start"] - settings.PAD_SECONDS), seg["end"] + settings.PAD_SECONDS


def _padded_duration(seg):
    start, end = _padded_bounds(seg)
    return end - start


def _dedupe_overlaps(segments_in_order):
    """Merge padded segments that overlap/touch once placed in chronological order,
    so we never cut the same footage twice. Keeps the highest overall_score and
    concatenates reasons/text when merging."""
    merged = []
    for seg in segments_in_order:
        start, end = _padded_bounds(seg)
        if merged and start <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], end)
            merged[-1]["overall_score"] = max(merged[-1]["overall_score"], seg["overall_score"])
            merged[-1]["reason"] = f"{merged[-1]['reason']} | {seg.get('reason', '')}"
            merged[-1]["text"] = f"{merged[-1]['text']} {seg.get('text', '')}".strip()
        else:
            start_p, end_p = start, end
            merged.append({
                "start": round(start_p, 2),
                "end": round(end_p, 2),
                "overall_score": seg["overall_score"],
                "reason": seg.get("reason", ""),
                "text": seg.get("text", ""),
            })
    return merged


def _select_diverse_by_duration(candidates: list) -> list:
    """Round-robin over score categories, picking the best not-yet-picked candidate in
    each category per round, until the duration budget is filled. This is what
    guarantees a mix instead of an edit dominated by one category."""
    target = settings.TARGET_OUTPUT_DURATION
    tolerance = settings.TARGET_DURATION_TOLERANCE
    max_duration = target + tolerance

    selected_ids = set()
    selected = []
    running_duration = 0.0

    remaining_by_category = {
        cat: sorted(candidates, key=lambda s: s[cat], reverse=True)
        for cat in settings.SCORE_CATEGORIES
    }

    made_progress = True
    while running_duration < target and made_progress:
        made_progress = False
        for cat in settings.SCORE_CATEGORIES:
            pool = remaining_by_category[cat]
            # find the next candidate in this category's ranking that isn't picked yet
            pick = next((s for s in pool if id(s) not in selected_ids), None)
            if pick is None:
                continue
            dur = _padded_duration(pick)
            if running_duration + dur > max_duration:
                continue  # would overshoot — skip this category this round, try others
            selected_ids.add(id(pick))
            selected.append(pick)
            running_duration += dur
            made_progress = True
            if running_duration >= target:
                break

    return selected


def _pick_hook(candidates: list, already_selected_ids: set):
    """Best cold-open candidate: highest of (quotability_score, overall_score),
    must clear HOOK_MIN_SCORE on at least one of those."""
    def hook_rank(s):
        return max(s["quotability_score"], s["overall_score"])

    eligible = [s for s in candidates if hook_rank(s) >= settings.HOOK_MIN_SCORE]
    if not eligible:
        return None
    return max(eligible, key=hook_rank)


def _polish_ending(candidates: list, selected: list, hook, video_end_time: float):
    """If the chronologically-last selected segment (excluding the hook) is weak on
    plot/emotion, look for a stronger plot beat in the final 20% of the source video
    to close on instead — a deliberate cliffhanger rather than an arbitrary cutoff."""
    body = [s for s in selected if s is not hook]
    if not body:
        return selected
    body_sorted = sorted(body, key=lambda s: s["start"])
    last = body_sorted[-1]

    if last["plot_score"] + last["emotion_score"] >= 80:
        return selected  # already a strong closer, leave it alone

    late_window_start = video_end_time * 0.8
    already_selected_ids = {id(s) for s in selected}
    late_candidates = [
        s for s in candidates
        if s["start"] >= late_window_start and id(s) not in already_selected_ids
    ]
    if not late_candidates:
        return selected

    better_closer = max(late_candidates, key=lambda s: s["plot_score"] + s["emotion_score"])
    if better_closer["plot_score"] + better_closer["emotion_score"] <= last["plot_score"] + last["emotion_score"]:
        return selected  # nothing actually better available

    # swap in: add the better closer, drop the current weakest non-hook segment if over budget
    new_selected = selected + [better_closer]
    total = sum(_padded_duration(s) for s in new_selected)
    max_duration = settings.TARGET_OUTPUT_DURATION + settings.TARGET_DURATION_TOLERANCE
    if total > max_duration:
        droppable = [s for s in new_selected if s is not hook and s is not better_closer]
        if droppable:
            weakest = min(droppable, key=lambda s: s["overall_score"])
            new_selected.remove(weakest)
    return new_selected


def select_segments(video_stem: str) -> Path:
    scored_path = settings.ANALYSIS_DIR / f"{video_stem}_scored.json"
    all_segments = json.loads(scored_path.read_text(encoding="utf-8"))["segments"]

    candidates = [s for s in all_segments if s["overall_score"] >= settings.MIN_SCORE_FLOOR]
    if not candidates:
        raise RuntimeError(
            f"No segments cleared MIN_SCORE_FLOOR ({settings.MIN_SCORE_FLOOR}). "
            "Lower it in config/settings.py and re-run selection."
        )

    video_end_time = max(s["end"] for s in all_segments)

    selected = _select_diverse_by_duration(candidates)
    selected_ids = {id(s) for s in selected}

    hook = _pick_hook(candidates, selected_ids)
    if hook is not None and id(hook) not in selected_ids:
        selected.append(hook)
        total = sum(_padded_duration(s) for s in selected)
        max_duration = settings.TARGET_OUTPUT_DURATION + settings.TARGET_DURATION_TOLERANCE
        if total > max_duration:
            droppable = [s for s in selected if s is not hook]
            if droppable:
                weakest = min(droppable, key=lambda s: s["overall_score"])
                selected.remove(weakest)

    selected = _polish_ending(candidates, selected, hook, video_end_time)

    # Final play order: hook first (if any), then the rest chronologically
    if hook is not None:
        body = sorted([s for s in selected if s is not hook], key=lambda s: s["start"])
        play_order = [hook] + body
    else:
        play_order = sorted(selected, key=lambda s: s["start"])

    final_segments = _dedupe_overlaps_preserving_hook(play_order, hook)

    total_duration = sum(s["end"] - s["start"] for s in final_segments)

    out_path = settings.ANALYSIS_DIR / f"{video_stem}_selected.json"
    out_path.write_text(json.dumps({
        "target_duration_sec": settings.TARGET_OUTPUT_DURATION,
        "total_duration_sec": round(total_duration, 1),
        "hook_included": hook is not None,
        "segments": final_segments,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[stage7] selected {len(final_segments)} segments "
          f"(target {settings.TARGET_OUTPUT_DURATION}s, got {total_duration:.1f}s), "
          f"hook={'yes' if hook is not None else 'no'} -> {out_path}")
    return out_path


def _dedupe_overlaps_preserving_hook(play_order: list, hook) -> list:
    """Same overlap-merging as _dedupe_overlaps, but keeps the hook as element 0 rather
    than letting it merge into whatever comes chronologically near it."""
    if hook is None:
        return _dedupe_overlaps(play_order)
    hook_merged = _dedupe_overlaps([hook])
    body_merged = _dedupe_overlaps([s for s in play_order if s is not hook])
    return hook_merged + body_merged


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python stage6_select.py <video_stem>")
        sys.exit(1)
    select_segments(sys.argv[1])
