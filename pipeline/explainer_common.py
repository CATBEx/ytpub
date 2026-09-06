"""
Shared helpers used by both explainer_stage4_tts.py and explainer_stage5_assemble.py,
so the two stages agree on where each beat's footage actually comes from.
"""


def beat_envelope(beats: list, i: int):
    """(start, end) footage window for beat i. Cited beats use their own
    transcript-derived envelope; an uncited (pure-transition) beat gets a
    zero-width point at the midpoint between its neighbors' envelopes, so it
    still lands on real footage from roughly the right place in the movie."""
    beat = beats[i]
    if beat["start"] is not None:
        return beat["start"], beat["end"]

    prev_end = next((beats[j]["end"] for j in range(i - 1, -1, -1) if beats[j]["end"] is not None), None)
    next_start = next((beats[j]["start"] for j in range(i + 1, len(beats)) if beats[j]["start"] is not None), None)
    if prev_end is not None and next_start is not None:
        mid = (prev_end + next_start) / 2
    elif prev_end is not None:
        mid = prev_end
    elif next_start is not None:
        mid = next_start
    else:
        mid = 0.0
    return mid, mid
