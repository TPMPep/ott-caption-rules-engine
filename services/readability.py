"""
Readability Engine

Ensures captions are readable on screen by enforcing:

• Minimum cue durations
• Micro-cue merging
• Sound cue stability
• Balanced pacing
"""

MIN_DIALOGUE_MS = 800
MIN_SOUND_MS = 800
MICRO_WORD_LIMIT = 2
MICRO_DURATION_MS = 1000


def enforce_min_duration(cues):
    """
    Ensure captions stay on screen long enough to read.
    """
    for i, cue in enumerate(cues):

        duration = cue["end_ms"] - cue["start_ms"]

        min_duration = MIN_SOUND_MS if cue["type"] == "sound" else MIN_DIALOGUE_MS

        if duration >= min_duration:
            continue

        if i < len(cues) - 1:

            gap = cues[i + 1]["start_ms"] - cue["end_ms"]
            needed = min_duration - duration

            if gap >= needed:
                cue["end_ms"] += needed

    return cues


def merge_micro_cues(cues):
    """
    Merge tiny captions with neighbors.

    Fixes:
    • single word captions
    • two word flashes
    • unnatural segmentation
    """

    merged = []
    i = 0

    while i < len(cues):

        cue = cues[i]

        duration = cue["end_ms"] - cue["start_ms"]

        text = " ".join(cue["lines"])
        word_count = len(text.split())

        # merge micro dialogue cues
        if (
            cue["type"] == "dialogue"
            and duration < MICRO_DURATION_MS
            and word_count <= MICRO_WORD_LIMIT
            and i < len(cues) - 1
            and cues[i + 1]["type"] == "dialogue"
        ):

            nxt = cues[i + 1]

            gap = nxt["start_ms"] - cue["end_ms"]

            if gap <= 200:

                cue["lines"] = [
                    (text + " " + " ".join(nxt["lines"])).strip()
                ]

                cue["end_ms"] = nxt["end_ms"]

                merged.append(cue)

                i += 2
                continue

        # merge consecutive sound cues
        if (
            cue["type"] == "sound"
            and i < len(cues) - 1
            and cues[i + 1]["type"] == "sound"
        ):

            nxt = cues[i + 1]

            a = cue["lines"][0].strip("[]")
            b = nxt["lines"][0].strip("[]")

            cue["lines"] = [f"[{a} AND {b}]"]

            cue["end_ms"] = max(cue["end_ms"], nxt["end_ms"])

            merged.append(cue)

            i += 2
            continue

        merged.append(cue)
        i += 1

    return merged


def apply_readability_rules(cues):
    """
    Master readability pass.

    Run this AFTER AI formatting.
    """

    cues = enforce_min_duration(cues)
    cues = merge_micro_cues(cues)

    return cues
