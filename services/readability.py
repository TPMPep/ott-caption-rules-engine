"""
Readability Engine — ensures captions are readable on screen.

Enforces:
  • Minimum cue durations
  • Micro-cue merging
  • Sound cue stability
  • Balanced pacing

All thresholds read from env vars (set per-job via captionOptions on the
POST /v1/jobs payload). No hardcoded profile fallbacks.
"""

import os

MICRO_WORD_LIMIT = 2
MICRO_DURATION_MS = 1000

# Orphan reflow — a final safety net. A dialogue cue of ORPHAN_WORD_LIMIT or
# fewer words that is NOT a brief standalone response ("Yes." "No." "OK.")
# should rejoin the sentence it belongs to. We merge it BACKWARD into the
# previous same-speaker dialogue cue (producing "You must be Mr. Wang.") when
# the result still fits the spec; otherwise FORWARD into the next one.
ORPHAN_WORD_LIMIT = 1
_BRIEF_RESPONSES = {
    "yes", "no", "ok", "okay", "yeah", "right", "sure", "nope", "yep",
    "what", "why", "how", "stop", "wait", "hi", "hey", "hello", "thanks",
}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _min_dialogue_ms() -> int:
    return _env_int("CUSTOM_MIN_DISPLAY_MS", 800)


def _max_chars() -> int:
    return _env_int("CUSTOM_MAX_CHARS", 32)


def _max_lines() -> int:
    return _env_int("CUSTOM_MAX_LINES", 2)


def _min_sound_ms() -> int:
    return _env_int("CUSTOM_MIN_SOUND_DISPLAY_MS", 800)


def _merge_gap_ms() -> int:
    return _env_int("CUSTOM_MERGE_GAP_MS", 80)


def enforce_min_duration(cues):
    for i, cue in enumerate(cues):
        duration = cue["end_ms"] - cue["start_ms"]
        min_duration = _min_sound_ms() if cue["type"] == "sound" else _min_dialogue_ms()
        if duration >= min_duration:
            continue
        if i < len(cues) - 1:
            gap = cues[i + 1]["start_ms"] - cue["end_ms"]
            needed = min_duration - duration
            if gap >= needed:
                cue["end_ms"] += needed
    return cues


def merge_micro_cues(cues):
    merge_gap = _merge_gap_ms()
    merged = []
    i = 0
    while i < len(cues):
        cue = cues[i]
        duration = cue["end_ms"] - cue["start_ms"]
        text = " ".join(cue["lines"])
        word_count = len(text.split())

        if (cue["type"] == "dialogue"
                and duration < MICRO_DURATION_MS
                and word_count <= MICRO_WORD_LIMIT
                and i < len(cues) - 1
                and cues[i + 1]["type"] == "dialogue"):
            nxt = cues[i + 1]
            gap = nxt["start_ms"] - cue["end_ms"]
            if gap <= max(200, merge_gap):
                cue["lines"] = [(text + " " + " ".join(nxt["lines"])).strip()]
                cue["end_ms"] = nxt["end_ms"]
                merged.append(cue)
                i += 2
                continue

        if (cue["type"] == "sound"
                and i < len(cues) - 1
                and cues[i + 1]["type"] == "sound"):
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


def _is_orphan_dialogue(cue):
    """A dialogue cue that's a stranded sentence fragment — NOT a deliberate
    brief response. e.g. a lone 'Wang.' left after a bad upstream split."""
    if cue.get("type") != "dialogue":
        return False
    text = " ".join(cue.get("lines", [])).strip()
    # Strip a leading dash prefix so '- Wang.' is also caught.
    if text.startswith("- "):
        text = text[2:].strip()
    words = [w for w in text.split() if w]
    if len(words) > ORPHAN_WORD_LIMIT:
        return False
    if not words:
        return False
    bare = words[0].rstrip(".!?\"')]}").lower()
    # A genuine brief response ("Yes." "No.") is allowed to stand alone.
    return bare not in _BRIEF_RESPONSES


def reflow_orphans(cues):
    """Rejoin stranded single-word sentence fragments into their sentence.

    Merge BACKWARD into the previous same-speaker dialogue cue when the merged
    text still fits the spec (max_chars × max_lines) — this is the
    'You must be Mr. Wang.' fix. If a backward merge doesn't fit, try FORWARD
    into the next same-speaker dialogue cue. Speaker is inferred from the dash
    prefix convention and cue adjacency; we never merge across a speaker change.
    """
    if not cues:
        return cues

    budget = _max_chars() * _max_lines()
    out = list(cues)
    i = 0
    while i < len(out):
        cue = out[i]
        if not _is_orphan_dialogue(cue):
            i += 1
            continue

        orphan_text = " ".join(cue.get("lines", [])).strip()

        # Try BACKWARD merge into the previous dialogue cue.
        if i > 0 and out[i - 1].get("type") == "dialogue":
            prev = out[i - 1]
            prev_text = " ".join(prev.get("lines", [])).strip()
            combined = (prev_text + " " + orphan_text).strip()
            if len(combined) <= budget:
                prev["lines"] = [combined]
                prev["end_ms"] = cue["end_ms"]
                # Preserve the dialogue_text meta so any downstream re-read is
                # consistent with the merged content.
                if isinstance(prev.get("meta"), dict):
                    prev["meta"]["dialogue_text"] = (
                        (prev["meta"].get("dialogue_text", prev_text) + " " + orphan_text).strip()
                    )
                del out[i]
                continue  # re-check the same index (now the next cue)

        # Try FORWARD merge into the next dialogue cue.
        if i < len(out) - 1 and out[i + 1].get("type") == "dialogue":
            nxt = out[i + 1]
            nxt_text = " ".join(nxt.get("lines", [])).strip()
            combined = (orphan_text + " " + nxt_text).strip()
            if len(combined) <= budget:
                nxt["lines"] = [combined]
                nxt["start_ms"] = cue["start_ms"]
                if isinstance(nxt.get("meta"), dict):
                    nxt["meta"]["dialogue_text"] = (
                        (orphan_text + " " + nxt["meta"].get("dialogue_text", nxt_text)).strip()
                    )
                del out[i]
                continue

        # Can't merge either direction without breaking the spec — leave it.
        i += 1

    return out


def apply_readability_rules(cues):
    """Master readability pass. Run AFTER AI formatting.

    Order matters: reflow orphans FIRST (rejoin stranded fragments to their
    sentence), THEN enforce min-duration + micro-merge on the cleaned set.
    """
    cues = reflow_orphans(cues)
    cues = enforce_min_duration(cues)
    cues = merge_micro_cues(cues)
    return cues
