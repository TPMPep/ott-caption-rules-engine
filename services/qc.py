"""
QC (Quality Control) Report Generator.

Every profile/threshold is sourced from env vars at runtime so the
formatter and QC always see the same rules. Auditor-grade output:
qc_report() returns `_rules_used` so a reviewer can answer "what rules
graded this cue?" from the result alone.
"""

import os
import re
from typing import Dict, List

FUNCTION_WORDS = {
    "a", "an", "the", "of", "to", "and", "or", "but",
    "with", "from", "in", "on", "at", "for", "that",
    "this", "these", "those",
}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _caption_profile() -> str:
    return (os.getenv("CAPTION_PROFILE", "") or "").strip().lower()


def _max_lines() -> int:
    return _env_int("CUSTOM_MAX_LINES", 2)


def _max_chars() -> int:
    return _env_int("CUSTOM_MAX_CHARS", 32)


def _min_dialogue_ms() -> int:
    return _env_int("CUSTOM_MIN_DISPLAY_MS", 800)


def _min_sound_ms() -> int:
    return _env_int("CUSTOM_MIN_SOUND_DISPLAY_MS", 800)


def ends_with_function_word(line: str) -> bool:
    line = (line or "").strip().lower()
    if not line:
        return False
    line = re.sub(r"[^\w']+$", "", line)
    parts = line.split()
    if not parts:
        return False
    return parts[-1] in FUNCTION_WORDS


def count_function_word_endings(lines: List[str]) -> int:
    return sum(1 for line in lines if ends_with_function_word(line))


def is_one_word(lines: List[str]) -> bool:
    text = " ".join(lines).strip()
    words = re.findall(r"\b\w+\b", text)
    return len(words) == 1


def count_overlaps(cues: List[Dict]) -> int:
    overlap = 0
    for i in range(len(cues) - 1):
        if cues[i]["end_ms"] > cues[i + 1]["start_ms"]:
            overlap += 1
    return overlap


def count_sound_overlaps(cues: List[Dict]) -> int:
    overlap = 0
    for i in range(len(cues) - 1):
        if cues[i]["end_ms"] > cues[i + 1]["start_ms"]:
            if cues[i]["type"] == "sound" or cues[i + 1]["type"] == "sound":
                overlap += 1
    return overlap


def count_protected_phrase_splits(lines: List[str], protected_phrases: List[str]) -> int:
    joined = "\n".join(lines)
    count = 0
    for phrase in protected_phrases or []:
        if not phrase:
            continue
        if phrase in joined.replace("\n", " ") and phrase.replace(" ", "\n") in joined:
            count += 1
    return count


def qc_report(cues_in: int, cues_out: List[Dict], protected_phrases: List[str]) -> Dict:
    max_lines = _max_lines()
    max_chars = _max_chars()
    min_dialogue_ms = _min_dialogue_ms()
    min_sound_ms = _min_sound_ms()

    max_lines_violation = 0
    max_chars_violation = 0
    short_duration_violations = 0
    one_word_dialogue_cues = 0
    function_word_endings = 0
    protected_phrase_splits = 0

    for cue in cues_out:
        if len(cue["lines"]) > max_lines:
            max_lines_violation += 1
        if any(len(line) > max_chars for line in cue["lines"]):
            max_chars_violation += 1
        duration = cue["end_ms"] - cue["start_ms"]
        if cue["type"] == "sound":
            if duration < min_sound_ms:
                short_duration_violations += 1
        else:
            if duration < min_dialogue_ms:
                short_duration_violations += 1
            if is_one_word(cue["lines"]):
                one_word_dialogue_cues += 1
        function_word_endings += count_function_word_endings(cue["lines"])
        protected_phrase_splits += count_protected_phrase_splits(cue["lines"], protected_phrases)

    return {
        "cues_in": cues_in,
        "cues_out": len(cues_out),
        "overlaps": count_overlaps(cues_out),
        "max_lines_violation": max_lines_violation,
        "max_chars_violation": max_chars_violation,
        "short_duration_violations": short_duration_violations,
        "one_word_dialogue_cues": one_word_dialogue_cues,
        "sound_overlap_violations": count_sound_overlaps(cues_out),
        "function_word_endings": function_word_endings,
        "protected_phrase_splits": protected_phrase_splits,
        "_rules_used": {
            # Truthful audit label. The engine is spec-agnostic — every rule
            # below comes from the CUSTOM_* / SPEAKER_LABEL_MODE / MUSIC_CUE_*
            # knobs the Base44 producer derives from the pinned spec. When no
            # CAPTION_PROFILE env var is sent (the correct, spec-driven path),
            # report 'spec_driven' rather than stamping a misleading client
            # name on the deliverable. SOC 2 CC8.1 — the QC record never
            # claims a profile that wasn't actually applied.
            "profile": _caption_profile() or "spec_driven",
            "max_lines": max_lines,
            "max_chars": max_chars,
            "min_dialogue_ms": min_dialogue_ms,
            "min_sound_ms": min_sound_ms,
        },
    }
