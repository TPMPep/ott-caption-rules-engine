import re
from typing import Dict, List

FUNCTION_WORDS = {
    "a", "an", "the", "of", "to", "and", "or", "but",
    "with", "from", "in", "on", "at", "for", "that",
    "this", "these", "those"
}


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


def violates_line_limits(lines: List[str], max_lines: int = 2, max_chars: int = 32) -> bool:
    if len(lines) > max_lines:
        return True
    return any(len(line) > max_chars for line in lines)


def count_overlaps(cues: List[Dict]) -> int:
    overlap_count = 0
    for i in range(len(cues) - 1):
        if cues[i]["end_ms"] > cues[i + 1]["start_ms"]:
            overlap_count += 1
    return overlap_count


def count_sound_overlaps(cues: List[Dict]) -> int:
    overlap_count = 0
    for i in range(len(cues) - 1):
        if cues[i]["end_ms"] > cues[i + 1]["start_ms"]:
            if cues[i]["type"] == "sound" or cues[i + 1]["type"] == "sound":
                overlap_count += 1
    return overlap_count


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
    max_lines_violation = 0
    max_chars_violation = 0
    short_duration_violations = 0
    one_word_dialogue_cues = 0
    function_word_endings = 0
    protected_phrase_splits = 0

    for cue in cues_out:
        if len(cue["lines"]) > 2:
            max_lines_violation += 1

        if any(len(line) > 32 for line in cue["lines"]):
            max_chars_violation += 1

        duration = cue["end_ms"] - cue["start_ms"]
        if cue["type"] == "sound":
            if duration < 800:
                short_duration_violations += 1
        else:
            if duration < 800:
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
    }
