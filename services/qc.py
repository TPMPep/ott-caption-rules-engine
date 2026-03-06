import re

FUNCTION_WORDS = set(["a","an","the","of","to","and","or","but","with","from","in","on","at","for","that","this","these","those"])

def ends_with_function_word(line: str) -> bool:
    line = line.strip().lower()
    if not line:
        return False
    line = re.sub(r"[^\w']+$", "", line)
    parts = line.split()
    return (parts[-1] if parts else "") in FUNCTION_WORDS

def count_function_word_endings(lines):
    return sum(1 for l in lines if ends_with_function_word(l))

def is_one_word(lines):
    txt = " ".join(lines).strip()
    words = re.findall(r"\b\w+\b", txt)
    return len(words) == 1

def violates_line_limits(lines, max_lines=2, max_chars=32):
    if len(lines) > max_lines:
        return True
    return any(len(l) > max_chars for l in lines)

def overlaps(cues):
    overlaps = 0
    for i in range(len(cues)-1):
        if cues[i]["end_ms"] > cues[i+1]["start_ms"]:
            overlaps += 1
    return overlaps

def qc_report(cues_in, cues_out, protected_phrases):
    max_lines_violation = 0
    max_chars_violation = 0
    short_duration = 0
    one_word = 0
    function_word_endings = 0
    protected_splits = 0

    for c in cues_out:
        if len(c["lines"]) > 2:
            max_lines_violation += 1
        if any(len(l) > 32 for l in c["lines"]):
            max_chars_violation += 1

        dur = c["end_ms"] - c["start_ms"]
        if c["type"] == "sound":
            if dur < 800:
                short_duration += 1
        else:
            if dur < 800:
                short_duration += 1
            if is_one_word(c["lines"]):
                one_word += 1

        function_word_endings += count_function_word_endings(c["lines"])

        # protected phrase split check (simple)
        joined = "\n".join(c["lines"])
        for p in protected_phrases or []:
            if p and (p in joined.replace("\n"," ")) and (p.replace(" ","\n") in joined):
                protected_splits += 1

    return {
        "cues_in": cues_in,
        "cues_out": len(cues_out),
        "overlaps": overlaps(cues_out),
        "max_lines_violation": max_lines_violation,
        "max_chars_violation": max_chars_violation,
        "short_duration_violations": short_duration,
        "one_word_dialogue_cues": one_word,
        "function_word_endings": function_word_endings,
        "protected_phrase_splits": protected_splits,
    }
