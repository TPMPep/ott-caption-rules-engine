import json
import os
import re
from typing import Dict, List, Optional

MAX_LINES = 2
MAX_CHARS = 32
_WORD_RE = re.compile(r"\b[\w']+\b")
_SENT_BOUNDARY_RE = re.compile(r"[.!?]\s*$")
_WEAK_WORDS = {
    "is", "to", "of", "and", "or", "but", "for", "with", "a", "an", "the",
    "that", "this", "these", "those", "in", "on", "at", "from", "by",
}
_FRAGMENT_PATTERNS = {
    "and i", "and i had", "it’s", "it's", "who made", "did it", "do you",
    "from below", "speaking of", "by the way",
}


def editorial_refine_cues(cues: List[Dict], protected_phrases: List[str]) -> List[Dict]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return cues
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception:
        return cues

    refined: List[Dict] = []
    total = len(cues)
    for idx, cue in enumerate(cues):
        if cue.get("type") != "dialogue":
            refined.append(cue)
            continue

        runs = cue.get("meta", {}).get("runs", []) or []
        dialogue_text = cue.get("meta", {}).get("dialogue_text", " ".join(cue.get("lines", []))).strip()
        if not dialogue_text:
            refined.append(cue)
            continue

        prev_text = ""
        next_text = ""
        if idx > 0 and cues[idx - 1].get("type") == "dialogue":
            prev_text = cues[idx - 1].get("meta", {}).get("dialogue_text", " ".join(cues[idx - 1].get("lines", []))).strip()
        if idx < total - 1 and cues[idx + 1].get("type") == "dialogue":
            next_text = cues[idx + 1].get("meta", {}).get("dialogue_text", " ".join(cues[idx + 1].get("lines", []))).strip()

        if len(runs) == 2 and cue.get("meta", {}).get("two_speaker"):
            new_cue = _refine_two_speaker_cue(client, cue, runs, protected_phrases)
        else:
            new_cue = _refine_single_speaker_cue(client, cue, dialogue_text, prev_text, next_text, protected_phrases)

        refined.append(new_cue if new_cue is not None else cue)
    return refined


def _refine_two_speaker_cue(client, cue: Dict, runs: List[Dict], protected_phrases: List[str]):
    left = (runs[0].get("text") or "").strip()
    right = (runs[1].get("text") or "").strip()
    if not left or not right:
        return None
    if len(left) > MAX_CHARS - 2 or len(right) > MAX_CHARS - 2:
        return None

    payload = {
        "speaker_run_1": left,
        "speaker_run_2": right,
        "protected_phrases": protected_phrases[:50],
        "rules": {
            "format": ["- speaker 1 text", "- speaker 2 text"],
            "max_lines": 2,
            "max_chars_per_line": MAX_CHARS,
            "preserve_each_run_exactly": True,
            "do_not_merge_or_reorder_runs": True,
            "fix_punctuation_and_capitalization_only": True,
        },
    }
    try:
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            temperature=0,
            input=[
                {"role": "system", "content": (
                    "You are a broadcast closed-caption editorial assistant. "
                    "You are formatting EXACTLY TWO speaker turns. "
                    "Return exactly two lines. Each line must begin with '- '. "
                    "Line 1 must contain ONLY speaker_run_1. Line 2 must contain ONLY speaker_run_2. "
                    "Do not add, remove, replace, reorder, or transfer words between speakers. "
                    "Keep each speaker turn atomic. Never split one speaker's phrase across the other speaker's line. "
                    "You may only adjust punctuation and capitalization inside each speaker run. "
                    "Do not title-case words just because a new caption line starts. "
                    "Only capitalize a line-start word if it truly starts a sentence, is the pronoun I, or is already a proper noun/title. "
                    "Use commas instead of periods when a phrase continues as an appositive, for example 'I'm your host, Andy Cohen.' "
                    "Return JSON only with the shape {\"lines\":[\"...\",\"...\"]}."
                )},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        data = json.loads(response.output_text.strip())
        ai_lines = _normalize_lines([str(x).strip() for x in data.get("lines", []) if str(x).strip()])
    except Exception:
        return None

    if len(ai_lines) != 2 or not all(line.startswith('- ') for line in ai_lines):
        return None
    if any(len(line) > MAX_CHARS for line in ai_lines):
        return None
    if _word_fingerprint(ai_lines[0][2:]) != _word_fingerprint(left):
        return None
    if _word_fingerprint(ai_lines[1][2:]) != _word_fingerprint(right):
        return None
    if _has_bad_titlecase(left, ai_lines[0][2:]):
        return None
    if _has_bad_titlecase(right, ai_lines[1][2:]):
        return None
    joined = " ".join([ai_lines[0][2:], ai_lines[1][2:]])
    if _has_bad_period_lowercase_sequence(joined):
        return None
    if _looks_fragmentary(ai_lines[0][2:]) or _looks_fragmentary(ai_lines[1][2:]):
        return None

    new_cue = dict(cue)
    new_cue["lines"] = ai_lines
    return new_cue


def _refine_single_speaker_cue(client, cue: Dict, dialogue_text: str, prev_text: str, next_text: str, protected_phrases: List[str]):
    if not dialogue_text.strip():
        return None
    payload = {
        "dialogue_text": dialogue_text,
        "current_lines": cue.get("lines", []),
        "prev_dialogue": prev_text,
        "next_dialogue": next_text,
        "protected_phrases": protected_phrases[:50],
        "rules": {
            "max_lines": MAX_LINES,
            "max_chars_per_line": MAX_CHARS,
            "preserve_words_exactly": True,
            "prefer_phrase_and_punctuation_boundaries": True,
            "do_not_title_case_line_starts": True,
            "only_fix_real_sentence_capitalization": True,
            "fix_obvious_punctuation_only": True,
        },
    }
    try:
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            temperature=0,
            input=[
                {"role": "system", "content": (
                    "You are a broadcast closed-caption editorial assistant. "
                    "Do not add, remove, replace, or reorder words. "
                    "You may only adjust punctuation, capitalization, and line breaks. "
                    "Do NOT capitalize a word just because it starts a new caption line. "
                    "Only capitalize if it truly begins a sentence, is the pronoun I, or is already a proper noun/title. "
                    "Preserve sentence meaning and continuity across neighboring cues. "
                    "Prefer punctuation and phrase boundaries over visual balance. If punctuation is ambiguous and the thought clearly continues, prefer a comma or semicolon rather than a period. "
                    "Use commas rather than periods when the sentence clearly continues, for example 'I'm your host, Andy Cohen.' "
                    "Preserve titles and proper nouns. Do not split proper nouns or show titles awkwardly across lines. "
                    "If the source contains [INAUDIBLE], preserve it exactly as [INAUDIBLE]. "
                    "Return JSON only with the shape {\"lines\":[\"...\",\"...\"]}."
                )},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        data = json.loads(response.output_text.strip())
        ai_lines = _normalize_lines([str(x).strip() for x in data.get("lines", []) if str(x).strip()])
    except Exception:
        return None

    if not ai_lines or len(ai_lines) > MAX_LINES or any(len(line) > MAX_CHARS for line in ai_lines):
        return None

    joined = " ".join(ai_lines)
    if _word_fingerprint(joined) != _word_fingerprint(dialogue_text):
        return None
    if _has_bad_titlecase(dialogue_text, joined):
        return None
    if _bad_initial_capitalization_due_to_continuation(dialogue_text, joined, prev_text):
        return None
    if _has_bad_period_lowercase_sequence(joined):
        return None
    if _has_weak_one_word_line(ai_lines):
        return None
    if _breaks_protected_phrase(ai_lines, protected_phrases):
        return None
    if _looks_fragmentary(joined):
        return None
    if _creates_bad_boundary_with_neighbors(joined, prev_text, next_text):
        return None

    new_cue = dict(cue)
    new_cue["lines"] = ai_lines
    return new_cue


def _word_fingerprint(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def _normalize_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    for line in lines:
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            out.append(line)
    return out[:MAX_LINES]


def _has_bad_period_lowercase_sequence(edited: str) -> bool:
    return bool(re.search(r"[A-Za-z]\.(?:\s+)([a-z])", edited))


def _has_weak_one_word_line(lines: List[str]) -> bool:
    for line in lines:
        words = line.replace("- ", "").split()
        if len(words) == 1 and words[0].lower() in _WEAK_WORDS:
            return True
    return False


def _token_case_list(text: str) -> List[str]:
    return [m.group(0) for m in _WORD_RE.finditer(text)]


def _has_bad_titlecase(original: str, edited: str) -> bool:
    orig_tokens = _token_case_list(original)
    edit_tokens = _token_case_list(edited)
    if len(orig_tokens) != len(edit_tokens):
        return True

    allowed_sentence_starts = set()
    idx = 0
    first = True
    for m in _WORD_RE.finditer(edited):
        if first:
            allowed_sentence_starts.add(idx)
            first = False
        else:
            prefix = edited[:m.start()].rstrip()
            if prefix and prefix[-1] in '.!?':
                allowed_sentence_starts.add(idx)
        idx += 1

    for i, (o, e) in enumerate(zip(orig_tokens, edit_tokens)):
        if o == e:
            continue
        if o.lower() != e.lower():
            return True
        if o.lower() == 'i' and e == 'I':
            continue
        if o[:1].isupper():
            continue
        if i in allowed_sentence_starts and e[:1].isupper():
            continue
        return True
    return False


def _continuation_likely(prev_text: str) -> bool:
    prev_text = (prev_text or "").strip()
    if not prev_text:
        return False
    return not _SENT_BOUNDARY_RE.search(prev_text)


def _bad_initial_capitalization_due_to_continuation(original: str, edited: str, prev_text: str) -> bool:
    orig_tokens = _token_case_list(original)
    edit_tokens = _token_case_list(edited)
    if not orig_tokens or not edit_tokens:
        return False
    if not _continuation_likely(prev_text):
        return False
    o = orig_tokens[0]
    e = edit_tokens[0]
    if o.lower() != e.lower():
        return False
    if o.lower() == "i" and e == "I":
        return False
    if o[:1].islower() and e[:1].isupper():
        return True
    return False


def _breaks_protected_phrase(lines: List[str], protected_phrases: List[str]) -> bool:
    if len(lines) < 2:
        return False
    combined = "\n".join(lines)
    for phrase in protected_phrases:
        p = re.sub(r"\s+", " ", (phrase or "").strip())
        if not p:
            continue
        if p in combined.replace("\n", " ") and p.replace(" ", "\n") in combined:
            return True
    return False


def _looks_fragmentary(text: str) -> bool:
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if not clean:
        return True
    lowered = clean.lower().strip(" ,;:-")
    words = lowered.split()
    if lowered in _FRAGMENT_PATTERNS:
        return True
    if len(words) <= 2 and words[-1] in _WEAK_WORDS:
        return True
    if len(words) <= 3 and not re.search(r"[.!?]$", clean) and words[0] in {"and", "but", "or", "so", "because"}:
        return True
    if clean.endswith((" And", " and", " to", " of", " from", " with")):
        return True
    return False


def _first_word(text: str) -> Optional[str]:
    m = _WORD_RE.search(text or "")
    return m.group(0).lower() if m else None


def _last_word(text: str) -> Optional[str]:
    tokens = _word_fingerprint(text or "")
    return tokens[-1] if tokens else None


def _creates_bad_boundary_with_neighbors(joined: str, prev_text: str, next_text: str) -> bool:
    joined = (joined or "").strip()
    if not joined:
        return True
    first = _first_word(joined)
    last = _last_word(joined)
    if first is None or last is None:
        return True

    if prev_text:
        prev_last = _last_word(prev_text)
        if prev_last and prev_last == first and len(_word_fingerprint(joined)) <= 4:
            return True
    if next_text:
        next_first = _first_word(next_text)
        if next_first and next_first == last and len(_word_fingerprint(joined)) <= 4:
            return True
    return False
