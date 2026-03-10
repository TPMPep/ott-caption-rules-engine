import json
import os
import re
from typing import Dict, List, Optional

MAX_LINES = 2
MAX_CHARS = 32
WORD_RE = re.compile(r"\b[\w']+\b", re.UNICODE)
SPACE_RE = re.compile(r"\s+")
TERMINAL_PUNCT_RE = re.compile(r"[.!?…]$")
MID_SENTENCE_LOWER_RE = re.compile(r"[A-Za-z]\.(?:\s+)([a-z])")
PROTECTED_TOKEN_RE = re.compile(r"\b(?:[A-Z][a-z]+|[A-Z]{2,}|[Ii])\b")
WEAK_WORDS = {
    "a", "an", "the", "and", "or", "but", "to", "of", "for", "from", "with",
    "in", "on", "at", "is", "are", "was", "were", "be", "been", "being"
}


def editorial_refine_cues(cues: List[Dict], protected_phrases: List[str]) -> List[Dict]:
    client = _get_openai_client()
    refined: List[Dict] = []

    for idx, cue in enumerate(cues):
        if cue.get("type") != "dialogue":
            refined.append(cue)
            continue

        prev_cue = cues[idx - 1] if idx > 0 else None
        next_cue = cues[idx + 1] if idx + 1 < len(cues) else None

        if cue.get("meta", {}).get("two_speaker"):
            new_cue = _refine_two_speaker_cue(cue, protected_phrases, client)
        else:
            new_cue = _refine_single_speaker_cue(cue, prev_cue, next_cue, protected_phrases, client)

        refined.append(new_cue or cue)

    return refined


def _get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def _normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", (text or "")).strip()


def _word_fingerprint(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]


def _true_sentence_start(previous_text: str) -> bool:
    previous_text = _normalize_text(previous_text)
    if not previous_text:
        return True
    return bool(re.search(r"[.!?…]['\")\]]*$", previous_text))


def _split_protected_phrases(lines: List[str], protected_phrases: List[str]) -> bool:
    joined = "\n".join(lines)
    for phrase in protected_phrases or []:
        phrase = _normalize_text(phrase)
        if not phrase or len(phrase.split()) < 2:
            continue
        parts = phrase.split()
        for i in range(1, len(parts)):
            if f"{parts[i-1]}\n{parts[i]}" in joined:
                return True
    return False


def _bad_one_word_line(lines: List[str]) -> bool:
    for line in lines:
        words = [w for w in line.replace("- ", "").split() if w]
        if len(words) == 1 and words[0].lower().strip(".,?!;:") in WEAK_WORDS:
            return True
    return False


def _has_bad_titlecase(original: str, edited: str, prev_text: str) -> bool:
    orig = [m.group(0) for m in WORD_RE.finditer(original or "")]
    new = [m.group(0) for m in WORD_RE.finditer(edited or "")]
    if len(orig) != len(new):
        return True

    sentence_start = _true_sentence_start(prev_text)
    for i, (o, e) in enumerate(zip(orig, new)):
        if o == e:
            sentence_start = bool(re.search(rf"\b{re.escape(e)}\b[^\w]*$", edited[: max(0, edited.find(e) + len(e))]))
            continue
        if o.lower() != e.lower():
            return True
        if o == "i" and e == "I":
            sentence_start = False
            continue
        if o[:1].isupper():
            sentence_start = False
            continue
        if sentence_start and e[:1].isupper():
            sentence_start = False
            continue
        return True
    return False


def _light_punctuation_cleanup(text: str, prev_text: str = "") -> str:
    text = _normalize_text(text)
    if not text:
        return text

    text = text.replace(" ,", ",").replace(" .", ".").replace(" ?", "?").replace(" !", "!")
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    text = re.sub(r"([,.;:?!])(\w)", r"\1 \2", text)
    text = re.sub(r"\s+—\s+", " — ", text)
    text = re.sub(r"\s+([’'])", r"\1", text)
    text = re.sub(r"([\[(])\s+", r"\1", text)
    text = re.sub(r"\s+([\])])", r"\1", text)
    text = re.sub(r"\b([iI])\b", "I", text)

    if _true_sentence_start(prev_text) and text:
        first = re.search(r"\b[\w']+\b", text)
        if first:
            start, end = first.span()
            word = text[start:end]
            if word.lower() == "i":
                rep = "I"
            elif word[:1].islower():
                rep = word[:1].upper() + word[1:]
            else:
                rep = word
            text = text[:start] + rep + text[end:]

    text = re.sub(r"\b([a-z])(?=\s+[A-Z][a-z]+\.$)", lambda m: m.group(1), text)
    return text


def _refine_two_speaker_cue(cue: Dict, protected_phrases: List[str], client=None) -> Optional[Dict]:
    runs = cue.get("meta", {}).get("runs", []) or []
    if len(runs) != 2:
        return None

    left = _light_punctuation_cleanup((runs[0].get("text") or "").strip())
    right = _light_punctuation_cleanup((runs[1].get("text") or "").strip())
    if not left or not right:
        return None

    lines = [f"- {left}", f"- {right}"]
    if all(len(line) <= MAX_CHARS for line in lines):
        new_cue = dict(cue)
        new_cue["lines"] = lines
        return new_cue

    if client is None:
        return None

    payload = {
        "speaker_run_1": left,
        "speaker_run_2": right,
        "protected_phrases": protected_phrases[:50],
    }
    try:
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            temperature=0,
            input=[
                {"role": "system", "content": (
                    "You are formatting exactly two speaker turns for closed captions. "
                    "Return JSON only: {\"lines\":[\"- ...\",\"- ...\"]}. "
                    "Each line must begin with '- '. Do not add, remove, replace, or reorder words. "
                    "Only normalize punctuation and capitalization. Each speaker stays on their own line."
                )},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        data = json.loads(response.output_text.strip())
        ai_lines = [_normalize_text(x) for x in data.get("lines", []) if _normalize_text(str(x))]
    except Exception:
        return None

    if len(ai_lines) != 2 or any(not line.startswith("- ") or len(line) > MAX_CHARS for line in ai_lines):
        return None
    if _word_fingerprint(ai_lines[0][2:]) != _word_fingerprint(left):
        return None
    if _word_fingerprint(ai_lines[1][2:]) != _word_fingerprint(right):
        return None

    new_cue = dict(cue)
    new_cue["lines"] = ai_lines
    return new_cue


def _refine_single_speaker_cue(cue: Dict, prev_cue: Optional[Dict], next_cue: Optional[Dict], protected_phrases: List[str], client=None) -> Optional[Dict]:
    dialogue_text = _normalize_text(cue.get("meta", {}).get("dialogue_text") or " ".join(cue.get("lines", [])))
    if not dialogue_text:
        return None

    prev_text = ""
    if prev_cue and prev_cue.get("type") == "dialogue":
        prev_text = _normalize_text(prev_cue.get("meta", {}).get("dialogue_text") or " ".join(prev_cue.get("lines", [])))

    cleaned = _light_punctuation_cleanup(dialogue_text, prev_text)
    current_lines = [_normalize_text(x) for x in cue.get("lines", []) if _normalize_text(x)]
    fallback_lines = current_lines if current_lines else [cleaned]

    if (
        _word_fingerprint(" ".join(fallback_lines)) == _word_fingerprint(dialogue_text)
        and len(fallback_lines) <= MAX_LINES
        and all(len(line) <= MAX_CHARS for line in fallback_lines)
        and not _bad_one_word_line(fallback_lines)
        and not _split_protected_phrases(fallback_lines, protected_phrases)
    ):
        new_cue = dict(cue)
        new_cue["lines"] = fallback_lines
        return new_cue

    if client is None:
        return None

    next_text = ""
    if next_cue and next_cue.get("type") == "dialogue":
        next_text = _normalize_text(next_cue.get("meta", {}).get("dialogue_text") or " ".join(next_cue.get("lines", [])))

    payload = {
        "dialogue_text": cleaned,
        "current_lines": current_lines,
        "prev_dialogue": prev_text,
        "next_dialogue": next_text,
        "protected_phrases": protected_phrases[:50],
        "rules": {
            "max_lines": MAX_LINES,
            "max_chars_per_line": MAX_CHARS,
            "preserve_words_exactly": True,
            "preserve_order": True,
        },
    }

    try:
        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            temperature=0,
            input=[
                {"role": "system", "content": (
                    "You are a broadcast closed-caption line-break editor. "
                    "Return JSON only: {\"lines\":[\"...\",\"...\"]}. "
                    "Do not add, remove, replace, or reorder words. "
                    "You may only adjust punctuation, capitalization, and line breaks. "
                    "Prefer phrase boundaries. Avoid weak one-word lines. "
                    "Do not capitalize a continuing phrase just because it starts a new line."
                )},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        data = json.loads(response.output_text.strip())
        lines = [_normalize_text(x) for x in data.get("lines", []) if _normalize_text(str(x))]
    except Exception:
        return None

    if not lines or len(lines) > MAX_LINES or any(len(line) > MAX_CHARS for line in lines):
        return None
    if _word_fingerprint(" ".join(lines)) != _word_fingerprint(dialogue_text):
        return None
    if _split_protected_phrases(lines, protected_phrases):
        return None
    if _bad_one_word_line(lines):
        return None
    if MID_SENTENCE_LOWER_RE.search(" ".join(lines)):
        return None
    if _has_bad_titlecase(dialogue_text, " ".join(lines), prev_text):
        return None

    new_cue = dict(cue)
    new_cue["lines"] = lines
    return new_cue
