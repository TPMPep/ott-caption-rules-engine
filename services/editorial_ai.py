import json
import os
import re
from typing import Any, Dict, List

_WORD_RE = re.compile(r"\b[\w']+\b")
_STYLE_TAG_RE = re.compile(r"\{\\+an\d\}")
_ITALIC_TAG_RE = re.compile(r"</?i>")
_WEAK_ENDS = {"a", "an", "the", "of", "to", "and", "or", "but", "with", "from", "in", "on", "at", "for", "that"}


def _caption_profile() -> str:
    return (os.getenv("CAPTION_PROFILE", "") or "").strip().lower()


def _max_lines() -> int:
    if _caption_profile() == "custom":
        return int(os.getenv("CUSTOM_MAX_LINES", "2") or 2)
    return 2


def _max_chars() -> int:
    if _caption_profile() == "custom":
        return int(os.getenv("CUSTOM_MAX_CHARS", "32") or 32)
    return 32


def editorial_refine_cues(cues: List[Dict[str, Any]], protected_phrases: List[str]) -> List[Dict[str, Any]]:
    """
    Optional AI editorial pass.
    It may improve:
    - punctuation/capitalization
    - phrase-aware line breaks
    - two-speaker dash formatting

    Hard rule:
    - never accept an AI rewrite that changes the underlying words
      (ignoring case and punctuation).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return cues

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception:
        return cues

    refined: List[Dict[str, Any]] = []
    total = len(cues)
    max_lines = _max_lines()
    max_chars = _max_chars()

    for idx, cue in enumerate(cues):
        if cue.get("type") != "dialogue":
            refined.append(cue)
            continue

        runs = cue.get("meta", {}).get("runs", [])
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

        payload = {
            "dialogue_text": dialogue_text,
            "current_lines": cue.get("lines", []),
            "speaker_runs": runs,
            "prev_dialogue": prev_text,
            "next_dialogue": next_text,
            "protected_phrases": protected_phrases[:50],
            "rules": {
                "max_lines": max_lines,
                "max_chars_per_line": max_chars,
                "two_speaker_lines_must_start_with_dash": True,
                "preserve_words_exactly": True,
                "fix_punctuation_and_capitalization_only": True,
                "prefer_phrase_and_punctuation_boundaries": True,
                "avoid_weak_function_word_line_endings": True
            }
        }

        system_prompt = (
            "You are a broadcast closed-caption editorial assistant. Output must be suitable for any media (TV, streaming). "
            "Do not add, remove, replace, or reorder words. "
            "You may only change capitalization, punctuation, and line breaks. "
            "Capitalization: capitalize only the first word of a true sentence and proper nouns (names, titles, I). "
            "Do not capitalize a word just because it follows a comma or starts a new caption line. "
            "Punctuation (critical): use commas where the sentence or thought continues; use periods only at a real sentence stop. "
            "Only change a period to a comma when the next dialogue clearly continues the same thought (e.g. next starts with a lowercase continuation word). "
            "If this caption starts with a word that continues prev_dialogue (e.g. it's, well, and, but, so, then, where, really), output that word lowercased. "
            "When splitting into two lines, avoid a single word on the second line unless it is a brief response (Yes, No, OK, Yeah, Right). "
            "Prefer splitting at phrase or clause boundaries. "
            "Do not split protected phrases across lines. "
            "Avoid ending a line with weak function words (a, an, the, of, to, and, or, but, with, from, in, on, at, for, that) unless unavoidable. "
            "If there are exactly two speaker runs, you MUST output exactly two lines and begin each line with '- ' (dash space). "
            f"If text fits in one line (≤{max_chars} characters), output one line only; {max_lines} lines is the max, not required. "
            "Return JSON only with the shape "
            '{"lines":["...","..."]}.'
        )

        try:
            response = client.responses.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
                temperature=0,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            )
            raw = response.output_text.strip()
            data = json.loads(raw)
            ai_lines = [str(x).strip() for x in data.get("lines", []) if str(x).strip()]
        except Exception:
            refined.append(cue)
            continue

        ai_lines = _normalize_lines(ai_lines)
        if not ai_lines:
            refined.append(cue)
            continue

        # validate exact same words (ignore punctuation and case)
        original_words = _word_fingerprint(dialogue_text)
        ai_words = _word_fingerprint(" ".join(_strip_dashes(ai_lines)))
        if original_words != ai_words:
            refined.append(cue)
            continue

        if len(ai_lines) > max_lines or any(_visible_len(line) > max_chars for line in ai_lines):
            refined.append(cue)
            continue

        # if 2 speakers, preserve dash requirement
        if len(runs) == 2:
            if len(ai_lines) != 2 or not all(line.startswith("- ") for line in ai_lines):
                refined.append(cue)
                continue

        new_cue = dict(cue)
        new_cue["lines"] = ai_lines
        refined.append(new_cue)

    return refined


def _word_fingerprint(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def _strip_dashes(lines: List[str]) -> List[str]:
    out: List[str] = []
    for line in lines:
        if line.startswith("- "):
            out.append(line[2:])
        else:
            out.append(line)
    return out


def _normalize_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    for line in lines:
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            out.append(line)
    return out[:_max_lines()]


def _visible_len(text: str) -> int:
    text = _STYLE_TAG_RE.sub("", text or "")
    text = _ITALIC_TAG_RE.sub("", text or "")
    return len(text)
