import json
import os
import re
from typing import Dict, List

MAX_LINES = 2
MAX_CHARS = 32
_WORD_RE = re.compile(r"[\w']+")
_SENT_BOUNDARY_RE = re.compile(r"[.!?]\s*$")


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

        if len(runs) == 2:
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
    new_cue = dict(cue)
    new_cue["lines"] = ai_lines
    return new_cue


def _refine_single_speaker_cue(client, cue: Dict, dialogue_text: str, prev_text: str, next_text: str, protected_phrases: List[str]):
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
                    "Prefer punctuation and phrase boundaries over visual balance. "
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
    if _word_fingerprint(" ".join(ai_lines)) != _word_fingerprint(dialogue_text):
        return None
    if _has_bad_titlecase(dialogue_text, " ".join(ai_lines)):
        return None
    if _bad_initial_capitalization_due_to_continuation(dialogue_text, " ".join(ai_lines), prev_text):
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


def _token_case_list(text: str) -> List[str]:
    return [m.group(0) for m in _WORD_RE.finditer(text)]


def _has_bad_titlecase(original: str, edited: str) -> bool:
    """
    Reject edits that randomly title-case mid-sentence words.
    Allowed case changes:
    - i -> I
    - original already capitalized
    - first word of a true sentence after punctuation
    """
    orig_tokens = _token_case_list(original)
    edit_tokens = _token_case_list(edited)
    if len(orig_tokens) != len(edit_tokens):
        return True
    # determine allowed sentence starts from edited text char positions
    allowed_sentence_starts = set()
    idx = 0
    first = True
    for m in _WORD_RE.finditer(edited):
        token = m.group(0)
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
        # otherwise reject random capitalization change
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
