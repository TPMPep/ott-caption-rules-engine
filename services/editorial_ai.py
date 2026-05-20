"""
Editorial AI — GPT-powered caption refinement.

UPDATED: All profile helpers now read env vars unconditionally.
The frontend sends correct values for each profile (NBCU=2/32, custom=user values).
The GPT prompt is now dynamic based on SPEAKER_LABEL_MODE and other env vars.
"""

import json
import os
import re
from typing import Any, Dict, List

_WORD_RE = re.compile(r"\b[\w']+\b")
_STYLE_TAG_RE = re.compile(r"\{\+an\d\}")
_ITALIC_TAG_RE = re.compile(r"</?i>")
_WEAK_ENDS = {"a", "an", "the", "of", "to", "and", "or", "but", "with", "from", "in", "on", "at", "for", "that"}


# ─── Profile Helpers (UPDATED: always read env vars) ────────────────

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
    """Always read from env. NBCU default = 2."""
    return _env_int("CUSTOM_MAX_LINES", 2)


def _max_chars() -> int:
    """Always read from env. NBCU default = 32."""
    return _env_int("CUSTOM_MAX_CHARS", 32)


def _speaker_label_mode() -> str:
    """Read speaker label mode from env. NBCU default = dash."""
    return (os.getenv("SPEAKER_LABEL_MODE", "") or "dash").strip().lower()


# ─── Build Dynamic System Prompt ────────────────────────────────────

def _build_system_prompt(max_lines: int, max_chars: int, speaker_mode: str) -> str:
    """
    Build the GPT system prompt dynamically based on profile settings.
    """
    # Base rules (always apply)
    prompt_parts = [
        "You are a broadcast closed-caption editorial assistant. "
        "Output must be suitable for any media (TV, streaming).",

        "Do not add, remove, replace, or reorder words.",
        "You may only change capitalization, punctuation, and line breaks.",

        "Capitalization: capitalize only the first word of a true sentence and proper nouns "
        "(names, titles, I). Do not capitalize a word just because it follows a comma or "
        "starts a new caption line.",

        "Punctuation (critical): use commas where the sentence or thought continues; "
        "use periods only at a real sentence stop. "
        "Only change a period to a comma when the next dialogue clearly continues the same "
        "thought (e.g. next starts with a lowercase continuation word).",

        "If this caption starts with a word that continues prev_dialogue "
        "(e.g. it's, well, and, but, so, then, where, really), output that word lowercased.",

        "When splitting into two lines, avoid a single word on the second line unless it is "
        "a brief response (Yes, No, OK, Yeah, Right).",

        "Prefer splitting at phrase or clause boundaries.",
        "Do not split protected phrases across lines.",

        "Avoid ending a line with weak function words "
        "(a, an, the, of, to, and, or, but, with, from, in, on, at, for, that) unless unavoidable.",
    ]

    # Speaker-mode-specific rules
    if speaker_mode == "dash":
        prompt_parts.append(
            "If there are exactly two speaker runs, you MUST output exactly two lines "
            "and begin each line with '- ' (dash space)."
        )
    elif speaker_mode == "alpha":
        prompt_parts.append(
            "If there are multiple speaker runs, prefix each speaker's line with their "
            "letter label followed by a colon (e.g. 'A: Hello' / 'B: Hi there')."
        )
    elif speaker_mode == "generic":
        generic_prefix = os.getenv("SPEAKER_GENERIC_PREFIX", "SPEAKER") or "SPEAKER"
        prompt_parts.append(
            f"If there are multiple speaker runs, prefix each speaker's line with "
            f"'{generic_prefix} N:' where N is the speaker number."
        )
    elif speaker_mode == "named":
        prompt_parts.append(
            "If there are multiple speaker runs, prefix each speaker's line with "
            "the speaker's name followed by a colon."
        )

    # Line/char limits
    prompt_parts.append(
        f"If text fits in one line (≤{max_chars} characters), output one line only; "
        f"{max_lines} lines is the max, not required."
    )

    # Output format
    prompt_parts.append(
        'Return JSON only with the shape {"lines":["...","..."]}.'
    )

    return " ".join(prompt_parts)


# ─── Main Refinement Function ───────────────────────────────────────

def editorial_refine_cues(cues: List[Dict[str, Any]], protected_phrases: List[str]) -> List[Dict[str, Any]]:
    """
    Optional AI editorial pass.
    Improves: punctuation/capitalization, phrase-aware line breaks, speaker formatting.

    Hard rule: never accept an AI rewrite that changes the underlying words
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

    max_lines = _max_lines()
    max_chars = _max_chars()
    speaker_mode = _speaker_label_mode()

    system_prompt = _build_system_prompt(max_lines, max_chars, speaker_mode)

    refined: List[Dict[str, Any]] = []
    total = len(cues)

    for idx, cue in enumerate(cues):
        if cue.get("type") != "dialogue":
            refined.append(cue)
            continue

        runs = cue.get("meta", {}).get("runs", [])
        dialogue_text = cue.get("meta", {}).get(
            "dialogue_text", " ".join(cue.get("lines", []))
        ).strip()
        if not dialogue_text:
            refined.append(cue)
            continue

        # Context: previous and next cue text
        prev_text = ""
        next_text = ""
        if idx > 0 and cues[idx - 1].get("type") == "dialogue":
            prev_text = cues[idx - 1].get("meta", {}).get(
                "dialogue_text", " ".join(cues[idx - 1].get("lines", []))
            ).strip()
        if idx < total - 1 and cues[idx + 1].get("type") == "dialogue":
            next_text = cues[idx + 1].get("meta", {}).get(
                "dialogue_text", " ".join(cues[idx + 1].get("lines", []))
            ).strip()

        # Build speaker formatting rule based on mode
        two_speaker_rule = True  # default for dash mode
        if speaker_mode != "dash":
            two_speaker_rule = False

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
                "speaker_mode": speaker_mode,
                "two_speaker_lines_must_start_with_dash": two_speaker_rule,
                "preserve_words_exactly": True,
                "fix_punctuation_and_capitalization_only": True,
                "prefer_phrase_and_punctuation_boundaries": True,
                "avoid_weak_function_word_line_endings": True,
            },
        }

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

        # Validate exact same words (ignore punctuation and case)
        original_words = _word_fingerprint(dialogue_text)
        ai_words = _word_fingerprint(" ".join(_strip_dashes(ai_lines)))
        if original_words != ai_words:
            refined.append(cue)
            continue

        # Check line/char limits
        if len(ai_lines) > max_lines or any(_visible_len(line) > max_chars for line in ai_lines):
            refined.append(cue)
            continue

        # Speaker-mode validation
        if speaker_mode == "dash" and len(runs) == 2:
            if len(ai_lines) != 2 or not all(line.startswith("- ") for line in ai_lines):
                refined.append(cue)
                continue

        new_cue = dict(cue)
        new_cue["lines"] = ai_lines
        refined.append(new_cue)

    return refined


# ─── Helpers ────────────────────────────────────────────────────────

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
