"""
CJK (Chinese / Japanese / Korean) text awareness — the single source of truth
for every place the caption engine must NOT assume space-delimited Latin text.

WHY THIS MODULE EXISTS
──────────────────────
The entire formatter was built for space-delimited Latin script. Every text
operation used `text.split()` (split on spaces) and `len(" ".join(words))`.
Japanese has NO spaces and uses 。！？ as sentence terminators and 、 as the
clause separator. On a Japanese transcript this produced:

  • segmentation.is_sentence_end never fired (looked only for . ! ?), so the
    WHOLE transcript collapsed into ~1 sentence group per speaker run.
  • text.split() returned ONE giant "word" (no spaces), so wrapping could not
    break it → 54-char lines.
  • CPS/char counting measured len() of a string with no spaces, mixing the
    spec's character convention with the wrong tokenization.

Result: 66 mega-cues with oversized lines instead of the ~191 short,
speech-timed cues a professional Japanese caption file carries.

This module is the ONE place that knows about CJK. Every existing stage calls
into it through a tiny, surgical hook so the Latin path stays byte-identical
(zero regression risk) and CJK becomes a first-class, auditable transform.

Pure functions only — no env writes, no I/O. Deterministic. SOC 2 CC8.1:
identical (text, spec) always produces identical segmentation + wrapping.

KINSOKU (禁則処理)
─────────────────
Japanese line-breaking has hard rules: certain characters may NOT start a line
(closing brackets, small kana, sentence-final punctuation) and certain
characters may NOT end a line (opening brackets). We honor the high-frequency
subset that matters for captions — never break right before 。、！？」』）, and
never leave an opening 「『（ dangling at a line end.
"""

from typing import List

# ─── Script detection ────────────────────────────────────────────────
# Unicode ranges that are written WITHOUT inter-word spaces. If a meaningful
# fraction of a string's characters fall in these ranges, the string is CJK
# and must be tokenized/wrapped per-character, not per-space-word.
#   CJK Unified Ideographs        4E00–9FFF  (Kanji / Hanzi)
#   CJK Ext-A                     3400–4DBF
#   Hiragana                      3040–309F
#   Katakana                      30A0–30FF
#   Hangul syllables              AC00–D7AF
#   Halfwidth/Fullwidth forms     FF00–FFEF  (fullwidth punctuation)
#   CJK symbols & punctuation     3000–303F  (。、「」『』（） etc.)
def _is_cjk_char(ch: str) -> bool:
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF or
        0x3400 <= o <= 0x4DBF or
        0x3040 <= o <= 0x309F or
        0x30A0 <= o <= 0x30FF or
        0xAC00 <= o <= 0xD7AF or
        0x3000 <= o <= 0x303F or
        0xFF00 <= o <= 0xFFEF
    )


def is_cjk_text(text: str) -> bool:
    """True when the text is predominantly CJK (no-space script). We sample the
    non-whitespace characters and treat the text as CJK when ≥30% are CJK — a
    deliberately low bar so a mostly-Japanese line with a few Latin loanwords
    ('ヨガ', 'OK') still routes through the CJK path. ASCII-only text is never
    CJK. Empty → False (Latin path)."""
    if not text:
        return False
    cjk = 0
    total = 0
    for ch in text:
        if ch.isspace():
            continue
        total += 1
        if _is_cjk_char(ch):
            cjk += 1
    if total == 0:
        return False
    return (cjk / total) >= 0.30


# ─── Sentence + clause punctuation ───────────────────────────────────
# Fullwidth (CJK) AND halfwidth forms, so a transcript that mixes 。 with a
# stray ASCII '.' is handled either way.
CJK_SENTENCE_ENDERS = ("。", "！", "？", "．", "!", "?", ".")
CJK_CLAUSE_ENDERS = ("、", "，", "；", "：", "・", ",", ";", ":")

# Kinsoku: characters that may NOT start a line (line-leading prohibited).
# Breaking right BEFORE these would orphan a terminator/closer at a line head.
_NO_LINE_START = set(
    "。、！？．，；：・）｝］」』〕〉》】〙〗｠"
    "ぁぃぅぇぉっゃゅょゎ"          # small hiragana
    "ァィゥェォッャュョヮ"          # small katakana
    "ー"                           # prolonged sound mark
    "!?,.:;)]}"                   # halfwidth equivalents
    "」』"
)
# Kinsoku: characters that may NOT end a line (line-trailing prohibited) —
# opening brackets that must stay attached to what follows.
_NO_LINE_END = set("（｛［「『〔〈《【〘〖｟([{")


def cjk_char_count(text: str) -> int:
    """Visible character count for CPS / max-chars on CJK text. We count every
    non-space character (Japanese captions count by character, including
    punctuation, and carry no inter-word spaces). Spaces — which only appear
    where the transcript inserted artificial gaps — are not counted."""
    return sum(1 for ch in (text or "") if not ch.isspace())


def split_cjk_into_sentences(text: str) -> List[str]:
    """Split a CJK string into sentences on 。！？ (keeping the terminator with
    its sentence). Used by the CJK segmentation path. A trailing fragment with
    no terminator is returned as its own sentence so nothing is dropped."""
    out: List[str] = []
    cur = []
    for ch in (text or ""):
        cur.append(ch)
        if ch in CJK_SENTENCE_ENDERS:
            s = "".join(cur).strip()
            if s:
                out.append(s)
            cur = []
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return out


def _best_cjk_break(s: str, max_chars: int) -> int:
    """Pick the break index for a CJK string longer than max_chars. Prefer the
    LAST clause/sentence boundary (、。！？等) at or before max_chars so the line
    ends on a natural pause; otherwise break at max_chars. Honor kinsoku:
    never put a no-line-start char at the head of line 2, and never leave a
    no-line-end char at the tail of line 1 — nudge the break by one when
    needed. Returns an index in 1..len(s)-1."""
    n = len(s)
    limit = min(max_chars, n - 1)
    if limit < 1:
        return max(1, n - 1)

    # Prefer the last clause/sentence punctuation within the window. The break
    # goes AFTER the punctuation (so it ends line 1), i.e. index = pos + 1.
    best = -1
    for i in range(limit):
        if s[i] in CJK_CLAUSE_ENDERS or s[i] in CJK_SENTENCE_ENDERS:
            best = i + 1
    idx = best if best >= 1 else limit

    # Kinsoku nudge: if the char that would START line 2 is line-start-
    # prohibited, push the break one char later so it rides on line 1.
    guard = 0
    while idx < n and s[idx] in _NO_LINE_START and guard < 8:
        idx += 1
        guard += 1
    # Kinsoku nudge: if the char that would END line 1 is line-end-prohibited
    # (an opening bracket), pull the break one char earlier.
    guard = 0
    while idx > 1 and s[idx - 1] in _NO_LINE_END and guard < 8:
        idx -= 1
        guard += 1

    return max(1, min(idx, n - 1))


def wrap_cjk(text: str, max_chars: int, max_lines: int) -> List[str]:
    """Wrap a CJK string into ≤max_lines lines of ≤max_chars characters each,
    breaking at 、。 clause/sentence boundaries where possible and honoring
    kinsoku. Pure character-count wrapping — never uses spaces. If the text
    cannot fit in max_lines×max_chars, the final line absorbs the remainder
    (the CPS/split passes upstream are responsible for keeping cues short
    enough that this is rare; QC still grades the result honestly)."""
    s = (text or "").strip()
    if not s:
        return [""]
    if cjk_char_count(s) <= max_chars:
        return [s]

    lines: List[str] = []
    remaining = s
    while remaining and len(lines) < max_lines - 1:
        idx = _best_cjk_break(remaining, max_chars)
        line = remaining[:idx].strip()
        remaining = remaining[idx:].strip()
        if line:
            lines.append(line)
        if not remaining:
            break
    if remaining:
        lines.append(remaining)
    return lines[:max_lines] if lines else [s]
