"""
Line-break intelligence — the deterministic "Universal Caption Builder" brain.

WHY THIS MODULE EXISTS
──────────────────────
Greedy left-to-right fill (the old wrap_text behaviour) packs line 1 to the
character brim, then dumps the rest on line 2. That satisfies max_chars but
routinely produces breaks a senior caption editor would reject:

    BAD  (greedy):                 BETTER (balanced + clause-aware):
    I went to the store because    I went to the store
    I needed milk.                 because I needed milk.

    The United                     The United States
    States government              government

This module makes the SINGLE break decision for a one-line-too-long string in
the common 2-line case. It is pure, deterministic, and parameterised only by
spec flags the ClosedCaptionSpec already declares:

    CUSTOM_PREFER_BALANCED_LINES   (default 1 / true)  → reward balanced lengths
    CUSTOM_PRESERVE_CLAUSE_INTEGRITY (default 1 / true)→ break on clause/phrase
                                                          boundaries, never mid-phrase

NO AI. NO I/O. Identical (text, max_chars) always returns the identical break.
SOC 2 CC8.1 — every line-break decision is reproducible and auditor-defensible.

CONTRACT: choose_two_line_break(text, max_chars) returns either
  • [line1, line2]  — the chosen split (both within max_chars), or
  • None            — no two-line split fits; caller falls back to greedy.

The caller (rendering.wrap_text) owns max_lines handling, speaker labels, and
the >2-line fallback. This module ONLY answers "where is the best single break
for two lines?" — the highest-frequency, highest-impact editorial decision.
"""

import os
import re
from typing import List, Optional


# ─── Spec flags (read once per call; cheap) ──────────────────────────
def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _prefer_balanced() -> bool:
    return _flag("CUSTOM_PREFER_BALANCED_LINES", True)


def _preserve_clause() -> bool:
    return _flag("CUSTOM_PRESERVE_CLAUSE_INTEGRITY", True)


# ─── Linguistic fact tables (baked-in English; NOT per-job knobs) ────
# These are linguistic constants, identical to how segmentation.py bakes in the
# abbreviation set. A spec NEVER overrides what an article or conjunction is.

# Words that must LEAD the next line, not trail the current one. Breaking right
# BEFORE these is good (they start the continuation); leaving them dangling at
# the end of a line is the classic "weak line ending" defect.
_LEADING_WORDS = {
    # Articles
    "a", "an", "the",
    # Coordinating + common subordinating conjunctions
    "and", "but", "or", "nor", "for", "so", "yet",
    "because", "although", "though", "while", "whereas", "since", "unless",
    "until", "if", "when", "whenever", "where", "wherever", "after", "before",
    "as", "that", "than", "whether",
    # Prepositions (high-frequency)
    "of", "to", "in", "on", "at", "by", "with", "from", "into", "onto",
    "upon", "about", "over", "under", "through", "between", "among",
    "during", "without", "within", "toward", "towards", "against",
    # Prepositions (second tier — previously missing; the '…25 laps around |
    # this place' defect: 'around' wasn't listed, so nothing penalized
    # stranding it at a line/cue end or splitting inside its phrase).
    # Deliberately EXCLUDES phrasal-verb particles (off/up/down/out) — those
    # legitimately end a line ('take off', 'sit down').
    "around", "across", "along", "behind", "beside", "besides", "beyond",
    "near", "past", "despite", "except", "inside", "outside", "underneath",
    "via", "amid", "atop",
    # Relative / possessive that bind tightly to what follows
    "which", "who", "whom", "whose",
}

# Possessive / determiner words that bind tightly to the FOLLOWING noun — never
# strand them at the end of a line either.
_DETERMINERS = {
    "my", "your", "his", "her", "its", "our", "their",
    "this", "that", "these", "those", "some", "any", "no", "every", "each",
}

_SENTENCE_PUNCT = (".", "!", "?")
_CLAUSE_PUNCT = (",", ";", ":", "—", "–")

# Strip trailing punctuation for word-class lookup.
_STRIP_RE = re.compile(r"[^\w']+$")


def _bare(word: str) -> str:
    return _STRIP_RE.sub("", (word or "")).strip().lower()


def _line_len(words: List[str], start: int, end: int) -> int:
    """Rendered length of words[start:end] joined by single spaces."""
    if start >= end:
        return 0
    seg = words[start:end]
    return sum(len(w) for w in seg) + (len(seg) - 1)


def _score_break(words: List[str], i: int, max_chars: int) -> Optional[float]:
    """
    Score a candidate break BETWEEN words[i-1] and words[i] (i.e. line 1 =
    words[0:i], line 2 = words[i:]). Higher is better. Returns None when the
    break is infeasible (either line exceeds max_chars).

    Scoring (all additive, deterministic):
      + balance       : reward small |len(line1) - len(line2)|
      + clause punct  : big reward if line 1's last word ends in , ; : — (break
                        AT a clause boundary reads naturally)
      + leading word  : reward if line 2's FIRST word is a conjunction/prep/
                        article (it correctly LEADS the continuation)
      - weak ending   : penalize if line 1's LAST word is an article / prep /
                        conjunction / determiner left dangling (the weak-ending
                        defect) — only when preserve_clause is on
    """
    n = len(words)
    len1 = _line_len(words, 0, i)
    len2 = _line_len(words, i, n)
    if len1 == 0 or len2 == 0:
        return None
    if len1 > max_chars or len2 > max_chars:
        return None

    score = 0.0

    # Balance — the further the two lines are from equal, the worse. Weight is
    # modest so it never overrides a clean clause break, but decisive between
    # otherwise-equal candidates. Disabled when the spec turns balance off.
    if _prefer_balanced():
        score -= abs(len1 - len2) * 1.0

    last_word = words[i - 1]
    first_word = words[i]
    last_bare = _bare(last_word)
    first_bare = _bare(first_word)

    if _preserve_clause():
        # Reward breaking right AFTER sentence/clause punctuation. A SENTENCE
        # boundary (. ! ?) is the STRONGEST possible line ending — a complete
        # grammatical unit — so it must outrank a mid-sentence CLAUSE boundary
        # (, ; :). The previous ordering (clause 30 > sentence 20) inverted this
        # and produced breaks like "Good work today," / "Hart. You've helped a
        # lot." — orphaning "Hart." from its own sentence. Ending a line on a
        # finished sentence and starting the next line with the next sentence is
        # the professional-captioner default. Sentence 32 > clause 28 so a full
        # stop always wins when both are feasible, but a lone clause break is
        # still strongly preferred over a mid-phrase break.
        if last_word.endswith(_SENTENCE_PUNCT):
            score += 32.0
        elif last_word.endswith(_CLAUSE_PUNCT):
            score += 28.0

        # Reward line 2 LEADING with a conjunction/preposition/article — the
        # canonical "I went to the store / because I needed milk" fix.
        if first_bare in _LEADING_WORDS:
            score += 15.0

        # Penalize a weak line-1 ending: a dangling article / prep / conjunction
        # / determiner that should have led the next line instead.
        if last_bare in _LEADING_WORDS or last_bare in _DETERMINERS:
            score -= 25.0

    return score


def choose_two_line_break(text: str, max_chars: int) -> Optional[List[str]]:
    """
    Decide the best single break that splits `text` into TWO lines, each within
    max_chars, honoring clause integrity + balance. Returns [line1, line2] or
    None if no feasible two-line split exists (caller falls back to greedy).

    Deterministic: scans every inter-word gap, scores each feasible one, and
    returns the highest-scoring split. Ties broken by the EARLIER break (stable,
    reproducible). For a string that fits on one line this returns None (no
    split needed — the caller short-circuits before calling us).
    """
    text = (text or "").strip()
    if not text:
        return None

    words = text.split()
    if len(words) < 2:
        return None  # can't split a single token across two lines

    best_i: Optional[int] = None
    best_score = float("-inf")

    # Candidate break i means line1=words[:i], line2=words[i:], for i in 1..n-1.
    for i in range(1, len(words)):
        s = _score_break(words, i, max_chars)
        if s is None:
            continue
        # Strict > keeps the EARLIEST best break on ties → stable output.
        if s > best_score:
            best_score = s
            best_i = i

    if best_i is None:
        return None

    line1 = " ".join(words[:best_i])
    line2 = " ".join(words[best_i:])
    return [line1, line2]
