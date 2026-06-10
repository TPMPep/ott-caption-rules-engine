"""
Sentence segmentation — the professional-grade linguistic core of cue building.

WHY THIS EXISTS
───────────────
A broadcast caption engine (Iyuno / Pixelogic / Zoo / Deluxe grade) segments
spoken text into captions on TRUE SENTENCE boundaries first, then fits whole
sentences to the spec's line/CPS budget — splitting WITHIN a sentence only at
clause boundaries, and NEVER orphaning a trailing word onto its own flashing
cue.

The legacy cue builder used a naive `[.!?]$` test on the previous word to
decide where a sentence ended. That fires on the period after an abbreviation
("Mr.", "Dr.", "St."), so "Good afternoon. You must be Mr. Wang." was split as:
    cue 1:  Good afternoon. You must be Mr.
    cue 2:  Wang.                              ← orphan, flashes on screen
The correct, industry-standard output is:
    cue 1:  Good afternoon.
    cue 2:  You must be Mr. Wang.

This module is the single source of truth for:
  1. ABBREVIATION-AWARE sentence-end detection (never breaks "Mr. Wang").
  2. Grouping a flat word-token stream into per-speaker SENTENCE GROUPS that
     are the atomic unit cues are built from.

Pure functions only — no env reads, no I/O. Deterministic and auditor-defensible:
the abbreviation set is a fixed linguistic fact of English, baked in (NOT a
per-job knob), so segmentation behaviour can never vary between jobs or clients.
SOC 2 CC8.1 — identical input always yields identical sentence boundaries.
"""

import re
from typing import Any, Dict, List, Optional

# CJK awareness — the single source of truth for no-space-script handling.
try:
    from .cjk import is_cjk_text, CJK_SENTENCE_ENDERS
except Exception:  # pragma: no cover — defensive for alternate import roots
    from cjk import is_cjk_text, CJK_SENTENCE_ENDERS

# ─── Canonical broadcast abbreviation set ────────────────────────────
# Fixed, baked-in. A period immediately following any of these tokens is an
# abbreviation period, NOT a sentence terminator. This is the standard set
# every professional captioning house treats as non-breaking. Stored
# lower-cased + stripped of the trailing dot for O(1) lookup.
_ABBREVIATIONS = {
    # Titles / honorifics
    "mr", "mrs", "ms", "dr", "prof", "rev", "fr", "sr", "jr", "st",
    "messrs", "mmes", "mlle",
    # Geographic / address
    "mt", "ave", "blvd", "rd", "ln", "ste", "apt", "dept", "fl",
    # Business / org
    "inc", "ltd", "co", "corp", "llc", "plc",
    # Common Latin / misc abbreviations
    "vs", "etc", "no", "al", "approx", "est", "min", "max",
    # Month / day abbreviations
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec", "mon", "tue", "wed", "thu", "fri", "sat", "sun",
}

_SENTENCE_END_CHARS = (".", "!", "?")
# A single capital letter optionally followed by a dot — an initial like
# "J." in "J. Smith". Treated as an abbreviation (never a sentence end).
_INITIAL_RE = re.compile(r"^[A-Z]$")
# Decimal number guard: "3.5" must not be treated as a sentence end. We detect
# this when the token ends in a digit-dot-digit pattern, but since tokens are
# whitespace-split words, a pure decimal like "3.5" never ends in a bare dot.
# An ellipsis ("...") is a continuation, not a hard stop.
_ELLIPSIS_RE = re.compile(r"\.\.\.$")


def _strip_trailing_sentence_punct(word: str) -> str:
    """Return the word with trailing sentence/closing punctuation removed,
    for abbreviation matching. Keeps internal dots (e.g. 'U.S.')."""
    return word.rstrip(".!?\"')]}»”’").strip()


def is_sentence_end(word: str, next_word: Optional[str] = None) -> bool:
    """
    Decide whether `word` ends a sentence, abbreviation-aware.

    Returns False (NOT a sentence end) when:
      • the word does not end with . ! or ?
      • the word is an ellipsis ("...") — a continuation
      • the bare token (sans punctuation) is a known abbreviation ("Mr", "Dr")
      • the bare token is a single-letter initial ("J", "A")
      • the word ends with a dot AND the next word begins lower-case AND the
        bare token is a known abbreviation (defensive double-check)

    Returns True only on a genuine sentence terminator.
    """
    w = (word or "").strip()
    if not w:
        return False

    # CJK sentence terminators (。！？) end a sentence with no abbreviation
    # ambiguity — there is no "Mr." problem in Japanese. Check these FIRST so a
    # Japanese token ending in 。 is always a sentence boundary even though it
    # never matches the ASCII _SENTENCE_END_CHARS test below.
    if w[-1] in ("。", "！", "？", "．"):
        return True

    # Must end with a terminator at all.
    if not w.endswith(_SENTENCE_END_CHARS):
        return False

    # Ellipsis is a continuation, never a hard stop.
    if _ELLIPSIS_RE.search(w):
        return False

    # Only a trailing '.' can be an abbreviation; '!' and '?' are always stops.
    if w.endswith("."):
        bare = _strip_trailing_sentence_punct(w)
        # Internal-dot abbreviations like "U.S." or "a.m." — if removing the
        # final dot still leaves an internal dot, it's an abbreviation.
        if "." in bare:
            return False
        token = bare.lower()
        if token in _ABBREVIATIONS:
            return False
        if _INITIAL_RE.match(bare):
            return False

    return True


def _explode_cjk_tokens(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Explode any CJK token carrying multiple 。-delimited sentences into one
    sub-token per sentence, interpolating timings by character count over the
    token's real window. Non-CJK tokens (and CJK tokens with ≤1 sentence) pass
    through unchanged. This is what lets the existing per-token sentence-end
    grouping produce one group per Japanese sentence instead of one giant group.
    """
    from .cjk import split_cjk_into_sentences  # local import (CI: no top-level cross-import drift)

    out: List[Dict[str, Any]] = []
    for token in tokens:
        text = (token.get("text") or "").strip()
        if not text or not is_cjk_text(text):
            out.append(token)
            continue
        sentences = split_cjk_into_sentences(text)
        if len(sentences) <= 1:
            out.append(token)
            continue
        start = int(token.get("start_ms", 0) or 0)
        end = int(token.get("end_ms", start) or start)
        span = max(1, end - start)
        total_chars = sum(len(s) for s in sentences) or 1
        cursor = 0
        for s in sentences:
            s_start = start + (span * cursor) // total_chars
            cursor += len(s)
            s_end = start + (span * cursor) // total_chars
            out.append({
                "text": s,
                "start_ms": int(s_start),
                "end_ms": int(max(s_end, s_start + 1)),
                "speaker": token.get("speaker"),
            })
    return out


def segment_into_sentence_groups(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group a flat word-token stream into SENTENCE GROUPS — the atomic unit the
    cue builder packs into captions.

    A new group starts when EITHER:
      • the speaker changes (a new speaker is always a new caption group), OR
      • the previous word was a genuine (abbreviation-aware) sentence end.

    Each returned group is:
      {
        "words":      [str, ...],          # the word texts in order
        "start_ms":   int,                 # first word start
        "end_ms":     int,                 # last word end
        "speaker_runs": [ {speaker, word_start}, ... ],  # per-speaker offsets
      }

    `speaker_runs` preserves the exact multi-speaker structure the downstream
    one-speaker-per-line renderer needs, identical to the legacy contract.

    Note on multi-speaker groups: because a speaker change always opens a new
    group, a single group can only contain ONE speaker in practice. The
    speaker_runs list therefore normally has length 1 — but we keep the
    structure so the renderer contract is unchanged and a future "merge tiny
    adjacent speakers" rule can populate it.
    """
    # CJK PRE-SPLIT: Scribe/AAI emit Japanese as a few enormous tokens, each
    # carrying many 。-delimited sentences with no spaces. The Latin grouping
    # loop below keys off per-token sentence-end detection, so a single mega-
    # token would collapse the whole utterance into ONE group (the 66-mega-cue
    # bug). We first explode any CJK token into sentence-level sub-tokens, with
    # timings interpolated proportionally to character count over the token's
    # real [start_ms, end_ms]. Each sub-token then ends in 。！？ so the existing
    # sentence-end logic opens a new group per sentence — exactly the behaviour
    # the Latin path already has. Non-CJK tokens pass through untouched, so the
    # English path is byte-identical. SOC 2 CC8.1 — deterministic, reproducible.
    tokens = _explode_cjk_tokens(tokens)

    groups: List[Dict[str, Any]] = []
    cur_words: List[str] = []
    cur_start: Optional[int] = None
    cur_end: int = 0
    cur_speaker: Optional[str] = None
    cur_runs: List[Dict[str, Any]] = []

    def _flush() -> None:
        nonlocal cur_words, cur_start, cur_end, cur_runs
        if cur_words:
            groups.append({
                "words": cur_words,
                "start_ms": cur_start if cur_start is not None else 0,
                "end_ms": cur_end,
                "speaker_runs": cur_runs,
            })
        cur_words = []
        cur_start = None
        cur_runs = []

    for token in tokens:
        text = (token.get("text") or "").strip()
        if not text:
            continue
        speaker = token.get("speaker")
        start_ms = int(token.get("start_ms", 0) or 0)
        end_ms = int(token.get("end_ms", start_ms) or start_ms)

        speaker_changed = cur_speaker is not None and speaker != cur_speaker
        prev_sentence_ended = bool(cur_words) and is_sentence_end(cur_words[-1])

        if (speaker_changed or prev_sentence_ended) and cur_words:
            _flush()

        if cur_start is None:
            cur_start = start_ms

        if speaker != cur_speaker:
            cur_speaker = speaker
            cur_runs.append({"speaker": speaker, "word_start": len(cur_words)})

        cur_words.append(text)
        cur_end = end_ms

    _flush()
    return groups
