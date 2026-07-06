"""
Regression tests for the RE-BALANCED, CLAUSE-GUARDED cue splitter in services.shaping.

TWO DEFECTS THIS LOCKS
──────────────────────
1. TINY OVER-CPS SLIVER (re-balance):
   The naive "boundary nearest the middle by word count" could hand a short-text
   half a tiny audio window when speech at that boundary is fast — e.g.
   "It's riboflavin, honey." → 0.9s → 25.9 cps FAIL. The splitter must now reject
   a split that mints an unreadable (< min_display) child and try the next-best
   boundary, so text and window scale together.

2. ORPHANED LEADING FUNCTION WORD (clause guard):
   The clause-boundary picker chose the punctuation break nearest the middle with
   NO word-class guard, so "...the regeneration" / "of glutathione" stranded "of"
   onto the tail cue. The clause path now runs through the same leading-word guard
   the word-fallback path uses, so a preposition/article/conjunction never leads
   the tail cue.

CPL SAFETY-NET (unchanged contract): enforce_cpl_fit must still break a genuinely
over-CPL line even when neither child clears the rhythm window — CPL is a hard FCC
limit. That regression is covered here too.

Deterministic; env knobs set explicitly and restored after each test.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.shaping import (  # noqa: E402
    shape_caption_rhythm,
    enforce_cpl_fit,
    _clause_boundary_ok,
    _pick_clause_boundary,
)


def _set_env(**kw):
    prev = {}
    for k, v in kw.items():
        prev[k] = os.environ.get(k)
        os.environ[k] = str(v)
    return prev


def _restore(prev):
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _cue(text, start_ms, end_ms, word_timings=None):
    return {
        "idx": 1, "start_ms": start_ms, "end_ms": end_ms,
        "lines": [text], "type": "dialogue",
        "meta": {
            "dialogue_text": text,
            "runs": [{"speaker": None, "word_start": 0}],
            "word_timings": word_timings or [],
        },
    }


def _cps(cue):
    body = cue["meta"]["dialogue_text"]
    dur = max(0.001, (cue["end_ms"] - cue["start_ms"]) / 1000.0)
    return len(body) / dur


# ── 1) Clause split never orphans a leading function word ───────────────────
def test_clause_boundary_rejects_orphaned_preposition():
    # "...the regeneration" | "of glutathione" — the tail leads with "of".
    words = "function is the regeneration of glutathione".split()
    # index 4 = split before "of" → tail "of glutathione" → REJECTED.
    of_idx = words.index("of")
    assert _clause_boundary_ok(words, of_idx) is False


def test_pick_clause_boundary_keeps_phrase_together():
    prev = _set_env(CUSTOM_MAX_CHARS=32, CUSTOM_MAX_LINES=2)
    try:
        # A sentence boundary exists mid-way; the picker must prefer it and must
        # never return a break that strands "of" on the tail.
        words = "It's a vitamin. Its function is the regeneration of glutathione.".split()
        idx = _pick_clause_boundary(words)
        if idx is not None:
            assert words[idx].lower().strip(".,;:") != "of", \
                f"clause picker orphaned 'of': {words[:idx]} | {words[idx:]}"
    finally:
        _restore(prev)


# ── 2) A fast-paced short boundary is NOT split into a tiny over-CPS sliver ──
def test_no_tiny_over_cps_child_when_split():
    prev = _set_env(
        CUSTOM_MAX_CHARS=32, CUSTOM_MAX_LINES=2, CUSTOM_SHAPING_ENABLED=1,
        CUSTOM_TARGET_DURATION_MS=3000, CUSTOM_MIN_DISPLAY_MS=1000,
        CUSTOM_MERGE_GAP_MS=83, SPEAKER_LABEL_MODE="none",
    )
    try:
        # Long-enough utterance to trigger shaping, with a leading short sentence
        # whose audio is fast — the naive splitter would isolate it into a sliver.
        text = ("It's riboflavin, honey. It's a vitamin, actually, "
                "and it helps with all sorts of things you might need.")
        # 6.0s window — over 1.5x target, so shaping fires.
        cue = _cue(text, 0, 6_000)
        out = shape_caption_rhythm([cue])
        # Every resulting child must clear the min reading window (no sliver).
        for c in out:
            dur_ms = c["end_ms"] - c["start_ms"]
            assert dur_ms >= 1000, \
                f"split produced a sub-min-display sliver: {dur_ms}ms — {c['lines']!r}"
    finally:
        _restore(prev)


# ── 3) CPL safety net STILL breaks an over-CPL line (hard FCC limit) ────────
def test_enforce_cpl_fit_still_splits_over_cpl_line():
    prev = _set_env(
        CUSTOM_MAX_CHARS=20, CUSTOM_MAX_LINES=2, CUSTOM_SHAPING_ENABLED=1,
        CUSTOM_MIN_DISPLAY_MS=1500, CUSTOM_MERGE_GAP_MS=83, SPEAKER_LABEL_MODE="none",
    )
    try:
        # A line that overflows 2×20ch but sits in a window too tight for two
        # min-display children — the safety net must break it anyway.
        text = "the quick brown fox jumped over the lazy sleeping dog again"
        cue = _cue(text, 0, 2_000)
        out = enforce_cpl_fit([cue])
        assert len(out) >= 2, "over-CPL line was not broken by the CPL safety net"
    finally:
        _restore(prev)
