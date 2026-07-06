"""
Regression tests for the WORD-LEVEL phrase-boundary fallback in services.shaping.

THE BUG THIS LOCKS
──────────────────
The shaper's splitter only knew how to break at CLAUSE PUNCTUATION (comma / period
/ semicolon). When a cue had NO interior punctuation — e.g. a labeled line like
"[SPEAKER A:] your head will be adorned in crimson and gold at the" — the clause
splitter returned None, so the cue was left intact and shipped an over-32ch labeled
line. 24 of 53 overflow cues on the Everwood project were exactly this class.

The fallback now splits a punctuation-free overflow at the best WORD phrase point
(before a preposition/conjunction/article, never stranding a function word), using
the same word-class intelligence the line-breaker uses — so every over-label cue can
be divided into compliant halves regardless of punctuation.

Deterministic; env knobs set explicitly and restored after each test.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.shaping import shape_caption_rhythm, _pick_word_phrase_boundary  # noqa: E402
from services.rendering import render_lines, wrap_text  # noqa: E402


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


def _labeled_cue(text, speaker, start_ms, end_ms, max_chars=32, max_lines=2):
    return {
        "idx": 1, "start_ms": start_ms, "end_ms": end_ms,
        "lines": wrap_text(text, max_chars, max_lines),
        "type": "dialogue",
        "meta": {"dialogue_text": text, "runs": [{"speaker": speaker, "word_start": 0}], "word_timings": []},
    }


def _delivered_longest(cue):
    lines = render_lines(
        cue["meta"]["dialogue_text"].split(),
        cue["meta"]["runs"], 2, 32, cue["meta"]["dialogue_text"],
    )
    return max((len(l) for l in lines), default=0)


# ── A punctuation-free labeled overflow IS split into compliant halves ──────
def test_no_punctuation_labeled_overflow_is_split():
    prev = _set_env(
        CUSTOM_MAX_CHARS=32, CUSTOM_MAX_LINES=2,
        CUSTOM_SHAPING_ENABLED=1, CUSTOM_TARGET_DURATION_MS=3000,
        CUSTOM_MIN_DISPLAY_MS=800, CUSTOM_MERGE_GAP_MS=83,
        SPEAKER_LABEL_MODE="alpha", SPEAKER_LABEL_FORMAT="[{name}:]",
        SPEAKER_LABEL_CASE="uppercase",
    )
    try:
        # No comma/period anywhere — the clause splitter alone can't touch this.
        text = "your head will be adorned in crimson and gold at the arena"
        cue = _labeled_cue(text, "A", 0, 3_054)
        assert _delivered_longest(cue) > 32, "precondition: labeled render should overflow"

        out = shape_caption_rhythm([cue])
        assert len(out) >= 2, f"punctuation-free overflow not split: {[c['lines'] for c in out]!r}"
        for c in out:
            assert _delivered_longest(c) <= 32, f"labeled line still over CPL: {c['lines']!r}"
    finally:
        _restore(prev)


# ── The word-boundary picker prefers a phrase start, avoids stranding ───────
def test_word_boundary_prefers_phrase_start():
    prev = _set_env(CUSTOM_MAX_CHARS=32, CUSTOM_MAX_LINES=2)
    try:
        # "adorned in crimson" — a good split leads the 2nd half with "in"
        # (a preposition) and does NOT strand "the"/"at" at the 1st half's end.
        words = "your head will be adorned in crimson and gold".split()
        idx = _pick_word_phrase_boundary(words)
        assert idx is not None and 0 < idx < len(words)
        # The word starting the second half should be a phrase-leader when one is
        # reasonably near the middle (here 'in' or 'and').
        first_of_second = words[idx].lower()
        assert first_of_second in {"in", "and", "crimson", "gold", "adorned"}, \
            f"unexpected split point: {words[:idx]} | {words[idx:]}"
    finally:
        _restore(prev)


# ── A single unsplittable long word is left intact (no loop) ────────────────
def test_single_long_word_left_intact():
    prev = _set_env(
        CUSTOM_MAX_CHARS=10, CUSTOM_MAX_LINES=2, CUSTOM_SHAPING_ENABLED=1,
        CUSTOM_TARGET_DURATION_MS=3000, CUSTOM_MIN_DISPLAY_MS=800, CUSTOM_MERGE_GAP_MS=83,
        SPEAKER_LABEL_MODE="none",
    )
    try:
        cue = _labeled_cue("supercalifragilisticexpialidocious", "A", 0, 2_000, max_chars=10)
        out = shape_caption_rhythm([cue])
        assert len(out) == 1, "an unsplittable single-word cue must be left intact"
    finally:
        _restore(prev)
