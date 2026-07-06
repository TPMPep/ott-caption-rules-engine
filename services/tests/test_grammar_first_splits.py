"""
Regression tests for the 0031/0032 defect class (2026-07-06):
"Okay, give me 25 laps around this place, then hit the lockers." shipped as
"Okay, give me 25 laps around" | "This place, then hit the lockers."

THREE ROOT CAUSES LOCKED HERE
─────────────────────────────
1. 'around' (and a tier of common prepositions) was missing from the
   line-break word tables, so nothing penalized stranding a preposition at a
   line/cue end or splitting inside its phrase.
2. The re-balanced boundary picker treated the readable-window check as a hard
   filter: when the grammatically-correct clause boundary failed it by a hair,
   it silently fell through to a mid-phrase word split. Grammar now outranks
   the window heuristic — clause boundaries win whenever they exist.
3. The editorial-AI pass wrote its casing fixes onto lines[] only, leaving
   meta.dialogue_text stale — so the deterministic capitalization pass saw a
   lowercase body, concluded nothing to fix, and the delivered line kept the
   wrong capital. The capitalization pass now reconciles the delivered line.

Deterministic; env knobs set explicitly and restored after each test.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.linebreak import _LEADING_WORDS, choose_two_line_break  # noqa: E402
from services.shaping import _pick_rebalanced_latin_boundary  # noqa: E402
from services.capitalization import apply_sentence_capitalization  # noqa: E402


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


def _cue(text, start_ms, end_ms, lines=None, word_timings=None):
    return {
        "idx": 1, "start_ms": start_ms, "end_ms": end_ms,
        "lines": lines if lines is not None else [text],
        "type": "dialogue",
        "meta": {
            "dialogue_text": text,
            "runs": [{"speaker": "C", "word_start": 0}],
            "word_timings": word_timings or [],
        },
    }


# ── 1) Preposition table covers the second tier ─────────────────────────────
def test_second_tier_prepositions_in_leading_words():
    for w in ("around", "across", "along", "behind", "beyond", "near", "past"):
        assert w in _LEADING_WORDS, f"'{w}' missing from _LEADING_WORDS"
    # Phrasal-verb particles must stay OUT (legitimately end a line).
    for w in ("off", "up", "down", "out"):
        assert w not in _LEADING_WORDS, f"particle '{w}' wrongly in _LEADING_WORDS"


def test_linebreak_never_strands_around():
    prev = _set_env(CUSTOM_PREFER_BALANCED_LINES=1, CUSTOM_PRESERVE_CLAUSE_INTEGRITY=1)
    try:
        lines = choose_two_line_break("Okay, give me 25 laps around this place,", 32)
        assert lines is not None
        assert not lines[0].rstrip().endswith("around"), \
            f"line 1 strands the preposition: {lines!r}"
        # The professional break: "Okay, give me 25 laps / around this place,"
        assert lines[1].startswith("around"), f"unexpected break: {lines!r}"
    finally:
        _restore(prev)


# ── 2) Grammar outranks the window heuristic ────────────────────────────────
def test_clause_boundary_wins_even_when_window_tight():
    prev = _set_env(CUSTOM_MAX_CHARS=32, CUSTOM_MAX_LINES=2)
    try:
        # Word timings reproduce the real 0031/0032 audio: 'place,' ends late
        # enough that the tail window is < min_display — the OLD picker then
        # fell through to the mid-phrase word split before 'this'.
        words = "Okay, give me 25 laps around this place, then hit the lockers.".split()
        wt, t = [], 68340
        durs = [400, 300, 200, 500, 400, 480, 350, 1070, 300, 300, 250, 900]
        for w, d in zip(words, durs):
            wt.append({"text": w, "start_ms": t, "end_ms": t + d})
            t += d
        cue = _cue(" ".join(words), 68340, 72337, word_timings=wt)
        idx = _pick_rebalanced_latin_boundary(cue, words, False, 1000, 83)
        assert idx is not None
        # The split must land at the clause comma (before 'then'), NEVER inside
        # the prepositional phrase (before 'this').
        assert words[idx] == "then", \
            f"grammar-wrong boundary: {words[:idx]} | {words[idx:]}"
    finally:
        _restore(prev)


# ── 3) Capitalization reconciles a line that diverged from dialogue_text ────
def test_line_meta_divergence_is_reconciled():
    # dialogue_text is correct (lowercase continuation) but the delivered line
    # was recased upstream — the pass must fix the LINE, not conclude no-op.
    c1 = _cue("Okay, give me 25 laps around", 68340, 70619)
    c2 = _cue("this place, then hit the lockers.", 70701, 72337,
              lines=["This place,", "then hit the lockers."])
    out = apply_sentence_capitalization([c1, c2])
    assert out[1]["lines"][0] == "this place,", \
        f"delivered line not reconciled: {out[1]['lines']!r}"


def test_capitalized_meta_continuation_still_lowered_on_lines():
    # Both meta AND lines carry the wrong capital (AI synced them) — the
    # continuation rule must lower both.
    c1 = _cue("Okay, give me 25 laps around", 68340, 70619)
    c2 = _cue("This place, then hit the lockers.", 70701, 72337,
              lines=["This place,", "then hit the lockers."])
    out = apply_sentence_capitalization([c1, c2])
    assert out[1]["lines"][0] == "this place,"
    assert out[1]["meta"]["dialogue_text"].startswith("this place,")
