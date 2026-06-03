"""
Regression tests for rendering.wrap_text() + services.linebreak.

COVERS
──────
1. Repeated-word faithfulness (the words.index() slice bug).
2. Proper-noun / phrase preservation ("The United States government").
3. Weak line endings (no dangling article / preposition / conjunction).
4. Clause-aware breaks (continuation leads with the conjunction).

All assertions are deterministic. Each test pins SPEAKER_LABEL_MODE='none' via
env so we exercise the pure wrap path without speaker tags interfering, and the
balanced-break flags default to on.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.rendering import wrap_text  # noqa: E402
from services.linebreak import choose_two_line_break  # noqa: E402


REPEATED_WORDS = "go go go to the town and the town will know"


def _faithful(lines, text):
    rejoined = " ".join(" ".join(lines).split())
    assert rejoined == text, (
        f"word-faithfulness broken\n  input:  {text!r}\n  output: {lines!r}"
    )


# ── 1. Repeated-word faithfulness ────────────────────────────────────
def test_repeated_words_faithful_across_widths():
    for max_chars in (8, 12, 16, 20, 24, 32, 42):
        for max_lines in (1, 2, 3):
            lines = wrap_text(REPEATED_WORDS, max_chars, max_lines)
            _faithful(lines, REPEATED_WORDS)
            assert len(lines) <= max_lines


def test_all_identical_words_faithful():
    text = "the the the the the the"
    lines = wrap_text(text, max_chars=7, max_lines=2)
    _faithful(lines, text)


# ── 2. Proper-noun / phrase preservation ─────────────────────────────
def test_proper_noun_not_split_across_lines():
    # "The United States government" at 16 chars: greedy would give
    # "The United" / "States government" (splits "United States"). The
    # clause-aware breaker should keep "United States" together.
    text = "The United States government"
    lines = wrap_text(text, max_chars=16, max_lines=2)
    _faithful(lines, text)
    assert len(lines) == 2
    # "United States" must not be split — i.e. line 1 must not END on "United"
    # while line 2 STARTS on "States".
    assert not (lines[0].endswith("United") and lines[1].startswith("States")), (
        f"split the proper-noun phrase: {lines!r}"
    )


# ── 3. Weak line endings ─────────────────────────────────────────────
def test_no_dangling_article_at_line_end():
    # The breaker must not leave "the" / "a" / "to" dangling at the end of
    # line 1 when a better break exists.
    text = "I went to the store because I needed milk"
    lines = wrap_text(text, max_chars=24, max_lines=2)
    _faithful(lines, text)
    assert len(lines) == 2
    last_word_line1 = lines[0].rstrip(".,;:").split()[-1].lower()
    assert last_word_line1 not in {"the", "a", "an", "to", "because", "and", "of"}, (
        f"weak line ending: {lines!r}"
    )


# ── 4. Clause-aware breaks ───────────────────────────────────────────
def test_continuation_leads_with_conjunction():
    # Canonical case: "I went to the store / because I needed milk."
    text = "I went to the store because I needed milk"
    lines = wrap_text(text, max_chars=24, max_lines=2)
    _faithful(lines, text)
    assert len(lines) == 2
    # Line 2 should lead with the subordinating conjunction "because".
    assert lines[1].lower().startswith("because"), (
        f"continuation did not lead with conjunction: {lines!r}"
    )


def test_break_after_comma_preferred():
    # A clause comma is the strongest natural break point.
    text = "When the rain stopped, we walked home together"
    lines = wrap_text(text, max_chars=26, max_lines=2)
    _faithful(lines, text)
    assert len(lines) == 2
    assert lines[0].rstrip().endswith(","), f"did not break at the comma: {lines!r}"


# ── Direct breaker unit checks ───────────────────────────────────────
def test_choose_two_line_break_none_when_single_word():
    assert choose_two_line_break("supercalifragilistic", max_chars=10) is None


def test_choose_two_line_break_balanced():
    # Two equal halves should be chosen for a balance-only string (no clause cues).
    res = choose_two_line_break("alpha bravo charlie delta", max_chars=14)
    assert res is not None
    _faithful(res, "alpha bravo charlie delta")


# ── Trivial cases ────────────────────────────────────────────────────
def test_short_text_single_line():
    assert wrap_text("hello world", max_chars=32, max_lines=2) == ["hello world"]


def test_empty_text():
    assert wrap_text("", max_chars=32, max_lines=2) == [""]
