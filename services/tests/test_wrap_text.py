"""
Regression tests for rendering.wrap_text().

GUARDS THE REPEATED-WORD SLICE BUG
──────────────────────────────────
wrap_text() previously sliced "remaining" words with words.index(word), which
returns the FIRST occurrence of a word — so a cue containing a repeated word
sliced from the wrong position and duplicated or dropped text. The canonical
proof string is:

    "go go go to the town and the town will know"

It contains "go" three times and "the"/"town" twice. Under the old code, when
the greedy fill hit the second "the" (or any repeat) at the last-line boundary,
words.index("the") returned the index of the FIRST "the", re-dumping a chunk of
already-placed text. The fix iterates with enumerate() and slices from the true
current index.

These tests assert the WORD-FAITHFULNESS invariant: the concatenation of all
rendered lines must equal the input text exactly — no duplicated, dropped, or
reordered words — for any max_chars / max_lines combination.
"""

import os
import sys

# Make services/ importable when run from the repo root or the tests/ dir.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.rendering import wrap_text  # noqa: E402


REPEATED_WORDS = "go go go to the town and the town will know"


def _assert_word_faithful(text, max_chars, max_lines):
    lines = wrap_text(text, max_chars, max_lines)
    rejoined = " ".join(" ".join(lines).split())
    assert rejoined == text, (
        f"word-faithfulness broken at max_chars={max_chars} max_lines={max_lines}\n"
        f"  input:  {text!r}\n"
        f"  output: {lines!r}\n"
        f"  rejoin: {rejoined!r}"
    )
    assert len(lines) <= max_lines, f"exceeded max_lines: {lines!r}"
    return lines


def test_repeated_words_no_duplication_two_lines():
    # The canonical bug repro — 2 lines, tight width forces a mid-string wrap
    # right on a repeated word.
    _assert_word_faithful(REPEATED_WORDS, max_chars=20, max_lines=2)


def test_repeated_words_across_widths_and_lines():
    # Sweep widths/line-counts — every combination must stay word-faithful.
    for max_chars in (8, 12, 16, 20, 24, 32, 42):
        for max_lines in (1, 2, 3):
            _assert_word_faithful(REPEATED_WORDS, max_chars, max_lines)


def test_all_identical_words():
    # Degenerate worst case: every word identical. index() would always return 0.
    _assert_word_faithful("the the the the the the", max_chars=7, max_lines=2)


def test_short_text_single_line():
    assert wrap_text("hello world", max_chars=32, max_lines=2) == ["hello world"]


def test_empty_text():
    assert wrap_text("", max_chars=32, max_lines=2) == [""]


def test_no_repeats_still_faithful():
    _assert_word_faithful(
        "the quick brown fox jumps over a lazy dog", max_chars=15, max_lines=3
    )
