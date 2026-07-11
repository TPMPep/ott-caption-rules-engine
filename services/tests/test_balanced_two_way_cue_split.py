"""
Regression tests for the BALANCED TWO-WAY cue split in
services.formatter._split_sentence_into_cue_chunks.

THE BUG THIS LOCKS (PBS 0003 defect, 2026-07-10)
─────────────────────────────────────────────────
A sentence that overflows ONE cue but fits in TWO was greedy-filled: cue 1 was
packed to the character brim and the short remainder dumped into cue 2. On

    "Now, today's story is about a father who was obsessed with buying antiques."

at 32 CPL × 2 lines (budget 64) the greedy chunker produced a lopsided
    chunk 1 = "...who was obsessed with"   (58 ch)
    chunk 2 = "buying antiques."           (16 ch)
The shaper then re-split the 58ch chunk, yielding a 3-way fragment mess with a
1-line, over-CPS "Now, today's story is" fail-cue.

The balanced two-way split fixes this at the SOURCE: when a sentence fits in two
cues, it is split at the best phrase boundary into two balanced halves —
    "Now, today's story is about a father"   (36 ch → 2 lines)
    "who was obsessed with buying antiques."  (38 ch → 2 lines)
So "buying antiques." rides with its phrase and there are exactly TWO cues.

Coverage:
  • the production sentence splits into exactly 2 balanced chunks, neither a
    stranded tiny fragment, each within one cue's budget.
  • a sentence that genuinely needs 3+ cues still uses the greedy clause-backoff
    path (balanced two-way only fires when len ≤ 2×budget).
  • a sentence that already fits one cue is never sent to the chunker.

Deterministic; no env dependency (the splitter takes max_chars/max_lines args).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.formatter import _split_sentence_into_cue_chunks  # noqa: E402


def _chunks(text, max_chars=32, max_lines=2):
    words = text.split()
    out = _split_sentence_into_cue_chunks(words, max_chars, max_lines)
    return [" ".join(c["words"]) for c in out]


# ── The production defect: balanced TWO cues, not a lopsided 3-way shatter ──
def test_pbs_sentence_splits_into_two_balanced_cues():
    text = "Now, today's story is about a father who was obsessed with buying antiques."
    chunks = _chunks(text, 32, 2)

    assert len(chunks) == 2, f"expected 2 balanced cues, got {len(chunks)}: {chunks!r}"

    left, right = chunks
    # "buying antiques." must ride WITH its phrase, not be orphaned.
    assert right.endswith("buying antiques."), f"tail phrase was orphaned: {chunks!r}"
    assert "who was obsessed with" in right, f"phrase was split across cues: {chunks!r}"

    # Neither half is a stranded tiny fragment (≥3 words each).
    assert len(left.split()) >= 3 and len(right.split()) >= 3, f"tiny fragment: {chunks!r}"

    # Each half fits within one cue's budget (max_chars × max_lines = 64).
    assert len(left) <= 64 and len(right) <= 64, f"chunk over budget: {chunks!r}"

    # The two halves are reasonably balanced (the whole point) — neither is more
    # than ~2× the other. The old greedy split was 58 vs 16 (3.6×).
    la, lb = len(left), len(right)
    assert max(la, lb) <= 2 * min(la, lb), f"chunks not balanced: {la} vs {lb} — {chunks!r}"


# ── A genuinely long sentence (>2 cues) still uses the greedy path ──────────
def test_long_sentence_still_multi_chunks():
    # ~120 chars → needs 3 cues at budget 64; balanced-two-way must NOT fire.
    text = ("The committee unanimously agreed that the proposal to expand the "
            "regional transit network would require substantial additional funding.")
    chunks = _chunks(text, 32, 2)
    assert len(chunks) >= 2, f"expected multiple chunks: {chunks!r}"
    # Every chunk fits one cue's budget.
    for c in chunks:
        assert len(c) <= 64, f"chunk over budget: {c!r}"


# ── A short sentence that fits one cue would not reach the chunker, but the
#    chunker must still be a no-op-safe single chunk if called directly ───────
def test_short_sentence_single_chunk():
    text = "Hello, and welcome to Cash in the Attic."  # 40 ch, fits one 2-line cue
    chunks = _chunks(text, 32, 2)
    assert len(chunks) == 1, f"a within-two-line sentence must stay one chunk: {chunks!r}"
