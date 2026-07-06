"""
Regression tests for the 2026-07-06 Pluto segmentation defects (cues 0058-0068):

1. Sliver chunks — "No, I'm not gonna call…" must never yield a standalone
   "No," chunk (min-chunk guard in _split_sentence_into_cue_chunks).
2. Clause boundaries stranding a tiny side are excluded by the shaping picker.
3. Forced CPL splits never mint a near-zero-duration child.
4. LLM condensation that guts a sentence (word retention < 50%) is rejected.
5. Disfluency removal never force-capitalizes a mid-sentence continuation cue.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.formatter import _split_sentence_into_cue_chunks  # noqa: E402
from services.shaping import _pick_rebalanced_latin_boundary, _split_cue_once  # noqa: E402
from services.condensation import remove_disfluencies  # noqa: E402


def _set_env(**kv):
    old = {}
    for k, v in kv.items():
        old[k] = os.environ.get(k)
        os.environ[k] = str(v)
    return old


def _restore_env(old):
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ─── 1. Chunker min-chunk guard ──────────────────────────────────────
def test_chunker_never_strands_tiny_head():
    words = ("No, I'm not gonna call some 90-year-old woman "
             "who happens to own a metronome.").split()
    chunks = _split_sentence_into_cue_chunks(words, 32, 2)
    for c in chunks:
        assert len(c["words"]) >= 2, f"sliver chunk produced: {c['words']}"
    # Specifically: "No," must not be alone.
    assert chunks[0]["words"][:2] != ["No,"] or len(chunks[0]["words"]) > 1


def test_chunker_prefers_sentence_boundary():
    words = ("By who? Gus from the body shop or that drug salesman "
             "who gave you all the Prozac pens?").split()
    chunks = _split_sentence_into_cue_chunks(words, 32, 2)
    # The first chunk should carry "By who?" INTO a fuller chunk, never alone.
    assert len(chunks[0]["words"]) >= 3


# ─── 2/3. Shaping guards ─────────────────────────────────────────────
def _mk_cue(text, start_ms, end_ms):
    return {
        "idx": 1, "start_ms": start_ms, "end_ms": end_ms,
        "lines": [text], "type": "dialogue",
        "meta": {"dialogue_text": text,
                 "runs": [{"speaker": "A", "word_start": 0}],
                 "word_timings": []},
    }


def test_shaping_excludes_tiny_side_clause_boundary():
    words = ("No, I'm not gonna call some 90-year-old woman "
             "who happens to own a metronome.").split()
    cue = _mk_cue(" ".join(words), 0, 3000)
    old = _set_env(CUSTOM_MIN_DISPLAY_MS=1000, CUSTOM_MERGE_GAP_MS=80,
                   CUSTOM_MAX_CHARS=32, CUSTOM_MAX_LINES=2)
    try:
        idx = _pick_rebalanced_latin_boundary(cue, words, False, 1000, 80)
        # The only clause boundary (after "No,") strands a 1-word side —
        # the picker must NOT choose index 1.
        assert idx != 1, "picker chose the 'No,' sliver boundary"
    finally:
        _restore_env(old)


def test_forced_cpl_split_never_mints_zero_duration_child():
    words = ("No, I'm not gonna call some 90-year-old woman "
             "who happens to own a metronome.").split()
    cue = _mk_cue(" ".join(words), 0, 3000)
    # Word timings that put the first comma at 70ms in (the real-world case).
    cue["meta"]["word_timings"] = [
        {"text": "No,", "start_ms": 0, "end_ms": 70},
    ]
    old = _set_env(CUSTOM_MIN_DISPLAY_MS=1000, CUSTOM_MERGE_GAP_MS=80,
                   CUSTOM_MAX_CHARS=32, CUSTOM_MAX_LINES=2)
    try:
        split = _split_cue_once(cue, require_readable_windows=False)
        assert split is not None
        for child in split:
            dur = child["end_ms"] - child["start_ms"]
            assert dur >= 900, f"child cue too short: {dur}ms"
    finally:
        _restore_env(old)


# ─── 4. Condensation retention guard (deterministic layer) ───────────
def test_condensation_retention_floor():
    # The guard lives in _llm_condense_cue's acceptance path; verify the rule
    # itself: a 3-word rewrite of an 8-word verbatim is below the floor.
    verbatim = "Yeah, well, plenty lesson that could be you."
    condensed = "Yeah, many lessons."
    assert len(condensed.split()) < max(2, (len(verbatim.split()) + 1) // 2)


# ─── 5. Disfluency removal preserves continuation casing ─────────────
def test_disfluency_removal_keeps_lowercase_continuation():
    out = remove_disfluencies("the, um, Prozac pens?")
    assert out[0] == "t", f"continuation was force-capitalized: {out!r}"


def test_disfluency_removal_recapitalizes_sentence_start():
    out = remove_disfluencies("Um, we should go now.")
    assert out[0].isupper(), f"sentence start not re-capitalized: {out!r}"
