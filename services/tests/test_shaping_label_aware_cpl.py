"""
Regression tests for LABEL-AWARE CPL enforcement in services.shaping.

THE BUG THIS LOCKS
──────────────────
The shaper's CPL-overflow trigger (`_wrap_overflows_cpl`) previously measured
the cue's BODY-ONLY stored lines. But the deliverable prepends the speaker label
(e.g. '[SPEAKER B:] ', 13ch) at render time. So a cue whose body wraps clean at
2×32 body-only becomes an over-wide labeled line the trigger never caught.

Production example (Everwood, Pluto/Paramount FAST, 32 CPL, named/alpha label):

    "For the town of Everwood, basketball is like kindling."   (54ch body)
      body-only wrap → "For the town of Everwood," (25) / "basketball is like
                        kindling." (28)   ← both ≤32, looked "clean"
      BUT rendered    → "[SPEAKER B:] For the town of" / "Everwood, basketball
                        is like kindling."  (38ch)   ← OVER 32 → QC FAIL

The trigger now renders the cue EXACTLY as delivered (label included) via
render_lines, so the label-induced overflow fires the shaper, which splits the
cue at the comma so each resulting cue wraps within CPL WITH its label.

Deterministic; env knobs set explicitly and restored after each test.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.shaping import shape_caption_rhythm  # noqa: E402
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
    """Build a cue the formatter's way: lines pre-wrapped BODY-ONLY (the exact
    condition that hid the bug), meta carrying dialogue_text + a single-speaker
    run so render_lines re-applies the label."""
    return {
        "idx": 1,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "lines": wrap_text(text, max_chars, max_lines),  # body-only, as stored
        "type": "dialogue",
        "meta": {"dialogue_text": text, "runs": [{"speaker": speaker, "word_start": 0}], "word_timings": []},
    }


def _delivered_longest(cue):
    """Longest line as the DELIVERABLE draws it — label included."""
    lines = render_lines(
        cue["meta"]["dialogue_text"].split(),
        cue["meta"]["runs"], 2, 32, cue["meta"]["dialogue_text"],
    )
    return max((len(l) for l in lines), default=0)


# ── The Everwood bug: a labeled cue that overflows CPL only WITH the label ──
def test_label_induced_cpl_overflow_is_split():
    prev = _set_env(
        CUSTOM_MAX_CHARS=32, CUSTOM_MAX_LINES=2,
        CUSTOM_SHAPING_ENABLED=1, CUSTOM_TARGET_DURATION_MS=3000,
        CUSTOM_MIN_DISPLAY_MS=800, CUSTOM_MERGE_GAP_MS=83,
        SPEAKER_LABEL_MODE="alpha", SPEAKER_LABEL_FORMAT="[{name}:]",
        SPEAKER_LABEL_CASE="uppercase",
    )
    try:
        text = "For the town of Everwood, basketball is like kindling."
        # 3.9s — UNDER 1.5×3000 = 4500ms, so ONLY the CPL trigger can fire.
        cue = _labeled_cue(text, "B", 2_270, 6_230)

        # Precondition: BODY-ONLY it looks clean (≤32)…
        assert max(len(l) for l in cue["lines"]) <= 32, f"body-only precondition: {cue['lines']!r}"
        # …but the DELIVERED (labeled) render overflows 32 — the real defect.
        assert _delivered_longest(cue) > 32, "test precondition failed: labeled render should overflow"

        out = shape_caption_rhythm([cue])

        # It must have been split into ≥2 cues…
        assert len(out) >= 2, f"label-overflow cue was not split: {[c['lines'] for c in out]!r}"
        # …and EVERY resulting cue must fit CPL as DELIVERED (label included).
        for c in out:
            assert _delivered_longest(c) <= 32, f"labeled line still over CPL after split: {c['lines']!r}"
    finally:
        _restore(prev)


# ── A labeled cue already within CPL (label included) is left alone ─────────
def test_labeled_within_cpl_untouched():
    prev = _set_env(
        CUSTOM_MAX_CHARS=32, CUSTOM_MAX_LINES=2,
        CUSTOM_SHAPING_ENABLED=1, CUSTOM_TARGET_DURATION_MS=3000,
        CUSTOM_MIN_DISPLAY_MS=800, CUSTOM_MERGE_GAP_MS=83,
        SPEAKER_LABEL_MODE="alpha", SPEAKER_LABEL_FORMAT="[{name}:]",
        SPEAKER_LABEL_CASE="uppercase",
    )
    try:
        # Short body that still fits 2×32 WITH the label prepended.
        text = "Let's go, team."
        cue = _labeled_cue(text, "A", 10_000, 12_500)
        assert _delivered_longest(cue) <= 32
        out = shape_caption_rhythm([cue])
        assert len(out) == 1, f"a within-CPL labeled cue was needlessly split: {[c['lines'] for c in out]!r}"
    finally:
        _restore(prev)


# ── Dash mode (no bracket tag on single-speaker cues) is unaffected ─────────
def test_dash_mode_single_speaker_not_over_split():
    prev = _set_env(
        CUSTOM_MAX_CHARS=32, CUSTOM_MAX_LINES=2,
        CUSTOM_SHAPING_ENABLED=1, CUSTOM_TARGET_DURATION_MS=3000,
        CUSTOM_MIN_DISPLAY_MS=800, CUSTOM_MERGE_GAP_MS=83,
        SPEAKER_LABEL_MODE="dash",
    )
    try:
        # Dash mode adds NO prefix to a single-speaker cue, so a body that fits
        # body-only still fits delivered — must NOT be split.
        text = "For the town of Everwood, it's kindling."
        cue = _labeled_cue(text, "B", 2_270, 6_230)
        out = shape_caption_rhythm([cue])
        # Body wraps clean and no label is added → left intact.
        assert len(out) == 1, f"dash-mode single-speaker cue over-split: {[c['lines'] for c in out]!r}"
    finally:
        _restore(prev)
