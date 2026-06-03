"""
Regression tests for Phase 2 (CPS enforcement) + Phase 3 (QC auto-fix gate).

Phase 2 — services.cps
  • extend_fast_cues  : over-fast cue stretches into idle gap, never overlaps.
  • split_fast_cues   : still-over-fast cue splits at a clause boundary; both
                        halves end up within max_cps.
  • trim_slow_cues    : lingering cue trims dead air, never below min_display.

Phase 3 — services.qc
  • export_blocked gate fires on a hard (fail) violation and is clean otherwise.
  • per-cue grades carry severity + a machine-readable `fix` hint.

All deterministic. Each test sets the spec env knobs explicitly so the
thresholds under test are unambiguous, and restores them after.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services import cps as cps_mod  # noqa: E402
from services.qc import qc_report  # noqa: E402


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


def _dialogue(idx, start, end, text):
    return {
        "idx": idx, "start_ms": start, "end_ms": end,
        "lines": [text], "type": "dialogue",
        "meta": {"dialogue_text": text, "runs": [{"speaker": "A", "word_start": 0}]},
    }


# ── Phase 2: EXTEND ──────────────────────────────────────────────────
def test_extend_fast_cue_into_idle_gap():
    prev = _set_env(CUSTOM_MAX_CPS=17, CUSTOM_TARGET_CPS=15,
                    CUSTOM_MERGE_GAP_MS=80, CUSTOM_MAX_DISPLAY_MS=7000)
    try:
        # 40 chars in 1.0s = 40 CPS (way over 17). Next cue starts at 6000ms,
        # so there's idle room to extend into.
        cue = _dialogue(1, 0, 1000, "I really need to read this whole line")
        nxt = _dialogue(2, 6000, 7000, "Next line here")
        cues = cps_mod.extend_fast_cues([cue, nxt])
        # Extended cue must read at/under max and never reach the next cue.
        assert cps_mod.cue_cps(cues[0]) <= 17.01
        assert cues[0]["end_ms"] <= 6000 - 80
    finally:
        _restore(prev)


def test_extend_never_overlaps_tight_next_cue():
    prev = _set_env(CUSTOM_MAX_CPS=17, CUSTOM_TARGET_CPS=15, CUSTOM_MERGE_GAP_MS=80)
    try:
        cue = _dialogue(1, 0, 1000, "I really need to read this whole line now")
        nxt = _dialogue(2, 1200, 2200, "Right after")  # almost no gap
        cues = cps_mod.extend_fast_cues([cue, nxt])
        assert cues[0]["end_ms"] <= cues[1]["start_ms"] - 80
    finally:
        _restore(prev)


# ── Phase 2: SPLIT ───────────────────────────────────────────────────
def test_split_fast_cue_at_clause_boundary():
    prev = _set_env(CUSTOM_MAX_CPS=17, CUSTOM_TARGET_CPS=15,
                    CUSTOM_MIN_DISPLAY_MS=500, CUSTOM_MERGE_GAP_MS=80,
                    CUSTOM_MAX_CHARS=42, CUSTOM_MAX_LINES=2)
    try:
        # Long line, no gap to extend into → must split. Clause comma in middle.
        text = "When the rain finally stopped, we walked home together quickly"
        cue = _dialogue(1, 0, 2000, text)  # ~31 CPS over a 2s window
        out = cps_mod.split_fast_cues([cue])
        assert len(out) == 2, f"expected a split, got {len(out)} cue(s)"
        # Both halves within budget.
        assert cps_mod.cue_cps(out[0]) <= 17.01
        assert cps_mod.cue_cps(out[1]) <= 17.01
        # Faithful: the two halves rejoin to the original text.
        rejoined = " ".join(" ".join(out[0]["lines"] + out[1]["lines"]).split())
        assert rejoined == text
    finally:
        _restore(prev)


def test_split_declined_when_no_clean_room():
    prev = _set_env(CUSTOM_MAX_CPS=17, CUSTOM_MIN_DISPLAY_MS=900, CUSTOM_MERGE_GAP_MS=80)
    try:
        # Over-fast but the 1000ms window can't host two ≥900ms halves → decline,
        # leave intact for the QC gate to flag (never make it worse).
        cue = _dialogue(1, 0, 1000, "This line is far too fast to read in time")
        out = cps_mod.split_fast_cues([cue])
        assert len(out) == 1
    finally:
        _restore(prev)


# ── Phase 2: TRIM ────────────────────────────────────────────────────
def test_trim_slow_lingering_cue():
    prev = _set_env(CUSTOM_MIN_CPS=5, CUSTOM_TARGET_CPS=15, CUSTOM_MIN_DISPLAY_MS=800)
    try:
        # "Hi." (3 chars) held for 10s = 0.3 CPS → trim toward target, floor 800ms.
        cue = _dialogue(1, 0, 10000, "Hi.")
        out = cps_mod.trim_slow_cues([cue])
        assert out[0]["end_ms"] - out[0]["start_ms"] >= 800
        assert out[0]["end_ms"] < 10000
    finally:
        _restore(prev)


# ── Phase 3: QC GATE ─────────────────────────────────────────────────
def test_qc_gate_blocks_on_over_max_chars():
    prev = _set_env(CUSTOM_MAX_CHARS=20, CUSTOM_MAX_LINES=2,
                    CUSTOM_MAX_CPS=17, CUSTOM_MIN_DISPLAY_MS=500)
    try:
        cue = _dialogue(1, 0, 3000, "This single line is way over twenty chars")
        qc = qc_report(1, [cue], [])
        assert qc["export_blocked"] is True
        assert qc["blocked_cue_count"] == 1
        rules = {v["rule"] for g in qc["cue_grades"] for v in g["violations"]}
        assert "max_chars_per_line" in rules
        # Hard failures carry a machine-readable fix hint.
        fail = next(v for g in qc["cue_grades"] for v in g["violations"]
                    if v["severity"] == "fail")
        assert fail["fix"]
    finally:
        _restore(prev)


def test_qc_gate_clean_passes():
    prev = _set_env(CUSTOM_MAX_CHARS=42, CUSTOM_MAX_LINES=2,
                    CUSTOM_MAX_CPS=20, CUSTOM_MIN_CPS=3, CUSTOM_MIN_DISPLAY_MS=500)
    try:
        cue = _dialogue(1, 0, 2500, "A clean readable line.")
        qc = qc_report(1, [cue], [])
        assert qc["export_blocked"] is False
        assert qc["blocked_cue_count"] == 0
        assert qc["total_fail"] == 0
    finally:
        _restore(prev)


def test_qc_warn_does_not_block_export():
    prev = _set_env(CUSTOM_MAX_CHARS=42, CUSTOM_MAX_LINES=2,
                    CUSTOM_MAX_CPS=20, CUSTOM_MIN_CPS=5, CUSTOM_MIN_DISPLAY_MS=2000)
    try:
        # Short duration → a 'warn' (min_caption_duration_ms), never a blocker.
        cue = _dialogue(1, 0, 600, "Quick.")
        qc = qc_report(1, [cue], [])
        assert qc["export_blocked"] is False
        assert qc["total_warn"] >= 1
    finally:
        _restore(prev)
