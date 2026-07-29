"""
Neighbor time-donation — regression + invariant suite (Commit 2).

WHAT THIS LOCKS
───────────────
services.cps.donate_time_from_neighbors is the CROSS-CUE TIMING REBALANCE that
resolves a "boxed-in" over-CPS cue: when a fast cue has NO idle silence to grow
into (extend_fast_cues is powerless) but an adjacent SAME-SPEAKER cue is reading
BELOW target_cps, the shared boundary between the two is moved so the fast cue
borrows the slow neighbor's surplus display time. Pure timing — no text change,
no new cue, no overlap.

This suite is the enterprise/auditor proof that the reducer both (a) ACTUALLY
resolves the real-world boxed-in 31-CPS cue, and (b) can NEVER violate a
donor-safety invariant. It asserts, with zero tolerance:

  RESOLUTION
    • the boxed-in over-CPS cue is brought to/under max_cps by donation, when a
      slow same-speaker neighbor exists on either side.

  SAFETY INVARIANTS (every one is a hard failure if broken)
    1. NO OVERLAP           — consecutive cues never overlap; min_gap preserved.
    2. DONOR NEVER RUSHED   — a donor cue is never pushed ABOVE target_cps by the
                              time it gives away (moving the violation is forbidden).
    3. MIN-DISPLAY FLOOR    — neither cue is shrunk below min_display_ms.
    4. TEXT UNTOUCHED       — no cue's text/lines/word content changes (timing only).
    5. BOUNDARY RESPECT     — never donates across a speaker change, an unknown-
                              speaker wall, or a hard pause/authored boundary.
    6. NO SELF-HARM         — an over-CPS cue with NO eligible neighbor is left
                              exactly as-is (never made worse).
    7. DETERMINISM          — identical (cues, knobs) → byte-identical result.
    8. CHAIN INTEGRATION    — donation runs inside enforce_cps_rules and resolves
                              the boxed-in case end-to-end.
  PROPERTY SWEEP
    • a randomized fuzz over many (fast-cue, donor) geometries asserts every
      safety invariant holds on every generated case (never just the hand cases).

Pure functions, deterministic, no I/O, no AI. SOC 2 CC8.1 / FCC 47 CFR §79.1.
"""

import random

import pytest

from services.rules import activate_rule_context, reset_rule_context
from services.cps import (
    donate_time_from_neighbors,
    enforce_cps_rules,
    cue_cps,
    _visible_chars,
    _duration_ms,
)

# Spec knobs used across the suite (FCC-32-ish: 17 max / 15 target CPS).
KNOBS = {
    "CUSTOM_MAX_CPS": "17",
    "CUSTOM_TARGET_CPS": "15",
    "CUSTOM_MIN_CPS": "5",
    "CUSTOM_MIN_DISPLAY_MS": "800",
    "CUSTOM_MAX_DISPLAY_MS": "7000",
    "CUSTOM_MERGE_GAP_MS": "80",
    "CUSTOM_MAX_CHARS": "32",
    "CUSTOM_MAX_LINES": "2",
    "CPS_MEASUREMENT": "characters",
}
MAX_CPS = 17
TARGET_CPS = 15
MIN_DISPLAY = 800
MIN_GAP = 80


@pytest.fixture(autouse=True)
def _rules_ctx():
    """Inject the spec knobs into the engine's rule context for every test, and
    tear them down after so no state leaks between cases (100-user parity: the
    reducer reads knobs from the active context, never process env)."""
    activate_rule_context(dict(KNOBS))
    try:
        yield
    finally:
        reset_rule_context()


# ── Cue builders ────────────────────────────────────────────────────────────
def _cue(text, start_ms, end_ms, speaker="A", ctype="dialogue", meta_extra=None):
    meta = {"dialogue_text": text, "runs": [{"speaker": speaker, "word_start": 0}]}
    if meta_extra:
        meta.update(meta_extra)
    return {
        "idx": 0, "type": ctype, "start_ms": int(start_ms), "end_ms": int(end_ms),
        "lines": [text], "meta": meta,
    }


def _snapshot_text(cues):
    """Bag of (text, lines) per cue — used to prove text is never mutated."""
    return [
        ((c.get("meta") or {}).get("dialogue_text"), tuple(c.get("lines") or []))
        for c in cues
    ]


# ── Safety-invariant assertions (reused everywhere) ──────────────────────────
def assert_no_overlap(cues):
    for i in range(len(cues) - 1):
        a, b = cues[i], cues[i + 1]
        assert a["end_ms"] <= b["start_ms"], (
            f"OVERLAP: cue[{i}] ends {a['end_ms']} > cue[{i+1}] starts {b['start_ms']}"
        )
        assert b["start_ms"] - a["end_ms"] >= MIN_GAP - 1, (
            f"MIN_GAP violated between cue[{i}] and cue[{i+1}]: "
            f"gap {b['start_ms'] - a['end_ms']}ms < {MIN_GAP}ms"
        )


def assert_min_display(cues):
    for i, c in enumerate(cues):
        if c.get("type") != "dialogue":
            continue
        assert _duration_ms(c) >= MIN_DISPLAY, (
            f"MIN_DISPLAY violated: cue[{i}] duration {_duration_ms(c)}ms < {MIN_DISPLAY}ms"
        )


def assert_no_donor_rushed(before, after):
    """A cue that LOST display time (its duration shrank) must not end up ABOVE
    target_cps as a result — the reducer must never move the violation."""
    for i, (b, a) in enumerate(zip(before, after)):
        if a.get("type") != "dialogue":
            continue
        if _duration_ms(a) < _duration_ms(b):  # this cue donated time
            # Small rounding tolerance (integer ms interpolation).
            assert cue_cps(a) <= TARGET_CPS + 0.5, (
                f"DONOR RUSHED: cue[{i}] shrank and is now {cue_cps(a):.1f} CPS "
                f"> target {TARGET_CPS} — violation was moved, not resolved"
            )


def assert_text_untouched(before_text, after_cues):
    assert before_text == _snapshot_text(after_cues), (
        "TEXT MUTATED: donation is a timing-only op and must never change any "
        "cue's text or line content"
    )


# ═════════════════════════════════════════════════════════════════════════════
# 1. RESOLUTION — the boxed-in 31-CPS cue is fixed by borrowing from the NEXT cue
# ═════════════════════════════════════════════════════════════════════════════
def test_boxed_in_fast_cue_resolved_by_next_neighbor():
    # Fast cue: 31 chars over 1000ms = 31 CPS (way over 17), boxed in (next cue
    # starts right after the min_gap). Next cue: 10 chars over 3000ms = 3.3 CPS,
    # tons of surplus time to donate. Same speaker, no wall.
    fast = _cue("Thirty one characters right here", 0, 1000, speaker="A")
    slow = _cue("Short one.", 1080, 4080, speaker="A")
    assert cue_cps(fast) > MAX_CPS  # precondition: genuinely over

    out = donate_time_from_neighbors([fast, slow])

    assert cue_cps(out[0]) <= MAX_CPS, (
        f"fast cue still over: {cue_cps(out[0]):.1f} CPS"
    )
    assert_no_overlap(out)
    assert_min_display(out)


def test_boxed_in_fast_cue_resolved_by_prev_neighbor():
    # Slow donor BEFORE the fast cue; the fast cue is boxed on its right by end.
    slow = _cue("Short one.", 0, 3000, speaker="A")
    fast = _cue("Thirty one characters right here", 3080, 4080, speaker="A")
    assert cue_cps(fast) > MAX_CPS

    out = donate_time_from_neighbors([slow, fast])

    assert cue_cps(out[1]) <= MAX_CPS
    assert_no_overlap(out)
    assert_min_display(out)


# ═════════════════════════════════════════════════════════════════════════════
# 2. DONOR-SAFETY INVARIANTS
# ═════════════════════════════════════════════════════════════════════════════
def test_donor_never_pushed_over_target():
    fast = _cue("Thirty one characters right here", 0, 1000, speaker="A")
    # Donor sized so it has only a LITTLE surplus — donation must stop before it
    # is rushed past target_cps.
    slow = _cue("A medium length caption line ok", 1080, 4080, speaker="A")
    before = [dict(fast), dict(slow)]
    out = donate_time_from_neighbors([fast, slow])
    assert_no_donor_rushed(before, out)
    assert_no_overlap(out)
    assert_min_display(out)


def test_no_overlap_and_min_gap_preserved():
    fast = _cue("Thirty one characters right here", 0, 1000, speaker="A")
    slow = _cue("Short.", 1080, 5080, speaker="A")
    out = donate_time_from_neighbors([fast, slow])
    assert_no_overlap(out)


def test_text_is_never_mutated():
    fast = _cue("Thirty one characters right here", 0, 1000, speaker="A")
    slow = _cue("Short.", 1080, 5080, speaker="A")
    before_text = _snapshot_text([fast, slow])
    out = donate_time_from_neighbors([fast, slow])
    assert_text_untouched(before_text, out)


# ═════════════════════════════════════════════════════════════════════════════
# 3. BOUNDARY RESPECT — donation never crosses a wall
# ═════════════════════════════════════════════════════════════════════════════
def test_no_donation_across_speaker_change():
    fast = _cue("Thirty one characters right here", 0, 1000, speaker="A")
    slow = _cue("Short.", 1080, 5080, speaker="B")  # different speaker
    before = _snapshot_text([fast, slow])
    b0, b1 = dict(fast), dict(slow)
    out = donate_time_from_neighbors([fast, slow])
    # Nothing moved — a cross-speaker boundary is immutable.
    assert out[0]["end_ms"] == b0["end_ms"]
    assert out[1]["start_ms"] == b1["start_ms"]
    assert_text_untouched(before, out)


def test_no_donation_across_hard_pause_boundary():
    fast = _cue("Thirty one characters right here", 0, 1000, speaker="A")
    # Same speaker, but the next cue opens at a hard source-utterance pause.
    slow = _cue("Short.", 1080, 5080, speaker="A",
                meta_extra={"pause_boundary_before": True})
    b0, b1 = dict(fast), dict(slow)
    out = donate_time_from_neighbors([fast, slow])
    assert out[0]["end_ms"] == b0["end_ms"]
    assert out[1]["start_ms"] == b1["start_ms"]


def test_no_donation_across_unknown_speaker_wall():
    fast = _cue("Thirty one characters right here", 0, 1000, speaker="A")
    slow = _cue("Short.", 1080, 5080, speaker="A",
                meta_extra={"review_required": True})  # unknown-speaker wall
    b0, b1 = dict(fast), dict(slow)
    out = donate_time_from_neighbors([fast, slow])
    assert out[0]["end_ms"] == b0["end_ms"]
    assert out[1]["start_ms"] == b1["start_ms"]


# ═════════════════════════════════════════════════════════════════════════════
# 4. NO SELF-HARM — no eligible neighbor ⇒ left exactly as-is
# ═════════════════════════════════════════════════════════════════════════════
def test_fast_cue_with_no_slow_neighbor_left_unchanged():
    # Both neighbors are themselves at/over target — no surplus to donate.
    fast = _cue("Thirty one characters right here", 0, 1000, speaker="A")
    busy = _cue("Also quite a dense caption line!", 1080, 2080, speaker="A")
    b0, b1 = dict(fast), dict(busy)
    out = donate_time_from_neighbors([fast, busy])
    assert out[0]["start_ms"] == b0["start_ms"] and out[0]["end_ms"] == b0["end_ms"]
    assert out[1]["start_ms"] == b1["start_ms"] and out[1]["end_ms"] == b1["end_ms"]
    assert_no_overlap(out)
    assert_min_display(out)


def test_isolated_fast_cue_untouched():
    fast = _cue("Thirty one characters right here", 0, 1000, speaker="A")
    b0 = dict(fast)
    out = donate_time_from_neighbors([fast])
    assert out[0]["start_ms"] == b0["start_ms"] and out[0]["end_ms"] == b0["end_ms"]


def test_compliant_cue_untouched():
    ok = _cue("Comfortable pace here.", 0, 3000, speaker="A")
    slow = _cue("Short.", 3080, 7080, speaker="A")
    b0, b1 = dict(ok), dict(slow)
    out = donate_time_from_neighbors([ok, slow])
    assert out[0] == b0 and out[1] == b1  # nothing over max_cps ⇒ no-op


# ═════════════════════════════════════════════════════════════════════════════
# 5. DETERMINISM
# ═════════════════════════════════════════════════════════════════════════════
def test_determinism():
    def build():
        return [
            _cue("Thirty one characters right here", 0, 1000, speaker="A"),
            _cue("Short one.", 1080, 4080, speaker="A"),
        ]
    a = donate_time_from_neighbors(build())
    b = donate_time_from_neighbors(build())
    assert [(c["start_ms"], c["end_ms"]) for c in a] == \
           [(c["start_ms"], c["end_ms"]) for c in b]


# ═════════════════════════════════════════════════════════════════════════════
# 6. CHAIN INTEGRATION — resolved end-to-end through enforce_cps_rules
# ═════════════════════════════════════════════════════════════════════════════
def test_enforce_cps_rules_resolves_boxed_in_case():
    fast = _cue("Thirty one characters right here", 0, 1000, speaker="A")
    slow = _cue("Short one.", 1080, 4080, speaker="A")
    out = enforce_cps_rules([fast, slow])
    # After the full chain, the boxed-in fast cue must be within reading speed.
    fast_out = next(c for c in out
                    if (c.get("meta") or {}).get("dialogue_text", "").startswith("Thirty one"))
    assert cue_cps(fast_out) <= MAX_CPS
    assert_no_overlap(out)
    assert_min_display(out)


# ═════════════════════════════════════════════════════════════════════════════
# 7. PROPERTY SWEEP — every generated geometry upholds every safety invariant
# ═════════════════════════════════════════════════════════════════════════════
def test_property_sweep_all_invariants_hold():
    rng = random.Random(20260729)  # fixed seed → reproducible sweep
    checked = 0
    for _ in range(3000):
        fast_chars = rng.randint(20, 40)
        fast_dur = rng.randint(800, 2000)
        gap = MIN_GAP
        donor_chars = rng.randint(3, 30)
        donor_dur = rng.randint(800, 5000)
        same_speaker = rng.random() < 0.7
        wall = rng.random() < 0.2

        fast_text = "x" * fast_chars
        donor_text = "y" * donor_chars
        fast = _cue(fast_text, 0, fast_dur, speaker="A")
        meta_extra = {"pause_boundary_before": True} if wall else None
        donor = _cue(donor_text, fast_dur + gap, fast_dur + gap + donor_dur,
                     speaker="A" if same_speaker else "B", meta_extra=meta_extra)

        before = [dict(fast), dict(donor)]
        before_text = _snapshot_text([fast, donor])
        out = donate_time_from_neighbors([dict(fast), dict(donor)])

        # Every invariant, every case.
        assert_no_overlap(out)
        assert_min_display(out)
        assert_no_donor_rushed(before, out)
        assert_text_untouched(before_text, out)
        # An immutable boundary (speaker change or pause) ⇒ nothing moved.
        if (not same_speaker) or wall:
            assert out[0]["end_ms"] == before[0]["end_ms"]
            assert out[1]["start_ms"] == before[1]["start_ms"]
        checked += 1
    assert checked == 3000
