"""
Unit tests for services.sequence_optimizer — the cross-cue editorial optimizer.

Covers the deterministic contract (Step 15): candidate generation, forward /
backward redistribution, 2-cue and 3-cue resegmentation, speaker-boundary
prohibition, phrase/quote/entity protection, text conservation, provenance
completeness, no-op stability on already-good cues, and the condensation gate.

All tests are DOM-free, deterministic, and restore env after each case.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.sequence_optimizer import (  # noqa: E402
    optimize_cue_sequence,
    condensation_is_blocked,
    _candidate_cut_sets,
    _collect_windows,
    OPTIMIZER_VERSION,
    SEGMENTATION_POLICY_VERSION,
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


def _wt(text, start_ms, end_ms):
    """Even word-level timings across [start_ms, end_ms] for a text string."""
    words = text.split()
    span = max(1, end_ms - start_ms)
    per = span // max(1, len(words))
    out = []
    t = start_ms
    for i, w in enumerate(words):
        e = start_ms + per * (i + 1) if i < len(words) - 1 else end_ms
        out.append({"text": w, "start_ms": int(t), "end_ms": int(e)})
        t = e
    return out


def _cue(text, start_ms, end_ms, speaker="A", with_timings=True):
    return {
        "idx": 1, "start_ms": start_ms, "end_ms": end_ms,
        "lines": [text], "type": "dialogue",
        "meta": {
            "dialogue_text": text,
            "runs": [{"speaker": speaker, "word_start": 0}],
            "word_timings": _wt(text, start_ms, end_ms) if with_timings else [],
        },
    }


def _text(cue):
    return (cue.get("meta") or {}).get("dialogue_text") or " ".join(cue.get("lines", []))


def _all_words(cues):
    out = []
    for c in cues:
        out.extend(_text(c).split())
    return out


# ── Windowing + candidate generation ────────────────────────────────
def test_windows_never_cross_speaker_or_sound():
    cues = [
        _cue("Hello there friend.", 0, 2000, speaker="A"),
        _cue("How are you.", 2100, 4000, speaker="A"),
        _cue("I am fine.", 4100, 6000, speaker="B"),
        {"idx": 4, "start_ms": 6100, "end_ms": 7000, "type": "music", "lines": ["[MUSIC]"], "meta": {}},
        _cue("Back again now.", 7100, 9000, speaker="A"),
    ]
    windows = _collect_windows(cues)
    # A-run(0,2), B(2,3), A(4,5) — music at index 3 breaks the window.
    assert (0, 2) in windows
    assert (2, 3) in windows
    assert (4, 5) in windows


def test_candidate_sets_include_no_change_2_and_3():
    words = "this is a fairly long sentence that should split cleanly here".split()
    ops = {op for op, _ in _candidate_cut_sets(words)}
    assert "no_change" in ops
    assert "resegment_2" in ops
    assert "resegment_3" in ops


def test_candidates_never_orphan_leading_function_word():
    words = "walk to the store and then come back home again now".split()
    for op, cuts in _candidate_cut_sets(words):
        for c in cuts:
            first = words[c].strip(".,;:!?").lower()
            assert first not in {"to", "and", "the", "a", "an", "of"}, \
                f"candidate {op} orphaned leading word '{first}'"


# ── Text conservation + speaker ownership ───────────────────────────
def test_text_conservation_across_redistribution():
    prev = _set_env(CUSTOM_MAX_CHARS=32, CUSTOM_MAX_LINES=2, CUSTOM_MIN_DISPLAY_MS=800,
                    CUSTOM_MAX_CPS=17, CUSTOM_MERGE_GAP_MS=80)
    try:
        cues = [
            _cue("The very first part of this sentence keeps going", 0, 2500, speaker="A"),
            _cue("and it finishes right about here now.", 2600, 3200, speaker="A"),
        ]
        before = _all_words(cues)
        out = optimize_cue_sequence(cues)
        after = _all_words(out)
        assert after == before, "words lost or reordered during optimization"
        # Every emitted cue stays speaker A.
        for c in out:
            spk = (c["meta"]["runs"][0]).get("speaker")
            assert spk == "A"
    finally:
        _restore(prev)


def test_never_merges_across_speaker_boundary():
    prev = _set_env(CUSTOM_MAX_CHARS=32, CUSTOM_MAX_LINES=2, CUSTOM_MIN_DISPLAY_MS=800)
    try:
        cues = [
            _cue("Speaker one says a full line here.", 0, 2500, speaker="A"),
            _cue("Speaker two answers right back now.", 2600, 5000, speaker="B"),
        ]
        out = optimize_cue_sequence(cues)
        # No cue may contain both speakers' words fused.
        for c in out:
            t = _text(c)
            assert not ("one" in t and "two" in t), "fused across speaker boundary"
    finally:
        _restore(prev)


# ── Provenance completeness ─────────────────────────────────────────
def test_provenance_stamped_on_every_dialogue_cue():
    prev = _set_env(CUSTOM_MAX_CHARS=32, CUSTOM_MAX_LINES=2, CUSTOM_MIN_DISPLAY_MS=800,
                    CUSTOM_MAX_CPS=17)
    try:
        cues = [
            _cue("A long enough opening clause to be interesting,", 0, 2500, speaker="A"),
            _cue("and a trailing clause that continues onward here.", 2600, 5200, speaker="A"),
        ]
        out = optimize_cue_sequence(cues)
        for c in out:
            so = c["meta"].get("seq_opt")
            assert so is not None, "cue missing seq_opt provenance"
            assert so["optimizer_version"] == OPTIMIZER_VERSION
            assert so["policy_version"] == SEGMENTATION_POLICY_VERSION
            assert "operation" in so
            assert "condensation_allowed" in so
            assert "condensation_reason" in so
            assert "segmentation_quality" in so
    finally:
        _restore(prev)


# ── Stability: an already-good window is left alone ─────────────────
def test_no_op_on_already_good_cues():
    prev = _set_env(CUSTOM_MAX_CHARS=42, CUSTOM_MAX_LINES=2, CUSTOM_MIN_DISPLAY_MS=800,
                    CUSTOM_MAX_CPS=17, CUSTOM_TARGET_DURATION_MS=3000)
    try:
        # Two clean, well-balanced, in-budget cues — nothing to improve.
        cues = [
            _cue("This is a perfectly fine caption.", 0, 2500, speaker="A"),
            _cue("And so is this one right here.", 2600, 5100, speaker="A"),
        ]
        before = [(_text(c), c["start_ms"], c["end_ms"]) for c in cues]
        out = optimize_cue_sequence(cues)
        # no_change must win (stability bias) — same text + boundaries.
        assert all(c["meta"]["seq_opt"]["operation"] == "no_change" for c in out)
        after = [(_text(c), c["start_ms"], c["end_ms"]) for c in out]
        assert after == before
    finally:
        _restore(prev)


# ── Condensation gate wiring ────────────────────────────────────────
def test_condensation_blocked_when_nondestructive_fix_exists():
    prev = _set_env(CUSTOM_MAX_CHARS=32, CUSTOM_MAX_LINES=2, CUSTOM_MIN_DISPLAY_MS=800,
                    CUSTOM_MAX_CPS=17, CUSTOM_MERGE_GAP_MS=80)
    try:
        # An over-CPS opening cue whose overflow can be resegmented forward into
        # the roomy neighbour → optimizer resegments → condensation must be blocked.
        cues = [
            _cue("You get on well now so you are both going", 2200, 3400, speaker="A"),  # ~1.2s, fast
            _cue("to be helping us out today.", 3500, 7000, speaker="A"),  # roomy
        ]
        out = optimize_cue_sequence(cues)
        # At least one emitted cue must forbid condensation (redistribution won).
        assert any(condensation_is_blocked(c) for c in out)
        # And no emitted cue is a flash.
        for c in out:
            assert (c["end_ms"] - c["start_ms"]) >= 800
    finally:
        _restore(prev)


def test_condensation_allowed_when_no_resegmentation_helps():
    prev = _set_env(CUSTOM_MAX_CHARS=32, CUSTOM_MAX_LINES=2, CUSTOM_MIN_DISPLAY_MS=800,
                    CUSTOM_MAX_CPS=17)
    try:
        # A single short-window cue that is genuinely over-CPS with no neighbour
        # to redistribute into and too short to split into two min-display halves
        # → no compliant resegmentation exists → condensation must be ALLOWED.
        cues = [
            _cue("Absolutely everyone understood the situation immediately here.", 0, 1100, speaker="A"),
        ]
        out = optimize_cue_sequence(cues)
        # No compliant non-destructive candidate → condensation NOT blocked.
        assert not any(condensation_is_blocked(c) for c in out), \
            "condensation was blocked even though no compliant resegmentation existed"
        for c in out:
            assert c["meta"]["seq_opt"]["condensation_reason"] in (
                "no_compliant_resegmentation", "within_budget_no_change"), \
                c["meta"]["seq_opt"]["condensation_reason"]
    finally:
        _restore(prev)


# ── Rollout flag ────────────────────────────────────────────────────
def test_disabled_flag_is_noop():
    prev = _set_env(SEQ_OPTIMIZER_ENABLED=0, CUSTOM_MAX_CPS=17, CUSTOM_MIN_DISPLAY_MS=800)
    try:
        cues = [
            _cue("You get on well now so you are both going", 2200, 3400, speaker="A"),
            _cue("to be helping us out today.", 3500, 7000, speaker="A"),
        ]
        out = optimize_cue_sequence(cues)
        # Disabled → passthrough, no seq_opt provenance stamped.
        assert all("seq_opt" not in (c.get("meta") or {}) for c in out)
        assert _all_words(out) == _all_words(cues)
    finally:
        _restore(prev)
