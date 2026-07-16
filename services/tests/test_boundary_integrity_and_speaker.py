"""
BOUNDARY-INTEGRITY + STRUCTURED-SPEAKER regression suite.

This is the auditor-passing evidence layer for the caption-composition
remediation (engine 5.31.0). It locks the two defect classes that no
single-cue stage could fix, plus the structured-speaker contract, against
regression — as GENERAL invariants, never per-example branches:

  A. CROSS-SPEAKER NON-FUSION — distinct speakers never share one silent,
     dash-less cue; the segmentation stage flushes on every speaker change.
  B. IMMUTABLE INTER-UTTERANCE PAUSE — a ≥ pause_boundary_ms silence between
     two DISTINCT source utterances is an unbreakable wall (the 1460ms
     "Cookie?" beat is its OWN cue), and the threshold is spec-driven
     (CUSTOM_PAUSE_BOUNDARY_MS, default 1200) end to end.
  C. NO STAGE MAY SPAN A WALL — the shared boundaries primitive vetoes a merge
     across a speaker/pause/unknown wall, and the sequence-optimizer windower
     terminates a window at a wall.
  D. STRUCTURED SPEAKER EMISSION — the formatter emits speaker_label (single),
     speaker_segments (genuine multi-speaker), and speaker_review_required
     (unknown) from source-token structure, NEVER by parsing rendered text.
  E. DECISION-LINKAGE INTEGRITY — a passthrough window produces its OWN unique
     decision_key_unbound (never a shared id) so unrelated passthrough cues can
     never collapse into one CCSegmentationDecision.

Deterministic, no I/O. SOC 2 CC8.1 / FCC 47 CFR §79.1.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.segmentation import (  # noqa: E402
    segment_into_sentence_groups,
    pause_boundary_ms,
    DEFAULT_PAUSE_BOUNDARY_MS,
)
from services.boundaries import (  # noqa: E402
    is_immutable_boundary,
    speakers_mergeable,
    boundary_reason,
    opens_hard_boundary,
)
from services.formatter import _structured_speaker_fields  # noqa: E402
from services.sequence_optimizer import (  # noqa: E402
    _collect_windows,
    _passthrough_identity,
)


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


def _tok(text, start_ms, end_ms, speaker, utt):
    return {"text": text, "start_ms": start_ms, "end_ms": end_ms,
            "speaker": speaker, "source_utterance_id": utt}


def _cue(text, start_ms, end_ms, runs, ctype="dialogue", **meta_extra):
    meta = {"dialogue_text": text, "runs": runs, "word_timings": []}
    meta.update(meta_extra)
    return {"idx": 1, "start_ms": start_ms, "end_ms": end_ms,
            "lines": [text], "type": ctype, "meta": meta}


# ═══ A. CROSS-SPEAKER NON-FUSION ════════════════════════════════════════════
def test_two_speakers_never_share_one_group():
    """Speaker A then Speaker B → two groups, each owning its own speaker.
    A single fused group would be the cross-speaker contamination defect."""
    tokens = [
        _tok("Good", 0, 200, "A", 0), _tok("morning.", 200, 600, "A", 0),
        _tok("Hello", 700, 900, "B", 1), _tok("there.", 900, 1300, "B", 1),
    ]
    groups = segment_into_sentence_groups(tokens)
    assert len(groups) == 2, f"expected 2 groups, got {len(groups)}"
    assert groups[0]["speaker"] == "A" and groups[0]["speaker_known"]
    assert groups[1]["speaker"] == "B" and groups[1]["speaker_known"]
    # Neither group's words leaked into the other.
    assert groups[0]["words"] == ["Good", "morning."]
    assert groups[1]["words"] == ["Hello", "there."]


def test_same_speaker_split_re_materializes_speaker():
    """The historical cross-speaker bug: after a same-speaker sentence split the
    FIRST following group emitted an EMPTY speaker_run (g_speaker=None) and the
    packer silently fused. Every non-empty group must carry an explicit run."""
    tokens = [
        _tok("First.", 0, 400, "A", 0),
        _tok("Second.", 500, 900, "A", 0),
    ]
    groups = segment_into_sentence_groups(tokens)
    assert len(groups) == 2
    for g in groups:
        assert g["speaker"] == "A"
        assert g["speaker_runs"] and g["speaker_runs"][0]["speaker"] == "A"


def test_unknown_speaker_never_inherits_neighbour():
    """A group whose tokens carried NO speaker is emitted review_required and
    never inherits an adjacent known speaker."""
    tokens = [
        _tok("Known.", 0, 400, "A", 0),
        _tok("Unknown.", 500, 900, None, 1),
    ]
    groups = segment_into_sentence_groups(tokens)
    assert len(groups) == 2
    assert groups[1]["speaker"] is None
    assert groups[1]["review_required"] is True


# ═══ B. IMMUTABLE INTER-UTTERANCE PAUSE (the 1460ms "Cookie?" fix) ══════════
def test_1460ms_pause_is_its_own_cue():
    """The regression: same speaker, but a 1460ms silence between two DISTINCT
    source utterances must open an IMMUTABLE new group (not merge the reply
    into the prior line)."""
    tokens = [
        _tok("Do", 0, 200, "A", 0), _tok("you", 200, 400, "A", 0),
        _tok("want", 400, 600, "A", 0), _tok("a", 600, 700, "A", 0),
        _tok("cookie?", 700, 1040, "A", 0),
        # 1460ms gap → distinct utterance 1.
        _tok("Cookie?", 2500, 2900, "A", 1),
    ]
    groups = segment_into_sentence_groups(tokens)
    assert len(groups) == 2, f"pause was not honored: {len(groups)} groups"
    assert groups[1]["words"] == ["Cookie?"]
    assert groups[1]["hard_boundary_before"] is True


def test_pause_below_threshold_does_not_split_same_utterance():
    """A long word-level hesitation WITHIN one source utterance must NOT force a
    hard split — the engine never over-fragments dramatic delivery."""
    tokens = [
        _tok("I", 0, 200, "A", 0),
        # 1400ms gap but SAME utterance 0 → no hard split.
        _tok("hesitate...", 1600, 2000, "A", 0),
    ]
    groups = segment_into_sentence_groups(tokens)
    assert len(groups) == 1
    assert groups[0]["hard_boundary_before"] is False


def test_pause_threshold_is_spec_driven():
    """CUSTOM_PAUSE_BOUNDARY_MS overrides the default; a malformed value falls
    back to 1200 so the invariant can never be silently disabled."""
    old = _set_env(CUSTOM_PAUSE_BOUNDARY_MS=800)
    try:
        assert pause_boundary_ms() == 800
        tokens = [
            _tok("First.", 0, 400, "A", 0),
            _tok("Second.", 1300, 1700, "A", 1),  # 900ms gap ≥ 800 → hard split
        ]
        groups = segment_into_sentence_groups(tokens)
        assert groups[1]["hard_boundary_before"] is True
    finally:
        _restore_env(old)
    # Malformed → default.
    old = _set_env(CUSTOM_PAUSE_BOUNDARY_MS="not-a-number")
    try:
        assert pause_boundary_ms() == DEFAULT_PAUSE_BOUNDARY_MS == 1200
    finally:
        _restore_env(old)
    # Non-positive → default (never disabled).
    old = _set_env(CUSTOM_PAUSE_BOUNDARY_MS=0)
    try:
        assert pause_boundary_ms() == 1200
    finally:
        _restore_env(old)


# ═══ C. NO STAGE MAY SPAN A WALL ════════════════════════════════════════════
def test_boundaries_veto_cross_speaker_merge():
    a = _cue("Hi.", 0, 800, [{"speaker": "A", "word_start": 0}])
    b = _cue("Hello.", 900, 1700, [{"speaker": "B", "word_start": 0}])
    assert speakers_mergeable(a, b) is False
    assert is_immutable_boundary(a, b) is True
    assert boundary_reason(a, b) is not None


def test_boundaries_veto_pause_wall():
    a = _cue("Cookie?", 0, 800, [{"speaker": "A", "word_start": 0}])
    b = _cue("Cookie?", 2500, 3300, [{"speaker": "A", "word_start": 0}],
             pause_boundary_before=True)
    assert opens_hard_boundary(b) is True
    assert is_immutable_boundary(a, b) is True


def test_boundaries_allow_same_speaker_no_wall():
    a = _cue("First part", 0, 800, [{"speaker": "A", "word_start": 0}])
    b = _cue("second part.", 900, 1700, [{"speaker": "A", "word_start": 0}])
    assert speakers_mergeable(a, b) is True
    assert is_immutable_boundary(a, b) is False


def test_optimizer_window_stops_at_wall():
    """A window is a maximal same-speaker run that never spans a cue opening an
    immutable wall. The pause cue must START its own window."""
    cues = [
        _cue("First.", 0, 800, [{"speaker": "A", "word_start": 0}]),
        _cue("Second.", 900, 1700, [{"speaker": "A", "word_start": 0}]),
        _cue("Cookie?", 3200, 4000, [{"speaker": "A", "word_start": 0}],
             pause_boundary_before=True),
    ]
    windows = _collect_windows(cues)
    # The first two same-speaker cues form one window [0,2); the pause cue is a
    # separate window [2,3) — never absorbed into the first.
    assert (0, 2) in windows
    assert (2, 3) in windows


def test_optimizer_window_stops_at_speaker_change():
    cues = [
        _cue("A one.", 0, 800, [{"speaker": "A", "word_start": 0}]),
        _cue("B two.", 900, 1700, [{"speaker": "B", "word_start": 0}]),
    ]
    windows = _collect_windows(cues)
    assert (0, 1) in windows and (1, 2) in windows


# ═══ D. STRUCTURED SPEAKER EMISSION (from source, never rendered text) ══════
def test_structured_speaker_single():
    cue = _cue("Hello there.", 0, 1200, [{"speaker": "A", "word_start": 0}])
    out = _structured_speaker_fields(cue)
    assert out.get("speaker_label") == "A"
    assert "speaker_segments" not in out


def test_structured_speaker_multi_emits_segments():
    """A legitimate dash-grouped two-speaker cue emits per-segment attribution
    AND a stable scalar primary — never mislabeling the whole cue as one."""
    cue = _cue("Hi there. Hello back.", 0, 1600,
               [{"speaker": "A", "word_start": 0},
                {"speaker": "B", "word_start": 2}])
    out = _structured_speaker_fields(cue)
    assert "speaker_segments" in out
    assert [s["speaker"] for s in out["speaker_segments"]] == ["A", "B"]
    assert out["speaker_segments"][0]["text"] == "Hi there."
    assert out["speaker_segments"][1]["text"] == "Hello back."
    # Scalar primary is the FIRST speaker, a stable legacy fallback.
    assert out["speaker_label"] == "A"


def test_structured_speaker_unknown_flags_review():
    cue = _cue("Who said this?", 0, 1200,
               [{"speaker": None, "word_start": 0}], review_required=True)
    out = _structured_speaker_fields(cue)
    assert out.get("speaker_review_required") is True
    assert "speaker_label" not in out


def test_structured_speaker_non_dialogue_is_empty():
    cue = _cue("[MUSIC]", 0, 1200, [], ctype="music")
    assert _structured_speaker_fields(cue) == {}


# ═══ E. DECISION-LINKAGE INTEGRITY (unique passthrough keys) ═══════════════
def test_passthrough_windows_get_distinct_keys():
    """Two unrelated passthrough windows must NOT share a decision_key_unbound —
    the linkage defect that collapsed unrelated cues into one decision."""
    w1_cues = [_cue("First window.", 0, 1200, [{"speaker": "A", "word_start": 0}])]
    w2_cues = [_cue("Second window.", 2000, 3200, [{"speaker": "B", "word_start": 0}])]
    id1 = _passthrough_identity(0, ["First", "window."], [0], w1_cues)
    id2 = _passthrough_identity(1, ["Second", "window."], [2000], w2_cues)
    assert id1["decision_key_unbound"] != id2["decision_key_unbound"]
    assert id1["input_hash"] != id2["input_hash"]
    # Each is reproducible: same inputs → same key.
    id1b = _passthrough_identity(0, ["First", "window."], [0], w1_cues)
    assert id1["decision_key_unbound"] == id1b["decision_key_unbound"]
