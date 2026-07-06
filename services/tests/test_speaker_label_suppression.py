"""
Regression tests for services.rendering.suppress_repeat_speaker_labels — the
UNIVERSAL, TURN-BASED repeat-speaker-label suppression pass.

The bug this locks: a project pinned to the 'named' speaker posture rendered
'[SPEAKER B:]' on every one of 4 consecutive back-to-back Speaker-B cues. The
professional convention (and every named / first-occurrence / every-change spec)
is to identify a speaker once and then omit the tag until the speaker CHANGES —
re-labeling the moment a different speaker speaks, INCLUDING when the original
speaker returns after an interruption (A…A…A…A → B → A re-labels A).

All repeat-suppressing modes ('named' / 'first_occurrence_per_scene' /
'every_change') resolve to the SAME turn-based rule. They differ only in what the
label SAYS (decided at render time), never in how often it appears. Once-per-scene
labeling would require real VISUAL scene detection, which the CC pipeline does not
do — so it is a future opt-in, not the default.

Coverage:
  • all three modes → label only on a speaker turn (re-label on any change).
  • the original speaker returning after another speaker → re-labeled.
  • scene boundaries re-assert labels (forward-looking hook).
  • 'always' / 'dash' → NEVER suppressed (no-op).
  • a non-dialogue cue between two same-speaker cues does NOT re-label the second.

All deterministic; env knobs set explicitly and restored after each test.
"""

import pytest  # noqa: E402  (import after module docstring is intentional)

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.rendering import suppress_repeat_speaker_labels  # noqa: E402


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


def _cue(idx, speaker, text, cue_type="dialogue"):
    """A dialogue cue rendered as the engine's bracket-tag modes would produce:
    the label rides on the first line, exactly like render_lines emits it."""
    runs = [{"speaker": speaker, "word_start": 0}] if speaker is not None else []
    return {
        "idx": idx, "start_ms": idx * 1000, "end_ms": idx * 1000 + 800,
        "lines": [text], "type": cue_type,
        "meta": {"dialogue_text": text, "runs": runs},
    }


def _labeled(name, body):
    return f"[{name}:] {body}"


def _first_line_has_label(cue):
    return cue["lines"][0].strip().startswith("[")


# ── all repeat-suppressing modes are turn-based (parametrized) ───────
# 'named', 'first_occurrence_per_scene', and 'every_change' resolve to the SAME
# turn-based rule: label only on a speaker change from the previous dialogue cue.
_TURN_BASED_MODES = ["named", "first_occurrence_per_scene", "every_change"]


@pytest.mark.parametrize("mode", _TURN_BASED_MODES)
def test_suppresses_consecutive_same_speaker(mode):
    prev = _set_env(SPEAKER_LABEL_MODE=mode, CUSTOM_MAX_CHARS=42, CUSTOM_MAX_LINES=2)
    try:
        cues = [
            _cue(1, "SPEAKER B", _labeled("SPEAKER B", "For players and fans alike,")),
            _cue(2, "SPEAKER B", _labeled("SPEAKER B", "the game is a crucible.")),
            _cue(3, "SPEAKER B", _labeled("SPEAKER B", "both those that are realized")),
            _cue(4, "SPEAKER B", _labeled("SPEAKER B", "Inexplicably awry.")),
        ]
        out = suppress_repeat_speaker_labels(cues)
        # First cue of the run keeps the tag; the next three drop it.
        assert _first_line_has_label(out[0])
        assert not _first_line_has_label(out[1])
        assert not _first_line_has_label(out[2])
        assert not _first_line_has_label(out[3])
    finally:
        _restore(prev)


@pytest.mark.parametrize("mode", _TURN_BASED_MODES)
def test_relabels_when_original_speaker_returns(mode):
    """The core behavior the user described: A…A…A…A → B → A must re-label the
    returning A. Every repeat-suppressing mode does this (turn-based, universal)."""
    prev = _set_env(SPEAKER_LABEL_MODE=mode, CUSTOM_MAX_CHARS=42, CUSTOM_MAX_LINES=2)
    try:
        cues = [
            _cue(1, "A", _labeled("A", "Hi there.")),
            _cue(2, "A", _labeled("A", "Still me.")),
            _cue(3, "B", _labeled("B", "Now me.")),
            _cue(4, "A", _labeled("A", "Back to me.")),  # change from B → keeps label
        ]
        out = suppress_repeat_speaker_labels(cues)
        assert _first_line_has_label(out[0])       # A (first)
        assert not _first_line_has_label(out[1])   # A repeat → stripped
        assert _first_line_has_label(out[2])       # change to B → kept
        assert _first_line_has_label(out[3])       # A returns after B → re-labeled
    finally:
        _restore(prev)


# ── scene boundaries re-assert labels (forward-looking hook) ─────────
def test_relabels_after_scene_boundary():
    prev = _set_env(SPEAKER_LABEL_MODE="named", CUSTOM_MAX_CHARS=42, CUSTOM_MAX_LINES=2)
    try:
        cues = [
            _cue(1, "A", _labeled("A", "Scene one.")),
            _cue(2, "A", _labeled("A", "Still scene one.")),
            _cue(3, "A", _labeled("A", "Scene two starts.")),
        ]
        out = suppress_repeat_speaker_labels(cues, scene_boundary_idxs={2})
        assert _first_line_has_label(out[0])       # A first in scene 1
        assert not _first_line_has_label(out[1])   # A repeat in scene 1
        assert _first_line_has_label(out[2])       # scene 2 boundary → A re-labeled
    finally:
        _restore(prev)


# ── non-dialogue cue between two same-speaker cues must not re-label ──
def test_music_cue_between_same_speaker_does_not_relabel():
    prev = _set_env(SPEAKER_LABEL_MODE="every_change", CUSTOM_MAX_CHARS=42, CUSTOM_MAX_LINES=2)
    try:
        cues = [
            _cue(1, "A", _labeled("A", "First A line.")),
            _cue(2, None, "\u266a music \u266a", cue_type="music"),
            _cue(3, "A", _labeled("A", "Second A line.")),  # still A → stripped
        ]
        out = suppress_repeat_speaker_labels(cues)
        assert _first_line_has_label(out[0])
        assert out[1]["lines"][0].startswith("\u266a")  # music untouched
        assert not _first_line_has_label(out[2])          # A run not broken by music
    finally:
        _restore(prev)


# ── no-op modes ──────────────────────────────────────────────────────
def test_always_mode_is_noop():
    prev = _set_env(SPEAKER_LABEL_MODE="always", CUSTOM_MAX_CHARS=42, CUSTOM_MAX_LINES=2)
    try:
        cues = [
            _cue(1, "B", _labeled("B", "One.")),
            _cue(2, "B", _labeled("B", "Two.")),
        ]
        out = suppress_repeat_speaker_labels(cues)
        assert _first_line_has_label(out[0])
        assert _first_line_has_label(out[1])  # 'always' keeps every label
    finally:
        _restore(prev)


def test_dash_mode_is_noop():
    prev = _set_env(SPEAKER_LABEL_MODE="dash", CUSTOM_MAX_CHARS=42, CUSTOM_MAX_LINES=2)
    try:
        cues = [
            _cue(1, "B", "For players and fans alike,"),
            _cue(2, "B", "the game is a crucible."),
        ]
        out = suppress_repeat_speaker_labels(cues)
        # Dash mode emits no bracket labels for single-speaker cues; the pass is
        # a no-op and the text is returned verbatim.
        assert out[0]["lines"][0] == "For players and fans alike,"
        assert out[1]["lines"][0] == "the game is a crucible."
    finally:
        _restore(prev)
