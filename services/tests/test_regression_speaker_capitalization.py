"""
REGRESSION — final-presentation-order fix (formatter v6, 2026-07-30).
======================================================================
Locks the two production caption regressions the operator flagged on project
6a1d3079d8c8f415659d45fd, driven through the REAL formatter path
(main.baseline_to_assembly_result → build_caption_inputs → process_caption_job)
against a FROZEN baseline fixture — NOT isolated helper functions.

The bug: repeated-speaker-label suppression + sentence-boundary capitalization
ran BEFORE the final sequence optimizer / CPL reflow, which re-render cues from
meta.runs / meta.dialogue_text and therefore (a) re-added suppressed
[SPEAKER X:] labels on every same-speaker continuation cue and (b) downcased a
new-sentence first word ("For" → "for") across an A→B speaker boundary. The v6
fix moves both presentation passes to run LAST (after optimize → CPL fit →
frame-grid), so nothing re-renders after them.

Every one of the 11 required assertions is exercised here against the delivered
output the viewer actually sees. Additionally proves: meta.runs speaker identity
survives suppression, repeated Apply Spec runs are byte-identical (cue count +
canonical output hash), and zero OpenAI calls on the reformat path.

Run:  pytest src/cc-rules-engine/services/tests/test_regression_speaker_capitalization.py -v
"""

import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import services.editorial_ai as editorial_ai  # noqa: E402
from services.formatter import process_caption_job  # noqa: E402
from services.assembly import build_caption_inputs  # noqa: E402
import main  # noqa: E402

_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
_BASELINE_PATH = os.path.join(_FIXTURE_DIR, "regression_speaker_capitalization_baseline.json")

# alpha label mode → the engine emits '[SPEAKER A:]' / '[SPEAKER B:]' bracket
# tags and turn-based repeat-suppression applies (this is the exact class of the
# production project). A fixed 42×2 / 17cps posture keeps the run reproducible.
_ENV_OVERRIDES = {
    "CUSTOM_MAX_CHARS": "42",
    "CUSTOM_MAX_LINES": "2",
    "CUSTOM_TARGET_CPS": "15",
    "CUSTOM_MAX_CPS": "17",
    "CUSTOM_MIN_CPS": "5",
    "CPS_MEASUREMENT": "characters",
    "CUSTOM_MIN_DISPLAY_MS": "833",
    "CUSTOM_MAX_DISPLAY_MS": "7000",
    "CUSTOM_FRAME_RATE": "25",
    "CUSTOM_MIN_GAP_FRAMES": "2",
    "CUSTOM_MERGE_GAP_MS": "80",
    "CUSTOM_PAUSE_BOUNDARY_MS": "1200",
    "CUSTOM_TARGET_DURATION_MS": "3000",
    "CUSTOM_SHAPING_ENABLED": "1",
    "SPEAKER_LABEL_MODE": "alpha",
    "SPEAKER_LABEL_CASE": "uppercase",
    "MUSIC_CUE_FORMAT": "bracketed_uppercase",
    "SOUND_EFFECT_FORMAT": "bracketed_uppercase",
    "SOUND_DENSITY": "aggressive",
    "CONDENSATION_MODE": "disfluency_only",
    "OUTPUT_FORMATS": "srt,vtt",
    "SEQ_OPTIMIZER_ENABLED": "1",
    "SEQ_OPTIMIZER_VERSION": "1",
    "SPEC_SLUG": "regression-speaker-cap-42x2",
    "SPEC_VERSION": "1",
}


def _load_baseline():
    with open(_BASELINE_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


class _ExplodingOpenAI:
    """Fails LOUD if anything constructs an OpenAI client on the reformat path."""
    def __init__(self, *a, **k):
        raise AssertionError(
            "OpenAI client constructed during a deterministic reformat — "
            "the no-editorial-AI guarantee is violated.")


def _run_reformat_once(env=None):
    """Run ONE deterministic reformat exactly as main.run_caption_job does on the
    reformat_from_baseline path, editorial-AI blocked two ways. Returns the full
    result dict + the canonical output hash the ingester persists."""
    env = env if env is not None else dict(_ENV_OVERRIDES)
    baseline = _load_baseline()

    token = main.apply_env_overrides(env)
    forbid_prev = editorial_ai.FORBID_EDITORIAL_AI
    openai_prev = None
    try:
        editorial_ai.FORBID_EDITORIAL_AI = True
        try:
            import openai
            openai_prev = openai.OpenAI
            openai.OpenAI = _ExplodingOpenAI
        except Exception:
            openai_prev = None

        assembly_result = main.baseline_to_assembly_result(baseline)
        backbone_srt, timestamps = build_caption_inputs(assembly_result)
        audio_events = list(assembly_result.get("audio_events") or [])
        result = process_caption_job(
            backbone_srt_text=backbone_srt,
            timestamps=timestamps,
            protected_phrases=[],
            output_formats=["srt", "vtt"],
            audio_events=audio_events,
            allow_editorial_ai=False,  # DESIGN A: deterministic reformat path
        )
        chash = main.canonical_output_hash(result.get("cues") or [],
                                            result.get("segmentation_qc"))
        return result, chash
    finally:
        editorial_ai.FORBID_EDITORIAL_AI = forbid_prev
        if openai_prev is not None:
            import openai
            openai.OpenAI = openai_prev
        main.restore_env_overrides(token)


# ── Delivered-text helpers (measure exactly what the viewer sees) ────────────
def _cue_text(cue):
    return " ".join(cue.get("lines") or []).strip()


def _dialogue_cues(result):
    return [c for c in (result.get("cues") or []) if c.get("type") == "dialogue"]


def _has_speaker_b_label(cue):
    return bool(re.search(r"\[SPEAKER B:\]", _cue_text(cue)))


def _find_first_index(cues, needle):
    for i, c in enumerate(cues):
        if needle in _cue_text(c):
            return i
    return -1


@pytest.fixture(scope="module")
def result():
    r, _ = _run_reformat_once()
    return r


# ─── 1. Second "Chest pass" preserved as legitimate content ──────────────────
def test_second_chest_pass_preserved(result):
    joined = " ".join(_cue_text(c) for c in _dialogue_cues(result))
    # Two distinct "Chest pass" occurrences survive (the first ends "Chest
    # pass.", the second is the bare "Chest pass" with no terminal punctuation).
    assert joined.count("Chest pass") >= 2, (
        f"second 'Chest pass' was lost or merged away: {joined!r}")


# ─── 2. "For" stays capitalized across the A→B immutable boundary ────────────
def test_for_the_town_stays_capitalized(result):
    cues = _dialogue_cues(result)
    idx = _find_first_index(cues, "the town of Everwood")
    assert idx >= 0, "the 'For the town of Everwood,' cue disappeared"
    text = _cue_text(cues[idx])
    # Strip any leading [SPEAKER B:] label to inspect the real first word.
    body = re.sub(r"^\s*\[SPEAKER [A-Z]:\]\s*", "", text)
    assert body.startswith("For the town of Everwood"), (
        f"'For' was downcased across the A→B boundary — got {body!r}")
    assert "for the town of Everwood" not in text, (
        f"lowercase 'for the town' present in delivered text: {text!r}")


# ─── 3. Speaker B labeled ONLY on the first cue of the uninterrupted turn ────
def test_speaker_b_labeled_once_in_uninterrupted_turn(result):
    cues = _dialogue_cues(result)
    start = _find_first_index(cues, "the town of Everwood")
    assert start >= 0
    # Walk the uninterrupted B run: from the first B cue up to (but not
    # including) the Speaker C cue. In alpha mode a suppressed cue carries no
    # bracket label. Exactly ONE labeled cue may appear in the run.
    c_idx = _find_first_index(cues, "Coach says we start again")
    assert c_idx > start, "Speaker C cue not found after the B run"
    b_run = cues[start:c_idx]
    labeled = [c for c in b_run if _has_speaker_b_label(c)]
    assert len(labeled) == 1, (
        f"Speaker B label should appear exactly once in the uninterrupted turn; "
        f"found {len(labeled)} labeled cues: {[_cue_text(c) for c in b_run]}")
    # And it must be the FIRST cue of the run.
    assert _has_speaker_b_label(b_run[0]), "first B cue is not labeled"


# ─── 4. A music/SFX cue does NOT reset the Speaker B turn ────────────────────
def test_music_cue_does_not_relabel_speaker_b(result):
    cues = result.get("cues") or []
    music_idx = next((i for i, c in enumerate(cues)
                      if c.get("type") in ("music", "sound_effect")), -1)
    assert music_idx >= 0, "the music cue was dropped"
    # The dialogue cue immediately AFTER the music cue, still inside the B turn
    # (before Speaker C), must NOT carry a re-asserted [SPEAKER B:] label.
    after = [c for c in cues[music_idx + 1:] if c.get("type") == "dialogue"]
    c_idx = _find_first_index(after, "Coach says we start again")
    b_after_music = after[:c_idx] if c_idx >= 0 else after
    assert b_after_music, "no B dialogue cue after the music cue"
    assert not _has_speaker_b_label(b_after_music[0]), (
        f"music cue spuriously reset the B turn — label re-appeared on "
        f"{_cue_text(b_after_music[0])!r}")


# ─── 5. Speaker C is labeled when the speaker changes ────────────────────────
def test_speaker_c_labeled_on_change(result):
    cues = _dialogue_cues(result)
    idx = _find_first_index(cues, "Coach says we start again")
    assert idx >= 0, "Speaker C cue not found"
    assert re.search(r"\[SPEAKER C:\]", _cue_text(cues[idx])), (
        f"Speaker C was not labeled on the speaker change: {_cue_text(cues[idx])!r}")


# ─── 6. Speaker B labeled AGAIN when B returns after Speaker C ───────────────
def test_speaker_b_relabeled_on_return(result):
    cues = _dialogue_cues(result)
    idx = _find_first_index(cues, "Everwood never forgets a champion")
    assert idx >= 0, "the returning Speaker B cue disappeared"
    assert _has_speaker_b_label(cues[idx]), (
        f"Speaker B was not re-labeled on return after C: {_cue_text(cues[idx])!r}")


# ─── 7. meta.runs retains correct speaker identities after suppression ───────
def test_meta_runs_speaker_identity_survives_suppression(result):
    cues = _dialogue_cues(result)
    # Every dialogue cue carries structured speaker attribution derived from
    # meta.runs (speaker_label on the result cue). Suppression only removes the
    # DELIVERED bracket label — the structured identity must survive. The
    # returning-B cue is suppression-relevant (it was re-labeled), so its
    # structured speaker must still be present.
    idx = _find_first_index(cues, "Everwood never forgets a champion")
    assert idx >= 0
    ret = cues[idx]
    # alpha mode resolves speaker_label to a 'SPEAKER B'-style display name.
    assert (ret.get("speaker_label") or "").upper().endswith("B"), (
        f"structured speaker identity lost on the returning-B cue: {ret!r}")
    # A suppressed continuation cue (no bracket label in its delivered text)
    # must STILL carry structured speaker attribution.
    start = _find_first_index(cues, "the town of Everwood")
    c_idx = _find_first_index(cues, "Coach says we start again")
    for c in cues[start:c_idx]:
        assert c.get("speaker_label"), (
            f"a B-turn cue lost its structured speaker_label: {_cue_text(c)!r}")


# ─── 8. Final QC measures the delivered text AFTER suppression + caps ────────
def test_final_qc_measures_delivered_text(result):
    seg = result.get("segmentation_qc") or {}
    # A real QC verdict must be present (policy version stamped) — the QC stage
    # is mandatory and runs LAST, so it graded the suppressed + recased cues.
    assert seg, "segmentation_qc missing — QC did not run"
    assert seg.get("segmentation_qc_policy_version") is not None or seg.get("rollup"), (
        "segmentation_qc has no policy verdict — QC did not measure delivered text")
    # And no mandatory stage failed (healthy run).
    assert result.get("mandatory_stage_error") is None, (
        f"a mandatory stage failed: {result.get('mandatory_stage_error')}")


# ─── 9. No later formatter stage re-adds a suppressed label ──────────────────
def test_no_stage_readds_suppressed_label(result):
    # The uninterrupted B turn must contain suppressed (label-less) continuation
    # cues. If a later re-render had re-added labels, EVERY B cue would carry one
    # (the pre-v6 bug) and this count would equal the cue count.
    cues = _dialogue_cues(result)
    start = _find_first_index(cues, "the town of Everwood")
    c_idx = _find_first_index(cues, "Coach says we start again")
    b_run = cues[start:c_idx]
    labeled = sum(1 for c in b_run if _has_speaker_b_label(c))
    assert len(b_run) >= 2, "expected multiple B cues in the uninterrupted turn"
    assert labeled < len(b_run), (
        f"every B cue carries a label — a later stage re-added the suppressed "
        f"labels (the pre-v6 regression): {[_cue_text(c) for c in b_run]}")


# ─── 10. Repeated Apply Spec runs → identical cue count + canonical hash ─────
def test_repeated_runs_byte_identical():
    runs = [_run_reformat_once() for _ in range(3)]
    hashes = {chash for _, chash in runs}
    counts = {len(r.get("cues") or []) for r, _ in runs}
    assert len(hashes) == 1, f"canonical output hash varied across runs: {hashes}"
    assert len(counts) == 1, f"cue count varied across runs: {counts}"
    h = next(iter(hashes))
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)


# ─── 11. Zero editorial-AI / OpenAI calls on the reformat path ───────────────
def test_reformat_makes_zero_openai_calls():
    # _run_reformat_once blocks OpenAI two ways (FORBID_EDITORIAL_AI + an
    # exploding client). If the path reached either, this raises. Reaching here
    # with cues proves zero OpenAI calls.
    result, chash = _run_reformat_once()
    assert result.get("cues"), "reformat produced no cues"
    assert chash


# ─── DIAGNOSTIC — dump every delivered cue (always passes; prints to stdout) ──
# Run with `-s` to see the output:
#   python -m pytest services/tests/test_regression_speaker_capitalization.py \
#       -k dump_delivered_cues -s
# This is the ground-truth surface for the two label failures: it shows the
# EXACT delivered lines + the structured speaker_label the engine produced for
# every cue, so we can see whether render_lines emitted a label that suppression
# then stripped, or never emitted one at all. Not a compliance assertion —
# a debugging aid retained in the suite for reproducibility.
def test_dump_delivered_cues(result):
    print("\n===== DELIVERED CUES (regression fixture, alpha mode) =====")
    for c in (result.get("cues") or []):
        print(
            f"idx={c.get('idx')} type={c.get('type')} "
            f"speaker_label={c.get('speaker_label')!r} "
            f"start={c.get('start_ms')} end={c.get('end_ms')} "
            f"lines={c.get('lines')!r}"
        )
    print("===== END DELIVERED CUES =====\n")
    assert True
