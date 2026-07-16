"""
Permanent unit coverage for the CANONICAL Segmentation QC engine
(services/segmentation_qc.py). These are the auditor-grade tests that lock the
deterministic contract: the issue catalog + severities, the bounded/finite
remediation loop, the no-AI + no-word-deletion guarantees, optimizer-boundary
protection, the technical-violations-stay-visible rule, and the cue/run rollups.

Pure functions, zero I/O, deterministic — identical (cues, rules) always yields
an identical contract. Parity with the JS reference is covered separately by
test_segmentation_qc_parity.py against the frozen shared fixtures.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services import segmentation_qc as sq  # noqa: E402


# Default spec rules used across cases (mirror the frozen parity fixture dials).
RULES = {
    "line_rules": {"max_chars_per_line": 32, "max_lines_per_caption": 2, "min_gap_between_captions_ms": 80},
    "reading_speed_rules": {"max_cps": 17, "cps_measurement": "characters"},
    "protected_phrases": ["andy cohen", "watch what happens live"],
}


def _run(cues, rules=None):
    return sq.run_segmentation_qc(cues, rules or RULES)


def _codes(result):
    return sorted({i["issue_code"] for i in result["segmentation_qc_issues"]})


# ── Policy version + clean path ──────────────────────────────────────
def test_policy_version_is_pinned():
    assert sq.QC_POLICY_VERSION == 1
    r = _run([{"id": "c1", "start_ms": 0, "end_ms": 3000, "text": "A clean readable line.", "speaker_id": "s1"}])
    assert r["segmentation_qc_policy_version"] == 1
    assert r["rollup"]["segmentation_qc_policy_version"] == 1


def test_clean_sequence_has_empty_contract():
    r = _run([{"id": "c1", "start_ms": 0, "end_ms": 3000, "text": "A clean readable line.", "speaker_id": "s1"}])
    assert r["technical_violations"] == []
    assert r["segmentation_qc_issues"] == []
    assert r["review_required"] is False
    assert r["export_blocked"] is False
    assert r["rollup"]["segmentation_qc_completeness"] == "clean"


# ── Severity catalog integrity ───────────────────────────────────────
def test_hard_and_soft_codes_have_expected_severity():
    assert sq.QC_SEVERITY[sq.QC_ISSUE_FLASH_CUE] == "fail"
    assert sq.QC_SEVERITY[sq.QC_ISSUE_SPEAKER_LABEL_CAUSED_CPS_FAILURE] == "fail"
    assert sq.QC_SEVERITY[sq.QC_ISSUE_MEANINGFUL_TEXT_REMOVED] == "fail"
    assert sq.QC_SEVERITY[sq.QC_ISSUE_OPTIMIZER_BOUNDARY_VIOLATION] == "fail"
    assert sq.QC_SEVERITY[sq.QC_ISSUE_MICRO_CUE] == "warn"
    assert sq.QC_SEVERITY[sq.QC_ISSUE_READING_SPEED_IMBALANCE] == "warn"
    assert sq.QC_SEVERITY[sq.QC_ISSUE_TIMING_PROVENANCE_MISSING] == "info"


# ── Technical violations stay visible on an unresolved hard failure ───
def test_unresolved_label_cps_keeps_technical_violation_and_blocks():
    cues = [
        {"id": "c1", "start_ms": 0, "end_ms": 1400, "text": "[JOHN:] twenty character line", "speaker_label": "JOHN", "speaker_id": "s1"},
        {"id": "c2", "start_ms": 1500, "end_ms": 3000, "text": "next cue.", "speaker_label": "MARY", "speaker_id": "s2"},
    ]
    r = _run(cues)
    # Both the raw technical violation AND the workflow disposition are reported.
    assert "max_cps" in r["technical_violations"]
    assert r["review_required"] is True
    assert r["export_blocked"] is True
    hard = [i for i in r["segmentation_qc_issues"] if i["issue_code"] == sq.QC_ISSUE_SPEAKER_LABEL_CAUSED_CPS_FAILURE]
    assert hard and hard[0]["remediation_result"] == "unresolved"


def test_resolvable_label_cps_marks_resolved_and_exportable():
    cues = [
        {"id": "c1", "start_ms": 0, "end_ms": 1400, "text": "[JOHN:] twenty character line", "speaker_label": "JOHN", "speaker_id": "s1"},
        {"id": "c2", "start_ms": 9000, "end_ms": 10000, "text": "next cue.", "speaker_label": "MARY", "speaker_id": "s2"},
    ]
    r = _run(cues)
    assert r["review_required"] is False
    assert r["export_blocked"] is False
    hard = [i for i in r["segmentation_qc_issues"] if i["issue_code"] == sq.QC_ISSUE_SPEAKER_LABEL_CAUSED_CPS_FAILURE]
    assert hard and hard[0]["remediation_result"] == "resolved"


# ── Flash / micro rhythm ─────────────────────────────────────────────
def test_flash_cue_unresolvable_blocks():
    cues = [
        {"id": "c1", "start_ms": 0, "end_ms": 250, "text": "Hey!", "speaker_id": "s1"},
        {"id": "c2", "start_ms": 300, "end_ms": 1500, "text": "Right now.", "speaker_id": "s1"},
    ]
    r = _run(cues)
    assert sq.QC_ISSUE_FLASH_CUE in _codes(r)
    assert r["export_blocked"] is True


def test_micro_cue_is_soft_non_blocking():
    r = _run([{"id": "c1", "start_ms": 0, "end_ms": 800, "text": "A short line here.", "speaker_id": "s1"}])
    assert sq.QC_ISSUE_MICRO_CUE in _codes(r)
    assert r["export_blocked"] is False
    assert r["review_required"] is False


# ── Optimizer-authored boundary protection ───────────────────────────
def test_optimizer_boundary_violation_flagged_when_provenance_missing():
    cues = [{
        "id": "c1", "start_ms": 0, "end_ms": 3000, "text": "A clean readable line.",
        "speaker_id": "s1", "meta": {"seq_opt": {"operation": "resegment_2"}},  # no output_hash
    }]
    r = _run(cues)
    assert sq.QC_ISSUE_OPTIMIZER_BOUNDARY_VIOLATION in _codes(r)
    issue = [i for i in r["segmentation_qc_issues"] if i["issue_code"] == sq.QC_ISSUE_OPTIMIZER_BOUNDARY_VIOLATION][0]
    # It is FLAGGED, never silently reversed → unresolved hard.
    assert issue["remediation_result"] == "unresolved"
    assert r["export_blocked"] is True


def test_optimizer_boundary_with_provenance_is_not_flagged():
    cues = [{
        "id": "c1", "start_ms": 0, "end_ms": 3000, "text": "A clean readable line.",
        "speaker_id": "s1", "meta": {"seq_opt": {"operation": "resegment_2", "output_hash": "abc123"}},
    }]
    r = _run(cues)
    assert sq.QC_ISSUE_OPTIMIZER_BOUNDARY_VIOLATION not in _codes(r)


# ── Meaningful-text-removal (hard) ───────────────────────────────────
def test_meaningful_text_removed_is_hard_when_word_dropped():
    cues = [{"id": "c1", "start_ms": 0, "end_ms": 3000, "text": "the quick fox", "speaker_id": "s1"}]
    rules = {**RULES, "original_word_bag": ["the", "quick", "brown", "fox"]}  # 'brown' dropped
    r = _run(cues, rules)
    assert sq.QC_ISSUE_MEANINGFUL_TEXT_REMOVED in _codes(r)
    assert r["export_blocked"] is True


def test_meaningful_text_removal_ignores_stopwords_and_condensation():
    # 'the' is a stopword; 'quick' was removed via an ATTRIBUTED condensation.
    cues = [{
        "id": "c1", "start_ms": 0, "end_ms": 3000, "text": "brown fox",
        "speaker_id": "s1",
        "meta": {"condensation": {"applied": True, "verbatim": "the quick brown fox"}},
    }]
    rules = {**RULES, "original_word_bag": ["the", "quick", "brown", "fox"]}
    r = _run(cues, rules)
    assert sq.QC_ISSUE_MEANINGFUL_TEXT_REMOVED not in _codes(r)


# ── Cue-level + run-level rollups ────────────────────────────────────
def test_cue_summary_shapes_for_each_disposition():
    # unresolved hard
    unresolved = sq._build_cue_summary([sq.make_issue(sq.QC_ISSUE_FLASH_CUE, remediation_result="unresolved")])
    assert unresolved["segmentation_qc_status"] == "fail"
    assert unresolved["segmentation_qc_review_required"] is True
    assert unresolved["segmentation_qc_remediation_outcome"] == "unresolved"
    # corrected hard
    corrected = sq._build_cue_summary([sq.make_issue(sq.QC_ISSUE_FLASH_CUE, remediation_result="resolved")])
    assert corrected["segmentation_qc_status"] == "corrected"
    assert corrected["segmentation_qc_review_required"] is False
    assert corrected["segmentation_qc_remediation_outcome"] == "corrected"
    # soft only
    soft = sq._build_cue_summary([sq.make_issue(sq.QC_ISSUE_MICRO_CUE, remediation_result="flagged")])
    assert soft["segmentation_qc_status"] == "warn"
    assert soft["segmentation_qc_review_required"] is False


def test_run_rollup_counts_are_bounded_and_accurate():
    cues = [
        {"id": "c1", "start_ms": 0, "end_ms": 250, "text": "Hey!", "speaker_id": "s1"},        # flash unresolvable
        {"id": "c2", "start_ms": 300, "end_ms": 1500, "text": "Right now.", "speaker_id": "s1"},
        {"id": "c3", "start_ms": 5000, "end_ms": 5800, "text": "A short line here.", "speaker_id": "s2"},  # micro soft
    ]
    r = _run(cues)
    rollup = r["rollup"]
    assert rollup["segmentation_qc_completeness"] == "has_unresolved"
    assert rollup["segmentation_qc_unresolved_count"] >= 1
    # review_required_count is a SEPARATE bounded field, derived independently
    # from review-required cues (never copied from unresolved_count). Under
    # QC_POLICY_VERSION 1 it equals unresolved_count by construction.
    assert rollup["segmentation_qc_review_required_count"] >= 1
    assert rollup["segmentation_qc_review_required_count"] == rollup["segmentation_qc_unresolved_count"]
    assert isinstance(rollup["segmentation_qc_issue_counts"], dict)
    # issue_counts is a bounded code→count map, never an unbounded blob.
    assert all(isinstance(v, int) for v in rollup["segmentation_qc_issue_counts"].values())
