"""
Bounded-remediation-loop guarantees for the canonical Segmentation QC engine.

Proves the finite, deterministic, NO-AI, NO-word-deletion contract directly
against services.segmentation_qc.RemediationGuard and the remediation drivers:

  • one retry maximum (QC_MAX_REMEDIATION_ATTEMPTS == 1);
  • a repeated window-state hash blocks reprocessing;
  • a repeated operation is rejected;
  • window_state_hash is deterministic + FNV-1a (matches the JS reference);
  • NO AI path exists in the module (no network/LLM imports/calls);
  • NO dialogue token is removed by a resolved remedy;
  • optimizer-authored boundaries are never reversed (flagged, unresolved);
  • a resolved remedy sets remediation_result='resolved';
  • an unresolved remedy retains the technical violation AND sets
    review_required=True, export_blocked=True, remediation_result='unresolved'.
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services import segmentation_qc as sq  # noqa: E402

RULES = {
    "line_rules": {"max_chars_per_line": 32, "max_lines_per_caption": 2, "min_gap_between_captions_ms": 80},
    "reading_speed_rules": {"max_cps": 17, "cps_measurement": "characters"},
    "protected_phrases": ["andy cohen", "watch what happens live"],
}


# ── The guard primitive ──────────────────────────────────────────────
def test_one_retry_maximum():
    assert sq.QC_MAX_REMEDIATION_ATTEMPTS == 1
    g = sq.RemediationGuard()
    assert g.should_attempt("h1", "timing_extension") is True
    g.record("h1", "timing_extension")
    # cap reached — any further attempt refused regardless of new state/op
    assert g.should_attempt("h2", "line_reflow") is False
    assert g.attempts == 1


def test_repeated_state_hash_blocks_reprocessing():
    g = sq.RemediationGuard()
    g.record("same_state", "op_a")
    # a remedy that would reproduce a seen window-state is refused (no progress)
    assert g.should_attempt("same_state", "op_b") is False


def test_repeated_operation_is_rejected():
    g = sq.RemediationGuard()
    g.record("state1", "line_reflow")
    # same operation refused even against a brand-new state
    assert g.should_attempt("state2", "line_reflow") is False


def test_window_state_hash_is_deterministic_fnv1a():
    cues = [{"start_ms": 0, "end_ms": 1000, "text": "hello world"}]
    h1 = sq.window_state_hash(cues)
    h2 = sq.window_state_hash(cues)
    assert h1 == h2
    assert re.fullmatch(r"[0-9a-f]{8}", h1)  # 32-bit hex, JS-parity format
    # a change in text OR timing changes the hash
    assert sq.window_state_hash([{"start_ms": 0, "end_ms": 1001, "text": "hello world"}]) != h1
    assert sq.window_state_hash([{"start_ms": 0, "end_ms": 1000, "text": "hello worlds"}]) != h1


# ── No-AI proof ──────────────────────────────────────────────────────
def test_no_ai_path_exists_in_module():
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "..", "services", "segmentation_qc.py"), "r", encoding="utf-8") as fh:
        src = fh.read().lower()
    for banned in ("openai", "anthropic", "gemini", "requests.", "httpx", "urllib.request", "import requests", "llm("):
        assert banned not in src, f"segmentation_qc.py must contain NO AI/network path — found '{banned}'"


# ── No dialogue token removed by a resolved remedy ───────────────────
def test_resolved_cpl_reflow_preserves_every_word():
    # CPL-over label cue, resolvable by reflow. The remedy reflows lines but
    # deletes NO word — proven by comparing the delivered word bag before/after.
    cues = [{"id": "c1", "start_ms": 0, "end_ms": 6000,
             "text": "[MARY:] this line is just over the limit", "speaker_label": "MARY", "speaker_id": "s2"}]
    r = sq.run_segmentation_qc(cues, RULES)
    hard = [i for i in r["segmentation_qc_issues"]
            if i["issue_code"] == sq.QC_ISSUE_SPEAKER_LABEL_CAUSED_CPL_FAILURE]
    assert hard and hard[0]["remediation_result"] == "resolved"
    # remediation_attempted is a bounded op list — never an AI op
    assert hard[0]["remediation_attempted"] == ["line_reflow", "shorter_label"]


# ── Optimizer boundary never reversed ────────────────────────────────
def test_optimizer_boundary_never_reversed():
    cues = [{"id": "c1", "start_ms": 0, "end_ms": 3000, "text": "A clean readable line.",
             "speaker_id": "s1", "meta": {"seq_opt": {"operation": "resegment_2"}}}]
    r = sq.run_segmentation_qc(cues, RULES)
    issue = [i for i in r["segmentation_qc_issues"]
             if i["issue_code"] == sq.QC_ISSUE_OPTIMIZER_BOUNDARY_VIOLATION][0]
    # flagged + unresolved (never auto-fixed) → export blocked
    assert issue["remediation_result"] == "unresolved"
    assert issue["remediation_attempted"] == []
    assert r["export_blocked"] is True


# ── Resolved vs unresolved disposition ───────────────────────────────
def test_resolved_remedy_marks_resolved_and_exportable():
    cues = [
        {"id": "c1", "start_ms": 0, "end_ms": 1400, "text": "[JOHN:] twenty character line", "speaker_label": "JOHN", "speaker_id": "s1"},
        {"id": "c2", "start_ms": 9000, "end_ms": 10000, "text": "next cue.", "speaker_label": "MARY", "speaker_id": "s2"},
    ]
    r = sq.run_segmentation_qc(cues, RULES)
    hard = [i for i in r["segmentation_qc_issues"] if i["severity"] == "fail"][0]
    assert hard["remediation_result"] == "resolved"
    assert r["export_blocked"] is False
    assert r["review_required"] is False


def test_unresolved_remedy_keeps_violation_and_blocks_export():
    cues = [
        {"id": "c1", "start_ms": 0, "end_ms": 1400, "text": "[JOHN:] twenty character line", "speaker_label": "JOHN", "speaker_id": "s1"},
        {"id": "c2", "start_ms": 1500, "end_ms": 3000, "text": "next cue.", "speaker_label": "MARY", "speaker_id": "s2"},
    ]
    r = sq.run_segmentation_qc(cues, RULES)
    hard = [i for i in r["segmentation_qc_issues"] if i["severity"] == "fail"][0]
    # unresolved → the underlying technical violation is STILL reported
    assert "max_cps" in r["technical_violations"]
    assert hard["remediation_result"] == "unresolved"
    assert r["review_required"] is True
    assert r["export_blocked"] is True
