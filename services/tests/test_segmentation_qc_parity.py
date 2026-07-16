"""
CROSS-LANGUAGE PARITY — the canonical Python Segmentation QC engine
(services/segmentation_qc.py) MUST produce the same NORMALIZED public contract
as the JS reference (src/lib/cc-segmentation-qc.js) for every FROZEN shared
fixture. The identical fixture file is consumed by the JS suite
(src/lib/__tests__/cc-segmentation-qc.test.js).

Normalized contract compared per fixture:
  • sorted technical_violations
  • the SET of {issue_code, severity, sorted(remediation_attempted), remediation_result}
  • review_required
  • export_blocked
  • segmentation_qc_policy_version

Internal debug fields (evidence blobs, metrics, hashes) are intentionally NOT
compared — only the normalized contract. NEVER renumber issue codes.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services import segmentation_qc as sq  # noqa: E402

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "segmentation_qc_parity.json")


def _load():
    with open(FIXTURE_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _normalize_issue(issue):
    return (
        issue["issue_code"],
        issue["severity"],
        tuple(sorted(issue.get("remediation_attempted") or [])),
        issue["remediation_result"],
    )


def _normalize_result(result):
    return {
        "technical_violations": sorted(result["technical_violations"]),
        "review_required": result["review_required"],
        "export_blocked": result["export_blocked"],
        "policy_version": result["segmentation_qc_policy_version"],
        "issues": sorted(_normalize_issue(i) for i in result["segmentation_qc_issues"]),
    }


def _normalize_expected(expect, policy_version):
    return {
        "technical_violations": sorted(expect["technical_violations"]),
        "review_required": expect["review_required"],
        "export_blocked": expect["export_blocked"],
        "policy_version": policy_version,
        "issues": sorted(_normalize_issue(i) for i in expect["issues"]),
    }


def test_fixture_file_is_present_and_versioned():
    data = _load()
    assert data["policy_version"] == sq.QC_POLICY_VERSION
    assert len(data["fixtures"]) == 12


def test_all_fixtures_match_normalized_contract():
    data = _load()
    rules = data["rules"]
    failures = []
    for fx in data["fixtures"]:
        result = sq.run_segmentation_qc(fx["cues"], rules)
        got = _normalize_result(result)
        want = _normalize_expected(fx["expect"], data["policy_version"])
        if got != want:
            failures.append((fx["name"], want, got))
    assert not failures, "PARITY MISMATCH:\n" + "\n".join(
        f"[{n}]\n  expected={w}\n  got     ={g}" for n, w, g in failures
    )


def test_review_required_count_and_unresolved_count_both_present_on_every_fixture():
    """The rollup MUST carry BOTH bounded counts on every fixture (the fields are
    persisted separately on CCFormatRun). Proves neither is dropped."""
    data = _load()
    rules = data["rules"]
    for fx in data["fixtures"]:
        rollup = sq.run_segmentation_qc(fx["cues"], rules)["rollup"]
        assert "segmentation_qc_unresolved_count" in rollup, fx["name"]
        assert "segmentation_qc_review_required_count" in rollup, fx["name"]
        assert isinstance(rollup["segmentation_qc_unresolved_count"], int)
        assert isinstance(rollup["segmentation_qc_review_required_count"], int)


def test_v1_invariant_review_required_count_equals_unresolved_count():
    """POLICY-VERSION-1 INVARIANT (versioned, NOT a permanent semantic guarantee):
    under QC_POLICY_VERSION 1 a cue is review-required IFF it carries an unresolved
    hard defect, so review_required_count == unresolved_count on EVERY fixture. A
    future policy may diverge them (a corrected-but-reviewable / soft-but-reviewable
    cue) — at which point this test is updated for the new policy version. The two
    fields exist separately precisely so that divergence needs no schema migration."""
    data = _load()
    assert sq.QC_POLICY_VERSION == 1, "This invariant is pinned to QC_POLICY_VERSION 1; update it when the policy bumps."
    rules = data["rules"]
    for fx in data["fixtures"]:
        rollup = sq.run_segmentation_qc(fx["cues"], rules)["rollup"]
        assert (
            rollup["segmentation_qc_review_required_count"]
            == rollup["segmentation_qc_unresolved_count"]
        ), f"[{fx['name']}] v1 equivalence broken: review_required={rollup['segmentation_qc_review_required_count']} unresolved={rollup['segmentation_qc_unresolved_count']}"
