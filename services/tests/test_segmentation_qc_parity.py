"""
Segmentation QC — PYTHON side of the FROZEN cross-engine parity harness.

This test and src/lib/__tests__/cc-segmentation-qc.test.js consume the SAME
version-controlled fixture — tests/fixtures/segmentation_qc_parity.json — and
must produce byte-identical NORMALIZED output. The fixture is the single source
of truth: it is NEVER regenerated from either engine's output, so a silent
regression in either runtime is caught the moment its output diverges from the
committed contract.

What this test proves for the Python authority (services.segmentation_qc):
  1. run_segmentation_qc(cues, rules) matches the fixture's expected run-level
     output (technical_violations, review_required, export_blocked, the ordered
     set of {issue_code, severity, remediation_attempted, remediation_result}).
  2. build_unresolved_untimed_evidence(...) matches the fixture's expected
     bounded evidence for every all-untimed group, including:
       • deterministic ug#N routing ids (array order),
       • segmentation_group_index DISTINCT from the routing ordinal,
       • bounded text truncation at UNRESOLVED_TEXT_MAX_CHARS,
       • missing optional provenance normalizing to null,
       • complete provenance copied through verbatim.
  3. FROZEN-WORDS NORMALIZATION: every group input is deep-frozen with
     services.immutable.freeze BEFORE being passed in, so `words` becomes a
     Python tuple. The engine must still emit a JSON ARRAY (the fixture's plain
     list) — proving the FrozenDict tuple-freeze does not change the normalized
     output. (This is the exact regression the list-only guard once caused:
     word_count → 0. It is now locked.)
  4. The QC_MAX_REMEDIATION_ATTEMPTS cap (locked at 1) is asserted INDIRECTLY
     through the "unresolved after allowed attempt" fixture, not by comparing
     the numeric constant alone — behavior is the contract.

Deterministic; no I/O beyond reading the committed fixture; no network. Runs in
the same environment as the sequence-optimizer + remediation-loop suites (none
of segmentation_qc's imports require `requests`).
"""

import json
import os

import pytest

from services.segmentation_qc import (
    run_segmentation_qc,
    build_unresolved_untimed_evidence,
    QC_POLICY_VERSION,
    QC_MAX_REMEDIATION_ATTEMPTS,
    UNRESOLVED_TEXT_MAX_CHARS,
)
from services.immutable import freeze


# ── Fixture loading ──────────────────────────────────────────────────────────

_FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "segmentation_qc_parity.json")

with open(_FIXTURE_PATH, "r", encoding="utf-8") as _fh:
    FIXTURE = json.load(_fh)

RULES = FIXTURE["rules"]


# ── Fixture-input expansion (kept DRY so no 2600-char literal lives in JSON) ──

def _expand_text_repeat(spec):
    """Materialize a { unit, count } repeat directive into its full string."""
    return str(spec["unit"]) * int(spec["count"])


def _expand_group(group):
    """Return a plain-dict copy of a fixture unresolved-group with any
    `_text_repeat` directive expanded into a real `text` field. Mirrors the JS
    parity test's expansion so both engines feed byte-identical inputs."""
    g = dict(group)
    tr = g.pop("_text_repeat", None)
    if tr is not None:
        g["text"] = _expand_text_repeat(tr)
    return g


def _expand_expected_evidence(ev):
    """Materialize any `_text_expect` directive in an expected-evidence record
    into the concrete `text` the engine must emit (truncated repeat string)."""
    e = dict(ev)
    te = e.pop("_text_expect", None)
    if te is not None:
        e["text"] = str(te["unit"]) * int(te["truncated_to"])
    return e


# ── Normalizers — collapse to the comparable JSON projection ─────────────────

def _norm_run(result):
    """Project the engine's run-level output onto the fixture's comparable shape:
    sorted technical_violations, the flags, the policy version, and the ordered
    set of issue {code, severity, sorted attempted ops, result}."""
    return {
        "tv": sorted(result["technical_violations"]),
        "rr": result["review_required"],
        "eb": result["export_blocked"],
        "pv": result["segmentation_qc_policy_version"],
        "issues": sorted(
            (
                {
                    "issue_code": i["issue_code"],
                    "severity": i["severity"],
                    "remediation_attempted": sorted(i["remediation_attempted"]),
                    "remediation_result": i["remediation_result"],
                }
                for i in result["segmentation_qc_issues"]
            ),
            key=lambda x: x["issue_code"] + x["remediation_result"],
        ),
    }


def _norm_expected_run(expect):
    return {
        "tv": sorted(expect["technical_violations"]),
        "rr": expect["review_required"],
        "eb": expect["export_blocked"],
        "pv": FIXTURE["policy_version"],
        "issues": sorted(
            (
                {
                    "issue_code": i["issue_code"],
                    "severity": i["severity"],
                    "remediation_attempted": sorted(i["remediation_attempted"]),
                    "remediation_result": i["remediation_result"],
                }
                for i in (expect.get("issues") or [])
            ),
            key=lambda x: x["issue_code"] + x["remediation_result"],
        ),
    }


def _rules_for_engine(unresolved_groups):
    """The rules dict the Python engine expects, threaded with the DEEP-FROZEN
    unresolved groups. Freezing here is the whole point of the frozen-words
    normalization check: the engine receives FrozenDict groups (words → tuple)
    and must still emit plain JSON arrays identical to the fixture."""
    return {
        "line_rules": RULES["line_rules"],
        "reading_speed_rules": RULES["reading_speed_rules"],
        "protected_phrases": RULES["protected_phrases"],
        "unresolved_groups": [freeze(_expand_group(g)) for g in (unresolved_groups or [])],
    }


# ── Static contract assertions ───────────────────────────────────────────────

def test_fixture_schema_version_present_and_pinned():
    # An explicit schema version so a future change cannot silently reinterpret
    # old fixture data. Bump it deliberately when the fixture shape changes.
    assert FIXTURE["fixture_schema_version"] == 1


def test_policy_version_matches_engine():
    assert QC_POLICY_VERSION == FIXTURE["policy_version"]


def test_max_remediation_attempts_locked_at_one():
    # The value lives in ONE central policy constant; the fixture records the
    # locked value so both engines are pinned to the same authority.
    assert QC_MAX_REMEDIATION_ATTEMPTS == 1
    assert FIXTURE["max_remediation_attempts"] == QC_MAX_REMEDIATION_ATTEMPTS


# ── Run-level parity — every fixture ─────────────────────────────────────────

@pytest.mark.parametrize("fx", FIXTURE["fixtures"], ids=[f["name"] for f in FIXTURE["fixtures"]])
def test_run_level_parity(fx):
    rules = _rules_for_engine(fx.get("unresolved_groups"))
    result = run_segmentation_qc(list(fx["cues"]), rules)
    assert _norm_run(result) == _norm_expected_run(fx["expect"]), (
        f"run-level parity drift on fixture '{fx['name']}'"
    )


# ── Evidence-level parity — every fixture that declares unresolved_evidence ──

_EVIDENCE_FIXTURES = [
    f for f in FIXTURE["fixtures"] if f["expect"].get("unresolved_evidence")
]


@pytest.mark.parametrize("fx", _EVIDENCE_FIXTURES, ids=[f["name"] for f in _EVIDENCE_FIXTURES])
def test_unresolved_evidence_parity(fx):
    groups = [freeze(_expand_group(g)) for g in fx["unresolved_groups"]]
    expected = [_expand_expected_evidence(e) for e in fx["expect"]["unresolved_evidence"]]
    assert len(groups) == len(expected)
    for routing_index, (group, exp) in enumerate(zip(groups, expected)):
        ev = build_unresolved_untimed_evidence(group, routing_index)
        # Deterministic ug#N routing id, distinct from segmentation_group_index.
        assert ev["unresolved_group_id"] == exp["unresolved_group_id"] == f"ug#{routing_index}"
        assert ev["segmentation_group_index"] == exp["segmentation_group_index"]
        assert ev["unresolved_group_id"] != str(ev["segmentation_group_index"])
        # FROZEN-WORDS NORMALIZATION: the engine received a FrozenDict (words is
        # a tuple) and must emit a JSON list identical to the fixture.
        assert isinstance(ev["words"], list)
        assert ev["words"] == exp["words"]
        assert ev["word_count"] == exp["word_count"]
        # Bounded text + truncation provenance.
        assert ev["text"] == exp["text"]
        assert ev["text_truncated"] == exp["text_truncated"]
        assert ev["original_text_length"] == exp["original_text_length"]
        if ev["text_truncated"]:
            assert len(ev["text"]) == UNRESOLVED_TEXT_MAX_CHARS
        # Provenance (present → copied through; missing → normalized to null).
        for key in (
            "provider", "model", "provider_version", "transcription_job_id",
            "confidence_source", "source_utterance_id", "speaker",
            "source_word_range", "reason", "disposition",
            "unresolved_group_version",
        ):
            assert ev[key] == exp[key], f"evidence field '{key}' drift in '{fx['name']}'"


# ── Frozen-input immutability guarantee (the contract that surfaced the bug) ──

def test_frozen_group_input_is_actually_immutable():
    """A frozen unresolved-group MUST reject mutation (FrozenDict), and the
    engine MUST still read it correctly. This is the runtime proof that the
    normalization tolerance is real, not incidental."""
    g = freeze({
        "unresolved_group_version": 1,
        "segmentation_group_index": 4,
        "words": ["Cookie", "?"],
        "text": "Cookie ?",
        "source_word_start": 5,
        "source_word_end": 7,
        "reason": "all_tokens_untimed",
    })
    with pytest.raises(TypeError):
        g["reason"] = "mutated"  # frozen — must raise
    ev = build_unresolved_untimed_evidence(g, 0)
    assert ev["words"] == ["Cookie", "?"]
    assert ev["word_count"] == 2


# ── Policy invariant: review_required_count == unresolved_count on every fixture ─

@pytest.mark.parametrize("fx", FIXTURE["fixtures"], ids=[f["name"] for f in FIXTURE["fixtures"]])
def test_v2_review_required_equals_unresolved(fx):
    assert QC_POLICY_VERSION == 3
    rules = _rules_for_engine(fx.get("unresolved_groups"))
    result = run_segmentation_qc(list(fx["cues"]), rules)
    rollup = result["rollup"]
    assert rollup["segmentation_qc_review_required_count"] == rollup["segmentation_qc_unresolved_count"]


# ── Attempt-cap behavior asserted INDIRECTLY (not by the numeric constant) ───

def test_attempt_cap_behavior_is_the_contract():
    """The 'unresolved after allowed attempt' fixture must stay unresolved and
    block export. This asserts the QC_MAX_REMEDIATION_ATTEMPTS=1 behavior
    through outcome, per the requirement to test the cap indirectly."""
    fx = next(f for f in FIXTURE["fixtures"]
              if f["name"] == "hard_issue_unresolved_after_allowed_attempt")
    rules = _rules_for_engine(fx.get("unresolved_groups"))
    result = run_segmentation_qc(list(fx["cues"]), rules)
    assert result["review_required"] is True
    assert result["export_blocked"] is True
    flash = next(i for i in result["segmentation_qc_issues"] if i["issue_code"] == "FLASH_CUE")
    assert flash["remediation_result"] == "unresolved"

    # And the resolvable twin (idle room present) resolves within the one attempt.
    fx_ok = next(f for f in FIXTURE["fixtures"]
                 if f["name"] == "hard_issue_resolved_first_attempt")
    result_ok = run_segmentation_qc(list(fx_ok["cues"]), _rules_for_engine(None))
    assert result_ok["review_required"] is False
    flash_ok = next(i for i in result_ok["segmentation_qc_issues"] if i["issue_code"] == "FLASH_CUE")
    assert flash_ok["remediation_result"] == "resolved"
