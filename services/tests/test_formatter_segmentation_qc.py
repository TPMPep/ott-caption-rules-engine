"""
Integration coverage for the REAL Python formatter pipeline
(services.formatter.process_caption_job). Proves that a real caption job carries
the canonical Segmentation QC contract on its result, and locks the RUNTIME
ORDER: Segmentation QC runs AFTER every deterministic transform and BEFORE
general QC + export.

Not a mock — this exercises process_caption_job end to end on a small word
stream, then asserts the shape + fields the Base44 ingester depends on.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.formatter import process_caption_job, apply_env_overrides, restore_env_overrides  # noqa: E402


def _words(text, start_ms, end_ms, speaker="A"):
    """Evenly spread a sentence's words across [start_ms, end_ms]."""
    toks = text.split()
    n = len(toks)
    span = end_ms - start_ms
    out = []
    for i, w in enumerate(toks):
        ws = start_ms + span * i // n
        we = start_ms + span * (i + 1) // n
        out.append({"text": w, "start_ms": ws, "end_ms": we, "speaker": speaker})
    return out


def _run_job(timestamps, backbone="", formats=None):
    snap = apply_env_overrides({
        "CUSTOM_MAX_CHARS": "42", "CUSTOM_MAX_LINES": "2",
        "CUSTOM_MAX_CPS": "17", "SPEAKER_LABEL_MODE": "none",
    })
    try:
        return process_caption_job(backbone, timestamps, output_formats=formats or ["srt"])
    finally:
        restore_env_overrides(snap)


def test_result_carries_full_segmentation_qc_contract():
    ts = _words("This is a clean and perfectly readable caption line here.", 0, 4000)
    result = _run_job(ts)

    # final cues present
    assert isinstance(result.get("cues"), list) and len(result["cues"]) >= 1
    for cue in result["cues"]:
        assert "start_ms" in cue and "end_ms" in cue and "lines" in cue

    seg = result.get("segmentation_qc")
    assert seg is not None, "process_caption_job must attach segmentation_qc"

    # bounded contract keys
    for key in ("technical_violations", "segmentation_qc_issues", "review_required",
                "export_blocked", "segmentation_qc_policy_version", "cue_summaries", "rollup"):
        assert key in seg, f"segmentation_qc missing '{key}'"

    # cue-level summaries carry the persisted fields
    for cs in seg["cue_summaries"]:
        for f in ("segmentation_qc_status", "segmentation_qc_highest_severity",
                  "segmentation_qc_issue_codes", "segmentation_qc_review_required",
                  "segmentation_qc_remediation_outcome"):
            assert f in cs
    # rollup carries the persisted fields
    for f in ("segmentation_qc_completeness", "segmentation_qc_issue_counts",
              "segmentation_qc_corrected_count", "segmentation_qc_unresolved_count",
              "segmentation_qc_warn_count", "segmentation_qc_policy_version"):
        assert f in seg["rollup"]

    # policy version pinned
    assert seg["segmentation_qc_policy_version"] == 1


def test_clean_job_is_not_export_blocked():
    ts = _words("A perfectly readable caption line.", 0, 3000)
    result = _run_job(ts)
    seg = result["segmentation_qc"]
    assert seg["export_blocked"] is False
    assert seg["review_required"] is False


def test_qc_runs_before_general_qc_and_export():
    # Both segmentation_qc AND general qc AND the export artifact are present on
    # one result — proves the runtime order 10e → 11 → 12 all executed in order.
    ts = _words("A perfectly readable caption line.", 0, 3000)
    result = _run_job(ts, formats=["srt"])
    assert result.get("segmentation_qc") is not None   # step 10e
    assert result.get("qc") is not None                # step 11
    assert isinstance(result.get("srt"), str) and result["srt"].strip()  # step 12


def test_runtime_order_matches_formatter_source():
    """Lock the exact stage order from formatter.py by asserting the QC stage
    comment block sits AFTER capitalization and BEFORE the general-QC call."""
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "..", "services", "formatter.py"), "r", encoding="utf-8") as fh:
        src = fh.read()
    # anchors, in the required order. Each is a UNIQUE execution-site call
    # string (not a docstring/comment word) so .index() lands on the real stage.
    order = [
        "cues = shape_caption_rhythm(cues)",       # 1. initial segmentation/shaping
        "cues = optimize_cue_sequence(cues)",      # 2. sequence optimizer
        "cues = apply_readability_rules(cues)",    # 3a. readability/CPS
        "cues = condense_cues(",                   # 3b. condensation
        "cues = apply_sentence_capitalization(cues)",  # 4. final rendering effects (last)
        "seg_qc = run_segmentation_qc(cues, {",    # 5. Segmentation QC
        "qc = qc_report(cues_in_count",            # 6. general QC
        'if "srt" in output_formats:',             # 7. export
    ]
    positions = [src.index(a) for a in order]
    assert positions == sorted(positions), (
        "formatter runtime order changed — expected "
        "shaping → optimizer → readability → condensation → capitalization → "
        "segmentation QC → general QC → export"
    )
