"""
test_export_action_key.py — proves the Python export-action key
(services/export_action_key.py) produces collision-resistant SHA-256 identities
BYTE-IDENTICAL to the JavaScript twin (lib/cc-export-action-key.js).

FROZEN_VECTORS below are the SAME digests asserted in
lib/__tests__/cc-export-action-key.test.js. If the Python or JS canonicalization
ever forks, a vector fails in one language and the cross-language CI guarantee
catches it. SOC 2 CC7.2 (exactly-once) / CC8.1 (attributable + reproducible).

Run: python -m pytest tests/test_export_action_key.py -q
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.export_action_key import (  # noqa: E402
    compute_export_action_key,
    build_export_action_payload,
    EXPORT_ACTION_KEY_VERSION,
)

# FROZEN — MUST match lib/__tests__/cc-export-action-key.test.js exactly.
FROZEN_VECTORS = {
    "clean_minimal": (
        {"project_id": "proj_1", "format": "srt", "actor_id": "u1", "actor_email": "a@x.com"},
        "ccexp:cbeaf53abdebbebca25bff1e820515a5b73ae68bec1ff9772fec609003e61a51",
    ),
    "clean_with_spec": (
        {"project_id": "proj_1", "format_run_id": "run_1", "format": "srt", "spec_slug": "netflix-ttal", "spec_version": 3, "actor_id": "u1", "actor_email": "a@x.com"},
        "ccexp:d0a366335ea1276c5a889d0b3240f8d2d63c53d01c7feac9587c9fd8ef28a367",
    ),
    "override_full": (
        {"project_id": "proj_9", "format_run_id": "run_9", "format": "scc", "spec_slug": "fcc", "spec_version": 2, "override": True, "override_reason": "Client-critical; label-CPS cue reviewed.", "issue_codes": ["SPEAKER_LABEL_CAUSED_CPS_FAILURE", "FLASH_CUE"], "actor_id": "u9", "actor_email": "Admin@X.com", "client_action_id": "act-123"},
        "ccexp:23c3f77fabb138de911d3ab5ce7bfe56e6348eea453ea393194fa50d2d813ae9",
    ),
    # Messy inputs (padding, case, unsorted + duplicate codes) → MUST equal override_full.
    "override_full_messy": (
        {"project_id": "proj_9", "format_run_id": "run_9", "format": "  SCC ", "spec_slug": "fcc", "spec_version": 2, "override": True, "override_reason": "  Client-critical; label-CPS cue reviewed.  ", "issue_codes": ["FLASH_CUE", "SPEAKER_LABEL_CAUSED_CPS_FAILURE", "FLASH_CUE"], "actor_id": "u9", "actor_email": "  admin@x.com  ", "client_action_id": " act-123 "},
        "ccexp:23c3f77fabb138de911d3ab5ce7bfe56e6348eea453ea393194fa50d2d813ae9",
    ),
    "distinct_client_action": (
        {"project_id": "proj_9", "format_run_id": "run_9", "format": "scc", "spec_slug": "fcc", "spec_version": 2, "override": True, "override_reason": "Client-critical; label-CPS cue reviewed.", "issue_codes": ["SPEAKER_LABEL_CAUSED_CPS_FAILURE", "FLASH_CUE"], "actor_id": "u9", "actor_email": "admin@x.com", "client_action_id": "act-456"},
        "ccexp:9676fcb1e559ed0902fe30cb25d5e0ee613a29a1450959ddb8321abb9176bbf1",
    ),
}


def test_frozen_vectors_match_js():
    for label, (payload, expected) in FROZEN_VECTORS.items():
        got = compute_export_action_key(payload)
        assert got == expected, f"{label}: {got} != {expected}"


def test_is_full_sha256_with_prefix():
    k = compute_export_action_key({"project_id": "p", "format": "srt", "actor_id": "u", "actor_email": "u@x.com"})
    assert k.startswith("ccexp:")
    hexpart = k.split(":", 1)[1]
    assert len(hexpart) == 64
    assert all(c in "0123456789abcdef" for c in hexpart)


def test_deterministic():
    p = {"project_id": "p", "format": "srt", "actor_id": "u", "actor_email": "u@x.com"}
    assert compute_export_action_key(p) == compute_export_action_key(p)


def test_normalization_collapses_trivial_differences():
    a = compute_export_action_key({"project_id": "p", "format": "srt", "actor_id": "u", "actor_email": "u@x.com", "override": True, "override_reason": "Reason X", "issue_codes": ["B", "A"]})
    b = compute_export_action_key({"project_id": "p", "format": "SRT", "actor_id": "u", "actor_email": "U@X.com", "override": True, "override_reason": " reason x ", "issue_codes": ["A", "B", "A"]})
    assert a == b


def test_client_action_id_forks_key():
    base = {"project_id": "p", "format": "srt", "actor_id": "u", "actor_email": "u@x.com"}
    assert compute_export_action_key({**base, "client_action_id": "one"}) != compute_export_action_key({**base, "client_action_id": "two"})


def test_identity_fields_fork_key():
    base = {"project_id": "p", "format_run_id": "r", "format": "srt", "spec_slug": "s", "spec_version": 1, "actor_id": "u", "actor_email": "u@x.com"}
    a = compute_export_action_key(base)
    assert compute_export_action_key({**base, "override": True, "override_reason": "x", "issue_codes": ["FLASH_CUE"]}) != a
    assert compute_export_action_key({**base, "format": "scc"}) != a
    assert compute_export_action_key({**base, "actor_email": "other@x.com"}) != a
    assert compute_export_action_key({**base, "format_run_id": "r2"}) != a


def test_payload_never_contains_artifact_bytes():
    payload = build_export_action_payload({"project_id": "p", "format": "srt", "actor_id": "u", "actor_email": "u@x.com", "artifact_id": "cchash:deadbeef", "byte_hash": "deadbeef"})
    assert "artifact_id" not in payload
    assert "byte_hash" not in payload
    assert payload["v"] == EXPORT_ACTION_KEY_VERSION
    assert payload["schema"] == "ccexp"


if __name__ == "__main__":
    for name in list(globals()):
        if name.startswith("test_"):
            globals()[name]()
    print("export_action_key parity: ALL PASS")
