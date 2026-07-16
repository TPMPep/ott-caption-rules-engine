"""
export_action_key.py — CANONICAL export-ACTION identity (Python twin of
lib/cc-export-action-key.js). Byte-identical SHA-256 output for every payload.

This is the ENGINE-SIDE twin of the JavaScript export-action key helper. It
exists so the rules-engine repo can compute / verify the same export action_key
the Base44 ccExportProject function produces, and so tests/test_export_action_key.py
can lock cross-language parity against the frozen vectors shared with
lib/__tests__/cc-export-action-key.test.js.

WHAT IT IS NOT: the action_key is NOT a concurrency mechanism. Exactly-once
export behavior is provided by the CCExportAction compare-and-set claim rows
(winner election + loser replay + stale recovery). The key only groups the rows
that are contenders for the SAME action.

Canonicalization reuses services/canonical_hash.py (the single source of truth
for audit-identity hashing): schema prefix 'cchash/v1\\n', sorted keys, NFC
strings, integral-number normalization, fixed separators, full 64-char
lowercase-hex SHA-256. The rendered-bytes hash is NEVER an input (that is a
receipt, not identity). SOC 2 CC7.2 / CC8.1. FCC 47 CFR §79.1.
"""

from __future__ import annotations

from typing import Any, Dict

from .canonical_hash import canonical_sha256

# Must equal EXPORT_ACTION_KEY_VERSION in lib/cc-export-action-key.js. Lives
# inside the hashed payload so a future shape change can't collide with a v1 key.
EXPORT_ACTION_KEY_VERSION = 1


def build_export_action_payload(p: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Build the ORDERED canonical payload for an export action. Pure +
    deterministic. Normalizes every identity-shaping input (strip/lower/sort/
    dedupe) so trivially-different requests collapse to ONE action and
    identity-shaping ones fork. NO artifact/byte-hash field — identity only.
    Mirrors buildExportActionPayload in lib/cc-export-action-key.js."""
    p = p or {}

    def s(key: str) -> str:
        v = p.get(key)
        return "" if v is None else str(v)

    spec_version = p.get("spec_version")
    if not isinstance(spec_version, int) or isinstance(spec_version, bool):
        spec_version = 0

    raw_codes = p.get("issue_codes") or []
    # De-dupe preserving nothing (we sort), then sort ascending — matches JS
    # [...new Set(...)].sort() (lexicographic string sort).
    issue_codes = sorted({str(c) for c in raw_codes})

    return {
        "schema": "ccexp",
        "v": EXPORT_ACTION_KEY_VERSION,
        "project_id": s("project_id"),
        "format_run_id": s("format_run_id"),
        "format": s("format").strip().lower(),
        "spec_slug": s("spec_slug"),
        "spec_version": spec_version,
        "override": p.get("override") is True,
        "override_reason": s("override_reason").strip().lower(),
        "issue_codes": issue_codes,
        "actor_id": s("actor_id"),
        "actor_email": s("actor_email").strip().lower(),
        "client_action_id": s("client_action_id").strip(),
    }


def compute_export_action_key(p: Dict[str, Any] | None = None) -> str:
    """Compute the collision-resistant export action key: 'ccexp:<64-hex SHA-256>'.
    Deterministic + cross-language-identical with computeExportActionKey (JS)."""
    return f"ccexp:{canonical_sha256(build_export_action_payload(p))}"
