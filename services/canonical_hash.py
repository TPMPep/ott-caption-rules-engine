"""
canonical_hash.py — COLLISION-RESISTANT canonical hashing for segmentation audit
identity. THE single source of truth for every hash that participates in audit
identity or idempotency: input_hash, candidate_set_hash, output_hash, and the
CCSegmentationDecision.decision_key.

WHY SHA-256 (not FNV-1a)
────────────────────────
FNV-1a is a 32-bit NON-cryptographic hash: ~65k inputs give a ~40% collision
chance (birthday bound). For an audit identity that MUST be unique per decision
across a whole project's history — and for an idempotency key that decides
whether a second ingestion creates a duplicate row — a 32-bit space is
indefensible. Two logically-distinct windows colliding would silently drop or
overwrite a decision. SHA-256 (256-bit) makes accidental collision impossible in
practice and is the auditor-defensible choice. SOC 2 CC7.2 / CC8.1.

CANONICALIZATION CONTRACT (must match cc-segmentation-audit.js byte-for-byte)
─────────────────────────────────────────────────────────────────────────────
The hash is SHA-256 over a DETERMINISTIC canonical JSON serialization of the
payload, so the SAME logical input hashes IDENTICALLY in Python and JavaScript:

  1. A schema/version PREFIX is prepended to the hashed bytes:
        "cchash/v1\n" + canonical_json
     so a future canonicalization change can never silently collide with a v1
     hash — the version is INSIDE the hashed bytes.
  2. Object keys are sorted ascending (stable ordering).
  3. Strings are NFC-normalized (so a composed vs decomposed accent can't fork).
  4. Numbers are normalized: integral floats render as integers (2.0 → "2"),
     non-integral floats use repr's shortest round-trip form; NaN/Infinity are
     forbidden (raise) — they have no canonical form.
  5. No timestamps, no transient fields — the CALLER is responsible for passing
     only deterministic inputs. This module never injects a clock value.
  6. Separators are fixed (',' and ':') with NO whitespace, so formatting can't
     fork the bytes.
  7. Output is lowercase hex, full 64-char SHA-256 digest.

A short 8-char PREFIX of the digest MAY be used for internal cache bucketing
ONLY (see short_bucket) — NEVER for audit identity or idempotency.
"""

import hashlib
import json
import math
import unicodedata
from typing import Any

CANONICAL_HASH_VERSION = 1
_PREFIX = f"cchash/v{CANONICAL_HASH_VERSION}\n"


def _normalize(value: Any) -> Any:
    """Recursively normalize a value into a canonical, JSON-serializable form.
    Deterministic: dict keys sorted, strings NFC-normalized, numbers normalized,
    NaN/Infinity rejected. Mirrors _canonicalize in cc-segmentation-audit.js."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("NaN/Infinity have no canonical hash form")
        # Integral float → int (2.0 == 2), else shortest round-trip repr.
        if value.is_integer():
            return int(value)
        return float(repr(value))
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        # Sort keys ascending for stable ordering; coerce keys to str.
        return {str(k): _normalize(value[k]) for k in sorted(value.keys(), key=lambda x: str(x))}
    # Fallback — stringify unknown types deterministically.
    return unicodedata.normalize("NFC", str(value))


def canonical_json(payload: Any) -> str:
    """Deterministic canonical JSON string (sorted keys, fixed separators, no
    whitespace, normalized values). NOT prefixed — the raw canonical form."""
    normalized = _normalize(payload)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_sha256(payload: Any) -> str:
    """Full 64-char lowercase-hex SHA-256 over the version-prefixed canonical
    serialization. THE audit-identity / idempotency hash. Deterministic and
    cross-language-identical with cc-segmentation-audit.js canonicalSha256."""
    body = _PREFIX + canonical_json(payload)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def short_bucket(payload: Any) -> str:
    """8-char prefix of the canonical digest — for INTERNAL CACHE BUCKETING ONLY.
    NEVER use for audit identity or idempotency (truncation reintroduces
    collision risk). Named explicitly so a reviewer can grep for misuse."""
    return canonical_sha256(payload)[:8]
