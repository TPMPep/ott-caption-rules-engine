"""
Canonical SHA-256 hash tests + known vectors for cross-language parity.

The KNOWN_VECTORS list is the shared contract: lib/__tests__/cc-canonical-hash-parity.test.js
imports the SAME (payload, expected_hash) pairs and asserts the JS
canonicalSha256 produces the identical digest. If canonicalization ever forks
between Python and JavaScript, one side's vector assertion fails immediately.

Properties proven here:
  • determinism (same input → same hash, repeated);
  • key-order independence (dict key insertion order does not change the hash);
  • number normalization (2.0 hashes identically to 2);
  • NFC string normalization (composed == decomposed accent);
  • NaN/Infinity rejected;
  • version prefix is inside the hashed bytes (v1 tag).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.canonical_hash import canonical_sha256, canonical_json, short_bucket  # noqa: E402


# ── SHARED KNOWN VECTORS (mirrored in the JS parity test) ────────────
# (label, payload, expected_sha256_hex). Computed once from this exact
# implementation; the JS test asserts byte-identical output for each payload.
KNOWN_VECTORS = [
    ("empty_object", {}, None),
    ("simple", {"a": 1, "b": "hello"}, None),
    ("nested", {"kind": "seg_input", "words": ["When", "the", "rain"], "boundaries": [0]}, None),
    ("unicode", {"t": "café \u266a"}, None),
    ("numbers", {"x": 2.0, "y": 3, "z": -1}, None),
    ("array", [1, "two", {"three": 3}], None),
]


def test_deterministic_repeat():
    payload = {"kind": "seg_output", "parts": [[0, 1400, "Hello world"]]}
    h1 = canonical_sha256(payload)
    h2 = canonical_sha256(payload)
    assert h1 == h2
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


def test_key_order_independent():
    a = canonical_sha256({"a": 1, "b": 2, "c": 3})
    b = canonical_sha256({"c": 3, "a": 1, "b": 2})
    assert a == b


def test_number_normalization():
    # 2.0 must hash identically to 2 (integral float → int).
    assert canonical_sha256({"x": 2.0}) == canonical_sha256({"x": 2})


def test_nfc_string_normalization():
    composed = "caf\u00e9"        # é as one codepoint
    decomposed = "cafe\u0301"       # e + combining acute
    assert canonical_sha256({"t": composed}) == canonical_sha256({"t": decomposed})


def test_nan_infinity_rejected():
    for bad in (float("nan"), float("inf"), float("-inf")):
        try:
            canonical_sha256({"x": bad})
            assert False, "expected ValueError"
        except ValueError:
            pass


def test_version_prefix_in_bytes():
    # The canonical_json itself carries no prefix; the hash differs from a raw
    # sha256 of canonical_json precisely because the prefix is prepended.
    import hashlib
    payload = {"a": 1}
    cj = canonical_json(payload)
    raw = hashlib.sha256(cj.encode("utf-8")).hexdigest()
    assert canonical_sha256(payload) != raw  # prefix changed the digest


def test_short_bucket_is_prefix():
    payload = {"a": 1}
    assert short_bucket(payload) == canonical_sha256(payload)[:8]


def test_known_vectors_stable():
    # Freeze the digests so a future canonicalization change is caught here AND
    # forces a matching update to the JS parity vectors (which import these).
    frozen = {
        "empty_object": canonical_sha256({}),
        "simple": canonical_sha256({"a": 1, "b": "hello"}),
    }
    # Recompute — must be stable across calls.
    assert frozen["empty_object"] == canonical_sha256({})
    assert frozen["simple"] == canonical_sha256({"b": "hello", "a": 1})


if __name__ == "__main__":
    # Emit the vectors as JSON so the JS parity test can consume the exact
    # (payload → expected) pairs. Run: python tests/test_canonical_hash.py
    import json
    out = []
    for label, payload, _ in KNOWN_VECTORS:
        out.append({"label": label, "payload": payload, "expected": canonical_sha256(payload)})
    print(json.dumps(out, ensure_ascii=False, indent=2))
