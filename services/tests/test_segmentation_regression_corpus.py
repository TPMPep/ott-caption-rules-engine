"""
Segmentation regression corpus — the two REPORTED failure patterns.

IMPORTANT (honest labeling, per the forensic instruction): these are
REPRODUCTIONS of the reported failure PATTERNS, not exact forensic
reconstructions of the specific stored runs. The historical engine runs are not
reconstructable — the engine's job store is in-memory/transient and the
formatter persisted no per-stage transformation history — so the original source
tokens, word timings, and CONDENSATION_MODE for those exact runs are gone. These
fixtures use representative word-timed input that triggers the SAME architectural
path, and assert the invariants the fix must guarantee.

PATTERN 1 — FLASH FRAGMENT
  A long same-speaker sentence must not produce an avoidable momentary trailing
  cue when a compliant two-/three-cue redistribution gives better reading rhythm.
  (The screenshot's "to be helping us out today." flash.)

PATTERN 2 — REMOVED PHRASE
  Meaningful words must not be deleted merely because one cue exceeds CPS when an
  adjacent resegmentation preserves every word compliantly. The optimizer must
  BLOCK condensation in this case.
  (The screenshot's dropped "clearly you..." words.)

Both tests prove condensation is NOT authorized when the deterministic optimizer
finds a compliant solution.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.sequence_optimizer import (  # noqa: E402
    optimize_cue_sequence,
    condensation_is_blocked,
)


def _set_env(**kw):
    prev = {}
    for k, v in kw.items():
        prev[k] = os.environ.get(k)
        os.environ[k] = str(v)
    return prev


def _restore(prev):
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _wt(text, start_ms, end_ms):
    words = text.split()
    span = max(1, end_ms - start_ms)
    per = span // max(1, len(words))
    out, t = [], start_ms
    for i, w in enumerate(words):
        e = start_ms + per * (i + 1) if i < len(words) - 1 else end_ms
        out.append({"text": w, "start_ms": int(t), "end_ms": int(e)})
        t = e
    return out


def _cue(text, start_ms, end_ms, speaker="A"):
    return {
        "idx": 1, "start_ms": start_ms, "end_ms": end_ms,
        "lines": [text], "type": "dialogue",
        "meta": {"dialogue_text": text,
                 "runs": [{"speaker": speaker, "word_start": 0}],
                 "word_timings": _wt(text, start_ms, end_ms)},
    }


def _text(c):
    return (c.get("meta") or {}).get("dialogue_text") or " ".join(c.get("lines", []))


def _words(cues):
    out = []
    for c in cues:
        out.extend(_text(c).split())
    return out


# Broadcast-grade spec used for both reproductions (Netflix-ish adult: 42×2, 17cps).
_SPEC = dict(
    CUSTOM_MAX_CHARS=42, CUSTOM_MAX_LINES=2, CUSTOM_MAX_CPS=17,
    CUSTOM_TARGET_CPS=15, CUSTOM_MIN_DISPLAY_MS=833, CUSTOM_MAX_DISPLAY_MS=7000,
    CUSTOM_MERGE_GAP_MS=80, CUSTOM_TARGET_DURATION_MS=3000,
    SEQ_OPTIMIZER_ENABLED=1,
)


# ── PATTERN 1: flash fragment ───────────────────────────────────────
def test_flash_fragment_is_redistributed_not_left_as_flash():
    """REPRODUCTION of the "to be helping us out today." flash.

    Input: one long same-speaker utterance whose upstream split left a short
    trailing fragment. The optimizer must resegment the word-timed window so NO
    emitted cue is a momentary flash — every cue clears the editorial flash
    threshold, and the words are all preserved."""
    prev = _set_env(**_SPEC)
    try:
        cues = [
            _cue("You get on well now, so you're both going", 2200, 5240, speaker="A"),
            _cue("to be helping us out today.", 5300, 6810, speaker="A"),
        ]
        before = _words(cues)
        out = optimize_cue_sequence(cues)

        # Invariant 1: no words lost.
        assert _words(out) == before

        # Invariant 2: no emitted cue is a flash (editorial threshold, not just
        # the hard min). Flash ceiling = min(1200, 1.4*min_display).
        flash_ceiling = min(1200, int(833 * 1.4))
        for c in out:
            dur = c["end_ms"] - c["start_ms"]
            assert dur >= flash_ceiling, \
                f"flash cue survived: {dur}ms — {_text(c)!r}"

        # Invariant 3: the trailing phrase rides WITH its neighbour context — it
        # is not stranded as a lone final cue with only that fragment.
        last = _text(out[-1])
        assert last.strip() != "to be helping us out today.", \
            "trailing fragment was left as its own cue instead of redistributed"
    finally:
        _restore(prev)


# ── PATTERN 2: removed phrase (condensation blocked) ────────────────
def test_removed_phrase_pattern_blocks_condensation():
    """REPRODUCTION of the dropped "clearly you..." words.

    Input: an over-CPS opening cue whose overflow can be resegmented forward into
    a roomy same-speaker neighbour. The optimizer must find a compliant
    non-destructive arrangement AND record condensation_allowed=False, so the
    condensation stage is provably forbidden from deleting words. This is the
    executable redistribution-before-condensation guarantee."""
    prev = _set_env(**_SPEC)
    try:
        # Opening cue: ~22 chars/sec (over 17) — clearly over budget. The
        # following cue has slack. A 2-cue resegmentation of the combined window
        # brings both within 17 cps without touching a single word.
        cues = [
            _cue("But clearly you get on really well together now,", 2210, 4400, speaker="A"),
            _cue("which is why you're both helping us out.", 4500, 8000, speaker="A"),
        ]
        before = _words(cues)
        out = optimize_cue_sequence(cues)

        # Every word preserved.
        assert _words(out) == before

        # Condensation must be BLOCKED on the optimized cues — a compliant
        # non-destructive arrangement existed, so words may not be deleted.
        assert any(condensation_is_blocked(c) for c in out), \
            "condensation was not blocked despite a compliant resegmentation"

        # And every emitted cue is within CPS (the whole point — no reword needed).
        for c in out:
            body = " ".join(c.get("lines", [])).strip() or _text(c)
            dur_s = max(0.001, (c["end_ms"] - c["start_ms"]) / 1000.0)
            assert (len(body) / dur_s) <= 17.0 + 0.5, \
                f"emitted cue still over CPS: {_text(c)!r}"
    finally:
        _restore(prev)


# ── Determinism / reproducibility ───────────────────────────────────
def test_identical_input_and_version_produce_identical_output():
    prev = _set_env(**_SPEC)
    try:
        def _run():
            cues = [
                _cue("You get on well now, so you're both going", 2200, 5240, speaker="A"),
                _cue("to be helping us out today.", 5300, 6810, speaker="A"),
            ]
            out = optimize_cue_sequence(cues)
            return [(_text(c), c["start_ms"], c["end_ms"]) for c in out]
        assert _run() == _run(), "optimizer output is not reproducible"
    finally:
        _restore(prev)
