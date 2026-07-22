"""
services.timing_repair — Class A timing-data integrity (Phase 1).
=================================================================
Deterministic, provider/model-specific repair of MALFORMED provider word
timings, run at NORMALIZATION time so NO downstream stage (segmentation,
shaping, sequence optimizer, readability, condensation, QC, export) ever sees a
corrupt frame. This is the canonical, production-authority implementation; the
JS mirror (src/lib/cc-timing-repair.js) is a byte-for-byte behavioural parity
copy for the Base44 side and the parity test.

WHY THIS EXISTS — the evidence (read-only prevalence scan, 2026-07-19/20):
  ElevenLabs Scribe v2 (English corpus, 10 projects / 52,855 words) emits, on
  rare utterance-final words, an implausible `end` that swallows the SILENCE
  GAP before the NEXT utterance's first word. The anchor defect:

    project 6a1d3079…: utt 870 = ["There" 2,589,518 → 2,603,858]  (14,340ms, spk B)
                       utt 871 = ["are" 2,603,898 → …]            (spk B, +40ms)

  Downstream (formatter._word_timings_in_window midpoint assignment →
  shaping.shape_caption_rhythm) this becomes a one-word cue with a 14s window:
  fan-out + a blank on-screen interval + CPS/min-duration distortion. That is
  the failure this module eliminates.

SCOPE — PROVIDER-SPECIFIC + VERSIONED (do NOT universalize):
  Only ElevenLabs Scribe v2 has a genuine per-word capture defect in the
  corpus. imported::vtt_import "long words" are import interpolation artifacts
  (monotonic, internally consistent) and are OUT OF SCOPE. AssemblyAI is not a
  target. This policy is "ELS-v2 timing-repair policy v1"; broadening to another
  provider/language REQUIRES its own corpus validation and a policy-version bump.

DETECTION ≠ REPAIRED DURATION (the two are separate concepts):
  Detection fires ONLY when ALL hold:
    (1) duration > IMPLAUSIBLE_DURATION_MS (corpus-calibrated ceiling, well above
        the p999 real word ≈ 1,740ms — set to 2,600ms so the non-empty
        2,000–2,500ms band of REAL long words is never touched), AND
    (2) the word's end is NOT corroborated by a trustworthy neighbour (see
        _find_trustworthy_neighbour), AND
    (3) the word would produce a DEGENERATE cue downstream — i.e. the excess
        beyond the neighbour is a large blank interval (fan-out). A merely
        slightly-long word before a real micro-gap (G3) does NOT qualify.
  A flat duration ceiling alone is NOT the detector (the 2,000–2,500ms band is
  non-empty with real words). Corroboration + degeneracy are required.

THREE SEPARATE CONCEPTS — DO NOT CONFLATE (the core Phase 1 discipline):
  This module deliberately keeps three distinct things apart, each with its own
  evidence and its own constant. Collapsing any two of them is the mistake this
  policy exists to avoid.

    (1) DETECTION evidence — "is this timing demonstrably wrong?"
        Governed by IMPLAUSIBLE_DURATION_MS (+ corroboration + degeneracy). This
        constant is a DETECTION THRESHOLD only. It never becomes a repaired value.

    (2) REPAIR-BOUND evidence — "how late is the word PROVABLY not still running?"
        Governed by the trustworthy neighbour's start (this or the next
        utterance's first valid same-speaker word). The neighbour is a hard UPPER
        BOUND: the corrupt word must end before it (minus MIN_GAP_MS). The
        neighbour PROVES the word cannot extend past that point — it does NOT
        prove the word's actual spoken length, and it does NOT license collapsing
        the genuine silence between the word and the neighbour.

    (3) REPAIRED-DURATION ESTIMATE — "what is the word's likely spoken length?"
        Governed by ESTIMATED_REPAIR_DURATION_MS (a versioned, corpus-derived
        CONSERVATIVE HEURISTIC — see its docstring) OR by statistically
        trustworthy interpolation. This is an ESTIMATE bounded by (2), never a
        value the neighbour "established". A corrupt word is repaired to
        start + estimate, then clamped down so it never crosses the (2) bound.
        Because the estimate is short (≤ a plausible word) and the bound is only
        an upper limit, legitimate multi-second silence BETWEEN the repaired word
        and a far-away neighbour is preserved, not eliminated.

DECISION MODEL — STRICTLY EVIDENCE-DRIVEN (Phase 1 philosophy):
  The objective is to maximize TRUSTWORTHY repairs, NOT the repair count. The
  engine follows the evidence, in two questions:

    Step 1 — Can we DEMONSTRATE the provider timing is actually incorrect?
             (duration > IMPLAUSIBLE_DURATION_MS, end uncorroborated, and the
             cue would be degenerate downstream). If no → PRESERVE unchanged.
    Step 2 — Can we DERIVE a demonstrably better timing from TRUSTWORTHY
             evidence (a bounded estimate)? If yes → REPAIR. If no → QUARANTINE.

  There is intentionally NO generic capped fallback in Phase 1: the engine never
  invents a duration merely because its evidence is exhausted. A conservative
  fallback may be re-evaluated in a later policy version ONLY if corpus analysis
  proves it introduces no false positives.

REPAIR EVIDENCE HIERARCHY (the repaired `end` is DERIVED from evidence, or none):
  1. neighbour — the first valid same-speaker word of THIS utterance,
     chronologically adjacent, crossing no hard speaker boundary.
  2. cross_utterance_neighbour — same, but the first valid same-speaker word of
     the immediately-following utterance. (1) and (2) supply an UPPER BOUND: the
     real word must end before that trustworthy anchor (minus MIN_GAP_MS). The
     assigned length is the SHORTER of the versioned heuristic estimate (concept 3)
     and that bound — so a FAR-AWAY neighbour leaves genuine silence intact and a
     NEAR neighbour clamps the word down. The neighbour never fixes the length.
  3. interpolation — deterministic median-word-duration extrapolation from the
     surrounding valid words of the SAME utterance, admitted ONLY when the
     evidence is STATISTICALLY TRUSTWORTHY: at least MIN_INTERPOLATION_SAMPLES
     supporting words AND a median that is itself a plausible word length
     (≥ MIN_PLAUSIBLE_WORD_MS). A 3-word / 60ms median is NOT trustworthy → skip.
  4. quarantine — mark timing_anomaly.quarantined=True when NO tier produced a
     defensible duration. Quarantined words are flagged, NEVER silently shipped
     and NEVER assigned a fabricated duration.

GUARANTEES (auditor invariants):
  • Valid timing preserved byte-for-byte (no false repair).
  • Original provider values preserved on `timing_anomaly` (immutable evidence).
  • A repaired duration is ALWAYS shorter than the original (repair only shortens).
  • Never negative/reversed/overlapping/non-monotonic.
  • Never moves a valid neighbour.
  • A repaired end is ONLY ever a BOUNDED ESTIMATE (a conservative heuristic or a
    trustworthy interpolation), never a floor constant, never an invented
    duration, and never the neighbour's start presented as the word's length.
    Low-evidence → quarantine.
  • QUARANTINE SEPARATES AN AUDIT FACT FROM AN EXECUTION FACT (Option A). The
    AUDIT fact — disposition="quarantined" — is owned by THIS repair layer and
    means "no trustworthy duration could be derived", with the full evidence trail
    + immutable ORIGINAL values on timing_anomaly. The EXECUTION fact — the token
    has NO effective timing — is represented by REMOVING the timing (start/end :=
    None), NEVER by fabricating a synthetic zero-width point. A zero-width span is
    itself an invented duration; removing timing keeps the "never fabricate"
    principle internally consistent end-to-end. The word TEXT, ORDER, and SPEAKER
    survive untouched — it stays a fully valid linguistic token. Every timing-aware
    downstream stage excludes it via the GENERAL has_timing() predicate (null
    timing → excluded from group window, pause gap, shaper midpoint) WITHOUT ever
    knowing it was quarantined or why. The provider BASELINE is untouched. The job
    is NEVER stopped for a quarantined word (halting a whole broadcast caption job
    over one uncertain word would convert a bounded, containable data defect into
    an outage — the wrong enterprise call under 100+ concurrent users). Instead the
    quarantine is surfaced as a run-level review signal.
  • Legitimate silence is preserved: the neighbour is an upper bound, so a repaired
    word ending at start + estimate leaves any genuine multi-second gap before a
    far-away neighbour intact — the repair never "closes the gap" to the neighbour.
  • A quarantined frame cannot influence segmentation/shaping as though it were
    valid timing: its end is neutralized to its start (zero-width), so it neither
    stretches a cue window, manufactures a false pause gap, nor poisons a
    word-timing midpoint. It is excluded from timing-based segmentation while its
    text stays in the reading order.
  • Deterministic: identical input → identical output. Idempotent: repairing an
    already-repaired stream is a no-op (a word carrying timing_anomaly is skipped).

COMPLIANCE: FCC 47 CFR §79.1, TPN MS-4.x, SOC 2 CC7.4 (reversible/attributable
transformation) / CC8.1 (change management — every repair is auditable).
"""

from __future__ import annotations

from statistics import median
from typing import Any, Dict, List, Optional

# ── Policy constants (ELS-v2 timing-repair policy v1) ────────────────────────
# Numeric values are corpus-calibrated with margin; see the module docstring and
# docs/phase0-root-cause-report.md §5. RULE_VERSION is stamped on every repair.
RULE_VERSION = 1
POLICY_NAME = "elevenlabs_scribe_v2_timing_repair_v1"

# ── CONCEPT (1): DETECTION THRESHOLD ────────────────────────────────────────
# A word longer than this is a CANDIDATE for repair (not an automatic repair).
# p999 real word ≈ 1,740ms; the 2,000–2,500ms band holds real long words, so the
# threshold sits above it at 2,600ms. This is ONLY a detector — it is never used
# as, or clamped to, a repaired duration.
IMPLAUSIBLE_DURATION_MS = 2600

# ── CONCEPT (3): REPAIRED-DURATION ESTIMATE (versioned conservative heuristic) ─
# The ESTIMATED spoken length assigned to a corrupt word when no trustworthy
# interpolation is available. This is a HEURISTIC, not a proven value:
#   • It is NOT established by the neighbour. The neighbour only proves the word
#     must END BEFORE it (an upper bound); it says nothing about the word's true
#     length. The estimate answers the separate question "how long was this word
#     LIKELY spoken?".
#   • 1,200ms is the CONSERVATIVE corpus-derived choice for ELS-v2 policy v1:
#     it sits above the p999 real Scribe-v2 word (≈1,740ms is the p999; 1,200ms is
#     ~p998 and comfortably longer than the p99 of 920ms), so a repaired word is a
#     believable long word — long enough not to under-clip, short enough not to
#     re-introduce a fan-out or eat legitimate silence. It is bounded DOWN by the
#     neighbour when the neighbour is close, and left intact (silence preserved)
#     when the neighbour is far away.
#   • It is pinned to RULE_VERSION. A future policy version may recalibrate it from
#     a broader corpus; changing it is a versioned decision, not a silent tweak.
# RENAMED from the earlier MAX_PLAUSIBLE_WORD_MS specifically so neither the code
# nor the audit trail can imply the neighbour "established" a 1,200ms word length.
ESTIMATED_REPAIR_DURATION_MS = 1200

# A plausible word floor. Used TWO ways: (1) the minimum a repaired word may be,
# and (2) the trustworthiness gate on interpolation — a median BELOW this is not
# credible evidence, so interpolation is refused and the word quarantines.
MIN_PLAUSIBLE_WORD_MS = 200

# Interpolation is only statistically trustworthy with at least this many
# supporting same-utterance words. Below this the median is noise (the "together."
# case had 3 samples / a 60ms median) → interpolation is refused → quarantine.
MIN_INTERPOLATION_SAMPLES = 5

# Minimum gap left before a trustworthy neighbour when clamping.
MIN_GAP_MS = 20

# Degeneracy threshold — the excess beyond the corroborating neighbour that makes
# the resulting cue degenerate (blank interval / fan-out). Below this, a long
# word before a real micro-gap (G3) is NOT repaired.
DEGENERATE_EXCESS_MS = 2000

# Providers/models this policy applies to. Anything else passes through untouched.
_IN_SCOPE = {("elevenlabs", "scribe_v2")}


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _in_scope(provider: Optional[str], model: Optional[str]) -> bool:
    return (_norm(provider), _norm(model)) in _IN_SCOPE


def _wstart(w: Dict[str, Any]) -> Optional[int]:
    v = w.get("start", w.get("start_ms"))
    return None if v is None else int(v)


def _wend(w: Dict[str, Any]) -> Optional[int]:
    v = w.get("end", w.get("end_ms"))
    return None if v is None else int(v)


def _valid(w: Dict[str, Any]) -> bool:
    s, e = _wstart(w), _wend(w)
    return s is not None and e is not None and e >= s


def has_timing(token: Dict[str, Any]) -> bool:
    """GENERAL TIMING-DOMAIN PREDICATE — the single source of truth for
    "does this token have usable timing?" Returns True iff BOTH start and end are
    present (non-None) AND end >= start. Returns False for an UNTIMED token —
    whatever the reason (a quarantined repair disposition today; a future
    provider gap, a future normalization policy, a manual insertion tomorrow).

    This is deliberately NOT quarantine-specific. Timing-aware stages
    (segmentation, formatter, shaping) consume ONLY has_timing() tokens for every
    timestamp-based calculation, and never ask WHY a token is untimed — the reason
    is an AUDIT fact that lives on timing_anomaly, owned by the repair policy.
    Untimed tokens remain fully valid linguistic tokens (text, order, speaker,
    provenance preserved); they are simply excluded from timing math.

    Accepts both key shapes (start/end and start_ms/end_ms). Never coerces a
    missing value to 0 — a missing timestamp is UNTIMED, never a real 0ms event.
    """
    s, e = _wstart(token), _wend(token)
    return s is not None and e is not None and e >= s


def _dur(w: Dict[str, Any]) -> Optional[int]:
    s, e = _wstart(w), _wend(w)
    return None if s is None or e is None else e - s


def _flatten(utterances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten utterance words into an index the repair can walk, carrying the
    owning utterance index + speaker so cross-utterance neighbour lookup and the
    hard speaker-boundary guard are answerable per word."""
    flat: List[Dict[str, Any]] = []
    for ui, u in enumerate(utterances):
        u_spk = u.get("speaker")
        for wi, w in enumerate(u.get("words") or []):
            flat.append({
                "w": w, "ui": ui, "wi": wi,
                "speaker": w.get("speaker", u_spk),
                "u_end": _wend(u) if _wend(u) is not None else None,
            })
    return flat


def _find_trustworthy_neighbour(flat: List[Dict[str, Any]], fi: int) -> Optional[Dict[str, Any]]:
    """The first valid, chronologically-later, SAME-speaker word after fi —
    including the next utterance's first word — that crosses no hard speaker
    boundary. Returns {start_ms, cross_utterance} or None."""
    cur = flat[fi]
    cur_spk = cur["speaker"]
    cur_start = _wstart(cur["w"])
    for j in range(fi + 1, len(flat)):
        nxt = flat[j]
        if not _valid(nxt["w"]):
            continue
        ns = _wstart(nxt["w"])
        if cur_start is not None and ns is not None and ns < cur_start:
            continue  # not chronologically later — never clamp backwards
        # Hard speaker boundary — a different speaker is not a trustworthy anchor.
        if nxt["speaker"] != cur_spk:
            return None
        return {"start_ms": ns, "cross_utterance": nxt["ui"] != cur["ui"]}
    return None


def _interpolation_evidence(flat: List[Dict[str, Any]], fi: int) -> Optional[Dict[str, Any]]:
    """Gather TRUSTWORTHY interpolation evidence from the SAME utterance's other
    valid words. Returns {end_ms, median_ms, sample_count} ONLY when the evidence
    is statistically credible — at least MIN_INTERPOLATION_SAMPLES supporting
    words AND a median that is itself a plausible word length. Returns None
    otherwise (→ the caller quarantines rather than fabricating a duration).

    This is the gate that refuses the "together." case: 3 samples / 60ms median
    is noise, not evidence."""
    cur = flat[fi]
    start = _wstart(cur["w"])
    if start is None:
        return None
    durs = [
        _dur(o["w"]) for o in flat
        if o["ui"] == cur["ui"] and o is not cur and _valid(o["w"])
        and _dur(o["w"]) is not None and 0 < _dur(o["w"]) <= IMPLAUSIBLE_DURATION_MS
    ]
    if len(durs) < MIN_INTERPOLATION_SAMPLES:
        return None  # insufficient sample size — not statistically trustworthy
    med = int(median(durs))
    if med < MIN_PLAUSIBLE_WORD_MS:
        return None  # median itself implausible — the evidence is not credible
    return {"end_ms": start + med, "median_ms": med, "sample_count": len(durs)}


def _repair_one(
    flat: List[Dict[str, Any]], fi: int,
) -> Optional[Dict[str, Any]]:
    """Decide + apply the repair for one flat word. Returns the timing_anomaly
    audit block (with the new end already written onto the word) or None when the
    word is preserved untouched. Never mutates any OTHER word."""
    cur = flat[fi]
    w = cur["w"]
    if not _valid(w):
        return None
    # Idempotence: a word already carrying a timing_anomaly was processed by a
    # prior pass — never re-process (no duplicated audit metadata, no re-repair).
    if "timing_anomaly" in w:
        return None
    start = _wstart(w)
    end = _wend(w)
    duration = end - start

    # STEP 1 — plausible duration → preserve byte-for-byte (no evidence of a fault).
    if duration <= IMPLAUSIBLE_DURATION_MS:
        return None

    neighbour = _find_trustworthy_neighbour(flat, fi)

    # (2)+(3) detection. The end is CORROBORATED (→ preserve, the G3 case) only
    # when it lands close to a trustworthy neighbour's start from EITHER side:
    #   |end - neighbour_start| < DEGENERATE_EXCESS_MS
    # The "There" shape has end (2,603,858) just BEFORE the neighbour start
    # (2,603,898) → excess = -40 → |−40| < 2000, which would look corroborated —
    # BUT the word's DURATION is 14,340ms of mostly-silence, so it is a fan-out
    # regardless of where its end lands relative to the neighbour. Corroboration
    # therefore requires BOTH a near-neighbour end AND a plausible duration; a
    # word whose duration alone is a fan-out is never "rescued" by a coincidental
    # near-neighbour end. G3 (a genuinely ~3-3.4s word ending ~20-50ms before its
    # neighbour) stays preserved because its duration is not a fan-out.
    if neighbour is not None and neighbour["start_ms"] is not None:
        near_neighbour = abs(end - neighbour["start_ms"]) < DEGENERATE_EXCESS_MS
        duration_is_fanout = duration >= (IMPLAUSIBLE_DURATION_MS + DEGENERATE_EXCESS_MS)
        if near_neighbour and not duration_is_fanout:
            # End corroborated AND duration plausible-ish → G3, preserve.
            return None

    # ── STEP 2 — derive a demonstrably better timing from TRUSTWORTHY evidence ──
    # CORE INVARIANT: a repair ALWAYS SHORTENS to a PLAUSIBLE WORD LENGTH. The
    # defect is an implausibly LONG word that swallowed silence; the repaired end
    # is therefore strictly < the original end. Each tier is EVIDENCE — a
    # trustworthy neighbour, or a statistically credible interpolation. There is
    # NO capped fallback: if no tier yields defensible evidence, the word
    # QUARANTINES. The engine never invents a duration.
    repaired_end: Optional[int] = None
    evidence: Optional[str] = None
    neighbour_evidence: Optional[Dict[str, Any]] = None

    # CONCEPT (3) — the ESTIMATED spoken length (a heuristic, not proven by any
    # neighbour). The word is estimated to run start + ESTIMATED_REPAIR_DURATION_MS.
    estimated_end = start + ESTIMATED_REPAIR_DURATION_MS

    # Tiers 1+2 — trustworthy (cross-)utterance neighbour as an UPPER BOUND only
    # (concept 2). The neighbour proves the word must end before it; it does NOT
    # dictate the length. We therefore take the SHORTER of the estimate and the
    # bound:
    #   • Neighbour CLOSE (e.g. the "There" shape: next-utt "are" @2,603,898, far
    #     past the estimate) → the estimate wins → end = start + 1,200 = 2,590,718,
    #     leaving the ~13s that follows as preserved silence before "are".
    #   • Neighbour NEARER than the estimate (a tight adjacent word) → the bound
    #     wins → the word is clamped just before the neighbour.
    # Because we take the MIN, a far-away neighbour never inflates the repair to
    # eat the silence, and a near neighbour is never overrun.
    if neighbour is not None and neighbour["start_ms"] is not None:
        cand = min(estimated_end, neighbour["start_ms"] - MIN_GAP_MS)
        if cand > start:
            repaired_end = cand
            evidence = "cross_utterance_neighbour" if neighbour["cross_utterance"] else "neighbour"
            neighbour_evidence = {
                "neighbour_start_ms": neighbour["start_ms"],
                "cross_utterance": neighbour["cross_utterance"],
                # Audit: the neighbour is the BOUND; record whether it or the
                # estimate governed, so a reviewer sees silence was preserved.
                "bound_governed": cand < estimated_end,
            }

    # Tier 3 — statistically trustworthy interpolation ONLY. _interpolation_evidence
    # returns None unless there are ≥ MIN_INTERPOLATION_SAMPLES supporting words AND
    # a plausible median. This is the gate that refuses the "together." case
    # (3 samples / 60ms median → None → quarantine), instead of flooring a noise
    # median up to a constant and calling it a repair.
    interp_evidence: Optional[Dict[str, Any]] = None
    if repaired_end is None:
        interp_evidence = _interpolation_evidence(flat, fi)
        if interp_evidence is not None:
            cand = min(interp_evidence["end_ms"], estimated_end)
            if cand > start:
                repaired_end = cand
                evidence = "interpolation"

    anomaly: Dict[str, Any] = {
        "provider": POLICY_NAME.split("_")[0],
        "model": "scribe_v2",
        "policy": POLICY_NAME,
        "rule_version": RULE_VERSION,
        "detection_reason": "implausible_word_duration",
        "original_start_ms": start,
        "original_end_ms": end,
        "original_duration_ms": duration,
    }

    # Tier 4 — QUARANTINE. No trustworthy tier produced a duration (no neighbour,
    # no credible interpolation), OR the only candidate would not actually shorten
    # the word. We NEVER fabricate a duration — AND (Option A) we never fabricate a
    # synthetic zero-width point either. Instead the word's EFFECTIVE TIMING IS
    # REMOVED: start/end set to None (the canonical "no trustworthy timing" state),
    # not to any invented value. This DECOUPLES the two facts the pipeline had
    # wrongly coupled:
    #   • AUDIT FACT (owned HERE, the repair layer): disposition="quarantined" —
    #     the policy could not derive a trustworthy duration, with the full
    #     evidence-rejection trail + immutable ORIGINAL values on timing_anomaly.
    #   • EXECUTION FACT (owned by the TIMING LAYER, has_timing()): the token has
    #     no effective timing, so every timing-aware stage excludes it via the
    #     GENERAL predicate — WITHOUT knowing it was quarantined, or why.
    # Text + reading order + speaker attribution are preserved: the token remains a
    # fully valid linguistic token; only its untrustworthy timing is withheld.
    if repaired_end is None or repaired_end <= start or repaired_end >= end:
        anomaly["disposition"] = "quarantined"
        anomaly["quarantined"] = True
        anomaly["repair_branch"] = "quarantine"
        anomaly["repair_evidence"] = "quarantine"
        # WHY each evidence tier was rejected (auditor-answerable from the row).
        anomaly["neighbour_evidence"] = neighbour_evidence  # what was (or wasn't) available
        if neighbour is None or neighbour.get("start_ms") is None:
            anomaly["neighbour_rejected"] = "no_trustworthy_same_speaker_neighbour"
        else:
            anomaly["neighbour_rejected"] = "bound_would_not_shorten"
        if interp_evidence is None:
            anomaly["interpolation_rejected"] = "insufficient_or_implausible_evidence"
        # EXECUTION REPRESENTATION — timing REMOVED, never fabricated. start/end
        # become None (the "untimed token" state). NO zero-width point, NO invented
        # timestamp. The ORIGINAL corrupt values live on the audit block as
        # immutable evidence, so nothing is lost.
        anomaly["downstream_timing"] = "timing_removed"
        anomaly["effective_timing"] = None
        anomaly["excluded_from_timing_segmentation"] = True
        anomaly["operator_review_state"] = "review_required"
        # Null BOTH shapes so no reader can recover a synthetic value. A timing
        # consumer MUST go through has_timing() (which treats None as untimed).
        if "start" in w:
            w["start"] = None
        if "end" in w:
            w["end"] = None
        if "start_ms" in w:
            w["start_ms"] = None
        if "end_ms" in w:
            w["end_ms"] = None
        # timing_quarantined stays TRUE — it is the AUDIT marker (disposition), NOT
        # the execution gate. Execution is gated purely by null timing via
        # has_timing(). Kept so an auditor can locate quarantined tokens directly.
        w["timing_quarantined"] = True
        w["timing_anomaly"] = anomaly
        return anomaly

    # Final safety clamps — never overlap a neighbour, never reverse, never grow.
    if neighbour is not None and neighbour["start_ms"] is not None:
        repaired_end = min(repaired_end, neighbour["start_ms"] - MIN_GAP_MS)
    repaired_end = max(repaired_end, start + MIN_PLAUSIBLE_WORD_MS)  # plausible floor
    repaired_end = min(repaired_end, end - 1)  # SHORTEN invariant — strictly less

    anomaly["disposition"] = "repaired"
    anomaly["quarantined"] = False
    anomaly["repair_branch"] = evidence
    anomaly["repair_evidence"] = evidence
    anomaly["repaired_start_ms"] = start
    anomaly["repaired_end_ms"] = repaired_end
    anomaly["repaired_duration_ms"] = repaired_end - start
    # CONCEPT (3) provenance — the repaired duration is an ESTIMATE, and this
    # records HOW it was estimated + WHICH constant/heuristic governed, so the
    # audit never implies the neighbour proved the length.
    if evidence == "interpolation":
        anomaly["repaired_duration_estimation"] = "trustworthy_interpolation"
    else:
        # neighbour / cross_utterance_neighbour branches: the length is the
        # versioned heuristic estimate, upper-bounded by the neighbour.
        anomaly["repaired_duration_estimation"] = "conservative_heuristic"
        anomaly["estimated_repair_duration_ms"] = ESTIMATED_REPAIR_DURATION_MS
    if neighbour_evidence is not None:
        anomaly["neighbour_evidence"] = neighbour_evidence
    if evidence == "interpolation" and interp_evidence is not None:
        anomaly["interpolation_evidence"] = {
            "median_ms": interp_evidence["median_ms"],
            "sample_count": interp_evidence["sample_count"],
        }

    # Write the repaired end onto BOTH shapes so any downstream reader (start/end
    # or start_ms/end_ms) sees the corrected frame. The original is on `timing_anomaly`.
    if "end" in w:
        w["end"] = repaired_end
    if "end_ms" in w:
        w["end_ms"] = repaired_end
    w["timing_anomaly"] = anomaly
    return anomaly


def repair_word_timings(
    utterances: List[Dict[str, Any]],
    provider: Optional[str] = "elevenlabs",
    model: Optional[str] = "scribe_v2",
) -> List[Dict[str, Any]]:
    """Repair implausible word timings in a list of provider UTTERANCES.

    Mutates + returns the same list (words repaired/neutralized in place;
    originals recorded on word.timing_anomaly). No-op (byte-identical) for
    out-of-scope provider/model, and for every already-plausible word. Never
    moves a valid neighbour. Deterministic.

    Contract note: the caller (assembly.build_word_timestamps_from_result) runs
    this ONCE on the NORMALIZED provider result BEFORE tokenization, so the
    immutable baseline JSON on S3 is untouched — only the in-memory normalized
    stream is repaired; a quarantined word additionally has its effective timing
    REMOVED (start/end := None, the untimed-token state) + is stamped
    timing_quarantined=True as an AUDIT marker. No synthetic timestamp is ever
    written; downstream stages exclude it purely via has_timing().

    Back-compat: returns the same utterances list. For the run-level operator
    review summary use repair_word_timings_with_summary (this delegates to it).
    """
    repair_word_timings_with_summary(utterances, provider=provider, model=model)
    return utterances


def repair_word_timings_with_summary(
    utterances: List[Dict[str, Any]],
    provider: Optional[str] = "elevenlabs",
    model: Optional[str] = "scribe_v2",
) -> Dict[str, Any]:
    """Same repair pass as repair_word_timings, but returns a BOUNDED run-level
    summary the caller can surface as an operator-visible review signal:

        {
          "in_scope": bool,          # was the provider/model the Scribe v2 policy?
          "detected": int,           # implausible words detected
          "repaired": int,           # words given a bounded repaired duration
          "quarantined": int,        # words neutralized (review_required)
          "review_required": bool,   # quarantined > 0
          "rule_version": int,
          "policy": str,
        }

    Idempotent: a second pass over an already-processed stream skips words
    carrying timing_anomaly, so the summary of a re-run counts zero new events.
    Deterministic. The utterances are mutated in place exactly as in
    repair_word_timings (this is the primitive; that function delegates here)."""
    summary = {
        "in_scope": _in_scope(provider, model),
        "detected": 0, "repaired": 0, "quarantined": 0,
        "review_required": False,
        "rule_version": RULE_VERSION, "policy": POLICY_NAME,
    }
    if not utterances or not summary["in_scope"]:
        return summary
    flat = _flatten(utterances)
    for fi in range(len(flat)):
        anomaly = _repair_one(flat, fi)
        if anomaly is None:
            continue
        summary["detected"] += 1
        if anomaly.get("disposition") == "quarantined":
            summary["quarantined"] += 1
        else:
            summary["repaired"] += 1
    summary["review_required"] = summary["quarantined"] > 0
    return summary
