"""
Immutable Cue Boundaries — the SINGLE SOURCE OF TRUTH for "may these two cues
be merged / may a window span them?" consulted by EVERY downstream composition
stage (shaping, sequence optimizer, readability merges, CPS split/rebalance,
condensation continuation-merge, sliver/fragment absorption).

WHY THIS MODULE EXISTS
──────────────────────
Two production defects shared one root cause — many stages independently decided
whether adjacent cues could be joined, and each re-derived its own (incomplete)
notion of "same speaker / safe to merge":

  1. CROSS-SPEAKER CONTAMINATION — Speaker A's "You okay?" packed into Speaker
     G's cue, because a stage read an empty/absent speaker run as "unknown =
     safe to join".
  2. PAUSE MERGE — "Cookie?" merged into the previous utterance despite a
     measured 1,460 ms inter-utterance source gap, because no stage knew about
     a source-utterance pause boundary at all.

The professional fix is ONE primitive every stage calls. A boundary between two
cues is IMMUTABLE — no stage may merge across it, absorb across it, redistribute
words across it, or build an optimizer window spanning it — when ANY of these
hold:

  • SPEAKER CHANGE          — the two cues have different known speakers, OR one
                               is a known speaker and the other unknown.
  • UNKNOWN-SPEAKER WALL    — either cue is marked review-required (its source
                               group had no reliable speaker). An unknown-speaker
                               cue is never absorbed into a known-speaker cue.
  • PAUSE BOUNDARY          — the RIGHT cue opened at a hard inter-utterance
                               pause (meta.pause_boundary_before), i.e. a
                               source-utterance change with gap ≥ the configured
                               pause threshold.
  • OPTIMIZER-AUTHORED WALL — the RIGHT cue carries meta.hard_boundary_before
                               (an immutable boundary an earlier stage authored).

CONTRACT (cue meta the primitive reads — all bounded, no raw arrays):
  cue["meta"]["runs"]                 : [{speaker, word_start}, ...]
  cue["meta"]["review_required"]      : bool  (unknown-speaker source group)
  cue["meta"]["pause_boundary_before"]: bool  (opened at a hard source pause)
  cue["meta"]["hard_boundary_before"] : bool  (generic immutable boundary)

Pure functions only — no env writes, no I/O. Deterministic. SOC 2 CC8.1 /
FCC 47 CFR §79.1 — a boundary a stage was told about can never be silently
crossed, and the reason it exists is answerable from the cue meta alone.
"""

from typing import Any, Dict, List, Optional


def cue_speaker(cue: Dict[str, Any]) -> Optional[str]:
    """The cue's primary (first non-None) speaker id, or None when unknown."""
    for run in ((cue.get("meta") or {}).get("runs") or []):
        if run.get("speaker") is not None:
            return run.get("speaker")
    return None


def cue_speaker_set(cue: Dict[str, Any]) -> set:
    """Set of distinct known speaker ids in the cue. Empty = unknown speaker."""
    return {r.get("speaker") for r in ((cue.get("meta") or {}).get("runs") or [])
            if r.get("speaker") is not None}


def is_review_required(cue: Dict[str, Any]) -> bool:
    """True when this cue's source group had no reliable speaker (an unknown-
    speaker cue that must never inherit or be absorbed by a known speaker)."""
    return bool((cue.get("meta") or {}).get("review_required"))


def opens_hard_boundary(cue: Dict[str, Any]) -> bool:
    """True when this cue opens at an IMMUTABLE boundary — either a hard source-
    utterance pause (pause_boundary_before) or a generic authored wall
    (hard_boundary_before). No stage may fuse the previous cue into this one."""
    meta = cue.get("meta") or {}
    return bool(meta.get("pause_boundary_before") or meta.get("hard_boundary_before"))


def speakers_mergeable(prev_cue: Dict[str, Any], next_cue: Dict[str, Any]) -> bool:
    """Speaker-integrity test for a prev→next merge. Mergeable ONLY when BOTH
    cues share the exact same single known speaker. Two unknown cues are NOT
    auto-mergeable here (that is left to the legacy same-file heuristics that
    predate structured speakers) — but a KNOWN↔UNKNOWN pair is never mergeable.

    This is stricter than the historical `_speakers_compatible` (which treated
    any unknown as safe-to-join) precisely because that leniency is what let
    Speaker A's fragment fuse into Speaker G's cue when diarization dropped a
    speaker tag. A review-required cue is never mergeable in either direction."""
    if is_review_required(prev_cue) or is_review_required(next_cue):
        return False
    sa = cue_speaker_set(prev_cue)
    sb = cue_speaker_set(next_cue)
    if not sa or not sb:
        # At least one side has no known speaker → cannot PROVE same-speaker →
        # do not merge (the enterprise-safe default that closes the A-into-G bug).
        return False
    return sa == sb and len(sa) == 1


def is_immutable_boundary(prev_cue: Dict[str, Any], next_cue: Dict[str, Any]) -> bool:
    """THE primitive. True when the boundary BETWEEN prev_cue and next_cue is
    immutable and no stage may merge/absorb/redistribute/window across it.

    Immutable when ANY of:
      • next_cue opens a hard pause / authored wall (opens_hard_boundary), OR
      • either cue is review-required (unknown-speaker wall), OR
      • the two cues are not provably the same single speaker (speaker change).
    """
    if opens_hard_boundary(next_cue):
        return True
    if is_review_required(prev_cue) or is_review_required(next_cue):
        return True
    return not speakers_mergeable(prev_cue, next_cue)


def boundary_reason(prev_cue: Dict[str, Any], next_cue: Dict[str, Any]) -> Optional[str]:
    """Bounded machine-readable reason a boundary is immutable, or None when the
    pair is freely mergeable. For provenance / audit surfacing (never free text)."""
    meta = next_cue.get("meta") or {}
    if meta.get("pause_boundary_before"):
        return "pause_boundary"
    if meta.get("hard_boundary_before"):
        return "authored_hard_boundary"
    if is_review_required(prev_cue) or is_review_required(next_cue):
        return "unknown_speaker"
    if not speakers_mergeable(prev_cue, next_cue):
        return "speaker_change"
    return None


def propagate_boundary_to_children(parent: Dict[str, Any],
                                   children: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """When a stage SPLITS one cue into ordered children, the parent's opening
    boundary belongs ONLY to the FIRST child (the split introduced NO new source-
    utterance pause between the children). Every child inherits the parent's
    speaker-integrity review flag. This is the "boundary survives split, on the
    correct child only" contract used by shaping / CPS / duration splitters.

    Idempotent + defensive: no-op on an empty child list."""
    if not children:
        return children
    pmeta = parent.get("meta") or {}
    review = bool(pmeta.get("review_required"))
    pause_first = bool(pmeta.get("pause_boundary_before"))
    hard_first = bool(pmeta.get("hard_boundary_before"))
    # Bounded pause provenance rides with the opening pause wall — first child only.
    pause_prov = pmeta.get("pause_provenance")
    for i, child in enumerate(children):
        cmeta = child.setdefault("meta", {})
        # Speaker-integrity review flag rides with every child (all children are
        # the same source group as the parent).
        if review:
            cmeta["review_required"] = True
        # The opening pause / hard wall belongs to the FIRST child only.
        if i == 0:
            if pause_first:
                cmeta["pause_boundary_before"] = True
            if hard_first:
                cmeta["hard_boundary_before"] = True
            if pause_prov is not None:
                cmeta["pause_provenance"] = pause_prov
        else:
            # A mid-sentence split child must NOT claim the parent's opening wall.
            cmeta.pop("pause_boundary_before", None)
            cmeta.pop("hard_boundary_before", None)
            cmeta.pop("pause_provenance", None)
    return children
