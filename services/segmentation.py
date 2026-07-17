"""
Sentence segmentation — the professional-grade linguistic core of cue building.

WHY THIS EXISTS
───────────────
A broadcast caption engine (Iyuno / Pixelogic / Zoo / Deluxe grade) segments
spoken text into captions on TRUE SENTENCE boundaries first, then fits whole
sentences to the spec's line/CPS budget — splitting WITHIN a sentence only at
clause boundaries, and NEVER orphaning a trailing word onto its own flashing
cue.

The legacy cue builder used a naive `[.!?]$` test on the previous word to
decide where a sentence ended. That fires on the period after an abbreviation
("Mr.", "Dr.", "St."), so "Good afternoon. You must be Mr. Wang." was split as:
    cue 1:  Good afternoon. You must be Mr.
    cue 2:  Wang.                              ← orphan, flashes on screen
The correct, industry-standard output is:
    cue 1:  Good afternoon.
    cue 2:  You must be Mr. Wang.

This module is the single source of truth for:
  1. ABBREVIATION-AWARE sentence-end detection (never breaks "Mr. Wang").
  2. Grouping a flat word-token stream into per-speaker SENTENCE GROUPS that
     are the atomic unit cues are built from.
  3. TWO immutable composition invariants every downstream stage must honor:

     A. SPEAKER OWNERSHIP (the cross-speaker fusion fix). Every non-empty
        sentence group carries EXPLICIT speaker ownership in `speaker_runs`
        (never empty, never inherited from a neighbour). A group whose source
        tokens have no speaker is marked `speaker_known=False` +
        `review_required=True` and is NEVER allowed to inherit or merge into an
        adjacent known-speaker group. The historical defect — `_flush()` left
        `cur_speaker` un-reset so the group immediately following a same-speaker
        sentence-split emitted an EMPTY `speaker_runs`, which the packer read as
        `g_speaker=None` and silently fused a following different-speaker group
        into it — is closed at the root: `_flush()` resets ALL grouping state,
        and `speaker_runs` is materialized from the group's OWN first token, so
        the invariant is "every non-empty group has explicit speaker ownership",
        not merely "one variable was reset". FCC 47 CFR §79.1 / SOC 2 CC8.1.

     B. HARD PAUSE BOUNDARY (the conversational-pause fix). A new group ALWAYS
        opens — as an IMMUTABLE boundary — when the current token belongs to a
        DIFFERENT source utterance than the previous token AND the inter-token
        silence gap ≥ `pause_boundary_ms` (spec-driven, default 1200ms). This is
        deliberately scoped to inter-UTTERANCE gaps: a long word-level hesitation
        INSIDE one source utterance (slow/dramatic delivery) does NOT force a
        hard split unless the provider already marked a new utterance — so the
        engine never over-fragments dramatic speech. The group is tagged
        `hard_boundary_before=True`; the packer, sequence optimizer windowing,
        and every readability/merge stage treat that flag as an unbreakable wall
        (a cue can never span it). SOC 2 CC8.1 — the boundary is a versioned,
        reproducible function of source-utterance identity + measured gap.

Pure functions only — no I/O. `pause_boundary_ms` is read from the spec env knob
`CUSTOM_PAUSE_BOUNDARY_MS` (default 1200). Deterministic and auditor-defensible:
the abbreviation set is a fixed linguistic fact of English, baked in (NOT a
per-job knob), so segmentation behaviour can never vary between jobs or clients.
SOC 2 CC8.1 — identical input always yields identical sentence boundaries.
"""

import os
from typing import Any, Dict, List, Optional

# CJK awareness — the single source of truth for no-space-script handling.
try:
    from .cjk import is_cjk_text, CJK_SENTENCE_ENDERS
except Exception:  # pragma: no cover — defensive for alternate import roots
    from cjk import is_cjk_text, CJK_SENTENCE_ENDERS

# ─── Segmentation policy version ─────────────────────────────────────
# Bumped when the grouping LOGIC changes in a way that could alter group
# boundaries or speaker ownership. Independent of the sequence-optimizer policy
# version — this is the SENTENCE-GROUPING generation. SOC 2 CC8.1.
SEGMENTATION_GROUPING_VERSION = 2

# Default hard pause boundary (ms). A gap of AT LEAST this long BETWEEN two
# distinct source utterances is an immutable cue boundary. 1200ms is the
# broadcast/FAST-SDH norm (below the 1500ms conversational-turn threshold, so a
# genuine beat like the 1460ms "Cookie?" gap is honored as its own cue). Spec-
# overridable via CUSTOM_PAUSE_BOUNDARY_MS (pinned per-spec, e.g. Pluto=1200).
DEFAULT_PAUSE_BOUNDARY_MS = 1200


def pause_boundary_ms() -> int:
    """Resolve the hard inter-utterance pause boundary (ms) from the spec env
    knob CUSTOM_PAUSE_BOUNDARY_MS, defaulting to DEFAULT_PAUSE_BOUNDARY_MS. A
    non-positive / unparseable value falls back to the default so the invariant
    can never be silently disabled by a malformed spec value. SOC 2 CC8.1 — the
    boundary a run used is the pinned spec value, reproducible from the row."""
    raw = os.getenv("CUSTOM_PAUSE_BOUNDARY_MS")
    if raw is None or raw == "":
        return DEFAULT_PAUSE_BOUNDARY_MS
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_PAUSE_BOUNDARY_MS
    return v if v > 0 else DEFAULT_PAUSE_BOUNDARY_MS


# ─── Canonical broadcast abbreviation set ────────────────────────────
# Fixed, baked-in. A period immediately following any of these tokens is an
# abbreviation period, NOT a sentence terminator. This is the standard set
# every professional captioning house treats as non-breaking. Stored
# lower-cased + stripped of the trailing dot for O(1) lookup.
_ABBREVIATIONS = {
    # Titles / honorifics
    "mr", "mrs", "ms", "dr", "prof", "rev", "fr", "sr", "jr", "st",
    "messrs", "mmes", "mlle",
    # Geographic / address
    "mt", "ave", "blvd", "rd", "ln", "ste", "apt", "dept", "fl",
    # Business / org
    "inc", "ltd", "co", "corp", "llc", "plc",
    # Common Latin / misc abbreviations
    "vs", "etc", "no", "al", "approx", "est", "min", "max",
    # Month / day abbreviations
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec", "mon", "tue", "wed", "thu", "fri", "sat", "sun",
}

_SENTENCE_END_CHARS = (".", "!", "?")
# A single capital letter optionally followed by a dot — an initial like
# "J." in "J. Smith". Treated as an abbreviation (never a sentence end).
import re
_INITIAL_RE = re.compile(r"^[A-Z]$")
# A decimal number guard: "3.5" must not be treated as a sentence end. We detect
# this when the token ends in a digit-dot-digit pattern, but since tokens are
# whitespace-split words, a pure decimal like "3.5" never ends in a bare dot.
# An ellipsis ("...") is a continuation, not a hard stop.
_ELLIPSIS_RE = re.compile(r"\.\.\.$")


def _strip_trailing_sentence_punct(word: str) -> str:
    """Return the word with trailing sentence/closing punctuation removed,
    for abbreviation matching. Keeps internal dots (e.g. 'U.S.')."""
    return word.rstrip(".!?\"')]}»”’").strip()


def is_sentence_end(word: str, next_word: Optional[str] = None) -> bool:
    """
    Decide whether `word` ends a sentence, abbreviation-aware.

    Returns False (NOT a sentence end) when:
      • the word does not end with . ! or ?
      • the word is an ellipsis ("...") — a continuation
      • the bare token (sans punctuation) is a known abbreviation ("Mr", "Dr")
      • the bare token is a single-letter initial ("J", "A")
      • the word ends with a dot AND the next word begins lower-case AND the
        bare token is a known abbreviation (defensive double-check)

    Returns True only on a genuine sentence terminator.
    """
    w = (word or "").strip()
    if not w:
        return False

    # CJK sentence terminators (。！？) end a sentence with no abbreviation
    # ambiguity — there is no "Mr." problem in Japanese. Check these FIRST so a
    # Japanese token ending in 。 is always a sentence boundary even though it
    # never matches the ASCII _SENTENCE_END_CHARS test below.
    if w[-1] in ("。", "！", "？", "．"):
        return True

    # Must end with a terminator at all.
    if not w.endswith(_SENTENCE_END_CHARS):
        return False

    # Ellipsis is a continuation, never a hard stop.
    if _ELLIPSIS_RE.search(w):
        return False

    # Only a trailing '.' can be an abbreviation; '!' and '?' are always stops.
    if w.endswith("."):
        bare = _strip_trailing_sentence_punct(w)
        # Internal-dot abbreviations like "U.S." or "a.m." — if removing the
        # final dot still leaves an internal dot, it's an abbreviation.
        if "." in bare:
            return False
        token = bare.lower()
        if token in _ABBREVIATIONS:
            return False
        if _INITIAL_RE.match(bare):
            return False

    return True


def _token_utterance_id(token: Dict[str, Any]) -> Optional[Any]:
    """Return the source-utterance identity of a token, if the builder threaded
    one. `source_utterance_id` is the authoritative provider-utterance index
    (baseline utterance position / AAI-Scribe utterance id). When absent (legacy
    tokens), returns None and the caller falls back to a speaker-change + gap
    heuristic that never OVER-splits a single utterance. SOC 2 CC8.1."""
    for key in ("source_utterance_id", "utterance_id", "utterance_index"):
        v = token.get(key)
        if v is not None:
            return v
    return None


def _explode_cjk_tokens(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Explode any CJK token carrying multiple 。-delimited sentences into one
    sub-token per sentence, interpolating timings by character count over the
    token's real window. Non-CJK tokens (and CJK tokens with ≤1 sentence) pass
    through unchanged. This is what lets the existing per-token sentence-end
    grouping produce one group per Japanese sentence instead of one giant group.
    The source_utterance_id + speaker are carried onto every sub-token so the
    speaker-ownership + pause-boundary invariants hold for CJK too.
    """
    from .cjk import split_cjk_into_sentences  # local import (CI: no top-level cross-import drift)

    out: List[Dict[str, Any]] = []
    for token in tokens:
        text = (token.get("text") or "").strip()
        if not text or not is_cjk_text(text):
            out.append(token)
            continue
        sentences = split_cjk_into_sentences(text)
        if len(sentences) <= 1:
            out.append(token)
            continue
        start = int(token.get("start_ms", 0) or 0)
        end = int(token.get("end_ms", start) or start)
        span = max(1, end - start)
        total_chars = sum(len(s) for s in sentences) or 1
        cursor = 0
        for s in sentences:
            s_start = start + (span * cursor) // total_chars
            cursor += len(s)
            s_end = start + (span * cursor) // total_chars
            out.append({
                "text": s,
                "start_ms": int(s_start),
                "end_ms": int(max(s_end, s_start + 1)),
                "speaker": token.get("speaker"),
                # Preserve source-utterance identity so a multi-sentence CJK
                # utterance still counts as ONE source utterance (a 。 split is
                # a sentence boundary, NOT an utterance boundary — the pause rule
                # only fires between DISTINCT source utterances).
                "source_utterance_id": _token_utterance_id(token),
            })
    return out


def segment_into_sentence_groups(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Group a flat word-token stream into SENTENCE GROUPS — the atomic unit the
    cue builder packs into captions.

    A new group starts when ANY of:
      • the speaker changes (a new speaker is always a new caption group), OR
      • the previous word was a genuine (abbreviation-aware) sentence end, OR
      • a HARD PAUSE BOUNDARY exists: the current token belongs to a DIFFERENT
        source utterance than the previous token AND the inter-token silence gap
        ≥ pause_boundary_ms (spec default 1200ms). This boundary is IMMUTABLE —
        the group carries `hard_boundary_before=True` and no downstream stage may
        fuse across it.

    Each returned group is:
      {
        "words":               [str, ...],   # the word texts in order
        "start_ms":            int,          # first word start
        "end_ms":              int,          # last word end
        "speaker":             str | None,   # the group's OWN speaker (explicit)
        "speaker_known":       bool,         # False when the source had none
        "review_required":     bool,         # True when speaker unknown (never inherit)
        "source_utterance_id": Any | None,   # the group's opening source utterance
        "hard_boundary_before":bool,         # immutable pause boundary before this group
        "speaker_runs":        [ {speaker, word_start}, ... ],  # per-speaker offsets
      }

    SPEAKER-OWNERSHIP INVARIANT (the cross-speaker fusion fix):
    `speaker_runs` is ALWAYS materialized from the group's OWN first token — it
    is never empty on a non-empty group and never inherits from a neighbour.
    Because a speaker change always opens a new group, a group holds exactly ONE
    speaker in practice; `speaker_runs` therefore has length 1. A group whose
    source tokens carried NO speaker is emitted with speaker=None,
    speaker_known=False, review_required=True — and the packer/optimizer are
    forbidden from merging it into an adjacent known-speaker cue.
    """
    # CJK PRE-SPLIT (see _explode_cjk_tokens). Non-CJK tokens pass through
    # untouched, so the English path is byte-identical.
    tokens = _explode_cjk_tokens(tokens)

    threshold = pause_boundary_ms()

    groups: List[Dict[str, Any]] = []
    cur_words: List[str] = []
    cur_start: Optional[int] = None
    cur_end: int = 0
    cur_speaker: Optional[str] = None
    cur_speaker_known: bool = False
    cur_utterance_id: Optional[Any] = None
    cur_hard_boundary: bool = False
    # Bounded pause-provenance for a group opened at a hard pause: the measured
    # gap (ms) and the prior utterance id. None on a group that did not open at
    # a hard pause. Materialized on the first token of a hard-boundary group.
    cur_gap_before_ms: Optional[int] = None
    cur_prev_utterance_id: Optional[Any] = None
    # Timing + speaker of the LAST appended token — the reference for the next
    # token's pause-boundary + speaker-change decisions. Kept explicitly (not
    # re-derived from cur_words) so a flush never loses the reference frame.
    prev_end_ms: Optional[int] = None
    prev_utterance_id: Optional[Any] = None
    prev_speaker: Optional[str] = None
    prev_speaker_present: bool = False

    def _flush() -> None:
        # RESET ALL grouping state — this is the root of the cross-speaker fix.
        # The prior bug left cur_speaker set across a flush, so the FIRST group
        # after a same-speaker sentence-split never re-appended its speaker_run
        # (the `speaker != cur_speaker` test below was False), emitting an EMPTY
        # speaker_runs the packer read as g_speaker=None → silent fusion. Now a
        # flush zeroes cur_speaker/cur_speaker_known/cur_utterance_id/
        # cur_hard_boundary, so the next group ALWAYS re-materializes its own
        # explicit speaker_run from its first token. The prev_* reference frame
        # is intentionally NOT reset here — it tracks the last real token for the
        # NEXT group's boundary math.
        nonlocal cur_words, cur_start, cur_end, cur_speaker, cur_speaker_known
        nonlocal cur_utterance_id, cur_hard_boundary
        nonlocal cur_gap_before_ms, cur_prev_utterance_id
        if cur_words:
            speaker_known = cur_speaker_known and cur_speaker is not None
            groups.append({
                "words": cur_words,
                "start_ms": cur_start if cur_start is not None else 0,
                "end_ms": cur_end,
                "speaker": cur_speaker if speaker_known else None,
                "speaker_known": speaker_known,
                "review_required": not speaker_known,
                "source_utterance_id": cur_utterance_id,
                "hard_boundary_before": cur_hard_boundary,
                # BOUNDED PAUSE PROVENANCE (item-4 contract). Only meaningful on a
                # group that opened at a hard pause: the measured inter-utterance
                # gap (ms) and the PRIOR utterance id, so a downstream reader can
                # prove the boundary reason from bounded scalars — never raw words.
                "gap_before_ms": cur_gap_before_ms,
                "prev_utterance_id": cur_prev_utterance_id,
                # EXPLICIT ownership — always exactly one run, materialized from
                # the group's own speaker. Never empty on a non-empty group.
                "speaker_runs": [{"speaker": cur_speaker if speaker_known else None,
                                  "word_start": 0}],
            })
        cur_words = []
        cur_start = None
        cur_end = 0
        cur_speaker = None
        cur_speaker_known = False
        cur_utterance_id = None
        cur_hard_boundary = False
        cur_gap_before_ms = None
        cur_prev_utterance_id = None

    for token in tokens:
        text = (token.get("text") or "").strip()
        if not text:
            continue
        # A token's speaker is "present" only when the key exists AND is non-None.
        speaker_present = token.get("speaker") is not None
        speaker = token.get("speaker") if speaker_present else None
        start_ms = int(token.get("start_ms", 0) or 0)
        end_ms = int(token.get("end_ms", start_ms) or start_ms)
        utt_id = _token_utterance_id(token)

        # ── Boundary decisions against the LAST real token ───────────────────
        # (1) Speaker change. A KNOWN→KNOWN change, a KNOWN→unknown transition,
        #     or an unknown→KNOWN transition all open a new group so an unknown-
        #     speaker run can never silently inherit a neighbour's speaker.
        speaker_changed = bool(cur_words) and (
            speaker_present != prev_speaker_present
            or (speaker_present and prev_speaker_present and speaker != prev_speaker)
        )
        # (2) Sentence end on the previous appended word.
        prev_sentence_ended = bool(cur_words) and is_sentence_end(cur_words[-1])
        # (3) HARD PAUSE BOUNDARY — different source utterance AND gap ≥ thresh.
        #     Scoped to distinct utterances so a slow/dramatic pause WITHIN one
        #     utterance never force-splits. When the builder threaded no
        #     utterance ids (legacy tokens), we conservatively treat a gap ≥
        #     threshold as a distinct-utterance boundary ONLY when the provider
        #     signalled a new utterance via a differing id; a None==None pair is
        #     the same (unknown) utterance and does NOT hard-split — matching the
        #     "don't over-fragment a single utterance" requirement.
        different_utterance = (
            cur_words
            and utt_id is not None
            and prev_utterance_id is not None
            and utt_id != prev_utterance_id
        )
        gap_ms = (start_ms - prev_end_ms) if (cur_words and prev_end_ms is not None) else 0
        hard_pause = different_utterance and gap_ms >= threshold

        # Capture the measured gap + prior utterance id BEFORE the flush resets
        # the reference frame — so a hard-pause group can carry bounded pause
        # provenance (item-4 contract). Read only when the pause actually fires.
        pause_gap_ms = gap_ms if hard_pause else None
        pause_prev_utt = prev_utterance_id if hard_pause else None
        if (speaker_changed or prev_sentence_ended or hard_pause) and cur_words:
            _flush()
            # A group opened by a hard pause carries the immutable-boundary flag
            # plus its bounded provenance (measured gap + prior utterance id).
            if hard_pause:
                cur_hard_boundary = True
                cur_gap_before_ms = int(pause_gap_ms) if pause_gap_ms is not None else None
                cur_prev_utterance_id = pause_prev_utt

        if cur_start is None:
            cur_start = start_ms
            cur_utterance_id = utt_id

        # Materialize the group's OWN speaker on its first token. Once set for a
        # group it is fixed (a speaker change would have flushed above), so this
        # only ever fires on the group's first token — giving every non-empty
        # group explicit, self-owned speaker identity.
        if not cur_words:
            cur_speaker = speaker
            cur_speaker_known = speaker_present

        cur_words.append(text)
        cur_end = end_ms

        # Advance the reference frame to THIS token for the next iteration.
        prev_end_ms = end_ms
        prev_utterance_id = utt_id
        prev_speaker = speaker
        prev_speaker_present = speaker_present

    _flush()
    return groups
