"""
CPS enforcement — reading-speed rules as ACTIVE DRIVERS, not just QC metrics.

WHY THIS MODULE EXISTS
──────────────────────
Characters-per-second (CPS = visible chars / on-screen seconds) is the single
most-violated caption rule and the #1 reason a deliverable fails a broadcast /
streaming QC pass. Before this module, the engine only *measured* CPS in the QC
report. A cue reading at 28 CPS against a 17-CPS spec was reported and shipped.

Phase 2 makes CPS *act*, deterministically, in strict priority order:

  TOO FAST  (cps > max_cps):
    1. EXTEND  — push the cue's end_ms later into the idle gap before the next
                 cue (never into it). Free: no text change, no overlap, honors
                 min_gap. Brings CPS down at zero editorial cost.
    2. SPLIT   — if still over budget after extending, split the cue at the best
                 clause boundary into two cues. Each half then carries fewer
                 chars over its own time window, so both read within budget.
                 Timings are interpolated proportionally over the original
                 window; the split honors min_gap and min_display.

  TOO SLOW  (cps < min_cps):
    3. TRIM    — a caption lingering far longer than its content needs reads as
                 "sleepy". Trim trailing dead air down toward target_cps, never
                 below min_display_ms and never stealing from the next cue.

NO AI. NO I/O. Pure functions over the cue list + spec env knobs. Identical
(cues, env) always produces the identical result. Every action is reproducible
and auditor-defensible. SOC 2 CC8.1 / FCC 47 CFR §79.1 (readability).

Spec knobs consumed (all already declared on ClosedCaptionSpec.reading_speed_rules
and surfaced via the producer's caption-options mapping):
    CUSTOM_MAX_CPS         max chars/sec (hard ceiling)
    CUSTOM_TARGET_CPS      the CPS we aim for when extending/trimming
    CUSTOM_MIN_CPS         below this = too slow (lingering)
    CUSTOM_MIN_DISPLAY_MS  floor on dialogue cue duration
    CUSTOM_MAX_DISPLAY_MS  ceiling on a single cue's duration
    CUSTOM_MERGE_GAP_MS    min gap to preserve between consecutive cues
    CUSTOM_MAX_CHARS / CUSTOM_MAX_LINES  geometry for the split re-render
"""

import os
from typing import Any, Dict, List, Optional

from .rules import get_rule as _rule_get

try:
    from .rendering import render_lines
except Exception:  # pragma: no cover — defensive for alternate import roots
    def render_lines(words, runs, max_lines=None, max_chars=None, dialogue_text=None):
        return [dialogue_text if dialogue_text is not None else " ".join(words)]

# CJK awareness — character-based measurement + splitting for no-space scripts.
try:
    from .cjk import is_cjk_text as _is_cjk, cjk_char_count as _cjk_count, CJK_CLAUSE_ENDERS, CJK_SENTENCE_ENDERS
except Exception:  # pragma: no cover
    def _is_cjk(text):
        return False

    def _cjk_count(text):
        return len((text or "").replace(" ", ""))

    CJK_CLAUSE_ENDERS = ("、", "，", "；", "：")
    CJK_SENTENCE_ENDERS = ("。", "！", "？")

# Shared delivered-geometry fit check for the sliver-absorb merge acceptance.
try:
    from .rendering import cue_fits_delivered as _cue_fits
except Exception:  # pragma: no cover
    _cue_fits = None

# Immutable-boundary primitive — sliver absorption must never merge across a
# pause/authored/unknown-speaker wall; split children inherit the parent wall.
try:
    from .boundaries import (
        is_immutable_boundary as _is_immutable_boundary,
        propagate_boundary_to_children as _propagate_boundary,
    )
except Exception:  # pragma: no cover
    def _is_immutable_boundary(prev_cue, next_cue):
        return False

    def _propagate_boundary(parent, children):
        return children


def _measurement() -> str:
    """The spec's cps_measurement: 'characters' | 'words' | 'characters_no_spaces'.
    Sent from the spec via CPS_MEASUREMENT. Drives how _visible_chars counts."""
    return (_rule_get("CPS_MEASUREMENT", "characters") or "characters").strip().lower()


# ─── Spec knobs ──────────────────────────────────────────────────────
def _env_int(name: str, default: int) -> int:
    raw = _rule_get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _max_cps() -> int:
    return _env_int("CUSTOM_MAX_CPS", 45)


def _target_cps() -> int:
    return _env_int("CUSTOM_TARGET_CPS", 27)


def _min_cps() -> int:
    return _env_int("CUSTOM_MIN_CPS", 5)


def _min_display_ms() -> int:
    return _env_int("CUSTOM_MIN_DISPLAY_MS", 800)


def _max_display_ms() -> int:
    return _env_int("CUSTOM_MAX_DISPLAY_MS", 7000)


def _merge_gap_ms() -> int:
    return _env_int("CUSTOM_MERGE_GAP_MS", 80)


def _max_chars() -> int:
    return _env_int("CUSTOM_MAX_CHARS", 32)


def _max_lines() -> int:
    return _env_int("CUSTOM_MAX_LINES", 2)


# ─── Helpers ─────────────────────────────────────────────────────────
_CLAUSE_END = (",", ";", ":", "—", "–")
_SENTENCE_END = (".", "!", "?")

# UNIVERSAL LAW (mirrors shaping._split_breaks_paired_delimiter): a quoted /
# parenthetical / bracketed span is never split so its opener and closer land
# on different cues. The CPS split path is a separate splitter, so it needs the
# same guard — otherwise a CPS-driven split re-introduces the orphaned quote the
# shaping guard prevented. Deterministic, zero-cost, every timed-text form.
_OPEN_TO_CLOSE_CPS = {"\u201c": "\u201d", "(": ")", "[": "]", "\u2018": "\u2019", "\u00ab": "\u00bb"}
_CLOSE_TO_OPEN_CPS = {v: k for k, v in _OPEN_TO_CLOSE_CPS.items()}


def _split_breaks_paired_delimiter(words: List[str], idx: int) -> bool:
    if idx <= 0 or idx >= len(words):
        return False
    head = " ".join(words[:idx])
    stack = []
    dq_open = False
    for ch in head:
        if ch == '"':
            dq_open = not dq_open
        elif ch in _OPEN_TO_CLOSE_CPS:
            stack.append(ch)
        elif ch in _CLOSE_TO_OPEN_CPS:
            if stack and stack[-1] == _CLOSE_TO_OPEN_CPS[ch]:
                stack.pop()
    return bool(stack) or dq_open


def _visible_chars(cue: Dict[str, Any]) -> int:
    """Character count actually on screen, measured per the spec's
    CPS_MEASUREMENT. CJK always counts by character (no spaces). For Latin,
    'characters' = total incl. spaces (industry default), 'characters_no_spaces'
    drops spaces, 'words' counts whitespace tokens. This is what CPS is measured
    against, so it must match the spec the deliverable is graded against.
    POLICY (2026-07-06, operator-confirmed): EVERY character that renders on
    screen counts toward reading speed — speaker labels and dash prefixes
    INCLUDED. The viewer must read the label to know who is talking (Netflix
    TTSG / BBC posture: CPS is computed on the full delivered subtitle event).
    Parity with qc.py, condensation.py, and the Base44 graders."""
    text = " ".join(cue.get("lines", [])).replace("\n", " ").strip()
    if _is_cjk(text):
        return _cjk_count(text)
    mode = _measurement()
    if mode == "characters_no_spaces":
        return len(text.replace(" ", ""))
    if mode == "words":
        return len(text.split())
    return len(text)


def _duration_ms(cue: Dict[str, Any]) -> int:
    return max(1, int(cue.get("end_ms", 0)) - int(cue.get("start_ms", 0)))


def cue_cps(cue: Dict[str, Any]) -> float:
    """Reading speed of a cue in characters/second."""
    return _visible_chars(cue) / (_duration_ms(cue) / 1000.0)


def _cue_words(cue: Dict[str, Any]) -> List[str]:
    """The cue's spoken words, preferring the structured meta over the rendered
    lines so a split re-renders faithfully (and never re-splits a dash prefix)."""
    meta = cue.get("meta") or {}
    dt = meta.get("dialogue_text")
    if dt:
        return dt.split()
    return " ".join(cue.get("lines", [])).split()


def _primary_runs(cue: Dict[str, Any]) -> List[Dict[str, Any]]:
    runs = (cue.get("meta") or {}).get("runs") or []
    if runs:
        # Reset to a single run at word 0 for each split half — the half is one
        # contiguous slice of one speaker's words.
        return [{"speaker": runs[0].get("speaker"), "word_start": 0}]
    return [{"speaker": None, "word_start": 0}]


# ─── (1) EXTEND ──────────────────────────────────────────────────────
def extend_fast_cues(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Push each over-fast dialogue cue's end_ms later into the idle gap before
    the next cue (or, for the last cue, up to max_display_ms). Never creates an
    overlap, always preserves merge_gap, never exceeds max_display_ms. Pure
    timing change — text untouched."""
    max_cps = _max_cps()
    target_cps = _target_cps()
    min_gap = _merge_gap_ms()
    max_dur = _max_display_ms()

    for i, cue in enumerate(cues):
        if cue.get("type") != "dialogue":
            continue
        if cue_cps(cue) <= max_cps:
            continue
        chars = _visible_chars(cue)
        # The duration that would put this cue at target_cps (headroom below max).
        ideal_ms = int((chars / max(1, target_cps)) * 1000)
        ideal_ms = min(ideal_ms, max_dur)
        if ideal_ms <= _duration_ms(cue):
            continue
        new_end = cue["start_ms"] + ideal_ms
        # Clamp to the gap before the next cue.
        if i < len(cues) - 1:
            ceiling = cues[i + 1]["start_ms"] - min_gap
            new_end = min(new_end, ceiling)
        if new_end > cue["end_ms"]:
            cue["end_ms"] = new_end
    return cues


# ─── (2) SPLIT ───────────────────────────────────────────────────────
def _best_split_index(words: List[str]) -> int:
    """Pick the word index to split a cue's words into two halves. Prefer the
    clause/sentence boundary nearest the MIDDLE (so both halves are balanced and
    each ends on a natural break); fall back to the exact middle when no
    punctuation boundary exists. Never returns 0 or len(words)."""
    n = len(words)
    mid = n // 2
    best_i: Optional[int] = None
    best_dist = n + 1
    # A boundary AFTER word j means split index = j+1.
    # MIN-SIDE GUARD (2026-07-07): a boundary that strands a side under 3 words
    # ('By who?' | rest) is skipped — same rule as the shaping + chunker
    # pickers, so no split path in the engine can mint a tiny fragment cue.
    for j in range(n - 1):
        w = words[j].rstrip()
        if w.endswith(_CLAUSE_END) or w.endswith(_SENTENCE_END):
            idx = j + 1
            if idx <= 0 or idx >= n:
                continue
            if min(idx, n - idx) < 3:
                continue
            # UNIVERSAL: never split so an opening quote/bracket is orphaned.
            if _split_breaks_paired_delimiter(words, idx):
                continue
            dist = abs(idx - mid)
            if dist < best_dist:
                best_dist = dist
                best_i = idx
    if best_i is not None:
        return best_i
    return max(1, mid)


def _best_cjk_split_index(s: str) -> int:
    """Pick a split point for a CJK string near the MIDDLE, preferring the
    clause/sentence boundary (、。！？) closest to the middle. Returns a char
    index in 1..len(s)-1; falls back to the exact middle when no boundary."""
    n = len(s)
    mid = n // 2
    best = None
    best_dist = n + 1
    for i in range(n - 1):
        if s[i] in CJK_CLAUSE_ENDERS or s[i] in CJK_SENTENCE_ENDERS:
            idx = i + 1
            if idx <= 0 or idx >= n:
                continue
            dist = abs(idx - mid)
            if dist < best_dist:
                best_dist = dist
                best = idx
    return best if best is not None else max(1, mid)


def _split_cjk_cue(cue, max_cps, min_display, min_gap, max_chars, max_lines):
    """Split an over-fast CJK dialogue cue into two cues at the nearest clause
    boundary, interpolating timings by character count. Returns [left, right]
    only when BOTH halves end up ≤ max_cps and ≥ min_display — otherwise None
    (leave intact for the QC gate; never ship a worse state). Used ONLY by the
    deprecated split_fast_cues (no longer in the CPS-remediation chain) and by
    split_overlong_cues (which passes a permissive CPS ceiling for DURATION
    splitting). By character instead of word."""
    meta = cue.get("meta") or {}
    text = (meta.get("dialogue_text") or " ".join(cue.get("lines", []))).strip()
    if len(text) < 2:
        return None
    split_i = _best_cjk_split_index(text)
    left_text = text[:split_i].strip()
    right_text = text[split_i:].strip()
    if not left_text or not right_text:
        return None

    start = int(cue["start_ms"])
    end = int(cue["end_ms"])
    span = max(2, end - start)
    cut = start + (span * len(left_text)) // max(1, len(text))
    left_end = min(cut - (min_gap // 2), end - min_display)
    right_start = max(cut + (min_gap // 2), start + min_display)
    if left_end - start < min_display or end - right_start < min_display:
        return None

    runs = _primary_runs(cue)
    left_cue = {
        "idx": 0, "start_ms": start, "end_ms": int(left_end),
        "lines": render_lines(left_text.split(), runs, max_lines, max_chars, left_text),
        "type": "dialogue",
        "meta": {"dialogue_text": left_text, "runs": runs, "deprecated_split": True},
    }
    right_cue = {
        "idx": 0, "start_ms": int(right_start), "end_ms": end,
        "lines": render_lines(right_text.split(), runs, max_lines, max_chars, right_text),
        "type": "dialogue",
        "meta": {"dialogue_text": right_text, "runs": runs, "deprecated_split": True},
    }
    if cue_cps(left_cue) <= max_cps and cue_cps(right_cue) <= max_cps:
        return [left_cue, right_cue]
    return None


def split_fast_cues(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """DEPRECATED — REMOVED FROM THE CPS-REMEDIATION CHAIN (2026-07-16).

    This function is NO LONGER invoked by enforce_cps_rules. It is retained ONLY
    for test/external-caller compatibility and is scheduled for deletion once no
    caller references it. Do NOT reintroduce it into any remediation pipeline.

    PROVEN IMPOSSIBILITY (a timing-neutral split can never fix an over-CPS cue):
    For a cue with total visible characters C over total duration T that is over
    the limit,        C / T > max_cps.
    A timing-neutral split partitions it into children with
        chars   C1 + C2 = C
        durations T1 + T2 = T.
    If BOTH children were compliant we would have
        C1 / T1 <= max_cps   and   C2 / T2 <= max_cps
    hence  C1 <= max_cps·T1  and  C2 <= max_cps·T2, so
        C1 + C2 <= max_cps·(T1 + T2)   ⇒   C / T <= max_cps,
    contradicting the premise C / T > max_cps. Therefore splitting the SAME text
    over the SAME total window can NEVER turn one over-CPS cue into two compliant
    children — it is impossible by construction, not merely rare (the property
    sweep in tests/test_split_fast_cues_viability.py empirically confirms 0/2640,
    and tests/test_cps_invariant.py encodes the algebraic invariant directly).

    The genuine, non-destructive CPS remedies live elsewhere and run in this
    order (see enforce_cps_rules): (1) legal timing EXTENSION into idle silence,
    (2) cross-cue REDISTRIBUTION/RESEGMENTATION over the larger word-timed window
    (services.sequence_optimizer, upstream), (3) timing REBALANCE where measured
    word timings + adjacent capacity permit, (4) approved DISFLUENCY removal,
    (5) authorized CONDENSATION, (6) REVIEW-required when no safe remedy exists.
    DURATION/rhythm splitting of an over-length or wall-of-text cue is a SEPARATE
    concern owned by split_overlong_cues / shaping / sequence_optimizer, never by
    this function. SOC 2 CC8.1."""
    max_cps = _max_cps()
    min_display = _min_display_ms()
    min_gap = _merge_gap_ms()
    max_chars = _max_chars()
    max_lines = _max_lines()

    out: List[Dict[str, Any]] = []
    for cue in cues:
        if cue.get("type") != "dialogue" or cue_cps(cue) <= max_cps:
            out.append(cue)
            continue

        # CJK over-fast cues split by CHARACTER at 、。 boundaries — the
        # space-word splitter below can't break a no-space Japanese string.
        meta = cue.get("meta") or {}
        cjk_text = meta.get("dialogue_text") or " ".join(cue.get("lines", []))
        if _is_cjk(cjk_text):
            split = _split_cjk_cue(cue, max_cps, min_display, min_gap, max_chars, max_lines)
            if split:
                out.extend(split)
            else:
                out.append(cue)
            continue

        words = _cue_words(cue)
        if len(words) < 2:
            out.append(cue)  # can't split a single token
            continue

        split_i = _best_split_index(words)
        left_words = words[:split_i]
        right_words = words[split_i:]
        if not left_words or not right_words:
            out.append(cue)
            continue

        start = int(cue["start_ms"])
        end = int(cue["end_ms"])
        span = max(2, end - start)
        # Interpolate the split point proportionally to word count.
        cut = start + (span * len(left_words)) // len(words)
        # Honor min_gap between the two halves and min_display on each.
        left_end = min(cut - (min_gap // 2), end - min_display)
        right_start = max(cut + (min_gap // 2), start + min_display)
        if left_end - start < min_display or end - right_start < min_display:
            out.append(cue)  # no room to split cleanly — leave for QC gate
            continue

        left_text = " ".join(left_words)
        right_text = " ".join(right_words)
        left_cue = {
            "idx": 0, "start_ms": start, "end_ms": int(left_end),
            "lines": render_lines(left_words, _primary_runs(cue), max_lines, max_chars, left_text),
            "type": "dialogue",
            "meta": {"dialogue_text": left_text, "runs": _primary_runs(cue),
                     "deprecated_split": True},
        }
        right_cue = {
            "idx": 0, "start_ms": int(right_start), "end_ms": end,
            "lines": render_lines(right_words, _primary_runs(cue), max_lines, max_chars, right_text),
            "type": "dialogue",
            "meta": {"dialogue_text": right_text, "runs": _primary_runs(cue),
                     "deprecated_split": True},
        }
        # Accept the split only if it actually fixed the reading speed on both
        # halves. Otherwise keep the original cue (never ship a worse state).
        if cue_cps(left_cue) <= max_cps and cue_cps(right_cue) <= max_cps:
            out.append(left_cue)
            out.append(right_cue)
        else:
            out.append(cue)
    return out


# ─── (3) TRIM slow cues ──────────────────────────────────────────────
def trim_slow_cues(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Trim trailing dead air from cues reading BELOW min_cps (lingering). Pull
    end_ms back toward the duration that yields target_cps, never below
    min_display_ms and never below the cue's own start. Pure timing change."""
    min_cps = _min_cps()
    target_cps = _target_cps()
    min_display = _min_display_ms()

    for cue in cues:
        if cue.get("type") != "dialogue":
            continue
        if cue_cps(cue) >= min_cps:
            continue
        chars = _visible_chars(cue)
        if chars == 0:
            continue
        ideal_ms = int((chars / max(1, target_cps)) * 1000)
        ideal_ms = max(ideal_ms, min_display)
        new_end = cue["start_ms"] + ideal_ms
        if new_end < cue["end_ms"]:
            cue["end_ms"] = new_end
    return cues


# ─── (4) DURATION SPLIT ──────────────────────────────────────────────
def split_overlong_cues(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split any dialogue cue exceeding max_display_ms into shorter cues at a
    clause boundary, recursively, until every piece is within the duration
    ceiling (or can't be split further). This is the DURATION driver: a caption
    on screen for 8–13s is never spec-compliant regardless of CPS — Apple TV+
    caps a cue at 6s. CJK splits by character at 、。 boundaries; Latin by word
    at clause boundaries. Each accepted split preserves min_display + min_gap.
    Runs BEFORE the CPS pass so the CPS driver then refines what's left."""
    max_dur = _max_display_ms()
    min_display = _min_display_ms()
    min_gap = _merge_gap_ms()
    max_chars = _max_chars()
    max_lines = _max_lines()

    # Iterate to a fixed point — a single split may still leave a half over the
    # ceiling (a very long utterance). Bounded by cue count to avoid runaway.
    for _ in range(8):
        out: List[Dict[str, Any]] = []
        changed = False
        for cue in cues:
            if cue.get("type") != "dialogue" or _duration_ms(cue) <= max_dur:
                out.append(cue)
                continue
            meta = cue.get("meta") or {}
            text = meta.get("dialogue_text") or " ".join(cue.get("lines", []))
            split = None
            if _is_cjk(text):
                # Reuse the CJK clause splitter but with a permissive CPS ceiling
                # (we're splitting for DURATION here, not CPS) so it always
                # accepts a clause-boundary halving when room exists.
                split = _split_cjk_cue(cue, max_cps=10_000, min_display=min_display,
                                       min_gap=min_gap, max_chars=max_chars, max_lines=max_lines)
            else:
                words = _cue_words(cue)
                if len(words) >= 2:
                    i = _best_split_index(words)
                    lw, rw = words[:i], words[i:]
                    start, end = int(cue["start_ms"]), int(cue["end_ms"])
                    span = max(2, end - start)
                    cut = start + (span * len(lw)) // len(words)
                    le = min(cut - (min_gap // 2), end - min_display)
                    rs = max(cut + (min_gap // 2), start + min_display)
                    if le - start >= min_display and end - rs >= min_display:
                        lt, rt = " ".join(lw), " ".join(rw)
                        runs = _primary_runs(cue)
                        split = [
                            {"idx": 0, "start_ms": start, "end_ms": int(le), "type": "dialogue",
                             "lines": render_lines(lw, runs, max_lines, max_chars, lt),
                             "meta": {"dialogue_text": lt, "runs": runs, "duration_split": True}},
                            {"idx": 0, "start_ms": int(rs), "end_ms": end, "type": "dialogue",
                             "lines": render_lines(rw, runs, max_lines, max_chars, rt),
                             "meta": {"dialogue_text": rt, "runs": runs, "duration_split": True}},
                        ]
            if split:
                # A duration split introduces no new source-utterance pause —
                # carry the parent's opening wall onto child[0] only and the
                # review flag onto all children. SOC 2 CC8.1.
                out.extend(_propagate_boundary(cue, split))
                changed = True
            else:
                out.append(cue)
        cues = out
        if not changed:
            break
    return cues


# ─── (5) SLIVER ABSORB ───────────────────────────────────────────────
def _make_sliver_merge(start_ms, end_ms, joined_text, runs_source, max_lines, max_chars):
    """Build the merged candidate for a sliver absorption. Accepted only when
    the merged cue still fits the delivered line geometry (label included) —
    otherwise None and the sliver ships flagged for QC (never a worse state)."""
    runs = _primary_runs(runs_source)
    candidate = {
        "idx": 0, "start_ms": int(start_ms), "end_ms": int(end_ms),
        "type": "dialogue",
        "lines": render_lines(joined_text.split(), runs, max_lines, max_chars, joined_text),
        "meta": {"dialogue_text": joined_text, "runs": runs, "sliver_absorbed": True},
    }
    if _cue_fits is not None and not _cue_fits(candidate, max_lines, max_chars):
        return None
    return candidate


def absorb_sliver_cues(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """SLIVER GUARD — a dialogue cue below min_display_ms (e.g. an 80ms
    'Well,' fragment left behind by an upstream split) must never ship as-is:
    it is unreadable at ANY speed. Deterministic remedies, in order:
      (a) EXTEND into the trailing idle gap up to min_display (free);
      (b) MERGE into the adjacent same-speaker dialogue cue (next preferred —
          a sliver is usually the head of the following utterance; prev as
          fallback), re-rendered through the shared renderer, accepted only
          when the merged cue fits the delivered geometry and stays within
          max_display_ms;
      (c) leave it for the QC gate to flag honestly (a human must decide).
    Pure + reproducible. SOC 2 CC8.1 / FCC 47 CFR §79.1 (readability)."""
    min_display = _min_display_ms()
    min_gap = _merge_gap_ms()
    max_dur = _max_display_ms()
    max_chars = _max_chars()
    max_lines = _max_lines()

    def _speaker(c):
        for r in ((c.get("meta") or {}).get("runs") or []):
            if r.get("speaker") is not None:
                return r.get("speaker")
        return None

    def _text(c):
        return ((c.get("meta") or {}).get("dialogue_text") or " ".join(c.get("lines", []))).strip()

    out: List[Dict[str, Any]] = []
    absorbed = 0
    i = 0
    while i < len(cues):
        cue = cues[i]
        if cue.get("type") != "dialogue" or _duration_ms(cue) >= min_display:
            out.append(cue)
            i += 1
            continue
        # (a) extend into the trailing idle gap
        ceiling = int(cue["start_ms"]) + max_dur
        nxt = cues[i + 1] if i + 1 < len(cues) else None
        if nxt is not None:
            ceiling = min(ceiling, int(nxt.get("start_ms", ceiling)) - min_gap)
        needed_end = int(cue["start_ms"]) + min_display
        if needed_end <= ceiling:
            cue["end_ms"] = max(int(cue["end_ms"]), needed_end)
            out.append(cue)
            i += 1
            continue
        # (b) merge into a same-speaker neighbor
        sliver_text = _text(cue)
        joiner = "" if _is_cjk(sliver_text) else " "
        if (nxt is not None and nxt.get("type") == "dialogue"
                and _speaker(nxt) == _speaker(cue)
                and not _is_immutable_boundary(cue, nxt)
                and int(nxt["end_ms"]) - int(cue["start_ms"]) <= max_dur):
            merged = _make_sliver_merge(
                cue["start_ms"], nxt["end_ms"],
                (sliver_text + joiner + _text(nxt)).strip(),
                cue, max_lines, max_chars)
            if merged is not None:
                out.append(merged)
                absorbed += 1
                i += 2
                continue
        prev = out[-1] if out else None
        if (prev is not None and prev.get("type") == "dialogue"
                and _speaker(prev) == _speaker(cue)
                and not _is_immutable_boundary(prev, cue)
                and int(cue["end_ms"]) - int(prev["start_ms"]) <= max_dur):
            merged = _make_sliver_merge(
                prev["start_ms"], cue["end_ms"],
                (_text(prev) + joiner + sliver_text).strip(),
                prev, max_lines, max_chars)
            if merged is not None:
                out[-1] = merged
                absorbed += 1
                i += 1
                continue
        # (c) leave — QC flags it honestly
        out.append(cue)
        i += 1
    if absorbed:
        print(f"[CPS] sliver_absorb merged {absorbed} sub-min-duration fragment(s)")
    return out


# ─── Master pass ─────────────────────────────────────────────────────
def enforce_cps_rules(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply the deterministic drivers in the CORRECT, responsibility-separated
    order. Two distinct concerns, never conflated:

    ── DURATION / RHYTHM SPLITTING (geometry, NOT reading speed) ──
      0. split_overlong_cues — split any cue exceeding max_display_ms at a clause
         boundary. This is the ONLY splitter in this chain, and it exists purely
         to fix on-screen DURATION (a cue held 8–13s is non-compliant regardless
         of CPS). Wall-of-text / better-grammatical-segmentation splitting is
         owned UPSTREAM by services.shaping + services.sequence_optimizer, which
         run before this pass in the formatter. A timing-neutral single-cue split
         is NEVER used to lower CPS (see the impossibility proof below).

    ── CPS REMEDIATION (reading speed) — non-destructive remedies, in order ──
      1. LEGAL TIMING EXTENSION — extend_fast_cues stretches an over-fast cue into
         adjacent idle SILENCE toward target_cps, bounded by min_gap to the next
         cue. The primary free reducer applied here.
      2. CROSS-CUE REDISTRIBUTION / RESEGMENTATION over the larger word-timed
         window — services.sequence_optimizer (runs EARLIER in the formatter),
         which can move words across cue boundaries to rebalance local CPS. The
         real structural CPS improver.
      3. TIMING REBALANCE where actual word timings + adjacent capacity permit
         (extend_fast_cues + the optimizer, using measured word timings).
      4. APPROVED DISFLUENCY REMOVAL — deterministic filler/false-start removal
         (services.condensation, disfluency_only mode).
      5. AUTHORIZED CONDENSATION — bounded, entity/number-locked paraphrase, ONLY
         when the pinned spec permits it (condensation_rules.mode=condense_to_cps)
         AND the optimizer did not already record a non-destructive fix.
      6. REVIEW-REQUIRED — when no safe remedy exists, the QC gate (Phase 3) marks
         the cue over-CPS for a human. The engine never silently ships a violation
         and never destroys words to hide one.

    Then re-index. A TIMING-NEUTRAL SPLIT OF ONE ISOLATED CUE IS NOT A CPS REMEDY
    and is deliberately absent from this chain. Proof: for a cue with C visible
    chars over T ms where C/T > max_cps, any split into children (C1+C2=C,
    T1+T2=T) with both compliant (C1/T1 ≤ max_cps, C2/T2 ≤ max_cps) implies
    C1+C2 ≤ max_cps·(T1+T2), i.e. C/T ≤ max_cps — contradicting the premise. So
    both children can never be compliant; splitting the same text over the same
    window is impossible-by-construction as a CPS fix. The deprecated
    split_fast_cues (formerly step 2 here) was REMOVED from this chain on
    2026-07-16. See tests/test_cps_invariant.py (algebraic invariant) and
    tests/test_split_fast_cues_viability.py (empirical sweep). SOC 2 CC8.1."""
    cues = split_overlong_cues(cues)
    cues = extend_fast_cues(cues)
    cues = absorb_sliver_cues(cues)
    cues = trim_slow_cues(cues)
    for i, cue in enumerate(cues):
        cue["idx"] = i + 1
    return cues
