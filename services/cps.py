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


def _measurement() -> str:
    """The spec's cps_measurement: 'characters' | 'words' | 'characters_no_spaces'.
    Sent from the spec via CPS_MEASUREMENT. Drives how _visible_chars counts."""
    return (os.getenv("CPS_MEASUREMENT", "characters") or "characters").strip().lower()


# ─── Spec knobs ──────────────────────────────────────────────────────
def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
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


def _visible_chars(cue: Dict[str, Any]) -> int:
    """Character count actually on screen, measured per the spec's
    CPS_MEASUREMENT. CJK always counts by character (no spaces). For Latin,
    'characters' = total incl. spaces (industry default), 'characters_no_spaces'
    drops spaces, 'words' counts whitespace tokens. This is what CPS is measured
    against, so it must match the spec the deliverable is graded against."""
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
    for j in range(n - 1):
        w = words[j].rstrip()
        if w.endswith(_CLAUSE_END) or w.endswith(_SENTENCE_END):
            idx = j + 1
            if idx <= 0 or idx >= n:
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
    (leave intact for the QC gate; never ship a worse state). Mirrors the Latin
    split_fast_cues acceptance contract exactly, by character instead of word."""
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
        "meta": {"dialogue_text": left_text, "runs": runs, "cps_split": True},
    }
    right_cue = {
        "idx": 0, "start_ms": int(right_start), "end_ms": end,
        "lines": render_lines(right_text.split(), runs, max_lines, max_chars, right_text),
        "type": "dialogue",
        "meta": {"dialogue_text": right_text, "runs": runs, "cps_split": True},
    }
    if cue_cps(left_cue) <= max_cps and cue_cps(right_cue) <= max_cps:
        return [left_cue, right_cue]
    return None


def split_fast_cues(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """For dialogue cues STILL over max_cps after extension, split into two cues
    at the best clause boundary. Each half is re-rendered through the shared
    renderer and its time window is interpolated proportionally to word count
    over the original [start_ms, end_ms]. The split is only accepted when BOTH
    halves end up at or below max_cps AND each meets min_display_ms — otherwise
    the cue is left intact for the QC gate to flag (we never make it worse)."""
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
                     "cps_split": True},
        }
        right_cue = {
            "idx": 0, "start_ms": int(right_start), "end_ms": end,
            "lines": render_lines(right_words, _primary_runs(cue), max_lines, max_chars, right_text),
            "type": "dialogue",
            "meta": {"dialogue_text": right_text, "runs": _primary_runs(cue),
                     "cps_split": True},
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
                out.extend(split)
                changed = True
            else:
                out.append(cue)
        cues = out
        if not changed:
            break
    return cues


# ─── Master pass ─────────────────────────────────────────────────────
def enforce_cps_rules(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Apply the deterministic drivers in priority order:
      0. split cues exceeding max_display_ms at a clause boundary (duration),
      1. extend over-fast cues into idle gap (free),
      2. split the still-over-fast cues at a clause boundary (CPS),
      3. trim over-slow lingering cues.
    Then re-index. Anything still out of budget after this is a genuine
    editorial decision the QC gate (Phase 3) surfaces — the engine has done
    every deterministic thing it safely can."""
    cues = split_overlong_cues(cues)
    cues = extend_fast_cues(cues)
    cues = split_fast_cues(cues)
    cues = trim_slow_cues(cues)
    for i, cue in enumerate(cues):
        cue["idx"] = i + 1
    return cues
