"""
Cross-Cue Sequence Optimizer — the canonical Timed-Text Editorial Segmentation
stage. THIS is the professional-captioner brain that evaluates a CONTIGUOUS
SEQUENCE of same-speaker dialogue cues as one word-timed window and REPLACES the
local boundary structure with the best compliant arrangement — instead of
treating each upstream cue boundary as sacred.

WHY THIS EXISTS (the two failure classes it closes)
───────────────────────────────────────────────────
Every stage before this one decides one cue at a time:
  • shaping.py splits a single over-rhythm cue in isolation;
  • cps.py extends/splits/absorbs per cue (only its pairwise sliver-absorb looks
    at a neighbour, and only below min_display);
  • condensation.py is the ONLY over-CPS remedy after split/extend fail — so an
    over-CPS cue whose overflow could have been RESEGMENTED across the window
    instead gets its WORDS DELETED/REWORDED.

Neither of the reported defects is fixable one-cue-at-a-time:
  1. FLASH FRAGMENT — a long sentence gets a trailing chunk with a tiny
     proportional time-slice; it is technically ≥ min_display yet reads as a
     flash, and nothing re-balances it against the under-filled neighbour.
  2. REMOVED PHRASE — an over-CPS cue reaches condensation as a full sentence
     and its words are cut, when a two- or three-cue RESEGMENTATION of the same
     word-timed span would have kept every word within budget.

Both share one root cause: NO stage evaluates the adjacent cue window as a whole
and is allowed to REPLACE the boundaries. This module is that stage.

CONTRACT / GUARANTEES (enterprise A+, auditor-passing)
──────────────────────────────────────────────────────
• DETERMINISTIC. No AI, no I/O. Identical (cues, env, version) → identical
  output, byte-for-byte. SOC 2 CC8.1.
• BOUNDARY REPLACEMENT, not word-appending. For each same-speaker window the
  optimizer regenerates candidate 1-/2-/3-cue arrangements from the ORIGINAL
  word-timed token stream and picks the best compliant one. "No change" is
  always a candidate and wins on ties (stability).
• HARD VETOES before scoring: a candidate that loses/reorders a word, breaks a
  paired delimiter, crosses a speaker boundary, or violates line/CPL/duration/
  gap geometry is rejected outright — never scored, never selected.
• TEXT CONSERVATION: the concatenated dialogue tokens of a selected candidate
  MUST equal the window's original tokens, in order. Enforced as a veto.
• WORD TIMINGS ARE AUTHORITATIVE. Boundaries land on real word end_ms; timing
  is frame-faithful. Interpolation is only a labelled fallback when a window
  lacks per-word timings.
• REDISTRIBUTION-BEFORE-CONDENSATION. Every cue the optimizer touches (and every
  cue it deliberately left whole after evaluating candidates) is stamped
  meta.seq_opt.condensation_allowed with an explicit reason. The condensation
  stage refuses to reword an over-CPS cue unless the optimizer recorded that NO
  compliant non-destructive candidate existed. This is the executable guard your
  precedence rules require.
• PROVENANCE. Every optimized window writes a bounded meta.seq_opt audit block
  (versions, source cue ids, original window text/boundaries, candidate
  summaries, selected op, rejected reason codes, moved-word count, timing-change
  count, text-conservation result, condensation gate + reason). The formatter
  surfaces the meaningful summary onto the result cue (not just condensation).

PLACEMENT (formatter.process_caption_job)
──────────────────────────────────────────
Runs AFTER initial rhythm shaping (word timings intact on meta) and BEFORE
editorial-AI / readability+CPS / condensation. This is the placement that (a)
still has word timings, (b) can replace boundaries before any destructive stage,
and (c) is not undone later — readability's merge/CPS passes only fire on cues
that are still non-compliant, and a window the optimizer made compliant has no
trigger left. Documented + test-locked (see tests/test_sequence_optimizer*.py).

SCRIPT / LANGUAGE
─────────────────
Latin path implemented in this slice (both reported failures are English). CJK
windows are passed through UNCHANGED (the shaper's CJK char-splitter already owns
that path); a CJK window is detected and skipped with a recorded reason, never
mis-split. The language-policy interface (Step 11) lands in a later slice; the
seam is the _phrase_boundaries / _bare helpers, already delegated to linebreak.

Pure functions only. No env writes.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

# ─── Versioning (Step 17) ────────────────────────────────────────────
# Bump SEGMENTATION_POLICY_VERSION when the candidate-generation or scoring
# LOGIC changes in a way that could alter selected boundaries. Bump
# OPTIMIZER_VERSION for any change to this module. Both are pinned into every
# provenance block so an auditor can answer "which optimizer produced this cue?"
OPTIMIZER_VERSION = 1
SEGMENTATION_POLICY_VERSION = 1

# Shared linguistic tables + bare-word normalizer — the SAME ones the line
# breaker and shaper use, so boundary quality is judged identically everywhere.
try:
    from .linebreak import _LEADING_WORDS, _DETERMINERS, _bare as _bare_word
except Exception:  # pragma: no cover
    _LEADING_WORDS = frozenset()
    _DETERMINERS = frozenset()

    def _bare_word(w):
        return (w or "").strip(".,;:!?—–\"')]}").lower()

try:
    from .cjk import is_cjk_text as _is_cjk
except Exception:  # pragma: no cover
    def _is_cjk(text):
        return False

try:
    from .rendering import render_lines as _render_lines, cue_fits_delivered as _cue_fits_delivered
except Exception:  # pragma: no cover
    def _render_lines(words, runs, max_lines=None, max_chars=None, dialogue_text=None):
        return [dialogue_text if dialogue_text is not None else " ".join(words)]

    def _cue_fits_delivered(cue, max_lines=None, max_chars=None):
        return True

_SENTENCE_END = (".", "!", "?")
_CLAUSE_END = (",", ";", ":", "—", "–")
_OPEN_TO_CLOSE = {"\u201c": "\u201d", "(": ")", "[": "]", "\u2018": "\u2019", "\u00ab": "\u00bb"}
_CLOSE_TO_OPEN = {v: k for k, v in _OPEN_TO_CLOSE.items()}


# ─── Spec knobs (identical names/defaults to shaping.py + cps.py) ─────
def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _enabled() -> bool:
    """Controlled rollout flag (Step 17). Default ON — the optimizer is the
    canonical path. A spec/run can disable it (SEQ_OPTIMIZER_ENABLED=0) for an
    A/B or a 1:1 import posture that must not be resegmented."""
    return os.getenv("SEQ_OPTIMIZER_ENABLED", "1") not in ("0", "false", "False")


def _max_chars() -> int:
    return _env_int("CUSTOM_MAX_CHARS", 32)


def _max_lines() -> int:
    return _env_int("CUSTOM_MAX_LINES", 2)


def _min_display_ms() -> int:
    return _env_int("CUSTOM_MIN_DISPLAY_MS", 800)


def _max_display_ms() -> int:
    return _env_int("CUSTOM_MAX_DISPLAY_MS", 7000)


def _min_gap_ms() -> int:
    return _env_int("CUSTOM_MERGE_GAP_MS", 80)


def _max_cps() -> int:
    return _env_int("CUSTOM_MAX_CPS", 45)


def _target_cps() -> int:
    return _env_int("CUSTOM_TARGET_CPS", 27)


def _target_duration_ms() -> int:
    target = _env_int("CUSTOM_TARGET_DURATION_MS", 3000)
    return max(target, 2 * _min_display_ms())


def _micro_flash_ms() -> int:
    """Editorial flash ceiling (Step 6). A cue below this is a FLASH even if it
    technically clears min_display — the professional 'momentary trailing cue'
    the operator flagged. Distinct from min_display (the hard floor). Default =
    the smaller of 1200ms or 1.4×min_display, so a spec with a large min_display
    doesn't get an even larger flash window."""
    v = _env_int("SEQ_OPT_FLASH_MS", 0)
    if v > 0:
        return v
    return min(1200, int(_min_display_ms() * 1.4))


# ─── Window helpers ──────────────────────────────────────────────────
def _cue_text(cue: Dict[str, Any]) -> str:
    meta = cue.get("meta") or {}
    return (meta.get("dialogue_text") or " ".join(cue.get("lines", []))).strip()


def _cue_words(cue: Dict[str, Any]) -> List[str]:
    t = _cue_text(cue)
    return t.split() if t else []


def _primary_speaker(cue: Dict[str, Any]) -> Optional[str]:
    for r in ((cue.get("meta") or {}).get("runs") or []):
        if r.get("speaker") is not None:
            return r.get("speaker")
    return None


def _word_timings(cue: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for w in ((cue.get("meta") or {}).get("word_timings") or []):
        try:
            out.append({
                "text": str(w.get("text", "")),
                "start_ms": int(w.get("start_ms", w.get("start", 0)) or 0),
                "end_ms": int(w.get("end_ms", w.get("end", 0)) or 0),
            })
        except Exception:
            continue
    return out


def _ends_sentence(word: str) -> bool:
    return bool(word) and word.rstrip("\"')]}»”’").endswith(_SENTENCE_END)


def _split_breaks_paired_delimiter(words: List[str], idx: int) -> bool:
    """True when splitting `words` at index `idx` orphans an open quote/bracket
    on the head from its closer in the tail. Mirrors shaping/cps guards exactly."""
    if idx <= 0 or idx >= len(words):
        return False
    head = " ".join(words[:idx])
    stack = []
    dq_open = False
    for ch in head:
        if ch == '"':
            dq_open = not dq_open
        elif ch in _OPEN_TO_CLOSE:
            stack.append(ch)
        elif ch in _CLOSE_TO_OPEN:
            if stack and stack[-1] == _CLOSE_TO_OPEN[ch]:
                stack.pop()
    return bool(stack) or dq_open


# ─── Window = a maximal run of adjacent SAME-SPEAKER dialogue cues ───
def _collect_windows(cues: List[Dict[str, Any]]) -> List[Tuple[int, int]]:
    """Return [start, end) index ranges of maximal same-speaker dialogue runs.
    A window is the optimizer's evaluation unit. Windows never cross a speaker
    change or a non-dialogue (sound/music) cue — those are protected boundaries
    (Step 4 hard veto: never move a word across a speaker boundary)."""
    windows: List[Tuple[int, int]] = []
    i = 0
    n = len(cues)
    while i < n:
        cue = cues[i]
        if cue.get("type") != "dialogue":
            i += 1
            continue
        spk = _primary_speaker(cue)
        j = i + 1
        while j < n and cues[j].get("type") == "dialogue" and _primary_speaker(cues[j]) == spk:
            j += 1
        # Only windows of ≥2 cues are candidates for resegmentation; a single
        # standalone cue has no neighbour to redistribute with (the shaper
        # already handled its internal split). We still evaluate a lone cue's
        # condensation gate below, so include size-1 windows too.
        windows.append((i, j))
        i = j
    return windows


def _window_word_stream(cues: List[Dict[str, Any]], start: int, end: int
                        ) -> Tuple[List[str], List[Dict[str, Any]], int, int, bool]:
    """Flatten a window into (words, per_word_timings, win_start_ms, win_end_ms,
    have_timings). Word order is preserved exactly. Timings prefer the cue's real
    meta.word_timings; a cue lacking them contributes interpolated placeholders so
    the stream stays index-aligned to words (have_timings=False when ANY cue in
    the window lacked real timings → interpolation-fallback provenance)."""
    words: List[str] = []
    timings: List[Dict[str, Any]] = []
    have_all = True
    for k in range(start, end):
        cue = cues[k]
        cw = _cue_words(cue)
        wt = _word_timings(cue)
        c_start = int(cue.get("start_ms", 0))
        c_end = int(cue.get("end_ms", c_start + 1))
        if len(wt) == len(cw) and cw:
            for w, t in zip(cw, wt):
                words.append(w)
                timings.append({"start_ms": t["start_ms"], "end_ms": t["end_ms"]})
        else:
            # Interpolate this cue's words proportionally over its window.
            have_all = False
            span = max(1, c_end - c_start)
            total = sum(len(w) for w in cw) or 1
            cursor = 0
            for w in cw:
                s = c_start + span * cursor // total
                cursor += len(w)
                e = c_start + span * cursor // total
                words.append(w)
                timings.append({"start_ms": int(s), "end_ms": int(max(e, s + 1))})
    win_start = int(cues[start].get("start_ms", 0))
    win_end = int(cues[end - 1].get("end_ms", win_start + 1))
    return words, timings, win_start, win_end, have_all


# ─── Candidate = a proposed arrangement of the window's words into cues ─
def _cut_indices_to_cues(words, timings, cuts, win_start, win_end, speaker
                         ) -> List[Dict[str, Any]]:
    """Turn a sorted list of word cut-indices into concrete cue dicts with
    frame-faithful timings (word end_ms at each boundary), honoring min_gap and
    min_display clamping. Returns [] when a cut can't be timed legally."""
    max_chars, max_lines = _max_chars(), _max_lines()
    min_display, min_gap = _min_display_ms(), _min_gap_ms()
    bounds = [0] + list(cuts) + [len(words)]
    runs = [{"speaker": speaker, "word_start": 0}]
    out: List[Dict[str, Any]] = []
    n_parts = len(bounds) - 1
    for p in range(n_parts):
        a, b = bounds[p], bounds[p + 1]
        seg_words = words[a:b]
        if not seg_words:
            return []
        seg_start = win_start if p == 0 else int(timings[a]["start_ms"])
        seg_end = win_end if p == n_parts - 1 else int(timings[b - 1]["end_ms"])
        # Honor min_gap between consecutive parts.
        if p > 0 and out:
            prev = out[-1]
            if seg_start - prev["end_ms"] < min_gap:
                seg_start = prev["end_ms"] + min_gap
        if seg_end - seg_start < 1:
            return []
        text = " ".join(seg_words)
        out.append({
            "idx": 0,
            "start_ms": int(seg_start),
            "end_ms": int(seg_end),
            "lines": _render_lines(seg_words, runs, max_lines, max_chars, text),
            "type": "dialogue",
            "meta": {"dialogue_text": text, "runs": runs,
                     "word_timings": [dict(timings[i]) | {"text": words[i]} for i in range(a, b)]},
        })
    return out


def _candidate_cut_sets(words: List[str], orig_cut_indices: Tuple[int, ...]
                        ) -> List[Tuple[str, Tuple[int, ...]]]:
    """Generate deterministic candidate cut-index sets over the word stream.
    Each is (operation_code, tuple_of_cut_indices).

    CRITICAL: "no_change" reproduces the ORIGINAL per-cue boundaries
    (orig_cut_indices — the word indices where the incoming cues actually
    divided), NOT a single collapsed part. Collapsing the window into one cue is
    a real resegmentation (a MERGE), which must compete on merit and be beaten by
    a balanced split when the merged cue is a wall of text — it is emitted here
    as a distinct 'merge_all' candidate, never masquerading as 'no change'.

    Candidates: no_change (original boundaries) · merge_all (one cue) · every
    clean single cut (2-cue) · every clean double cut (3-cue). "Clean" = passes
    the paired-delimiter guard and doesn't strand a leading function word.
    Bounded O(n^2); n = window word count (tens). Deterministic."""
    n = len(words)
    cands: List[Tuple[str, Tuple[int, ...]]] = [("no_change", tuple(orig_cut_indices))]
    # The single-cue arrangement is a legitimate candidate (a small window may
    # read best merged) but it is NOT the baseline — only offer it when it isn't
    # already the original arrangement.
    if orig_cut_indices:
        cands.append(("merge_all", ()))
    if n < 4:
        return cands

    def _clean_cut(idx: int) -> bool:
        if idx <= 0 or idx >= n:
            return False
        if _split_breaks_paired_delimiter(words, idx):
            return False
        # Never start a part with a stranded leading function word / determiner.
        first = _bare_word(words[idx])
        if first in _LEADING_WORDS or first in _DETERMINERS:
            return False
        # Tiny-side guard — same ≥3-word rule as every other picker in the engine.
        return min(idx, n - idx) >= 3

    single = [i for i in range(1, n) if _clean_cut(i)]
    for i in single:
        cands.append(("resegment_2", (i,)))
    # 3-cue: pairs of clean cuts, each part ≥3 words.
    for a_i in range(len(single)):
        for b_i in range(a_i + 1, len(single)):
            a, b = single[a_i], single[b_i]
            if b - a < 3:
                continue
            cands.append(("resegment_3", (a, b)))
    return cands


# ─── Hard vetoes (Step 4 / candidate scoring) ────────────────────────
def _veto(parts: List[Dict[str, Any]], orig_words: List[str]
          ) -> Optional[str]:
    """Return a veto reason code, or None when the candidate is compliant.
    Vetoes run BEFORE scoring — a vetoed candidate is never selectable."""
    max_chars, max_lines = _max_chars(), _max_lines()
    min_display, max_display = _min_display_ms(), _max_display_ms()
    max_cps = _max_cps()
    # RHYTHM CEILING (Step 6, editorial): a single cue running noticeably longer
    # than the reading-rhythm target reads as a wall of text even when it's under
    # the hard max_display and within CPS. Ceiling = target × 1.5 (same threshold
    # shaping._needs_shaping uses to decide a cue is over-rhythm). A candidate
    # containing an over-rhythm part is vetoed so a balanced split always beats
    # the merged wall-of-text — UNLESS no split is possible, in which case every
    # candidate vetoes here and the caller's chosen-is-None path lets the
    # downstream rhythm/CPS stages handle it. Deterministic, spec-driven.
    rhythm_ceiling = int(_target_duration_ms() * 1.5)
    if not parts:
        return "EMPTY_CANDIDATE"
    # TEXT CONSERVATION — concatenated tokens must equal the original, in order.
    got = []
    for p in parts:
        got.extend((p["meta"]["dialogue_text"]).split())
    if got != orig_words:
        return "TEXT_CONSERVATION_FAILED"
    for p in parts:
        dur = p["end_ms"] - p["start_ms"]
        if dur < min_display:
            return "DURATION_BELOW_MIN"
        if dur > max_display:
            return "DURATION_ABOVE_MAX"
        if dur > rhythm_ceiling:
            return "OVER_RHYTHM_CEILING"
        if not _cue_fits_delivered(p, max_lines, max_chars):
            return "LINE_OR_CPL_FAIL"
        # CPS veto — a resegmentation that is STILL over budget is not a
        # non-destructive win; it must not be selected over condensation's
        # honest handling. (no_change is exempt: it's the baseline, not a fix.)
        body = " ".join(p.get("lines", [])).strip() or p["meta"]["dialogue_text"]
        dur_s = max(0.001, dur / 1000.0)
        if (len(body) / dur_s) > max_cps:
            return "CPS_OVER_MAX"
    return None


# ─── Deterministic scoring (Step 4) ──────────────────────────────────
def _score(parts: List[Dict[str, Any]], op: str, orig_part_count: int) -> float:
    """Higher is better. All-additive, deterministic. Rewards natural grammar
    boundaries, phrase integrity, CPS balance, duration balance, line balance;
    penalizes flashes, fragmentation, and disturbance (a no_change/stability
    bias so already-good windows are never churned)."""
    max_cps = _max_cps()
    target_cps = _target_cps()
    target_dur = _target_duration_ms()
    flash_ms = _micro_flash_ms()
    score = 0.0

    cps_vals, durs = [], []
    for i, p in enumerate(parts):
        dur = p["end_ms"] - p["start_ms"]
        durs.append(dur)
        body = " ".join(p.get("lines", [])).strip() or p["meta"]["dialogue_text"]
        cps = len(body) / max(0.001, dur / 1000.0)
        cps_vals.append(cps)
        words = p["meta"]["dialogue_text"].split()
        # Grammar boundary: reward a part that ENDS on sentence/clause punct
        # (a natural break), except the final part which just ends the window.
        if i < len(parts) - 1:
            last = words[-1].rstrip()
            if last.endswith(_SENTENCE_END):
                score += 12.0
            elif last.endswith(_CLAUSE_END):
                score += 8.0
            else:
                score -= 6.0  # mid-phrase boundary is worse
        # Flash penalty (editorial, not just min_display).
        if dur < flash_ms:
            score -= 20.0
        # Duration proximity to the rhythm target.
        score -= abs(dur - target_dur) / 1000.0 * 1.5

    # CPS balance across the window — reward every part comfortably under max,
    # penalize variance so we don't trade one over-fast cue for a lopsided pair.
    if cps_vals:
        score -= max(0.0, max(cps_vals) - target_cps) * 1.2
        if len(cps_vals) > 1:
            spread = max(cps_vals) - min(cps_vals)
            score -= spread * 0.4
    # Duration balance.
    if len(durs) > 1:
        score -= (max(durs) - min(durs)) / 1000.0 * 0.8
    # Fragmentation penalty — more parts = more fragmentation; slight bias to
    # the fewest cues that solve the problem (2 beats 3 on ties).
    score -= (len(parts) - 1) * 2.0
    # STABILITY bias — prefer leaving an already-good window alone.
    if op == "no_change":
        score += 5.0
    return score


def _summarize_candidate(op: str, cuts, parts, veto_reason) -> Dict[str, Any]:
    return {
        "op": op,
        "cuts": list(cuts),
        "part_count": len(parts) if parts else 0,
        "vetoed": veto_reason is not None,
        "veto_reason": veto_reason,
    }


# ─── Public entry ────────────────────────────────────────────────────
def optimize_cue_sequence(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Canonical cross-cue optimization pass. For each maximal same-speaker
    dialogue window: rebuild the word-timed stream, generate candidate 1/2/3-cue
    arrangements, veto the non-compliant, score the rest, select the best, and
    replace the window's cues with it — stamping full meta.seq_opt provenance +
    the condensation gate on every emitted cue. Non-dialogue cues, CJK windows,
    and single-cue windows with no better candidate pass through unchanged
    (with a recorded reason). Disabled (SEQ_OPTIMIZER_ENABLED=0) → no-op.

    SOC 2 CC8.1 — deterministic, attributable, reproducible."""
    if not _enabled() or not cues:
        for i, c in enumerate(cues):
            c["idx"] = i + 1
        return cues

    windows = _collect_windows(cues)
    out: List[Dict[str, Any]] = []
    cursor = 0
    for (start, end) in windows:
        # Emit any non-dialogue cues sitting before this window verbatim.
        while cursor < start:
            out.append(cues[cursor])
            cursor += 1

        window_cues = cues[start:end]
        speaker = _primary_speaker(window_cues[0])
        words, timings, win_start, win_end, have_timings = _window_word_stream(cues, start, end)
        orig_text = " ".join(words)
        orig_boundaries = [int(c.get("start_ms", 0)) for c in window_cues]
        # Original cut indices = cumulative word counts at each internal cue
        # boundary. This makes "no_change" reproduce the incoming arrangement
        # exactly (not a collapsed single cue).
        orig_cuts: List[int] = []
        acc = 0
        for c in window_cues[:-1]:
            acc += len(_cue_words(c))
            if 0 < acc < len(words):
                orig_cuts.append(acc)
        orig_cuts_t = tuple(orig_cuts)

        # CJK windows: skip (the shaper's char-splitter owns that path). Stamp a
        # reason so the audit shows the optimizer saw it and deferred.
        if _is_cjk(orig_text):
            for c in window_cues:
                _stamp_passthrough(c, "cjk_window_deferred", have_timings)
                out.append(c)
            cursor = end
            continue

        candidates = _candidate_cut_sets(words, orig_cuts_t)
        summaries: List[Dict[str, Any]] = []
        scored: List[Tuple[float, str, tuple, List[Dict[str, Any]]]] = []
        for op, cuts in candidates:
            parts = _cut_indices_to_cues(words, timings, cuts, win_start, win_end, speaker)
            veto = _veto(parts, words) if parts else "EMPTY_CANDIDATE"
            # no_change reproduces the ORIGINAL boundaries; if it vetoes on CPS or
            # over-rhythm that's the SIGNAL a fix is needed — recorded, not hidden.
            summaries.append(_summarize_candidate(op, cuts, parts, veto))
            if veto is None:
                scored.append((_score(parts, op, len(window_cues)), op, cuts, parts))

        # Deterministic selection: highest score; tie → fewest parts → earliest
        # cuts → no_change. Sort key makes this total + reproducible.
        chosen = None
        if scored:
            scored.sort(key=lambda t: (-t[0], len(t[3]), t[2] if t[2] else (), 0 if t[1] == "no_change" else 1))
            chosen = scored[0]

        # ── Condensation gate (redistribution-before-condensation) ──────────
        # Did a COMPLIANT non-destructive candidate exist that keeps every word
        # within CPS? If yes → condensation is FORBIDDEN on these cues. If the
        # only survivor is no_change AND no_change itself is over-CPS (i.e. no
        # resegmentation could rescue it), condensation is ALLOWED with a reason.
        non_destructive_fix_exists = any(
            s[1] != "no_change" for s in scored
        )
        # Was the baseline (no_change) over budget? (Was it vetoed on CPS?)
        no_change_over_cps = any(
            c["op"] == "no_change" and c["veto_reason"] == "CPS_OVER_MAX"
            for c in summaries
        )

        if chosen is None:
            # Nothing compliant at all (rare) — leave originals, allow condensation
            # to try (it's the honest last resort), reason recorded.
            for c in window_cues:
                _stamp_passthrough(c, "no_compliant_candidate", have_timings,
                                   condensation_allowed=True,
                                   condensation_reason="no_compliant_resegmentation",
                                   summaries=summaries, orig_text=orig_text,
                                   orig_boundaries=orig_boundaries, speaker=speaker)
                out.append(c)
            cursor = end
            continue

        _score_val, op, cuts, parts = chosen
        # Was any word moved across an original boundary? (op != no_change ⇒ yes.)
        moved_words = 0
        if op != "no_change":
            moved_words = _count_moved_words(window_cues, parts)

        condensation_allowed = (op == "no_change" and no_change_over_cps and not non_destructive_fix_exists)
        condensation_reason = (
            "no_compliant_resegmentation" if condensation_allowed
            else ("resegmented_within_budget" if op != "no_change"
                  else "within_budget_no_change")
        )

        timing_source = "word_timings" if have_timings else "interpolated"
        seg_quality = "optimized" if op != "no_change" else "compliant"

        prov = {
            "optimizer_version": OPTIMIZER_VERSION,
            "policy_version": SEGMENTATION_POLICY_VERSION,
            "source_cue_ids": [c.get("idx") for c in window_cues],
            "source_cue_count": len(window_cues),
            "original_window_text": orig_text[:2000],
            "original_boundaries_ms": orig_boundaries,
            "speaker": speaker,
            "operation": op,
            "candidates_considered": len(summaries),
            "candidate_summaries": summaries[:40],  # bounded audit
            "selected_cuts": list(cuts),
            "moved_word_count": moved_words,
            "timing_change": op != "no_change",
            "timing_source": timing_source,
            "text_conservation": "passed",  # a vetoed candidate never reaches here
            "condensation_evaluated": True,
            "condensation_allowed": condensation_allowed,
            "condensation_reason": condensation_reason,
            "segmentation_quality": seg_quality,
        }

        for pi, p in enumerate(parts):
            meta = p.get("meta") or {}
            meta["seq_opt"] = dict(prov) | {"part_index": pi, "part_total": len(parts)}
            p["meta"] = meta
            out.append(p)
        cursor = end

    # Trailing non-dialogue cues after the last window.
    while cursor < len(cues):
        out.append(cues[cursor])
        cursor += 1

    for i, c in enumerate(out):
        c["idx"] = i + 1
    return out


def _count_moved_words(orig_cues: List[Dict[str, Any]], new_parts: List[Dict[str, Any]]) -> int:
    """Count words whose owning cue index changed between the original window
    arrangement and the new one. Bounded, deterministic — the 'minimal
    disturbance' metric surfaced in provenance."""
    def _owner_map(cues):
        m = []
        for ci, c in enumerate(cues):
            for _ in _cue_words(c):
                m.append(ci)
        return m
    o = _owner_map(orig_cues)
    n = _owner_map(new_parts)
    if len(o) != len(n):
        return len(o)
    # Normalize each side's part-index sequence to first-appearance order so a
    # pure re-timing with identical grouping counts as 0 moved words.
    return sum(1 for a, b in zip(o, n) if a != b)


def _stamp_passthrough(cue, reason, have_timings, condensation_allowed=False,
                       condensation_reason="not_evaluated", summaries=None,
                       orig_text=None, orig_boundaries=None, speaker=None):
    """Stamp a minimal seq_opt provenance block on a cue the optimizer left
    unchanged, so EVERY dialogue cue carries an audit trail (no silent gaps)."""
    meta = cue.get("meta") or {}
    meta["seq_opt"] = {
        "optimizer_version": OPTIMIZER_VERSION,
        "policy_version": SEGMENTATION_POLICY_VERSION,
        "operation": "no_change",
        "passthrough_reason": reason,
        "timing_source": "word_timings" if have_timings else "interpolated",
        "condensation_evaluated": reason != "cjk_window_deferred",
        "condensation_allowed": condensation_allowed,
        "condensation_reason": condensation_reason,
        "segmentation_quality": "passthrough",
    }
    if summaries is not None:
        meta["seq_opt"]["candidate_summaries"] = summaries[:40]
        meta["seq_opt"]["candidates_considered"] = len(summaries)
    if orig_text is not None:
        meta["seq_opt"]["original_window_text"] = orig_text[:2000]
    if orig_boundaries is not None:
        meta["seq_opt"]["original_boundaries_ms"] = orig_boundaries
    if speaker is not None:
        meta["seq_opt"]["speaker"] = speaker
    cue["meta"] = meta


def condensation_is_blocked(cue: Dict[str, Any]) -> bool:
    """Executable redistribution-before-condensation guard (Step 5 / precedence).
    The condensation stage calls this before rewording an over-CPS cue. Returns
    True (BLOCK the reword) when the optimizer ran and recorded that a compliant
    non-destructive arrangement existed — i.e. condensation_allowed is False.
    Returns False (permit) when the optimizer explicitly allowed condensation
    (no compliant resegmentation), or when no optimizer provenance exists at all
    (back-compat: a cue the optimizer never saw is handled the legacy way).
    SOC 2 CC8.1 — a words-deleting action is provably a last resort."""
    so = (cue.get("meta") or {}).get("seq_opt")
    if not so:
        return False  # optimizer didn't run on this cue → legacy behavior
    return not bool(so.get("condensation_allowed", False))
