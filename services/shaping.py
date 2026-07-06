"""
Caption Shaping — the universal, spec-driven "caption rhythm" stage.

WHAT THIS IS
────────────
The professional layer that turns raw-transcript-shaped cues (one cue per
utterance/sentence) into BROADCAST-RHYTHM cues (1–3s, clause-broken, reading
naturally) BEFORE the CPS/readability/QC passes run. This is the stage a
captioner does by hand and the stage EZTitles / WinCAPS / Swift do
automatically — and the one our engine was missing.

WHY A SEPARATE UNIVERSAL STAGE (not a Japanese patch)
─────────────────────────────────────────────────────
The raw transcript is the MASTER SOURCE. The selected CC spec drives the
transformation into compliant output. Every language benefits from the same
core workflow:

    raw cues
      → SHAPE to reading rhythm   (this module — split toward a TARGET
                                    duration, at clause/phrase boundaries)
      → editorial AI polish
      → readability (orphan reflow, micro-cue merge, two-speaker dash)
      → CPS enforcement (extend / split / trim — hard limits)
      → QC grade
      → export

The CPS stage already splits cues that exceed the HARD max duration
(max_caption_duration_ms, e.g. 7000ms) or overflow the char budget. But a 6.2s
single-sentence cue that fits the char budget and is under 7s sails through
unsplit — and reads as a wall of text. Professional rhythm targets ~2–3s per
cue. THIS stage closes that gap: it splits toward a spec-driven TARGET duration
(below the hard max), at natural phrase boundaries, so each resulting cue can
then be wrapped within the spec's CPL/line budget and graded clean.

SCRIPT-AWARENESS, NOT SCRIPT-BRANCHING
──────────────────────────────────────
The pipeline logic here is identical for every language. The ONLY place script
matters is HOW a single over-rhythm cue is cut into phrases — and that decision
is delegated to the existing, audited helpers:
  • CJK  → clause/sentence enders (、。！？) via services.cjk, kinsoku-aware.
  • Latin→ clause boundaries (, ; : — –) by word, orphan-guarded.
Both already exist (cps.py / formatter.py / cjk.py). Adding a new script later
means adding one phrase-splitter helper, NOT touching this pipeline.

TIMING PROVENANCE (enterprise-correct)
──────────────────────────────────────
When the cue carries real word-level timings (meta.word_timings — the
AAI/Scribe per-word start/end), the split uses the ACTUAL timestamp at the
phrase boundary, so the result is frame-faithful to the audio. Only when word
timings are absent do we interpolate proportionally to character/word count
over the cue's [start_ms, end_ms]. SOC 2 CC8.1 — every shaped cue's timing is
provably either measured (word-timed) or interpolated (fallback), recorded in
meta.shaping.

Pure functions only — no env writes, no I/O beyond reading the spec env knobs
through services.formatter helpers. Deterministic: identical (cues, spec)
always yields identical shaped cues.
"""

import os
from typing import Any, Dict, List, Optional

try:
    from .rendering import render_lines as _render_lines, cue_fits_delivered as _cue_fits_delivered
except Exception:  # pragma: no cover
    def _render_lines(words, runs, max_lines=None, max_chars=None, dialogue_text=None):
        return [dialogue_text if dialogue_text is not None else " ".join(words)]

    def _cue_fits_delivered(cue, max_lines=None, max_chars=None):
        return True

try:
    from .cjk import (
        is_cjk_text as _is_cjk,
        cjk_char_count as _cjk_count,
        CJK_CLAUSE_ENDERS,
        CJK_SENTENCE_ENDERS,
    )
except Exception:  # pragma: no cover
    def _is_cjk(text):
        return False

    def _cjk_count(text):
        return len((text or "").replace(" ", ""))

    CJK_CLAUSE_ENDERS = ("、", "，", "；", "：", "・", ",", ";", ":")
    CJK_SENTENCE_ENDERS = ("。", "！", "？", "．", "!", "?", ".")

# Phrase-boundary word classes — the SAME linguistic tables the line-breaker uses,
# so a punctuation-free cue is split at a real phrase point (before a preposition/
# conjunction/article, never stranding a function word at a cue end) rather than a
# blind midpoint. Shared import keeps segmentation identical to line-break logic.
try:
    from .linebreak import _LEADING_WORDS, _DETERMINERS, _bare as _bare_word
except Exception:  # pragma: no cover
    _LEADING_WORDS = frozenset()
    _DETERMINERS = frozenset()

    def _bare_word(w):
        return (w or "").strip(".,;:!?—–\"')]}").lower()


# ─── Spec knobs ──────────────────────────────────────────────────────
def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _shaping_enabled() -> bool:
    """Master switch. Default ON — caption shaping is the professional baseline.
    A spec can disable it (CUSTOM_SHAPING_ENABLED=0) to ship raw-utterance cues
    verbatim (e.g. a 1:1 import that must not be re-timed)."""
    return os.getenv("CUSTOM_SHAPING_ENABLED", "1") not in ("0", "false", "False")


def _condensation_mode() -> str:
    """CONDENSATION_MODE gate for the condense-to-fit-before-split attempt.
    'off' → verbatim spec, NEVER trim to avoid a split (go straight to split).
    Any other value permits the deterministic disfluency trim that can let a
    marginally-overflowing sentence stay ONE cue instead of being split."""
    return (os.getenv("CONDENSATION_MODE", "disfluency_only") or "disfluency_only").strip().lower()


def _max_chars() -> int:
    return _env_int("CUSTOM_MAX_CHARS", 32)


def _max_lines() -> int:
    return _env_int("CUSTOM_MAX_LINES", 2)


def _min_display_ms() -> int:
    return _env_int("CUSTOM_MIN_DISPLAY_MS", 800)


def _max_display_ms() -> int:
    return _env_int("CUSTOM_MAX_DISPLAY_MS", 7000)


def _merge_gap_ms() -> int:
    return _env_int("CUSTOM_MERGE_GAP_MS", 80)


def _target_duration_ms() -> int:
    """The READING-RHYTHM target — the per-cue duration the shaper aims for.
    Distinct from max_caption_duration_ms (the hard ceiling). A cue noticeably
    longer than this is split at a phrase boundary even though it's under the
    hard max. Spec-driven via CUSTOM_TARGET_DURATION_MS; default 3000ms, the
    broadcast-SDH norm (most professional cues land 1–3s). Floored at
    2×min_display so the shaper never targets a duration it can't split into two
    legal halves."""
    target = _env_int("CUSTOM_TARGET_DURATION_MS", 3000)
    floor = 2 * _min_display_ms()
    return max(target, floor)


# ─── Helpers ─────────────────────────────────────────────────────────
_LATIN_CLAUSE_END = (",", ";", ":", "—", "–")
_LATIN_SENTENCE_END = (".", "!", "?")


def _cue_duration_ms(cue: Dict[str, Any]) -> int:
    return max(1, int(cue.get("end_ms", 0)) - int(cue.get("start_ms", 0)))


def _cue_text(cue: Dict[str, Any]) -> str:
    meta = cue.get("meta") or {}
    return (meta.get("dialogue_text") or " ".join(cue.get("lines", []))).strip()


def _cue_words(cue: Dict[str, Any]) -> List[str]:
    txt = _cue_text(cue)
    if not txt:
        return []
    return txt.split()


def _primary_speaker(cue: Dict[str, Any]) -> Optional[str]:
    runs = (cue.get("meta") or {}).get("runs") or []
    return runs[0].get("speaker") if runs else None


def _word_timings(cue: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Real per-word timings carried on the cue, if any. Shape:
    [{text, start_ms, end_ms}, ...]. Empty list when absent (fallback path)."""
    wt = (cue.get("meta") or {}).get("word_timings") or []
    out = []
    for w in wt:
        try:
            out.append({
                "text": str(w.get("text", "")),
                "start_ms": int(w.get("start_ms", w.get("start", 0)) or 0),
                "end_ms": int(w.get("end_ms", w.get("end", 0)) or 0),
            })
        except Exception:
            continue
    return out


# ─── Phrase boundary selection (script-aware, single decision point) ──
def _latin_phrase_boundaries(words: List[str]) -> List[int]:
    """Word indices AFTER which a phrase boundary exists (clause/sentence
    punctuation). Returned as split indices (1..len-1). Sentence enders rank
    above clause enders, but both are valid rhythm breakpoints."""
    bounds = []
    for i in range(len(words) - 1):
        w = words[i].rstrip()
        if w.endswith(_LATIN_SENTENCE_END) or w.endswith(_LATIN_CLAUSE_END):
            bounds.append(i + 1)
    return bounds


def _cjk_phrase_boundaries(s: str) -> List[int]:
    """Character indices AFTER which a CJK phrase boundary exists (、。！？等).
    Returned as split indices (1..len-1)."""
    bounds = []
    for i in range(len(s) - 1):
        if s[i] in CJK_CLAUSE_ENDERS or s[i] in CJK_SENTENCE_ENDERS:
            bounds.append(i + 1)
    return bounds


def _pick_balanced_boundary(bounds: List[int], n: int) -> Optional[int]:
    """From the available phrase boundaries, pick the one nearest the MIDDLE so
    the two halves are balanced in length (each closest to the target rhythm).
    None when no usable interior boundary exists."""
    if not bounds:
        return None
    mid = n / 2.0
    best = min(bounds, key=lambda b: abs(b - mid))
    if 0 < best < n:
        return best
    return None


def _clause_boundary_ok(words: List[str], idx: int) -> bool:
    """Guard a Latin CLAUSE/SENTENCE boundary against the SAME orphaned-function-
    word defect the word-fallback path already prevents. A boundary at word index
    `idx` (line 1 = words[:idx], line 2 = words[idx:]) is REJECTED when the tail
    half LEADS with a stranded preposition/conjunction/article/determiner — e.g.
    '...the regeneration' / 'of glutathione' orphans 'of'. Because clause enders
    sit AFTER a word, the punctuation itself already guarantees the head doesn't
    end on a bare function word; the remaining risk is purely the tail's leading
    word. Returns True when the split reads cleanly."""
    if idx <= 0 or idx >= len(words):
        return False
    first_bare = _bare_word(words[idx])
    if first_bare in _LEADING_WORDS or first_bare in _DETERMINERS:
        return False
    return True


def _pick_clause_boundary(words: List[str]) -> Optional[int]:
    """Pick the best Latin clause/sentence boundary — nearest the middle for
    balance — that ALSO passes the orphaned-function-word guard. This routes the
    clause path through the same word-class intelligence the word-fallback path
    uses, so a punctuated cue never strands 'of'/'the'/'and' on the tail cue.
    Returns a split index in 1..len-1, or None when no clean clause boundary
    exists (caller falls back to the word-phrase splitter)."""
    bounds = _latin_phrase_boundaries(words)
    if not bounds:
        return None
    clean = [b for b in bounds if _clause_boundary_ok(words, b)]
    return _pick_balanced_boundary(clean, len(words))


def _pick_word_phrase_boundary(words: List[str]) -> Optional[int]:
    """FALLBACK when a cue has no clause/sentence punctuation to split at (the
    "no interior boundary" cases like "your head will be adorned in crimson and
    gold at the"). Pick the word split index nearest the middle that produces a
    GOOD phrase break — one where the second half LEADS with a preposition/
    conjunction/article/determiner (so the break lands before a natural phrase
    start) and the first half does NOT end on a stranded function word. Same
    word-class intelligence the line-breaker uses, applied to CUE splitting so a
    punctuation-free line still divides at a real phrase point, never a blind
    midpoint. Returns a split index in 1..len-1, or None if too short (≤1 word)."""
    n = len(words)
    if n < 2:
        return None
    mid = n / 2.0
    scored: List[tuple] = []
    for i in range(1, n):
        last_bare = _bare_word(words[i - 1])
        first_bare = _bare_word(words[i])
        score = 0.0
        if first_bare in _LEADING_WORDS or first_bare in _DETERMINERS:
            score += 10.0
        if last_bare in _LEADING_WORDS or last_bare in _DETERMINERS:
            score -= 12.0
        score -= abs(i - mid) * 1.0
        scored.append((score, i))
    if not scored:
        return None
    scored.sort(key=lambda t: (-t[0], t[1]))
    return scored[0][1]


# ─── Timing at a split point ─────────────────────────────────────────
def _split_time_at(
    cue: Dict[str, Any],
    left_text: str,
    full_text: str,
    is_cjk: bool,
) -> int:
    """Resolve the boundary timestamp between the left and right halves.

    PREFERS real word timings: find the word boundary closest to where the left
    text ends and use that word's end_ms — frame-faithful to the audio. Falls
    back to character/word-proportional interpolation over the cue window when
    no word timings are present. SOC 2 CC8.1 — provenance recorded by caller."""
    start = int(cue.get("start_ms", 0))
    end = int(cue.get("end_ms", start + 1))
    span = max(2, end - start)

    wt = _word_timings(cue)
    if wt:
        # Find the cumulative-text position the left half ends at, then the
        # word whose running text best matches that boundary.
        target_chars = _cjk_count(left_text) if is_cjk else len(left_text)
        running = 0
        chosen_end = None
        for w in wt:
            piece = w["text"]
            running += _cjk_count(piece) if is_cjk else (len(piece) + 1)
            if running >= target_chars:
                chosen_end = w["end_ms"]
                break
        if chosen_end is not None and start < chosen_end < end:
            return int(chosen_end)

    # Fallback: proportional interpolation by visible character count.
    left_len = _cjk_count(left_text) if is_cjk else len(left_text)
    full_len = _cjk_count(full_text) if is_cjk else len(full_text)
    if full_len <= 0:
        return start + span // 2
    return start + int(span * (left_len / full_len))


# ─── Re-balance a split so neither child is a too-short, over-CPS cue ──
def _child_windows_readable(
    cue: Dict[str, Any],
    left_text: str,
    right_text: str,
    full_text: str,
    is_cjk: bool,
    min_display: int,
    min_gap: int,
) -> bool:
    """True when splitting `cue` at the boundary between left_text/right_text
    yields TWO children that each clear the min reading window (min_display).

    This is the RE-BALANCE guard: the naive "boundary nearest the middle by word
    count" can hand a short-text half a tiny audio window (e.g. 'It's riboflavin,
    honey.' → 0.9s → 25.9 cps FAIL) when speech at that boundary is fast. By
    resolving the REAL boundary timestamp and checking both child durations, we
    reject a split that would create an unreadable sliver and let the caller try
    the next-best boundary instead — so text and window scale together. SOC 2
    CC8.1 — the split decision is provably duration-aware, not blind midpoint."""
    start = int(cue.get("start_ms", 0))
    end = int(cue.get("end_ms", start + 1))
    cut = _split_time_at(cue, left_text, full_text, is_cjk)
    left_end = min(cut - (min_gap // 2), end - min_display)
    right_start = max(cut + (min_gap // 2), start + min_display)
    return (left_end - start) >= min_display and (end - right_start) >= min_display


def _pick_rebalanced_latin_boundary(
    cue: Dict[str, Any],
    words: List[str],
    is_cjk: bool,
    min_display: int,
    min_gap: int,
) -> Optional[int]:
    """Choose the Latin split index that (a) reads cleanly (clause-guarded, no
    orphaned function word) AND (b) gives BOTH children a readable window. Tries
    clause/sentence boundaries first — ranked by balance — keeping only those
    where both halves clear min_display; then the word-phrase fallback under the
    same window check. Returns the best index, or None to signal 'no split that
    both reads well and times legally' (caller leaves the cue intact rather than
    minting a tiny fail-cue)."""
    full_text = " ".join(words)

    # Candidate clause boundaries, cleanest-and-most-balanced first.
    clause = [b for b in _latin_phrase_boundaries(words) if _clause_boundary_ok(words, b)]
    mid = len(words) / 2.0
    clause.sort(key=lambda b: abs(b - mid))
    for idx in clause:
        if _child_windows_readable(
            cue, " ".join(words[:idx]), " ".join(words[idx:]),
            full_text, is_cjk, min_display, min_gap,
        ):
            return idx

    # GRAMMAR OUTRANKS THE WINDOW HEURISTIC (2026-07-06). When clean clause
    # boundaries EXIST but none passes the readable-window preference, split at
    # the most balanced clause boundary anyway rather than falling through to a
    # word split. The window check is a preference, not a law: a clause split
    # with a slightly tight window is professionally correct, while a mid-
    # phrase word split ('…25 laps around' | 'this place,') breaks the
    # universal phrase-integrity rule AND still fails CPS. The caller's own
    # min-window recheck still vetoes the split in rhythm posture (cue stays
    # whole); the CPL safety net (enforce_cpl_fit) then force-splits at this
    # same clause boundary when geometry demands it. Downstream stages (CPS
    # extend, sliver absorb, condensation, QC) own the timing remedy.
    if clause:
        return clause[0]

    # Word-phrase fallback — ONLY for punctuation-free cues, window-checked.
    widx = _pick_word_phrase_boundary(words)
    if widx is not None and _child_windows_readable(
        cue, " ".join(words[:widx]), " ".join(words[widx:]),
        full_text, is_cjk, min_display, min_gap,
    ):
        return widx
    return None


# ─── Core: split one over-rhythm cue into two ────────────────────────
def _split_cue_once(
    cue: Dict[str, Any],
    require_readable_windows: bool = True,
) -> Optional[List[Dict[str, Any]]]:
    """Split a single dialogue cue into two cues at the best phrase boundary.
    Returns [left, right] or None when no clean split is possible.

    require_readable_windows (default True — the RHYTHM-shaping posture): the
    Latin split must give BOTH children a readable window (min_display), so a
    split is only taken when it improves rhythm without minting a tiny over-CPS
    sliver. When no such split exists the cue is left intact (a long-but-legal
    cue beats a fail-cue).

    require_readable_windows=False (the CPL SAFETY-NET posture, used by
    enforce_cpl_fit): a cue that overflows the hard per-line char cap MUST be
    broken — CPL is an FCC hard limit that outranks the rhythm-window preference.
    The clause-orphan guard still applies (breaking well never hurts fit), but
    the min-window balance check is relaxed so an over-CPL line is always split.

    Script-aware ONLY in which boundary list is used; everything else is shared."""
    text = _cue_text(cue)
    if not text:
        return None
    is_cjk = _is_cjk(text)
    speaker = _primary_speaker(cue)
    max_chars = _max_chars()
    max_lines = _max_lines()
    min_display = _min_display_ms()
    min_gap = _merge_gap_ms()
    start = int(cue.get("start_ms", 0))
    end = int(cue.get("end_ms", start + 1))

    # 1) Pick the phrase boundary nearest the middle (balanced halves).
    if is_cjk:
        n = len(text)
        bounds = _cjk_phrase_boundaries(text)
        idx = _pick_balanced_boundary(bounds, n)
        if idx is None:
            return None
        left_text = text[:idx].strip()
        right_text = text[idx:].strip()
    else:
        words = _cue_words(cue)
        if len(words) < 2:
            return None
        # Re-balanced, clause-guarded boundary: prefers a clause/sentence break
        # that (a) never orphans a leading function word onto the tail cue and
        # (b) gives BOTH children a readable window (no tiny over-CPS sliver like
        # 'It's riboflavin, honey.' @ 0.9s). Falls back to the word-phrase
        # splitter under the same guards. None → no split both reads and times
        # cleanly, so leave the cue intact rather than mint a fail-cue.
        widx = _pick_rebalanced_latin_boundary(cue, words, is_cjk, min_display, min_gap)
        if widx is None:
            if require_readable_windows:
                return None
            # CPL SAFETY-NET posture: an over-CPL line MUST break even if neither
            # child clears the rhythm window. Use the clause-guarded boundary
            # (still no orphaned function word), then the word-phrase fallback,
            # WITHOUT the min-window balance check. CPL is a hard FCC limit.
            widx = _pick_clause_boundary(words)
            if widx is None:
                widx = _pick_word_phrase_boundary(words)
            if widx is None:
                return None
        left_words = words[:widx]
        right_words = words[widx:]
        left_text = " ".join(left_words)
        right_text = " ".join(right_words)

    if not left_text or not right_text:
        return None

    # 2) Resolve the boundary timestamp (real word timing preferred).
    cut = _split_time_at(cue, left_text, text, is_cjk)
    left_end = min(cut - (min_gap // 2), end - min_display)
    right_start = max(cut + (min_gap // 2), start + min_display)
    if left_end - start < min_display or end - right_start < min_display:
        # No room for two readable windows. In the RHYTHM posture, leave intact
        # (a long-but-legal cue beats a fail-cue). In the CPL SAFETY-NET posture,
        # an over-CPL line must still break — clamp the boundary to the cue
        # window and split anyway (each child gets whatever window remains).
        if require_readable_windows:
            return None
        cut = max(start + 1, min(cut, end - 1))
        left_end = cut
        right_start = cut

    runs = [{"speaker": speaker, "word_start": 0}]
    timing_source = "word_timings" if _word_timings(cue) else "interpolated"

    def _mk(txt: str, s_ms: int, e_ms: int) -> Dict[str, Any]:
        words = txt.split()
        lines = _render_lines(words, runs, max_lines, max_chars, txt)
        return {
            "idx": 0,
            "start_ms": int(s_ms),
            "end_ms": int(e_ms),
            "lines": lines,
            "type": "dialogue",
            "meta": {
                "dialogue_text": txt,
                "runs": runs,
                "shaping": {"split": True, "timing_source": timing_source},
                # Carry the word timings forward so a recursive second split on
                # either half is also frame-faithful.
                "word_timings": _word_timings(cue),
            },
        }

    return [_mk(left_text, start, left_end), _mk(right_text, right_start, end)]


def _wrap_overflows_cpl(cue: Dict[str, Any]) -> bool:
    """True when the cue, RENDERED AS DELIVERED (speaker label included), does NOT
    fit the spec geometry — its best labeled wrap still leaves a line over the CPL
    cap or exceeds max_lines. Delegates to the ONE shared fit primitive
    (rendering.cue_fits_delivered) so the shaper's "does this fit?" question is
    byte-identical to the formatter's, the readability merge's, and the exported
    file's. Catches both the body-only overflow (33ch line) and the LABEL-INDUCED
    overflow (54ch body that wraps clean body-only, but '[SPEAKER B:]' pushes the
    render to 38ch). SOC 2 CC8.1 / FCC §79.1."""
    return not _cue_fits_delivered(cue, _max_lines(), _max_chars())


def _needs_shaping(cue: Dict[str, Any], target_ms: int) -> bool:
    """A dialogue cue needs rhythm-shaping when EITHER:
      • it runs noticeably longer than the target reading rhythm (1.5×target, so
        a slightly-over cue is left alone — splitting a 3.2s cue into two 1.6s
        cues adds churn for no readability gain; a 6s cue clearly reads better as
        two ~3s cues), OR
      • its best 2-line wrap still overflows the spec's max_chars_per_line — a
        genuine CPL violation the whole-cue budget check upstream missed. In this
        case duration is irrelevant; the line simply doesn't fit and must be
        split at a phrase boundary. SOC 2 CC8.1 / FCC §79.1 — the delivered line
        never silently exceeds the spec's per-line cap."""
    if cue.get("type") != "dialogue":
        return False
    return _cue_duration_ms(cue) > int(target_ms * 1.5) or _wrap_overflows_cpl(cue)


# ─── Condense-to-fit-before-split (C when needed) ────────────────────
def _try_condense_to_fit(
    cue: Dict[str, Any],
    target_ms: int,
) -> Optional[Dict[str, Any]]:
    """Attempt to keep a marginally-overflowing sentence as ONE cue by trimming
    trivial disfluency, instead of splitting it across two cues.

    Only fires when the cue needs shaping PURELY for CPL fit — i.e. it is NOT
    genuinely too long to read (duration ≤ 1.5×target) but its best delivered
    wrap overflows the per-line cap. A cue that is also over-rhythm (too long)
    genuinely needs the split; we don't paper over that with a trim.

    Spec-gated: CONDENSATION_MODE='off' (verbatim spec) → return None (no trim;
    caller splits). Otherwise run the DETERMINISTIC disfluency remover (no AI —
    the reproducible layer) and, if the trimmed text now fits delivered, return
    a single condensed cue with meta.condensation provenance so the Base44
    ingester records the original verbatim + one-click revert. Returns None when
    a trim can't achieve fit (caller falls through to the split)."""
    # Verbatim spec → never trim to avoid a split.
    if _condensation_mode() == "off":
        return None
    # Only for cues that overflow CPL but are NOT over-rhythm (those must split).
    if _cue_duration_ms(cue) > int(target_ms * 1.5):
        return None
    if not _wrap_overflows_cpl(cue):
        return None

    try:
        from .condensation import remove_disfluencies
    except Exception:
        return None

    verbatim = _cue_text(cue)
    if not verbatim:
        return None
    trimmed = remove_disfluencies(verbatim)
    if not trimmed or trimmed == verbatim:
        return None  # nothing to trim → can't rescue → split

    speaker = _primary_speaker(cue)
    runs = [{"speaker": speaker, "word_start": 0}]
    max_chars = _max_chars()
    max_lines = _max_lines()
    label, body = "", trimmed
    # Preserve a leading speaker label if the cue text carried one.
    m = None
    try:
        import re as _re
        m = _re.match(r"^\s*(?:-\s+|\[[^\]]*\]:?\s*|[^\s:]{1,24}:\s+)", trimmed)
    except Exception:
        m = None
    if m:
        label, body = trimmed[:m.end()], trimmed[m.end():]

    candidate = {
        "idx": cue.get("idx", 0),
        "start_ms": int(cue.get("start_ms", 0)),
        "end_ms": int(cue.get("end_ms", 0)),
        "lines": _render_lines(body.split(), runs, max_lines, max_chars, body),
        "type": "dialogue",
        "meta": {
            "dialogue_text": trimmed,
            "runs": runs,
            "word_timings": _word_timings(cue),
            "condensation": {
                "applied": True,
                "kind": "disfluency",
                "verbatim": verbatim,
            },
        },
    }
    # Only accept the trim if it ACTUALLY fits delivered now (else split reads better).
    if _wrap_overflows_cpl(candidate):
        return None
    return candidate


# ─── Public entry ────────────────────────────────────────────────────
def shape_caption_rhythm(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Universal caption-rhythm pass. Iteratively splits over-rhythm dialogue
    cues at phrase boundaries toward the spec's target duration, until every
    dialogue cue is within rhythm or can't be split further. Sound cues are
    never touched. Spec-driven, script-aware via delegated helpers, real word
    timings preferred over interpolation. Re-indexes on exit.

    Disabled (CUSTOM_SHAPING_ENABLED=0) → returns cues unchanged (raw-utterance
    posture for 1:1 imports)."""
    if not _shaping_enabled() or not cues:
        for i, c in enumerate(cues):
            c["idx"] = i + 1
        return cues

    target_ms = _target_duration_ms()

    # Iterate to a fixed point. Each pass splits every over-rhythm cue once;
    # a long utterance becomes 2 → 4 → … cues until each is within rhythm.
    # Bounded so a pathological cue (no interior boundary) can never loop.
    for _ in range(8):
        out: List[Dict[str, Any]] = []
        changed = False
        for cue in cues:
            if not _needs_shaping(cue, target_ms):
                out.append(cue)
                continue
            # CONDENSE-TO-FIT-BEFORE-SPLIT ("A always, C when needed"). If this
            # cue needs shaping ONLY because its wrap marginally overflows CPL
            # (not because it's genuinely too long to read), first try a
            # deterministic disfluency trim to make the WHOLE sentence fit one
            # cue — the professional captioner's move. Keeps a single readable
            # cue instead of minting a split fragment + fail-cue. Spec-gated on
            # CONDENSATION_MODE (verbatim specs skip straight to the split). If
            # the trim achieves fit, keep the one condensed cue; otherwise fall
            # through to the split. SOC 2 CC8.1 — the trim is attributed via
            # meta.condensation, exactly like the condensation stage.
            fitted = _try_condense_to_fit(cue, target_ms)
            if fitted is not None:
                out.append(fitted)
                changed = True
                continue
            split = _split_cue_once(cue)
            if split:
                out.extend(split)
                changed = True
            else:
                out.append(cue)
        cues = out
        if not changed:
            break

    for i, cue in enumerate(cues):
        cue["idx"] = i + 1
    return cues


def enforce_cpl_fit(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """FINAL delivered-fit guarantee, run at the very END of the readability
    pipeline (after CPS extend/split/merge). Splits ANY dialogue cue that still
    doesn't fit DELIVERED (speaker label included) at a phrase boundary — clause
    first, word-phrase fallback — until every cue fits or can't be split further.
    Pure CPL: the duration trigger is ignored here (a cue can be perfectly-timed
    yet still overflow its line once labeled). This is the safety net that
    guarantees no over-label line survives any later stage that re-wrapped or
    re-merged after the initial shaping pass. Idempotent — a cue that already
    fits is never touched. Bounded fixed-point loop. SOC 2 CC8.1 / FCC §79.1."""
    if not cues:
        return cues
    for _ in range(8):
        out: List[Dict[str, Any]] = []
        changed = False
        for cue in cues:
            if cue.get("type") != "dialogue" or not _wrap_overflows_cpl(cue):
                out.append(cue)
                continue
            # CPL is a hard FCC limit — force the break even if neither child
            # clears the rhythm window (require_readable_windows=False).
            split = _split_cue_once(cue, require_readable_windows=False)
            if split:
                out.extend(split)
                changed = True
            else:
                out.append(cue)
        cues = out
        if not changed:
            break
    for i, cue in enumerate(cues):
        cue["idx"] = i + 1
    return cues
