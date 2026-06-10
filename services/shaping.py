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
    from .rendering import render_lines as _render_lines
except Exception:  # pragma: no cover
    def _render_lines(words, runs, max_lines=None, max_chars=None, dialogue_text=None):
        return [dialogue_text if dialogue_text is not None else " ".join(words)]

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


# ─── Core: split one over-rhythm cue into two ────────────────────────
def _split_cue_once(cue: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Split a single dialogue cue into two cues at the best phrase boundary
    nearest the middle. Returns [left, right] or None when no clean split is
    possible (no interior boundary, or the halves would violate min_display).
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
        bounds = _latin_phrase_boundaries(words)
        widx = _pick_balanced_boundary(bounds, len(words))
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
        return None  # no room to split cleanly — leave intact

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


def _needs_shaping(cue: Dict[str, Any], target_ms: int) -> bool:
    """A dialogue cue needs rhythm-shaping when it runs noticeably longer than
    the target reading rhythm. We use 1.5×target as the trigger so cues that are
    only slightly over the target are left alone (splitting a 3.2s cue into two
    1.6s cues adds churn for no readability gain); a 6s cue (2×target) clearly
    reads better as two ~3s cues."""
    if cue.get("type") != "dialogue":
        return False
    return _cue_duration_ms(cue) > int(target_ms * 1.5)


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
