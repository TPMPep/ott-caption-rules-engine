"""
Rendering — the SINGLE SOURCE OF TRUTH for turning a cue's words + speaker
structure into on-screen lines.

WHY THIS MODULE EXISTS
──────────────────────
Before this module, line-drawing lived inside formatter.py (`_speaker_render_lines`,
`_segments_from_runs`, `_wrap_text`). The readability cleanup passes
(merge_micro_cues, reflow_orphans) could NOT reach it, so when they glued two
cues together they did naive string concatenation:

    cue["lines"] = [prev_text + " " + orphan_text]   # ← one flat line, no dash

That single shortcut caused two production defects on the same cue:
  • SPEAKER A and SPEAKER B got fused into one caption with NO dash / label
    (the cleanup step never checked who was speaking).
  • The line break (\n) was destroyed — the editor showed one long line even
    though the spec said two.

The professional fix (Iyuno / Pixelogic / Zoo / Deluxe grade) is to have ONE
function that every stage calls to render lines. Formatter calls it when it
first builds a cue; the readability merge passes call it again after they
recombine words. They can never disagree because there is only one of them.

CONTRACT — every cue carries enough structure to be re-rendered losslessly:
    cue["meta"]["dialogue_text"]  : the full spoken text (no dash/label markup)
    cue["meta"]["runs"]           : [{speaker, word_start}, ...] speaker offsets
The renderer reconstructs per-speaker segments from `runs` + the word list and
applies the UNIVERSAL one-speaker-per-line invariant plus the spec's
SPEAKER_LABEL_MODE (dash / none / named / etc.).

Pure functions only — no env writes, no I/O. Deterministic. SOC 2 CC8.1:
identical (words, runs, spec) always renders identical lines, on every stage.
"""

import os
from typing import Any, Dict, List, Optional


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _max_lines() -> int:
    return _env_int("CUSTOM_MAX_LINES", 2)


def _max_chars() -> int:
    return _env_int("CUSTOM_MAX_CHARS", 32)


def _speaker_label_mode() -> str:
    return (os.getenv("SPEAKER_LABEL_MODE", "") or "dash").strip().lower()


def segments_from_runs(words: List[str], speaker_runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split the flat word list into per-speaker text segments using each
    run's `word_start` offset. Returns [{speaker, text}, ...] in order."""
    if not speaker_runs:
        return [{"speaker": None, "text": " ".join(words)}]
    out: List[Dict[str, Any]] = []
    for i, run in enumerate(speaker_runs):
        start = run.get("word_start", 0)
        end = speaker_runs[i + 1].get("word_start", len(words)) if i + 1 < len(speaker_runs) else len(words)
        seg_words = words[start:end]
        if seg_words:
            out.append({"speaker": run.get("speaker"), "text": " ".join(seg_words)})
    return out


def wrap_text(text: str, max_chars: int, max_lines: int) -> List[str]:
    """
    Wrap text into lines respecting max_chars per line.
    Greedy fill; prefers natural word boundaries.
    """
    text = (text or "").strip()
    if not text:
        return [""]

    if len(text) <= max_chars:
        return [text]

    words = text.split()
    lines: List[str] = []
    current_line = ""

    for word in words:
        test = (current_line + " " + word).strip() if current_line else word

        if len(test) <= max_chars:
            current_line = test
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

            if len(lines) >= max_lines - 1:
                # Last allowed line — dump the remaining words onto it.
                remaining = " ".join(words[words.index(word):])
                lines.append(remaining)
                return lines[:max_lines]

    if current_line:
        lines.append(current_line)

    return lines[:max_lines]


def render_lines(
    words: List[str],
    speaker_runs: List[Dict[str, Any]],
    max_lines: Optional[int] = None,
    max_chars: Optional[int] = None,
    dialogue_text: Optional[str] = None,
) -> List[str]:
    """
    UNIVERSAL speaker-aware line builder, parameterised by the spec's
    SPEAKER_LABEL_MODE. The engine never names a client — it only obeys the
    mode the spec sent.

    Hard universal invariant (every mode, never configurable):
      • One speaker per line. Two distinct speakers are NEVER placed on the
        same physical line. This is the Pluto "- Speaker One / - Speaker Two"
        rule and is correct for every spec.

    Modes:
      • 'dash' (Pluto/FAST): multi-speaker group → each speaker's text on its
        own line, prefixed with '- '. Single-speaker group → plain wrap, no dash.
      • 'none': plain wrap, no speaker identifier.
      • anything else ('first_occurrence_per_scene' / 'every_change' /
        'always' / 'named' / 'alpha' / 'generic'): single-speaker groups wrap
        plainly here (named-tag insertion is the editorial-AI / downstream
        concern); multi-speaker groups still get one-speaker-per-line so reading
        order is unambiguous.
    """
    max_lines = max_lines if max_lines is not None else _max_lines()
    max_chars = max_chars if max_chars is not None else _max_chars()
    dialogue_text = dialogue_text if dialogue_text is not None else " ".join(words)

    mode = _speaker_label_mode()
    segments = segments_from_runs(words, speaker_runs)
    distinct_speakers = len({
        s.get("speaker") for s in (speaker_runs or []) if s.get("speaker") is not None
    })

    # Single speaker (or no diarization) → plain greedy wrap. No dash, no split.
    if distinct_speakers <= 1 or len(segments) <= 1:
        return wrap_text(dialogue_text, max_chars, max_lines)

    # Multi-speaker group — one speaker per line (universal invariant).
    prefix = "- " if mode == "dash" else ""
    lines: List[str] = []
    for seg in segments:
        seg_text = seg["text"].strip()
        if not seg_text:
            continue
        lines.append(f"{prefix}{seg_text}")

    if not lines:
        return wrap_text(dialogue_text, max_chars, max_lines)
    return lines


def merge_cue_meta(prev_cue: Dict[str, Any], next_cue: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine the word + speaker structure of two ADJACENT cues into a single
    meta block, preserving per-speaker run offsets so the merged cue can be
    re-rendered losslessly by render_lines.

    Returns { "dialogue_text": str, "runs": [{speaker, word_start}, ...],
              "words": [str, ...] }. The caller decides whether the merge is
    allowed (speaker compatibility, budget) BEFORE calling this.
    """
    prev_meta = prev_cue.get("meta") or {}
    next_meta = next_cue.get("meta") or {}

    prev_text = (prev_meta.get("dialogue_text") or " ".join(prev_cue.get("lines", []))).strip()
    next_text = (next_meta.get("dialogue_text") or " ".join(next_cue.get("lines", []))).strip()

    prev_words = prev_text.split()
    next_words = next_text.split()
    combined_words = prev_words + next_words

    prev_runs = prev_meta.get("runs") or [{"speaker": None, "word_start": 0}]
    next_runs = next_meta.get("runs") or [{"speaker": None, "word_start": 0}]

    # Re-base the next cue's run offsets by the prev cue's word count, then
    # collapse a redundant boundary run if the two adjoining speakers match
    # (so "same speaker" merges stay a single run = single line).
    offset = len(prev_words)
    combined_runs = list(prev_runs)
    for run in next_runs:
        rebased = {"speaker": run.get("speaker"), "word_start": run.get("word_start", 0) + offset}
        if combined_runs and combined_runs[-1].get("speaker") == rebased["speaker"]:
            # Same speaker continues — no new run boundary.
            continue
        combined_runs.append(rebased)

    return {
        "dialogue_text": (prev_text + " " + next_text).strip(),
        "runs": combined_runs,
        "words": combined_words,
    }


def cue_speakers(cue: Dict[str, Any]) -> set:
    """Return the set of distinct speaker ids present in a cue's meta.runs.
    Empty set when the cue has no speaker structure (treated as 'unknown')."""
    runs = (cue.get("meta") or {}).get("runs") or []
    return {r.get("speaker") for r in runs if r.get("speaker") is not None}
