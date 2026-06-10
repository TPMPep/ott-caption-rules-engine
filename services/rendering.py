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

try:
    from services.linebreak import choose_two_line_break
except ImportError:  # pragma: no cover — defensive for alternate import roots
    from linebreak import choose_two_line_break

# CJK awareness — route no-space-script text to the character-aware wrapper.
try:
    from services.cjk import is_cjk_text as _is_cjk, wrap_cjk as _wrap_cjk, cjk_char_count as _cjk_count
except ImportError:  # pragma: no cover
    try:
        from cjk import is_cjk_text as _is_cjk, wrap_cjk as _wrap_cjk, cjk_char_count as _cjk_count
    except ImportError:
        def _is_cjk(text):
            return False

        def _wrap_cjk(text, max_chars, max_lines):
            return [text]

        def _cjk_count(text):
            return len((text or "").replace(" ", ""))


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


def _label_case() -> str:
    return (os.getenv("SPEAKER_LABEL_CASE", "uppercase") or "uppercase").strip().lower()


def _format_speaker_name(raw: Optional[str], mode: str) -> str:
    """Turn a raw diarization speaker id (e.g. 'A', 'SPEAKER_01', 'John') into
    the display name the spec's mode wants:
      • alpha   → 'SPEAKER A'  (letter from the id, A/B/C…)
      • generic → 'SPEAKER 1'  (1-indexed number from the id)
      • named / first_occurrence / every_change / always → the real name as-is
    Case is applied per SPEAKER_LABEL_CASE (broadcast default uppercase)."""
    s = (raw or "").strip()
    if not s:
        return ""

    if mode == "alpha":
        # Pull a single A–Z letter. AssemblyAI/Scribe ids are usually already
        # 'A','B','C' or 'SPEAKER_A'; fall back to the first alpha char.
        letter = next((ch for ch in s.upper() if ch.isalpha()), "A")
        name = f"SPEAKER {letter}"
    elif mode == "generic":
        digits = "".join(ch for ch in s if ch.isdigit())
        if digits:
            num = int(digits) + (0 if digits != "0" else 1)
        else:
            # Map a letter id to a 1-based number (A→1, B→2…).
            first = next((ch for ch in s.upper() if ch.isalpha()), "A")
            num = ord(first) - ord("A") + 1
        name = f"SPEAKER {num}"
    else:
        # named / first_occurrence_per_scene / every_change / always
        name = s

    case = _label_case()
    if case == "uppercase":
        return name.upper()
    if case == "title":
        return name.title()
    return name


def _label_format(off_camera: bool = False) -> str:
    """Tag template from the spec. {name} is substituted. Off-camera speakers
    may use a distinct template (Pluto: '({name}):')."""
    if off_camera:
        oc = (os.getenv("OFF_CAMERA_LABEL_FORMAT", "") or "").strip()
        if oc:
            return oc
    return (os.getenv("SPEAKER_LABEL_FORMAT", "[{name}:]") or "[{name}:]").strip()


def _render_speaker_tag(raw_speaker: Optional[str], mode: str) -> str:
    """Build the full speaker tag string (e.g. '[SPEAKER A:]') for a cue's
    speaker, per the spec's mode + format template. Returns '' when the mode
    emits no tag ('none') or there is no speaker."""
    if mode in ("none", "dash"):
        return ""  # 'dash' is handled separately (prefix, not a name tag)
    name = _format_speaker_name(raw_speaker, mode)
    if not name:
        return ""
    template = _label_format()
    return template.replace("{name}", name)


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


def _greedy_wrap(words: List[str], max_chars: int, max_lines: int) -> List[str]:
    """Greedy left-to-right fill. The deterministic fallback used for the
    1-line case, the 3+-line case, and whenever the syntax-aware 2-line breaker
    finds no feasible balanced split. Word-faithful (enumerate-based slice)."""
    lines: List[str] = []
    current_line = ""
    for idx, word in enumerate(words):
        test = (current_line + " " + word).strip() if current_line else word
        if len(test) <= max_chars:
            current_line = test
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
            if len(lines) >= max_lines - 1:
                # Last allowed line — dump remaining words from the CURRENT
                # index (never words.index(word), which finds the first match).
                lines.append(" ".join(words[idx:]))
                return lines[:max_lines]
    if current_line:
        lines.append(current_line)
    return lines[:max_lines]


def wrap_text(text: str, max_chars: int, max_lines: int) -> List[str]:
    """
    Wrap text into lines respecting max_chars per line.

    For the common 2-line case the SYNTAX-AWARE breaker (services.linebreak)
    chooses the break that preserves clause/phrase integrity AND balances the
    two line lengths — the professional-captioner behaviour (honors the spec's
    PREFER_BALANCED_LINES + PRESERVE_CLAUSE_INTEGRITY flags). For 1-line, 3+
    lines, or when no feasible balanced split exists, falls back to greedy fill.
    Both paths are deterministic and word-faithful. SOC 2 CC8.1.
    """
    text = (text or "").strip()
    if not text:
        return [""]

    # CJK (no-space script): wrap by CHARACTER count at 、。 boundaries with
    # kinsoku, never by space-split words. This is the core fix for the Japanese
    # 54-char-line / mega-cue defect — wrap_cjk produces ≤max_chars lines.
    if _is_cjk(text):
        if _cjk_count(text) <= max_chars:
            return [text]
        return _wrap_cjk(text, max_chars, max_lines)

    if len(text) <= max_chars:
        return [text]

    words = text.split()

    # ── Syntax-aware balanced break — only the exact-2-line case ──────────
    # When max_lines >= 2 and the whole string fits across two lines, prefer the
    # clause-aware balanced split over greedy. choose_two_line_break returns None
    # if no feasible two-line split exists (e.g. content needs 3 lines), in which
    # case we fall through to greedy below.
    if max_lines >= 2:
        two = choose_two_line_break(text, max_chars)
        if two is not None:
            return two

    return _greedy_wrap(words, max_chars, max_lines)


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

    Two universal hard invariants (every mode, never configurable):
      • One speaker per line. Two distinct speakers are NEVER placed on the
        same physical line.
      • A speaker label is emitted at the start of each speaker's text, in the
        style the spec's mode dictates. This is the FCC/industry baseline
        ("signal a new speaker with a hyphen or a clear label") — true for
        every spec, not just Pluto. The DETERMINISTIC renderer owns this so
        labels are present even when the editorial-AI is skipped / times out /
        hits its run budget under 100-user load. SOC 2 CC8.1 / FCC §79.1 —
        speaker identification is never silently lost.

    Styles (the ONLY per-spec variation — the glyph, never whether a label
    appears):
      • 'dash'    : prefix each speaker's line with '- '  (Pluto/FAST).
      • 'alpha'   : '[SPEAKER A:]' bracket tag.
      • 'generic' : '[SPEAKER 1:]' bracket tag.
      • 'named' / 'first_occurrence_per_scene' / 'every_change' / 'always'
                  : '[NAME:]' bracket tag using the real diarized name.
      • 'none'    : no label (plain wrap) — the only mode that omits a label.

    NOTE on "every speaker change": render_lines is per-cue and the packer
    guarantees one speaker per dialogue cue (the speaker-integrity invariant),
    so labelling the cue's speaker == labelling on every change. A genuine
    multi-speaker cue (intentional dash grouping) labels each segment.
    """
    max_lines = max_lines if max_lines is not None else _max_lines()
    max_chars = max_chars if max_chars is not None else _max_chars()
    dialogue_text = dialogue_text if dialogue_text is not None else " ".join(words)

    mode = _speaker_label_mode()
    segments = segments_from_runs(words, speaker_runs)
    distinct_speakers = len({
        s.get("speaker") for s in (speaker_runs or []) if s.get("speaker") is not None
    })

    # The cue's single speaker (for single-speaker cues, which is most cues).
    primary_speaker = None
    for run in (speaker_runs or []):
        if run.get("speaker") is not None:
            primary_speaker = run.get("speaker")
            break

    # ── Single speaker (the common case) ─────────────────────────────────
    if distinct_speakers <= 1 or len(segments) <= 1:
        if mode == "none" or primary_speaker is None:
            return wrap_text(dialogue_text, max_chars, max_lines)
        if mode == "dash":
            # Dash mode does NOT prefix a single-speaker cue (the dash only
            # disambiguates a multi-speaker exchange). Plain wrap.
            return wrap_text(dialogue_text, max_chars, max_lines)
        # Bracket-style modes (alpha / generic / named / first_occurrence /
        # every_change / always): prepend the tag to the wrapped text so the
        # label rides on the first line. The tag counts toward the char budget
        # of line 1 — wrap the tagged text as a whole so QC stays honest.
        tag = _render_speaker_tag(primary_speaker, mode)
        if not tag:
            return wrap_text(dialogue_text, max_chars, max_lines)
        return wrap_text(f"{tag} {dialogue_text}".strip(), max_chars, max_lines)

    # ── Multi-speaker cue — one speaker per line + per-speaker label ──────
    lines: List[str] = []
    for seg in segments:
        seg_text = seg["text"].strip()
        if not seg_text:
            continue
        if mode == "dash":
            lines.append(f"- {seg_text}")
        elif mode == "none":
            lines.append(seg_text)
        else:
            tag = _render_speaker_tag(seg.get("speaker"), mode)
            lines.append(f"{tag} {seg_text}".strip() if tag else seg_text)

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


# ─── first_occurrence_per_scene label suppression ───────────────────────────
# render_lines is intentionally STATELESS per cue (single source of truth, called
# by the formatter AND the readability merge passes). It therefore cannot decide
# "is this the speaker's FIRST cue in the scene?" — so on its own it renders a
# bracket label on EVERY cue (correct for 'always' / 'every_change', wrong for
# 'first_occurrence_per_scene', which is what the bug was: the catch-all branch
# silently degraded first_occurrence to always).
#
# The spec-faithful fix is a SINGLE stateful post-pass over the finished cue list.
# It runs ONCE, after every cue is built/refined/readability-processed, walks the
# cues in display order, and for label_mode='first_occurrence_per_scene' strips the
# bracket/paren label from any cue whose speaker has already been labeled earlier
# in the scene. MVP scene definition (per directive): the ENTIRE output is one
# scene unless explicit scene boundaries are provided — so each speaker is labeled
# exactly once, on their first cue.
#
# Why this can never break layout: removing a leading label only ever SHORTENS a
# line, never lengthens it, so no cue can newly overflow max_chars. We re-wrap the
# stripped first line defensively anyway (free, deterministic) so a label that had
# pushed text onto a second line collapses back to one when it now fits.
#
# Modes other than first_occurrence_per_scene are untouched (dash / alpha / generic
# / named / every_camera / always / none all keep their current behaviour). SOC 2
# CC8.1 / FCC §79.1 — speaker identification still appears (once), and the pass is
# a deterministic, reproducible function of (cues, mode, scene boundaries).

# Match a leading bracket tag '[NAME:] ' or paren tag '(NAME): ' the renderer
# emitted. Dash ('- ') is NOT stripped here — dash mode is a separate label_mode
# and is never first_occurrence_per_scene.
import re as _re_for_labels
# CRITICAL: the inner class must be LAZY ([^\]]*?). The rendered label is
# '[SPEAKER A:]' — the ':' is itself a non-']' char, so a GREEDY [^\]]* eats
# 'SPEAKER A:' and leaves only ']', making the required ':\]' suffix unmatchable.
# A greedy class therefore NEVER matches a real label (the silent no-op bug).
# Lazy expansion stops at the first ':]' and matches correctly.
_BRACKET_LABEL_RE = _re_for_labels.compile(r"^\s*\[[^\]]*?:\]\s*")
_PAREN_LABEL_RE = _re_for_labels.compile(r"^\s*\([^)]*?\):\s*")


def _strip_leading_bracket_label(line: str) -> str:
    s = _BRACKET_LABEL_RE.sub("", line or "", count=1)
    s = _PAREN_LABEL_RE.sub("", s, count=1)
    return s


def _line_has_bracket_label(line: str) -> bool:
    return _strip_leading_bracket_label(line) != (line or "")


def suppress_repeat_speaker_labels(
    cues: List[Dict[str, Any]],
    scene_boundary_idxs: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Enforce label_mode='first_occurrence_per_scene' across a finished cue list.

    For each scene, the FIRST cue of each speaker keeps its rendered label; every
    later cue of that same speaker in the same scene has its label stripped. No-op
    for any other label_mode. Pure, deterministic, in-place-safe (returns the same
    list mutated). `scene_boundary_idxs` is a set of cue indices that START a new
    scene; when None/empty the entire output is treated as ONE scene (MVP).
    """
    mode = _speaker_label_mode()
    if mode != "first_occurrence_per_scene":
        return cues

    max_lines = _max_lines()
    max_chars = _max_chars()
    boundaries = scene_boundary_idxs or set()
    seen_speakers: set = set()

    for i, cue in enumerate(cues):
        if i in boundaries:
            seen_speakers = set()  # new scene → labels re-appear
        if cue.get("type") != "dialogue":
            continue
        speaker = None
        for run in ((cue.get("meta") or {}).get("runs") or []):
            if run.get("speaker") is not None:
                speaker = run.get("speaker")
                break
        # A cue with no resolvable speaker carries no first-occurrence identity;
        # leave it exactly as rendered.
        if speaker is None:
            continue

        if speaker in seen_speakers:
            # Repeat occurrence — strip the label off this cue's lines, then
            # re-wrap the now-shorter spoken text so a label that had forced a
            # 2nd line collapses cleanly. Shortening can never overflow.
            lines = cue.get("lines", [])
            if any(_line_has_bracket_label(l) for l in lines):
                spoken = " ".join(_strip_leading_bracket_label(l) for l in lines).strip()
                cue["lines"] = wrap_text(spoken, max_chars, max_lines)
        else:
            seen_speakers.add(speaker)

    return cues
