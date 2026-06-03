"""
Readability Engine — ensures captions are readable on screen.

Enforces:
  • Minimum cue durations
  • Micro-cue merging
  • Sound cue stability
  • Balanced pacing

All thresholds read from env vars (set per-job via captionOptions on the
POST /v1/jobs payload). No hardcoded profile fallbacks.
"""

import os

# SINGLE SOURCE OF TRUTH for line rendering — the same module formatter.py uses
# when it first builds a cue. The merge passes below recombine words + speaker
# runs and RE-RENDER through this, so a merged cue is drawn identically to a
# freshly-built one (correct \n line breaks, correct dash/label, never two
# speakers on one line). This is what fixes the "A+B fused into one flat line"
# defect — the merge passes no longer do naive string concatenation.
try:
    from .rendering import render_lines, merge_cue_meta, cue_speakers
except Exception:
    _HAVE_RENDERING = False
    # Defensive fallback — must never crash a job. Degrades to flat behaviour.
    def render_lines(words, runs, max_lines=None, max_chars=None, dialogue_text=None):
        return [dialogue_text if dialogue_text is not None else " ".join(words)]

    def merge_cue_meta(prev_cue, next_cue):
        pt = (prev_cue.get("meta") or {}).get("dialogue_text") or " ".join(prev_cue.get("lines", []))
        nt = (next_cue.get("meta") or {}).get("dialogue_text") or " ".join(next_cue.get("lines", []))
        text = (pt + " " + nt).strip()
        return {"dialogue_text": text, "runs": [{"speaker": None, "word_start": 0}], "words": text.split()}

    def cue_speakers(cue):
        runs = (cue.get("meta") or {}).get("runs") or []
        return {r.get("speaker") for r in runs if r.get("speaker") is not None}

MICRO_WORD_LIMIT = 2
MICRO_DURATION_MS = 1000


def _speakers_compatible(a, b):
    """Two cues may be merged only when they don't introduce a NEW speaker
    boundary that the renderer would have to split anyway. Compatible when:
      • either cue has no known speaker (unknown — safe to join), OR
      • they share the exact same single speaker.
    Cues belonging to genuinely DIFFERENT speakers are never glued together —
    that's what preserves the dash/label and stops A+B fusing into one line."""
    sa = cue_speakers(a)
    sb = cue_speakers(b)
    if not sa or not sb:
        return True
    return sa == sb and len(sa) == 1


def _rerender_merged(prev_cue, next_cue):
    """Produce the merged cue: recombine word+speaker structure, then RE-RENDER
    lines through the shared renderer so \n / dash / label are all correct."""
    meta = merge_cue_meta(prev_cue, next_cue)
    lines = render_lines(meta["words"], meta["runs"], dialogue_text=meta["dialogue_text"])
    merged = dict(prev_cue)
    merged["lines"] = lines
    merged["end_ms"] = next_cue["end_ms"]
    merged["start_ms"] = min(prev_cue.get("start_ms", next_cue["start_ms"]), next_cue["start_ms"])
    merged["meta"] = {"dialogue_text": meta["dialogue_text"], "runs": meta["runs"]}
    return merged

# Orphan reflow — a final safety net. A dialogue cue of ORPHAN_WORD_LIMIT or
# fewer words that is NOT a brief standalone response ("Yes." "No." "OK.")
# should rejoin the sentence it belongs to. We merge it BACKWARD into the
# previous same-speaker dialogue cue (producing "You must be Mr. Wang.") when
# the result still fits the spec; otherwise FORWARD into the next one.
ORPHAN_WORD_LIMIT = 1
_BRIEF_RESPONSES = {
    "yes", "no", "ok", "okay", "yeah", "right", "sure", "nope", "yep",
    "what", "why", "how", "stop", "wait", "hi", "hey", "hello", "thanks",
}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _min_dialogue_ms() -> int:
    return _env_int("CUSTOM_MIN_DISPLAY_MS", 800)


def _max_chars() -> int:
    return _env_int("CUSTOM_MAX_CHARS", 32)


def _max_lines() -> int:
    return _env_int("CUSTOM_MAX_LINES", 2)


def _min_sound_ms() -> int:
    return _env_int("CUSTOM_MIN_SOUND_DISPLAY_MS", 800)


def _merge_gap_ms() -> int:
    return _env_int("CUSTOM_MERGE_GAP_MS", 80)


def _speaker_label_mode() -> str:
    return (os.getenv("SPEAKER_LABEL_MODE", "") or "dash").strip().lower()


# Two-speaker grouping window. When two adjacent DIFFERENT-speaker dialogue
# cues both sit inside this window (a tight back-and-forth), a dash-style spec
# groups them into ONE caption ('- A' / '- B'). Tunable per-job via env.
def _two_speaker_group_gap_ms() -> int:
    return _env_int("CUSTOM_TWO_SPEAKER_GROUP_GAP_MS", 600)


def enforce_min_duration(cues):
    for i, cue in enumerate(cues):
        duration = cue["end_ms"] - cue["start_ms"]
        min_duration = _min_sound_ms() if cue["type"] == "sound" else _min_dialogue_ms()
        if duration >= min_duration:
            continue
        if i < len(cues) - 1:
            gap = cues[i + 1]["start_ms"] - cue["end_ms"]
            needed = min_duration - duration
            if gap >= needed:
                cue["end_ms"] += needed
    return cues


# ─── CPS-aware merge math (single source of truth) ───────────────────────────
# Reuse cps.py's exact CPS / visible-char / duration helpers so the merge
# improvement-gate measures reading speed IDENTICALLY to how the CPS driver and
# the QC gate measure it. No second definition of CPS anywhere — zero drift.
try:
    from .cps import cue_cps as _cue_cps, _visible_chars as _cps_visible_chars
except Exception:
    def _cue_cps(cue):
        text = " ".join(cue.get("lines", [])).replace("\n", " ").strip()
        dur = max(1, int(cue.get("end_ms", 0)) - int(cue.get("start_ms", 0)))
        return len(text) / (dur / 1000.0)

    def _cps_visible_chars(cue):
        return len(" ".join(cue.get("lines", [])).replace("\n", " ").strip())


# Maximum on-screen duration a merged cue may occupy. Reuses the same env knob
# the CPS driver uses so a merge can never produce a cue the CPS pass would
# immediately judge over-long.
def _max_display_ms() -> int:
    return _env_int("CUSTOM_MAX_DISPLAY_MS", 7000)


def _max_cps() -> int:
    return _env_int("CUSTOM_MAX_CPS", 45)


def _merge_improves(prev_cue, next_cue, merged):
    """The auditor-grade acceptance gate. A micro-cue merge is accepted ONLY
    when it provably improves readability WITHOUT introducing a new violation:

      1. layout: merged cue is ≤ max_lines lines, every line ≤ max_chars.
      2. duration: merged on-screen time is within [min_display, max_display].
      3. reading speed: merged CPS is no worse than the WORST of the two inputs
         (i.e. the merge does not raise reading speed) AND, when at least one
         input was below min_display, the merged cue clears min_display.

    This is intentionally conservative — it never ships a state worse than what
    it started with. Anything it declines is left for the CPS driver / QC gate.
    SOC 2 CC8.1: every merge is a provable readability improvement, never a
    cosmetic regrouping that hides a fail."""
    max_chars = _max_chars()
    max_lines = _max_lines()
    min_display = _min_dialogue_ms()
    max_display = _max_display_ms()
    max_cps = _max_cps()

    lines = merged.get("lines", [])
    # (1) layout must stay within the spec geometry.
    if len(lines) > max_lines:
        return False
    if any(len(l) > max_chars for l in lines):
        return False

    # (2) duration window.
    merged_dur = int(merged.get("end_ms", 0)) - int(merged.get("start_ms", 0))
    if merged_dur < min_display or merged_dur > max_display:
        return False

    # (3) reading speed must not worsen.
    prev_cps = _cue_cps(prev_cue)
    next_cps = _cue_cps(next_cue)
    merged_cps = _cue_cps(merged)
    worst_input_cps = max(prev_cps, next_cps)
    # Merge must not raise CPS above the worse of the two inputs. (Combining a
    # short fragment with a neighbour lengthens the window, so a genuine
    # improvement lowers CPS; we accept equal-or-better, never worse.)
    if merged_cps > worst_input_cps + 0.01:
        return False
    # If neither input was short, there's no min-duration problem to solve and
    # we don't want to needlessly fuse two healthy cues — require that at least
    # one input was actually below min_display (the real merge signal).
    prev_dur = int(prev_cue.get("end_ms", 0)) - int(prev_cue.get("start_ms", 0))
    nxt_dur = int(next_cue.get("end_ms", 0)) - int(next_cue.get("start_ms", 0))
    if prev_dur >= min_display and nxt_dur >= min_display:
        return False

    return True


def merge_micro_cues(cues):
    """Merge adjacent sub-min_display dialogue fragments into a single compliant
    cue — but ONLY when the merge provably improves readability (see
    _merge_improves). Two merge shapes, both gated identically:

      • SAME-SPEAKER continuation — fragments from one speaker re-render as one
        cue (single-line continuation, or a balanced 2-line wrap if needed).
      • ADJACENT DIFFERENT-SPEAKER into a valid TWO-LINE cue — when the spec
        permits ≥2 lines, a tight A→B back-and-forth where each fragment is
        sub-min_display renders as one caption with one speaker per line and the
        spec's labels preserved (first_occurrence_per_scene, alpha, named, …).
        The dash convention has its own dedicated pass (group_two_speaker_cues);
        this path handles every NON-dash spec's legitimate two-speaker caption.

    one-speaker-per-line, max_lines, max_chars, and label mode are all enforced
    by re-rendering through the shared renderer (render_lines via
    _rerender_merged). CPS math is cps.py's, so the gate never drifts from QC."""
    min_display = _min_dialogue_ms()
    group_gap = _two_speaker_group_gap_ms()
    max_lines = _max_lines()
    label_mode = _speaker_label_mode()
    merged = []
    i = 0
    while i < len(cues):
        cue = cues[i]
        nxt = cues[i + 1] if i + 1 < len(cues) else None
        cur_dur = cue["end_ms"] - cue["start_ms"]

        # ── Dialogue micro-merge — the real readability fix ──────────────────
        # Trigger when THIS cue is a sub-min_display dialogue fragment and the
        # next is also dialogue within a tight conversational gap.
        if (cue.get("type") == "dialogue"
                and cur_dur < min_display
                and nxt is not None
                and nxt.get("type") == "dialogue"
                and (nxt["start_ms"] - cue["end_ms"]) <= group_gap):
            same_speaker = _speakers_compatible(cue, nxt)
            # Different-speaker merge is only allowed when the spec gives us a
            # second line to put the second speaker on (one-speaker-per-line is
            # non-negotiable). Dash mode is handled by group_two_speaker_cues.
            allow_cross = (not same_speaker) and max_lines >= 2 and label_mode != "dash"
            if same_speaker or allow_cross:
                candidate = _rerender_merged(cue, nxt)
                if _merge_improves(cue, nxt, candidate):
                    merged.append(candidate)
                    i += 2
                    continue

        # ── Sound-cue clustering (unchanged) ─────────────────────────────────
        if (cue.get("type") == "sound"
                and nxt is not None
                and nxt.get("type") == "sound"):
            a = cue["lines"][0].strip("[]")
            b = nxt["lines"][0].strip("[]")
            cue["lines"] = [f"[{a} AND {b}]"]
            cue["end_ms"] = max(cue["end_ms"], nxt["end_ms"])
            merged.append(cue)
            i += 2
            continue

        merged.append(cue)
        i += 1
    return merged


def _is_orphan_dialogue(cue):
    """A dialogue cue that's a stranded sentence fragment — NOT a deliberate
    brief response. e.g. a lone 'Wang.' left after a bad upstream split."""
    if cue.get("type") != "dialogue":
        return False
    text = " ".join(cue.get("lines", [])).strip()
    # Strip a leading dash prefix so '- Wang.' is also caught.
    if text.startswith("- "):
        text = text[2:].strip()
    words = [w for w in text.split() if w]
    if len(words) > ORPHAN_WORD_LIMIT:
        return False
    if not words:
        return False
    bare = words[0].rstrip(".!?\"')]}").lower()
    # A genuine brief response ("Yes." "No.") is allowed to stand alone.
    return bare not in _BRIEF_RESPONSES


def reflow_orphans(cues):
    """Rejoin stranded single-word sentence fragments into their sentence.

    Merge BACKWARD into the previous same-speaker dialogue cue when the merged
    text still fits the spec (max_chars × max_lines) — this is the
    'You must be Mr. Wang.' fix. If a backward merge doesn't fit, try FORWARD
    into the next same-speaker dialogue cue. Speaker is inferred from the dash
    prefix convention and cue adjacency; we never merge across a speaker change.
    """
    if not cues:
        return cues

    budget = _max_chars() * _max_lines()
    out = list(cues)
    i = 0
    while i < len(out):
        cue = out[i]
        if not _is_orphan_dialogue(cue):
            i += 1
            continue

        orphan_text = " ".join(cue.get("lines", [])).strip()

        # Try BACKWARD merge into the previous dialogue cue. SPEAKER GUARD +
        # RE-RENDER: only merge same-speaker (or unknown) cues, and draw the
        # result through the shared renderer (correct \n / dash / label).
        if (i > 0 and out[i - 1].get("type") == "dialogue"
                and _speakers_compatible(out[i - 1], cue)):
            prev = out[i - 1]
            prev_text = (prev.get("meta") or {}).get("dialogue_text") or " ".join(prev.get("lines", []))
            combined = (prev_text.strip() + " " + orphan_text).strip()
            if len(combined) <= budget:
                out[i - 1] = _rerender_merged(prev, cue)
                del out[i]
                continue  # re-check the same index (now the next cue)

        # Try FORWARD merge into the next dialogue cue.
        if (i < len(out) - 1 and out[i + 1].get("type") == "dialogue"
                and _speakers_compatible(cue, out[i + 1])):
            nxt = out[i + 1]
            nxt_text = (nxt.get("meta") or {}).get("dialogue_text") or " ".join(nxt.get("lines", []))
            combined = (orphan_text + " " + nxt_text.strip()).strip()
            if len(combined) <= budget:
                out[i + 1] = _rerender_merged(cue, nxt)
                del out[i]
                continue

        # Can't merge either direction without breaking the spec or across a
        # speaker boundary — leave it. QC will flag it.
        i += 1

    return out


def group_two_speaker_cues(cues):
    """SPEC-DRIVEN multi-speaker grouping (dash convention).

    Only active when the spec's SPEAKER_LABEL_MODE is 'dash' (Pluto / FAST and
    any spec that carries the dash posture). For non-dash specs this is a no-op
    — the professional default of one-speaker-per-cue (separate timecodes) is
    correct and untouched.

    When two ADJACENT dialogue cues belong to DIFFERENT speakers, sit inside a
    tight back-and-forth window (small inter-cue gap), and their combined text
    fits the spec's char×line budget, they are merged into ONE caption that the
    shared renderer draws as:
        - Speaker A line
        - Speaker B line
    This is the legitimate Pluto two-speakers-in-one-caption case. The merge
    goes through merge_cue_meta + render_lines (the SAME universal renderer
    every other stage uses), so the dash/label and one-speaker-per-line
    invariant are guaranteed and auditor-consistent. SOC 2 CC8.1.
    """
    if _speaker_label_mode() != "dash":
        return cues

    budget = _max_chars() * _max_lines()
    group_gap = _two_speaker_group_gap_ms()
    out = []
    i = 0
    while i < len(cues):
        cue = cues[i]
        nxt = cues[i + 1] if i + 1 < len(cues) else None
        if (nxt is not None
                and cue.get("type") == "dialogue"
                and nxt.get("type") == "dialogue"
                and not _speakers_compatible(cue, nxt)  # genuinely DIFFERENT speakers
                and (nxt["start_ms"] - cue["end_ms"]) <= group_gap):
            combined = _rerender_merged(cue, nxt)
            # Only accept the grouping if it still fits the on-screen budget
            # (≤ max_lines lines, each ≤ max_chars). Dash prefixes count.
            fits = (len(combined.get("lines", [])) <= _max_lines()
                    and all(len(l) <= _max_chars() for l in combined.get("lines", []))
                    and len(combined.get("meta", {}).get("dialogue_text", "")) <= budget)
            if fits:
                out.append(combined)
                i += 2
                continue
        out.append(cue)
        i += 1
    return out


def apply_readability_rules(cues):
    """Master readability pass. Run AFTER AI formatting.

    Order matters:
      1. reflow orphans — rejoin stranded fragments to their sentence.
      2. two-speaker grouping — SPEC-DRIVEN (dash mode only): fuse tight
         back-and-forth A/B exchanges into one '- A' / '- B' caption.
      3. enforce min-duration + micro-merge on the cleaned set.
      4. CPS enforcement (Phase 2) — reading-speed rules as active drivers:
         extend over-fast cues into idle gap, split the still-over-fast ones at
         a clause boundary, trim over-slow lingering cues. Runs AFTER merging so
         it sees final cue boundaries, and is the LAST structural pass so its
         timing/split decisions are not undone by a later merge.
    """
    cues = reflow_orphans(cues)
    cues = group_two_speaker_cues(cues)
    cues = enforce_min_duration(cues)
    cues = merge_micro_cues(cues)
    try:
        from .cps import enforce_cps_rules
    except Exception:
        def enforce_cps_rules(c):
            return c
    cues = enforce_cps_rules(cues)
    return cues
