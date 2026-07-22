"""Deterministic frame-grid timing policy for final caption sequences."""
from fractions import Fraction
from math import ceil
from typing import Any, Dict, List

from .rules import get_rule
from .canonical_hash import canonical_sha256

_COMMON_RATES = {
    "23.976": Fraction(24000, 1001),
    "23.98": Fraction(24000, 1001),
    "29.97": Fraction(30000, 1001),
    "59.94": Fraction(60000, 1001),
}


def frame_rate() -> Fraction:
    raw = str(get_rule("CUSTOM_FRAME_RATE", get_rule("SCC_FRAME_RATE", "25")) or "25").strip()
    if raw in _COMMON_RATES:
        return _COMMON_RATES[raw]
    try:
        value = Fraction(raw).limit_denominator(1001)
        return value if value > 0 else Fraction(25, 1)
    except Exception:
        return Fraction(25, 1)


def ms_to_frame(ms: int, mode: str = "nearest") -> int:
    value = Fraction(int(ms), 1000) * frame_rate()
    if mode == "floor":
        return value.numerator // value.denominator
    if mode == "ceil":
        return ceil(value)
    return int(value + Fraction(1, 2))


def frame_to_ms(frame: int) -> int:
    value = Fraction(int(frame) * 1000, 1) / frame_rate()
    return int(value + Fraction(1, 2))


def minimum_gap_frames() -> int:
    raw = get_rule("CUSTOM_MIN_GAP_FRAMES")
    if raw not in (None, ""):
        try:
            return max(0, int(raw))
        except Exception:
            pass
    legacy_ms = int(get_rule("CUSTOM_MERGE_GAP_MS", "80") or 80)
    return max(0, int(Fraction(legacy_ms, 1000) * frame_rate() + Fraction(1, 2)))


def minimum_gap_ms() -> int:
    return frame_to_ms(minimum_gap_frames())


def minimum_duration_frames() -> int:
    minimum_ms = int(get_rule("CUSTOM_MIN_DISPLAY_MS", "800") or 800)
    return max(1, ceil(Fraction(minimum_ms, 1000) * frame_rate()))


def minimum_duration_ms_on_grid() -> int:
    return frame_to_ms(minimum_duration_frames())


def is_on_grid(ms: int) -> bool:
    return frame_to_ms(ms_to_frame(ms)) == int(ms)


def normalize_cue_timing(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Project the complete delivered sequence onto one legal frame lattice.

    The projection is a deterministic least-squares isotonic solve: every cue
    keeps its original duration unless a dialogue cue needs the contractual
    minimum; starts are then moved by the smallest aggregate amount that can
    satisfy ordering, non-overlap, and frame-gap constraints. Dense rapid-dialogue
    groups borrow capacity from the nearest surrounding pauses instead of leaving
    an illegal child for final QC. No text or cue-specific rule is involved.
    """
    if not cues:
        return cues
    gap_f = minimum_gap_frames()
    min_f = minimum_duration_frames()

    original = []
    durations = []
    desired_starts = []
    max_cps = max(1, int(get_rule("CUSTOM_MAX_CPS", "17") or 17))
    max_duration_f = max(1, ms_to_frame(int(get_rule("CUSTOM_MAX_DISPLAY_MS", "7000") or 7000), "floor"))
    measurement = str(get_rule("CPS_MEASUREMENT", "characters") or "characters").strip().lower()
    for cue in cues:
        start_f = ms_to_frame(int(cue.get("start_ms", 0)), "nearest")
        end_f = ms_to_frame(int(cue.get("end_ms", 0)), "nearest")
        raw_duration = max(1, end_f - start_f)
        duration = raw_duration
        if cue.get("type", "dialogue") == "dialogue":
            delivered = " ".join(str(line) for line in (cue.get("lines") or [])).strip()
            if measurement == "characters_no_spaces":
                units = len("".join(delivered.split()))
            elif measurement == "words":
                units = len(delivered.split())
            else:
                units = len(delivered)
            cps_required_f = ceil(Fraction(units, max_cps) * frame_rate()) if units else min_f
            duration = max(raw_duration, min_f, min(cps_required_f, max_duration_f))
        original.append((start_f, end_f))
        durations.append(duration)
        desired_starts.append(Fraction(start_f + end_f - duration, 2))

    offsets = [0]
    for i in range(1, len(cues)):
        both_dialogue = (cues[i - 1].get("type", "dialogue") == "dialogue"
                         and cues[i].get("type", "dialogue") == "dialogue")
        offsets.append(offsets[-1] + durations[i - 1] + (gap_f if both_dialogue else 0))

    # Pool-adjacent-violators over desired_start - cumulative_constraints.
    blocks = []
    for i, desired in enumerate(desired_starts):
        blocks.append({"start": i, "end": i + 1, "sum": desired - offsets[i], "weight": 1})
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left["sum"] / left["weight"] <= right["sum"] / right["weight"]:
                break
            blocks[-2:] = [{
                "start": left["start"], "end": right["end"],
                "sum": left["sum"] + right["sum"],
                "weight": left["weight"] + right["weight"],
            }]

    projected_base = [0] * len(cues)
    for block in blocks:
        mean = block["sum"] / block["weight"]
        rounded = int(mean + Fraction(1, 2)) if mean >= 0 else -int(-mean + Fraction(1, 2))
        for i in range(block["start"], block["end"]):
            projected_base[i] = rounded
    starts = [projected_base[i] + offsets[i] for i in range(len(cues))]
    if starts[0] < 0:
        shift = -starts[0]
        starts = [s + shift for s in starts]

    # ── MEASURED-START FLOOR — the timecode-integrity invariant ──────────────
    # A caption may be nudged LATER (to resolve an overlap / honor the min gap),
    # but it may NEVER start earlier than the moment its words are actually
    # spoken. Without this, the isotonic solve treats a long inter-cue silence
    # (e.g. a 14s gap between a music cue ending and the next line being spoken)
    # as slack to squeeze out, and drags the dialogue cue seconds earlier onto
    # the tail of the prior cue — the caption then fires long before the audio.
    # The measured on-grid start is the source of truth; here it becomes a HARD
    # LOWER BOUND, applied in one forward pass so ordering + non-overlap are
    # preserved (each clamped start also pushes its successors' minimum). This
    # preserves the legitimate rapid-dialogue behavior (which only ever moves
    # starts LATER to fit min-duration) while making a real silence uncollapsible.
    # Deterministic; SOC 2 CC8.1 / FCC 47 CFR §79.1 — delivered timecodes never
    # precede the measured audio.
    running_min = None
    for i in range(len(cues)):
        # Round the measured source start UP to the frame grid. Using the
        # nearest frame could still place a caption up to half a frame before
        # the spoken word, which violates the no-early-caption invariant.
        measured_start_f = ms_to_frame(int(cues[i].get("start_ms", 0)), "ceil")
        floored = max(starts[i], measured_start_f)
        if running_min is not None:
            floored = max(floored, running_min)
        starts[i] = floored
        both_dialogue = (
            i + 1 < len(cues)
            and cues[i].get("type", "dialogue") == "dialogue"
            and cues[i + 1].get("type", "dialogue") == "dialogue"
        )
        running_min = starts[i] + durations[i] + (gap_f if both_dialogue else 0)

    decision_groups: Dict[str, List[Dict[str, Any]]] = {}
    for i, cue in enumerate(cues):
        start_f, end_f = starts[i], starts[i] + durations[i]
        old_start_ms = int(cue.get("start_ms", 0))
        old_end_ms = int(cue.get("end_ms", 0))
        cue["start_ms"] = frame_to_ms(start_f)
        cue["end_ms"] = frame_to_ms(end_f)
        changed = cue["start_ms"] != old_start_ms or cue["end_ms"] != old_end_ms
        meta = dict(cue.get("meta") or {})
        meta["frame_grid"] = {
            "projection_version": 2,
            "frame_rate_num": frame_rate().numerator,
            "frame_rate_den": frame_rate().denominator,
            "minimum_gap_frames": gap_f,
            "minimum_duration_frames": min_f,
            "original_start_ms": old_start_ms,
            "original_end_ms": old_end_ms,
            "projected_start_shift_ms": cue["start_ms"] - old_start_ms,
            "projected_end_shift_ms": cue["end_ms"] - old_end_ms,
        }
        seq_opt = dict(meta.get("seq_opt") or {})
        if changed and seq_opt:
            detail = dict(seq_opt.get("timing_provenance_detail") or {})
            detail["extended_boundary_count"] = int(detail.get("extended_boundary_count", 0) or 0) + 1
            seq_opt["timing_provenance"] = "extended"
            seq_opt["timing_provenance_detail"] = detail
            meta["seq_opt"] = seq_opt
            group_key = str(seq_opt.get("decision_key_unbound") or seq_opt.get("input_hash") or "")
            if group_key:
                decision_groups.setdefault(group_key, []).append(cue)
        cue["meta"] = meta

    # Projection changes timing but not composition. Rebind each affected
    # optimizer decision's output hash to the actual delivered boundaries.
    for group in decision_groups.values():
        output_hash = canonical_sha256({
            "kind": "seg_output",
            "parts": [[int(c["start_ms"]), int(c["end_ms"]),
                       (c.get("meta") or {}).get("dialogue_text")
                       or " ".join(c.get("lines", []))] for c in group],
        })
        for cue in group:
            cue["meta"]["seq_opt"]["output_hash"] = output_hash
    return cues
    gap_f = minimum_gap_frames()
    min_f = minimum_duration_frames()

    frames = []
    for cue in cues:
        start_f = ms_to_frame(int(cue.get("start_ms", 0)), "nearest")
        end_f = ms_to_frame(int(cue.get("end_ms", 0)), "nearest")
        if end_f <= start_f:
            end_f = start_f + 1
        frames.append([start_f, end_f])

    for i in range(len(frames) - 1):
        cur, nxt = frames[i], frames[i + 1]
        needed_start = cur[1] + gap_f
        if nxt[0] >= needed_start:
            continue
        # Prefer moving the next start later when its legal duration survives.
        if nxt[1] - needed_start >= min_f:
            nxt[0] = needed_start
            continue
        # Otherwise pull the prior end earlier when its legal duration survives.
        legal_prev_end = nxt[0] - gap_f
        if legal_prev_end - cur[0] >= min_f:
            cur[1] = legal_prev_end

    for cue, (start_f, end_f) in zip(cues, frames):
        cue["start_ms"] = frame_to_ms(start_f)
        cue["end_ms"] = frame_to_ms(end_f)
        meta = dict(cue.get("meta") or {})
        meta["frame_grid"] = {
            "frame_rate_num": frame_rate().numerator,
            "frame_rate_den": frame_rate().denominator,
            "minimum_gap_frames": gap_f,
            "minimum_duration_frames": min_f,
        }
        cue["meta"] = meta
    return cues
