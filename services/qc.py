"""
QC (Quality Control) Report Generator.

Every profile/threshold is sourced from env vars at runtime so the
formatter and QC always see the same rules. Auditor-grade output:
qc_report() returns `_rules_used` so a reviewer can answer "what rules
graded this cue?" from the result alone.
"""

import os
import re
from typing import Dict, List

FUNCTION_WORDS = {
    "a", "an", "the", "of", "to", "and", "or", "but",
    "with", "from", "in", "on", "at", "for", "that",
    "this", "these", "those",
}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _caption_profile() -> str:
    return (os.getenv("CAPTION_PROFILE", "") or "").strip().lower()


def _max_lines() -> int:
    return _env_int("CUSTOM_MAX_LINES", 2)


def _max_chars() -> int:
    return _env_int("CUSTOM_MAX_CHARS", 32)


def _min_dialogue_ms() -> int:
    return _env_int("CUSTOM_MIN_DISPLAY_MS", 800)


def _min_sound_ms() -> int:
    return _env_int("CUSTOM_MIN_SOUND_DISPLAY_MS", 800)


def _max_cps() -> int:
    return _env_int("CUSTOM_MAX_CPS", 45)


def _target_cps() -> int:
    return _env_int("CUSTOM_TARGET_CPS", 27)


def _min_cps() -> int:
    return _env_int("CUSTOM_MIN_CPS", 5)


def _visible_chars(cue: Dict) -> int:
    text = " ".join(cue.get("lines", []))
    return len(text.replace("\n", " ").strip())


def _cue_cps(cue: Dict) -> float:
    dur_ms = max(1, int(cue.get("end_ms", 0)) - int(cue.get("start_ms", 0)))
    return _visible_chars(cue) / (dur_ms / 1000.0)


# Phase 3 — the auto-fix gate. Each rule carries a stable key, a severity, and
# (where deterministic) a suggested repair the editor can one-click apply.
# 'fail' severity is an EXPORT BLOCKER; 'warn' is a suggestion. The engine
# stamps these per-cue so Base44 can gate the export button + offer repairs
# without re-deriving the rules. SOC 2 CC8.1 — the verdict is reproducible.
def _grade_cue(cue: Dict, max_chars: int, max_lines: int,
               min_dialogue_ms: int, min_sound_ms: int,
               max_cps: int, min_cps: int) -> List[Dict]:
    """Return the list of violations for ONE cue. Each: {rule, severity,
    message, actual, limit, fix}. `fix` is a machine-readable repair hint the
    Base44 QC panel maps to a one-click action (or null when no safe automatic
    repair exists and a human must decide)."""
    violations: List[Dict] = []
    lines = cue.get("lines", [])
    is_sound = cue.get("type") in ("sound", "music", "sound_effect")
    duration = int(cue.get("end_ms", 0)) - int(cue.get("start_ms", 0))

    longest = max((len(l) for l in lines), default=0)
    if longest > max_chars:
        violations.append({
            "rule": "max_chars_per_line", "severity": "fail",
            "message": f"Line of {longest} chars exceeds max {max_chars}",
            "actual": longest, "limit": max_chars, "fix": "rewrap_lines",
        })
    if len(lines) > max_lines:
        violations.append({
            "rule": "max_lines_per_caption", "severity": "fail",
            "message": f"{len(lines)} lines exceeds max {max_lines}",
            "actual": len(lines), "limit": max_lines, "fix": "split_cue",
        })
    if not is_sound:
        cps = round(_cue_cps(cue), 2)
        if cps > max_cps:
            violations.append({
                "rule": "max_cps", "severity": "fail",
                "message": f"Reading speed {cps} CPS exceeds max {max_cps}",
                "actual": cps, "limit": max_cps, "fix": "split_or_extend_cue",
            })
        elif cps < min_cps and _visible_chars(cue) > 0:
            violations.append({
                "rule": "min_cps", "severity": "warn",
                "message": f"Reading speed {cps} CPS below min {min_cps} (lingering)",
                "actual": cps, "limit": min_cps, "fix": "trim_cue",
            })
        if is_one_word(lines):
            violations.append({
                "rule": "one_word_cue", "severity": "warn",
                "message": "Single-word dialogue cue",
                "actual": 1, "limit": None, "fix": "merge_neighbor",
            })
        if count_function_word_endings(lines) > 0:
            violations.append({
                "rule": "weak_line_ending", "severity": "warn",
                "message": "Line ends on a function word (weak break)",
                "actual": None, "limit": None, "fix": "rewrap_lines",
            })
    min_dur = min_sound_ms if is_sound else min_dialogue_ms
    if duration < min_dur:
        violations.append({
            "rule": "min_caption_duration_ms", "severity": "warn",
            "message": f"On-screen {duration}ms below min {min_dur}ms",
            "actual": duration, "limit": min_dur, "fix": "extend_cue",
        })
    return violations


def ends_with_function_word(line: str) -> bool:
    line = (line or "").strip().lower()
    if not line:
        return False
    line = re.sub(r"[^\w']+$", "", line)
    parts = line.split()
    if not parts:
        return False
    return parts[-1] in FUNCTION_WORDS


def count_function_word_endings(lines: List[str]) -> int:
    return sum(1 for line in lines if ends_with_function_word(line))


def is_one_word(lines: List[str]) -> bool:
    text = " ".join(lines).strip()
    words = re.findall(r"\b\w+\b", text)
    return len(words) == 1


def count_overlaps(cues: List[Dict]) -> int:
    overlap = 0
    for i in range(len(cues) - 1):
        if cues[i]["end_ms"] > cues[i + 1]["start_ms"]:
            overlap += 1
    return overlap


def count_sound_overlaps(cues: List[Dict]) -> int:
    overlap = 0
    for i in range(len(cues) - 1):
        if cues[i]["end_ms"] > cues[i + 1]["start_ms"]:
            if cues[i]["type"] == "sound" or cues[i + 1]["type"] == "sound":
                overlap += 1
    return overlap


def count_protected_phrase_splits(lines: List[str], protected_phrases: List[str]) -> int:
    joined = "\n".join(lines)
    count = 0
    for phrase in protected_phrases or []:
        if not phrase:
            continue
        if phrase in joined.replace("\n", " ") and phrase.replace(" ", "\n") in joined:
            count += 1
    return count


def qc_report(cues_in: int, cues_out: List[Dict], protected_phrases: List[str]) -> Dict:
    max_lines = _max_lines()
    max_chars = _max_chars()
    min_dialogue_ms = _min_dialogue_ms()
    min_sound_ms = _min_sound_ms()

    max_lines_violation = 0
    max_chars_violation = 0
    short_duration_violations = 0
    one_word_dialogue_cues = 0
    function_word_endings = 0
    protected_phrase_splits = 0
    # DIAGNOSTIC (speaker-fusion fingerprint): count dialogue cues whose
    # meta.runs carry 2+ DISTINCT speakers. A correctly-segmented alpha/named
    # deliverable should have ZERO multi-speaker cues (one speaker per cue);
    # any non-zero count localizes the A/B fusion to the packer. Pure audit
    # signal — never alters output. SOC 2 CC8.1.
    multi_speaker_dialogue_cues = 0

    # Phase 3 — per-cue grading + export-block gate.
    max_cps = _max_cps()
    min_cps = _min_cps()
    cue_grades: List[Dict] = []   # one entry per cue with violations
    blocked_cues: List[Dict] = []  # cues carrying at least one 'fail'
    total_fail = 0
    total_warn = 0

    for idx, cue in enumerate(cues_out):
        is_sound = cue.get("type") in ("sound", "music", "sound_effect")
        if cue.get("type") == "dialogue":
            _runs = (cue.get("meta") or {}).get("runs") or []
            _distinct = {r.get("speaker") for r in _runs if r.get("speaker") is not None}
            if len(_distinct) >= 2:
                multi_speaker_dialogue_cues += 1
        if len(cue["lines"]) > max_lines:
            max_lines_violation += 1
        if any(len(line) > max_chars for line in cue["lines"]):
            max_chars_violation += 1
        duration = cue["end_ms"] - cue["start_ms"]
        if is_sound:
            if duration < min_sound_ms:
                short_duration_violations += 1
        else:
            if duration < min_dialogue_ms:
                short_duration_violations += 1
            if is_one_word(cue["lines"]):
                one_word_dialogue_cues += 1
        function_word_endings += count_function_word_endings(cue["lines"])
        protected_phrase_splits += count_protected_phrase_splits(cue["lines"], protected_phrases)

        violations = _grade_cue(cue, max_chars, max_lines, min_dialogue_ms,
                                min_sound_ms, max_cps, min_cps)
        if violations:
            fails = [v for v in violations if v["severity"] == "fail"]
            warns = [v for v in violations if v["severity"] == "warn"]
            total_fail += len(fails)
            total_warn += len(warns)
            grade = {
                "idx": idx + 1,
                "status": "fail" if fails else "warn",
                "violations": violations,
            }
            cue_grades.append(grade)
            if fails:
                blocked_cues.append({"idx": idx + 1, "violations": fails})

    export_blocked = len(blocked_cues) > 0

    return {
        "cues_in": cues_in,
        "cues_out": len(cues_out),
        "overlaps": count_overlaps(cues_out),
        "max_lines_violation": max_lines_violation,
        "max_chars_violation": max_chars_violation,
        "short_duration_violations": short_duration_violations,
        "one_word_dialogue_cues": one_word_dialogue_cues,
        "sound_overlap_violations": count_sound_overlaps(cues_out),
        "function_word_endings": function_word_endings,
        "protected_phrase_splits": protected_phrase_splits,
        # Fusion fingerprint — see comment above. Surfaced top-level so the
        # Base44 telemetry diagnostic reads it without digging into _rules_used.
        "multi_speaker_dialogue_cues": multi_speaker_dialogue_cues,
        # ── Phase 3 — auto-fix gate ──────────────────────────────────────
        # export_blocked: true when ANY cue carries a 'fail'-severity violation
        # (over max_chars / max_lines / max_cps). Base44 reads this to gate the
        # export button. cue_grades: per-cue {idx,status,violations[]} where each
        # violation has {rule,severity,message,actual,limit,fix}. `fix` is the
        # one-click repair hint the Base44 QC panel maps to an action. blocked_cues
        # is the subset carrying a hard failure. total_fail/total_warn are the
        # headline counts. SOC 2 CC8.1 — the verdict + every repair is reproducible.
        "export_blocked": export_blocked,
        "total_fail": total_fail,
        "total_warn": total_warn,
        "blocked_cue_count": len(blocked_cues),
        "blocked_cues": blocked_cues,
        "cue_grades": cue_grades,
        "_rules_used": {
            # Truthful audit label. The engine is spec-agnostic — every rule
            # below comes from the CUSTOM_* / SPEAKER_LABEL_MODE / MUSIC_CUE_*
            # knobs the Base44 producer derives from the pinned spec. When no
            # CAPTION_PROFILE env var is sent (the correct, spec-driven path),
            # report 'spec_driven' rather than stamping a misleading client
            # name on the deliverable. SOC 2 CC8.1 — the QC record never
            # claims a profile that wasn't actually applied.
            "profile": _caption_profile() or "spec_driven",
            "max_lines": max_lines,
            "max_chars": max_chars,
            "min_dialogue_ms": min_dialogue_ms,
            "min_sound_ms": min_sound_ms,
            "max_cps": max_cps,
            "target_cps": _target_cps(),
            "min_cps": min_cps,
            # The speaker mode the engine ACTUALLY resolved at format time.
            # Confirms whether the Base44 producer's project posture ('alpha')
            # reached the engine, or whether it fell to the 'dash' default
            # (which would explain dash-path grouping running unexpectedly).
            "speaker_label_mode": (os.getenv("SPEAKER_LABEL_MODE", "") or "dash").strip().lower(),
        },
    }
