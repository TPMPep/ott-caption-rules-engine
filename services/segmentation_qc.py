"""
Segmentation QC — CANONICAL, PRODUCTION-AUTHORITY deterministic inspection stage.

This module is the PRODUCTION authority for Segmentation QC. The JavaScript
reference implementation (src/lib/cc-segmentation-qc.js) mirrors this contract
for the editor's live badges + Base44 parity tests; THIS module is what actually
runs inside the Railway formatter pipeline and produces the audited verdict.

It is a DISTINCT inspection stage that runs AFTER the rules engine has produced
the final rendered cue sequence — shaping done, sequence optimizer done,
readability + CPS remediation done, condensation done, speaker labels emitted,
line rendering final — and BEFORE general QC (services.qc) and export
serialization. It answers a different question than its neighbours:

  • services.cps / services.readability → deterministic CPS/duration remedies on
    the raw cue timings.
  • services.qc → general technical limits (CPS / CPL / duration / one-word) with
    a fail/warn gate.
  • segmentation_qc (THIS module) → "is the DELIVERED sequence — speaker tags,
    dashes, protected phrases, reading rhythm and all — actually shippable, and
    if a hard defect remains, has a bounded deterministic remedy been attempted,
    and what is the honest disposition (review-required / export-blocked)?"

Design invariants (hard constraints — auditor posture):
  1. TAG-AWARE — every cue is measured AS DELIVERED (speaker label prefix +
     dialogue), so a label that tips an otherwise-fine cue over CPS/CPL is caught.
  2. ISSUE-CODED — every finding is a stable QC_ISSUE code with a declared
     QC_SEVERITY, a bounded deterministic evidence blob, the remediation
     operations attempted, and the remediation result.
  3. BOUNDED + FINITE remediation — one retry maximum per cue window, a
     deterministic window-state hash to detect no-progress, attempted-operation
     tracking (no repeated op / no repeated state), NO AI, NO meaningful-word
     deletion, and optimizer-authored boundaries are never reversed.
  4. TECHNICAL VIOLATIONS STAY VISIBLE — a review-required cue never erases its
     underlying technical violation. For an unresolved speaker-label case both
     `technical_violations` (max_cps / max_cpl) AND review_required/export_blocked
     are reported.
  5. EXPORT-GATING — an UNRESOLVED hard failure sets review_required=True and
     export_blocked=True. The engine never silently ships an unresolved hard
     segmentation failure.

Pure functions. Zero I/O. Zero AI. Deterministic — identical (cues, rules)
always produce an identical contract.

Compliance refs: FCC 47 CFR §79.1, TPN MS-4.x, SOC 2 CC7.4 (bounded +
attributable auto-correction) / CC8.1 (every verdict reproducible + pinned to
QC_POLICY_VERSION).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple

from .timing_grid import is_on_grid, minimum_gap_frames, minimum_gap_ms

# Versioned so a later policy change never retro-changes the audit meaning of a
# stored verdict. Bump ONLY when the issue catalog / thresholds / remediation
# contract change. MUST stay in lockstep with cc-segmentation-qc.js
# QC_POLICY_VERSION so JS/Python parity fixtures agree.
QC_POLICY_VERSION = 3

# Hard cap on bounded remediation attempts per cue window. One retry maximum:
# the loop may make at most this many APPLIED remedy attempts regardless of how
# many distinct operations remain. The finite guarantee (SOC 2 CC7.4).
QC_MAX_REMEDIATION_ATTEMPTS = 1

# ── Issue catalog ───────────────────────────────────────────────────────────
# Stable machine-readable codes. NEVER renumber/rename — persisted on
# CaptionCue.segmentation_qc_issue_codes + CCFormatRun.segmentation_qc_issue_counts
# and asserted byte-for-byte by the JS/Python parity fixtures.
QC_ISSUE_DURATION_BELOW_MIN = "DURATION_BELOW_MIN"
QC_ISSUE_FRAME_GRID_MISALIGNMENT = "FRAME_GRID_MISALIGNMENT"
QC_ISSUE_MIN_GAP_FRAMES = "MIN_GAP_FRAMES"
QC_ISSUE_FLASH_CUE = "FLASH_CUE"
QC_ISSUE_PROTECTED_PHRASE_SPLIT = "PROTECTED_PHRASE_SPLIT"
QC_ISSUE_PROTECTED_PHRASE_GEOMETRY_CONFLICT = "PROTECTED_PHRASE_GEOMETRY_CONFLICT"
# HARD. The delivered cue reads FASTER than the spec's max_cps for the ordinary
# reason (the speaker simply spoke faster than the reading limit) — NOT because a
# speaker label tipped it over (that is SPEAKER_LABEL_CAUSED_CPS_FAILURE). This is
# the plain over-reading-speed floor: every genuinely-too-fast cue is caught here,
# the bounded timing-extension remedy is attempted (grow into idle silence at head
# OR tail), and if no safe extension resolves it the cue is left review-required so
# the export gate blocks a clean export. The last-resort B1 floor — never a silent
# over-CPS ship. FCC 47 CFR §79.1 (readability) / SOC 2 CC8.1.
QC_ISSUE_CPS_OVER_MAX = "CPS_OVER_MAX"
QC_ISSUE_SPEAKER_LABEL_CAUSED_CPS_FAILURE = "SPEAKER_LABEL_CAUSED_CPS_FAILURE"
QC_ISSUE_SPEAKER_LABEL_CAUSED_CPL_FAILURE = "SPEAKER_LABEL_CAUSED_CPL_FAILURE"
QC_ISSUE_SPEAKER_LABEL_UNRESOLVED = "SPEAKER_LABEL_UNRESOLVED"
QC_ISSUE_MICRO_CUE = "MICRO_CUE"
QC_ISSUE_AVOIDABLE_FRAGMENT = "AVOIDABLE_FRAGMENT"
QC_ISSUE_READING_SPEED_IMBALANCE = "READING_SPEED_IMBALANCE"
QC_ISSUE_OVERLOADED_UNDERFILLED_ADJACENCY = "OVERLOADED_UNDERFILLED_ADJACENCY"
QC_ISSUE_MEANINGFUL_TEXT_REMOVED = "MEANINGFUL_TEXT_REMOVED"
QC_ISSUE_TIMING_PROVENANCE_MISSING = "TIMING_PROVENANCE_MISSING"
QC_ISSUE_SEGMENTATION_PROVENANCE_MISSING = "SEGMENTATION_PROVENANCE_MISSING"
QC_ISSUE_OPTIMIZER_BOUNDARY_VIOLATION = "OPTIMIZER_BOUNDARY_VIOLATION"
QC_ISSUE_PROVENANCE_INCOMPLETE = "PROVENANCE_INCOMPLETE"
# HARD + NON-OVERRIDABLE. A linguistic group whose EVERY source token had no
# trustworthy timing (all quarantined/untimed) → it is excluded from every timed
# cue and would be SILENTLY OMITTED from the exported deliverable. This is not a
# quality imperfection (content present but imperfect) — it is MISSING content.
# The export gate refuses it unconditionally (see NON_OVERRIDABLE_QC_CODES in
# lib/cc-export-gate.js). FCC 47 CFR §79.1 / SOC 2 CC8.1.
QC_ISSUE_UNRESOLVED_UNTIMED_CONTENT = "UNRESOLVED_UNTIMED_CONTENT"

# ── Evidence bound constants (deterministic, hostile-input-safe) ─────────────
# A malformed/oversized provider response must NEVER produce an unbounded issue
# or database record. Every unresolved-group evidence field is clamped to these.
UNRESOLVED_TEXT_MAX_CHARS = 2000       # preserve the text up to this; then truncate + flag
UNRESOLVED_MAX_WORDS = 400             # cap on the words[] array length
UNRESOLVED_ID_MAX_LEN = 128            # cap on any identifier string (speaker/utterance)
UNRESOLVED_REASON_MAX_LEN = 64         # cap on the reason code string

QC_SEVERITY = {
    QC_ISSUE_CPS_OVER_MAX: "fail",
    QC_ISSUE_DURATION_BELOW_MIN: "fail",
    QC_ISSUE_FRAME_GRID_MISALIGNMENT: "fail",
    QC_ISSUE_MIN_GAP_FRAMES: "fail",
    QC_ISSUE_UNRESOLVED_UNTIMED_CONTENT: "fail",
    QC_ISSUE_FLASH_CUE: "fail",
    QC_ISSUE_PROTECTED_PHRASE_SPLIT: "fail",
    QC_ISSUE_PROTECTED_PHRASE_GEOMETRY_CONFLICT: "fail",
    QC_ISSUE_SPEAKER_LABEL_CAUSED_CPS_FAILURE: "fail",
    QC_ISSUE_SPEAKER_LABEL_CAUSED_CPL_FAILURE: "fail",
    QC_ISSUE_SPEAKER_LABEL_UNRESOLVED: "fail",
    QC_ISSUE_MEANINGFUL_TEXT_REMOVED: "fail",
    QC_ISSUE_OPTIMIZER_BOUNDARY_VIOLATION: "fail",
    QC_ISSUE_MICRO_CUE: "warn",
    QC_ISSUE_AVOIDABLE_FRAGMENT: "warn",
    QC_ISSUE_READING_SPEED_IMBALANCE: "warn",
    QC_ISSUE_OVERLOADED_UNDERFILLED_ADJACENCY: "warn",
    QC_ISSUE_PROVENANCE_INCOMPLETE: "info",
    QC_ISSUE_TIMING_PROVENANCE_MISSING: "info",
    QC_ISSUE_SEGMENTATION_PROVENANCE_MISSING: "info",
}

_HARD_CODES = frozenset(c for c, s in QC_SEVERITY.items() if s == "fail")
_SEVERITY_RANK = {"info": 0, "warn": 1, "fail": 2}

# Map a hard QC issue back to the underlying TECHNICAL RULE it violates. Used to
# build technical_violations — the honest compliance surface that is NEVER
# filtered by workflow disposition. Parity with cc-segmentation-qc.js ISSUE_TO_RULE.
_ISSUE_TO_RULE = {
    QC_ISSUE_CPS_OVER_MAX: "max_cps",
    QC_ISSUE_DURATION_BELOW_MIN: "min_caption_duration_ms",
    QC_ISSUE_FRAME_GRID_MISALIGNMENT: "frame_grid",
    QC_ISSUE_MIN_GAP_FRAMES: "min_gap_between_captions_frames",
    QC_ISSUE_SPEAKER_LABEL_CAUSED_CPS_FAILURE: "max_cps",
    QC_ISSUE_SPEAKER_LABEL_CAUSED_CPL_FAILURE: "max_chars_per_line",
    QC_ISSUE_PROTECTED_PHRASE_GEOMETRY_CONFLICT: "protected_phrase",
    QC_ISSUE_PROTECTED_PHRASE_SPLIT: "protected_phrase",
    QC_ISSUE_FLASH_CUE: "min_caption_duration_ms",
    QC_ISSUE_MEANINGFUL_TEXT_REMOVED: "text_conservation",
    QC_ISSUE_OPTIMIZER_BOUNDARY_VIOLATION: "segmentation_boundary",
    QC_ISSUE_UNRESOLVED_UNTIMED_CONTENT: "timing_completeness",
}


def _clamp_str(v: Any, limit: int) -> str:
    """Coerce to str and hard-cap length. Deterministic; hostile-input-safe."""
    s = str(v if v is not None else "")
    return s[:limit]


def build_unresolved_untimed_evidence(group: Dict[str, Any], routing_index: int) -> Dict[str, Any]:
    """Bounded, deterministic evidence for ONE all-untimed linguistic group.

    OWNERSHIP: the `group` is the IMMUTABLE unresolved-group object SEGMENTATION
    owns (segmentation.build_unresolved_group). Segmentation QC READS its fields
    and COPIES the relevant ones into bounded issue evidence — it never reinterprets
    or augments the object. Two distinct index concepts are kept separate:
      • `segmentation_group_index` — the group's INTRINSIC absolute index in
        segmentation's output (read from the object, echoed for audit).
      • `unresolved_group_id` (ug#N) — a QC ROUTING artifact assigned HERE from
        `routing_index` (the ordinal AMONG unresolved groups). Never a cue id.

    Provenance ONLY — never a timestamp, never a cue id. Every field is clamped so
    a malformed/oversized provider response cannot bloat the issue or the row.
    Transcription provenance (provider/model/version/job-id/confidence-source) is
    copied from the object's immutable `transcription_provenance` block — the
    monotonic chain that originated at normalization. SOC 2 CC8.1.
    """
    # Accept BOTH list and tuple — the immutable unresolved-group object freezes
    # `words` to a tuple (FrozenDict deep-freeze), so a `list`-only guard would
    # silently drop every word (word_count → 0). Any other type is rejected.
    words = group.get("words")
    words = list(words) if isinstance(words, (list, tuple)) else []
    full_text = group.get("text")
    if full_text is None:
        full_text = " ".join(str(w) for w in words)
    full_text = str(full_text)
    original_len = len(full_text)
    truncated = original_len > UNRESOLVED_TEXT_MAX_CHARS
    text = full_text[:UNRESOLVED_TEXT_MAX_CHARS]
    bounded_words = [str(w) for w in words[:UNRESOLVED_MAX_WORDS]]

    # Absolute source-word offsets (half-open) owned by segmentation.
    word_start = group.get("source_word_start")
    word_end = group.get("source_word_end")
    source_word_range = (
        [int(word_start), int(word_end)]
        if isinstance(word_start, int) and isinstance(word_end, int)
        else None
    )

    # Transcription provenance copied from the immutable object (never fabricated).
    # Accept any Mapping — the frozen unresolved-group nests a FrozenDict here,
    # which is a collections.abc.Mapping but NOT a `dict`. A `dict`-only guard
    # would silently discard real provenance (provider/model → None) the instant
    # segmentation emits a frozen group. SOC 2 CC8.1 / FCC §79.1.
    prov = group.get("transcription_provenance")
    prov = prov if isinstance(prov, Mapping) else {}

    seg_index = group.get("segmentation_group_index")

    return {
        # QC ROUTING identity — the ordinal among unresolved groups. Never a cue id.
        "unresolved_group_id": f"ug#{routing_index}",
        # INTRINSIC segmentation identity — echoed for audit, distinct from ug#N.
        "segmentation_group_index": seg_index if isinstance(seg_index, int) else None,
        # Schema version the object was emitted at (deterministic version pinning).
        "unresolved_group_version": group.get("unresolved_group_version"),
        # Preserved text (bounded) + truncation provenance.
        "text": text,
        "text_truncated": truncated,
        "original_text_length": original_len,
        "words": bounded_words,
        "word_count": len(words),
        # Non-temporal source locator (absolute word offsets from segmentation).
        "source_word_range": source_word_range,
        "source_utterance_id": _clamp_str(group.get("source_utterance_id"), UNRESOLVED_ID_MAX_LEN) or None,
        "speaker": _clamp_str(group.get("speaker"), UNRESOLVED_ID_MAX_LEN) or None,
        # Transcription provenance (copied from the immutable object).
        "provider": _clamp_str(prov.get("provider"), UNRESOLVED_ID_MAX_LEN) or None,
        "model": _clamp_str(prov.get("model"), UNRESOLVED_ID_MAX_LEN) or None,
        "provider_version": _clamp_str(prov.get("provider_version"), UNRESOLVED_ID_MAX_LEN) or None,
        "transcription_job_id": _clamp_str(prov.get("transcription_job_id"), UNRESOLVED_ID_MAX_LEN) or None,
        "confidence_source": _clamp_str(prov.get("confidence_source"), UNRESOLVED_ID_MAX_LEN) or None,
        "reason": _clamp_str(group.get("reason") or "all_tokens_untimed", UNRESOLVED_REASON_MAX_LEN),
        "disposition": "unresolved_untimed",
    }

# Deterministic remedy operations the engine attempts per hard issue, in order.
# NO AI. NO word deletion. Recorded even when a remedy fails so the audit shows
# the bounded attempt set. Parity with cc-segmentation-qc.js attemptedOps().
_REMEDIATION_OPS = {
    QC_ISSUE_CPS_OVER_MAX: ["timing_extension"],
    QC_ISSUE_SPEAKER_LABEL_CAUSED_CPS_FAILURE: ["timing_extension", "line_reflow"],
    QC_ISSUE_SPEAKER_LABEL_CAUSED_CPL_FAILURE: ["line_reflow", "shorter_label"],
    QC_ISSUE_PROTECTED_PHRASE_GEOMETRY_CONFLICT: ["phrase_safe_reflow"],
    QC_ISSUE_PROTECTED_PHRASE_SPLIT: ["phrase_safe_reflow"],
    QC_ISSUE_FLASH_CUE: ["timing_extension"],
}

# ── Thresholds (policy-versioned) ───────────────────────────────────────────
FLASH_CUE_MAX_MS = 400          # below this a pop-on caption physically flashes
MICRO_CUE_MAX_MS = 1000         # below this a cue is uncomfortably short
READING_SPEED_IMBALANCE_RATIO = 2.2  # adjacent same-speaker CPS ratio
OVERLOAD_UNDERFILL_CHAR_RATIO = 3.0  # adjacent same-speaker char-load ratio

# Optimizer operations whose boundaries are DELIBERATE professional decisions.
# The QC stage must not silently reverse them; a violation is flagged, not fixed.
_OPTIMIZER_AUTHORED_OPS = frozenset({"resegment_2", "resegment_3", "merge_all"})

_CJK_RE = re.compile(r"[\u3000-\u9fff\uac00-\ud7af\uff00-\uffef]")
_LABEL_PREFIX_RE = re.compile(r"^(\[[^\]]+:\]\s+|>>\s*[^:]+:\s+|\([^)]+\)\s+)")
_SENTENCE_END_RE = re.compile(r"[.!?\u2026][\"'\u201d\u2019)\]]?\s*$")
_WORD_STRIP_RE = re.compile(r"[.,;:!?\"'()\[\]]")


# =============================================================================
# Text / measurement helpers
# =============================================================================

def _is_cjk(text: str) -> bool:
    return bool(_CJK_RE.search(str(text or "")))


def _cp_len(s: str) -> int:
    # Python str length is already codepoint-based (no surrogate miscounts).
    return len(str(s or ""))


def _words(text: str) -> List[str]:
    return [w for w in str(text or "").replace("\n", " ").split() if w]


def _normalize_word(w: str) -> str:
    return _WORD_STRIP_RE.sub("", str(w or "")).lower()


def _strip_to_dialogue(text: str) -> str:
    """Remove leading dash prefixes on multi-speaker dash captions so per-line
    word counting reflects real dialogue, not the '- ' presentation marker."""
    return re.sub(r"^-\s+", "", str(text or ""), flags=re.MULTILINE)


def _cue_text(cue: Dict[str, Any]) -> str:
    """The delivered text of a cue as a single \n-joined string. Engine cues
    carry `lines`; the JS reference carries `text`. Accept both for parity."""
    if isinstance(cue.get("text"), str):
        return cue["text"]
    lines = cue.get("lines") or []
    return "\n".join(str(l) for l in lines)


def _cue_lines(cue: Dict[str, Any]) -> List[str]:
    if isinstance(cue.get("lines"), list):
        return [str(l) for l in cue["lines"]]
    return str(cue.get("text") or "").split("\n")


def measure_delivered_cue(
    dialogue_text: str,
    label_prefix: str,
    timing: Dict[str, Any],
    spec_rules: Dict[str, Any],
) -> Dict[str, Any]:
    """Measure a cue AS DELIVERED — label prefix + dialogue together — so
    geometry (chars, longest line, CPS) reflects what the viewer reads."""
    prefix = label_prefix or ""
    delivered = f"{prefix}{dialogue_text or ''}"
    duration_ms = max(0, int(timing.get("end_ms", 0) or 0) - int(timing.get("start_ms", 0) or 0))
    duration_sec = duration_ms / 1000.0
    lines = str(delivered).split("\n")
    chars = _cp_len(delivered.replace("\n", ""))
    longest_line = max((_cp_len(l) for l in lines), default=0)
    measurement = (spec_rules.get("reading_speed_rules") or {}).get("cps_measurement", "characters")
    if measurement == "characters_no_spaces":
        measurement_chars = _cp_len(re.sub(r"\s", "", delivered))
    else:
        measurement_chars = chars
    cps = (measurement_chars / duration_sec) if duration_sec > 0 else float("inf")
    return {
        "delivered": delivered,
        "chars": chars,
        "longest_line_chars": longest_line,
        "cps": cps,
        "duration_ms": duration_ms,
    }


def assess_label_impact(
    dialogue_body: str,
    label_prefix: str,
    timing: Dict[str, Any],
    spec_rules: Dict[str, Any],
) -> Dict[str, Any]:
    """Determine whether the SPEAKER LABEL is what tips a cue into a CPS/CPL
    failure — compares the cue WITH the label against the same cue WITHOUT it."""
    max_cps = (spec_rules.get("reading_speed_rules") or {}).get("max_cps", 17)
    max_cpl = (spec_rules.get("line_rules") or {}).get("max_chars_per_line", 42)
    with_label = measure_delivered_cue(dialogue_body, label_prefix, timing, spec_rules)
    without_label = measure_delivered_cue(dialogue_body, "", timing, spec_rules)
    cps_newly_broken = without_label["cps"] <= max_cps and with_label["cps"] > max_cps
    cpl_newly_broken = (
        without_label["longest_line_chars"] <= max_cpl
        and with_label["longest_line_chars"] > max_cpl
    )
    return {
        "cps_newly_broken": cps_newly_broken,
        "cpl_newly_broken": cpl_newly_broken,
        "label_induced_failure": cps_newly_broken or cpl_newly_broken,
        "with_label": with_label,
        "without_label": without_label,
        "max_cps": max_cps,
        "max_cpl": max_cpl,
    }


# =============================================================================
# Protected-phrase detection + phrase-safe layout feasibility
# =============================================================================

def _phrase_split_across_line(text: str, protected_phrases: List[str]) -> Optional[str]:
    """Return the first protected phrase that straddles a line break (\n) inside
    the cue text, else None. Case-normalized (parity with JS)."""
    if not protected_phrases or "\n" not in str(text or ""):
        return None
    lines = str(text).split("\n")
    flat: List[str] = []
    break_indices = set()
    for li, line in enumerate(lines):
        for w in _words(line):
            flat.append(_normalize_word(w))
        if li < len(lines) - 1:
            break_indices.add(len(flat))  # break BEFORE this index
    for phrase in protected_phrases:
        pw = [_normalize_word(w) for w in str(phrase).lower().split()]
        pw = [w for w in pw if w]
        if len(pw) < 2:
            continue
        for i in range(0, len(flat) - len(pw) + 1):
            if flat[i:i + len(pw)] == pw:
                for b in range(i + 1, i + len(pw)):
                    if b in break_indices:
                        return phrase
    return None


def phrase_safe_layout_exists(text: str, phrase: str, max_cpl: int) -> bool:
    """True when SOME 2-line layout keeps `phrase` intact on one line without
    exceeding max_cpl on either line. If True the observed split was AVOIDABLE;
    if False the phrase genuinely cannot coexist with the geometry (conflict).
    Deterministic. Parity with cc-segmentation-qc.js phraseSafeLayoutExists()."""
    flat = _words(text)
    if len(flat) < 2:
        return True
    pw = [_normalize_word(w) for w in str(phrase).lower().split()]
    pw = [w for w in pw if w]
    norm = [_normalize_word(w) for w in flat]
    phrase_start = -1
    for i in range(0, len(norm) - len(pw) + 1):
        if norm[i:i + len(pw)] == pw:
            phrase_start = i
            break
    if phrase_start < 0:
        return True  # phrase not present as tokens — nothing to protect
    phrase_end = phrase_start + len(pw)  # exclusive
    for b in range(1, len(flat)):
        if phrase_start < b < phrase_end:
            continue  # would split the phrase
        l1 = " ".join(flat[:b])
        l2 = " ".join(flat[b:])
        if _cp_len(l1) <= max_cpl and _cp_len(l2) <= max_cpl:
            return True
    return False


def _ends_complete_sentence(text: str) -> bool:
    return bool(_SENTENCE_END_RE.search(str(text or "").strip()))


def _is_avoidable_fragment(cue: Dict[str, Any]) -> bool:
    dialogue = _strip_to_dialogue(_cue_text(cue)).strip()
    if not dialogue:
        return False
    if len(_words(dialogue)) > 2:
        return False
    if _ends_complete_sentence(dialogue):
        return False
    return True


# =============================================================================
# Issue construction
# =============================================================================

def make_issue(
    code: str,
    cue_ids: Optional[List[Any]] = None,
    window_index: Optional[int] = None,
    evidence: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    remediation_attempted: Optional[List[str]] = None,
    remediation_result: str = "none",
) -> Dict[str, Any]:
    """Build one fully-auditable QC issue in the canonical bounded shape. Always
    the same keys so downstream rollups never branch on missing fields."""
    return {
        "issue_code": code,
        "severity": QC_SEVERITY.get(code, "warn"),
        "cue_ids": list(cue_ids or []),
        "window_index": window_index,
        "evidence": dict(evidence or {}),
        "metrics": dict(metrics or {}),
        "remediation_attempted": list(remediation_attempted or []),
        "remediation_result": remediation_result,
        "qc_policy_version": QC_POLICY_VERSION,
    }


# =============================================================================
# Bounded remediation loop guard
# =============================================================================

def window_state_hash(cues: List[Dict[str, Any]]) -> str:
    """FNV-1a (32-bit) hash of a window's rendered state (timings + text). Used
    to detect a no-progress remediation loop — a remedy that reproduces a
    window-state we've already seen is refused. Parity with JS windowStateHash
    (same FNV-1a constants) so the loop's finiteness is identical cross-language."""
    canonical = "\u241e".join(
        f"{int(c.get('start_ms', 0) or 0)}|{int(c.get('end_ms', 0) or 0)}|{_cue_text(c)}"
        for c in (cues or [])
    )
    h = 0x811C9DC5
    for ch in canonical:
        h ^= ord(ch)
        h = (h * 0x01000193) & 0xFFFFFFFF
    return format(h, "08x")


class RemediationGuard:
    """Bounded remediation guard for ONE cue window. The loop MUST call
    should_attempt() before every remedy and record() after applying one.

    Refuses when: the attempt cap is hit, the resulting window-state repeats
    (no progress), or the same operation is retried. Guarantees a finite loop —
    at most QC_MAX_REMEDIATION_ATTEMPTS applied attempts. SOC 2 CC7.4."""

    def __init__(self) -> None:
        self._seen_states: set = set()
        self._seen_ops: set = set()
        self._attempts = 0

    @property
    def attempts(self) -> int:
        return self._attempts

    def should_attempt(self, next_state_hash: Optional[str], operation: Optional[str]) -> bool:
        if self._attempts >= QC_MAX_REMEDIATION_ATTEMPTS:
            return False
        if next_state_hash is not None and next_state_hash in self._seen_states:
            return False
        if operation is not None and operation in self._seen_ops:
            return False
        return True

    def record(self, state_hash: Optional[str], operation: Optional[str]) -> None:
        self._attempts += 1
        if state_hash is not None:
            self._seen_states.add(state_hash)
        if operation is not None:
            self._seen_ops.add(operation)


# =============================================================================
# Deterministic remedies (NO AI, NO word deletion, boundary-safe)
# =============================================================================

def _reflow_two_lines(dialogue_body: str, max_cpl: int) -> Optional[str]:
    """Deterministic best-balanced 2-line reflow of a dialogue body that keeps
    every word (no deletion). Returns the reflowed \n-joined text when a layout
    fits both lines under max_cpl, else None. Word order is preserved."""
    flat = _words(dialogue_body)
    if len(flat) < 2:
        return None
    best: Optional[Tuple[int, str]] = None
    for b in range(1, len(flat)):
        l1 = " ".join(flat[:b])
        l2 = " ".join(flat[b:])
        if _cp_len(l1) <= max_cpl and _cp_len(l2) <= max_cpl:
            diff = abs(_cp_len(l1) - _cp_len(l2))
            if best is None or diff < best[0]:
                best = (diff, f"{l1}\n{l2}")
    return best[1] if best else None


def _phrase_safe_reflow(dialogue_body: str, phrase: str, max_cpl: int) -> Optional[str]:
    """Deterministic 2-line reflow that KEEPS `phrase` intact on one line and
    fits both lines under max_cpl. Returns the \n-joined text or None. No word
    deletion; word order preserved."""
    flat = _words(dialogue_body)
    if len(flat) < 2:
        return None
    pw = [_normalize_word(w) for w in str(phrase).lower().split()]
    pw = [w for w in pw if w]
    norm = [_normalize_word(w) for w in flat]
    phrase_start = -1
    for i in range(0, len(norm) - len(pw) + 1):
        if norm[i:i + len(pw)] == pw:
            phrase_start = i
            break
    if phrase_start < 0:
        return _reflow_two_lines(dialogue_body, max_cpl)
    phrase_end = phrase_start + len(pw)
    best: Optional[Tuple[int, str]] = None
    for b in range(1, len(flat)):
        if phrase_start < b < phrase_end:
            continue  # would split the protected phrase
        l1 = " ".join(flat[:b])
        l2 = " ".join(flat[b:])
        if _cp_len(l1) <= max_cpl and _cp_len(l2) <= max_cpl:
            diff = abs(_cp_len(l1) - _cp_len(l2))
            if best is None or diff < best[0]:
                best = (diff, f"{l1}\n{l2}")
    return best[1] if best else None


def _extend_timing_for_cps(
    cue: Dict[str, Any],
    next_cue: Optional[Dict[str, Any]],
    delivered_chars: int,
    max_cps: int,
    min_gap_ms: int,
) -> Optional[int]:
    """Return a new end_ms that would bring the DELIVERED cue within max_cps by
    extending into idle time before the next cue, or None when no compliant
    extension is available (no idle room). Never overlaps the next cue; never
    shortens. Deterministic."""
    if max_cps <= 0 or delivered_chars <= 0:
        return None
    required_dur = int((delivered_chars / max_cps) * 1000) + 1
    start = int(cue.get("start_ms", 0) or 0)
    cur_end = int(cue.get("end_ms", 0) or 0)
    upper = (int(next_cue.get("start_ms", 0)) - min_gap_ms) if next_cue else (start + required_dur)
    new_end = min(start + required_dur, upper)
    if new_end > cur_end:
        return new_end
    return None


def _extend_timing_bidirectional_for_cps(
    cue: Dict[str, Any],
    prev_cue: Optional[Dict[str, Any]],
    next_cue: Optional[Dict[str, Any]],
    delivered_chars: int,
    max_cps: int,
    min_gap_ms: int,
) -> Optional[Tuple[int, int]]:
    """Return (new_start_ms, new_end_ms) that bring the DELIVERED cue within
    max_cps by growing the window into idle silence at the TAIL first (preferred —
    never moves the in-time), then, if still over, into idle silence at the HEAD.
    Returns None when no safe extension in either direction is available.

    Never overlaps a neighbour (honors min_gap on both sides), never shortens the
    cue. Deterministic. This is the B1-floor free reducer — head-room the tail-only
    extender left on the table is exactly what let a genuinely-over-CPS cue ship."""
    if max_cps <= 0 or delivered_chars <= 0:
        return None
    required_dur = int((delivered_chars / max_cps) * 1000) + 1
    start = int(cue.get("start_ms", 0) or 0)
    end = int(cue.get("end_ms", 0) or 0)
    cur_dur = max(1, end - start)
    if required_dur <= cur_dur:
        return None
    # TAIL first — extend end into the gap before the next cue (no in-time move).
    tail_ceiling = (int(next_cue.get("start_ms", 0)) - min_gap_ms) if next_cue else (start + required_dur)
    new_end = min(start + required_dur, tail_ceiling)
    new_end = max(new_end, end)  # never shorten
    if (new_end - start) >= required_dur:
        return (start, new_end)
    # Still short → also pull the start earlier into leading idle silence.
    head_floor = (int(prev_cue.get("end_ms", 0)) + min_gap_ms) if prev_cue else (new_end - required_dur)
    new_start = max(new_end - required_dur, head_floor)
    new_start = min(new_start, start)  # never move the in-time later
    if (new_end - new_start) >= required_dur and new_start < start:
        return (new_start, new_end)
    return None


# =============================================================================
# Per-cue label-splitting
# =============================================================================

def _split_label(cue: Dict[str, Any]) -> Tuple[str, str]:
    """Return (label_prefix, dialogue_body). Prefix is '' when the cue carries
    no rendered speaker label. Only splits when speaker_label is present AND the
    text actually starts with a recognizable label token (engine-emitted)."""
    text = _cue_text(cue)
    if not cue.get("speaker_label"):
        return "", text
    m = _LABEL_PREFIX_RE.match(text)
    if not m:
        return "", text
    return m.group(0), text[len(m.group(0)):]


# =============================================================================
# Sequence inspection + remediation — the main entry point
# =============================================================================

def run_segmentation_qc(cues: List[Dict[str, Any]], rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Inspect a final rendered cue sequence, run the bounded deterministic
    remediation loop on hard defects, and return the canonical QC contract.

    @param cues  final rendered cues (post shaping / optimizer / readability /
                 CPS remediation / condensation / label + line rendering). Each
                 cue: { id?, start_ms, end_ms, text|lines, speaker_label?,
                 speaker_id?, meta? }.
    @param rules { line_rules, reading_speed_rules, protected_phrases,
                 original_word_bag? } — original_word_bag (optional list of
                 lowercased spoken words) enables the meaningful-text-removal
                 check.

    @return the bounded contract:
        {
          technical_violations: [str],
          segmentation_qc_issues: [ {issue_code, severity, cue_ids, window_index,
                                     evidence, metrics, remediation_attempted,
                                     remediation_result, qc_policy_version} ],
          review_required: bool,
          export_blocked: bool,
          segmentation_qc_policy_version: int,
          cue_summaries: [ {cue_id, ...bounded per-cue summary} ],
          rollup: { ...run-level rollup } ,
        }
    """
    rules = rules or {}
    spec_rules = {
        "line_rules": rules.get("line_rules") or {},
        "reading_speed_rules": rules.get("reading_speed_rules") or {},
    }
    protected_phrases = [str(p).lower() for p in (rules.get("protected_phrases") or [])]
    line_rules = spec_rules["line_rules"]
    reading_rules = spec_rules["reading_speed_rules"]
    max_cpl = int(line_rules.get("max_chars_per_line", 42) or 42)
    max_cps = int(reading_rules.get("max_cps", 17) or 17)
    # The frame lattice is authoritative. A millisecond guide such as 83 ms at
    # 25 fps resolves to two frames (80 ms); grading the projected 80 ms gap
    # against the unprojected 83 ms number would falsely fail every legal pair.
    min_gap_ms = minimum_gap_ms()
    min_caption_ms = int(line_rules.get("min_caption_duration_ms", 800) or 800)

    provenance_expected = bool(rules.get("provenance_expected", False))
    original_word_bag = rules.get("original_word_bag")

    cue_list = list(cues or [])
    issues: List[Dict[str, Any]] = []
    cue_issue_map: Dict[Any, List[Dict[str, Any]]] = {}

    def _cue_id(idx: int, cue: Dict[str, Any]) -> Any:
        cid = cue.get("id")
        return cid if cid is not None else f"#{idx}"

    def _push(cid: Any, issue: Dict[str, Any]) -> None:
        issues.append(issue)
        cue_issue_map.setdefault(cid, []).append(issue)

    for i, cue in enumerate(cue_list):
        cid = _cue_id(i, cue)
        duration_ms = max(0, int(cue.get("end_ms", 0) or 0) - int(cue.get("start_ms", 0) or 0))
        cjk = _is_cjk(_cue_text(cue))
        next_cue = cue_list[i + 1] if i + 1 < len(cue_list) else None
        meta = cue.get("meta") or {}
        seq_op = (meta.get("seq_opt") or {}).get("operation")

        # ── Contractual timing floor + frame-grid policy ───────────────────
        if cue.get("type", "dialogue") == "dialogue" and duration_ms < min_caption_ms:
            _push(cid, make_issue(
                QC_ISSUE_DURATION_BELOW_MIN, cue_ids=[cid], window_index=i,
                evidence={"duration_ms": duration_ms, "minimum_ms": min_caption_ms},
                metrics={"duration_ms": duration_ms},
                remediation_attempted=["local_group_recomposition"],
                remediation_result="unresolved",
            ))
        if not is_on_grid(int(cue.get("start_ms", 0))) or not is_on_grid(int(cue.get("end_ms", 0))):
            _push(cid, make_issue(
                QC_ISSUE_FRAME_GRID_MISALIGNMENT, cue_ids=[cid], window_index=i,
                evidence={"start_ms": cue.get("start_ms"), "end_ms": cue.get("end_ms")},
                remediation_result="unresolved",
            ))
        if (next_cue is not None and cue.get("type", "dialogue") == "dialogue"
                and next_cue.get("type", "dialogue") == "dialogue"):
            actual_gap = int(next_cue.get("start_ms", 0)) - int(cue.get("end_ms", 0))
            if actual_gap < min_gap_ms:
                _push(cid, make_issue(
                    QC_ISSUE_MIN_GAP_FRAMES, cue_ids=[cid, _cue_id(i + 1, next_cue)], window_index=i,
                    evidence={"actual_gap_ms": actual_gap, "minimum_gap_ms": min_gap_ms,
                              "minimum_gap_frames": minimum_gap_frames()},
                    remediation_result="unresolved",
                ))

        # ── Rhythm: flash + micro cue (every script incl. CJK) ──────────────
        if 0 < duration_ms < FLASH_CUE_MAX_MS:
            issue = make_issue(
                QC_ISSUE_FLASH_CUE, cue_ids=[cid], window_index=i,
                evidence={"duration_ms": duration_ms, "min_ms": FLASH_CUE_MAX_MS},
                metrics={"duration_ms": duration_ms},
            )
            _remediate_flash(issue, cue, next_cue, spec_rules, cue_list, i, min_gap_ms, max_cps)
            _push(cid, issue)
        elif 0 < duration_ms < MICRO_CUE_MAX_MS:
            _push(cid, make_issue(
                QC_ISSUE_MICRO_CUE, cue_ids=[cid], window_index=i,
                evidence={"duration_ms": duration_ms, "min_ms": MICRO_CUE_MAX_MS},
                metrics={"duration_ms": duration_ms},
                remediation_result="flagged",
            ))

        # ── Optimizer-authored boundary protection ──────────────────────────
        # If a cue is tagged with an optimizer resegmentation op but its
        # provenance hash is missing/mismatched, the boundary was tampered with
        # downstream — flag, never silently reverse.
        if seq_op in _OPTIMIZER_AUTHORED_OPS:
            so = meta.get("seq_opt") or {}
            if not so.get("output_hash"):
                _push(cid, make_issue(
                    QC_ISSUE_OPTIMIZER_BOUNDARY_VIOLATION, cue_ids=[cid], window_index=i,
                    evidence={"operation": seq_op, "reason": "missing_optimizer_output_hash"},
                    remediation_attempted=[], remediation_result="unresolved",
                ))

        # ── Provenance checks (info) ─────────────────────────────────────────
        if provenance_expected:
            has_timing_prov = bool(meta.get("word_timings")) or bool((meta.get("seq_opt") or {}).get("timing_provenance"))
            if not has_timing_prov:
                _push(cid, make_issue(
                    QC_ISSUE_TIMING_PROVENANCE_MISSING, cue_ids=[cid], window_index=i,
                    evidence={"reason": "no_word_timings_and_no_timing_provenance"},
                    remediation_result="flagged",
                ))
            if cue.get("type", "dialogue") == "dialogue" and not meta.get("seq_opt"):
                _push(cid, make_issue(
                    QC_ISSUE_SEGMENTATION_PROVENANCE_MISSING, cue_ids=[cid], window_index=i,
                    evidence={"reason": "no_seq_opt_provenance"},
                    remediation_result="flagged",
                ))

        # ── Latin-only editorial checks ──────────────────────────────────────
        if not cjk:
            if _is_avoidable_fragment(cue):
                _push(cid, make_issue(
                    QC_ISSUE_AVOIDABLE_FRAGMENT, cue_ids=[cid], window_index=i,
                    evidence={"text": _cue_text(cue)},
                    metrics={"word_count": len(_words(_strip_to_dialogue(_cue_text(cue))))},
                    remediation_result="flagged",
                ))
            broken_phrase = _phrase_split_across_line(_cue_text(cue), protected_phrases)
            if broken_phrase:
                phrase_safe_fits = phrase_safe_layout_exists(_cue_text(cue), broken_phrase, max_cpl)
                code = (QC_ISSUE_PROTECTED_PHRASE_SPLIT if phrase_safe_fits
                        else QC_ISSUE_PROTECTED_PHRASE_GEOMETRY_CONFLICT)
                issue = make_issue(
                    code, cue_ids=[cid], window_index=i,
                    evidence={
                        "phrase": broken_phrase, "text": _cue_text(cue),
                        "max_cpl": max_cpl, "phrase_safe_layout_exists": phrase_safe_fits,
                    },
                )
                _remediate_protected_phrase(issue, cue, broken_phrase, max_cpl, code)
                _push(cid, issue)

        # ── Tag-aware CPS / CPL: is the speaker label the cause? ─────────────
        if cue.get("speaker_label"):
            label_prefix, dialogue_body = _split_label(cue)
            if label_prefix:
                impact = assess_label_impact(dialogue_body, label_prefix, cue, spec_rules)
                if impact["cps_newly_broken"]:
                    issue = make_issue(
                        QC_ISSUE_SPEAKER_LABEL_CAUSED_CPS_FAILURE, cue_ids=[cid], window_index=i,
                        evidence={
                            "label": label_prefix.strip(),
                            "cps_with": impact["with_label"]["cps"],
                            "cps_without": impact["without_label"]["cps"],
                            "max_cps": impact["max_cps"],
                        },
                        metrics={"cps": impact["with_label"]["cps"]},
                    )
                    _remediate_label_cps(issue, cue, next_cue, label_prefix, dialogue_body,
                                         spec_rules, cue_list, i, max_cps, max_cpl, min_gap_ms)
                    _push(cid, issue)
                if impact["cpl_newly_broken"]:
                    issue = make_issue(
                        QC_ISSUE_SPEAKER_LABEL_CAUSED_CPL_FAILURE, cue_ids=[cid], window_index=i,
                        evidence={
                            "label": label_prefix.strip(),
                            "cpl_with": impact["with_label"]["longest_line_chars"],
                            "cpl_without": impact["without_label"]["longest_line_chars"],
                            "max_cpl": impact["max_cpl"],
                        },
                        metrics={"longest_line_chars": impact["with_label"]["longest_line_chars"]},
                    )
                    _remediate_label_cpl(issue, cue, label_prefix, dialogue_body, max_cpl)
                    _push(cid, issue)

        # ── Plain over-CPS floor (hard) — the B1 last-resort gate ───────────
        # The delivered cue reads faster than max_cps for the ORDINARY reason
        # (spoken too fast), independent of any speaker label. This is the check
        # that was missing: the label-CPS check above fires ONLY when the label
        # itself tipped an otherwise-compliant cue over, so a cue that is over on
        # its own merits reached export silently. Here every genuinely-too-fast
        # dialogue cue is caught, the bounded bidirectional timing-extension remedy
        # is attempted, and if no safe extension resolves it the cue is left
        # review-required → the export gate blocks. All the safe upstream fixes
        # (reflow, resegmentation, extend, spec-gated condense) already ran before
        # QC; this is the honest floor after them. FCC §79.1 / SOC 2 CC8.1.
        if cue.get("type", "dialogue") == "dialogue":
            delivered_measure = measure_delivered_cue(_cue_text(cue), "", cue, spec_rules)
            if delivered_measure["cps"] > max_cps:
                issue = make_issue(
                    QC_ISSUE_CPS_OVER_MAX, cue_ids=[cid], window_index=i,
                    evidence={
                        "cps": delivered_measure["cps"],
                        "max_cps": max_cps,
                        "delivered_chars": delivered_measure["chars"],
                        "duration_ms": delivered_measure["duration_ms"],
                    },
                    metrics={"cps": delivered_measure["cps"]},
                )
                _remediate_cps_over_max(
                    issue, cue,
                    cue_list[i - 1] if i > 0 else None,
                    next_cue, spec_rules, max_cps, min_gap_ms)
                _push(cid, issue)

        # ── Meaningful-text removal (hard) ──────────────────────────────────
        # If an original spoken-word bag is supplied and the delivered sequence
        # dropped a meaningful (non-stopword) token that was NOT an attributed
        # condensation, that is a hard defect (never a silent deletion).
        if original_word_bag is not None and i == 0:
            removed = _detect_meaningful_removal(cue_list, original_word_bag)
            if removed:
                _push(cid, make_issue(
                    QC_ISSUE_MEANINGFUL_TEXT_REMOVED, cue_ids=[cid], window_index=None,
                    evidence={"removed_words": removed[:20], "removed_count": len(removed)},
                    remediation_attempted=[], remediation_result="unresolved",
                ))

        # ── Adjacency: reading-speed imbalance + overload/underfill ─────────
        if next_cue and (cue.get("speaker_id") or cue.get("speaker_label")) and \
                (cue.get("speaker_id") or cue.get("speaker_label")) == \
                (next_cue.get("speaker_id") or next_cue.get("speaker_label")):
            a = measure_delivered_cue(_strip_to_dialogue(_cue_text(cue)), "", cue, spec_rules)
            b = measure_delivered_cue(_strip_to_dialogue(_cue_text(next_cue)), "", next_cue, spec_rules)
            next_id = _cue_id(i + 1, next_cue)
            hi_cps, lo_cps = max(a["cps"], b["cps"]), min(a["cps"], b["cps"])
            if hi_cps != float("inf") and lo_cps > 0 and (hi_cps / lo_cps) >= READING_SPEED_IMBALANCE_RATIO:
                _push(cid, make_issue(
                    QC_ISSUE_READING_SPEED_IMBALANCE, cue_ids=[cid, next_id], window_index=i,
                    evidence={"cps_a": a["cps"], "cps_b": b["cps"], "ratio": hi_cps / lo_cps},
                    metrics={"ratio": hi_cps / lo_cps},
                    remediation_result="flagged",
                ))
            hi_ch, lo_ch = max(a["chars"], b["chars"]), min(a["chars"], b["chars"])
            if lo_ch > 0 and (hi_ch / lo_ch) >= OVERLOAD_UNDERFILL_CHAR_RATIO:
                _push(cid, make_issue(
                    QC_ISSUE_OVERLOADED_UNDERFILLED_ADJACENCY, cue_ids=[cid, next_id], window_index=i,
                    evidence={"chars_a": a["chars"], "chars_b": b["chars"], "ratio": hi_ch / lo_ch},
                    metrics={"ratio": hi_ch / lo_ch},
                    remediation_result="flagged",
                ))

    # ── Unresolved all-untimed groups (transient input from the formatter) ──
    # Detection lives in segmentation; ADJUDICATION lives HERE. The formatter
    # passes each all-untimed linguistic group (never a cue) via rules; we emit
    # ONE canonical fail/unresolved issue per group. This is the ONLY persisted
    # representation of the condition — there is no parallel unresolved_content
    # channel. Each issue carries bounded provenance evidence and a NON-CUE
    # identifier. Its severity=fail + remediation_result=unresolved drive
    # has_unresolved_hard → review_required + export_blocked, exactly like every
    # other hard segmentation defect. SOC 2 CC8.1 / FCC §79.1.
    unresolved_groups = rules.get("unresolved_groups") or []
    for routing_index, group in enumerate(unresolved_groups):
        # Accept any Mapping — the immutable unresolved-group object segmentation
        # emits is a FrozenDict (a collections.abc.Mapping, NOT a `dict`). A
        # `dict`-only guard would SKIP every frozen group, silently omitting the
        # UNRESOLVED_UNTIMED_CONTENT gate → untimed content ships unnoticed. This
        # is the exact FCC §79.1 failure the gate exists to prevent. SOC 2 CC8.1.
        if not isinstance(group, Mapping):
            continue
        # routing_index = the ordinal AMONG unresolved groups (drives ug#N).
        # The group's INTRINSIC segmentation_group_index rides inside the object.
        ev = build_unresolved_untimed_evidence(group, routing_index)
        issue = make_issue(
            QC_ISSUE_UNRESOLVED_UNTIMED_CONTENT,
            cue_ids=[], window_index=None,
            evidence=ev,
            remediation_attempted=[], remediation_result="unresolved",
        )
        # Anchor on the group's non-cue identity so the per-cue rollup counts it
        # without inventing a fake cue row. The key is the stable ug# id.
        _push(ev["unresolved_group_id"], issue)

    # ── Assemble the canonical contract ─────────────────────────────────────
    technical_violations = sorted({
        _ISSUE_TO_RULE[i["issue_code"]]
        for i in issues
        if i["issue_code"] in _HARD_CODES and i["issue_code"] in _ISSUE_TO_RULE
    })

    has_unresolved_hard = any(
        _SEVERITY_RANK.get(i["severity"], 0) == 2 and i["remediation_result"] != "resolved"
        for i in issues
    )

    cue_summaries = [
        {"cue_id": cid, **_build_cue_summary(lst)}
        for cid, lst in cue_issue_map.items()
    ]
    rollup = _build_run_rollup(issues, cue_issue_map)

    return {
        "technical_violations": technical_violations,
        "segmentation_qc_issues": issues,
        "review_required": has_unresolved_hard,
        "export_blocked": has_unresolved_hard,
        "segmentation_qc_policy_version": QC_POLICY_VERSION,
        "cue_summaries": cue_summaries,
        "rollup": rollup,
    }


# =============================================================================
# Remediation drivers — each consults a fresh bounded guard for its window
# =============================================================================

def _remediate_flash(issue, cue, next_cue, spec_rules, cue_list, idx, min_gap_ms, max_cps) -> None:
    guard = RemediationGuard()
    ops = _REMEDIATION_OPS.get(QC_ISSUE_FLASH_CUE, [])
    issue["remediation_attempted"] = list(ops)
    delivered = measure_delivered_cue(_cue_text(cue), "", cue, spec_rules)
    for op in ops:
        if op != "timing_extension":
            continue
        new_end = _extend_timing_for_cps(cue, next_cue, delivered["chars"], max_cps, min_gap_ms)
        # For a flash cue, "resolved" means we can lift duration past the flash
        # floor without overlapping the next cue.
        start = int(cue.get("start_ms", 0) or 0)
        target_end = start + FLASH_CUE_MAX_MS
        upper = (int(next_cue.get("start_ms", 0)) - min_gap_ms) if next_cue else target_end
        candidate_end = min(target_end, upper)
        if candidate_end - start >= FLASH_CUE_MAX_MS:
            trial = list(cue_list)
            trial[idx] = {**cue, "end_ms": candidate_end}
            state = window_state_hash([trial[idx]])
            if guard.should_attempt(state, op):
                guard.record(state, op)
                issue["remediation_result"] = "resolved"
                issue["evidence"]["resolved_by"] = op
                issue["evidence"]["new_end_ms"] = candidate_end
                return
    issue["remediation_result"] = "unresolved"


def _remediate_cps_over_max(issue, cue, prev_cue, next_cue, spec_rules,
                            max_cps, min_gap_ms) -> None:
    """Bounded remedy for a plain over-CPS cue: attempt a bidirectional timing
    extension (grow into idle silence at tail, then head) and mark 'resolved'
    only when the re-measured delivered CPS clears max_cps. Otherwise
    'unresolved' → review_required + export_blocked (the B1 floor). One retry
    max via the shared guard. No word change, no reflow — every words-changing /
    resegmentation remedy already ran upstream before QC. SOC 2 CC7.4."""
    guard = RemediationGuard()
    ops = _REMEDIATION_OPS.get(QC_ISSUE_CPS_OVER_MAX, [])
    issue["remediation_attempted"] = list(ops)
    delivered = measure_delivered_cue(_cue_text(cue), "", cue, spec_rules)
    extended = _extend_timing_bidirectional_for_cps(
        cue, prev_cue, next_cue, delivered["chars"], max_cps, min_gap_ms)
    if extended is not None:
        new_start, new_end = extended
        trial = {**cue, "start_ms": new_start, "end_ms": new_end}
        state = window_state_hash([trial])
        if guard.should_attempt(state, "timing_extension"):
            guard.record(state, "timing_extension")
            recheck = measure_delivered_cue(_cue_text(cue), "", trial, spec_rules)
            if recheck["cps"] <= max_cps:
                issue["remediation_result"] = "resolved"
                issue["evidence"]["resolved_by"] = "timing_extension"
                issue["evidence"]["new_start_ms"] = new_start
                issue["evidence"]["new_end_ms"] = new_end
                return
    issue["remediation_result"] = "unresolved"


def _remediate_label_cps(issue, cue, next_cue, label_prefix, dialogue_body,
                         spec_rules, cue_list, idx, max_cps, max_cpl, min_gap_ms) -> None:
    guard = RemediationGuard()
    ops = _REMEDIATION_OPS.get(QC_ISSUE_SPEAKER_LABEL_CAUSED_CPS_FAILURE, [])
    issue["remediation_attempted"] = list(ops)
    delivered = measure_delivered_cue(dialogue_body, label_prefix, cue, spec_rules)
    for op in ops:
        if op == "timing_extension":
            new_end = _extend_timing_for_cps(cue, next_cue, delivered["chars"], max_cps, min_gap_ms)
            if new_end is not None:
                trial = {**cue, "end_ms": new_end}
                state = window_state_hash([trial])
                if guard.should_attempt(state, op):
                    guard.record(state, op)
                    recheck = measure_delivered_cue(dialogue_body, label_prefix, trial, spec_rules)
                    if recheck["cps"] <= max_cps:
                        issue["remediation_result"] = "resolved"
                        issue["evidence"]["resolved_by"] = op
                        issue["evidence"]["new_end_ms"] = new_end
                        return
        # line_reflow cannot lower CPS (timing-neutral) — recorded as attempted
        # per the bounded set, but it can only resolve a CPL failure, not CPS.
    issue["remediation_result"] = "unresolved"


def _remediate_label_cpl(issue, cue, label_prefix, dialogue_body, max_cpl) -> None:
    guard = RemediationGuard()
    ops = _REMEDIATION_OPS.get(QC_ISSUE_SPEAKER_LABEL_CAUSED_CPL_FAILURE, [])
    issue["remediation_attempted"] = list(ops)
    for op in ops:
        if op == "line_reflow":
            # Reflow the LABELLED delivered text so the label rides on one line
            # and no line exceeds max_cpl. No word deletion.
            reflowed = _reflow_two_lines(f"{label_prefix}{dialogue_body}", max_cpl)
            if reflowed is not None:
                state = window_state_hash([{**cue, "text": reflowed}])
                if guard.should_attempt(state, op):
                    guard.record(state, op)
                    longest = max((_cp_len(l) for l in reflowed.split("\n")), default=0)
                    if longest <= max_cpl:
                        issue["remediation_result"] = "resolved"
                        issue["evidence"]["resolved_by"] = op
                        return
        # shorter_label is a human decision (which abbreviation to use) — the
        # engine records it as attempted but never invents a label; unresolved.
    issue["remediation_result"] = "unresolved"


def _remediate_protected_phrase(issue, cue, phrase, max_cpl, code) -> None:
    # PROTECTED_PHRASE_SPLIT (avoidable) → the engine should already have chosen
    # the phrase-safe layout upstream; if it appears here a reflow resolves it.
    # PROTECTED_PHRASE_GEOMETRY_CONFLICT → no phrase-safe layout fits; unresolved.
    guard = RemediationGuard()
    ops = _REMEDIATION_OPS.get(code, [])
    issue["remediation_attempted"] = list(ops)
    if code == QC_ISSUE_PROTECTED_PHRASE_GEOMETRY_CONFLICT:
        issue["remediation_result"] = "unresolved"
        return
    dialogue_body = _strip_to_dialogue(_cue_text(cue))
    reflowed = _phrase_safe_reflow(dialogue_body, phrase, max_cpl)
    if reflowed is not None:
        state = window_state_hash([{**cue, "text": reflowed}])
        if guard.should_attempt(state, "phrase_safe_reflow"):
            guard.record(state, "phrase_safe_reflow")
            issue["remediation_result"] = "resolved"
            issue["evidence"]["resolved_by"] = "phrase_safe_reflow"
            return
    issue["remediation_result"] = "unresolved"


# =============================================================================
# Meaningful-text-removal detection
# =============================================================================

_STOPWORDS = frozenset({
    "a", "an", "the", "of", "to", "and", "or", "but", "with", "from", "in", "on",
    "at", "for", "that", "this", "these", "those", "is", "are", "was", "were",
})


def _detect_meaningful_removal(cue_list: List[Dict[str, Any]], original_word_bag: List[str]) -> List[str]:
    """Return meaningful (non-stopword) original words that no longer appear in
    the delivered sequence AND were not part of an attributed condensation. A
    non-empty result is a hard defect — the engine never silently deletes words."""
    delivered = []
    condensation_verbatim = []
    for c in cue_list:
        delivered.extend(_normalize_word(w) for w in _words(_strip_to_dialogue(_cue_text(c))))
        cond = (c.get("meta") or {}).get("condensation") or {}
        if cond.get("applied") and cond.get("verbatim"):
            condensation_verbatim.extend(_normalize_word(w) for w in _words(cond["verbatim"]))
    delivered_set = set(delivered)
    # Words removed by an attributed condensation are expected — exclude them.
    allowed_removed = set(condensation_verbatim) - delivered_set
    removed = []
    seen = set()
    for w in original_word_bag:
        nw = _normalize_word(w)
        if not nw or nw in _STOPWORDS:
            continue
        if nw not in delivered_set and nw not in allowed_removed and nw not in seen:
            removed.append(nw)
            seen.add(nw)
    return removed


# =============================================================================
# Per-cue + run rollups
# =============================================================================

def _build_cue_summary(cue_issue_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Bounded per-cue QC summary persisted onto CaptionCue. Parity with
    cc-segmentation-qc.js buildCueQcSummary()."""
    lst = cue_issue_list or []
    if not lst:
        return {
            "segmentation_qc_status": "pass",
            "segmentation_qc_highest_severity": "info",
            "segmentation_qc_issue_codes": [],
            "segmentation_qc_review_required": False,
            "segmentation_qc_remediation_outcome": "none",
        }
    codes = list(dict.fromkeys(i["issue_code"] for i in lst))
    hard = [i for i in lst if i["issue_code"] in _HARD_CODES]
    unresolved_hard = [i for i in hard if i["remediation_result"] != "resolved"]
    resolved_hard = [i for i in hard if i["remediation_result"] == "resolved"]
    if hard:
        highest = "fail"
    elif any(i["severity"] == "warn" for i in lst):
        highest = "warn"
    else:
        highest = "info"
    if unresolved_hard:
        status, review, outcome = "fail", True, "unresolved"
    elif resolved_hard:
        status, review, outcome = "corrected", False, "corrected"
    else:
        status, review, outcome = "warn", False, "flagged"
    return {
        "segmentation_qc_status": status,
        "segmentation_qc_highest_severity": highest,
        "segmentation_qc_issue_codes": codes,
        "segmentation_qc_review_required": review,
        "segmentation_qc_remediation_outcome": outcome,
    }


def _build_run_rollup(issues: List[Dict[str, Any]], cue_issue_map: Dict[Any, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Run-level rollup persisted onto CCFormatRun. Parity with
    cc-segmentation-qc.js buildRunQcRollup()."""
    issue_counts: Dict[str, int] = {}
    remediation_attempts = 0
    for iss in issues:
        issue_counts[iss["issue_code"]] = issue_counts.get(iss["issue_code"], 0) + 1
        if iss["remediation_attempted"]:
            remediation_attempts += 1
    # unresolved       = cues whose disposition is a hard UNRESOLVED deterministic
    #                    defect (segmentation_qc_status == 'fail').
    # review_required  = cues requiring HUMAN REVIEW (segmentation_qc_review_required
    #                    is True), derived INDEPENDENTLY from the per-cue summary —
    #                    NOT copied from `unresolved`. Under QC_POLICY_VERSION 1 the
    #                    two are equal by construction (a cue is review-required iff
    #                    it carries an unresolved hard defect); a version-pinned
    #                    invariant test asserts that equality. Kept as a SEPARATE
    #                    field so a future policy version may diverge them (a
    #                    corrected-but-reviewable or soft-but-reviewable cue) WITHOUT
    #                    a schema migration or reinterpretation of historical rows.
    #                    SOC 2 CC8.1.
    unresolved = corrected = warn = review_required = 0
    for lst in cue_issue_map.values():
        summ = _build_cue_summary(lst)
        if summ["segmentation_qc_status"] == "fail":
            unresolved += 1
        elif summ["segmentation_qc_status"] == "corrected":
            corrected += 1
        elif summ["segmentation_qc_status"] == "warn":
            warn += 1
        if summ["segmentation_qc_review_required"]:
            review_required += 1
    if not issues:
        completeness = "clean"
    elif unresolved > 0:
        completeness = "has_unresolved"
    elif corrected > 0:
        completeness = "corrected"
    else:
        completeness = "clean"
    return {
        "segmentation_qc_policy_version": QC_POLICY_VERSION,
        "segmentation_qc_completeness": completeness,
        "segmentation_qc_unresolved_count": unresolved,
        "segmentation_qc_review_required_count": review_required,
        "segmentation_qc_corrected_count": corrected,
        "segmentation_qc_warn_count": warn,
        "segmentation_qc_issue_counts": issue_counts,
        "segmentation_qc_remediation_attempts": remediation_attempts,
    }
