"""
Caption Formatter — Core pipeline.

UPDATED:
- All profile helpers always read env vars (no hardcoded NBCU fallbacks)
- Now consumes SOUND_DENSITY, SPEAKER_LABEL_MODE, TIMECODE_OFFSET_MS
- Now consumes ITALICIZE_TITLES, ITALICIZE_PHRASES
- Now consumes ALIGNMENT_DEFAULT, ALIGNMENT_WINDOWS
- Now consumes VALIDATE_TTML, FAIL_ON_TTML_VALIDATION

NOTE: This file is a reconstruction based on the original formatter.py.
      If you have custom logic in your current formatter.py beyond what's shown here,
      merge the changes from the "UPDATED" sections marked with comments.
"""

import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .rules import activate_rule_context, reset_rule_context, get_rule as _rule_get

# Project imports with local fallbacks
try:
    from .rendering import render_lines as _render_lines, segments_from_runs as _segments_from_runs
except Exception:
    # Defensive: must never crash a job. Falls back to a plain join. This
    # path should never fire in production (rendering.py ships with the engine).
    def _render_lines(words, speaker_runs, max_lines, max_chars, dialogue_text=None):
        return [dialogue_text if dialogue_text is not None else " ".join(words)]

    def _segments_from_runs(words, speaker_runs):
        if not speaker_runs:
            return [{"speaker": None, "text": " ".join(words)}]
        out = []
        for i, run in enumerate(speaker_runs):
            start = run.get("word_start", 0)
            end = (speaker_runs[i + 1].get("word_start", len(words))
                   if i + 1 < len(speaker_runs) else len(words))
            seg_words = words[start:end]
            if seg_words:
                out.append({"speaker": run.get("speaker"), "text": " ".join(seg_words)})
        return out

# CJK awareness — single source of truth for no-space-script handling.
try:
    from .cjk import is_cjk_text as _is_cjk, cjk_char_count as _cjk_count, wrap_cjk as _wrap_cjk
except Exception:  # pragma: no cover — defensive for alternate import roots
    def _is_cjk(text):
        return False

    def _cjk_count(text):
        return len((text or "").replace(" ", ""))

    def _wrap_cjk(text, max_chars, max_lines):
        return [text]

# GENERAL TIMING-DOMAIN PREDICATE — the formatter is TIMING-AWARE, never
# QUARANTINE-aware. Every timestamp-based decision consumes only has_timing()
# tokens; the formatter never asks WHY a token is untimed (that reason is an audit
# fact owned by the repair layer). Keeps the formatter reusable for any future
# provider / normalization / timing policy. SOC 2 CC8.1.
try:
    from .timing_repair import has_timing as _has_timing
except Exception:  # pragma: no cover — defensive for alternate import roots
    def _has_timing(token):
        s = token.get("start_ms", token.get("start"))
        e = token.get("end_ms", token.get("end"))
        return s is not None and e is not None and e >= s

try:
    from .assembly import normalize_tokens as _normalize_tokens, is_sound_token as _is_sound_token
except Exception:
    def _normalize_tokens(timestamps: Any) -> List[Dict[str, Any]]:
        source: List[Dict[str, Any]] = []
        if isinstance(timestamps, dict):
            if isinstance(timestamps.get("words"), list):
                source = list(timestamps.get("words") or [])
            elif isinstance(timestamps.get("utterances"), list):
                for utt in timestamps.get("utterances") or []:
                    if isinstance(utt, dict) and isinstance(utt.get("words"), list):
                        source.extend(utt.get("words") or [])
        elif isinstance(timestamps, list):
            source = timestamps
        out = []
        for item in source:
            # AUDIT: preserve UNTIMED tokens as untimed — never coerce a missing/
            # None timestamp to 0 (that would fabricate a real 0ms event). A token
            # with either endpoint absent keeps start_ms/end_ms=None so the general
            # has_timing() predicate excludes it from every timing calculation.
            _s = item.get("start_ms", item.get("start"))
            _e = item.get("end_ms", item.get("end"))
            token = {
                "text": item.get("text", ""),
                "start_ms": int(_s) if _s is not None else None,
                "end_ms": int(_e) if _e is not None else None,
                "speaker": item.get("speaker") or "A",
            }
            # Carry the audit marker through the fallback path too (timing gating
            # is via has_timing(), but keep the disposition flag for auditors).
            if item.get("timing_quarantined"):
                token["timing_quarantined"] = True
            # PRESERVE the source-utterance identity — the immutable pause-boundary
            # invariant (segmentation._token_utterance_id) reads this to detect a
            # distinct-utterance hard pause. Dropping it here silently disabled the
            # invariant on the fallback normalization path (assembly.normalize_tokens
            # keeps it verbatim; this fallback must match). SOC 2 CC8.1 / FCC §79.1.
            utt = item.get("source_utterance_id",
                           item.get("utterance_index", item.get("aai_utterance_index")))
            if utt is not None:
                token["source_utterance_id"] = utt
            # PRESERVE the immutable transcription-provenance object verbatim —
            # the real assembly.normalize_tokens returns tokens untouched (it
            # never strips keys), so this fallback MUST do the same or an
            # all-untimed group loses its provider/model/job-id in the
            # UNRESOLVED_UNTIMED_CONTENT evidence. Carrying it here keeps the
            # fallback at parity with the production normalizer. SOC 2 CC8.1 /
            # FCC §79.1 — provenance is never dropped by the fallback path.
            prov = item.get("transcription_provenance")
            if prov is not None:
                token["transcription_provenance"] = prov
            out.append(token)
        return out

    def _is_sound_token(text: str) -> bool:
        text = (text or "").strip().upper()
        return text.startswith("[") and text.endswith("]")

try:
    from .exporters import (
        parse_srt as _parse_srt,
        export_srt as _export_srt,
        export_vtt as _export_vtt,
        export_scc as _export_scc,
    )
except Exception:
    def _ms_to_srt(ms: int) -> str:
        ms = max(0, int(ms))
        hh, rem = divmod(ms, 3600000)
        mm, rem = divmod(rem, 60000)
        ss, rem = divmod(rem, 1000)
        return f"{hh:02d}:{mm:02d}:{ss:02d},{rem:03d}"

    def _parse_srt(text: str) -> List[Dict[str, Any]]:
        text = (text or "").replace("\r\n", "\n").strip()
        blocks = re.split(r"\n\s*\n", text)
        cues = []
        for block in blocks:
            lines = [ln for ln in block.split("\n") if ln.strip() != ""]
            if len(lines) < 2:
                continue
            if re.match(r"^\d+$", lines[0].strip()):
                idx = int(lines[0].strip())
                timing = lines[1]
                body = lines[2:]
            else:
                idx = len(cues) + 1
                timing = lines[0]
                body = lines[1:]
            m = re.match(r"(\d\d:\d\d:\d\d,\d\d\d)\s+-->\s+(\d\d:\d\d:\d\d,\d\d\d)", timing)
            if not m:
                continue
            cues.append({"idx": idx, "start_ms": _parse_tc(m.group(1)), "end_ms": _parse_tc(m.group(2)), "lines": body})
        return cues

    def _parse_tc(tc: str) -> int:
        hh, mm, ssms = tc.split(":")
        ss, ms = ssms.split(",")
        return ((int(hh) * 60 + int(mm)) * 60 + int(ss)) * 1000 + int(ms)

    def _export_srt(cues: List[Dict[str, Any]]) -> str:
        out = []
        for i, cue in enumerate(cues, 1):
            out.append(str(i))
            out.append(f"{_ms_to_srt(cue['start_ms'])} --> {_ms_to_srt(cue['end_ms'])}")
            out.extend(cue.get("lines") or [""])
            out.append("")
        return "\n".join(out).strip() + "\n"

    def _export_vtt(cues: List[Dict[str, Any]]) -> str:
        body = _export_srt(cues).replace(",", ".")
        return "WEBVTT\n\n" + body

    def _export_scc(cues: List[Dict[str, Any]]) -> Optional[str]:
        return None


STYLE_TAG_RE = re.compile(r"\{\+an\d\}")
BRACKET_TAG_RE = re.compile(r"\[[^\]]+\]")


# ─── Env Helpers (UPDATED: always read env vars) ────────────────────

def _env_int(name: str, default: int) -> int:
    raw = _rule_get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_str(name: str, default: str = "") -> str:
    return (_rule_get(name, default) or default).strip()


def _caption_profile() -> str:
    return _env_str("CAPTION_PROFILE", "nbcu").lower()


def _max_lines() -> int:
    return _env_int("CUSTOM_MAX_LINES", 2)


def _max_chars() -> int:
    return _env_int("CUSTOM_MAX_CHARS", 32)


def _target_cps() -> int:
    return _env_int("CUSTOM_TARGET_CPS", 27)


def _max_cps() -> int:
    return _env_int("CUSTOM_MAX_CPS", 45)


def _min_display_ms() -> int:
    return _env_int("CUSTOM_MIN_DISPLAY_MS", 800)


def _min_sound_display_ms() -> int:
    return _env_int("CUSTOM_MIN_SOUND_DISPLAY_MS", 800)


def _min_sound_ms() -> int:
    return _env_int("CUSTOM_MIN_SOUND_MS", 250)


def _sound_cluster_gap_ms() -> int:
    return _env_int("CUSTOM_SOUND_CLUSTER_GAP_MS", 1500)


def _merge_gap_ms() -> int:
    return _env_int("CUSTOM_MERGE_GAP_MS", 80)


def _timecode_offset_ms() -> int:
    """UPDATED: Now reads TIMECODE_OFFSET_MS env var."""
    return _env_int("TIMECODE_OFFSET_MS", 0)


def _sound_density() -> str:
    """UPDATED: Now reads SOUND_DENSITY env var."""
    return _env_str("SOUND_DENSITY", "conservative").lower()


def _speaker_label_mode() -> str:
    """UPDATED: Now reads SPEAKER_LABEL_MODE env var."""
    return _env_str("SPEAKER_LABEL_MODE", "dash").lower()


def _music_cue_format() -> str:
    """Per-spec music rendering. 'musical_note_prefix' wraps with ♪ (Pluto);
    'bracketed_uppercase' renders [MUSIC] (broadcast). Sent from the spec via
    MUSIC_CUE_FORMAT — the engine never hardcodes a client convention."""
    return _env_str("MUSIC_CUE_FORMAT", "bracketed_uppercase").lower()


def _sound_effect_format() -> str:
    """Per-spec sound-effect rendering. 'bracketed_uppercase' = [DOOR SLAMS];
    'parenthetical' = (door slams). Pluto accepts either. From SOUND_EFFECT_FORMAT."""
    return _env_str("SOUND_EFFECT_FORMAT", "bracketed_uppercase").lower()


def _lyrics_marker() -> str:
    """Glyph used to wrap music cues / lyrics (♪). From LYRICS_MARKER."""
    return _env_str("LYRICS_MARKER", "\u266a") or "\u266a"


def _no_formatting_tags() -> bool:
    """When true, strip ALL inline <b>/<i> markup at render time (Pluto forbids
    formatting tags). From NO_FORMATTING_TAGS."""
    return _env_str("NO_FORMATTING_TAGS", "0") in ("1", "true", "True")


def _render_audio_event_text(raw_text: str, event_type: str) -> str:
    """
    UNIVERSAL audio-event renderer driven by the per-spec env knobs.

    `raw_text` is the human label (e.g. 'MUSIC', 'DOOR SLAMS'); `event_type`
    is the structured kind from the provider's audio_events[] (e.g. 'music',
    'music_playing', 'crowd_noise', 'door_slam'). Music-family events are
    rendered per MUSIC_CUE_FORMAT; everything else per SOUND_EFFECT_FORMAT.

    The engine knows nothing about any specific client — it only obeys the
    format knobs the spec sent. Pluto sends MUSIC_CUE_FORMAT='musical_note_prefix'
    + SOUND_EFFECT_FORMAT='bracketed_uppercase' → '♪ music ♪' + '[DOOR SLAMS]'.
    """
    label = (raw_text or event_type or "").strip()
    if not label:
        return ""
    et = (event_type or "").lower()
    is_music = et in ("music", "music_playing", "music_note") or "music" in et

    if is_music:
        fmt = _music_cue_format()
        if fmt == "musical_note_prefix":
            note = _lyrics_marker()
            # ♪ wraps the music cue; title-case the human label for readability.
            body = label.title() if label.isupper() else label
            return f"{note} {body} {note}"
        if fmt == "italic":
            return f"<i>{label}</i>"
        if fmt == "none":
            return ""
        # bracketed_uppercase (default)
        return f"[{label.upper()}]"

    fmt = _sound_effect_format()
    if fmt == "parenthetical":
        return f"({label.lower()})"
    if fmt == "italic":
        return f"<i>{label}</i>"
    if fmt == "none":
        return ""
    # bracketed_uppercase (default)
    return f"[{label.upper()}]"


def _strip_formatting_tags(text: str) -> str:
    """Remove <b>/<i> markup. Used when NO_FORMATTING_TAGS is set (Pluto)."""
    if not text:
        return text
    return re.sub(r"</?[bi]>", "", text)


def _apply_no_formatting_tags(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """When the spec forbids formatting tags, strip <b>/<i> from every line."""
    if not _no_formatting_tags():
        return cues
    for cue in cues:
        cue["lines"] = [_strip_formatting_tags(l) for l in cue.get("lines", [])]
    return cues


def _italicize_titles() -> bool:
    """UPDATED: Now reads ITALICIZE_TITLES env var."""
    return _env_str("ITALICIZE_TITLES", "1") == "1"


def _italicize_phrases() -> List[str]:
    """UPDATED: Now reads ITALICIZE_PHRASES env var."""
    raw = _env_str("ITALICIZE_PHRASES", "")
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]


def _alignment_default() -> str:
    """UPDATED: Now reads ALIGNMENT_DEFAULT env var."""
    return _env_str("ALIGNMENT_DEFAULT", "none").lower()


def _alignment_windows() -> List[Dict[str, Any]]:
    """UPDATED: Now reads ALIGNMENT_WINDOWS env var (JSON array)."""
    raw = _env_str("ALIGNMENT_WINDOWS", "")
    if not raw:
        return []
    try:
        windows = json.loads(raw)
        if isinstance(windows, list):
            return windows
    except Exception:
        pass
    return []


# ─── TTML Export ────────────────────────────────────────────────────

def _export_ttml(cues: List[Dict[str, Any]], frame_rate: str, frame_rate_multiplier: str, time_base: str, align: str) -> str:
    """Minimal IMSC-1.1 TTML text profile export."""

    def ms_to_ttml(ms: int) -> str:
        ms = max(0, int(ms))
        hh, rem = divmod(ms, 3600000)
        mm, rem = divmod(rem, 60000)
        ss, rem = divmod(rem, 1000)
        return f"{hh:02d}:{mm:02d}:{ss:02d}.{rem:03d}"

    def strip_tags(s: str) -> str:
        s = STYLE_TAG_RE.sub("", s or "")
        return s

    def convert_italics(text: str) -> str:
        out = []
        parts = re.split(r"(</?i>)", text)
        italic = False
        for part in parts:
            if part == "<i>":
                italic = True
                continue
            if part == "</i>":
                italic = False
                continue
            if italic:
                out.append(f'<span tts:fontStyle="italic">{part}</span>')
            else:
                out.append(part)
        return "".join(out)

    def to_ttml_text(lines: List[str]) -> str:
        safe_lines = [convert_italics(strip_tags(l)) for l in lines]
        if not safe_lines:
            return ""
        if len(safe_lines) == 1:
            return safe_lines[0]
        return "<br/>".join(safe_lines)

    # Build TTML
    ttml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<tt xml:lang="en" xmlns="http://www.w3.org/ns/ttml"'
        f' xmlns:ttp="http://www.w3.org/ns/ttml#parameter"'
        f' xmlns:tts="http://www.w3.org/ns/ttml#styling"'
        f' xmlns:ttm="http://www.w3.org/ns/ttml#metadata"'
        f' ttp:timeBase="{time_base}"'
        f' ttp:frameRate="{frame_rate}"'
        f' ttp:frameRateMultiplier="{frame_rate_multiplier}">',
        '  <head>',
        '    <styling>',
        f'      <style xml:id="default" tts:textAlign="{align}" tts:fontFamily="proportionalSansSerif" tts:fontSize="80%"/>',
        '    </styling>',
        '    <layout>',
        '      <region xml:id="bottom" tts:origin="10% 80%" tts:extent="80% 20%" tts:displayAlign="after"/>',
        '    </layout>',
        '  </head>',
        '  <body>',
        '    <div>',
    ]

    for cue in cues:
        begin = ms_to_ttml(cue["start_ms"])
        end = ms_to_ttml(cue["end_ms"])
        text = to_ttml_text(cue.get("lines") or [])
        if text:
            ttml_lines.append(f'      <p begin="{begin}" end="{end}" region="bottom" style="default">{text}</p>')

    ttml_lines.extend([
        '    </div>',
        '  </body>',
        '</tt>',
    ])

    return "\n".join(ttml_lines) + "\n"


# ─── Env Override Management ────────────────────────────────────────

def apply_env_overrides(env_dict: Dict[str, Any]):
    """Activate one immutable job-scoped rule context.

    The legacy name is retained for the main runner contract, but this function
    no longer mutates process-wide environment variables.
    """
    token = activate_rule_context(env_dict if isinstance(env_dict, dict) else {})
    print(f"[RULES] Profile: {_rule_get('CAPTION_PROFILE', '')}")
    print(f"[RULES] Max lines: {_rule_get('CUSTOM_MAX_LINES', 'default')}")
    print(f"[RULES] Max chars: {_rule_get('CUSTOM_MAX_CHARS', 'default')}")
    print(f"[RULES] Frame rate: {_rule_get('CUSTOM_FRAME_RATE', 'default')}")
    print(f"[RULES] Gap frames: {_rule_get('CUSTOM_MIN_GAP_FRAMES', 'default')}")
    return token


def restore_env_overrides(token) -> None:
    """Reset the current job's rule context."""
    reset_rule_context(token)


# ─── Sound Density Filtering (UPDATED) ──────────────────────────────

def _filter_sound_cues_by_density(sound_tokens: List[Dict[str, Any]], total_duration_ms: int) -> List[Dict[str, Any]]:
    """
    UPDATED: Reads SOUND_DENSITY env var to control how many sound cues to keep.
    - conservative: strict, fewer cues (NBCU style)
    - balanced: moderate
    - aggressive: most cues
    """
    density = _sound_density()

    if density == "aggressive" or not sound_tokens:
        return sound_tokens

    min_sound = _min_sound_ms()
    cluster_gap = _sound_cluster_gap_ms()

    filtered = []
    last_end = -999999

    for token in sound_tokens:
        duration = token["end_ms"] - token["start_ms"]
        gap_from_last = token["start_ms"] - last_end

        if density == "conservative":
            # Conservative: require minimum duration AND minimum gap between sounds
            if duration >= min_sound and gap_from_last >= cluster_gap:
                filtered.append(token)
                last_end = token["end_ms"]
        else:
            # Balanced: just require minimum gap
            if gap_from_last >= (cluster_gap // 2):
                filtered.append(token)
                last_end = token["end_ms"]

    return filtered


# ─── Timecode Offset (UPDATED) ──────────────────────────────────────

def _apply_timecode_offset(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """UPDATED: Apply TIMECODE_OFFSET_MS to all cue timings."""
    offset = _timecode_offset_ms()
    if offset == 0:
        return cues

    for cue in cues:
        cue["start_ms"] = cue["start_ms"] + offset
        cue["end_ms"] = cue["end_ms"] + offset

    return cues


# ─── Italics Post-Processing (UPDATED) ─────────────────────────────

def _apply_italics(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """UPDATED: Apply italic formatting to title phrases if enabled."""
    if not _italicize_titles():
        return cues

    phrases = _italicize_phrases()
    if not phrases:
        return cues

    for cue in cues:
        if cue.get("type") != "dialogue":
            continue
        new_lines = []
        for line in cue.get("lines", []):
            for phrase in phrases:
                if phrase and phrase in line:
                    line = line.replace(phrase, f"<i>{phrase}</i>")
            new_lines.append(line)
        cue["lines"] = new_lines

    return cues


# ─── Alignment Post-Processing (UPDATED) ───────────────────────────

def _apply_alignment(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """UPDATED: Apply alignment tags based on ALIGNMENT_DEFAULT and ALIGNMENT_WINDOWS."""
    default_align = _alignment_default()
    windows = _alignment_windows()

    if default_align == "none" and not windows:
        return cues

    def _tc_str_to_ms(tc_str: str) -> int:
        """Parse HH:MM:SS,mmm or HH:MM:SS.mmm to ms."""
        tc_str = tc_str.strip().replace(".", ",")
        try:
            parts = tc_str.split(":")
            if len(parts) == 3:
                hh = int(parts[0])
                mm = int(parts[1])
                ss_ms = parts[2].split(",")
                ss = int(ss_ms[0])
                ms = int(ss_ms[1]) if len(ss_ms) > 1 else 0
                return ((hh * 60 + mm) * 60 + ss) * 1000 + ms
        except Exception:
            pass
        return 0

    # Parse windows into ms ranges
    parsed_windows = []
    for w in windows:
        if w.get("start") and w.get("end") and w.get("align"):
            parsed_windows.append({
                "start_ms": _tc_str_to_ms(w["start"]),
                "end_ms": _tc_str_to_ms(w["end"]),
                "align": w["align"],
            })

    for cue in cues:
        # Check if cue falls in any alignment window
        align_tag = None
        for pw in parsed_windows:
            if cue["start_ms"] >= pw["start_ms"] and cue["end_ms"] <= pw["end_ms"]:
                align_tag = pw["align"]
                break

        if not align_tag and default_align != "none":
            align_tag = default_align

        if align_tag and align_tag != "none":
            # Prepend SRT-style alignment tag to first line
            if cue.get("lines"):
                first_line = cue["lines"][0]
                if not first_line.startswith("{\\+"):
                    cue["lines"][0] = "{\\+" + align_tag + "}" + first_line

    return cues


# ─── Structured speaker attribution ─────────────────────────────────

def _structured_speaker_fields(c: Dict[str, Any]) -> Dict[str, Any]:
    """Derive STRUCTURED speaker fields for a result cue FROM SOURCE TOKEN
    STRUCTURE (the cue's meta.runs word-offset runs) — NEVER by parsing the
    rendered "[SPEAKER X:]" caption text. Returns a dict merged into the result
    cue by _result_cue. Contract:

      • Non-dialogue cue (music / sound_effect / …) → {} (no speaker fields).
      • Unknown-speaker cue (meta.review_required, or no known speaker in runs)
        → {"speaker_review_required": True}. NO speaker_label is emitted — the
        speaker was never silently defaulted to a neighbour.
      • Exactly ONE distinct known speaker → {"speaker_label": <speaker>}.
      • ≥2 distinct known speakers (a legitimate dash-grouped multi-speaker cue)
        → {"speaker_label": <first speaker, stable scalar primary>,
           "speaker_segments": [{speaker, text}, …]} — one bounded entry per
        source run (capped at 8), each carrying that run's own text span, so the
        ingester persists each line's TRUE speaker without ever labeling the
        whole cue as one speaker.

    SOC 2 CC8.1 / FCC 47 CFR §79.1 — every line's speaker is provable from source
    structure, and the cross-speaker-fusion defect class is answerable from data.
    """
    if c.get("type", "dialogue") != "dialogue":
        return {}

    meta = c.get("meta") or {}
    runs = meta.get("runs") or []
    review_required = bool(meta.get("review_required"))

    # Reconstruct the cue's word list from the source dialogue text (never from
    # rendered lines, which may carry speaker labels / dashes).
    dialogue_text = meta.get("dialogue_text")
    if dialogue_text is None:
        dialogue_text = " ".join(c.get("lines") or [])
    words = dialogue_text.split() if dialogue_text else []

    known_speakers = [r.get("speaker") for r in runs if r.get("speaker") is not None]
    distinct_known = []
    for s in known_speakers:
        if s not in distinct_known:
            distinct_known.append(s)

    # Unknown speaker: flagged for review, never given a label.
    if review_required or not distinct_known:
        out: Dict[str, Any] = {}
        if review_required or not known_speakers:
            out["speaker_review_required"] = True
        return out

    if len(distinct_known) == 1:
        return {"speaker_label": distinct_known[0]}

    # Legitimate multi-speaker cue — bounded per-run attribution + scalar primary.
    segments = _segments_from_runs(words, runs)
    speaker_segments = [
        {"speaker": str(s.get("speaker")), "text": str(s.get("text", ""))}
        for s in segments if s.get("speaker") is not None
    ][:8]
    return {
        "speaker_label": distinct_known[0],
        "speaker_segments": speaker_segments,
    }


# ─── Main Pipeline ──────────────────────────────────────────────────

def process_caption_job(
    backbone_srt_text: str,
    timestamps: Any,
    protected_phrases: Optional[List[str]] = None,
    output_formats: Optional[List[str]] = None,
    heartbeat: Optional[Any] = None,
    audio_events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Main caption processing pipeline.

    Steps:
    1. Parse backbone SRT
    2. Normalize timestamps into tokens
    3. Separate spoken vs sound tokens
    4. Filter sound cues by density (UPDATED)
    5. Build cue list with proper line breaks
    6. Run editorial AI refinement
    7. Apply readability rules
    8. Apply timecode offset (UPDATED)
    9. Apply italics (UPDATED)
    10. Apply alignment (UPDATED)
    11. Run QC
    12. Export to requested formats
    """
    protected_phrases = protected_phrases or []

    # Import processing modules
    try:
        from .editorial_ai import editorial_refine_cues
    except Exception:
        def editorial_refine_cues(cues, pp):
            return cues

    try:
        from .readability import apply_readability_rules
    except Exception:
        def apply_readability_rules(cues):
            return cues

    try:
        from .qc import qc_report
    except Exception:
        def qc_report(cin, cout, pp):
            return {"cues_in": cin, "cues_out": len(cout)}

    # Read settings
    max_lines = _max_lines()
    max_chars = _max_chars()

    print(f"[FORMATTER] Profile: {_caption_profile()}")
    print(f"[FORMATTER] Max lines: {max_lines}, Max chars: {max_chars}")
    print(f"[FORMATTER] Sound density: {_sound_density()}")
    print(f"[FORMATTER] Speaker mode: {_speaker_label_mode()}")
    print(f"[FORMATTER] Timecode offset: {_timecode_offset_ms()} ms")

    # 1. Parse backbone SRT
    backbone_cues = _parse_srt(backbone_srt_text)
    print(f"[FORMATTER] Backbone SRT cues: {len(backbone_cues)}")

    # 2. Normalize timestamps
    tokens = _normalize_tokens(timestamps)
    print(f"[FORMATTER] Tokens: {len(tokens)}")

    # 3. Separate spoken vs sound tokens
    spoken_tokens = [t for t in tokens if not _is_sound_token(t.get("text", ""))]
    sound_tokens = [t for t in tokens if _is_sound_token(t.get("text", ""))]
    print(f"[FORMATTER] Spoken: {len(spoken_tokens)}, Sound: {len(sound_tokens)}")

    # 4. Filter sound cues by density (UPDATED)
    # AUDIT: only TIMED tokens contribute to total_duration_ms — an untimed
    # token's end_ms is None and must never be coerced to 0 (nor crash max()).
    _timed_ends = [int(t["end_ms"]) for t in tokens if _has_timing(t)]
    total_duration_ms = max(_timed_ends) if _timed_ends else 0
    sound_tokens = _filter_sound_cues_by_density(sound_tokens, total_duration_ms)
    print(f"[FORMATTER] Sound after density filter: {len(sound_tokens)}")

    # 5. Build cue list. Native audio_events (Scribe v2 / baseline) are passed
    # through so they become real sound cues rendered per the spec's
    # MUSIC_CUE_FORMAT / SOUND_EFFECT_FORMAT. Without this they were silently
    # dropped — the "no sound cues" regression. SOC 2 / FCC §79.1 non-dialogue
    # coverage is now reproduced from the baseline on every reformat.
    cues, unresolved_groups = _build_cues_from_tokens(
        spoken_tokens, sound_tokens, max_lines, max_chars,
        audio_events=audio_events or [],
    )
    cues_in_count = len(cues)
    print(f"[FORMATTER] Initial cues built: {cues_in_count} "
          f"(native audio_events in: {len(audio_events or [])}, "
          f"unresolved groups: {len(unresolved_groups)})")

    # 5b. CAPTION SHAPING (universal, spec-driven "caption rhythm" stage).
    # Splits over-rhythm dialogue cues toward the spec's TARGET duration at
    # natural phrase boundaries (clause/sentence), using real word timings when
    # present and interpolation as fallback. This is the professional layer that
    # turns raw-utterance cues into broadcast-rhythm cues BEFORE editorial/CPS/
    # readability run — the stage the engine was missing. Script-aware via the
    # shared CJK helpers; identical pipeline for every language. Disabled per
    # spec via CUSTOM_SHAPING_ENABLED=0 (1:1 import posture). SOC 2 CC8.1.
    try:
        from .shaping import shape_caption_rhythm
        before_shape = len(cues)
        cues = shape_caption_rhythm(cues)
        print(f"[FORMATTER] After caption shaping: {len(cues)} cues "
              f"(was {before_shape})")
    except Exception as _e:
        print(f"[FORMATTER] caption shaping skipped (non-fatal): {_e}")

    # 5c. CROSS-CUE SEQUENCE OPTIMIZER (canonical Timed-Text Editorial
    # Segmentation stage). Runs AFTER shaping (word timings intact) and BEFORE
    # editorial-AI / readability+CPS / condensation. This is the ONLY stage that
    # evaluates a contiguous same-speaker cue WINDOW as a whole and is allowed to
    # REPLACE the local boundaries — finding a bounded complete segmentation
    # path with no result-count ceiling from the original word-timed stream,
    # vetoing the
    # non-compliant, scoring the rest, and stamping the redistribution-before-
    # condensation gate + full provenance on every emitted cue. It fixes the
    # flash-fragment and removed-phrase failure classes through general rules,
    # not per-example branches. Disabled per run via SEQ_OPTIMIZER_ENABLED=0.
    # SOC 2 CC8.1 — deterministic, attributable, reproducible.
    try:
        from .sequence_optimizer import optimize_cue_sequence
        before_opt = len(cues)
        cues = optimize_cue_sequence(cues)
        print(f"[FORMATTER] After sequence optimizer: {len(cues)} cues "
              f"(was {before_opt})")
    except Exception as _e:
        print(f"[FORMATTER] sequence optimizer skipped (non-fatal): {_e}")

    # 6. Editorial AI — pass the heartbeat through so each cue's OpenAI
    # call can pulse the job's updated_at timestamp. Closes the "engine
    # looks hung from outside even though it's still working" perception
    # gap that produced the Pluto-Test 16-min auto-fail.
    cues = editorial_refine_cues(cues, protected_phrases, heartbeat=heartbeat)
    print(f"[FORMATTER] After editorial AI: {len(cues)} cues")

    # 7. Readability
    cues = apply_readability_rules(cues)
    print(f"[FORMATTER] After readability: {len(cues)} cues")

    # 7a. FINAL LOCAL RECOMPOSITION — readability may have merged, extended, or
    # split cues. Re-run the bounded optimizer on the resulting local groups so
    # no illegal child created downstream is accepted without reconsideration.
    try:
        from .sequence_optimizer import optimize_cue_sequence
        cues = optimize_cue_sequence(cues)
        print(f"[FORMATTER] After final local recomposition: {len(cues)} cues")
    except Exception as _e:
        print(f"[FORMATTER] final local recomposition skipped (non-fatal): {_e}")

    # 7b. CONDENSATION — the "captions aren't transcripts" editorial stage.
    # Runs AFTER the deterministic CPS extend/split (applied inside readability)
    # and BEFORE the QC gate, so it fires ONLY on cues STILL over max_cps that
    # timing alone couldn't rescue. Spec-gated via CONDENSATION_MODE:
    #   off             → no-op (verbatim-required specs)
    #   disfluency_only → deterministic filler/repeat removal (default, no AI)
    #   condense_to_cps → + bounded, entity/number-locked LLM paraphrase
    # Every changed cue is stamped meta.condensation={applied,kind,verbatim} so
    # the Base44 ingester writes CaptionCue.original_verbatim_text + provenance
    # and the editor can show a "Condensed — verify" chip with one-click revert.
    # SOC 2 CC8.1 — words are never silently reworded; every change is attributed.
    condensation_stats: Dict[str, Any] = {}
    try:
        from .condensation import condense_cues
        before_condense = len(cues)
        cues = condense_cues(cues, heartbeat=heartbeat, stats_out=condensation_stats)
        print(f"[FORMATTER] After condensation: {len(cues)} cues (was {before_condense})")
    except Exception as _e:
        print(f"[FORMATTER] condensation skipped (non-fatal): {_e}")
        condensation_stats = {"error": str(_e)[:300]}

    # 8. Timecode offset (UPDATED)
    cues = _apply_timecode_offset(cues)

    # 9. Italics (UPDATED)
    cues = _apply_italics(cues)

    # 10. Alignment (UPDATED)
    cues = _apply_alignment(cues)

    # 10b. No-formatting-tags policy — strip <b>/<i> when the spec forbids
    # markup (Pluto). Runs AFTER italics so a spec that forbade tags can never
    # emit one even if an earlier stage added it. Universal mechanism, per-spec flag.
    cues = _apply_no_formatting_tags(cues)

    # 10c. first_occurrence_per_scene label suppression. render_lines is stateless
    # per-cue so it labels EVERY cue; this single stateful post-pass strips the
    # repeat labels so each speaker is labeled only on their first cue in the scene
    # (MVP: whole output = one scene). No-op for every other label_mode. Runs AFTER
    # readability (cues are final) and BEFORE QC so CPS/CPL grade the delivered text.
    try:
        from .rendering import suppress_repeat_speaker_labels
        cues = suppress_repeat_speaker_labels(cues)
    except Exception as _e:
        print(f"[FORMATTER] label-suppression skipped (non-fatal): {_e}")

    # 10d. SENTENCE-BOUNDARY CAPITALIZATION — the DETERMINISTIC final authority.
    # Runs LAST (after every stage that could have split, merged, condensed, or
    # AI-polished a cue) so the delivered text always has correct sentence-
    # boundary casing regardless of whether the optional editorial-AI ran:
    #   • a cue that CONTINUES the prior sentence keeps its first word lowercase
    #     (proper-noun-safe — never downcases "I", an acronym, or a proven
    #     proper noun), and
    #   • a cue that STARTS a new sentence capitalizes its first word.
    # This closes the two shaping/transcript casing defects (mid-sentence "This
    # place" left capitalized after a split; a new sentence "you've" left
    # lowercase) deterministically, with NO AI dependency. Idempotent. SOC 2
    # CC8.1 — reproducible casing an auditor can re-derive.
    try:
        from .capitalization import apply_sentence_capitalization
        cues = apply_sentence_capitalization(cues)
    except Exception as _e:
        print(f"[FORMATTER] sentence-capitalization skipped (non-fatal): {_e}")

    # 10d.1 FINAL LOCAL RECONSIDERATION — condensation and repeat-label
    # suppression run after the earlier optimizer pass and can change delivered
    # geometry. Reconsider complete local windows once more before the final CPL
    # fit so quote/phrase spans are rebalanced as a sequence, not split in
    # isolation. Deterministic and idempotent.
    try:
        from .sequence_optimizer import optimize_cue_sequence
        cues = optimize_cue_sequence(
            cues, defer_timing_constraints=True, repair_hard_geometry_only=True)
    except Exception as _e:
        print(f"[FORMATTER] final local reconsideration skipped (non-fatal): {_e}")

    # 10d.2 FINAL DELIVERED-FIT REFLOW — run the same deterministic shaper on
    # the actual delivered text before timing projection.
    try:
        from .shaping import enforce_cpl_fit
        cues = enforce_cpl_fit(cues)
    except Exception as _e:
        print(f"[FORMATTER] final delivered-fit reflow skipped (non-fatal): {_e}")

    # 10d.3 FRAME-GRID NORMALIZATION — final generated boundaries are resolved
    # against the immutable per-job frame policy before delivery validation.
    try:
        from .timing_grid import normalize_cue_timing
        cues = normalize_cue_timing(cues)
    except Exception as _e:
        print(f"[FORMATTER] frame-grid normalization skipped (non-fatal): {_e}")

    # 10e. SEGMENTATION QC — the canonical (production-authority) deterministic
    # inspection + bounded remediation stage. Runs AFTER every deterministic
    # transform that could change delivered text or timing (shaping → sequence
    # optimizer → readability + CPS remediation → condensation → timecode/italics/
    # alignment → label suppression → sentence capitalization) so it measures the
    # cue exactly as the viewer sees it, and BEFORE general QC (step 11) + export
    # serialization (step 12). It keeps TECHNICAL COMPLIANCE separate from WORKFLOW
    # DISPOSITION: technical_violations always lists the raw failing rules; an
    # unresolved hard defect additionally sets review_required + export_blocked.
    # Its per-cue summaries + run rollup are carried on result.segmentation_qc for
    # the Base44 ingester to persist verbatim — the ingester NEVER recomputes QC.
    # SOC 2 CC7.4 (bounded, attributable auto-correction) / CC8.1.
    seg_qc: Dict[str, Any] = {}
    try:
        from .segmentation_qc import run_segmentation_qc
        seg_qc = run_segmentation_qc(cues, {
            "line_rules": {
                "max_chars_per_line": max_chars,
                "max_lines_per_caption": max_lines,
                "min_gap_between_captions_ms": _merge_gap_ms(),
                "min_caption_duration_ms": _min_display_ms(),
            },
            "reading_speed_rules": {
                "max_cps": _max_cps(),
                "cps_measurement": _env_str("CPS_MEASUREMENT", "characters") or "characters",
            },
            "protected_phrases": protected_phrases,
            # TRANSIENT INPUT — the all-untimed linguistic groups detected in
            # segmentation. Segmentation QC adjudicates each into ONE canonical
            # UNRESOLVED_UNTIMED_CONTENT fail issue. This is the ONLY place the
            # unresolved groups are consumed; they are NOT exposed on the result.
            # Detection (segmentation) → adjudication (QC) → one canonical issue.
            "unresolved_groups": unresolved_groups,
        })
        print(f"[FORMATTER] Segmentation QC: completeness="
              f"{seg_qc.get('rollup', {}).get('segmentation_qc_completeness')} "
              f"review_required={seg_qc.get('review_required')} "
              f"export_blocked={seg_qc.get('export_blocked')}")
    except Exception as _e:
        # NEVER crash a job on the QC stage. Record an honest 'unavailable'
        # verdict so the Base44 ingester marks the audit not-applicable (not
        # failed) rather than silently shipping without a QC record.
        print(f"[FORMATTER] segmentation QC skipped (non-fatal): {_e}")
        seg_qc = {"error": str(_e)[:300], "segmentation_qc_policy_version": None}

    # 11. QC
    qc = qc_report(cues_in_count, cues, protected_phrases)
    print(f"[FORMATTER] QC: {qc}")

    # 12. Export
    if not output_formats:
        output_formats = [_env_str("OUTPUT_FORMATS", "srt")]
    output_formats = [f.strip().lower() for f in ",".join(output_formats).split(",") if f.strip()]

    # NOTE on meta: the Base44 ingesters (ccIngestReformatResult /
    # ccIngestRailwayResult) read cue.meta.condensation to persist
    # original_verbatim_text + condensation_applied provenance onto CaptionCue.
    # We carry ONLY the condensation block (not full meta — word_timings would
    # bloat the payload by megabytes on a feature). Dropping this was the bug
    # that made condensed cues land with no audit trail. SOC 2 CC8.1.
    def _result_cue(i: int, c: Dict[str, Any]) -> Dict[str, Any]:
        out = {
            "idx": i + 1,
            "start_ms": c["start_ms"],
            "end_ms": c["end_ms"],
            "lines": c["lines"],
            "type": c.get("type", "dialogue"),
        }
        # ── STRUCTURED SPEAKER ATTRIBUTION (from source token data, NOT from
        # parsing rendered "[SPEAKER X:]" text). Emitted from the cue's
        # meta.runs (per-speaker word offsets carried since packing). A single-
        # speaker cue emits a scalar speaker_label; a legitimate multi-speaker
        # cue (dash grouping) emits bounded speaker_segments — one entry per run
        # with the speaker + that segment's text span — so the ingester can
        # persist each line's true speaker without ever assigning the whole cue
        # to one speaker. review_required rides through so an unknown-speaker cue
        # is flagged, never silently defaulted. SOC 2 CC8.1 / FCC §79.1.
        speaker_fields = _structured_speaker_fields(c)
        out.update(speaker_fields)
        cond = (c.get("meta") or {}).get("condensation")
        if cond and cond.get("applied"):
            out.setdefault("meta", {})["condensation"] = cond
        # ── BOUNDED PAUSE PROVENANCE (item-4 audit contract) ──────────────────
        # A cue that opened at a hard inter-utterance pause carries bounded
        # provenance: the source + prior utterance ids, the MEASURED gap (ms),
        # the EFFECTIVE pause_boundary_ms threshold, the boundary reason, and the
        # immutable-boundary status. Emitted on result.meta.pause_provenance so
        # the final engine cue (and the Base44 ingester) retain the provenance
        # that was previously null. NEVER raw word arrays — bounded scalars only.
        # SOC 2 CC8.1 / FCC 47 CFR §79.1 — the boundary is provable from the row.
        pause_prov = (c.get("meta") or {}).get("pause_provenance")
        if pause_prov:
            out.setdefault("meta", {})["pause_provenance"] = {
                "source_utterance_id": pause_prov.get("source_utterance_id"),
                "prev_utterance_id": pause_prov.get("prev_utterance_id"),
                "measured_gap_ms": pause_prov.get("measured_gap_ms"),
                "effective_pause_boundary_ms": pause_prov.get("effective_pause_boundary_ms"),
                "boundary_reason": pause_prov.get("boundary_reason") or "source_utterance_pause",
                "immutable_boundary": bool(pause_prov.get("immutable_boundary", True)),
            }
        # Sequence-optimizer provenance — the bounded, meaningful audit summary
        # the application contract exposes (Step 13). We carry ONLY the summary
        # fields (not the full candidate_summaries debug payload — that stays in
        # the engine log to keep the result payload small on a feature). The
        # Base44 ingester persists these onto CaptionCue for the editor's
        # "compliant / optimized / condensed / review-required" distinction.
        # BOUNDED external seq_opt payload — the fields the Base44 ingester
        # persists onto CaptionCue (cue-level summary) and CCSegmentationDecision
        # (run-level audit). Deliberately EXCLUDES the heavy debug detail
        # (candidate_summaries, original_window_text, word_timings, raw candidate
        # text) — those stay in the engine's Railway run log, referenced by
        # candidate_set_hash. Every field here is a version, count, enum,
        # bounded reason-code list, or hash — never an unbounded array. SOC 2 CC8.1.
        so = (c.get("meta") or {}).get("seq_opt")
        if so:
            out.setdefault("meta", {})["seq_opt"] = {
                # identity / version
                "optimizer_version": so.get("optimizer_version"),
                "policy_version": so.get("policy_version"),
                "segmentation_policy_version": so.get("segmentation_policy_version", so.get("policy_version")),
                "language_policy_version": so.get("language_policy_version"),
                "decision_schema_version": so.get("decision_schema_version"),
                # ancestry (bounded)
                "source_cue_ids": (so.get("source_cue_ids") or [])[:40],
                "source_cue_count": so.get("source_cue_count"),
                "result_cue_count": so.get("result_cue_count"),
                "part_index": so.get("part_index"),
                "part_total": so.get("part_total"),
                # decision
                "operation": so.get("operation"),
                "segmentation_quality": so.get("segmentation_quality"),
                "selected_candidate_id": so.get("selected_candidate_id"),
                "candidate_count": so.get("candidate_count", so.get("candidates_considered")),
                "candidates_considered": so.get("candidates_considered"),
                "selected_reason_codes": (so.get("selected_reason_codes") or [])[:12],
                "rejected_reason_categories": (so.get("rejected_reason_categories") or [])[:12],
                "moved_word_count": so.get("moved_word_count", 0),
                # conservation + timing
                "text_conservation_status": so.get("text_conservation_status", so.get("text_conservation")),
                "token_conservation_method": so.get("token_conservation_method"),
                "timing_source": so.get("timing_source"),
                "timing_provenance": so.get("timing_provenance"),
                "timing_provenance_detail": so.get("timing_provenance_detail"),
                # condensation gate
                "condensation_evaluated": so.get("condensation_evaluated", False),
                "condensation_allowed": so.get("condensation_allowed", False),
                "condensation_reason": so.get("condensation_reason"),
                # review
                "review_required": so.get("review_required", False),
                "review_reason_codes": (so.get("review_reason_codes") or [])[:8],
                # audit hashes
                "input_hash": so.get("input_hash"),
                "candidate_set_hash": so.get("candidate_set_hash"),
                "output_hash": so.get("output_hash"),
                # bounded diagnostic aid (speaker + original boundaries, capped)
                "speaker": so.get("speaker"),
                "original_boundaries_ms": (so.get("original_boundaries_ms") or [])[:10],
            }
        return out

    result: Dict[str, Any] = {
        "cues": [_result_cue(i, c) for i, c in enumerate(cues)],
        "qc": qc,
        # Canonical Segmentation QC contract — the production-authority verdict.
        # The Base44 ingester (ccIngestReformatResult) VALIDATES + persists this
        # verbatim (cue-level summaries + run-level rollup) and NEVER recomputes
        # QC itself. Carries technical_violations, segmentation_qc_issues[],
        # review_required, export_blocked, segmentation_qc_policy_version,
        # cue_summaries[], rollup{}. SOC 2 CC8.1.
        "segmentation_qc": seg_qc or None,
        # Condensation verdict counts — persisted onto CCFormatRun.summary by
        # the Base44 ingesters so the rewrite stage is never a black box.
        "condensation_stats": condensation_stats or None,
        # NOTE: there is deliberately NO top-level unresolved_content /
        # unresolved_content_count / publishable / review_required field. An
        # all-untimed group is adjudicated into a canonical UNRESOLVED_UNTIMED_
        # CONTENT issue inside result.segmentation_qc — the SINGLE publication
        # authority. Segmentation QC is the one source of truth; the transient
        # unresolved groups never leave the formatter. SOC 2 CC8.1.
    }

    if "srt" in output_formats:
        result["srt"] = _export_srt(cues)
    if "vtt" in output_formats:
        result["vtt"] = _export_vtt(cues)
    if "scc" in output_formats:
        scc = _export_scc(cues)
        if scc:
            result["scc"] = scc
    if "ttml" in output_formats:
        fr = _env_str("TTML_FRAME_RATE", "30")
        frm = _env_str("TTML_FRAME_RATE_MULTIPLIER", "1000 1001")
        tb = _env_str("TTML_TIMEBASE", "media")
        ta = _env_str("TTML_TEXT_ALIGN", "center")
        result["ttml"] = _export_ttml(cues, fr, frm, tb, ta)

    return result


# ─── Cue Building ───────────────────────────────────────────────────

def _build_cues_from_tokens(
    spoken_tokens: List[Dict[str, Any]],
    sound_tokens: List[Dict[str, Any]],
    max_lines: int,
    max_chars: int,
    audio_events: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build caption cues from spoken tokens, in-text sound tokens, AND the
    provider's native audio_events[] array.

    Returns (timed_cues, unresolved_groups) — the transient unresolved groups
    are consumed by Segmentation QC only, never merged into the cue list.

    Two non-dialogue sources, both rendered through the per-spec renderer:
      1. sound_tokens — bracketed tags found inside utterance text ([MUSIC]).
      2. audio_events — the structured {event_type,start,end} array Scribe v2
         emits natively (and the baseline carries verbatim). These were
         previously dropped on the floor — the "no sound cues" regression.

    Cue `type` is normalised to 'music' for music-family events and
    'sound_effect' otherwise, so the Base44 ingester maps them to the correct
    cue_type. Rendering (♪ vs [BRACKETED]) is decided purely by the spec's
    MUSIC_CUE_FORMAT / SOUND_EFFECT_FORMAT — the engine never names a client.
    """
    audio_events = audio_events or []

    # Build dialogue cues from spoken tokens. Returns (timed_cues,
    # unresolved_groups) — the transient unresolved channel carries all-untimed
    # groups that must never become timed cues. Threaded to process_caption_job
    # ONLY to feed Segmentation QC (the single publication authority); never
    # exposed on the external result.
    dialogue_cues, unresolved_groups = _build_dialogue_cues(spoken_tokens, max_lines, max_chars)

    sound_cues: List[Dict[str, Any]] = []

    # (1) In-text bracketed sound tokens. Re-render through the spec renderer
    # so even legacy in-text tags honour the spec's music/sfx convention.
    for token in sound_tokens:
        inner = (token.get("text") or "").strip().strip("[]").strip()
        et = "music" if "music" in inner.lower() else "sound_effect"
        rendered = _render_audio_event_text(inner, et)
        if not rendered:
            continue
        sound_cues.append({
            "start_ms": token["start_ms"],
            "end_ms": max(token["end_ms"], token["start_ms"] + _min_sound_ms()),
            "lines": [rendered],
            "type": "music" if et == "music" else "sound_effect",
            "meta": {},
        })

    # (2) Native structured audio_events — the source we used to drop.
    for ev in audio_events:
        start = ev.get("start", ev.get("start_ms"))
        end = ev.get("end", ev.get("end_ms"))
        event_type = ev.get("event_type") or ev.get("type") or "sound"
        if start is None or end is None:
            continue
        try:
            start = int(start)
            end = int(end)
        except (TypeError, ValueError):
            continue
        if end <= start:
            end = start + _min_sound_ms()
        label = (ev.get("text") or event_type).replace("_", " ")
        rendered = _render_audio_event_text(label, event_type)
        if not rendered:
            continue
        is_music = "music" in (event_type or "").lower()
        sound_cues.append({
            "start_ms": start,
            "end_ms": max(end, start + _min_sound_ms()),
            "lines": [rendered],
            "type": "music" if is_music else "sound_effect",
            "meta": {"native_audio_event": True, "event_type": event_type},
        })

    # Merge dialogue + sound cues by start time. Dialogue sorts before a
    # co-incident sound cue so on-screen reading order stays natural.
    all_cues = dialogue_cues + sound_cues
    all_cues.sort(key=lambda c: (c["start_ms"], 0 if c["type"] == "dialogue" else 1))

    # Re-index
    for i, cue in enumerate(all_cues):
        cue["idx"] = i + 1

    # Return the merged TIMED cue list AND the transient unresolved-group
    # collection. Unresolved groups are NEVER placed into all_cues — the
    # exporters serialize all_cues directly, so keeping them out is what
    # guarantees no invalid (untimed) cue can ever reach SRT/VTT/SCC/TTML.
    return all_cues, unresolved_groups


def _build_dialogue_cues(
    tokens: List[Dict[str, Any]],
    max_lines: int,
    max_chars: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build dialogue cues from spoken tokens — PROFESSIONAL-GRADE pipeline.
    Returns (timed_cues, unresolved_groups). The unresolved groups are the
    transient all-untimed groups fed to Segmentation QC (single authority).

    Order (the industry-correct ordering — Iyuno / Pixelogic / Zoo / Deluxe):
      1. SEGMENT the word stream into true SENTENCE GROUPS first
         (abbreviation-aware — "Mr. Wang" is never split). A new group starts
         only on a real sentence end or a speaker change. This is what stops
         the "Wang." orphan at the source.
      2. PACK whole sentences into cues, fitting the spec's char/line budget.
         A cue holds as many whole sentences as fit on screen; a sentence is
         split across cues ONLY when it genuinely overflows the budget, and
         then only at a clause boundary that never leaves a one-word tail.
      3. Orphan reflow (a final safety net) runs later in the master pipeline.

    Sentence segmentation and spec-fitting are co-solved here — you cannot fit
    to the line budget without knowing sentence boundaries, and you cannot
    finalise boundaries without knowing the budget. Doing them as one pass is
    why this produces "Good afternoon." / "You must be Mr. Wang." instead of
    the orphaned fragment the naive per-word splitter produced.
    """
    if not tokens:
        return [], []

    try:
        from .segmentation import (
            segment_into_sentence_groups,
            build_unresolved_group as _build_unresolved_group,
        )
    except Exception:
        # Defensive: if the segmentation module can't import, fall back to a
        # single-group-per-speaker grouping so we never crash a job. This
        # path should never fire in production. The fallback also provides a
        # minimal unresolved-group factory so the router contract still holds.
        def _build_unresolved_group(**kwargs):
            return dict(kwargs)

        def segment_into_sentence_groups(toks):
            return [{
                "words": [t.get("text", "") for t in toks],
                "start_ms": toks[0].get("start_ms", 0) if toks else 0,
                "end_ms": toks[-1].get("end_ms", 0) if toks else 0,
                "speaker_runs": [{"speaker": toks[0].get("speaker") if toks else None, "word_start": 0}],
            }] if toks else []

    sentence_groups = segment_into_sentence_groups(tokens)

    cues: List[Dict[str, Any]] = []
    # Accumulator for packing consecutive whole sentences (same speaker) into
    # one cue while they still fit the on-screen budget.
    pack_words: List[str] = []
    pack_start: Optional[int] = None
    pack_end: int = 0
    pack_runs: List[Dict[str, Any]] = []
    pack_speaker: Optional[str] = None
    # Whether the current pack is an unknown-speaker (review-required) run — it
    # must never absorb, or be absorbed by, a known-speaker group.
    pack_review_required: bool = False
    # Whether the current pack OPENED at a hard inter-utterance pause boundary.
    # Stamped onto the finalized cue's meta so the sequence optimizer's window
    # collector treats it as an immutable wall (no window may span it).
    pack_pause_boundary: bool = False
    # Bounded pause provenance for a pack that opened at a hard pause: the
    # group's source utterance id, prior utterance id, and measured gap (ms).
    # None on a pack that did not open at a hard pause. Item-4 audit contract.
    pack_pause_prov: Optional[Dict[str, Any]] = None

    budget = max_chars * max_lines

    # ── UNRESOLVED (ALL-UNTIMED) GROUP CHANNEL — pure routing, zero mutation ─
    # A group with timing_unresolved=True (segmentation could resolve NO window
    # because EVERY token in it was untimed) has NO trustworthy timing. We do NOT
    # invent one: no 0ms, no start==end, no anchor to the previous/next cue, no
    # evidence-free interpolation. The group NEVER enters the timed-cue path, so it
    # can never be serialized into SRT/VTT/SCC/TTML.
    #
    # OWNERSHIP (the monotonic provenance chain): SEGMENTATION owns the immutable
    # unresolved-group SCHEMA (segmentation.build_unresolved_group) — it stamps the
    # absolute source_word_start/end, carries the normalizer's transcription
    # provenance, and assigns the intrinsic segmentation_group_index. The FORMATTER
    # is a PURE ROUTER: it forwards each segmentation-emitted group VERBATIM into
    # Segmentation QC. It adds no field, removes no field, reinterprets no field —
    # and no longer tracks any word cursor (segmentation owns the offsets).
    # Segmentation QC adjudicates each group into ONE canonical
    # UNRESOLVED_UNTIMED_CONTENT fail issue (the single publication authority) and
    # assigns its own ug#N routing ordinal independently. SOC 2 CC8.1 / FCC §79.1.
    unresolved_groups: List[Dict[str, Any]] = []

    def _flush_pack() -> None:
        nonlocal pack_words, pack_start, pack_end, pack_runs, pack_speaker
        nonlocal pack_review_required, pack_pause_boundary, pack_pause_prov
        if pack_words:
            cue = _finalize_dialogue_cue(
                pack_words, pack_start, pack_end, pack_runs, max_lines, max_chars,
                word_timings=_word_timings_in_window(tokens, pack_start, pack_end),
                review_required=pack_review_required,
            )
            if pack_pause_boundary:
                cue["meta"]["pause_boundary_before"] = True
                if pack_pause_prov is not None:
                    cue["meta"]["pause_provenance"] = pack_pause_prov
            cues.append(cue)
        pack_words = []
        pack_start = None
        pack_end = 0
        pack_runs = []
        pack_speaker = None
        pack_review_required = False
        pack_pause_boundary = False
        pack_pause_prov = None

    for _group_index, group in enumerate(sentence_groups):
        # ── DIVERT AN UNRESOLVED (ALL-UNTIMED) GROUP — before ANY timing math ─
        # A group whose every token was untimed carries timing_unresolved=True and
        # start_ms=None. It has NO trustworthy window, so it must never reach cue
        # arithmetic (which would coerce None → 0) and must never become a timed
        # cue. We flush any pending pack first (so reading order is preserved:
        # unresolved content is recorded at its true position in the stream), then
        # record the group's text + provenance in the unresolved channel and skip
        # it. No fabricated timestamp is ever produced — not 0,0, not start==end,
        # not a neighbour anchor, not evidence-free interpolation. The text is
        # NEVER lost; it is surfaced separately and gates publication. This is the
        # untimed-token contract carried up to the group level. SOC 2 CC8.1 / FCC §79.1.
        if group.get("timing_unresolved") or group.get("start_ms") is None:
            _flush_pack()
            # PURE ROUTING: build the canonical immutable unresolved-group object
            # using SEGMENTATION's owned schema factory, populated ENTIRELY from
            # segmentation-provided fields (absolute word offsets, provenance,
            # speaker, utterance id). The formatter computes nothing here — it does
            # not track a word cursor, does not enrich provenance, does not
            # reinterpret any field. The segmentation_group_index is the group's
            # absolute index within segmentation's output (enumerate index below).
            unresolved_groups.append(_build_unresolved_group(
                words=list(group.get("words") or []),
                source_word_start=group.get("source_word_start", 0),
                source_word_end=group.get("source_word_end", 0),
                speaker=group.get("speaker"),
                source_utterance_id=group.get("source_utterance_id"),
                provenance=group.get("transcription_provenance"),
                segmentation_group_index=_group_index,
                reason="all_tokens_untimed",
            ))
            continue
        g_words = group["words"]
        g_text = " ".join(g_words)
        # EXPLICIT speaker ownership — read the group's OWN speaker, never a
        # neighbour's. segmentation now guarantees a non-empty group always
        # carries exactly one materialized speaker_run, so g_speaker is reliable
        # (it was the None-from-empty-runs value here that drove the historical
        # cross-speaker fusion). g_review_required marks an unknown-speaker group
        # that must NEVER be merged into an adjacent known-speaker cue.
        g_speaker = group.get("speaker", (group.get("speaker_runs") or [{}])[0].get("speaker"))
        g_speaker_known = bool(group.get("speaker_known", g_speaker is not None))
        g_review_required = bool(group.get("review_required", not g_speaker_known))
        # IMMUTABLE PAUSE BOUNDARY — the group opens after a ≥pause_boundary_ms
        # silence between distinct source utterances. A cue may never span it.
        g_hard_boundary_before = bool(group.get("hard_boundary_before", False))
        g_is_cjk = _is_cjk(g_text)

        # The on-screen budget. For CJK the budget is measured in CHARACTERS
        # (no spaces); a Japanese caption packs max_chars × max_lines glyphs.
        g_len = _cjk_count(g_text) if g_is_cjk else len(g_text)

        # A sentence that itself overflows the on-screen budget must be split
        # across multiple cues. CJK splits by character at 、。 clause boundaries
        # (kinsoku-aware); Latin splits at clause boundaries by word. Both
        # interpolate timings proportionally over the group's real window.
        #
        # NOTE on label-aware fit: this is the COARSE body-only first pass. The
        # authoritative delivered-fit (speaker-label-inclusive) enforcement is the
        # caption-shaping stage (services.shaping), which runs immediately after
        # this and owns the ONE shared cue_fits_delivered primitive plus clause-
        # AND word-phrase splitting with a fixed-point loop. Keeping ONE owner for
        # label-aware splitting avoids two stages half-solving it and drifting.
        if g_len > budget:
            # SHORT-LEAD ABSORPTION (universal): a tiny same-speaker sentence
            # sitting in the pack ("By who?") must ride INTO the split of the
            # following over-budget sentence instead of being flushed as its
            # own micro-cue with a micro window. The chunker then packs
            # "By who? Gus from the body shop…" naturally, splitting at real
            # clause/sentence boundaries. Latin only (CJK packs by glyph).
            # Short-lead absorption is FORBIDDEN across a hard pause boundary,
            # into/out of an unknown-speaker group, or across a speaker change —
            # the same immutable invariants the main packing branch enforces.
            absorbed_start: Optional[int] = None
            if (pack_words and not g_is_cjk and pack_speaker == g_speaker
                    and g_speaker_known and pack_speaker is not None
                    and not g_hard_boundary_before
                    and len(" ".join(pack_words)) <= max_chars):
                g_words = pack_words + g_words
                absorbed_start = pack_start
                pack_words = []
                pack_start = None
                pack_end = 0
                pack_runs = []
                pack_speaker = None
            else:
                _flush_pack()
            g_start = absorbed_start if absorbed_start is not None else group["start_ms"]
            g_end = max(group["end_ms"], g_start + 1)
            span = g_end - g_start

            if g_is_cjk:
                # Split the whole CJK sentence string into cue-sized character
                # chunks (each ≤ budget), then interpolate by character count.
                cjk_str = "".join(g_words) if len(g_words) > 1 else g_text
                chunk_strs = _split_cjk_sentence_into_cue_chunks(cjk_str, max_chars, max_lines)
                total_chars = sum(_cjk_count(c) for c in chunk_strs) or 1
                char_cursor = 0
                for c_str in chunk_strs:
                    c_start = g_start + (span * char_cursor) // total_chars
                    char_cursor += _cjk_count(c_str)
                    c_end = g_start + (span * char_cursor) // total_chars
                    cues.append(_finalize_dialogue_cue(
                        [c_str], int(c_start), int(max(c_end, c_start + 1)),
                        [{"speaker": g_speaker, "word_start": 0}], max_lines, max_chars,
                        word_timings=_word_timings_in_window(tokens, int(c_start), int(max(c_end, c_start + 1))),
                    ))
            else:
                chunk_list = _split_sentence_into_cue_chunks(g_words, max_chars, max_lines)
                total_words = sum(len(c["words"]) for c in chunk_list) or 1
                word_cursor = 0
                for chunk in chunk_list:
                    c_words = chunk["words"]
                    c_start = g_start + (span * word_cursor) // total_words
                    word_cursor += len(c_words)
                    c_end = g_start + (span * word_cursor) // total_words
                    cues.append(_finalize_dialogue_cue(
                        c_words, int(c_start), int(max(c_end, c_start + 1)),
                        [{"speaker": g_speaker, "word_start": 0}], max_lines, max_chars,
                        word_timings=_word_timings_in_window(tokens, int(c_start), int(max(c_end, c_start + 1))),
                    ))
            continue

        # Packing: would adding this whole sentence to the current pack still
        # fit, and is it the same speaker? If yes, append; else flush + start new.
        #
        # SPEAKER-INTEGRITY INVARIANT (Option A — universal, every spec):
        # A finalized dialogue cue must NEVER silently contain words from two
        # different speakers without the renderer being told. The packer
        # therefore flushes on ANY speaker change — including the case where
        # the current pack's speaker is known and the incoming group's speaker
        # is unknown/null (or vice-versa). Treating null≠known as a change is
        # what stops "Speaker A sentence" + "Speaker B sentence" fusing into a
        # single flat, dash-less line when diarization is imperfect. A
        # single-speaker cue is the guaranteed unit; intentional two-speaker
        # dash captions are produced afterwards by group_two_speaker_cues in
        # readability.py, which works ONLY on clean single-speaker cues.
        # A speaker change is ANY difference in known speaker OR a
        # known↔unknown transition. Because segmentation now guarantees explicit
        # per-group ownership, this compares real values, not a None-from-empty-
        # runs artifact. An unknown-speaker group (g_review_required) also always
        # flushes so it can never be absorbed into a known-speaker cue.
        speaker_changed = bool(pack_words) and (
            g_speaker != pack_speaker or g_review_required or pack_review_required
        )
        # CJK packs by joining with no space (no inter-word spaces in Japanese);
        # Latin joins by space. Length is measured the same way the budget is.
        if g_is_cjk:
            candidate = ("".join(pack_words + g_words)).strip() if pack_words else g_text
            would_overflow = _cjk_count(candidate) > budget
        else:
            candidate = (" ".join(pack_words + g_words)).strip() if pack_words else g_text
            would_overflow = len(candidate) > budget

        # IMMUTABLE PAUSE BOUNDARY — a group opened by a ≥pause_boundary_ms
        # inter-utterance silence ALWAYS forces a flush, so a cue can never span
        # the pause (the "Cookie?" fix). Evaluated before the overflow/speaker
        # tests so it can never be bypassed.
        if (g_hard_boundary_before or speaker_changed or would_overflow) and pack_words:
            _flush_pack()

        if pack_start is None:
            pack_start = group["start_ms"]
        # Pack ownership is materialized from the FIRST group in the pack and is
        # fixed for its lifetime (a change flushes above). pack_review_required
        # rides with it so a downstream reader can flag the cue for review;
        # pack_pause_boundary records that this pack opened at a hard pause so
        # the optimizer treats the cue's leading edge as an immutable wall.
        if not pack_words:
            pack_speaker = g_speaker
            pack_review_required = g_review_required
            pack_pause_boundary = g_hard_boundary_before
            # Carry the group's bounded pause provenance onto the pack when it
            # opened at a hard pause. Effective threshold is the resolved spec
            # value (segmentation.pause_boundary_ms). Item-4 audit contract —
            # bounded scalars only, never raw words. SOC 2 CC8.1 / FCC §79.1.
            if g_hard_boundary_before:
                try:
                    from .segmentation import pause_boundary_ms as _pbm
                    _threshold = _pbm()
                except Exception:
                    _threshold = 1200
                _gap = group.get("gap_before_ms")
                pack_pause_prov = {
                    "source_utterance_id": group.get("source_utterance_id"),
                    "prev_utterance_id": group.get("prev_utterance_id"),
                    "measured_gap_ms": int(_gap) if _gap is not None else None,
                    "effective_pause_boundary_ms": int(_threshold),
                    "boundary_reason": "source_utterance_pause",
                    "immutable_boundary": True,
                }
            pack_runs.append({"speaker": g_speaker if g_speaker_known else None,
                              "word_start": 0})
        pack_words.extend(g_words)
        pack_end = group["end_ms"]

    _flush_pack()
    # Return the timed cues AND the transient unresolved-group collection. The
    # caller (_build_cues_from_tokens → process_caption_job) feeds the groups into
    # Segmentation QC (the single publication authority); it NEVER merges them
    # into the timed cue list and NEVER exposes them on the external result.
    # SOC 2 CC8.1.
    return cues, unresolved_groups


# Phrase-boundary word classes — the SAME linguistic tables the line-breaker
# and shaper use, so the CUE-level balanced split ranks boundaries identically
# (break BEFORE a preposition/conjunction/article, never strand one at a chunk
# end). Shared import keeps every split path in the engine consistent.
try:
    from .linebreak import _LEADING_WORDS as _BAL_LEADING, _DETERMINERS as _BAL_DET, _bare as _bal_bare
except Exception:  # pragma: no cover
    _BAL_LEADING = frozenset()
    _BAL_DET = frozenset()

    def _bal_bare(w):
        return (w or "").strip(".,;:!?\"')]}").lower()

_BAL_SENT_END = (".", "!", "?")
_BAL_CLAUSE_END = (",", ";", ":", "—", "–")


def _balanced_two_way_split(
    words: List[str],
    budget: int,
) -> Optional[tuple]:
    """Split `words` into TWO balanced, phrase-aware halves for the common
    "sentence needs exactly two cues" case. Returns (left_words, right_words)
    or None when no boundary gives two halves that EACH fit one cue's budget
    (caller then falls through to the greedy multi-chunk splitter).

    Scoring mirrors linebreak.choose_two_line_break at the CUE level:
      + reward balance (small |len(left) - len(right)|)
      + strong reward for breaking AFTER sentence punctuation, then clause punct
      + reward the right half LEADING with a conjunction/preposition/article
      - penalize a left half ENDING on a stranded function word
    A ≥3-word tiny-side guard (same threshold as every other picker) keeps a
    3-char fragment from becoming its own cue. Deterministic; ties → earlier
    break. SOC 2 CC8.1."""
    n = len(words)
    if n < 4:
        return None

    def _llen(start: int, end: int) -> int:
        if start >= end:
            return 0
        seg = words[start:end]
        return sum(len(w) for w in seg) + (len(seg) - 1)

    best_i: Optional[int] = None
    best_score = float("-inf")
    for i in range(1, n):
        len1 = _llen(0, i)
        len2 = _llen(i, n)
        # Each half must fit a single cue (max_chars × max_lines).
        if len1 > budget or len2 > budget:
            continue
        # Tiny-side guard — never strand a <3-word fragment as its own cue.
        if min(i, n - i) < 3:
            continue
        last = words[i - 1]
        first = words[i]
        score = -abs(len1 - len2) * 1.0
        if last.rstrip().endswith(_BAL_SENT_END):
            score += 32.0
        elif last.rstrip().endswith(_BAL_CLAUSE_END):
            score += 28.0
        if _bal_bare(first) in _BAL_LEADING or _bal_bare(first) in _BAL_DET:
            score += 15.0
        if _bal_bare(last) in _BAL_LEADING or _bal_bare(last) in _BAL_DET:
            score -= 25.0
        if score > best_score:
            best_score = score
            best_i = i

    if best_i is None:
        return None
    return (words[:best_i], words[best_i:])


def _split_sentence_into_cue_chunks(
    words: List[str],
    max_chars: int,
    max_lines: int,
) -> List[Dict[str, Any]]:
    """
    Split a single over-long sentence into multiple cue-sized chunks at the
    best clause/phrase boundary, never leaving a one-word orphan tail.

    Greedy fill up to (max_chars * max_lines), but back off to the last clause
    boundary (comma, semicolon, colon, dash) inside the window when one exists,
    and guarantee the final chunk is never a lone word — if the tail would be a
    single word, pull a word back from the previous chunk.

    Returns [{ "words": [...] }, ...]. The caller interpolates real timecodes
    across the returned chunks proportionally to word count over the group's
    actual [start_ms, end_ms] window.
    """
    budget = max_chars * max_lines
    _CLAUSE_END = (",", ";", ":", "—", "–")
    _SENT_END = (".", "!", "?")
    # MIN-CHUNK GUARD (2026-07-06, Pluto 0062 sliver defect): never back off to
    # a boundary that strands a tiny head chunk. "No, I'm not gonna call…" must
    # NOT become a 3-char "No," cue with a ~0.2s proportional window — the
    # boundary after "No," is grammatically real but editorially unusable as a
    # standalone caption. A back-off boundary is eligible only when the head it
    # produces carries at least MIN_CHUNK_WORDS words.
    MIN_CHUNK_WORDS = 3

    # ── BALANCED TWO-WAY SPLIT (2026-07-10, PBS 0003 defect) ─────────────────
    # When a sentence overflows ONE cue but fits in TWO (len ≤ 2×budget), a
    # professional captioner BALANCES the two cues at the best phrase boundary
    # — NOT greedy-fill cue 1 to the brim and dump the short remainder into cue
    # 2. The old greedy loop turned
    #   "Now, today's story is about a father who was obsessed with buying antiques."
    # into a lopsided 58ch chunk + a 16ch "buying antiques." tail; the shaper
    # then re-split the 58ch chunk, yielding a 3-way fragment mess with a
    # 1-line, over-CPS "Now, today's story is" fail-cue. Balancing splits it into
    # two clean ~37ch cues ("...about a father" | "who was obsessed with buying
    # antiques.") — "buying antiques." rides with its phrase, exactly the
    # expected professional output. This mirrors the balanced+clause-aware
    # scoring the LINE breaker (linebreak.choose_two_line_break) already uses,
    # applied one level up at the CUE boundary. Longer sentences (3+ cues) fall
    # through to the greedy clause-backoff loop below unchanged. SOC 2 CC8.1 —
    # deterministic function of phrase structure, identical every run.
    if budget < len(" ".join(words)) <= 2 * budget and len(words) >= 4:
        two = _balanced_two_way_split(words, budget)
        if two is not None:
            return [{"words": two[0]}, {"words": two[1]}]

    chunks: List[List[str]] = []
    cur: List[str] = []
    for w in words:
        test = (" ".join(cur + [w])).strip()
        if len(test) <= budget or not cur:
            cur.append(w)
        else:
            # Back off to the last clause OR sentence boundary inside `cur`
            # whose head chunk is at least MIN_CHUNK_WORDS words. Sentence
            # enders are included so an absorbed short lead ("By who?") ends
            # its chunk at the sentence break, never mid-phrase.
            split_at = None
            for i in range(len(cur) - 1, 0, -1):
                tok = cur[i].rstrip()
                if (tok.endswith(_CLAUSE_END) or tok.endswith(_SENT_END)) \
                        and (i + 1) >= MIN_CHUNK_WORDS:
                    split_at = i + 1
                    break
            if split_at and split_at < len(cur):
                chunks.append(cur[:split_at])
                cur = cur[split_at:] + [w]
            else:
                chunks.append(cur)
                cur = [w]
    if cur:
        chunks.append(cur)

    # Orphan guard: never leave a single-word final chunk — pull the last word
    # of the previous chunk forward so the tail reads as a phrase.
    if len(chunks) >= 2 and len(chunks[-1]) == 1 and len(chunks[-2]) > 1:
        moved = chunks[-2].pop()
        chunks[-1] = [moved] + chunks[-1]

    return [{"words": c} for c in chunks]


def _split_cjk_sentence_into_cue_chunks(
    sentence: str,
    max_chars: int,
    max_lines: int,
) -> List[str]:
    """Split an over-long CJK sentence into cue-sized character chunks, each ≤
    max_chars × max_lines characters, breaking at 、。clause boundaries where
    possible (kinsoku-aware). Returns a list of CJK strings — each becomes one
    cue, then wrap_cjk lays it out across ≤max_lines lines. This is the CJK twin
    of _split_sentence_into_cue_chunks; it splits by CHARACTER, never by space."""
    from .cjk import CJK_CLAUSE_ENDERS, CJK_SENTENCE_ENDERS, cjk_char_count

    budget = max_chars * max_lines
    s = (sentence or "").strip()
    if cjk_char_count(s) <= budget:
        return [s]

    chunks: List[str] = []
    remaining = s
    while cjk_char_count(remaining) > budget:
        window = remaining[:budget]
        # Prefer the last clause/sentence boundary inside the window so the cue
        # ends on a natural pause (、 or 。). Break AFTER the punctuation.
        cut = -1
        for i in range(len(window)):
            if window[i] in CJK_CLAUSE_ENDERS or window[i] in CJK_SENTENCE_ENDERS:
                cut = i + 1
        if cut < 1:
            cut = budget  # no boundary — hard-cut at the budget
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return [c for c in chunks if c]


def _word_timings_in_window(
    tokens: List[Dict[str, Any]],
    start_ms: int,
    end_ms: int,
) -> List[Dict[str, Any]]:
    """Slice the per-word spoken tokens whose timing falls inside this cue's
    [start_ms, end_ms] window. These real word timings are attached to the cue's
    meta so the universal shaping stage can split at FRAME-FAITHFUL boundaries
    (the enterprise-correct answer) instead of interpolating. Positional +
    time-window match — no dependency on segmentation carrying timings through.
    Returns [] when no tokens match (shaper then interpolates).

    GENERAL TIMING CONTRACT: an UNTIMED token (has_timing() False — its timing was
    withheld upstream, e.g. a quarantined repair disposition, but the formatter
    neither knows nor cares WHY) is EXCLUDED from the word-timing slice — it has no
    trustworthy timing, so it must never anchor a shaper split boundary. Its TEXT
    still reaches the cue via the sentence group (segmentation keeps the word in
    order); only its (absent) timing is withheld. The has_timing() guard runs
    BEFORE any int() read, so a None timestamp is never coerced to 0. SOC 2 CC8.1 /
    FCC §79.1."""
    out = []
    for t in tokens:
        if not _has_timing(t):
            continue  # untimed — never a split anchor, never coerced to 0
        ts = int(t.get("start_ms", t.get("start")) or 0)
        te = int(t.get("end_ms", t.get("end")) or ts)
        # A token belongs to this cue when its midpoint sits in the window.
        mid = (ts + te) // 2
        if start_ms <= mid <= end_ms:
            out.append({"text": t.get("text", ""), "start_ms": ts, "end_ms": te})
    return out


def _finalize_dialogue_cue(
    words: List[str],
    start_ms: int,
    end_ms: int,
    speaker_runs: List[Dict[str, Any]],
    max_lines: int,
    max_chars: int,
    word_timings: Optional[List[Dict[str, Any]]] = None,
    review_required: bool = False,
) -> Dict[str, Any]:
    """Wrap words into lines respecting max_chars and max_lines, then apply
    the deterministic speaker-render baseline (dash / off-camera labels).

    The editorial-AI pass refines this further, but the DETERMINISTIC render
    here guarantees speaker identification is correct even when AI is skipped,
    times out, or hits its run budget — closing the "speaker IDs gone" gap.
    UNIVERSAL one-speaker-per-line rule is enforced for multi-speaker groups
    regardless of label mode (never two speakers on one line)."""
    # CJK joins with NO space (Japanese has no inter-word spaces); a stray space
    # between sentences would inflate the char count and read wrong. Latin joins
    # with a single space. render_lines/wrap_text route CJK to wrap_cjk via the
    # same is_cjk_text check, so the dialogue_text built here stays consistent.
    if len(words) == 1:
        dialogue_text = words[0]
    elif _is_cjk(" ".join(words)):
        dialogue_text = "".join(words)
    else:
        dialogue_text = " ".join(words)
    # SINGLE SOURCE OF TRUTH — render through the shared module so the cue is
    # drawn the exact same way the readability merge passes will re-draw it.
    lines = _render_lines(words, speaker_runs, max_lines, max_chars, dialogue_text)

    return {
        "idx": 0,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "lines": lines,
        "type": "dialogue",
        "meta": {
            "dialogue_text": dialogue_text,
            "runs": speaker_runs,
            # Real per-word timings for THIS cue's window (when available). The
            # universal shaping stage prefers these for frame-faithful splits;
            # empty list → shaper interpolates. SOC 2 CC8.1 — timing provenance.
            "word_timings": word_timings or [],
            # SPEAKER-OWNERSHIP REVIEW FLAG — True when this cue's source group
            # had no known speaker. Carried so the optimizer/readability never
            # merge it into a known-speaker cue and the ingester can mark it
            # review-required. SOC 2 CC8.1 — an unknown speaker is never silently
            # defaulted to a neighbour.
            "review_required": bool(review_required),
        },
    }
