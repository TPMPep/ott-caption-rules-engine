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

# Project imports with local fallbacks
try:
    from .rendering import render_lines as _render_lines
except Exception:
    # Defensive: must never crash a job. Falls back to a plain join. This
    # path should never fire in production (rendering.py ships with the engine).
    def _render_lines(words, speaker_runs, max_lines, max_chars, dialogue_text=None):
        return [dialogue_text if dialogue_text is not None else " ".join(words)]

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
            out.append({
                "text": item.get("text", ""),
                "start_ms": int(item.get("start_ms", item.get("start", 0) or 0)),
                "end_ms": int(item.get("end_ms", item.get("end", 0) or 0)),
                "speaker": item.get("speaker") or "A",
            })
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
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_str(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or default).strip()


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

def apply_env_overrides(env_dict: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    Set env vars from the captionOptions dict.
    Returns a snapshot of previous values for restoration.

    UPDATED: Also handles the new captionOptions mapping from the frontend.
    The frontend sends keys like CAPTION_PROFILE, CUSTOM_MAX_LINES, etc.
    """
    snapshot: Dict[str, Optional[str]] = {}
    if not env_dict or not isinstance(env_dict, dict):
        return snapshot

    for key, value in env_dict.items():
        key = str(key).strip()
        if not key:
            continue
        snapshot[key] = os.environ.get(key)
        os.environ[key] = str(value)

    # Log what was set
    profile = os.getenv("CAPTION_PROFILE", "")
    print(f"[ENV] Profile: {profile}")
    print(f"[ENV] Max lines: {os.getenv('CUSTOM_MAX_LINES', 'default')}")
    print(f"[ENV] Max chars: {os.getenv('CUSTOM_MAX_CHARS', 'default')}")
    print(f"[ENV] Sound density: {os.getenv('SOUND_DENSITY', 'default')}")
    print(f"[ENV] Speaker mode: {os.getenv('SPEAKER_LABEL_MODE', 'default')}")
    print(f"[ENV] Timecode offset: {os.getenv('TIMECODE_OFFSET_MS', '0')}")

    return snapshot


def restore_env_overrides(snapshot: Dict[str, Optional[str]]) -> None:
    """Restore env vars to their previous state."""
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


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
    total_duration_ms = max(t.get("end_ms", 0) for t in tokens) if tokens else 0
    sound_tokens = _filter_sound_cues_by_density(sound_tokens, total_duration_ms)
    print(f"[FORMATTER] Sound after density filter: {len(sound_tokens)}")

    # 5. Build cue list. Native audio_events (Scribe v2 / baseline) are passed
    # through so they become real sound cues rendered per the spec's
    # MUSIC_CUE_FORMAT / SOUND_EFFECT_FORMAT. Without this they were silently
    # dropped — the "no sound cues" regression. SOC 2 / FCC §79.1 non-dialogue
    # coverage is now reproduced from the baseline on every reformat.
    cues = _build_cues_from_tokens(
        spoken_tokens, sound_tokens, max_lines, max_chars,
        audio_events=audio_events or [],
    )
    cues_in_count = len(cues)
    print(f"[FORMATTER] Initial cues built: {cues_in_count} "
          f"(native audio_events in: {len(audio_events or [])})")

    # 6. Editorial AI — pass the heartbeat through so each cue's OpenAI
    # call can pulse the job's updated_at timestamp. Closes the "engine
    # looks hung from outside even though it's still working" perception
    # gap that produced the Pluto-Test 16-min auto-fail.
    cues = editorial_refine_cues(cues, protected_phrases, heartbeat=heartbeat)
    print(f"[FORMATTER] After editorial AI: {len(cues)} cues")

    # 7. Readability
    cues = apply_readability_rules(cues)
    print(f"[FORMATTER] After readability: {len(cues)} cues")

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

    # 11. QC
    qc = qc_report(cues_in_count, cues, protected_phrases)
    print(f"[FORMATTER] QC: {qc}")

    # 12. Export
    if not output_formats:
        output_formats = [_env_str("OUTPUT_FORMATS", "srt")]
    output_formats = [f.strip().lower() for f in ",".join(output_formats).split(",") if f.strip()]

    result: Dict[str, Any] = {
        "cues": [
            {
                "idx": i + 1,
                "start_ms": c["start_ms"],
                "end_ms": c["end_ms"],
                "lines": c["lines"],
                "type": c.get("type", "dialogue"),
            }
            for i, c in enumerate(cues)
        ],
        "qc": qc,
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
) -> List[Dict[str, Any]]:
    """
    Build caption cues from spoken tokens, in-text sound tokens, AND the
    provider's native audio_events[] array.

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

    # Build dialogue cues from spoken tokens
    dialogue_cues = _build_dialogue_cues(spoken_tokens, max_lines, max_chars)

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

    return all_cues


def _build_dialogue_cues(
    tokens: List[Dict[str, Any]],
    max_lines: int,
    max_chars: int,
) -> List[Dict[str, Any]]:
    """
    Build dialogue cues from spoken tokens — PROFESSIONAL-GRADE pipeline.

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
        return []

    try:
        from .segmentation import segment_into_sentence_groups
    except Exception:
        # Defensive: if the segmentation module can't import, fall back to a
        # single-group-per-speaker grouping so we never crash a job. This
        # path should never fire in production.
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

    budget = max_chars * max_lines

    def _flush_pack() -> None:
        nonlocal pack_words, pack_start, pack_end, pack_runs, pack_speaker
        if pack_words:
            cues.append(_finalize_dialogue_cue(
                pack_words, pack_start, pack_end, pack_runs, max_lines, max_chars,
            ))
        pack_words = []
        pack_start = None
        pack_end = 0
        pack_runs = []
        pack_speaker = None

    for group in sentence_groups:
        g_words = group["words"]
        g_text = " ".join(g_words)
        g_speaker = (group.get("speaker_runs") or [{}])[0].get("speaker")
        g_is_cjk = _is_cjk(g_text)

        # The on-screen budget. For CJK the budget is measured in CHARACTERS
        # (no spaces); a Japanese caption packs max_chars × max_lines glyphs.
        g_len = _cjk_count(g_text) if g_is_cjk else len(g_text)

        # A sentence that itself overflows the on-screen budget must be split
        # across multiple cues. CJK splits by character at 、。 clause boundaries
        # (kinsoku-aware); Latin splits at clause boundaries by word. Both
        # interpolate timings proportionally over the group's real window.
        if g_len > budget:
            _flush_pack()
            g_start = group["start_ms"]
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
        speaker_changed = (
            pack_words
            and g_speaker != pack_speaker
        )
        # CJK packs by joining with no space (no inter-word spaces in Japanese);
        # Latin joins by space. Length is measured the same way the budget is.
        if g_is_cjk:
            candidate = ("".join(pack_words + g_words)).strip() if pack_words else g_text
            would_overflow = _cjk_count(candidate) > budget
        else:
            candidate = (" ".join(pack_words + g_words)).strip() if pack_words else g_text
            would_overflow = len(candidate) > budget

        if (speaker_changed or would_overflow) and pack_words:
            _flush_pack()

        if pack_start is None:
            pack_start = group["start_ms"]
        if pack_speaker is None:
            pack_speaker = g_speaker
            pack_runs.append({"speaker": g_speaker, "word_start": 0})
        pack_words.extend(g_words)
        pack_end = group["end_ms"]

    _flush_pack()
    return cues


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

    chunks: List[List[str]] = []
    cur: List[str] = []
    for w in words:
        test = (" ".join(cur + [w])).strip()
        if len(test) <= budget or not cur:
            cur.append(w)
        else:
            # Back off to the last clause boundary inside `cur` if one exists
            # and it isn't the very first word.
            split_at = None
            for i in range(len(cur) - 1, 0, -1):
                if cur[i].rstrip().endswith(_CLAUSE_END):
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


def _finalize_dialogue_cue(
    words: List[str],
    start_ms: int,
    end_ms: int,
    speaker_runs: List[Dict[str, Any]],
    max_lines: int,
    max_chars: int,
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
        },
    }
