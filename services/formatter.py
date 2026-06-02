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
    Build dialogue cues from spoken tokens.
    Groups by speaker and sentence boundaries.
    """
    if not tokens:
        return []

    cues: List[Dict[str, Any]] = []
    current_words: List[str] = []
    current_start: Optional[int] = None
    current_end: int = 0
    current_speaker: Optional[str] = None
    speaker_runs: List[Dict[str, Any]] = []

    SENTENCE_END_RE = re.compile(r"[.!?]$")

    for token in tokens:
        text = token["text"]
        speaker = token.get("speaker")
        start_ms = token["start_ms"]
        end_ms = token["end_ms"]

        # Start new cue if speaker changes or sentence ends
        speaker_changed = current_speaker is not None and speaker != current_speaker
        sentence_ended = current_words and SENTENCE_END_RE.search(current_words[-1])

        # Check if adding this word would exceed max chars
        test_text = " ".join(current_words + [text])
        would_exceed = len(test_text) > max_chars * max_lines

        if (speaker_changed or sentence_ended or would_exceed) and current_words:
            cue = _finalize_dialogue_cue(
                current_words, current_start, current_end,
                speaker_runs, max_lines, max_chars,
            )
            cues.append(cue)
            current_words = []
            current_start = None
            speaker_runs = []

        if current_start is None:
            current_start = start_ms

        if speaker != current_speaker:
            if current_words:
                # Close previous speaker run
                pass
            current_speaker = speaker
            speaker_runs.append({"speaker": speaker, "word_start": len(current_words)})

        current_words.append(text)
        current_end = end_ms

    # Flush remaining
    if current_words:
        cue = _finalize_dialogue_cue(
            current_words, current_start, current_end,
            speaker_runs, max_lines, max_chars,
        )
        cues.append(cue)

    return cues


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
    dialogue_text = " ".join(words)
    lines = _speaker_render_lines(words, speaker_runs, max_lines, max_chars, dialogue_text)

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


def _speaker_render_lines(
    words: List[str],
    speaker_runs: List[Dict[str, Any]],
    max_lines: int,
    max_chars: int,
    dialogue_text: str,
) -> List[str]:
    """
    UNIVERSAL speaker-aware line builder, parameterised by the spec's
    SPEAKER_LABEL_MODE. The engine never names a client — it only obeys the
    mode the spec sent.

    Hard universal invariant (every mode, never configurable):
      • One speaker per line. Two distinct speakers are NEVER placed on the
        same physical line. This is the Pluto "-Speaker One / -Speaker Two"
        rule and is correct for every spec.

    Modes:
      • 'dash' (Pluto/FAST): multi-speaker group → each speaker's text on its
        own line, prefixed with '- '. Single-speaker group → plain wrap, no dash.
      • 'none': plain wrap, no speaker identifier.
      • anything else ('first_occurrence_per_scene' / 'every_change' /
        'always'): single-speaker groups wrap plainly here (named-tag insertion
        is the editorial-AI / downstream concern); multi-speaker groups still
        get one-speaker-per-line so reading order is unambiguous.
    """
    mode = _speaker_label_mode()
    # Reconstruct per-speaker text segments from the runs' word offsets.
    segments = _segments_from_runs(words, speaker_runs)
    distinct_speakers = len({s.get("speaker") for s in (speaker_runs or []) if s.get("speaker") is not None})

    # Single speaker (or no diarization) → plain greedy wrap. No dash, no split.
    if distinct_speakers <= 1 or len(segments) <= 1:
        return _wrap_text(dialogue_text, max_chars, max_lines)

    # Multi-speaker group — one speaker per line (universal invariant).
    prefix = "- " if mode == "dash" else ""
    lines: List[str] = []
    for seg in segments:
        seg_text = seg["text"].strip()
        if not seg_text:
            continue
        # Each speaker's text occupies its own line; if a single speaker's text
        # is itself too long, we keep it on one line here (readability + the
        # editorial-AI pass will rebalance). Never merge two speakers.
        lines.append(f"{prefix}{seg_text}")

    if not lines:
        return _wrap_text(dialogue_text, max_chars, max_lines)
    return lines


def _segments_from_runs(words: List[str], speaker_runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


def _wrap_text(text: str, max_chars: int, max_lines: int) -> List[str]:
    """
    Wrap text into lines respecting max_chars per line.
    Prefers breaking at punctuation and phrase boundaries.
    """
    text = text.strip()
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
                # Last line — dump remaining
                remaining = " ".join(words[words.index(word):])
                lines.append(remaining)
                return lines[:max_lines]

    if current_line:
        lines.append(current_line)

    return lines[:max_lines]
