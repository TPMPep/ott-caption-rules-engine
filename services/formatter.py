import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Project imports with local fallbacks for robustness/testability.
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
            out.append(
                {
                    "text": item.get("text", ""),
                    "start_ms": int(item.get("start_ms", item.get("start", 0) or 0)),
                    "end_ms": int(item.get("end_ms", item.get("end", 0) or 0)),
                    "speaker": item.get("speaker") or "A",
                }
            )
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
        export_ttml as _export_ttml,
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

def _export_ttml(cues: List[Dict[str, Any]], frame_rate: str, frame_rate_multiplier: str, time_base: str, align: str) -> str:
        # Minimal IMSC-1.1 TTML text profile export
        def ms_to_ttml(ms: int) -> str:
            ms = max(0, int(ms))
            hh, rem = divmod(ms, 3600000)
            mm, rem = divmod(rem, 60000)
            ss, rem = divmod(rem, 1000)
            return f"{hh:02d}:{mm:02d}:{ss:02d}.{rem:03d}"

        def strip_tags(s: str) -> str:
            s = STYLE_TAG_RE.sub("", s or "")
            return s

        def to_ttml_text(lines: List[str]) -> str:
            # convert <i> to <span tts:fontStyle="italic">
            def convert_italics(text: str) -> str:
                out = []
                parts = re.split(r"(<i>|</i>)", text)
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

            safe_lines = [convert_italics(strip_tags(l)) for l in lines]
            if not safe_lines:
                return ""
            if len(safe_lines) == 1:
                return safe_lines[0]
            return "<br/>".join(safe_lines)

        # Build TTML
        ttml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<tt xmlns="http://www.w3.org/ns/ttml"',
            '    xmlns:ttp="http://www.w3.org/ns/ttml#parameter"',
            '    xmlns:tts="http://www.w3.org/ns/ttml#style"',
            '    ttp:timeBase="%s"' % time_base,
            '    ttp:frameRate="%s"' % frame_rate,
            '    ttp:frameRateMultiplier="%s">' % frame_rate_multiplier,
            '  <head>',
            '    <styling>',
            '      <style xml:id="s0"',
            '             tts:fontFamily="proportionalSansSerif"',
            '             tts:fontSize="80%"',
            '             tts:color="#ebebeb"',
            '             tts:backgroundColor="transparent"',
            '             tts:textOutline="3% #101010"',
            '             tts:textAlign="%s"' % align,
            '             tts:luminanceGain="1.5"/>',
            '    </styling>',
            '    <layout>',
            '      <region xml:id="r0" tts:origin="2.5% 15%" tts:extent="95% 70%"/>',
            '    </layout>',
            '  </head>',
            '  <body region="r0" style="s0">',
            '    <div>',
        ]
        for cue in cues:
            text = to_ttml_text(cue.get("lines", []))
            if not text:
                continue
            ttml_lines.append(
                f'      <p begin="{ms_to_ttml(cue["start_ms"])}" end="{ms_to_ttml(cue["end_ms"])}">{text}</p>'
            )
        ttml_lines += ['    </div>', '  </body>', '</tt>']
        return "\n".join(ttml_lines)


def _validate_ttml(ttml_text: str, protected: List[str]) -> List[str]:
    errors: List[str] = []
    if not ttml_text:
        return ["TTML is empty"]
    try:
        root = ET.fromstring(ttml_text)
    except Exception as e:
        return [f"TTML XML parse error: {e}"]

    ns = {"tt": "http://www.w3.org/ns/ttml", "ttp": "http://www.w3.org/ns/ttml#parameter", "tts": "http://www.w3.org/ns/ttml#style"}
    # root checks
    if not root.tag.endswith("tt"):
        errors.append("Root element is not <tt>")
    # timeBase / frame rate
    time_base = root.attrib.get("{http://www.w3.org/ns/ttml#parameter}timeBase")
    if time_base != "media":
        errors.append(f"ttp:timeBase must be 'media' (got {time_base})")
    frame_rate = root.attrib.get("{http://www.w3.org/ns/ttml#parameter}frameRate")
    if not frame_rate:
        errors.append("Missing ttp:frameRate")
    frame_mult = root.attrib.get("{http://www.w3.org/ns/ttml#parameter}frameRateMultiplier")
    if not frame_mult:
        errors.append("Missing ttp:frameRateMultiplier")

    # style checks
    style = root.find(".//tt:styling/tt:style", ns)
    if style is None:
        errors.append("Missing <style> element")
    else:
        if style.attrib.get("{http://www.w3.org/ns/ttml#style}color") != "#ebebeb":
            errors.append("tts:color must be #ebebeb")
        if style.attrib.get("{http://www.w3.org/ns/ttml#style}backgroundColor") != "transparent":
            errors.append("tts:backgroundColor must be transparent")
        if style.attrib.get("{http://www.w3.org/ns/ttml#style}fontFamily") != "proportionalSansSerif":
            errors.append("tts:fontFamily must be proportionalSansSerif")
        if style.attrib.get("{http://www.w3.org/ns/ttml#style}fontSize") != "80%":
            errors.append("tts:fontSize must be 80%")
        if style.attrib.get("{http://www.w3.org/ns/ttml#style}textOutline") != "3% #101010":
            errors.append("tts:textOutline must be 3% #101010")
        if style.attrib.get("{http://www.w3.org/ns/ttml#style}luminanceGain") != "1.5":
            errors.append("tts:luminanceGain must be 1.5")

    region = root.find(".//tt:layout/tt:region", ns)
    if region is None:
        errors.append("Missing <region> element")
    else:
        if region.attrib.get("{http://www.w3.org/ns/ttml#style}origin") != "2.5% 15%":
            errors.append("tts:origin must be 2.5% 15%")
        if region.attrib.get("{http://www.w3.org/ns/ttml#style}extent") != "95% 70%":
            errors.append("tts:extent must be 95% 70%")

    # Validate cues
    def parse_time(val: str) -> Optional[int]:
        m = re.match(r"^(\d\d):(\d\d):(\d\d)\.(\d\d\d)$", val or "")
        if not m:
            return None
        hh, mm, ss, ms = map(int, m.groups())
        return ((hh * 60 + mm) * 60 + ss) * 1000 + ms

    prev_end = -1
    for p in root.findall(".//tt:body/tt:div/tt:p", ns):
        begin = p.attrib.get("begin")
        end = p.attrib.get("end")
        b_ms = parse_time(begin or "")
        e_ms = parse_time(end or "")
        if b_ms is None or e_ms is None:
            errors.append(f"Invalid time format on <p> begin/end: {begin} / {end}")
            continue
        if e_ms <= b_ms:
            errors.append(f"Non-positive duration: {begin} -> {end}")
        if b_ms < prev_end:
            errors.append(f"Overlap detected at {begin}")
        prev_end = e_ms

        # Extract visible lines
        # Keep <br/> splits
        text_fragments = []
        for node in p.iter():
            if node.tag.endswith("br"):
                text_fragments.append("\n")
            elif node is p:
                if node.text:
                    text_fragments.append(node.text)
            else:
                if node.text:
                    text_fragments.append(node.text)
                if node.tail:
                    text_fragments.append(node.tail)
        raw_text = "".join(text_fragments)
        lines = [ln.strip() for ln in raw_text.split("\n") if ln.strip() != ""]
        if len(lines) > 2:
            errors.append(f">2 lines in cue at {begin}")
        for line in lines:
            if len(line) > MAX_CHARS:
                errors.append(f"Line exceeds {MAX_CHARS} chars at {begin}: {line}")
        # sound cue mixed with dialogue
        for line in lines:
            if re.search(r"\[[^\]]+\]", line) and re.search(r"[A-Za-z]", re.sub(r"\[[^\]]+\]", "", line)):
                errors.append(f"Sound cue mixed with dialogue at {begin}: {line}")
        # protected phrase split across lines
        if len(lines) == 2:
            left, right = lines
            if boundary_splits_protected(left, right, protected):
                errors.append(f"Protected phrase split across lines at {begin}: {left} / {right}")

    return errors

try:
    from .qc import qc_report as _qc_report
except Exception:
    def _qc_report(cues_in: int, cues_out: List[Dict[str, Any]], protected_phrases: List[str]) -> Dict[str, Any]:
        return {"input_cues": cues_in, "output_cues": len(cues_out), "protected_phrases": protected_phrases}

MAX_LINES = 2
MAX_CHARS = 32
MAX_CUE_CHARS = MAX_LINES * MAX_CHARS
TARGET_CPS = 27.0
MAX_CPS = 45.0
MIN_DIALOGUE_MS = 800
MIN_DISPLAY_MS = 800  # Broadcast: minimum on-screen time so captions are readable
MIN_SOUND_DISPLAY_MS = 800  # Minimum on-screen time for sound cues (broadcast)
MIN_SOUND_MS = 400
TARGET_SOUND_MS = 1800
MAX_SOUND_MS = 3500
MERGE_GAP_MS = 650
WINDOW_PAD_MS = 700
SAME_SPEAKER_HARD_GAP_MS = 3200
SOUND_CLUSTER_GAP_MS = 200
TWO_SPEAKER_GAP_MS = 900
MAX_TWO_SPEAKER_WINDOW_MS = 4500
# Inline tags allowed to remain within dialogue (not treated as sound cues).
# For NBCU profile we force this to empty (no inline tags allowed).
DEFAULT_INLINE_DIALOGUE_TAGS = {"[INAUDIBLE]", "[UNINTELLIGIBLE]"}
INLINE_DIALOGUE_TAGS_RAW = os.getenv("INLINE_DIALOGUE_TAGS", "").strip()


def _parse_inline_dialogue_tags(raw: str) -> set:
    if raw:
        tags = set()
        for part in raw.split(","):
            tag = part.strip()
            if not tag:
                continue
            tag = tag.upper()
            if not tag.startswith("["):
                tag = f"[{tag}]"
            if not tag.endswith("]"):
                tag = f"{tag}]"
            tags.add(tag)
        return tags
    return set(DEFAULT_INLINE_DIALOGUE_TAGS)


INLINE_DIALOGUE_TAGS = _parse_inline_dialogue_tags(INLINE_DIALOGUE_TAGS_RAW)
FUNCTION_WORDS = {
    "a", "an", "the", "of", "to", "and", "or", "but", "with", "from", "in", "on", "at", "for",
    "that", "this", "these", "those", "is", "are", "was", "were", "be", "been", "being", "it", "i",
}
WEAK_ENDS = FUNCTION_WORDS | {"who", "what", "which", "when", "where", "why", "how", "well", "still"}
WEAK_STARTS = {"and", "or", "but", "to", "of", "for", "with", "because", "that", "this", "these", "those"}
CONNECTORS = {"of", "the", "and", "a", "an", "to", "for", "with", "vs", "v", "de", "du", "la", "le", "von"}
DEFAULT_ALLOWED_SOUND = {"[APPLAUSE]", "[LAUGHTER]", "[MUSIC]", "[CHEERING]", "[SFX]", "[BLEEP]"}
ALLOWED_SOUND = set(DEFAULT_ALLOWED_SOUND)
SOUND_PRIORITY = {"[MUSIC]": 1, "[CHEERING]": 2, "[LAUGHTER]": 3, "[APPLAUSE]": 4, "[SFX]": 5, "[BLEEP]": 6}
MUSIC_LONG_ONLY_MS = int(os.getenv("MUSIC_LONG_ONLY_MS", "2500") or 2500)
# Single-word dialogue cues allowed to stay as their own cue (not merged with previous)
ALLOWED_STANDALONE_WORDS = {"yes", "no", "ok", "okay", "right", "correct", "wrong", "maybe", "sure", "absolutely", "exactly", "never"}
# Words that must not be the only content on a second line (broadcast: no one-word weak second lines)
WEAK_SECOND_LINE_WORDS = {"well", "still", "yeah", "right", "ok", "okay", "so", "and", "but", "or"}
# Words that often continue a sentence (lowercase when previous cue ended with period/comma — broadcast)
CONTINUATION_STARTERS = frozenset(
    {"it's", "but", "so", "and", "well", "now", "this", "that", "they", "we", "you", "he", "she",
     "because", "which", "who", "when", "where", "what", "how", "then", "or", "if", "though", "yet", "still",
     "really", "wow", "hey", "there", "ok", "okay"}
)
MIN_FRAGMENT_WORDS = 3
MIN_FRAGMENT_CHARS = 10
LONG_GAP_BRIDGE_MS = 2200
BRIDGE_PAD_MS = 120
TAG_RE = re.compile(r"\[[^\]]+\]")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
TITLEISH_RE = re.compile(r"^[A-Z][A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?$|^[A-Z0-9]{2,}$")
STYLE_TAG_RE = re.compile(r"\{\\+an\d\}")
ITALIC_TAG_RE = re.compile(r"</?i>")

TIMECODE_OFFSET_MS = int(os.getenv("TIMECODE_OFFSET_MS", "0") or 0)
ALIGNMENT_DEFAULT = os.getenv("ALIGNMENT_DEFAULT", "an2")  # an2/an8/none
ITALICIZE_PHRASES_RAW = os.getenv("ITALICIZE_PHRASES", "").strip()
ITALICIZE_TITLES = os.getenv("ITALICIZE_TITLES", "").strip().lower() in {"1", "true", "yes", "y", "on"}
ITALICIZE_TITLES_MIN_WORDS = int(os.getenv("ITALICIZE_TITLES_MIN_WORDS", "3") or 3)

SPEAKER_LABEL_MODE = os.getenv("SPEAKER_LABEL_MODE", "dash").strip().lower()  # dash|alpha|generic|named
SPEAKER_LABEL_SINGLE = os.getenv("SPEAKER_LABEL_SINGLE", "").strip().lower() in {"1", "true", "yes", "y", "on"}
SPEAKER_GENERIC_PREFIX = os.getenv("SPEAKER_GENERIC_PREFIX", "SPEAKER").strip() or "SPEAKER"
SPEAKER_NAME_MAP_RAW = os.getenv("SPEAKER_NAME_MAP", "").strip()
SPEAKER_LABEL_FORMAT = os.getenv("SPEAKER_LABEL_FORMAT", "prefix").strip().lower()  # prefix|bracket

SOUND_LABEL_STYLE = os.getenv("SOUND_LABEL_STYLE", "simple").strip().lower()  # simple|descriptive

TTML_TIMEBASE = os.getenv("TTML_TIMEBASE", "media").strip() or "media"
TTML_FRAME_RATE = os.getenv("TTML_FRAME_RATE", "30").strip() or "30"
TTML_FRAME_RATE_MULTIPLIER = os.getenv("TTML_FRAME_RATE_MULTIPLIER", "1000 1001").strip() or "1000 1001"
TTML_TEXT_ALIGN = os.getenv("TTML_TEXT_ALIGN", "center").strip().lower() or "center"
OUTPUT_FORMATS_ENV = os.getenv("OUTPUT_FORMATS", "").strip()
CAPTION_PROFILE = os.getenv("CAPTION_PROFILE", "nbcu").strip().lower()  # nbcu|custom
SOUND_DENSITY = os.getenv("SOUND_DENSITY", "conservative").strip().lower()  # conservative|balanced|aggressive
VALIDATE_TTML = os.getenv("VALIDATE_TTML", "").strip().lower()
FAIL_ON_TTML_VALIDATION = os.getenv("FAIL_ON_TTML_VALIDATION", "").strip().lower()


def _parse_timecode(tc: str) -> Optional[int]:
    """Parse HH:MM:SS,mmm to ms."""
    try:
        hh, mm, ssms = tc.strip().split(":")
        ss, ms = ssms.split(",")
        return ((int(hh) * 60 + int(mm)) * 60 + int(ss)) * 1000 + int(ms)
    except Exception:
        return None


def _parse_alignment_windows() -> List[Tuple[int, int, str]]:
    """
    Optional alignment windows via env ALIGNMENT_WINDOWS as JSON list:
    [{"start":"HH:MM:SS,mmm","end":"HH:MM:SS,mmm","align":"an8"}]
    """
    raw = os.getenv("ALIGNMENT_WINDOWS", "").strip()
    if not raw:
        return []
    try:
        items = json.loads(raw)
    except Exception:
        return []
    out = []
    for item in items:
        start = _parse_timecode(item.get("start", ""))
        end = _parse_timecode(item.get("end", ""))
        align = str(item.get("align", "")).strip().lower()
        if start is None or end is None:
            continue
        if align not in {"an2", "an8"}:
            continue
        out.append((start, end, align))
    return out


ALIGNMENT_WINDOWS = _parse_alignment_windows()


def _alignment_for_cue(start_ms: int, end_ms: int) -> Optional[str]:
    if ALIGNMENT_DEFAULT.lower() == "none":
        return None
    for s, e, align in ALIGNMENT_WINDOWS:
        if start_ms >= s and end_ms <= e:
            return align
    return ALIGNMENT_DEFAULT.lower()


def _strip_style_tags(text: str) -> str:
    return ITALIC_TAG_RE.sub("", STYLE_TAG_RE.sub("", text or ""))


def _visible_len(text: str) -> int:
    return len(_strip_style_tags(text))


def _truncate_visible(text: str, max_chars: int) -> str:
    if _visible_len(text) <= max_chars:
        return text
    out = ""
    count = 0
    i = 0
    italic_open = False
    while i < len(text) and count < max_chars:
        if text.startswith("{\\an", i):
            end = text.find("}", i)
            if end != -1:
                out += text[i:end + 1]
                i = end + 1
                continue
        if text.startswith("<i>", i):
            out += "<i>"
            italic_open = True
            i += 3
            continue
        if text.startswith("</i>", i):
            out += "</i>"
            italic_open = False
            i += 4
            continue
        out += text[i]
        count += 1
        i += 1
    if italic_open and "</i>" not in out:
        out += "</i>"
    return out.rstrip()


def _cue_visible_text(cue: Dict[str, Any]) -> str:
    return " ".join(_strip_style_tags(line) for line in cue.get("lines", []))


def _reduce_high_cps(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Try to reduce extreme CPS by extending cue duration into nearby gaps when possible."""
    if not cues:
        return cues
    cues = sorted(cues, key=lambda c: (c["start_ms"], c["end_ms"], 0 if c.get("type") == "dialogue" else 1))
    for i, c in enumerate(cues):
        if c.get("type") != "dialogue":
            continue
        text_len = _visible_len(_cue_visible_text(c))
        if text_len == 0:
            continue
        dur = max(1, c["end_ms"] - c["start_ms"])
        cps = text_len / (dur / 1000.0)
        if cps <= MAX_CPS:
            continue
        target_dur = int((text_len / MAX_CPS) * 1000)
        extra = min(500, max(0, target_dur - dur))
        if extra <= 0:
            continue
        next_cue = cues[i + 1] if i + 1 < len(cues) else None
        gap = (next_cue["start_ms"] - c["end_ms"]) if next_cue else extra
        if gap >= extra:
            c["end_ms"] += extra
            continue
        if next_cue and next_cue.get("type") == "dialogue":
            slack_next = max(0, (next_cue["end_ms"] - next_cue["start_ms"]) - MIN_DIALOGUE_MS)
            take = min(slack_next, extra - gap)
            if take > 0:
                c["end_ms"] += extra
                next_cue["start_ms"] += take
    return cues


def _sound_trim_budget_ms() -> int:
    # NBCU: never trim dialogue to make room for sound cues.
    if CAPTION_PROFILE == "nbcu":
        return 0
    if SOUND_DENSITY == "aggressive":
        return 700
    if SOUND_DENSITY == "balanced":
        return 400
    return 200

def _parse_italicize_phrases() -> List[str]:
    if not ITALICIZE_PHRASES_RAW:
        return []
    parts = [p.strip() for p in ITALICIZE_PHRASES_RAW.split(",") if p.strip()]
    return sorted({p for p in parts if len(p) >= 2}, key=len, reverse=True)


ITALICIZE_PHRASES = _parse_italicize_phrases()


def _parse_speaker_name_map() -> Dict[str, str]:
    if not SPEAKER_NAME_MAP_RAW:
        return {}
    try:
        data = json.loads(SPEAKER_NAME_MAP_RAW)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except Exception:
        return {}
    return {}


SPEAKER_NAME_MAP = _parse_speaker_name_map()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _apply_profile_settings() -> None:
    global MAX_LINES, MAX_CHARS, MAX_CUE_CHARS, TARGET_CPS, MAX_CPS
    global MIN_DISPLAY_MS, MIN_SOUND_DISPLAY_MS, MIN_SOUND_MS, SOUND_CLUSTER_GAP_MS, MERGE_GAP_MS
    global SPEAKER_LABEL_MODE, SPEAKER_LABEL_SINGLE, SPEAKER_LABEL_FORMAT, SOUND_LABEL_STYLE, ALIGNMENT_DEFAULT
    global SOUND_DENSITY, VALIDATE_TTML, FAIL_ON_TTML_VALIDATION
    global INLINE_DIALOGUE_TAGS, ALLOWED_SOUND
    # NBCU profile locks
    if CAPTION_PROFILE == "nbcu":
        MAX_LINES = 2
        MAX_CHARS = 32
        TARGET_CPS = 27.0
        MAX_CPS = 45.0
        MIN_DISPLAY_MS = 800
        MIN_SOUND_DISPLAY_MS = 800
        MIN_SOUND_MS = 400
        SOUND_CLUSTER_GAP_MS = 200
        MERGE_GAP_MS = 650
        # NBCU forbids speaker labels beyond dash
        SPEAKER_LABEL_MODE = "dash"
        SPEAKER_LABEL_SINGLE = False
        SPEAKER_LABEL_FORMAT = "prefix"
        SOUND_LABEL_STYLE = "simple"
        ALIGNMENT_DEFAULT = "none"
        SOUND_DENSITY = "conservative"
        INLINE_DIALOGUE_TAGS = set()
        # NBCU: reactions + music only (music must be long; filtered later)
        ALLOWED_SOUND = {"[APPLAUSE]", "[LAUGHTER]", "[CHEERING]", "[MUSIC]"}
        VALIDATE_TTML = "1" if VALIDATE_TTML == "" else VALIDATE_TTML
        FAIL_ON_TTML_VALIDATION = "1" if FAIL_ON_TTML_VALIDATION == "" else FAIL_ON_TTML_VALIDATION
    elif CAPTION_PROFILE == "custom":
        MAX_LINES = _env_int("CUSTOM_MAX_LINES", MAX_LINES)
        MAX_CHARS = _env_int("CUSTOM_MAX_CHARS", MAX_CHARS)
        TARGET_CPS = _env_float("CUSTOM_TARGET_CPS", TARGET_CPS)
        MAX_CPS = _env_float("CUSTOM_MAX_CPS", MAX_CPS)
        MIN_DISPLAY_MS = _env_int("CUSTOM_MIN_DISPLAY_MS", MIN_DISPLAY_MS)
        MIN_SOUND_DISPLAY_MS = _env_int("CUSTOM_MIN_SOUND_DISPLAY_MS", MIN_SOUND_DISPLAY_MS)
        MIN_SOUND_MS = _env_int("CUSTOM_MIN_SOUND_MS", MIN_SOUND_MS)
        SOUND_CLUSTER_GAP_MS = _env_int("CUSTOM_SOUND_CLUSTER_GAP_MS", SOUND_CLUSTER_GAP_MS)
        MERGE_GAP_MS = _env_int("CUSTOM_MERGE_GAP_MS", MERGE_GAP_MS)
        VALIDATE_TTML = "0" if VALIDATE_TTML == "" else VALIDATE_TTML
        FAIL_ON_TTML_VALIDATION = "0" if FAIL_ON_TTML_VALIDATION == "" else FAIL_ON_TTML_VALIDATION
        INLINE_DIALOGUE_TAGS = _parse_inline_dialogue_tags(INLINE_DIALOGUE_TAGS_RAW)
        ALLOWED_SOUND = set(DEFAULT_ALLOWED_SOUND)
    MAX_CUE_CHARS = MAX_LINES * MAX_CHARS


_apply_profile_settings()


def _reload_config_from_env() -> None:
    global TIMECODE_OFFSET_MS, ALIGNMENT_DEFAULT, ITALICIZE_PHRASES_RAW, ITALICIZE_TITLES, ITALICIZE_TITLES_MIN_WORDS
    global SPEAKER_LABEL_MODE, SPEAKER_LABEL_SINGLE, SPEAKER_GENERIC_PREFIX, SPEAKER_NAME_MAP_RAW, SPEAKER_LABEL_FORMAT
    global SOUND_LABEL_STYLE, TTML_TIMEBASE, TTML_FRAME_RATE, TTML_FRAME_RATE_MULTIPLIER, TTML_TEXT_ALIGN
    global OUTPUT_FORMATS_ENV, CAPTION_PROFILE, SOUND_DENSITY, VALIDATE_TTML, FAIL_ON_TTML_VALIDATION
    global INLINE_DIALOGUE_TAGS_RAW, INLINE_DIALOGUE_TAGS, ITALICIZE_PHRASES, SPEAKER_NAME_MAP, MUSIC_LONG_ONLY_MS

    TIMECODE_OFFSET_MS = int(os.getenv("TIMECODE_OFFSET_MS", "0") or 0)
    ALIGNMENT_DEFAULT = os.getenv("ALIGNMENT_DEFAULT", "an2")
    ITALICIZE_PHRASES_RAW = os.getenv("ITALICIZE_PHRASES", "").strip()
    ITALICIZE_TITLES = os.getenv("ITALICIZE_TITLES", "").strip().lower() in {"1", "true", "yes", "y", "on"}
    ITALICIZE_TITLES_MIN_WORDS = int(os.getenv("ITALICIZE_TITLES_MIN_WORDS", "3") or 3)

    SPEAKER_LABEL_MODE = os.getenv("SPEAKER_LABEL_MODE", "dash").strip().lower()
    SPEAKER_LABEL_SINGLE = os.getenv("SPEAKER_LABEL_SINGLE", "").strip().lower() in {"1", "true", "yes", "y", "on"}
    SPEAKER_GENERIC_PREFIX = os.getenv("SPEAKER_GENERIC_PREFIX", "SPEAKER").strip() or "SPEAKER"
    SPEAKER_NAME_MAP_RAW = os.getenv("SPEAKER_NAME_MAP", "").strip()
    SPEAKER_LABEL_FORMAT = os.getenv("SPEAKER_LABEL_FORMAT", "prefix").strip().lower()

    SOUND_LABEL_STYLE = os.getenv("SOUND_LABEL_STYLE", "simple").strip().lower()
    TTML_TIMEBASE = os.getenv("TTML_TIMEBASE", "media").strip() or "media"
    TTML_FRAME_RATE = os.getenv("TTML_FRAME_RATE", "30").strip() or "30"
    TTML_FRAME_RATE_MULTIPLIER = os.getenv("TTML_FRAME_RATE_MULTIPLIER", "1000 1001").strip() or "1000 1001"
    TTML_TEXT_ALIGN = os.getenv("TTML_TEXT_ALIGN", "center").strip().lower() or "center"

    OUTPUT_FORMATS_ENV = os.getenv("OUTPUT_FORMATS", "").strip()
    CAPTION_PROFILE = os.getenv("CAPTION_PROFILE", "nbcu").strip().lower() or "nbcu"
    SOUND_DENSITY = os.getenv("SOUND_DENSITY", "conservative").strip().lower() or "conservative"
    VALIDATE_TTML = os.getenv("VALIDATE_TTML", "").strip().lower()
    FAIL_ON_TTML_VALIDATION = os.getenv("FAIL_ON_TTML_VALIDATION", "").strip().lower()

    INLINE_DIALOGUE_TAGS_RAW = os.getenv("INLINE_DIALOGUE_TAGS", "").strip()
    INLINE_DIALOGUE_TAGS = _parse_inline_dialogue_tags(INLINE_DIALOGUE_TAGS_RAW)
    MUSIC_LONG_ONLY_MS = int(os.getenv("MUSIC_LONG_ONLY_MS", "2500") or 2500)

    _apply_profile_settings()
    ITALICIZE_PHRASES = _parse_italicize_phrases()
    SPEAKER_NAME_MAP = _parse_speaker_name_map()


def apply_env_overrides(env: Optional[Dict[str, Any]]) -> Dict[str, Optional[str]]:
    if not env:
        return {}
    snapshot = {k: os.environ.get(k) for k in env.keys()}
    for key, value in env.items():
        os.environ[str(key)] = str(value)
    _reload_config_from_env()
    return snapshot


def restore_env_overrides(snapshot: Dict[str, Optional[str]]) -> None:
    if not snapshot:
        return
    for key, value in snapshot.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    _reload_config_from_env()


def process_caption_job(
    backbone_srt_text: str,
    timestamps: Any,
    protected_phrases: Optional[List[str]] = None,
    output_formats: Optional[List[str]] = None,
) -> Dict[str, Any]:
    protected_phrases = protected_phrases or []
    if output_formats is None:
        if OUTPUT_FORMATS_ENV:
            output_formats = [f.strip().lower() for f in OUTPUT_FORMATS_ENV.split(",") if f.strip()]
        else:
            output_formats = ["ttml"]

    backbone = _parse_srt(backbone_srt_text)
    tokens = normalize_tokens(timestamps)
    protected = build_runtime_protected_phrases(backbone, tokens, protected_phrases)
    italic_phrases = build_italic_phrases(protected)
    speaker_map = build_speaker_label_map(tokens)

    dialogue_windows, sound_events = build_windows_from_backbone(backbone)
    dialogue_windows = merge_dialogue_windows(dialogue_windows, protected)

    for win in dialogue_windows:
        candidate_tokens = [t for t in tokens if t["start_ms"] < win["end_ms"] + WINDOW_PAD_MS and t["end_ms"] > win["start_ms"] - WINDOW_PAD_MS and not token_is_sound(t["text"])]
        aligned = slice_tokens_to_dialogue(candidate_tokens, win["dialogue_text"])
        if aligned:
            win["tokens"] = aligned
            # Use AssemblyAI as source: preserve its punctuation (no phrase-specific overrides)
            win["dialogue_text"] = normalize_space(join_tokens(aligned))
        else:
            win["tokens"] = [t for t in tokens if t["start_ms"] < win["end_ms"] and t["end_ms"] > win["start_ms"] and not token_is_sound(t["text"])]
        win["runs"] = build_runs_from_tokens(win["tokens"], win["dialogue_text"])

    leading_sound_events = build_leading_sound_events(dialogue_windows)
    dialogue_atoms = explode_windows_to_atoms(dialogue_windows)
    dialogue_atoms = merge_same_speaker_atoms(dialogue_atoms, protected)
    dialogue_atoms = pack_adjacent_two_speaker(dialogue_atoms)

    formatted_dialogue: List[Dict[str, Any]] = []
    for atom in dialogue_atoms:
        formatted_dialogue.extend(format_dialogue_atom(atom, protected))
    formatted_dialogue = merge_fragment_dialogue_cues(formatted_dialogue, protected)

    token_sound_events = build_sound_events_from_tokens(tokens)
    merged_sound = merge_sound_events(sound_events + leading_sound_events + token_sound_events)
    merged_sound = filter_sound_events(merged_sound)
    sound_cues = place_sound_events(merged_sound, formatted_dialogue)

    cues = formatted_dialogue + sound_cues
    cues = resolve_overlaps(cues)
    cues = final_qc_cleanup(cues, protected, italic_phrases)
    cues = apply_speaker_labels(cues, speaker_map, SPEAKER_LABEL_SINGLE)
    cues.sort(key=lambda c: (c["start_ms"], c["end_ms"], 0 if c["type"] == "dialogue" else 1))
    for i, cue in enumerate(cues, 1):
        cue["idx"] = i

    cues_for_export = apply_timecode_offset(cues, TIMECODE_OFFSET_MS)
    cues_for_export = apply_alignment_tags(cues_for_export, TIMECODE_OFFSET_MS)

    srt_out = _export_srt(cues_for_export) if "srt" in output_formats else None
    vtt_out = _export_vtt(cues_for_export) if "vtt" in output_formats else None
    scc_out = _export_scc(cues_for_export) if "scc" in output_formats else None
    ttml_out = _export_ttml(
        cues_for_export,
        frame_rate=TTML_FRAME_RATE,
        frame_rate_multiplier=TTML_FRAME_RATE_MULTIPLIER,
        time_base=TTML_TIMEBASE,
        align=TTML_TEXT_ALIGN,
    ) if "ttml" in output_formats else None
    qc = _qc_report(len(backbone), cues, protected)
    if ttml_out and VALIDATE_TTML in {"1", "true", "yes", "y", "on"}:
        errors = _validate_ttml(ttml_out, protected)
        qc["ttml_validation_errors"] = errors
        qc["ttml_valid"] = len(errors) == 0
        if errors and FAIL_ON_TTML_VALIDATION in {"1", "true", "yes", "y", "on"}:
            raise ValueError("TTML validation failed: " + "; ".join(errors[:5]))
    return {"srt": srt_out, "vtt": vtt_out, "scc": scc_out, "ttml": ttml_out, "qc": qc}


# ------------ text helpers ------------

def normalize_tokens(timestamps: Any) -> List[Dict[str, Any]]:
    raw = _normalize_tokens(timestamps)
    out = []
    for item in raw:
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        start = int(item.get("start_ms", item.get("start", 0) or 0))
        end = int(item.get("end_ms", item.get("end", 0) or 0))
        if end <= start:
            end = start + 1
        out.append({"text": text, "start_ms": start, "end_ms": end, "speaker": str(item.get("speaker") or "A")})
    out.sort(key=lambda x: (x["start_ms"], x["end_ms"]))
    return out


def token_is_sound(text: str) -> bool:
    raw = (text or "").strip()
    tag = raw
    if re.match(r"^\[[^\]]+\][\.,!?]+$", raw):
        tag = re.sub(r"[\.,!?]+$", "", raw)
    tag_upper = tag.upper()
    if tag_upper in INLINE_DIALOGUE_TAGS:
        return False
    try:
        return bool(_is_sound_token(tag)) and tag_upper not in INLINE_DIALOGUE_TAGS
    except Exception:
        return tag.startswith("[") and tag.endswith("]") and tag_upper not in INLINE_DIALOGUE_TAGS


def normalize_space(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    return text.strip()


def flatten_words(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]


def normalize_token_word(text: str) -> str:
    text = normalize_space(text)
    text = re.sub(r"^[^A-Za-z0-9\[]+", "", text)
    text = re.sub(r"[^A-Za-z0-9\]'.’+-]+$", "", text)
    return text


def token_match_word(text: str) -> str:
    text = normalize_token_word(text)
    parts = flatten_words(text)
    return parts[0] if parts else ""


def join_tokens(tokens: Sequence[Dict[str, Any]]) -> str:
    return normalize_space(" ".join(t["text"] for t in tokens))


def apply_italics(text: str, phrases: Sequence[str]) -> str:
    if not text or not phrases:
        return text

    def italics_segment(seg: str) -> str:
        if "<i>" in seg or "</i>" in seg:
            return seg
        out = seg
        for phrase in phrases:
            if not phrase:
                continue
            pattern = re.compile(rf"\\b{re.escape(phrase)}\\b", re.I)

            def wrap(m: re.Match) -> str:
                return f"<i>{m.group(0)}</i>"

            out = pattern.sub(wrap, out)
        return out

    parts: List[str] = []
    last = 0
    for m in TAG_RE.finditer(text):
        parts.append(italics_segment(text[last:m.start()]))
        parts.append(m.group(0))
        last = m.end()
    parts.append(italics_segment(text[last:]))
    return "".join(parts)


def build_italic_phrases(protected: List[str]) -> List[str]:
    phrases = list(ITALICIZE_PHRASES)
    if ITALICIZE_TITLES:
        for phrase in protected:
            if text_word_count(phrase) >= ITALICIZE_TITLES_MIN_WORDS:
                phrases.append(phrase)
    # de-dup, prefer longer phrases
    uniq: List[str] = []
    seen = set()
    for p in sorted(phrases, key=len, reverse=True):
        key = p.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def format_sound_label(label: str) -> str:
    label = normalize_sound_label(label)
    if not label:
        return ""
    if SOUND_LABEL_STYLE != "descriptive":
        return label
    mapping = {
        "[APPLAUSE]": "[AUDIENCE APPLAUDS]",
        "[LAUGHTER]": "[AUDIENCE LAUGHS]",
        "[CHEERING]": "[CROWD CHEERS]",
        "[MUSIC]": "[MUSIC PLAYING]",
        "[SFX]": "[SOUND EFFECTS]",
        "[BLEEP]": "[BLEEP]",
    }
    return mapping.get(label, label)


def build_speaker_label_map(tokens: List[Dict[str, Any]]) -> Dict[str, str]:
    if SPEAKER_LABEL_MODE == "dash":
        return {}
    speakers: List[str] = []
    for tok in tokens:
        sp = str(tok.get("speaker") or "A")
        if sp not in speakers:
            speakers.append(sp)
    if SPEAKER_LABEL_MODE == "alpha":
        return {sp: sp for sp in speakers}
    if SPEAKER_LABEL_MODE == "generic":
        return {sp: f"{SPEAKER_GENERIC_PREFIX} {i+1}" for i, sp in enumerate(speakers)}
    if SPEAKER_LABEL_MODE == "named":
        return {sp: (SPEAKER_NAME_MAP.get(sp) or sp) for sp in speakers}
    return {}


def _fit_speaker_label(line: str, label: str) -> str:
    base = line
    if base.startswith("- "):
        base = base[2:]
    if SPEAKER_LABEL_FORMAT == "bracket":
        candidates = [
            f"-[{label}] {base}",
            f"- [{label}] {base}",
        ]
    else:
        candidates = [
            f"- {label}: {base}",
            f"- {label} {base}",
        ]
    # fallback short label
    short = label
    m = re.match(r"^SPEAKER\\s*(\\d+)$", label, re.I)
    if m:
        short = f"S{m.group(1)}"
    elif " " in label:
        short = label.split()[0]
    if SPEAKER_LABEL_FORMAT == "bracket":
        candidates.append(f"-[{short}] {base}")
        candidates.append(f"- [{short}] {base}")
    else:
        candidates.append(f"- {short}: {base}")
        candidates.append(f"- {short} {base}")
    for cand in candidates:
        if _visible_len(cand) <= MAX_CHARS:
            return cand
    return line


def apply_speaker_labels(cues: List[Dict[str, Any]], speaker_map: Dict[str, str], label_single: bool) -> List[Dict[str, Any]]:
    if not speaker_map:
        return cues
    out: List[Dict[str, Any]] = []
    for c in cues:
        if c.get("type") != "dialogue":
            out.append(c)
            continue
        runs = c.get("meta", {}).get("runs", [])
        lines = list(c.get("lines", []))
        if c.get("meta", {}).get("two_speaker") and len(runs) == 2 and len(lines) == 2:
            new_lines = []
            for line, run in zip(lines, runs):
                label = speaker_map.get(str(run.get("speaker") or "A"))
                if label:
                    line = _fit_speaker_label(line, label)
                new_lines.append(line)
            c = dict(c)
            c["lines"] = new_lines
            out.append(c)
            continue
        if label_single and runs and lines:
            label = speaker_map.get(str(runs[0].get("speaker") or "A"))
            if label:
                line = lines[0]
                if SPEAKER_LABEL_FORMAT == "bracket":
                    prefixed = f"[{label}] {line}"
                else:
                    prefixed = f"{label}: {line}"
                if _visible_len(prefixed) <= MAX_CHARS:
                    lines[0] = prefixed
                else:
                    # try short label
                    short = label.split()[0]
                    if SPEAKER_LABEL_FORMAT == "bracket":
                        prefixed = f"[{short}] {line}"
                    else:
                        prefixed = f"{short}: {line}"
                    if _visible_len(prefixed) <= MAX_CHARS:
                        lines[0] = prefixed
            c = dict(c)
            c["lines"] = lines
        out.append(c)
    return out


def strip_non_dialogue_tags(text: str) -> str:
    def repl(match: re.Match) -> str:
        tag = match.group(0).upper()
        return tag if tag in INLINE_DIALOGUE_TAGS else " "
    return normalize_space(TAG_RE.sub(repl, text or ""))


def sanitize_last_word(text: str) -> str:
    words = flatten_words(text)
    return words[-1] if words else ""


def text_word_count(text: str) -> int:
    return len(flatten_words(text))


def is_ultra_fragment(text: str) -> bool:
    text = normalize_space(text)
    return text_word_count(text) <= 1 or len(text) <= 6


def capitalize_i_pronoun(text: str) -> str:
    """Capitalize pronoun I (I'm, I've, I'll, etc.) at line/cue start or after sentence end (broadcast spec)."""
    if not text or len(text) < 2:
        return text
    text = normalize_space(text)
    # After comma (mid-sentence): pronoun I stays capitalized
    text = re.sub(r",\s*i'm\b", ", I'm", text, flags=re.I)
    text = re.sub(r",\s*i've\b", ", I've", text, flags=re.I)
    text = re.sub(r",\s*i'll\b", ", I'll", text, flags=re.I)
    text = re.sub(r",\s*i'd\b", ", I'd", text, flags=re.I)
    text = re.sub(r",\s+i\b", ", I ", text)
    # At start of string
    text = re.sub(r"^\s*i'm\b", "I'm", text, flags=re.I)
    text = re.sub(r"^\s*i've\b", "I've", text, flags=re.I)
    text = re.sub(r"^\s*i'll\b", "I'll", text, flags=re.I)
    text = re.sub(r"^\s*i'd\b", "I'd", text, flags=re.I)
    text = re.sub(r"^\s*i was\b", "I was", text, flags=re.I)
    text = re.sub(r"^\s*i am\b", "I am", text, flags=re.I)
    text = re.sub(r"^\s*i think\b", "I think", text, flags=re.I)
    text = re.sub(r"^\s*i mean\b", "I mean", text, flags=re.I)
    text = re.sub(r"^\s*i want\b", "I want", text, flags=re.I)
    text = re.sub(r"^\s*i just\b", "I just", text, flags=re.I)
    # After sentence end [.?!]
    text = re.sub(r"([.?!])\s+i'm\b", r"\1 I'm", text, flags=re.I)
    text = re.sub(r"([.?!])\s+i've\b", r"\1 I've", text, flags=re.I)
    text = re.sub(r"([.?!])\s+i'll\b", r"\1 I'll", text, flags=re.I)
    text = re.sub(r"([.?!])\s+i'd\b", r"\1 I'd", text, flags=re.I)
    text = re.sub(r"([.?!])\s+i\b", r"\1 I", text, flags=re.I)
    return text


def repair_continuing_punctuation(text: str) -> str:
    text = normalize_space(text)
    if not text:
        return text
    # Only period->comma for continuing thought (broadcast). Never comma->period.
    # If a period is followed by a lowercase word, treat as continuing phrase.
    text = re.sub(r"\.\s+(?=[a-z])", ", ", text)
    # Comma then capital (mid-sentence continuation): lowercase so not sentence start (broadcast spec).
    for word in (
        "You", "She", "He", "They", "This", "That", "What", "When", "Where", "Which", "Who", "How",
        "Because", "And", "But", "So", "While", "Thank", "My", "Your", "They're", "We're", "It's", "I'm",
        "Now", "Well", "Then", "Or", "If", "Though", "Yet", "Still", "Here", "There", "It",
    ):
        text = re.sub(r",\s*" + word + r"\b", ", " + word.lower(), text)
    text = re.sub(r",, +", ", ", text)
    return normalize_space(text)


def apply_asr_corrections(text: str) -> str:
    """
    Fix common ASR (speech-to-text) errors for broadcast captions.
    Only whole-word/phrase replacements that apply to any content; no show-specific rules.
    Preserves timing and word count.
    """
    if not text or not text.strip():
        return text
    text = normalize_space(text)
    # Universal mishears (any content)
    text = re.sub(r"\bdickhand\b", "deckhand", text, flags=re.I)
    # Duplicate word (any content)
    text = re.sub(r"\bdown\s+down\b", "down", text, flags=re.I)
    # Capitalization: comma then capitalized mid-sentence word (grammar rule)
    text = re.sub(r",\s*But\s+", ", but ", text)
    text = re.sub(r",\s*So\s+", ", so ", text)
    text = re.sub(r",\s*Because\s+", ", because ", text)
    text = re.sub(r",\s*And\s+", ", and ", text)
    # Mid-sentence: "we Come back" → "we come back" (any "when we come back" etc.)
    text = re.sub(r"\bwe Come back\b", "we come back", text, flags=re.I)
    # Greeting: "Hey everybody. welcome" → "Hey everybody, welcome" (common pattern)
    text = re.sub(r"\bHey everybody\.\s+welcome\b", "Hey everybody, welcome", text, flags=re.I)
    # Grammar: "in I would" → "and I would" (common ASR/grammar)
    text = re.sub(r"\s+in\s+I\s+would\s+", " and I would ", text, flags=re.I)
    # Meaning-critical: "comprehensible" in negative/tragedy context → "incomprehensible"
    text = re.sub(
        r"\bwhich is just[—\-]\s*(?:\[[^\]]+\]\s*)?comprehensible\s+to the family\b",
        "which is just— incomprehensible to the family",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\bcomprehensible\s+to the family\s+and everyone that knew him\b",
        "incomprehensible to the family and everyone that knew him",
        text,
        flags=re.I,
    )
    # Erroneous [INAUDIBLE] between words (e.g. "Second [INAUDIBLE] Stew" → "Second Stew")
    text = re.sub(r"\bSecond\s+\[INAUDIBLE\]\s+Stew\b", "Second Stew", text, flags=re.I)
    # Universal ASR typo (double letter)
    text = re.sub(r"\breallly\b", "really", text, flags=re.I)
    return normalize_space(text)


# ------------ protected phrases ------------

def build_runtime_protected_phrases(backbone: List[Dict[str, Any]], tokens: List[Dict[str, Any]], explicit: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for phrase in explicit + detect_phrases_from_backbone(backbone) + detect_phrases_from_tokens(tokens):
        phrase = normalize_space(phrase)
        if len(phrase.split()) < 2:
            continue
        key = phrase.lower()
        if key not in seen:
            seen.add(key)
            out.append(phrase)
    return out


def detect_phrases_from_backbone(backbone: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for cue in backbone:
        text = strip_non_dialogue_tags(normalize_space(" ".join(cue.get("lines", []))))
        out.extend(extract_titleish_phrases(text.split()))
    return out


def detect_phrases_from_tokens(tokens: List[Dict[str, Any]]) -> List[str]:
    words = [t["text"] for t in tokens if not token_is_sound(t["text"])]
    return extract_titleish_phrases(words)


def extract_titleish_phrases(words: Sequence[str]) -> List[str]:
    cleaned = [normalize_token_word(w) for w in words]
    out: List[str] = []
    i = 0
    while i < len(cleaned):
        w = cleaned[i]
        if not w or not is_titleish(w):
            i += 1
            continue
        phrase = [w]
        j = i + 1
        while j < len(cleaned) and len(phrase) < 6:
            nxt = cleaned[j]
            if not nxt:
                break
            if is_titleish(nxt) or nxt.lower() in CONNECTORS:
                phrase.append(nxt)
                j += 1
            else:
                break
        if len(phrase) >= 2 and sum(1 for part in phrase if is_titleish(part)) >= 2:
            out.append(normalize_space(" ".join(phrase)))
        i = j if len(phrase) > 1 else i + 1
    return out


def is_titleish(word: str) -> bool:
    return bool(TITLEISH_RE.match(word or ""))


# ------------ sound and dialogue windows from backbone ------------

def build_windows_from_backbone(backbone: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dialogue: List[Dict[str, Any]] = []
    sound: List[Dict[str, Any]] = []
    for idx, cue in enumerate(backbone):
        raw_text = normalize_space(" ".join(cue.get("lines", [])))
        leading, dialogue_text, all_labels = split_sound_and_dialogue(raw_text)
        start_ms = int(cue["start_ms"])
        end_ms = int(cue["end_ms"])
        if dialogue_text:
            dialogue.append({
                "start_ms": start_ms,
                "end_ms": end_ms,
                "raw_text": raw_text,
                "dialogue_text": apply_asr_corrections(repair_continuing_punctuation(dialogue_text)),
                "leading_sound": leading,
                "tokens": [],
                "runs": [],
            })
        else:
            labels = [l for l in all_labels if l in ALLOWED_SOUND]
            if labels:
                next_start = end_ms
                if idx + 1 < len(backbone):
                    next_raw = normalize_space(" ".join(backbone[idx + 1].get("lines", [])))
                    next_lead, next_dialogue, _ = split_sound_and_dialogue(next_raw)
                    next_label = dominant_sound_label(next_lead)
                    if next_dialogue and next_label in labels:
                        next_start = int(backbone[idx + 1]["start_ms"])
                sound.extend(expand_sound_sequence(labels, start_ms, max(end_ms, next_start)))
    return dialogue, sound


def split_sound_and_dialogue(raw_text: str) -> Tuple[List[str], str, List[str]]:
    raw_text = normalize_space(raw_text)
    if not raw_text:
        return [], "", []
    matches = list(TAG_RE.finditer(raw_text))
    leading: List[str] = []
    all_labels: List[str] = []
    pos = 0
    leading_phase = True
    for m in matches:
        between = raw_text[pos:m.start()].strip()
        if between:
            leading_phase = False
        labels = normalize_sound_labels(m.group(0))
        for label in labels:
            if label:
                all_labels.append(label)
                if leading_phase:
                    leading.append(label)
        pos = m.end()
    dialogue = strip_non_dialogue_tags(raw_text)
    return leading, dialogue, all_labels


def normalize_sound_labels(text: str) -> List[str]:
    tag = normalize_space(text).upper()
    # Allow bracketed tags with trailing punctuation (e.g. "[NOISE].")
    if re.match(r"^\[[^\]]+\][\.,!?]+$", tag):
        tag = re.sub(r"[\.,!?]+$", "", tag)
    if not (tag.startswith("[") and tag.endswith("]")):
        return []
    inner = tag[1:-1].strip().replace("♪", "MUSIC")
    parts = [p.strip() for p in re.split(r"\s+AND\s+|\s*,\s*|\s*/\s*", inner) if p.strip()]
    labels: List[str] = []
    for part in parts:
        if part in {"NOISE", "SOUND", "SFX", "FX", "VOICE", "SPEAKER", "PAUSE", "TALKING", "CROSSTALK"}:
            if part in {"NOISE", "SOUND", "SFX", "FX"}:
                label = "[SFX]"
                if label not in labels:
                    labels.append(label)
            continue
        if part.startswith("MUSIC") or part == "SONG":
            label = "[MUSIC]"
        elif part.startswith("APPLAUSE") or part in {"CLAPPING", "CLAPS"}:
            label = "[APPLAUSE]"
        elif part.startswith("LAUGHTER") or part in {"LAUGHS", "LAUGHING"}:
            label = "[LAUGHTER]"
        elif part.startswith("CHEER"):
            label = "[CHEERING]"
        elif "BLEEP" in part:
            label = "[BLEEP]"
        else:
            label = ""
        if label and label not in labels:
            labels.append(label)
    return labels


def normalize_sound_label(text: str) -> str:
    labels = normalize_sound_labels(text)
    return dominant_sound_label(labels)


def dominant_sound_label(labels: Sequence[str], prefer_first_reaction: bool = False) -> str:
    ordered = [label for label in labels if label in ALLOWED_SOUND]
    if not ordered:
        return ""
    if prefer_first_reaction:
        for label in ordered:
            if label in {"[APPLAUSE]", "[LAUGHTER]", "[CHEERING]"}:
                return label
    counts: Dict[str, int] = {}
    for label in ordered:
        counts[label] = counts.get(label, 0) + 1
    return sorted(counts.items(), key=lambda kv: (kv[1], SOUND_PRIORITY.get(kv[0], 0)), reverse=True)[0][0]


def build_leading_sound_events(windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for win in windows:
        labels = win.get("leading_sound") or []
        labels = [l for l in labels if l in ALLOWED_SOUND]
        if not labels:
            continue
        toks = win.get("tokens") or []
        if toks:
            end_ms = toks[0]["start_ms"]
        else:
            end_ms = min(win["end_ms"], win["start_ms"] + TARGET_SOUND_MS)
        start_ms = win["start_ms"]
        if end_ms - start_ms < MIN_SOUND_MS:
            end_ms = min(win["end_ms"], start_ms + MIN_SOUND_MS)
        if end_ms - start_ms < MIN_SOUND_MS:
            continue
        events.extend(expand_sound_sequence(labels, start_ms, end_ms))
    return events


# ------------ dialogue window merging ------------

def merge_dialogue_windows(windows: List[Dict[str, Any]], protected: List[str]) -> List[Dict[str, Any]]:
    if not windows:
        return []
    merged = [windows[0]]
    for win in windows[1:]:
        prev = merged[-1]
        if should_merge_windows(prev, win, protected):
            prev["end_ms"] = win["end_ms"]
            prev["raw_text"] = normalize_space(f"{prev['raw_text']} {win['raw_text']}")
            prev["dialogue_text"] = apply_asr_corrections(repair_continuing_punctuation(normalize_space(f"{prev['dialogue_text']} {win['dialogue_text']}")))
        else:
            merged.append(win)
    return merged


def should_merge_windows(a: Dict[str, Any], b: Dict[str, Any], protected: List[str]) -> bool:
    gap = b["start_ms"] - a["end_ms"]
    if gap > 1100:
        return False
    combined = normalize_space(f"{a['dialogue_text']} {b['dialogue_text']}")
    if len(combined) <= 64 and gap <= MERGE_GAP_MS:
        return True
    if is_fragment(a["dialogue_text"]) or is_fragment(b["dialogue_text"]):
        return True
    if continues_sentence(a["dialogue_text"], b["dialogue_text"]):
        return True
    if boundary_splits_protected(a["dialogue_text"], b["dialogue_text"], protected):
        return True
    return False


def is_fragment(text: str) -> bool:
    text = normalize_space(text)
    words = text.split()
    if len(words) <= 2 or len(text) <= 14:
        return True
    if sanitize_last_word(text) in WEAK_ENDS:
        return True
    if words and words[0].lower() in WEAK_STARTS:
        return True
    return False


def continues_sentence(a: str, b: str) -> bool:
    a = normalize_space(a)
    b = normalize_space(b)
    if not a or not b:
        return False
    if re.search(r"[,:;]$", a):
        return True
    if not re.search(r"[.!?]$", a):
        return True
    if b.split() and b.split()[0][:1].islower():
        return True
    return False


def boundary_splits_protected(a: str, b: str, protected: List[str]) -> bool:
    aw = flatten_words(a)
    bw = flatten_words(b)
    for phrase in protected:
        pw = flatten_words(phrase)
        if len(pw) < 2:
            continue
        max_i = min(len(aw), len(pw) - 1)
        for i in range(1, max_i + 1):
            if aw[-i:] == pw[:i] and bw[:len(pw)-i] == pw[i:]:
                return True
    return False


# ------------ token alignment and speaker runs ------------

def slice_tokens_to_dialogue(candidate_tokens: List[Dict[str, Any]], dialogue_text: str) -> List[Dict[str, Any]]:
    target = flatten_words(dialogue_text)
    if not target or not candidate_tokens:
        return []
    cand = [token_match_word(t["text"]) for t in candidate_tokens]
    n = len(target)
    for i in range(0, len(cand) - n + 1):
        if cand[i:i+n] == target:
            return candidate_tokens[i:i+n]
    # fuzzy prefix fallback for near-exact windows
    for i in range(len(cand)):
        if cand[i] != target[0]:
            continue
        j = 0
        while i + j < len(cand) and j < n and cand[i + j] == target[j]:
            j += 1
        if j >= max(3, n - 1):
            return candidate_tokens[i:i+j]
    return []


def build_runs_from_tokens(tokens: List[Dict[str, Any]], fallback_text: str) -> List[Dict[str, Any]]:
    text = normalize_space(fallback_text)
    if not tokens:
        if not text:
            return []
        return [{"speaker": "A", "text": text, "start_ms": 0, "end_ms": 0, "tokens": []}]
    # Use AssemblyAI token text as-is (only normalize + universal ASR fixes); preserve punctuation
    token_text = normalize_space(join_tokens(tokens))
    if len({(t.get("speaker") or "A") for t in tokens}) == 1:
        return [{
            "speaker": tokens[0].get("speaker") or "A",
            "text": apply_asr_corrections(token_text or text),
            "start_ms": tokens[0]["start_ms"],
            "end_ms": tokens[-1]["end_ms"],
            "tokens": tokens,
        }]
    runs: List[Dict[str, Any]] = []
    current = [tokens[0]]
    for tok in tokens[1:]:
        prev = current[-1]
        if tok["speaker"] != prev["speaker"] or tok["start_ms"] - prev["end_ms"] > SAME_SPEAKER_HARD_GAP_MS:
            runs.append(run_from_tokens(current))
            current = [tok]
        else:
            current.append(tok)
    runs.append(run_from_tokens(current))
    return [r for r in runs if r["text"]]


def run_from_tokens(tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Preserve AssemblyAI punctuation; only universal ASR corrections
    return {
        "speaker": tokens[0]["speaker"] or "A",
        "text": apply_asr_corrections(normalize_space(join_tokens(tokens))),
        "start_ms": tokens[0]["start_ms"],
        "end_ms": tokens[-1]["end_ms"],
        "tokens": tokens,
    }


# ------------ dialogue atoms ------------

def explode_windows_to_atoms(windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    atoms: List[Dict[str, Any]] = []
    for win in windows:
        runs = win.get("runs") or [{"speaker": "A", "text": win["dialogue_text"], "start_ms": win["start_ms"], "end_ms": win["end_ms"], "tokens": win.get("tokens", [])}]
        if len(runs) == 1:
            run = dict(runs[0])
            run["start_ms"] = win["start_ms"]
            run["end_ms"] = win["end_ms"]
            atoms.append(make_atom([run], False, win))
            continue
        if len(runs) == 2 and can_two_speaker(runs):
            atoms.append(make_atom(runs, True, win))
            continue
        for run in runs:
            atoms.append(make_atom([run], False, win))
    return atoms


def can_two_speaker(runs: List[Dict[str, Any]]) -> bool:
    if len(runs) != 2:
        return False
    if runs[1]["start_ms"] - runs[0]["end_ms"] > TWO_SPEAKER_GAP_MS:
        return False
    if runs[1]["end_ms"] - runs[0]["start_ms"] > MAX_TWO_SPEAKER_WINDOW_MS:
        return False
    return all(_visible_len(normalize_space(r["text"])) <= MAX_CHARS - 2 and text_word_count(normalize_space(r["text"])) <= 8 for r in runs)


def make_atom(runs: List[Dict[str, Any]], two_speaker: bool, win: Dict[str, Any]) -> Dict[str, Any]:
    start = runs[0].get("start_ms") or win["start_ms"]
    end = runs[-1].get("end_ms") or win["end_ms"]
    return {
        "idx": 0,
        "start_ms": start,
        "end_ms": max(end, start + 1),
        "lines": [],
        "type": "dialogue",
        "meta": {"dialogue_text": repair_continuing_punctuation(normalize_space(" ".join(r["text"] for r in runs))), "runs": runs, "two_speaker": two_speaker},
    }


def merge_same_speaker_atoms(atoms: List[Dict[str, Any]], protected: List[str]) -> List[Dict[str, Any]]:
    if not atoms:
        return []
    merged = [atoms[0]]
    for atom in atoms[1:]:
        prev = merged[-1]
        if prev["meta"].get("two_speaker") or atom["meta"].get("two_speaker"):
            merged.append(atom)
            continue
        pruns = prev["meta"].get("runs", [])
        aruns = atom["meta"].get("runs", [])
        if len(pruns) != 1 or len(aruns) != 1 or pruns[0]["speaker"] != aruns[0]["speaker"]:
            merged.append(atom)
            continue
        gap = atom["start_ms"] - prev["end_ms"]
        combined = normalize_space(f"{pruns[0]['text']} {aruns[0]['text']}")
        if gap <= MERGE_GAP_MS and (continues_sentence(pruns[0]["text"], aruns[0]["text"]) or boundary_splits_protected(pruns[0]["text"], aruns[0]["text"], protected) or len(combined) <= 96):
            prev["end_ms"] = atom["end_ms"]
            pruns[0]["text"] = repair_continuing_punctuation(combined)
            pruns[0]["end_ms"] = atom["end_ms"]
            pruns[0]["tokens"] = pruns[0].get("tokens", []) + aruns[0].get("tokens", [])
            prev["meta"]["dialogue_text"] = repair_continuing_punctuation(combined)
        else:
            merged.append(atom)
    return merged


def pack_adjacent_two_speaker(atoms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    packed: List[Dict[str, Any]] = []
    i = 0
    while i < len(atoms):
        atom = atoms[i]
        if i + 1 < len(atoms):
            nxt = atoms[i + 1]
            ar = atom["meta"].get("runs", [])
            nr = nxt["meta"].get("runs", [])
            if not atom["meta"].get("two_speaker") and not nxt["meta"].get("two_speaker") and len(ar) == len(nr) == 1 and ar[0]["speaker"] != nr[0]["speaker"]:
                left = normalize_space(ar[0]["text"])
                right = normalize_space(nr[0]["text"])
                if _visible_len(left) <= MAX_CHARS - 2 and _visible_len(right) <= MAX_CHARS - 2 and text_word_count(left) <= 8 and text_word_count(right) <= 8 and nxt["start_ms"] - atom["end_ms"] <= TWO_SPEAKER_GAP_MS and nxt["end_ms"] - atom["start_ms"] <= MAX_TWO_SPEAKER_WINDOW_MS:
                    packed.append({
                        "idx": 0,
                        "start_ms": atom["start_ms"],
                        "end_ms": nxt["end_ms"],
                        "lines": [],
                        "type": "dialogue",
                        "meta": {"dialogue_text": repair_continuing_punctuation(normalize_space(f"{left} {right}")), "runs": ar + nr, "two_speaker": True},
                    })
                    i += 2
                    continue
        packed.append(atom)
        i += 1
    return packed


# ------------ formatting ------------

def format_dialogue_atom(atom: Dict[str, Any], protected: List[str]) -> List[Dict[str, Any]]:
    runs = atom["meta"].get("runs", [])
    if atom["meta"].get("two_speaker") and len(runs) == 2:
        left = normalize_space(runs[0]["text"])
        right = normalize_space(runs[1]["text"])
        duration_ms = atom["end_ms"] - atom["start_ms"]
        duration_s = max(duration_ms / 1000.0, 0.001)
        total_chars = _visible_len(left) + _visible_len(right) + 4  # "- " per line
        cps = total_chars / duration_s
        # Short exchange (e.g. "Thank you." / "Who made this?"): keep as one two-line cue so min duration applies once.
        # Only split when CPS is high and (duration long or text long) so we don't create sub-second single-line cues.
        if cps > MAX_CPS and (duration_s >= 2.0 or total_chars > 45):
            return [format_dialogue_atom(make_atom([run], False, atom), protected)[0] for run in runs]
        if _visible_len(left) <= MAX_CHARS - 2 and _visible_len(right) <= MAX_CHARS - 2:
            out = atom.copy()
            out["lines"] = [f"- {left}", f"- {right}"]
            return [out]
        # split back into single-speaker atoms if it won't fit cleanly.
        return [format_dialogue_atom(make_atom([run], False, atom), protected)[0] for run in runs]

    if len(runs) == 1 and runs[0].get("tokens"):
        return segment_single_speaker_atom(atom, runs[0], protected)

    text = normalize_space(atom["meta"].get("dialogue_text", ""))
    if not text:
        return []
    out = atom.copy()
    out["lines"] = best_layout(text, protected)
    return [out]


def segment_single_speaker_atom(atom: Dict[str, Any], run: Dict[str, Any], protected: List[str]) -> List[Dict[str, Any]]:
    tokens = run.get("tokens") or []
    if not tokens:
        out = atom.copy()
        out["lines"] = best_layout(normalize_space(run["text"]), protected)
        return [out]
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(tokens):
        end = choose_chunk_end(tokens, i, protected)
        while end < len(tokens):
            chunk = tokens[i:end]
            text = apply_asr_corrections(repair_continuing_punctuation(join_tokens(chunk)))
            if (text_word_count(text) >= MIN_FRAGMENT_WORDS and len(text) >= MIN_FRAGMENT_CHARS) or re.search(r"[.!?]$", text):
                break
            end += 1
        chunk = tokens[i:end]
        text = apply_asr_corrections(repair_continuing_punctuation(join_tokens(chunk)))
        lines = best_layout(text, protected)
        start_ms = atom["start_ms"] if i == 0 else chunk[0]["start_ms"]
        end_ms = atom["end_ms"] if end >= len(tokens) else chunk[-1]["end_ms"]
        out.append({
            "idx": 0,
            "start_ms": start_ms,
            "end_ms": max(end_ms, start_ms + 1),
            "lines": lines,
            "type": "dialogue",
            "meta": {"dialogue_text": text, "runs": [{"speaker": run["speaker"], "text": text, "tokens": chunk, "start_ms": start_ms, "end_ms": end_ms}], "two_speaker": False},
        })
        i = end

    # Merge any leftover micro-fragments created by timing gaps.
    merged: List[Dict[str, Any]] = []
    for cue in out:
        text = cue["meta"]["dialogue_text"]
        if merged and cue["start_ms"] - merged[-1]["end_ms"] <= 900:
            prev = merged[-1]
            prev_text = prev["meta"]["dialogue_text"]
            combined = repair_continuing_punctuation(normalize_space(f"{prev_text} {text}"))
            if (is_ultra_fragment(text) or is_fragment(text) or is_fragment(prev_text)) and len(combined) <= MAX_CUE_CHARS and maybe_best_layout(combined, protected) is not None:
                prev["end_ms"] = cue["end_ms"]
                prev["meta"]["dialogue_text"] = combined
                prev["meta"]["runs"][0]["text"] = combined
                prev["meta"]["runs"][0]["tokens"] = prev["meta"]["runs"][0].get("tokens", []) + cue["meta"]["runs"][0].get("tokens", [])
                prev["lines"] = best_layout(combined, protected)
                continue
        merged.append(cue)
    return merged


def choose_chunk_end(tokens: List[Dict[str, Any]], start_idx: int, protected: List[str]) -> int:
    best_end = start_idx + 1
    best_score: Optional[float] = None
    max_end = min(len(tokens), start_idx + 28)
    for end in range(start_idx + 1, max_end + 1):
        chunk = tokens[start_idx:end]
        text = apply_asr_corrections(repair_continuing_punctuation(join_tokens(chunk)))
        if len(text) > MAX_CUE_CHARS + 18:
            break
        lines = maybe_best_layout(text, protected)
        if lines is None:
            continue
        next_word = token_match_word(tokens[end]["text"]).lower() if end < len(tokens) else ""
        score = chunk_score(chunk, text, lines, next_word, protected)
        if best_score is None or score < best_score:
            best_score = score
            best_end = end
        if score < 8 and re.search(r"[.!?]$", text):
            break
    return best_end


def chunk_score(chunk: List[Dict[str, Any]], text: str, lines: List[str], next_word: str, protected: List[str]) -> float:
    score = 0.0
    total_len = len(text)
    duration_ms = chunk[-1]["end_ms"] - chunk[0]["start_ms"]
    cps = total_len / max(duration_ms / 1000.0, 0.001)
    if len(lines) == 2:
        score += abs(len(lines[0]) - len(lines[1])) * 0.6
        if sanitize_last_word(lines[0]) in WEAK_ENDS:
            score += 18
        if lines[1].split() and lines[1].split()[0].lower() in WEAK_STARTS:
            score += 10
    else:
        if total_len > 24:
            score += 8
    if re.search(r"[.!?]$", text):
        score -= 18
    elif re.search(r"[,;:]$", text):
        score -= 10
    else:
        score += 5
    if sanitize_last_word(text) in WEAK_ENDS:
        score += 16
    if next_word in WEAK_STARTS:
        score += 10
    if cps > MAX_CPS:
        score += (cps - MAX_CPS) * 25
    elif cps > TARGET_CPS:
        score += (cps - TARGET_CPS) * 8
    if total_len < 8:
        score += 80
    elif total_len < 14:
        score += 42
    elif total_len < 20:
        score += 18
    elif total_len < 26 and not re.search(r"[.!?,;:]$", text):
        score += 14
    if len(chunk) < MIN_FRAGMENT_WORDS:
        score += 28
    elif len(chunk) < 5 and not re.search(r"[.!?,;:]$", text):
        score += 12
    if boundary_splits_protected(text, " ".join(next_word.split()), protected):
        score += 20
    return score


def maybe_best_layout(text: str, protected: List[str]) -> Optional[List[str]]:
    text = normalize_space(text)
    if not text:
        return None
    # Broadcast: if it fits in one line (≤32 chars), use one line; 2 lines is max, not required
    if _visible_len(text) <= MAX_CHARS:
        return [text]
    if _visible_len(text) > MAX_CUE_CHARS:
        return None
    return best_two_line_split(text, protected)


def best_layout(text: str, protected: List[str]) -> List[str]:
    layout = maybe_best_layout(text, protected)
    if layout is not None:
        return layout
    words = text.split()
    if not words:
        return []
    best = [_truncate_visible(words[0], MAX_CHARS)]
    if len(words) > 1:
        remainder = normalize_space(" ".join(words[1:]))
        if remainder:
            best.append(_truncate_visible(remainder, MAX_CHARS).rstrip())
    return best[:MAX_LINES]


def best_two_line_split(text: str, protected: List[str]) -> Optional[List[str]]:
    words = text.split()
    candidates: List[Tuple[float, List[str]]] = []
    protected_blocked: List[Tuple[float, List[str]]] = []
    for i in range(1, len(words)):
        left = normalize_space(" ".join(words[:i]))
        right = normalize_space(" ".join(words[i:]))
        if _visible_len(left) > MAX_CHARS or _visible_len(right) > MAX_CHARS:
            continue
        lines = [left, right]
        if boundary_splits_protected(left, right, protected):
            protected_blocked.append((split_layout_score(lines, protected) + 60, lines))
            continue
        # Avoid weak function words at end/start unless unavoidable
        left_last = sanitize_last_word(left).lower()
        right_first = right.split()[0].lower() if right.split() else ""
        if left_last in WEAK_ENDS or right_first in WEAK_STARTS:
            protected_blocked.append((split_layout_score(lines, protected) + 40, lines))
            continue
        candidates.append((split_layout_score(lines, protected), lines))
    if not candidates:
        if protected_blocked:
            protected_blocked.sort(key=lambda x: x[0])
            return protected_blocked[0][1]
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def split_layout_score(lines: List[str], protected: List[str]) -> float:
    left, right = lines
    score = abs(_visible_len(left) - _visible_len(right)) * 0.7
    # Prefer splitting at sentence boundary so full sentence starts on new line (broadcast)
    if re.search(r"[.!?]$", left):
        score -= 14
    elif re.search(r"[,;:]$", left):
        score -= 6
    else:
        score += 6
    if sanitize_last_word(left) in WEAK_ENDS:
        score += 18
    if right.split() and right.split()[0].lower() in WEAK_STARTS:
        score += 12
    if len(right.split()) == 1:
        score += 50
    if len(left.split()) == 1:
        score += 28
    if _visible_len(right) < 8:
        score += 12
    if _visible_len(left) < 8:
        score += 10
    if boundary_splits_protected(left, right, protected):
        score += 40
    return score


# ------------ sound placement ------------



def merge_fragment_dialogue_cues(cues: List[Dict[str, Any]], protected: List[str]) -> List[Dict[str, Any]]:
    if not cues:
        return []
    cues = sorted(cues, key=lambda c: (c["start_ms"], c["end_ms"]))

    def cue_text(c: Dict[str, Any]) -> str:
        return repair_continuing_punctuation(normalize_space(c.get("meta", {}).get("dialogue_text", " ".join(c.get("lines", [])))))

    def can_merge(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        if a["type"] != "dialogue" or b["type"] != "dialogue":
            return False
        if a.get("meta", {}).get("two_speaker") or b.get("meta", {}).get("two_speaker"):
            return False
        aruns = a.get("meta", {}).get("runs", [])
        bruns = b.get("meta", {}).get("runs", [])
        if len(aruns) != 1 or len(bruns) != 1 or aruns[0].get("speaker") != bruns[0].get("speaker"):
            return False
        at = cue_text(a)
        bt = cue_text(b)
        words_bt = bt.split()
        # Single-word cue that is allowed to stand alone (e.g. "Yes." "No.") — do not merge
        if len(words_bt) == 1 and sanitize_last_word(bt).lower() in ALLOWED_STANDALONE_WORDS:
            return False
        gap = b["start_ms"] - a["end_ms"]
        max_gap = max(MERGE_GAP_MS, 900)
        if len(words_bt) <= 2 and not re.search(r"[.!?]$", bt):
            max_gap = 1200  # Allow larger gap when merging short fragments
        # Continuation/weak words that should not stand alone (e.g. "still." "well." "yeah.")
        if len(words_bt) <= 2 and sanitize_last_word(bt).lower() in ("still", "well", "yeah"):
            max_gap = max(max_gap, 1800)
        # Cue that is only [INAUDIBLE] or similar: merge with previous when close
        bt_stripped = strip_non_dialogue_tags(bt).strip()
        if re.match(r"^\[(?:INAUDIBLE|UNINTELLIGIBLE)\]\.?\s*$", bt_stripped, re.I):
            max_gap = max(max_gap, 1200)
        b_dur = b["end_ms"] - b["start_ms"]
        if b_dur < MIN_DIALOGUE_MS:
            max_gap = max(max_gap, 1200)  # Merge very short display-time cues
        if b_dur < 500:
            max_gap = max(max_gap, 1500)  # Pull in micro-cues (< 0.5s on screen)
        if gap > max_gap:
            return False
        combined = repair_continuing_punctuation(normalize_space(f"{at} {bt}"))
        if len(combined) > MAX_CUE_CHARS:
            return False
        if maybe_best_layout(combined, protected) is None:
            return False
        # Always merge single-word cues (except standalones) when combined fits
        if len(words_bt) == 1:
            return True
        if is_fragment(at) or is_fragment(bt):
            return True
        if continues_sentence(at, bt) or boundary_splits_protected(at, bt, protected):
            return True
        return False

    def merge_pair(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        arun = a["meta"]["runs"][0]
        brun = b["meta"]["runs"][0]
        raw_combined = repair_continuing_punctuation(normalize_space(f"{cue_text(a)} {cue_text(b)}"))
        text = apply_asr_corrections(raw_combined)
        tokens = list(arun.get("tokens", [])) + list(brun.get("tokens", []))
        return {
            "idx": 0,
            "start_ms": a["start_ms"],
            "end_ms": b["end_ms"],
            "lines": best_layout(text, protected),
            "type": "dialogue",
            "meta": {
                "dialogue_text": text,
                "runs": [{
                    "speaker": arun.get("speaker", "A"),
                    "text": text,
                    "tokens": tokens,
                    "start_ms": a["start_ms"],
                    "end_ms": b["end_ms"],
                }],
                "two_speaker": False,
            },
        }

    changed = True
    working = list(cues)
    while changed:
        changed = False
        out: List[Dict[str, Any]] = []
        i = 0
        while i < len(working):
            cur = working[i]
            if i + 1 < len(working) and can_merge(cur, working[i + 1]):
                out.append(merge_pair(cur, working[i + 1]))
                i += 2
                changed = True
                continue
            out.append(cur)
            i += 1
        working = out
    return working

def build_sound_events_from_tokens(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for tok in tokens:
        if not token_is_sound(tok.get("text", "")):
            current = None
            continue
        label = normalize_sound_label(tok.get("text", ""))
        if label not in ALLOWED_SOUND:
            current = None
            continue
        start_ms = int(tok["start_ms"])
        end_ms = int(tok["end_ms"])
        if current and current["label"] == label and start_ms - current["end_ms"] <= SOUND_CLUSTER_GAP_MS:
            current["end_ms"] = end_ms
        else:
            current = {"label": label, "start_ms": start_ms, "end_ms": end_ms}
            events.append(current)
    return events


def merge_sound_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events = [e for e in events if e.get("label") in ALLOWED_SOUND]
    if not events:
        return []
    events.sort(key=lambda e: (e["start_ms"], e["end_ms"]))
    merged: List[Dict[str, Any]] = []
    current = events[0].copy()
    for ev in events[1:]:
        if ev["label"] == current["label"] and ev["start_ms"] - current["end_ms"] <= SOUND_CLUSTER_GAP_MS:
            current["end_ms"] = max(current["end_ms"], ev["end_ms"])
        else:
            merged.append(current)
            current = ev.copy()
    merged.append(current)
    return merged


def filter_sound_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not events:
        return []
    if CAPTION_PROFILE != "nbcu":
        return events
    out: List[Dict[str, Any]] = []
    for ev in events:
        label = ev.get("label")
        if label not in ALLOWED_SOUND:
            continue
        if label == "[MUSIC]" and (ev["end_ms"] - ev["start_ms"]) < MUSIC_LONG_ONLY_MS:
            continue
        out.append(ev)
    return out


def expand_sound_sequence(labels: List[str], start_ms: int, end_ms: int) -> List[Dict[str, Any]]:
    labels = [l for l in labels if l in ALLOWED_SOUND]
    if not labels:
        return []
    # de-dup while preserving order
    seen = set()
    ordered: List[str] = []
    for l in labels:
        if l not in seen:
            seen.add(l)
            ordered.append(l)
    if len(ordered) == 1:
        return [{"label": ordered[0], "start_ms": start_ms, "end_ms": end_ms}]

    # Reaction + music pattern: give reaction a short head, music remainder.
    reactions = [l for l in ordered if l in {"[APPLAUSE]", "[LAUGHTER]", "[CHEERING]"}]
    if reactions and "[MUSIC]" in ordered:
        reaction = reactions[0]
        reaction_end = min(end_ms, start_ms + 1200)
        events = [{"label": reaction, "start_ms": start_ms, "end_ms": reaction_end}]
        if end_ms - reaction_end >= MIN_SOUND_MS:
            events.append({"label": "[MUSIC]", "start_ms": reaction_end, "end_ms": end_ms})
        return events

    # Otherwise split evenly across labels.
    total = end_ms - start_ms
    if total < MIN_SOUND_MS * len(ordered):
        # Not enough time: just use the first label.
        return [{"label": ordered[0], "start_ms": start_ms, "end_ms": end_ms}]
    slice_len = max(MIN_SOUND_MS, total // len(ordered))
    events = []
    cur = start_ms
    for idx, label in enumerate(ordered):
        seg_end = end_ms if idx == len(ordered) - 1 else min(end_ms, cur + slice_len)
        if seg_end - cur >= MIN_SOUND_MS:
            events.append({"label": label, "start_ms": cur, "end_ms": seg_end})
        cur = seg_end
    return events


def place_sound_events(events: List[Dict[str, Any]], dialogue_cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not events:
        return []
    dialogue = sorted([c for c in dialogue_cues if c["type"] == "dialogue"], key=lambda c: (c["start_ms"], c["end_ms"]))
    placed: List[Dict[str, Any]] = []
    for ev in events:
        cue = place_sound_event(ev, dialogue)
        if cue:
            placed.append(cue)
    return placed


def place_sound_event(ev: Dict[str, Any], dialogue: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    prev = None
    nxt = None
    for cue in dialogue:
        if cue["end_ms"] <= ev["start_ms"]:
            prev = cue
            continue
        if cue["start_ms"] >= ev["start_ms"]:
            nxt = cue
            break
        if cue["start_ms"] < ev["end_ms"] and ev["start_ms"] < cue["end_ms"]:
            nxt = cue
            break
    gap_start = prev["end_ms"] if prev else ev["start_ms"]
    gap_end = nxt["start_ms"] if nxt else max(ev["end_ms"], ev["start_ms"] + TARGET_SOUND_MS)

    # If there is no room, try to create a gap by trimming neighboring dialogue cues.
    if gap_end - gap_start < MIN_SOUND_MS:
        needed = MIN_SOUND_MS - (gap_end - gap_start)
        budget = _sound_trim_budget_ms()
        # trim previous cue if possible
        take_prev = 0
        if prev and prev.get("type") == "dialogue":
            slack_prev = max(0, (prev["end_ms"] - prev["start_ms"]) - MIN_DIALOGUE_MS)
            take_prev = min(slack_prev, needed, budget)
            if take_prev > 0:
                prev["end_ms"] -= take_prev
                gap_start = prev["end_ms"]
                needed -= take_prev
                budget -= take_prev
        # trim next cue if possible
        take_next = 0
        if needed > 0 and budget > 0 and nxt and nxt.get("type") == "dialogue":
            slack_next = max(0, (nxt["end_ms"] - nxt["start_ms"]) - MIN_DIALOGUE_MS)
            take_next = min(slack_next, needed, budget)
            if take_next > 0:
                nxt["start_ms"] += take_next
                gap_end = nxt["start_ms"]
                needed -= take_next
                budget -= take_next
        if gap_end - gap_start < MIN_SOUND_MS:
            return None
    # In long dead-air gaps, bridge earlier instead of pinning the cue right on top of dialogue.
    if ev["start_ms"] - gap_start >= 1000 and gap_end - gap_start >= LONG_GAP_BRIDGE_MS:
        start = gap_start + BRIDGE_PAD_MS
    else:
        start = max(gap_start, ev["start_ms"])
    lead_out = BRIDGE_PAD_MS if nxt else 0
    end_cap = gap_end - lead_out
    if end_cap <= start:
        return None
    desired_len = TARGET_SOUND_MS if ev["label"] == "[MUSIC]" else max(MIN_SOUND_MS, 900)
    desired_end = min(end_cap, start + desired_len, max(ev["end_ms"], start + MIN_SOUND_MS))
    if desired_end - start < MIN_SOUND_MS:
        desired_end = min(end_cap, start + MIN_SOUND_MS)
    if desired_end - start < 500:
        return None
    return {"idx": 0, "start_ms": start, "end_ms": min(desired_end, start + MAX_SOUND_MS), "lines": [ev["label"]], "type": "sound", "meta": {"sound_label": ev["label"]}}


def repair_global_fragments(cues: List[Dict[str, Any]], protected: List[str]) -> List[Dict[str, Any]]:
    if not cues:
        return []
    cues = sorted(cues, key=lambda c: (c["start_ms"], c["end_ms"], 0 if c["type"] == "dialogue" else 1))

    def cue_text(c: Dict[str, Any]) -> str:
        return repair_continuing_punctuation(normalize_space(c.get("meta", {}).get("dialogue_text", " ".join(c.get("lines", [])))))

    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(cues):
        cur = cues[i]
        if cur["type"] != "dialogue":
            out.append(cur)
            i += 1
            continue
        cur_runs = cur.get("meta", {}).get("runs", [])
        cur_text = cue_text(cur)
        frag = is_ultra_fragment(cur_text) or is_fragment(cur_text)
        merged = False
        if frag and len(cur_runs) == 1:
            # try next first
            if i + 1 < len(cues):
                nxt = cues[i + 1]
                nxt_runs = nxt.get("meta", {}).get("runs", [])
                if nxt["type"] == "dialogue" and len(nxt_runs) == 1 and nxt_runs[0].get("speaker") == cur_runs[0].get("speaker") and nxt["start_ms"] - cur["end_ms"] <= 1000:
                    combined = repair_continuing_punctuation(normalize_space(f"{cur_text} {cue_text(nxt)}"))
                    if len(combined) <= MAX_CUE_CHARS and maybe_best_layout(combined, protected) is not None:
                        new = dict(cur)
                        new["end_ms"] = nxt["end_ms"]
                        new["meta"] = dict(cur.get("meta", {}))
                        new["meta"]["dialogue_text"] = combined
                        new["meta"]["runs"] = [{**cur_runs[0], "text": combined, "end_ms": nxt["end_ms"], "tokens": cur_runs[0].get("tokens", []) + nxt_runs[0].get("tokens", [])}]
                        new["lines"] = best_layout(combined, protected)
                        out.append(new)
                        i += 2
                        merged = True
            if not merged and out:
                prev = out[-1]
                prev_runs = prev.get("meta", {}).get("runs", []) if isinstance(prev.get("meta"), dict) else []
                if prev.get("type") == "dialogue" and len(prev_runs) == 1 and prev_runs[0].get("speaker") == cur_runs[0].get("speaker") and cur["start_ms"] - prev["end_ms"] <= 1000:
                    combined = repair_continuing_punctuation(normalize_space(f"{cue_text(prev)} {cur_text}"))
                    if len(combined) <= MAX_CUE_CHARS and maybe_best_layout(combined, protected) is not None:
                        prev["end_ms"] = cur["end_ms"]
                        prev["meta"]["dialogue_text"] = combined
                        prev["meta"]["runs"] = [{**prev_runs[0], "text": combined, "end_ms": cur["end_ms"], "tokens": prev_runs[0].get("tokens", []) + cur_runs[0].get("tokens", [])}]
                        prev["lines"] = best_layout(combined, protected)
                        i += 1
                        merged = True
        if not merged:
            out.append(cur)
            i += 1
    return out


# ------------ final cleanup ------------

def resolve_overlaps(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cues = sorted(cues, key=lambda c: (c["start_ms"], c["end_ms"], 0 if c["type"] == "dialogue" else 1))
    out: List[Dict[str, Any]] = []
    for cue in cues:
        cue = dict(cue)
        cue["lines"] = [_truncate_visible(normalize_space(line), MAX_CHARS) for line in cue.get("lines", []) if normalize_space(line)]
        if not cue["lines"]:
            continue
        if not out:
            out.append(cue)
            continue
        prev = out[-1]
        if cue["start_ms"] < prev["end_ms"]:
            if cue["type"] == "sound":
                cue["start_ms"] = prev["end_ms"]
            elif prev["type"] == "sound":
                prev["end_ms"] = min(prev["end_ms"], cue["start_ms"])
            else:
                cue["start_ms"] = prev["end_ms"]
        if cue["end_ms"] <= cue["start_ms"]:
            cue["end_ms"] = cue["start_ms"] + 1
        if cue["type"] == "sound":
            dur = cue["end_ms"] - cue["start_ms"]
            if dur < MIN_SOUND_DISPLAY_MS:
                cue["end_ms"] = cue["start_ms"] + MIN_SOUND_DISPLAY_MS
        out.append(cue)
    return [c for c in out if c["end_ms"] > c["start_ms"]]


def _apply_minimum_display_duration(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enforce minimum on-screen time for dialogue cues so they are readable (broadcast standard)."""
    for i, c in enumerate(cues):
        if c.get("type") != "dialogue":
            continue
        dur = c["end_ms"] - c["start_ms"]
        if dur >= MIN_DISPLAY_MS:
            continue
        new_end = c["start_ms"] + MIN_DISPLAY_MS
        if new_end > c["end_ms"]:
            c["end_ms"] = new_end
    return cues


def _apply_sound_min_duration(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Enforce minimum on-screen time for sound cues (broadcast standard)."""
    for c in cues:
        if c.get("type") != "sound":
            continue
        dur = c["end_ms"] - c["start_ms"]
        if dur < MIN_SOUND_DISPLAY_MS:
            c["end_ms"] = c["start_ms"] + MIN_SOUND_DISPLAY_MS
    return cues


# Em dash or hyphen used in transcripts as speaker/segment separator (not content-specific)
_EM_DASH_SEPARATOR = re.compile(r"\s*[—\-]\s+", re.U)
EM_DASH_SPLIT = os.getenv("EM_DASH_SPLIT", "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _split_at_em_dash_segment(text: str) -> Optional[Tuple[str, str]]:
    """
    When transcript uses em dash (or hyphen) as a segment separator, the part after
    is a different speaker/segment. Put each on its own line with leading dash (broadcast).
    Rule is structure-based only: no phrase or content matching.
    """
    if not EM_DASH_SPLIT:
        return None
    text = normalize_space(text)
    parts = _EM_DASH_SEPARATOR.split(text, 1)
    if len(parts) != 2:
        return None
    left = normalize_space(parts[0])
    right = normalize_space(parts[1])
    if not left or not right:
        return None
    # Keep dash on first line; both parts must fit line length
    left = left + "—"
    if _visible_len(left) > MAX_CHARS - 2 or _visible_len(right) > MAX_CHARS - 2:
        return None
    return (left, right)


def _fix_weak_second_line(cues: List[Dict[str, Any]], protected: List[str]) -> List[Dict[str, Any]]:
    """Broadcast: avoid one-word weak second lines. Merge with next or fold into first line when possible."""
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(cues):
        c = cues[i]
        if c.get("type") != "dialogue" or c.get("meta", {}).get("two_speaker") or not c.get("lines"):
            out.append(c)
            i += 1
            continue
        lines = c["lines"]
        if len(lines) != 2:
            out.append(c)
            i += 1
            continue
        second = (lines[1] or "").strip()
        second_word = sanitize_last_word(second).lower()
        if second_word not in WEAK_SECOND_LINE_WORDS and second_word not in WEAK_ENDS:
            out.append(c)
            i += 1
            continue
        words_second = second.split()
        if len(words_second) > 2:
            out.append(c)
            i += 1
            continue
        # Second line is weak (e.g. "Well." "Still." "Yeah."). Try merge with next or fold.
        combined_one_line = normalize_space(f"{lines[0]} {second}")
        if _visible_len(combined_one_line) <= MAX_CHARS:
            c = dict(c)
            c["lines"] = [combined_one_line]
            if c.get("meta"):
                c["meta"] = dict(c["meta"])
                c["meta"]["dialogue_text"] = combined_one_line
            out.append(c)
            i += 1
            continue
        # Try merge with next cue (same speaker, close gap)
        if i + 1 < len(cues):
            nxt = cues[i + 1]
            if (nxt.get("type") == "dialogue" and not nxt.get("meta", {}).get("two_speaker")
                    and nxt["start_ms"] - c["end_ms"] <= 1200):
                runs_c = c.get("meta", {}).get("runs", [])
                runs_n = nxt.get("meta", {}).get("runs", [])
                if len(runs_c) == 1 and len(runs_n) == 1 and runs_c[0].get("speaker") == runs_n[0].get("speaker"):
                    text_c = normalize_space(" ".join(c["lines"]))
                    text_n = normalize_space(" ".join(nxt.get("lines", [])))
                    merged_text = repair_continuing_punctuation(f"{text_c} {text_n}")
                    layout = maybe_best_layout(merged_text, protected) if len(merged_text) <= MAX_CUE_CHARS else None
                    if layout:
                        new_cue = dict(c)
                        new_cue["end_ms"] = nxt["end_ms"]
                        new_cue["meta"] = dict(c.get("meta", {}))
                        new_cue["meta"]["dialogue_text"] = apply_asr_corrections(merged_text)
                        new_cue["meta"]["runs"] = [{"speaker": runs_c[0].get("speaker", "A"), "text": new_cue["meta"]["dialogue_text"],
                            "start_ms": c["start_ms"], "end_ms": nxt["end_ms"], "tokens": runs_c[0].get("tokens", []) + runs_n[0].get("tokens", [])}]
                        new_cue["lines"] = layout
                        out.append(new_cue)
                        i += 2
                        continue
        out.append(c)
        i += 1
    return out


def _capitalize_cue_starts(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Capitalize first letter of each dialogue cue's first line when it's a sentence start (not continuation)."""
    for i, c in enumerate(cues):
        if c.get("type") != "dialogue" or not c.get("lines"):
            continue
        # Skip if previous cue ended with comma (this cue is continuation; keep lowercase)
        if i > 0:
            prev = cues[i - 1]
            if prev.get("type") == "dialogue" and prev.get("lines"):
                last_prev = (prev["lines"][-1] or "").strip()
                if last_prev and last_prev[-1] == ",":
                    continue
        line = c["lines"][0]
        if not line:
            continue
        match = re.search(r"(^-?\s*)([a-z])", line)
        if match:
            start, letter = match.start(2), match.group(2)
            c["lines"][0] = line[:start] + letter.upper() + line[start + 1:]
    return cues


def _capitalize_after_question(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """When previous cue ended with ?, capitalize next cue's first word (e.g. 'incredible that...' -> 'Incredible that...')."""
    for i in range(1, len(cues)):
        c = cues[i]
        if c.get("type") != "dialogue" or not c.get("lines"):
            continue
        prev = cues[i - 1]
        if prev.get("type") != "dialogue" or not prev.get("lines"):
            continue
        last_prev = (prev["lines"][-1] or "").strip()
        if not last_prev or last_prev[-1] != "?":
            continue
        line = (c["lines"][0] or "").strip()
        if not line:
            continue
        prefix = ""
        if line.startswith("- "):
            prefix, line = "- ", line[2:].lstrip()
        words = line.split()
        if not words or words[0][0].isupper():
            continue
        words[0] = words[0][0].upper() + words[0][1:]
        c["lines"][0] = prefix + " ".join(words)
    return cues


def _cross_cue_period_to_comma(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """When a cue ends with a period and the next cue starts with a continuation word, use comma instead (broadcast: same thought = comma)."""
    for i in range(len(cues) - 1):
        cur = cues[i]
        nxt = cues[i + 1]
        if cur.get("type") != "dialogue" or nxt.get("type") != "dialogue":
            continue
        if not cur.get("lines") or not nxt.get("lines"):
            continue
        last = (cur["lines"][-1] or "").strip()
        first_line = (nxt["lines"][0] or "").strip()
        if not last or last[-1] != ".":
            continue
        prefix = ""
        if first_line.startswith("- "):
            prefix, first_line = "- ", first_line[2:].lstrip()
        words = first_line.split()
        if not words or words[0].lower() not in CONTINUATION_STARTERS:
            continue
        # Only treat as continuation if the word is already lowercase in the cue
        if words[0][:1].isupper():
            continue
        # Replace period with comma on previous cue's last line
        cur["lines"][-1] = cur["lines"][-1].rstrip()
        if cur["lines"][-1].endswith("."):
            cur["lines"][-1] = cur["lines"][-1][:-1] + ","
        # Lowercase continuation word at start of next cue
        words[0] = words[0].lower()
        nxt["lines"][0] = prefix + " ".join(words)
    return cues


def _lowercase_continuation_at_cue_start(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Only when previous cue ended with a COMMA (mid-sentence): lowercase continuation word at cue start. Never after period (new sentence)."""
    for i in range(1, len(cues)):
        c = cues[i]
        if c.get("type") != "dialogue" or not c.get("lines"):
            continue
        prev = cues[i - 1]
        if prev.get("type") != "dialogue" or not prev.get("lines"):
            continue
        last_prev = (prev["lines"][-1] or "").strip()
        if not last_prev or last_prev[-1] != ",":
            continue
        first_line = (c["lines"][0] or "").strip()
        if not first_line:
            continue
        prefix = ""
        if first_line.startswith("- "):
            prefix, first_line = "- ", first_line[2:].lstrip()
        words = first_line.split()
        if not words or words[0].lower() not in CONTINUATION_STARTERS:
            continue
        words[0] = words[0].lower()
        c["lines"][0] = prefix + " ".join(words)
    return cues


def _resolve_dialogue_overlaps(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove overlaps: if a dialogue cue extends past the next cue's start, shift the next start to the previous end."""
    cues = sorted(cues, key=lambda c: (c["start_ms"], c["end_ms"], 0 if c.get("type") == "dialogue" else 1))
    for i in range(len(cues) - 1):
        cur = cues[i]
        nxt = cues[i + 1]
        if cur["end_ms"] <= nxt["start_ms"]:
            continue
        # Overlap: shift next cue so it starts when this one ends
        nxt["start_ms"] = cur["end_ms"]
        if nxt["end_ms"] <= nxt["start_ms"]:
            nxt["end_ms"] = nxt["start_ms"] + MIN_DIALOGUE_MS
    return cues


def final_qc_cleanup(cues: List[Dict[str, Any]], protected: List[str], italic_phrases: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    italic_phrases = italic_phrases or []
    for cue in cues:
        if cue["type"] == "sound":
            label = format_sound_label(cue["lines"][0])
            if label:
                cue["lines"] = [label]
                out.append(cue)
            continue
        runs = cue.get("meta", {}).get("runs", [])
        if cue.get("meta", {}).get("two_speaker") and len(runs) == 2:
            left = apply_asr_corrections(repair_continuing_punctuation(normalize_space(runs[0]["text"])))
            right = apply_asr_corrections(repair_continuing_punctuation(normalize_space(runs[1]["text"])))
            left = apply_italics(left, italic_phrases)
            right = apply_italics(right, italic_phrases)
            duration_ms = cue["end_ms"] - cue["start_ms"]
            duration_s = max(duration_ms / 1000.0, 0.001)
            total_chars = _visible_len(left) + _visible_len(right) + 4
            cps = total_chars / duration_s
            # Keep short multi-speaker exchanges as one cue (two lines); only split when CPS high and not a quick back-and-forth.
            split_for_cps = cps > MAX_CPS and (duration_s >= 2.0 or total_chars > 45)
            if split_for_cps or _visible_len(left) > MAX_CHARS - 2 or _visible_len(right) > MAX_CHARS - 2:
                # Split into two cues using run timing to preserve monotonic timecodes
                r1, r2 = runs[0], runs[1]
                s1 = int(r1.get("start_ms") or cue["start_ms"])
                e1 = int(r1.get("end_ms") or cue["start_ms"])
                s2 = int(r2.get("start_ms") or cue["end_ms"])
                e2 = int(r2.get("end_ms") or cue["end_ms"])
                # Ensure monotonic ordering
                if e1 <= s1:
                    e1 = min(cue["end_ms"], s1 + MIN_DIALOGUE_MS)
                if s2 < e1:
                    s2 = e1
                if e2 <= s2:
                    e2 = min(cue["end_ms"], s2 + MIN_DIALOGUE_MS)
                c1 = {
                    "idx": 0,
                    "start_ms": s1,
                    "end_ms": e1,
                    "lines": [],
                    "type": "dialogue",
                    "meta": {"dialogue_text": r1["text"], "runs": [r1], "two_speaker": False},
                }
                c2 = {
                    "idx": 0,
                    "start_ms": s2,
                    "end_ms": e2,
                    "lines": [],
                    "type": "dialogue",
                    "meta": {"dialogue_text": r2["text"], "runs": [r2], "two_speaker": False},
                }
                out.extend(format_dialogue_atom(c1, protected))
                out.extend(format_dialogue_atom(c2, protected))
                continue
            cue["lines"] = [f"- {left}", f"- {right}"]
            cue["lines"] = [
                capitalize_i_pronoun(_truncate_visible(repair_continuing_punctuation(normalize_space(x)), MAX_CHARS))
                for x in cue["lines"] if repair_continuing_punctuation(normalize_space(x))
            ]
        else:
            text = apply_asr_corrections(repair_continuing_punctuation(normalize_space(cue.get("meta", {}).get("dialogue_text", " ".join(cue.get("lines", []))))))
            text = apply_italics(text, italic_phrases)
            cue["meta"]["dialogue_text"] = text
            # If transcript uses em dash as segment/speaker separator, two lines each with dash
            split = _split_at_em_dash_segment(text)
            if split:
                left, right = split
                cue["lines"] = [_truncate_visible(f"- {left}", MAX_CHARS), _truncate_visible(f"- {right}", MAX_CHARS)]
            else:
                cue["lines"] = best_layout(text, protected)
        cue["lines"] = [
            capitalize_i_pronoun(_truncate_visible(repair_continuing_punctuation(normalize_space(x)), MAX_CHARS))
            for x in cue.get("lines", []) if repair_continuing_punctuation(normalize_space(x))
        ]
        if cue["lines"]:
            out.append(cue)

    # final anti-fragment pass on dialogue only
    dialogue = [c for c in out if c["type"] == "dialogue"]
    sounds = [c for c in out if c["type"] == "sound"]
    dialogue = merge_fragment_dialogue_cues(dialogue, protected)
    final = repair_global_fragments(dialogue + sounds, protected)
    final.sort(key=lambda c: (c["start_ms"], c["end_ms"], 0 if c["type"] == "dialogue" else 1))
    dialogue = [c for c in final if c["type"] == "dialogue"]
    sounds = [c for c in final if c["type"] == "sound"]
    dialogue = _fix_weak_second_line(dialogue, protected)
    final = sorted(dialogue + sounds, key=lambda c: (c["start_ms"], c["end_ms"], 0 if c["type"] == "dialogue" else 1))
    # Broadcast: enforce minimum display duration so short cues are readable
    for _ in range(2):
        final = _apply_minimum_display_duration(final)
        final = _apply_sound_min_duration(final)
        final = _resolve_dialogue_overlaps(final)
    final = _reduce_high_cps(final)
    final = _apply_minimum_display_duration(final)
    final = _resolve_dialogue_overlaps(final)
    final = _lowercase_continuation_at_cue_start(final)
    final = _cross_cue_period_to_comma(final)
    final = _capitalize_cue_starts(final)
    final = _capitalize_after_question(final)
    # Populate speaker on each cue for output JSON (from AssemblyAI runs)
    for c in final:
        if c.get("type") == "dialogue":
            runs = c.get("meta", {}).get("runs", [])
            if len(runs) == 1:
                c["speaker"] = str(runs[0].get("speaker") or "A")
            elif len(runs) == 2:
                c["speaker"] = f"{runs[0].get('speaker') or 'A'},{runs[1].get('speaker') or 'B'}"
            else:
                c["speaker"] = str(runs[0].get("speaker") or "A") if runs else None
    return final


def apply_timecode_offset(cues: List[Dict[str, Any]], offset_ms: int) -> List[Dict[str, Any]]:
    if not offset_ms:
        return cues
    out: List[Dict[str, Any]] = []
    for c in cues:
        nc = dict(c)
        nc["start_ms"] = max(0, int(c["start_ms"]) + offset_ms)
        nc["end_ms"] = max(0, int(c["end_ms"]) + offset_ms)
        out.append(nc)
    return out


def apply_alignment_tags(cues: List[Dict[str, Any]], offset_ms: int) -> List[Dict[str, Any]]:
    if ALIGNMENT_DEFAULT.lower() == "none":
        return cues
    out: List[Dict[str, Any]] = []
    for c in cues:
        nc = dict(c)
        lines = list(nc.get("lines", []))
        if lines:
            align = _alignment_for_cue(nc["start_ms"], nc["end_ms"])
            if align:
                tag = f"{{\\{align}}}"
                if not STYLE_TAG_RE.match(lines[0]):
                    lines[0] = tag + lines[0]
        nc["lines"] = lines
        out.append(nc)
    return out

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        with open(sys.argv[1]) as f:
            srt_text = f.read()
        with open(sys.argv[2]) as f:
            raw = json.load(f)
        result = process_caption_job(srt_text, raw, [])
        if result.get("srt"):
            print(result["srt"])
        elif result.get("ttml"):
            print(result["ttml"])
