from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(title="OTT Caption Rules Engine", version="1.1.0")


# ----------------------------
# Models
# ----------------------------

class Rules(BaseModel):
    maxCharsPerLine: int = 32
    maxLines: int = 2
    maxCPS: float = 17.0
    minDurationMs: int = 1000
    maxDurationMs: int = 7000
    minGapMs: int = 80
    preferPunctuationBreaks: bool = True

    # SCC / Timecode options
    sccFrameRate: float = 29.97
    startAtHour00: bool = True


class Word(BaseModel):
    text: str
    start: int
    end: int
    speaker: Optional[str] = "A"


class Event(BaseModel):
    type: Literal["music", "foreign_language"]
    start: int
    end: int
    # music
    text: Optional[str] = None
    # foreign_language
    language: Optional[str] = None


class Cue(BaseModel):
    start: int
    end: int
    text: str
    speaker: str = "A"


class QCItem(BaseModel):
    cue: int
    type: str
    value: float


class QCReport(BaseModel):
    issuesCount: int
    issues: List[QCItem]


class BuildRequest(BaseModel):
    words: List[Word]
    rules: Rules = Field(default_factory=Rules)
    events: Optional[List[Event]] = None


class BuildResponse(BaseModel):
    rules: Rules
    cues: List[Cue]
    srt: str
    vtt: str
    scc: str
    qc: QCReport


# ----------------------------
# Guardrail: strip bracket tokens if events present
# ----------------------------

_BRACKET_TOKEN_RE = re.compile(r"[\[\]]")

def _events_include_types(events: Optional[List[Event]], types: set[str]) -> bool:
    if not events:
        return False
    for e in events:
        if e.type in types:
            return True
    return False

def strip_bracket_tokens_from_words_if_events_present(words: List[Word], events: Optional[List[Event]]) -> List[Word]:
    """
    If events include music/foreign_language, remove any word tokens that contain '[' or ']'
    to avoid double-insertion (e.g. "[Music]" or "[Speaking" "Spanish]") and 1ms sliver cues.
    """
    if not words:
        return words

    if not _events_include_types(events, {"music", "foreign_language"}):
        return words

    cleaned: List[Word] = []
    for w in words:
        if _BRACKET_TOKEN_RE.search(w.text or ""):
            continue
        cleaned.append(w)
    return cleaned


# ----------------------------
# Caption building utilities
# ----------------------------

_PUNCT_END_RE = re.compile(r"[.!?…]+$|[)\]]$")

def _normalize_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    # fix space before punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # em dash spacing
    text = text.replace(" — ", "—").replace(" - ", "-")
    return text

def _wrap_lines(text: str, max_chars: int, max_lines: int, prefer_punct: bool) -> str:
    """
    Greedy wrap, tries to break lines at spaces and (optionally) punctuation boundaries.
    """
    text = _normalize_spaces(text)
    if not text:
        return ""

    words = text.split(" ")
    lines: List[str] = []
    cur: List[str] = []

    def cur_len(parts: List[str]) -> int:
        if not parts:
            return 0
        return len(" ".join(parts))

    for w in words:
        if cur_len(cur + [w]) <= max_chars:
            cur.append(w)
            continue

        # Need to wrap
        if cur:
            lines.append(" ".join(cur))
            cur = [w]
        else:
            # single word longer than max_chars; hard-cut
            lines.append(w[:max_chars])
            rest = w[max_chars:]
            cur = [rest] if rest else []

        if len(lines) == max_lines:
            # truncate remaining
            break

    if len(lines) < max_lines and cur:
        lines.append(" ".join(cur))

    # If too many lines, compress into max_lines by truncation
    if len(lines) > max_lines:
        lines = lines[:max_lines]

    # Optional: if prefer punctuation breaks and we have exactly 2 lines,
    # try to rebalance so line1 ends at punctuation when possible.
    if prefer_punct and len(lines) == 2:
        l1 = lines[0].split(" ")
        l2 = lines[1].split(" ")
        # move a few words from end of l1 to start of l2 until l1 ends with punctuation
        # but do not exceed max_chars on l2.
        for _ in range(min(4, len(l1))):
            if _PUNCT_END_RE.search(" ".join(l1)):
                break
            moved = l1.pop()
            candidate_l2 = [moved] + l2
            if len(" ".join(candidate_l2)) <= max_chars:
                l2 = candidate_l2
            else:
                # can't move without breaking l2; undo
                l1.append(moved)
                break
        lines = [" ".join(l1), " ".join(l2)]

    return "\n".join(lines)


def _cps(text: str, dur_ms: int) -> float:
    if dur_ms <= 0:
        return 999999.0
    # CPS based on visible characters excluding line breaks
    visible = len(text.replace("\n", ""))
    return visible / (dur_ms / 1000.0)


def _merge_words_to_cues(words: List[Word], rules: Rules) -> List[Cue]:
    """
    Build cues from word-level timestamps.
    Heuristic: group words until we approach CPS or max duration;
    maintain min gap between cues; keep single speaker per cue.
    """
    cues: List[Cue] = []
    if not words:
        return cues

    cur_words: List[Word] = []
    cur_speaker = words[0].speaker or "A"
    cue_start = words[0].start
    cue_end = words[0].end

    def flush():
        nonlocal cur_words, cur_speaker, cue_start, cue_end
        if not cur_words:
            return
        raw_text = " ".join(w.text for w in cur_words)
        wrapped = _wrap_lines(raw_text, rules.maxCharsPerLine, rules.maxLines, rules.preferPunctuationBreaks)
        cues.append(Cue(start=cue_start, end=cue_end, text=wrapped, speaker=cur_speaker))
        cur_words = []

    for w in words:
        sp = w.speaker or "A"
        if not cur_words:
            cur_speaker = sp
            cue_start = w.start
            cue_end = w.end
            cur_words = [w]
            continue

        # If speaker changes, flush and start new cue
        if sp != cur_speaker:
            flush()
            cur_speaker = sp
            cue_start = w.start
            cue_end = w.end
            cur_words = [w]
            continue

        # Tentatively append and see if it violates constraints
        tentative_words = cur_words + [w]
        tentative_start = cue_start
        tentative_end = max(cue_end, w.end)
        tentative_dur = tentative_end - tentative_start
        tentative_text = _wrap_lines(
            " ".join(x.text for x in tentative_words),
            rules.maxCharsPerLine,
            rules.maxLines,
            rules.preferPunctuationBreaks,
        )
        tentative_cps = _cps(tentative_text, tentative_dur)

        # Hard stop conditions
        too_long = tentative_dur > rules.maxDurationMs
        cps_too_high = tentative_cps > rules.maxCPS and tentative_dur >= rules.minDurationMs

        # If gap between words is huge, also flush
        gap = w.start - cue_end
        big_gap = gap >= max(rules.minGapMs * 6, 400)

        if too_long or cps_too_high or big_gap:
            flush()

            # enforce min gap
            if cues and cue_start - cues[-1].end < rules.minGapMs:
                cue_start = cues[-1].end + rules.minGapMs

            cur_speaker = sp
            cue_start = max(w.start, cue_start)
            cue_end = w.end
            cur_words = [w]
        else:
            cur_words.append(w)
            cue_end = max(cue_end, w.end)

    flush()

    # Post-pass: enforce min durations by stretching (without overlap)
    for i in range(len(cues)):
        dur = cues[i].end - cues[i].start
        if dur < rules.minDurationMs:
            needed = rules.minDurationMs - dur
            # try to extend end forward if possible
            next_start = cues[i + 1].start if i + 1 < len(cues) else None
            if next_start is None:
                cues[i].end += needed
            else:
                max_extend = max(0, (next_start - rules.minGapMs) - cues[i].end)
                extend = min(needed, max_extend)
                cues[i].end += extend

    return cues


def _insert_events(cues: List[Cue], events: Optional[List[Event]], rules: Rules) -> List[Cue]:
    """
    Insert event cues and split any overlapping speech cues cleanly.
    """
    if not events:
        return cues

    # Build event cues
    event_cues: List[Cue] = []
    for e in events:
        if e.type == "music":
            txt = e.text or "[♪ MUSIC ♪]"
            event_cues.append(Cue(start=e.start, end=e.end, text=txt, speaker="A"))
        elif e.type == "foreign_language":
            lang = (e.language or "Unknown").strip()
            # Match NBCU instruction format: "[Speaking <language>]"
            event_cues.append(Cue(start=e.start, end=e.end, text=f"[Speaking {lang}]", speaker="A"))

    # Merge by splitting overlaps
    combined: List[Cue] = []
    all_events = sorted(event_cues, key=lambda c: (c.start, c.end))

    for cue in cues:
        working = [cue]
        for ev in all_events:
            new_working: List[Cue] = []
            for w in working:
                # no overlap
                if ev.end <= w.start or ev.start >= w.end:
                    new_working.append(w)
                    continue

                # overlap: split w around ev
                if w.start < ev.start:
                    left = Cue(start=w.start, end=max(w.start, ev.start - rules.minGapMs), text=w.text, speaker=w.speaker)
                    if left.end - left.start > 0:
                        new_working.append(left)

                # right segment
                if w.end > ev.end:
                    right = Cue(start=min(w.end, ev.end + rules.minGapMs), end=w.end, text=w.text, speaker=w.speaker)
                    if right.end - right.start > 0:
                        new_working.append(right)
            working = new_working
        combined.extend(working)

    combined.extend(all_events)

    # Normalize ordering and enforce minGap
    combined = sorted(combined, key=lambda c: (c.start, c.end))
    normalized: List[Cue] = []
    for c in combined:
        if not normalized:
            normalized.append(c)
            continue
        prev = normalized[-1]
        if c.start - prev.end < rules.minGapMs:
            # push this cue forward to maintain minGap (do not invert)
            shift = rules.minGapMs - (c.start - prev.end)
            c = Cue(start=c.start + shift, end=c.end + shift, text=c.text, speaker=c.speaker)
        if c.end <= c.start:
            continue
        normalized.append(c)

    return normalized


# ----------------------------
# SRT / VTT formatting
# ----------------------------

def _ms_to_srt_time(ms: int) -> str:
    if ms < 0:
        ms = 0
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def _ms_to_vtt_time(ms: int) -> str:
    if ms < 0:
        ms = 0
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def cues_to_srt(cues: List[Cue]) -> str:
    out = []
    for i, c in enumerate(cues, start=1):
        out.append(str(i))
        out.append(f"{_ms_to_srt_time(c.start)} --> {_ms_to_srt_time(c.end)}")
        out.append(c.text)
        out.append("")
    return "\n".join(out).strip() + "\n"

def cues_to_vtt(cues: List[Cue]) -> str:
    out = ["WEBVTT", ""]
    for c in cues:
        out.append(f"{_ms_to_vtt_time(c.start)} --> {_ms_to_vtt_time(c.end)}")
        out.append(c.text)
        out.append("")
    return "\n".join(out).strip() + "\n"


# ----------------------------
# SCC generation (minimal, “good enough for test”)
# NOTE: This is not a full SCC encoder; it’s a pragmatic encoder for the test harness.
# ----------------------------

def _ms_to_timecode_frames(ms: int, fps: float, start_at_hour00: bool) -> str:
    """
    Convert ms to SCC timecode with frames.
    For now: non-drop math is used for simplicity; for NBCU test we start at 00:00:00:00.
    """
    if ms < 0:
        ms = 0
    # startAtHour00 means start from 00:00:00:00 always (test exception)
    total_seconds = ms / 1000.0
    hours = int(total_seconds // 3600)
    total_seconds -= hours * 3600
    minutes = int(total_seconds // 60)
    total_seconds -= minutes * 60
    seconds = int(total_seconds)
    frac = total_seconds - seconds
    frames = int(round(frac * fps))
    if frames >= int(round(fps)):
        frames = 0
        seconds += 1
        if seconds >= 60:
            seconds = 0
            minutes += 1
            if minutes >= 60:
                minutes = 0
                hours += 1

    if start_at_hour00:
        hours = 0

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"

# very simple "text to hex words" placeholder encoder
# (Your existing engine already outputs lines like 94ae...; we keep that style stable-ish.)
def _pseudo_scc_payload_for_text(text: str) -> str:
    """
    This is NOT a spec-perfect SCC encoding.
    It creates a deterministic-ish hex payload so you can test piping/formatting.
    """
    # Remove newlines to simplify payload
    t = text.replace("\n", " ")
    t = _normalize_spaces(t)

    # Header-ish lead-in commonly seen in SCC samples
    lead = "94ae 94ae 9420 9420"
    # Map ASCII chars to pseudo hex words
    payload_words = []
    for ch in t[:96]:  # cap
        payload_words.append(f"{ord(ch):02x}{ord(ch):02x}")
    tail = "942c 942c 942f 942f"
    body = " ".join(payload_words)
    # Add a mid control marker to mimic your earlier output look
    return f"{lead} 9470 9470 {body} {tail}".strip()

def cues_to_scc(cues: List[Cue], rules: Rules) -> str:
    lines = ["Scenarist_SCC V1.0", ""]
    for c in cues:
        tc = _ms_to_timecode_frames(c.start, rules.sccFrameRate, rules.startAtHour00)
        payload = _pseudo_scc_payload_for_text(c.text)
        lines.append(f"{tc}\t{payload}")
        lines.append("")
    # Add final pop-on clear (your earlier outputs had a trailing 942c 942c at end)
    if cues:
        end_tc = _ms_to_timecode_frames(cues[-1].end, rules.sccFrameRate, rules.startAtHour00)
        lines.append(f"{end_tc}\t942c 942c")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


# ----------------------------
# QC
# ----------------------------

def qc_cues(cues: List[Cue], rules: Rules) -> QCReport:
    issues: List[QCItem] = []
    for i, c in enumerate(cues, start=1):
        dur = c.end - c.start
        if dur < rules.minDurationMs:
            issues.append(QCItem(cue=i, type="too_short_ms", value=float(dur)))
        if dur > rules.maxDurationMs:
            issues.append(QCItem(cue=i, type="too_long_ms", value=float(dur)))

        cps = _cps(c.text, dur)
        if cps > rules.maxCPS:
            issues.append(QCItem(cue=i, type="cps_high", value=float(round(cps, 2))))

        # line length check
        lines = c.text.split("\n")
        if len(lines) > rules.maxLines:
            issues.append(QCItem(cue=i, type="too_many_lines", value=float(len(lines))))
        for ln in lines:
            if len(ln) > rules.maxCharsPerLine:
                issues.append(QCItem(cue=i, type="line_too_long", value=float(len(ln))))

    return QCReport(issuesCount=len(issues), issues=issues)


# ----------------------------
# Endpoints
# ----------------------------

@app.get("/")
def root():
    return {"ok": True}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/build-captions", response_model=BuildResponse)
def build_captions(payload: BuildRequest):
    rules = payload.rules
    events = payload.events

    # ✅ PRO GUARDRAIL: strip bracket tokens from words when events exist
    words = strip_bracket_tokens_from_words_if_events_present(payload.words, events)

    # 1) build speech cues from words
    cues = _merge_words_to_cues(words, rules)

    # 2) insert events (music / foreign language)
    cues = _insert_events(cues, events, rules)

    # 3) QC + exports
    qc = qc_cues(cues, rules)
    srt = cues_to_srt(cues)
    vtt = cues_to_vtt(cues)
    scc = cues_to_scc(cues, rules)

    return BuildResponse(
        rules=rules,
        cues=cues,
        srt=srt,
        vtt=vtt,
        scc=scc,
        qc=qc,
    )
