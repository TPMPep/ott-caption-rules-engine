from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Literal
import re

app = FastAPI(title="OTT Caption Rules Engine", version="1.0.0")

# -----------------------------
# Models
# -----------------------------

class Word(BaseModel):
    text: str
    start: int
    end: int
    speaker: Optional[str] = None
    confidence: Optional[float] = None

class Event(BaseModel):
    # Use events to inject things like:
    # - [Speaking Spanish]
    # - [♪ MUSIC ♪]
    type: Literal["foreign_language", "music", "custom"]
    start: int
    end: int
    language: Optional[str] = None
    text: Optional[str] = None
    speaker: Optional[str] = "A"

class Rules(BaseModel):
    # Caption layout + timing
    maxCharsPerLine: int = 32
    maxLines: int = 2
    maxCPS: float = 17.0            # QC target (NOT used to split)
    minDurationMs: int = 1000
    maxDurationMs: int = 7000
    minGapMs: int = 80
    preferPunctuationBreaks: bool = True

    # SCC / broadcast-ish settings
    sccFrameRate: float = 29.97     # default; can override per job
    startAtHour00: bool = True      # Andrea test exception: start at 00:00:00:00

class BuildRequest(BaseModel):
    words: List[Word]
    events: Optional[List[Event]] = []
    rules: Optional[Rules] = Rules()

# -----------------------------
# Helpers
# -----------------------------

PUNCT_BOUNDARY = re.compile(r"[.!?]$|[,;:]$")

def ms_to_srt_ts(ms: int) -> str:
    ms = max(int(ms), 0)
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def ms_to_vtt_ts(ms: int) -> str:
    ms = max(int(ms), 0)
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"

def wrap_lines(text: str, max_len: int) -> List[str]:
    t = re.sub(r"\s+", " ", text.strip())
    if not t:
        return []
    words = t.split(" ")
    lines: List[str] = []
    cur = ""
    for w in words:
        nxt = w if not cur else f"{cur} {w}"
        if len(nxt) <= max_len:
            cur = nxt
        else:
            if cur:
                lines.append(cur)
                cur = w
            else:
                lines.append(w[:max_len])
                cur = w[max_len:]
    if cur:
        lines.append(cur)
    return lines

def calc_cps(text: str, dur_ms: int) -> float:
    chars = len(text.replace("\n", ""))
    dur = max(dur_ms / 1000.0, 0.001)
    return chars / dur

def is_boundary_token(token: str) -> bool:
    return bool(PUNCT_BOUNDARY.search(token))

def cue_text_from_words(words: List[Word], rules: Rules) -> str:
    raw = " ".join(w.text for w in words).strip()
    lines = wrap_lines(raw, rules.maxCharsPerLine)
    return "\n".join(lines[:rules.maxLines])

def would_violate_layout_or_duration(words: List[Word], start_ms: int, end_ms: int, rules: Rules) -> bool:
    dur = end_ms - start_ms
    if dur > rules.maxDurationMs:
        return True
    raw = " ".join(w.text for w in words).strip()
    wrapped = wrap_lines(raw, rules.maxCharsPerLine)
    if len(wrapped) > rules.maxLines:
        return True
    if any(len(ln) > rules.maxCharsPerLine for ln in wrapped):
        return True
    return False

def build_cues_from_words(words: List[Word], rules: Rules) -> List[Dict[str, Any]]:
    ws = [w for w in words if w.text and w.end >= w.start]
    ws.sort(key=lambda w: (w.start, w.end))

    cues: List[Dict[str, Any]] = []
    current: List[Word] = []
    cue_start: Optional[int] = None

    def finalize(chunk: List[Word]):
        if not chunk:
            return
        cues.append({
            "start": chunk[0].start,
            "end": chunk[-1].end,
            "text": cue_text_from_words(chunk, rules),
            "speaker": chunk[0].speaker or "A"
        })

    for w in ws:
        if not current:
            current = [w]
            cue_start = w.start
            continue

        candidate = current + [w]
        start = cue_start if cue_start is not None else candidate[0].start
        end = candidate[-1].end

        if would_violate_layout_or_duration(candidate, start, end, rules):
            finalize(current)
            current = [w]
            cue_start = w.start
            continue

        current = candidate

        dur = current[-1].end - (cue_start if cue_start is not None else current[0].start)
        if rules.preferPunctuationBreaks and dur >= rules.minDurationMs and is_boundary_token(w.text):
            finalize(current)
            current = []
            cue_start = None

    finalize(current)

    # Normalize timing: enforce gaps, extend ends when possible
    cues.sort(key=lambda c: (c["start"], c["end"]))

    for i, c in enumerate(cues):
        # enforce start after previous end + gap
        if i > 0:
            prev = cues[i - 1]
            min_start = prev["end"] + rules.minGapMs
            if c["start"] < min_start:
                c["start"] = min_start

        # ensure end >= start
        if c["end"] <= c["start"]:
            c["end"] = c["start"] + 1

        # clamp max duration
        if c["end"] - c["start"] > rules.maxDurationMs:
            c["end"] = c["start"] + rules.maxDurationMs

        # try to extend to minDuration and reduce CPS if there's room
        target_end = c["start"] + rules.minDurationMs
        if i < len(cues) - 1:
            next_c = cues[i + 1]
            latest_end = next_c["start"] - rules.minGapMs
            if c["end"] < target_end:
                c["end"] = min(target_end, latest_end)
            max_allowed = min(c["start"] + rules.maxDurationMs, latest_end)
            if c["end"] < max_allowed:
                c["end"] = min(max_allowed, c["end"] + 500)
        else:
            if c["end"] < target_end:
                c["end"] = target_end

    return cues

def event_to_cue(ev: Event) -> Dict[str, Any]:
    if ev.type == "foreign_language":
        lang = (ev.language or "language").strip()
        txt = f"[Speaking {lang}]"
    elif ev.type == "music":
        txt = ev.text.strip() if ev.text else "[♪ MUSIC ♪]"
    else:
        txt = ev.text.strip() if ev.text else "[EVENT]"
    return {"start": ev.start, "end": ev.end, "text": txt, "speaker": ev.speaker or "A"}

def merge_cues_with_events(cues: List[Dict[str, Any]], events: List[Event], rules: Rules) -> List[Dict[str, Any]]:
    # Inject event cues, then re-sort and enforce minGap.
    ev_cues = [event_to_cue(e) for e in (events or [])]
    all_cues = cues + ev_cues
    all_cues.sort(key=lambda c: (c["start"], c["end"]))

    # Enforce gap + no negative durations
    fixed: List[Dict[str, Any]] = []
    for c in all_cues:
        c["start"] = max(int(c["start"]), 0)
        c["end"] = max(int(c["end"]), c["start"] + 1)
        if not fixed:
            fixed.append(c)
            continue
        prev = fixed[-1]
        min_start = prev["end"] + rules.minGapMs
        if c["start"] < min_start:
            c["start"] = min_start
            if c["end"] <= c["start"]:
                c["end"] = c["start"] + 1
        fixed.append(c)
    return fixed

def cues_to_srt(cues: List[Dict[str, Any]]) -> str:
    out = []
    for i, c in enumerate(cues, start=1):
        out.append(str(i))
        out.append(f"{ms_to_srt_ts(c['start'])} --> {ms_to_srt_ts(c['end'])}")
        out.append(c["text"])
        out.append("")
    return "\n".join(out)

def cues_to_vtt(cues: List[Dict[str, Any]]) -> str:
    out = ["WEBVTT", ""]
    for c in cues:
        out.append(f"{ms_to_vtt_ts(c['start'])} --> {ms_to_vtt_ts(c['end'])}")
        out.append(c["text"])
        out.append("")
    return "\n".join(out)

def qc_report(cues: List[Dict[str, Any]], rules: Rules) -> Dict[str, Any]:
    issues = []
    for i, c in enumerate(cues, start=1):
        lines = c["text"].splitlines()
        dur = c["end"] - c["start"]

        if len(lines) > rules.maxLines:
            issues.append({"cue": i, "type": "too_many_lines", "value": len(lines)})

        for ln in lines:
            if len(ln) > rules.maxCharsPerLine:
                issues.append({"cue": i, "type": "line_too_long", "value": len(ln)})

        if dur < rules.minDurationMs:
            issues.append({"cue": i, "type": "too_short_ms", "value": dur})

        if dur > rules.maxDurationMs:
            issues.append({"cue": i, "type": "too_long_ms", "value": dur})

        cps = calc_cps(c["text"], dur)
        if cps > rules.maxCPS:
            issues.append({"cue": i, "type": "cps_high", "value": round(cps, 2)})

        if i > 1:
            prev = cues[i - 2]
            gap = c["start"] - prev["end"]
            if gap < rules.minGapMs:
                issues.append({"cue": i, "type": "overlap_or_gap_violation", "value": gap})

    return {"issuesCount": len(issues), "issues": issues}

def cues_to_scc(cues: List[Dict[str, Any]], rules: Rules) -> str:
    """
    SCC is binary-encoded CEA-608 hex — do NOT hand-roll it.
    We convert from SRT using a proven library.
    """
    try:
        from pycaption import CaptionConverter
        from pycaption import SRTReader, SCCWriter
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"SCC export requires 'pycaption' installed. Error: {str(e)}"
        )

    # Andrea test: start at 00:00:00:00
    # Our cue times already start near 0ms. If you want to force rebase to 0, do it here.
    if rules.startAtHour00 and cues:
        base = cues[0]["start"]
        for c in cues:
            c["start"] -= base
            c["end"] -= base
            c["start"] = max(c["start"], 0)
            c["end"] = max(c["end"], c["start"] + 1)

    srt = cues_to_srt(cues)

    converter = CaptionConverter()
    converter.read(srt, SRTReader())
    caption_set = converter.captions

    writer = SCCWriter()
    # Some versions of pycaption accept frame_rate; others ignore it.
    try:
        return writer.write(caption_set, frame_rate=rules.sccFrameRate)
    except TypeError:
        return writer.write(caption_set)

# -----------------------------
# Endpoints
# -----------------------------

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/build-captions")
def build(req: BuildRequest) -> Dict[str, Any]:
    rules = req.rules or Rules()

    # 1) Build from words
    cues = build_cues_from_words(req.words, rules)

    # 2) Inject NBCU/Andrea events (foreign language, music, etc.)
    cues = merge_cues_with_events(cues, req.events or [], rules)

    # 3) Outputs
    srt = cues_to_srt(cues)
    vtt = cues_to_vtt(cues)
    qc = qc_report(cues, rules)

    # 4) SCC output
    scc = cues_to_scc([dict(c) for c in cues], rules)

    return {
        "rules": rules.model_dump(),
        "cues": cues,
        "srt": srt,
        "vtt": vtt,
        "scc": scc,
        "qc": qc
    }

@app.post("/export-scc")
def export_scc(req: BuildRequest) -> Dict[str, Any]:
    rules = req.rules or Rules()
    cues = build_cues_from_words(req.words, rules)
    cues = merge_cues_with_events(cues, req.events or [], rules)
    scc = cues_to_scc([dict(c) for c in cues], rules)
    return {"scc": scc, "qc": qc_report(cues, rules), "rules": rules.model_dump()}
