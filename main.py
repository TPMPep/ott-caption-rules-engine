from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import re

app = FastAPI(title="OTT Caption Rules Engine", version="0.3.0")

# -----------------------------
# Models
# -----------------------------
class Word(BaseModel):
    text: str
    start: int  # ms
    end: int    # ms
    speaker: Optional[str] = None
    confidence: Optional[float] = None

class Rules(BaseModel):
    # Preset A defaults (OTT broadcast-ish)
    maxCharsPerLine: int = 32
    maxLines: int = 2
    maxCPS: float = 17.0
    minDurationMs: int = 1000
    maxDurationMs: int = 7000
    minGapMs: int = 80

    preferPunctuationBreaks: bool = True

class BuildRequest(BaseModel):
    words: List[Word]
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
    """Greedy word wrap."""
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
                # very long token: hard split
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

def would_violate(words: List[Word], start_ms: int, end_ms: int, rules: Rules) -> bool:
    dur = end_ms - start_ms
    text = cue_text_from_words(words, rules)
    lines = text.splitlines() if text else []
    # too many lines after wrapping
    if len(wrap_lines(" ".join(w.text for w in words).strip(), rules.maxCharsPerLine)) > rules.maxLines:
        return True
    # any line too long (shouldn't happen, but guard)
    if any(len(ln) > rules.maxCharsPerLine for ln in lines):
        return True
    # duration too long
    if dur > rules.maxDurationMs:
        return True
    # cps too high (only meaningful when dur > 0)
    if dur > 0 and calc_cps(text, dur) > rules.maxCPS:
        return True
    return False

# -----------------------------
# Core caption builder
# -----------------------------
def build_cues(words: List[Word], rules: Rules) -> List[Dict[str, Any]]:
    """
    Build phrase-level cues from word timestamps.

    Strategy:
      - Keep appending words to current cue until adding next word would violate constraints.
      - Prefer breaking at punctuation, but only after minDurationMs.
      - Enforce non-overlap + minGapMs in a post-pass.
    """
    # Clean input
    ws = [w for w in words if w.text and w.end >= w.start]
    ws.sort(key=lambda w: (w.start, w.end))

    cues: List[Dict[str, Any]] = []

    current: List[Word] = []
    cue_start: Optional[int] = None

    def finalize(chunk: List[Word]):
        if not chunk:
            return
        start = chunk[0].start
        end = chunk[-1].end
        text = cue_text_from_words(chunk, rules)
        cues.append({
            "start": start,
            "end": end,
            "text": text,
            "speaker": chunk[0].speaker
        })

    for w in ws:
        if not current:
            current = [w]
            cue_start = w.start
            continue

        candidate = current + [w]
        start = cue_start if cue_start is not None else candidate[0].start
        end = candidate[-1].end

        # If adding this word breaks hard constraints, finalize current and start new cue
        if would_violate(candidate, start, end, rules):
            finalize(current)
            current = [w]
            cue_start = w.start
            continue

        # Otherwise accept
        current = candidate

        # If we have enough duration and hit punctuation, it's a good place to break
        dur = current[-1].end - (cue_start if cue_start is not None else current[0].start)
        if rules.preferPunctuationBreaks and dur >= rules.minDurationMs and is_boundary_token(w.text):
            finalize(current)
            current = []
            cue_start = None

    finalize(current)

    # -----------------------------
    # Post-pass timing normalization
    # -----------------------------
    cues.sort(key=lambda c: (c["start"], c["end"]))

    # Enforce gaps + min duration without overlaps
    for i, c in enumerate(cues):
        # Start must be >= previous end + gap
        if i > 0:
            prev = cues[i - 1]
            min_start = prev["end"] + rules.minGapMs
            if c["start"] < min_start:
                c["start"] = min_start

        # Ensure end >= start (and try to satisfy minDuration)
        if c["end"] <= c["start"]:
            c["end"] = c["start"] + rules.minDurationMs

        # Clamp max duration
        if c["end"] - c["start"] > rules.maxDurationMs:
            c["end"] = c["start"] + rules.maxDurationMs

        # Try to enforce minDuration using available gap to next cue
        dur = c["end"] - c["start"]
        if dur < rules.minDurationMs:
            need = rules.minDurationMs - dur
            if i < len(cues) - 1:
                next_c = cues[i + 1]
                latest_end = next_c["start"] - rules.minGapMs
                c["end"] = min(latest_end, c["end"] + need)
            else:
                c["end"] = c["end"] + need

        # Final overlap guard with next cue
        if i < len(cues) - 1:
            next_c = cues[i + 1]
            if c["end"] > next_c["start"] - rules.minGapMs:
                c["end"] = max(c["start"] + 1, next_c["start"] - rules.minGapMs)

    return cues

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

        # overlap check
        if i > 1:
            prev = cues[i - 2]
            if c["start"] < prev["end"] + rules.minGapMs:
                issues.append({"cue": i, "type": "overlap_or_gap_violation", "value": c["start"] - prev["end"]})

    return {"issuesCount": len(issues), "issues": issues}

# -----------------------------
# API
# -----------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/build-captions")
def build(req: BuildRequest) -> Dict[str, Any]:
    rules = req.rules or Rules()
    cues = build_cues(req.words, rules)
    return {
        "rules": rules.model_dump(),
        "cues": cues,
        "srt": cues_to_srt(cues),
        "vtt": cues_to_vtt(cues),
        "qc": qc_report(cues, rules),
    }
