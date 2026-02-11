from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import re

app = FastAPI(title="OTT Caption Rules Engine", version="0.2.0")

# -----------------------------
# Models
# -----------------------------
class Word(BaseModel):
    # AssemblyAI word-level timestamps typically include these fields
    text: str
    start: int  # ms
    end: int    # ms
    speaker: Optional[str] = None
    confidence: Optional[float] = None

class Rules(BaseModel):
    # Common OTT defaults (configurable per request)
    maxCharsPerLine: int = 32
    maxLines: int = 2
    maxCPS: float = 17.0
    minDurationMs: int = 1000
    maxDurationMs: int = 7000
    minGapMs: int = 80

    # Optional behavior toggles (future-proof)
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
    words = t.split(" ") if t else []
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
                # extremely long token fallback
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

# -----------------------------
# Core caption builder
# -----------------------------
def build_cues(words: List[Word], rules: Rules) -> List[Dict[str, Any]]:
    """
    Build cues from word-level timestamps.

    Best-effort constraints:
      - max chars/line
      - max lines
      - max CPS
      - min/max duration
      - no overlaps
    """
    cues: List[Dict[str, Any]] = []
    current: List[Word] = []

    def finalize(chunk: List[Word]):
        if not chunk:
            return
        start = chunk[0].start
        end = chunk[-1].end
        raw = " ".join(w.text for w in chunk).strip()
        lines = wrap_lines(raw, rules.maxCharsPerLine)

        # If > maxLines, split chunk and retry (best-effort)
        if len(lines) > rules.maxLines and len(chunk) > 1:
            mid = max(1, len(chunk) // 2)
            finalize(chunk[:mid])
            finalize(chunk[mid:])
            return

        cues.append({
            "start": start,
            "end": end,
            "text": "\n".join(lines[:rules.maxLines]),
            "speaker": chunk[0].speaker
        })

    for w in words:
        if not w.text:
            continue

        current.append(w)

        start = current[0].start
        end = current[-1].end
        dur = end - start

        raw = " ".join(x.text for x in current).strip()
        lines = wrap_lines(raw, rules.maxCharsPerLine)
        text = "\n".join(lines[:rules.maxLines])

        too_many_lines = len(lines) > rules.maxLines
        too_fast = calc_cps(text, dur) > rules.maxCPS
        too_long = dur > rules.maxDurationMs

        if too_many_lines or too_fast or too_long:
            # Choose a cut point (prefer punctuation)
            cut_at = None
            if rules.preferPunctuationBreaks:
                for j in range(len(current) - 2, 0, -1):
                    if is_boundary_token(current[j].text):
                        cut_at = j + 1
                        break
            if cut_at is None:
                cut_at = max(1, len(current) - 1)

            left = current[:cut_at]
            right = current[cut_at:]

            finalize(left)
            current = right

    finalize(current)

    # Post-process: enforce min duration by extending into available gaps; prevent overlap
    cues.sort(key=lambda c: (c["start"], c["end"]))
    for i, c in enumerate(cues):
        # clamp max duration
        if c["end"] - c["start"] > rules.maxDurationMs:
            c["end"] = c["start"] + rules.maxDurationMs

        # enforce min duration
        dur = c["end"] - c["start"]
        if dur < rules.minDurationMs:
            needed = rules.minDurationMs - dur
            if i < len(cues) - 1:
                gap = cues[i + 1]["start"] - c["end"]
                take = min(needed, max(0, gap - rules.minGapMs))
                c["end"] += take
            else:
                c["end"] += needed

        # prevent overlap with next cue
        if i < len(cues) - 1 and c["end"] > cues[i + 1]["start"] - rules.minGapMs:
            c["end"] = max(c["start"] + rules.minDurationMs, cues[i + 1]["start"] - rules.minGapMs)

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

        value_cps = calc_cps(c["text"], dur)
        if value_cps > rules.maxCPS:
            issues.append({"cue": i, "type": "cps_high", "value": round(value_cps, 2)})

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
