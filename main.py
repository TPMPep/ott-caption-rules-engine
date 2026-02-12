import os
import uuid
import re
import math
from typing import Any, Dict, List, Optional, Literal, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ============================================================
# AI CC CREATOR — PRO BROADCAST ENFORCEMENT ENGINE
# FastAPI + AssemblyAI Orchestrator + NBCU/SDH Caption Engine
# ============================================================

app = FastAPI(title="AI CC Creator — Pro Broadcast Engine", version="2.0.0")

# ---------------------------
# Environment
# ---------------------------

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")

# Comma-separated list of allowed origins:
# Example:
# ALLOWED_ORIGINS="https://preview-sandbox--xxxx.base44.app,https://ai-cc-creator.yourdomain.com"
ALLOWED_ORIGINS_RAW = os.getenv("ALLOWED_ORIGINS", "").strip()

if not PUBLIC_BASE_URL:
    PUBLIC_BASE_URL = "http://localhost:8000"

def _parse_allowed_origins(raw: str) -> List[str]:
    if not raw:
        return []
    raw = raw.strip()
    if raw == "*":
        return ["*"]
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]

_allowed_origins = _parse_allowed_origins(ALLOWED_ORIGINS_RAW)

# CORS for Base44 preview/prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins if _allowed_origins else [],
    allow_origin_regex=None,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

# ---------------------------
# Models
# ---------------------------

class Rules(BaseModel):
    # Caption geometry
    maxCharsPerLine: int = 32
    maxLines: int = 2

    # Speed
    maxCPS: float = 17.0

    # Timing
    minDurationMs: int = 1000
    maxDurationMs: int = 7000
    minGapMs: int = 80

    # Behavior
    preferPunctuationBreaks: bool = True

    # Output modes
    mode: Literal["standard", "sdh", "nbcu_strict"] = "nbcu_strict"

    # Speaker policy:
    # - "none" -> do not show
    # - "brackets" -> [A] style
    # - "dash" -> — speaker
    speakerStyle: Literal["none", "brackets", "dash"] = "none"

    # SDH policy
    enableSoundCues: bool = True
    soundCueMinGapMs: int = 900         # silence threshold to consider SDH insert
    soundCueMaxDurationMs: int = 5000   # don't create absurdly long SDH cues

    # SCC specifics
    sccFrameRate: float = 29.97
    startAtHour00: bool = True

class Word(BaseModel):
    text: str
    start: int
    end: int
    speaker: Optional[str] = "A"
    confidence: Optional[float] = None

class Event(BaseModel):
    type: Literal["music", "foreign_language", "sound_effect"]
    start: int
    end: int
    text: Optional[str] = None
    language: Optional[str] = None

class FormatRequest(BaseModel):
    rules: Rules = Field(default_factory=Rules)
    words: Optional[List[Word]] = None
    cues: Optional[List[Dict[str, Any]]] = None
    events: Optional[List[Event]] = None

class JobCreateRequest(BaseModel):
    mediaUrl: str
    rules: Rules = Field(default_factory=Rules)
    speaker_labels: bool = True
    language_detection: bool = True

class JobStatus(BaseModel):
    id: str
    status: Literal["queued", "processing", "done", "error"]
    error: Optional[str] = None
    assembly_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None

# ---------------------------
# In-memory store (MVP)
# ---------------------------

JOBS: Dict[str, JobStatus] = {}
JOB_RULES: Dict[str, Rules] = {}

# ---------------------------
# Utilities
# ---------------------------

PUNCT_BREAK_RE = re.compile(r"[.!?…]+$")
SOFT_BREAK_RE = re.compile(r"[,;:]+$")

SFX_KEYWORDS = [
    "music",
    "applause",
    "laughter",
    "cheering",
    "crowd",
    "gasps",
    "sighs",
    "door",
    "ringing",
    "phone",
    "gunshot",
    "explosion",
    "thunder",
]

def _clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def _normalize_speaker(s: Optional[str]) -> str:
    if not s:
        return "A"
    s = str(s).strip()
    if not s:
        return "A"
    s = s.replace("Speaker ", "").strip()
    return s or "A"

def _is_bracket_token(t: str) -> bool:
    t = (t or "").strip()
    if not t:
        return False
    if t.startswith("[") and t.endswith("]"):
        return True
    if t.startswith("[") or t.endswith("]"):
        return True
    return False

def _strip_bracket_tokens(words: List[Word]) -> List[Word]:
    cleaned: List[Word] = []
    for w in words:
        t = (w.text or "").strip()
        if _is_bracket_token(t):
            continue
        cleaned.append(w)
    return cleaned

def _ms_to_srt_time(ms: int) -> str:
    h = ms // 3600000
    ms -= h * 3600000
    m = ms // 60000
    ms -= m * 60000
    s = ms // 1000
    ms -= s * 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def _ms_to_vtt_time(ms: int) -> str:
    h = ms // 3600000
    ms -= h * 3600000
    m = ms // 60000
    ms -= m * 60000
    s = ms // 1000
    ms -= s * 1000
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"

# SCC MVP (still placeholder — spec-perfect SCC is later)
def _ms_to_scc_timecode(ms: int, fps: float) -> str:
    fps_i = 30 if abs(fps - 29.97) < 0.05 else int(round(fps))
    total_seconds = ms / 1000.0
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = int(total_seconds % 60)
    frac = total_seconds - int(total_seconds)
    ff = int(round(frac * fps_i))
    ff = _clip(ff, 0, fps_i - 1)
    return f"{h:02}:{m:02}:{s:02}:{ff:02}"

def _text_to_fake_scc_hex(text: str) -> str:
    safe = text.encode("latin-1", errors="replace")
    parts = [f"{b:02x}" for b in safe[:120]]
    grouped = []
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            grouped.append(parts[i] + parts[i + 1])
        else:
            grouped.append(parts[i] + "20")
    return " ".join(grouped)

# ---------------------------
# Caption Helpers
# ---------------------------

def _split_into_semantic_units(words: List[Word]) -> List[List[Word]]:
    """
    Creates semantic chunks based on punctuation and speaker boundaries.
    This is the first major upgrade vs naive gap grouping.
    """
    if not words:
        return []

    units: List[List[Word]] = []
    cur: List[Word] = []

    prev_speaker = _normalize_speaker(words[0].speaker)

    for w in words:
        sp = _normalize_speaker(w.speaker)
        t = (w.text or "").strip()

        if not cur:
            cur = [w]
            prev_speaker = sp
            continue

        # speaker boundary = hard semantic boundary
        if sp != prev_speaker:
            units.append(cur)
            cur = [w]
            prev_speaker = sp
            continue

        cur.append(w)

        # punctuation boundary encourages semantic boundary
        if PUNCT_BREAK_RE.search(t):
            units.append(cur)
            cur = []

    if cur:
        units.append(cur)

    return units

def _wrap_lines(text: str, max_chars: int, max_lines: int) -> List[str]:
    """
    Greedy wrapper producing <= max_lines, each <= max_chars when possible.
    """
    words = text.split()
    if not words:
        return [""]

    lines: List[str] = []
    cur = ""

    def push(s: str):
        s = s.strip()
        if s:
            lines.append(s)

    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur = f"{cur} {w}"
        else:
            push(cur)
            cur = w
            if len(lines) >= max_lines:
                # overflow: stuff into last line
                lines[-1] = (lines[-1] + " " + cur).strip()
                cur = ""

    if cur:
        push(cur)

    # squash overflow
    if len(lines) > max_lines:
        head = lines[:max_lines - 1]
        tail = " ".join(lines[max_lines - 1:])
        lines = head + [tail]

    return lines

def _cue_text(text: str, rules: Rules, speaker: str) -> str:
    """
    Applies speaker style + wrapping.
    """
    speaker = _normalize_speaker(speaker)

    if rules.speakerStyle == "brackets":
        text = f"[{speaker}] {text}".strip()
    elif rules.speakerStyle == "dash":
        text = f"— {text}".strip()

    lines = _wrap_lines(text, rules.maxCharsPerLine, rules.maxLines)
    return "\n".join(lines).strip()

def _calc_cps(text: str, duration_ms: int) -> float:
    duration_ms = max(1, duration_ms)
    chars = len(text.replace("\n", ""))
    return chars / (duration_ms / 1000.0)

def _split_text_best_effort(text: str) -> Tuple[str, str]:
    """
    Splits text into 2 halves at best punctuation boundary.
    """
    tokens = text.split()
    if len(tokens) <= 1:
        return text, ""

    # try to split at comma/semicolon boundary near the middle
    mid = len(tokens) // 2
    best = None
    best_dist = 9999

    for i in range(1, len(tokens) - 1):
        tok = tokens[i]
        if tok.endswith(",") or tok.endswith(";") or tok.endswith(":"):
            dist = abs(i - mid)
            if dist < best_dist:
                best = i
                best_dist = dist

    if best is None:
        best = mid

    left = " ".join(tokens[:best + 1]).strip()
    right = " ".join(tokens[best + 1:]).strip()
    return left, right

# ---------------------------
# SDH Sound Cue Generation
# ---------------------------

def _infer_sound_events(words: List[Word], rules: Rules) -> List[Event]:
    """
    Heuristic SDH detection:
    - Inserts [♪ MUSIC ♪] / [APPLAUSE] / [LAUGHTER] when large gaps occur
    - This is AI-assist, not perfect, but massively better than nothing.
    """
    if not rules.enableSoundCues:
        return []

    if not words:
        return []

    words = sorted(words, key=lambda w: (w.start, w.end))
    events: List[Event] = []

    for i in range(len(words) - 1):
        a = words[i]
        b = words[i + 1]
        gap = b.start - a.end

        if gap < rules.soundCueMinGapMs:
            continue

        # create a short SDH cue in the gap window
        start = a.end
        end = min(b.start, start + rules.soundCueMaxDurationMs)

        # default guess: MUSIC
        txt = "[♪ MUSIC ♪]"

        # confidence-based: if last spoken word had low confidence, maybe laughter/crowd
        conf = a.confidence if a.confidence is not None else 1.0
        if conf < 0.55:
            txt = "[LAUGHTER]"

        events.append(Event(type="sound_effect", start=start, end=end, text=txt))

    # De-dupe overlapping events
    merged: List[Event] = []
    for e in events:
        if not merged:
            merged.append(e)
            continue
        prev = merged[-1]
        if e.start <= prev.end + 100:
            # merge
            prev.end = max(prev.end, e.end)
        else:
            merged.append(e)

    return merged

# ---------------------------
# Pro Cue Builder (Multi-pass Enforcement)
# ---------------------------

def _build_initial_cues(words: List[Word], rules: Rules) -> List[Dict[str, Any]]:
    """
    Build first-pass cues using semantic units.
    """
    units = _split_into_semantic_units(words)
    cues: List[Dict[str, Any]] = []

    for unit in units:
        if not unit:
            continue
        start = int(unit[0].start)
        end = int(unit[-1].end)
        speaker = _normalize_speaker(unit[0].speaker)

        text = " ".join((w.text or "").strip() for w in unit).strip()
        if not text:
            continue

        cues.append({
            "start": start,
            "end": end,
            "speaker": speaker,
            "rawText": text,
        })

    # Ensure monotonic
    cues = sorted(cues, key=lambda c: (c["start"], c["end"]))
    return cues

def _enforce_rules_on_cues(cues: List[Dict[str, Any]], rules: Rules) -> List[Dict[str, Any]]:
    """
    Iterative enforcement loop:
    - ensures max lines/char
    - ensures CPS
    - ensures min/max duration
    - splits/merges as needed
    """
    # hard limit to prevent infinite loops
    MAX_PASSES = 12

    def cue_text(c: Dict[str, Any]) -> str:
        return _cue_text(str(c.get("rawText") or ""), rules, str(c.get("speaker") or "A"))

    def cue_duration(c: Dict[str, Any]) -> int:
        return max(1, int(c["end"]) - int(c["start"]))

    def cue_has_hard_violations(c: Dict[str, Any]) -> bool:
        t = cue_text(c)
        dur = cue_duration(c)
        cps = _calc_cps(t, dur)
        lines = t.split("\n")
        if len(lines) > rules.maxLines:
            return True
        if any(len(ln) > rules.maxCharsPerLine for ln in lines):
            return True
        if cps > rules.maxCPS:
            return True
        if dur < rules.minDurationMs:
            return True
        if dur > rules.maxDurationMs:
            return True
        return False

    def split_cue(c: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw = str(c.get("rawText") or "")
        left, right = _split_text_best_effort(raw)
        if not right.strip():
            return [c]

        start = int(c["start"])
        end = int(c["end"])
        dur = max(1, end - start)
        mid = start + int(dur * 0.52)

        c1 = dict(c)
        c2 = dict(c)
        c1["rawText"] = left.strip()
        c2["rawText"] = right.strip()
        c1["end"] = mid
        c2["start"] = mid + rules.minGapMs

        # clamp
        if c2["start"] >= end:
            c2["start"] = mid
        c2["end"] = end

        return [c1, c2]

    def merge_with_next(i: int) -> bool:
        if i < 0 or i >= len(cues) - 1:
            return False
        a = cues[i]
        b = cues[i + 1]

        # only merge if same speaker
        if str(a.get("speaker")) != str(b.get("speaker")):
            return False

        a["rawText"] = (str(a.get("rawText") or "") + " " + str(b.get("rawText") or "")).strip()
        a["end"] = max(int(a["end"]), int(b["end"]))
        del cues[i + 1]
        return True

    # Pass loop
    for _pass in range(MAX_PASSES):
        changed = False

        i = 0
        while i < len(cues):
            c = cues[i]
            dur = cue_duration(c)

            # Enforce min duration by extending (if safe)
            if dur < rules.minDurationMs:
                c["end"] = int(c["start"]) + rules.minDurationMs
                changed = True

            # Enforce max duration by splitting
            if cue_duration(c) > rules.maxDurationMs:
                parts = split_cue(c)
                if len(parts) > 1:
                    cues[i:i + 1] = parts
                    changed = True
                    continue

            # Enforce CPS by splitting
            t = cue_text(c)
            cps = _calc_cps(t, cue_duration(c))
            if cps > rules.maxCPS:
                parts = split_cue(c)
                if len(parts) > 1:
                    cues[i:i + 1] = parts
                    changed = True
                    continue

            # Enforce line lengths by splitting
            lines = t.split("\n")
            if len(lines) > rules.maxLines or any(len(ln) > rules.maxCharsPerLine for ln in lines):
                parts = split_cue(c)
                if len(parts) > 1:
                    cues[i:i + 1] = parts
                    changed = True
                    continue

            # Enforce too-short cues by merging
            if cue_duration(c) < rules.minDurationMs and i < len(cues) - 1:
                if merge_with_next(i):
                    changed = True
                    continue

            i += 1

        # Normalize ordering + clamp overlaps
        cues = sorted(cues, key=lambda c: (int(c["start"]), int(c["end"])))
        for j in range(len(cues) - 1):
            a = cues[j]
            b = cues[j + 1]
            if int(b["start"]) < int(a["end"]) + rules.minGapMs:
                b["start"] = int(a["end"]) + rules.minGapMs
                if int(b["start"]) >= int(b["end"]):
                    b["end"] = int(b["start"]) + rules.minDurationMs

        if not changed:
            break

    # Finalize into delivery cues
    final: List[Dict[str, Any]] = []
    for c in cues:
        final.append({
            "start": int(c["start"]),
            "end": int(c["end"]),
            "speaker": _normalize_speaker(str(c.get("speaker") or "A")),
            "text": _cue_text(str(c.get("rawText") or ""), rules, str(c.get("speaker") or "A")),
            "kind": "dialogue",
        })

    return final

# ---------------------------
# QC (post-enforcement)
# ---------------------------

def qc_cues(cues: List[Dict[str, Any]], rules: Rules) -> Dict[str, Any]:
    issues = []

    for idx, c in enumerate(cues, start=1):
        start = int(c["start"])
        end = int(c["end"])
        text = str(c.get("text") or "")

        duration = max(1, end - start)
        lines = text.split("\n")

        for ln in lines:
            if len(ln) > rules.maxCharsPerLine:
                issues.append({"cue": idx, "type": "line_too_long", "value": len(ln)})

        if len(lines) > rules.maxLines:
            issues.append({"cue": idx, "type": "too_many_lines", "value": len(lines)})

        cps = _calc_cps(text, duration)
        if cps > rules.maxCPS:
            issues.append({"cue": idx, "type": "cps_high", "value": round(cps, 2)})

        if duration < rules.minDurationMs:
            issues.append({"cue": idx, "type": "too_short_ms", "value": duration})

        if duration > rules.maxDurationMs:
            issues.append({"cue": idx, "type": "too_long_ms", "value": duration})

    return {"issuesCount": len(issues), "issues": issues}

# ---------------------------
# Exports
# ---------------------------

def to_srt(cues: List[Dict[str, Any]]) -> str:
    out = []
    for i, c in enumerate(cues, start=1):
        out.append(str(i))
        out.append(f"{_ms_to_srt_time(int(c['start']))} --> {_ms_to_srt_time(int(c['end']))}")
        out.append(str(c.get("text") or ""))
        out.append("")
    return "\n".join(out).strip() + "\n"

def to_vtt(cues: List[Dict[str, Any]]) -> str:
    out = ["WEBVTT", ""]
    for c in cues:
        out.append(f"{_ms_to_vtt_time(int(c['start']))} --> {_ms_to_vtt_time(int(c['end']))}")
        out.append(str(c.get("text") or ""))
        out.append("")
    return "\n".join(out).strip() + "\n"

def to_scc(cues: List[Dict[str, Any]], rules: Rules) -> str:
    lines = ["Scenarist_SCC V1.0", ""]
    base_offset = 0

    for c in cues:
        start_ms = int(c["start"]) - base_offset
        if start_ms < 0:
            start_ms = 0
        tc = _ms_to_scc_timecode(start_ms, rules.sccFrameRate)
        text = str(c.get("text") or "").replace("\n", " ")
        hex_payload = "94ae 94ae 9420 9420 9470 9470 " + _text_to_fake_scc_hex(text) + " 942c 942c 942f 942f"
        lines.append(f"{tc}\t{hex_payload}")
        lines.append("")

    final_end = int(cues[-1]["end"]) if cues else 0
    lines.append(_ms_to_scc_timecode(max(0, final_end), rules.sccFrameRate) + "\t942c 942c")
    lines.append("")
    return "\n".join(lines)

# ---------------------------
# Core Formatter
# ---------------------------

def format_payload(req: FormatRequest) -> Dict[str, Any]:
    rules = req.rules

    # If cues provided, trust them (advanced usage)
    if req.cues is not None:
        cues = req.cues
        # normalize
        cues = sorted(cues, key=lambda c: (int(c["start"]), int(c["end"])))
        final_cues = []
        for c in cues:
            final_cues.append({
                "start": int(c["start"]),
                "end": int(c["end"]),
                "speaker": _normalize_speaker(c.get("speaker")),
                "text": _cue_text(str(c.get("text") or ""), rules, _normalize_speaker(c.get("speaker"))),
                "kind": c.get("kind") or "dialogue",
            })
        cues_out = final_cues

    else:
        if not req.words:
            raise HTTPException(status_code=422, detail="You must provide either 'words' or 'cues'.")

        words = [Word(**w.model_dump()) for w in req.words]

        # PRO GUARD: strip bracket tokens if events exist
        if req.events:
            words = _strip_bracket_tokens(words)

        # Build initial semantic cues
        initial = _build_initial_cues(words, rules)

        # Enforce hard rules
        cues_out = _enforce_rules_on_cues(initial, rules)

        # SDH inference layer
        inferred_events: List[Event] = []
        if rules.enableSoundCues:
            inferred_events = _infer_sound_events(words, rules)

        # Combine explicit events + inferred events
        combined_events: List[Event] = []
        if req.events:
            combined_events.extend(req.events)
        combined_events.extend(inferred_events)

        # Inject events as their own cues
        if combined_events:
            for e in combined_events:
                if e.type == "music":
                    txt = e.text or "[♪ MUSIC ♪]"
                elif e.type == "foreign_language":
                    lang = e.language or "Unknown"
                    txt = f"[Speaking {lang}]"
                else:
                    txt = e.text or "[SOUND]"

                cues_out.append({
                    "start": int(e.start),
                    "end": int(e.end),
                    "speaker": "A",
                    "text": txt.strip(),
                    "kind": "sdh",
                })

            cues_out = sorted(cues_out, key=lambda c: (int(c["start"]), int(c["end"])))

    qc = qc_cues(cues_out, rules)

    return {
        "rules": rules.model_dump(),
        "cues": cues_out,
        "srt": to_srt(cues_out),
        "vtt": to_vtt(cues_out),
        "scc": to_scc(cues_out, rules),
        "qc": qc,
        "meta": {
            "engineVersion": "2.0.0",
            "mode": rules.mode,
            "speakerStyle": rules.speakerStyle,
            "soundCuesEnabled": rules.enableSoundCues,
        }
    }

# ---------------------------
# Routes
# ---------------------------

@app.get("/")
def root():
    return {"ok": True, "service": "ai-cc-creator", "version": "2.0.0"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/v1/format")
def format_endpoint(req: FormatRequest):
    return format_payload(req)

@app.post("/v1/jobs", response_model=JobStatus)
async def create_job(req: JobCreateRequest):
    if not ASSEMBLYAI_API_KEY:
        raise HTTPException(status_code=500, detail="ASSEMBLYAI_API_KEY is not set in Railway Variables.")
    if not WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="WEBHOOK_SECRET is not set in Railway Variables.")
    if not PUBLIC_BASE_URL:
        raise HTTPException(status_code=500, detail="PUBLIC_BASE_URL is not set in Railway Variables.")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = JobStatus(id=job_id, status="queued", assembly_id=None)
    JOB_RULES[job_id] = req.rules

    webhook_url = f"{PUBLIC_BASE_URL}/v1/webhooks/assemblyai"

    payload = {
        "audio_url": req.mediaUrl,
        "speaker_labels": req.speaker_labels,
        "language_detection": req.language_detection,

        # AssemblyAI now requires this:
        "speech_models": ["universal-3-pro"],

        # Webhook orchestration
        "webhook_url": webhook_url,
        "webhook_auth_header_name": "X-Webhook-Token",
        "webhook_auth_header_value": WEBHOOK_SECRET,
    }

    headers = {"Authorization": ASSEMBLYAI_API_KEY, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=90.0) as client:
        r = await client.post("https://api.assemblyai.com/v2/transcript", json=payload, headers=headers)
        if r.status_code >= 300:
            JOBS[job_id].status = "error"
            JOBS[job_id].error = f"AssemblyAI submit failed: {r.status_code} {r.text}"
            return JOBS[job_id]

        data = r.json()
        assembly_id = data.get("id")
        JOBS[job_id].status = "processing"
        JOBS[job_id].assembly_id = assembly_id

    return JOBS[job_id]

@app.get("/v1/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    js = JOBS.get(job_id)
    if not js:
        raise HTTPException(status_code=404, detail="Job not found")
    return js

@app.post("/v1/webhooks/assemblyai")
async def assemblyai_webhook(request: Request):
    token = request.headers.get("X-Webhook-Token", "")
    if not WEBHOOK_SECRET or token != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid webhook token")

    body = await request.json()
    assembly_id = body.get("transcript_id") or body.get("id")
    status = body.get("status")

    # Find job
    job_id = None
    for jid, js in JOBS.items():
        if js.assembly_id == assembly_id:
            job_id = jid
            break

    if not job_id:
        return JSONResponse({"ok": True, "ignored": True})

    if status in ("error", "failed"):
        JOBS[job_id].status = "error"
        JOBS[job_id].error = body.get("error") or "AssemblyAI transcript failed"
        return JSONResponse({"ok": True})

    if status != "completed":
        return JSONResponse({"ok": True})

    # Fetch transcript details (words + confidence + speaker)
    headers = {"Authorization": ASSEMBLYAI_API_KEY}
    async with httpx.AsyncClient(timeout=90.0) as client:
        r = await client.get(f"https://api.assemblyai.com/v2/transcript/{assembly_id}", headers=headers)
        if r.status_code >= 300:
            JOBS[job_id].status = "error"
            JOBS[job_id].error = f"AssemblyAI get transcript failed: {r.status_code} {r.text}"
            return JSONResponse({"ok": True})
        t = r.json()

    words_raw = t.get("words") or []
    words: List[Word] = []
    for w in words_raw:
        words.append(Word(
            text=w.get("text", ""),
            start=int(w.get("start", 0)),
            end=int(w.get("end", 0)),
            speaker=_normalize_speaker(w.get("speaker") or "A"),
            confidence=w.get("confidence", None),
        ))

    # Events: We infer SDH in Railway (not AssemblyAI)
    events: List[Event] = []

    rules = JOB_RULES.get(job_id, Rules())

    result = format_payload(FormatRequest(words=words, events=events, rules=rules))

    JOBS[job_id].status = "done"
    JOBS[job_id].result = result
    JOBS[job_id].error = None

    return JSONResponse({"ok": True})
