import os
import uuid
import re
from typing import Any, Dict, List, Optional, Literal, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="AI CC Creator - Broadcast Rules Engine + AssemblyAI Orchestrator", version="2.0.0")

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
    # Visual layout rules
    maxCharsPerLine: int = 32
    maxLines: int = 2

    # Speed rule
    maxCPS: float = 17.0

    # Timing constraints
    minDurationMs: int = 1000
    maxDurationMs: int = 7000
    minGapMs: int = 80

    preferPunctuationBreaks: bool = True

    # SCC specifics
    sccFrameRate: float = 29.97
    startAtHour00: bool = True

    # SDH / Broadcast formatting
    enableDashSpeakerLines: bool = True   # "- Speaker line"
    speakerStyle: Literal["none", "brackets"] = "brackets"  # "[A]" or none

    # Sound cues
    enableSoundCues: bool = True


class Word(BaseModel):
    text: str
    start: int
    end: int
    speaker: Optional[str] = "A"


class Event(BaseModel):
    type: Literal["music", "foreign_language", "sound"]
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
# Helpers
# ---------------------------

PUNCT_END_RE = re.compile(r"[.!?…]+$")
HARD_BREAK_RE = re.compile(r"[:;]$")

SOUND_CUE_WHITELIST = {
    "[♪ MUSIC ♪]",
    "[MUSIC]",
    "[APPLAUSE]",
    "[LAUGHTER]",
    "[DOOR SLAMS]",
    "[DOOR SLAM]",
    "[GUNSHOT]",
    "[GUNSHOTS]",
    "[SCREAMING]",
}


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


# ---------------------------
# Broadcast-grade wrapping
# ---------------------------

def wrap_to_lines(text: str, max_chars: int, max_lines: int) -> List[str]:
    """
    Greedy wrap that tries to fill lines up to max_chars.
    Does NOT create tiny lines unless unavoidable.
    """
    text = (text or "").strip()
    if not text:
        return [""]

    words = text.split()
    lines: List[str] = []
    cur = ""

    def push():
        nonlocal cur
        if cur.strip():
            lines.append(cur.strip())
        cur = ""

    for w in words:
        if not cur:
            cur = w
        else:
            if len(cur) + 1 + len(w) <= max_chars:
                cur = cur + " " + w
            else:
                push()
                cur = w
                if len(lines) >= max_lines:
                    # Stuff everything into last line if overflow
                    lines[-1] = (lines[-1] + " " + cur).strip()
                    cur = ""

    if cur:
        push()

    if len(lines) > max_lines:
        head = lines[: max_lines - 1]
        tail = " ".join(lines[max_lines - 1 :])
        lines = head + [tail]

    return lines[:max_lines]


def format_speaker_prefix(speaker: str, rules: Rules) -> str:
    """
    For now:
    - brackets => "[A] "
    - none => ""
    """
    if rules.speakerStyle == "none":
        return ""
    return f"[{speaker}] "


def compute_cps(text: str, start_ms: int, end_ms: int) -> float:
    dur = max(1, end_ms - start_ms)
    chars = len(text.replace("\n", ""))
    return chars / (dur / 1000.0)


# ---------------------------
# Core cue builder (PRO)
# ---------------------------

def build_cues_from_words_pro(words: List[Word], rules: Rules) -> List[Dict[str, Any]]:
    """
    PRO cue builder:
    - Uses AssemblyAI word start/end as truth.
    - Packs words into cues until we hit max lines/chars or maxDuration.
    - Avoids 1-3 word flashes unless unavoidable.
    - Speaker-aware.
    - Does NOT extend end times (prevents drift).
    """
    if not words:
        return []

    words = sorted(words, key=lambda w: (w.start, w.end))

    cues: List[Dict[str, Any]] = []

    cur_words: List[Word] = []
    cur_speaker = _normalize_speaker(words[0].speaker)
    cur_start = words[0].start
    cur_end = words[0].end

    def render_words_to_text(ws: List[Word], speaker: str) -> str:
        raw = " ".join((w.text or "").strip() for w in ws).strip()
        prefix = format_speaker_prefix(speaker, rules)
        return (prefix + raw).strip()

    def can_add_word(ws: List[Word], w: Word, speaker: str) -> bool:
        # Speaker changes cannot be mixed in same cue (we handle dash lines separately later)
        if _normalize_speaker(w.speaker) != speaker:
            return False

        proposed = ws + [w]
        text = render_words_to_text(proposed, speaker)

        # Wrap and check constraints
        lines = wrap_to_lines(text, rules.maxCharsPerLine, rules.maxLines)
        if len(lines) > rules.maxLines:
            return False
        for ln in lines:
            if len(ln) > rules.maxCharsPerLine:
                return False

        # Duration constraint
        start = proposed[0].start
        end = proposed[-1].end
        if (end - start) > rules.maxDurationMs:
            return False

        return True

    def flush():
        nonlocal cur_words, cur_start, cur_end, cur_speaker
        if not cur_words:
            return

        start = cur_words[0].start
        end = cur_words[-1].end
        text = render_words_to_text(cur_words, cur_speaker)

        # Final wrap
        lines = wrap_to_lines(text, rules.maxCharsPerLine, rules.maxLines)
        text = "\n".join(lines).strip()

        # IMPORTANT: do not extend end time
        # If too short, we flag it in QC, but we DO NOT drift timing.
        cues.append({
            "start": int(start),
            "end": int(end),
            "text": text,
            "speaker": cur_speaker,
        })

        cur_words = []

    for w in words:
        w_sp = _normalize_speaker(w.speaker)

        # If speaker changes, flush
        if w_sp != cur_speaker:
            flush()
            cur_speaker = w_sp
            cur_words = [w]
            cur_start = w.start
            cur_end = w.end
            continue

        # Gap split: only if we already have enough content
        if cur_words:
            gap = w.start - cur_words[-1].end
            if gap >= rules.minGapMs:
                # If current cue is very short, try to still keep going
                # (prevents 1-2 word cues)
                current_text = render_words_to_text(cur_words, cur_speaker)
                if len(current_text) >= 12:
                    flush()
                    cur_words = [w]
                    cur_start = w.start
                    cur_end = w.end
                    continue

        # Try to add the word
        if not cur_words:
            cur_words = [w]
            cur_start = w.start
            cur_end = w.end
            continue

        if can_add_word(cur_words, w, cur_speaker):
            cur_words.append(w)
            cur_end = max(cur_end, w.end)

            # Punctuation encouragement:
            # Only flush if we have enough density (prevents tiny flashes)
            if rules.preferPunctuationBreaks and PUNCT_END_RE.search((w.text or "").strip()):
                rendered = render_words_to_text(cur_words, cur_speaker)
                if len(rendered) >= 22 and (cur_end - cur_start) >= 1200:
                    flush()
            continue

        # If we cannot add, flush and start new cue
        flush()
        cur_words = [w]
        cur_start = w.start
        cur_end = w.end

    flush()
    return cues


# ---------------------------
# Events / Sound cue injection (MVP)
# ---------------------------

def detect_basic_sound_cues_from_text(transcript_text: str) -> List[Event]:
    """
    MVP: If transcript includes explicit bracket tokens (rare),
    we convert them into events.
    """
    events: List[Event] = []
    if not transcript_text:
        return events

    # This is placeholder only (AssemblyAI doesn't usually emit these)
    return events


# ---------------------------
# QC
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

        cps = compute_cps(text, start, end)
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


# VERY simplified SCC encoder (MVP, not spec-perfect EIA-608 packing)
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
# Formatter
# ---------------------------

def format_payload(req: FormatRequest) -> Dict[str, Any]:
    rules = req.rules

    if req.cues is not None:
        cues = req.cues
    else:
        if not req.words:
            raise HTTPException(status_code=422, detail="You must provide either 'words' or 'cues'.")

        words = [Word(**w.model_dump()) for w in req.words]

        # PRO cue builder
        cues = build_cues_from_words_pro(words, rules)

    # Inject events (if any)
    if req.events:
        for e in req.events:
            if e.type == "music":
                txt = e.text or "[♪ MUSIC ♪]"
                cues.append({"start": e.start, "end": e.end, "text": txt, "speaker": "A"})
            elif e.type == "foreign_language":
                lang = e.language or "Unknown"
                cues.append({"start": e.start, "end": e.end, "text": f"[Speaking {lang}]", "speaker": "A"})
            elif e.type == "sound":
                txt = e.text or "[SOUND]"
                cues.append({"start": e.start, "end": e.end, "text": txt, "speaker": "A"})

        cues = sorted(cues, key=lambda c: (int(c["start"]), int(c["end"])))

    qc = qc_cues(cues, rules)

    return {
        "rules": rules.model_dump(),
        "cues": cues,
        "srt": to_srt(cues),
        "vtt": to_vtt(cues),
        "scc": to_scc(cues, rules),
        "qc": qc,
    }


# ---------------------------
# Routes
# ---------------------------

@app.get("/")
def root():
    return {"ok": True}


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

        # REQUIRED for Universal-3 Pro
        "speech_models": ["universal-3-pro"],

        "webhook_url": webhook_url,
        "webhook_auth_header_name": "X-Webhook-Token",
        "webhook_auth_header_value": WEBHOOK_SECRET,
    }

    headers = {"Authorization": ASSEMBLYAI_API_KEY, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=60.0) as client:
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

    headers = {"Authorization": ASSEMBLYAI_API_KEY}
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(f"https://api.assemblyai.com/v2/transcript/{assembly_id}", headers=headers)
        if r.status_code >= 300:
            JOBS[job_id].status = "error"
            JOBS[job_id].error = f"AssemblyAI get transcript failed: {r.status_code} {r.text}"
            return JSONResponse({"ok": True})
        t = r.json()

    # Words
    words_raw = t.get("words") or []
    words: List[Word] = []
    for w in words_raw:
        words.append(Word(
            text=w.get("text", ""),
            start=int(w.get("start", 0)),
            end=int(w.get("end", 0)),
            speaker=_normalize_speaker(w.get("speaker") or "A"),
        ))

    # EVENTS (sound cues):
    # AssemblyAI does NOT give these cleanly. This is where you'd plug in:
    # - a sound classifier model
    # - or a second pass using an LLM
    events: List[Event] = []

    rules = JOB_RULES.get(job_id, Rules())

    result = format_payload(FormatRequest(words=words, events=events, rules=rules))

    JOBS[job_id].status = "done"
    JOBS[job_id].result = result
    JOBS[job_id].error = None

    return JSONResponse({"ok": True})
