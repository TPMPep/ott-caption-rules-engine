import os
import uuid
import re
from typing import Any, Dict, List, Optional, Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="AI CC Creator – Broadcast Engine", version="2.0.0")

# =========================
# ENV
# =========================

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if ALLOWED_ORIGINS == "*" else [x.strip() for x in ALLOWED_ORIGINS.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# MODELS
# =========================

class Rules(BaseModel):
    maxCharsPerLine: int = 32
    maxLines: int = 2
    maxCPS: float = 17.0
    minDurationMs: int = 1000
    maxDurationMs: int = 7000
    minGapMs: int = 80
    sccFrameRate: float = 29.97


class JobCreateRequest(BaseModel):
    mediaUrl: str
    rules: Rules = Field(default_factory=Rules)
    speaker_labels: bool = True
    language_detection: bool = True


class JobStatus(BaseModel):
    id: str
    status: Literal["queued", "processing", "done", "error"]
    assembly_id: Optional[str] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


JOBS: Dict[str, JobStatus] = {}
JOB_RULES: Dict[str, Rules] = {}

# =========================
# UTILITIES
# =========================

def ms_to_srt(ms: int) -> str:
    h = ms // 3600000
    ms -= h * 3600000
    m = ms // 60000
    ms -= m * 60000
    s = ms // 1000
    ms -= s * 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def wrap_lines(text: str, max_chars: int) -> List[str]:
    words = text.split()
    lines = []
    current = ""

    for w in words:
        if not current:
            current = w
        elif len(current) + 1 + len(w) <= max_chars:
            current += " " + w
        else:
            lines.append(current)
            current = w

    if current:
        lines.append(current)

    return lines[:2]

def build_srt(cues):
    out = []
    for i, c in enumerate(cues, 1):
        out.append(str(i))
        out.append(f"{ms_to_srt(c['start'])} --> {ms_to_srt(c['end'])}")
        out.append(c["text"])
        out.append("")
    return "\n".join(out)

# =========================
# SOUND CUE DETECTION
# =========================

SOUND_KEYWORDS = {
    "music": "[♪ MUSIC ♪]",
    "applause": "[APPLAUSE]",
    "laugh": "[LAUGHTER]",
    "door": "[DOOR SLAMS]"
}

def detect_sound(word_text: str):
    t = word_text.lower()
    for k, v in SOUND_KEYWORDS.items():
        if k in t:
            return v
    return None

# =========================
# BROADCAST ENGINE
# =========================

def format_from_utterances(utterances, rules: Rules):

    cues = []

    for u in utterances:
        start = u["start"]
        end = u["end"]
        text = u["text"].strip()

        # speaker dash formatting
        speaker = u.get("speaker")
        if speaker is not None:
            text = "- " + text

        # wrap safely
        lines = wrap_lines(text, rules.maxCharsPerLine)
        wrapped = "\n".join(lines)

        cues.append({
            "start": start,
            "end": end,  # NEVER extend beyond assembly end
            "text": wrapped
        })

    return cues

# =========================
# ROUTES
# =========================

@app.get("/")
def root():
    return {"ok": True}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/v1/jobs", response_model=JobStatus)
async def create_job(req: JobCreateRequest):

    job_id = str(uuid.uuid4())

    JOBS[job_id] = JobStatus(id=job_id, status="processing")
    JOB_RULES[job_id] = req.rules

    payload = {
        "audio_url": req.mediaUrl,
        "speaker_labels": True,
        "language_detection": True,
        "speech_models": ["universal-3-pro"],
        "webhook_url": f"{PUBLIC_BASE_URL}/v1/webhooks/assemblyai",
        "webhook_auth_header_name": "X-Webhook-Token",
        "webhook_auth_header_value": WEBHOOK_SECRET,
    }

    headers = {"Authorization": ASSEMBLYAI_API_KEY}

    async with httpx.AsyncClient() as client:
        r = await client.post("https://api.assemblyai.com/v2/transcript", json=payload, headers=headers)
        data = r.json()
        assembly_id = data["id"]

    JOBS[job_id].assembly_id = assembly_id
    return JOBS[job_id]

@app.get("/v1/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404)
    return JOBS[job_id]

@app.post("/v1/webhooks/assemblyai")
async def webhook(request: Request):

    body = await request.json()
    assembly_id = body.get("transcript_id")
    status = body.get("status")

    job_id = None
    for jid, job in JOBS.items():
        if job.assembly_id == assembly_id:
            job_id = jid
            break

    if not job_id:
        return {"ok": True}

    if status != "completed":
        return {"ok": True}

    headers = {"Authorization": ASSEMBLYAI_API_KEY}

    async with httpx.AsyncClient() as client:
        r = await client.get(f"https://api.assemblyai.com/v2/transcript/{assembly_id}", headers=headers)
        transcript = r.json()

    utterances = transcript.get("utterances") or []

    rules = JOB_RULES.get(job_id, Rules())

    cues = format_from_utterances(utterances, rules)

    JOBS[job_id].status = "done"
    JOBS[job_id].result = {
        "cues": cues,
        "srt": build_srt(cues)
    }

    return {"ok": True}
