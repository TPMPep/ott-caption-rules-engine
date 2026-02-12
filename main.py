# ===========================
# PRO BROADCAST CAPTION ENGINE
# SDH + AssemblyAI + Hard Enforcement
# ===========================

import os
import uuid
import re
import json
from typing import Any, Dict, List, Optional, Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------
# App
# ---------------------------

app = FastAPI(title="Pro Broadcast Caption Engine", version="2.0.0")

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
ALLOWED_ORIGINS_RAW = os.getenv("ALLOWED_ORIGINS", "").strip()

if not PUBLIC_BASE_URL:
    PUBLIC_BASE_URL = "http://localhost:8000"

def _parse_allowed_origins(raw: str) -> List[str]:
    if not raw:
        return []
    if raw.strip() == "*":
        return ["*"]
    return [x.strip() for x in raw.split(",") if x.strip()]

_allowed_origins = _parse_allowed_origins(ALLOWED_ORIGINS_RAW)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins if _allowed_origins else [],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------
# Models
# ---------------------------

class Rules(BaseModel):
    maxCharsPerLine: int = 32
    maxLines: int = 2
    maxCPS: float = 17.0
    minDurationMs: int = 1000
    maxDurationMs: int = 7000
    minGapMs: int = 80

class Word(BaseModel):
    text: str
    start: int
    end: int
    speaker: Optional[str] = "A"

class Event(BaseModel):
    type: Literal["music", "sfx"]
    start: int
    end: int
    text: str

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

JOBS: Dict[str, JobStatus] = {}
JOB_RULES: Dict[str, Rules] = {}

# ---------------------------
# Hard Enforcement Caption Builder
# ---------------------------

def wrap_text(text: str, max_chars: int) -> str:
    words = text.split()
    lines = []
    current = ""

    for w in words:
        if len(current) + len(w) + 1 <= max_chars:
            current = (current + " " + w).strip()
        else:
            lines.append(current)
            current = w

    if current:
        lines.append(current)

    if len(lines) > 2:
        lines = [lines[0], " ".join(lines[1:])]

    return "\n".join(lines[:2])

def build_cues(words: List[Word], rules: Rules) -> List[Dict]:
    cues = []
    if not words:
        return cues

    words = sorted(words, key=lambda w: w.start)
    buffer = []
    start = words[0].start
    last_end = words[0].end
    speaker = words[0].speaker

    for w in words:
        duration = w.end - start
        cps = len(" ".join([x.text for x in buffer] + [w.text])) / max((duration / 1000), 0.001)

        if duration > rules.maxDurationMs or cps > rules.maxCPS or w.speaker != speaker:
            if buffer:
                text = wrap_text(" ".join([x.text for x in buffer]), rules.maxCharsPerLine)
                cues.append({
                    "start": start,
                    "end": last_end,
                    "text": text,
                    "speaker": speaker
                })
            buffer = [w]
            start = w.start
            speaker = w.speaker
        else:
            buffer.append(w)

        last_end = w.end

    if buffer:
        text = wrap_text(" ".join([x.text for x in buffer]), rules.maxCharsPerLine)
        cues.append({
            "start": start,
            "end": last_end,
            "text": text,
            "speaker": speaker
        })

    return cues

# ---------------------------
# SDH Inference (GPT)
# ---------------------------

async def infer_sdh_events(transcript_text: str) -> List[Event]:
    if not OPENAI_API_KEY:
        return []

    prompt = f"""
Analyze this transcript and return ONLY a JSON array of sound events.
Detect music, applause, laughter, door slams, car engines, etc.

Return format:
[
  {{"type":"music","start":12000,"end":18000,"text":"[♪ MUSIC ♪]"}},
  {{"type":"sfx","start":25000,"end":26000,"text":"[DOOR SLAMS]"}}
]

Transcript:
{transcript_text}
"""

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        if r.status_code >= 300:
            return []

        content = r.json()["choices"][0]["message"]["content"]

        try:
            events_raw = json.loads(content)
            return [Event(**e) for e in events_raw]
        except:
            return []

# ---------------------------
# Routes
# ---------------------------

@app.post("/v1/jobs", response_model=JobStatus)
async def create_job(req: JobCreateRequest):

    job_id = str(uuid.uuid4())
    JOBS[job_id] = JobStatus(id=job_id, status="queued")
    JOB_RULES[job_id] = req.rules

    webhook_url = f"{PUBLIC_BASE_URL}/v1/webhooks/assemblyai"

    payload = {
        "audio_url": req.mediaUrl,
        "speech_models": ["universal-3-pro"],
        "speaker_labels": req.speaker_labels,
        "language_detection": req.language_detection,
        "webhook_url": webhook_url,
        "webhook_auth_header_name": "X-Webhook-Token",
        "webhook_auth_header_value": WEBHOOK_SECRET,
    }

    headers = {"Authorization": ASSEMBLYAI_API_KEY}

    async with httpx.AsyncClient() as client:
        r = await client.post("https://api.assemblyai.com/v2/transcript", json=payload, headers=headers)
        data = r.json()

    JOBS[job_id].status = "processing"
    JOBS[job_id].assembly_id = data.get("id")

    return JOBS[job_id]

@app.post("/v1/webhooks/assemblyai")
async def assemblyai_webhook(request: Request):

    token = request.headers.get("X-Webhook-Token", "")
    if token != WEBHOOK_SECRET:
        raise HTTPException(status_code=401)

    body = await request.json()
    assembly_id = body.get("id")
    status = body.get("status")

    job_id = next((jid for jid, js in JOBS.items() if js.assembly_id == assembly_id), None)
    if not job_id:
        return {"ok": True}

    if status != "completed":
        return {"ok": True}

    headers = {"Authorization": ASSEMBLYAI_API_KEY}

    async with httpx.AsyncClient() as client:
        r = await client.get(f"https://api.assemblyai.com/v2/transcript/{assembly_id}", headers=headers)
        t = r.json()

    words = [Word(text=w["text"], start=w["start"], end=w["end"], speaker=w.get("speaker","A")) for w in t.get("words", [])]

    rules = JOB_RULES[job_id]
    cues = build_cues(words, rules)

    transcript_text = t.get("text","")
    sdh_events = await infer_sdh_events(transcript_text)

    for e in sdh_events:
        cues.append({
            "start": e.start,
            "end": e.end,
            "text": e.text,
            "speaker": None
        })

    cues = sorted(cues, key=lambda x: x["start"])

    JOBS[job_id].status = "done"
    JOBS[job_id].result = {"cues": cues}

    return {"ok": True}
