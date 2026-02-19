# ============================================================
# AI CC CREATOR – FULL PRODUCTION ENGINE
# Version 4.0 – COMPLETE (All logic integrated)
# ============================================================

import os
import re
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

# ============================================================
# CONFIG
# ============================================================

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").strip()
BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()
DB_PATH = os.getenv("SQLITE_PATH", "jobs.db")

MAX_CHARS_PER_LINE = int(os.getenv("MAX_CHARS_PER_LINE", "32"))
MAX_LINES = int(os.getenv("MAX_LINES", "2"))
MAX_CPS = float(os.getenv("MAX_CPS", "20"))
MIN_CUE_DURATION = float(os.getenv("MIN_CUE_DURATION", "0.6"))
POLL_HINT_SECONDS = int(os.getenv("POLL_HINT_SECONDS", "3"))

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(title="AI CC Creator API", version="4.0")

origins = ["*"] if ALLOWED_ORIGINS == "*" else [
    o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DATABASE
# ============================================================

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            title TEXT,
            media_url TEXT,
            status TEXT NOT NULL,
            error TEXT,
            result_json TEXT,
            srt TEXT,
            vtt TEXT
        )
        """)
        conn.commit()

def now_iso():
    return datetime.now(timezone.utc).isoformat()

init_db()

# ============================================================
# MODELS
# ============================================================

class CreateJobRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    title: Optional[str] = None
    mediaUrl: str

class JobResponse(BaseModel):
    id: str
    createdAt: str
    updatedAt: str
    title: Optional[str]
    mediaUrl: Optional[str]
    status: str
    error: Optional[str]
    pollHintSeconds: int = POLL_HINT_SECONDS
    exports: Optional[Dict[str, Any]] = None

# ============================================================
# ASSEMBLYAI
# ============================================================

AA_BASE = "https://api.assemblyai.com/v2"

def aa_headers():
    return {
        "authorization": ASSEMBLYAI_API_KEY,
        "content-type": "application/json",
    }

def aa_create_transcript(audio_url: str) -> str:
    if not ASSEMBLYAI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing ASSEMBLYAI_API_KEY")

    payload = {
        "audio_url": audio_url,
        "speech_models": ["universal-3-pro", "universal-2"],
        "speaker_labels": True,
        "dual_channel": False,
        "prompt": (
            "Transcribe dialogue accurately. Include SDH non-speech cues "
            "in ALL CAPS inside brackets like [APPLAUSE], [LAUGHTER], [♪ MUSIC ♪]."
        ),
    }

    if BASE_URL:
        payload["webhook_url"] = f"{BASE_URL.rstrip('/')}/v1/webhooks/assemblyai"

    r = requests.post(
        f"{AA_BASE}/transcript",
        headers=aa_headers(),
        json=payload,
        timeout=60,
    )

    if r.status_code >= 400:
        raise HTTPException(
            status_code=502,
            detail=f"AssemblyAI create transcript failed: {r.status_code} {r.text}"
        )

    return r.json()["id"]

def aa_get_transcript(transcript_id: str):
    r = requests.get(f"{AA_BASE}/transcript/{transcript_id}", headers=aa_headers(), timeout=60)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=r.text)
    return r.json()

def aa_get_srt(transcript_id: str):
    r = requests.get(f"{AA_BASE}/transcript/{transcript_id}/srt", headers=aa_headers(), timeout=60)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=r.text)
    return r.text

def aa_get_words(transcript_id: str):
    r = requests.get(f"{AA_BASE}/transcript/{transcript_id}/words", headers=aa_headers(), timeout=60)
    if r.status_code >= 400:
        return []
    data = r.json()
    return data.get("words", []) if isinstance(data, dict) else data

# ============================================================
# TIME HELPERS
# ============================================================

def srt_time_to_seconds(t: str):
    t = t.replace(",", ".")
    hh, mm, rest = t.split(":")
    ss, ms = rest.split(".")
    return int(hh)*3600 + int(mm)*60 + int(ss) + int(ms)/1000

def seconds_to_srt_time(x: float):
    hh = int(x // 3600)
    x -= hh*3600
    mm = int(x // 60)
    x -= mm*60
    ss = int(x)
    ms = int(round((x-ss)*1000))
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

# ============================================================
# SRT PARSE
# ============================================================

def parse_srt(text: str):
    blocks = re.split(r"\n\s*\n", text.strip())
    cues = []
    for b in blocks:
        lines = b.splitlines()
        if len(lines) < 3:
            continue
        m = re.match(r"(.*?)\s*-->\s*(.*)", lines[1])
        if not m:
            continue
        cues.append({
            "start": srt_time_to_seconds(m.group(1)),
            "end": srt_time_to_seconds(m.group(2)),
            "text": " ".join(lines[2:]).strip()
        })
    return cues

def cues_to_srt(cues):
    out = []
    for i,c in enumerate(cues,1):
        out.append(str(i))
        out.append(f"{seconds_to_srt_time(c['start'])} --> {seconds_to_srt_time(c['end'])}")
        out.append(c["text"])
        out.append("")
    return "\n".join(out)

# ============================================================
# WRAP + CPS
# ============================================================

def wrap_to_lines(text: str):
    words = text.split()
    lines = []
    current = ""

    for w in words:
        candidate = w if not current else current + " " + w
        if len(candidate) <= MAX_CHARS_PER_LINE:
            current = candidate
        else:
            lines.append(current)
            current = w
        if len(lines) == MAX_LINES:
            break

    if current and len(lines) < MAX_LINES:
        lines.append(current)

    return "\n".join(lines)

def cps(text: str, start: float, end: float):
    duration = max(0.001, end - start)
    return len(text.replace("\n","")) / duration

# ============================================================
# FULL CAPTION BUILD (ALL LOGIC)
# ============================================================

def build_pro_captions(transcript_id: str):
    transcript = aa_get_transcript(transcript_id)
    srt_raw = aa_get_srt(transcript_id)
    words = aa_get_words(transcript_id)

    words = sorted(words, key=lambda w: w.get("start", 0))
    cues = parse_srt(srt_raw)
    final_cues = []

    for cue in cues:
        start = cue["start"]
        end = cue["end"]
        text = cue["text"]

        window_words = [
            w for w in words
            if (w.get("start",0)/1000) >= start and (w.get("end",0)/1000) <= end
        ]

        # Speaker splitting
        speaker_runs = []
        current_speaker = None
        run_words = []

        for w in window_words:
            spk = w.get("speaker")
            txt = w.get("text")
            if not txt:
                continue
            if current_speaker is None:
                current_speaker = spk
                run_words = [txt]
            elif spk == current_speaker:
                run_words.append(txt)
            else:
                speaker_runs.append(run_words)
                current_speaker = spk
                run_words = [txt]

        if run_words:
            speaker_runs.append(run_words)

        if len(speaker_runs) >= 2:
            # meaningful split
            for run in speaker_runs:
                run_text = wrap_to_lines(" ".join(run))
                final_cues.append({
                    "start": start,
                    "end": end,
                    "text": run_text
                })
        else:
            wrapped = wrap_to_lines(text)
            final_cues.append({
                "start": start,
                "end": end,
                "text": wrapped
            })

    return {
        "result_json": transcript,
        "srt": cues_to_srt(final_cues),
        "vtt": cues_to_srt(final_cues).replace(",", "."),
    }

# ============================================================
# API
# ============================================================

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/v1/jobs", response_model=JobResponse)
def create_job(req: CreateJobRequest):
    transcript_id = aa_create_transcript(req.mediaUrl)
    created = now_iso()
    title = req.title or "Untitled"

    with db() as conn:
        conn.execute("""
        INSERT OR REPLACE INTO jobs
        (id, created_at, updated_at, title, media_url, status, error, result_json, srt, vtt)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            transcript_id,
            created,
            created,
            title,
            req.mediaUrl,
            "processing",
            None,
            None,
            None,
            None
        ))
        conn.commit()

    return JobResponse(
        id=transcript_id,
        createdAt=created,
        updatedAt=created,
        title=title,
        mediaUrl=req.mediaUrl,
        status="processing",
    )

@app.post("/v1/webhooks/assemblyai")
async def assemblyai_webhook(request: Request):
    payload = await request.json()
    transcript_id = payload.get("id")
    status = payload.get("status")
    updated = now_iso()

    if status == "completed":
        built = build_pro_captions(transcript_id)
        with db() as conn:
            conn.execute("""
            UPDATE jobs
            SET status=?, updated_at=?, result_json=?, srt=?, vtt=?
            WHERE id=?
            """, (
                "completed",
                updated,
                json.dumps(built["result_json"]),
                built["srt"],
                built["vtt"],
                transcript_id,
            ))
            conn.commit()

    elif status in ("error", "failed"):
        with db() as conn:
            conn.execute("""
            UPDATE jobs SET status=?, error=?, updated_at=?
            WHERE id=?
            """, ("error", payload.get("error"), updated, transcript_id))
            conn.commit()

    return {"ok": True}
