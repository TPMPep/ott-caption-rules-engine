import os
import uuid
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, AnyUrl
from typing import Optional, Dict, Any
from services.assembly import submit_transcription, fetch_transcript
from services.formatter import build_captions_from_assembly
from services.exporters import captions_to_srt, captions_to_vtt, vtt_to_scc_safe
from services.qc import run_qc

APP_VERSION = "1.0"

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

if not ASSEMBLYAI_API_KEY:
    print("WARNING: ASSEMBLYAI_API_KEY is not set")

app = FastAPI(title="Caption Engine", version=APP_VERSION)

# CORS
origins = ["*"] if ALLOWED_ORIGINS.strip() == "*" else [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory store (fine for prototype). For production: Redis/Postgres.
JOBS: Dict[str, Dict[str, Any]] = {}

class CreateJobPayload(BaseModel):
    mediaUrl: AnyUrl
    speaker_labels: bool = True
    language_detection: bool = True
    rules: Dict[str, Any] = {}

@app.get("/health")
def health():
    return {
        "ok": True,
        "version": APP_VERSION,
        "hasAssemblyKey": bool(ASSEMBLYAI_API_KEY),
        "assemblyKeyLength": len(ASSEMBLYAI_API_KEY or ""),
        "hasPublicBaseUrl": bool(PUBLIC_BASE_URL)
    }

@app.post("/v1/jobs")
def create_job(payload: CreateJobPayload):
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    JOBS[job_id] = {
        "id": job_id,
        "status": "queued",
        "payload": payload.model_dump(),
        "assembly_transcript_id": None,
        "exports": None,
        "error": None,
    }

    # Submit to AssemblyAI asynchronously (webhook-driven when possible)
    try:
        webhook_url = None
        if PUBLIC_BASE_URL:
            webhook_url = f"{PUBLIC_BASE_URL}/v1/webhooks/assemblyai?job_id={job_id}&secret={WEBHOOK_SECRET}"
        transcript_id = submit_transcription(
            api_key=ASSEMBLYAI_API_KEY,
            media_url=str(payload.mediaUrl),
            speaker_labels=payload.speaker_labels,
            language_detection=payload.language_detection,
            webhook_url=webhook_url,
        )
        JOBS[job_id]["assembly_transcript_id"] = transcript_id
        JOBS[job_id]["status"] = "processing"
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = f"AssemblyAI submit failed: {e}"
        return {"id": job_id, "status": "failed", "error": JOBS[job_id]["error"]}

    return {"id": job_id, "status": JOBS[job_id]["status"]}

@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # If webhook isn’t used, you can optionally poll Assembly here.
    # For now: if status processing and no exports, keep returning processing.
    resp = {"id": job["id"], "status": job["status"]}
    if job.get("error"):
        resp["error"] = job["error"]
    if job.get("exports"):
        resp["exports"] = job["exports"]
    return resp

@app.post("/v1/webhooks/assemblyai")
async def assembly_webhook(request: Request):
    job_id = request.query_params.get("job_id")
    secret = request.query_params.get("secret")

    if not job_id or job_id not in JOBS:
        raise HTTPException(status_code=400, detail="Missing/invalid job_id")
    if WEBHOOK_SECRET and secret != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid webhook secret")

    body = await request.json()
    # AssemblyAI sends events; safest is to fetch transcript by id we stored.
    transcript_id = JOBS[job_id].get("assembly_transcript_id")
    if not transcript_id:
        raise HTTPException(status_code=400, detail="No transcript id for job")

    try:
        transcript = fetch_transcript(api_key=ASSEMBLYAI_API_KEY, transcript_id=transcript_id)
        status = transcript.get("status")

        if status == "error":
            JOBS[job_id]["status"] = "failed"
            JOBS[job_id]["error"] = transcript.get("error", "AssemblyAI error")
            return {"ok": True}

        if status != "completed":
            # still processing; leave job as processing
            JOBS[job_id]["status"] = "processing"
            return {"ok": True}

        # Completed → build broadcast captions
        rules = JOBS[job_id]["payload"].get("rules", {})
        captions, meta = build_captions_from_assembly(transcript, rules)

        # Exports
        vtt = captions_to_vtt(captions)
        srt = captions_to_srt(captions)
        scc = vtt_to_scc_safe(vtt)  # returns None if pycaption not installed

        qc = run_qc(captions, rules)
        result = {
            "meta": {"engine": "assemblyai", "appVersion": APP_VERSION, **meta},
            "qc": qc,
        }

        JOBS[job_id]["exports"] = {
            "vtt": vtt,
            "srt": srt,
            "scc": scc,
            "result": result,
        }
        JOBS[job_id]["status"] = "completed"
        return {"ok": True}

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = f"Post-process failed: {e}"
        return {"ok": True}
