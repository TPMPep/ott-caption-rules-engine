from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any
from datetime import datetime
import uuid
import asyncio
import traceback

from services.assembly import (
    submit_transcription_job,
    wait_for_transcription_result,
    build_caption_inputs_from_assembly_result,
)
from services.formatter import process_caption_job

app = FastAPI(title="OTT Caption Rules Engine", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store
# Good enough for now. Later you can move to Redis/Postgres.
JOBS: Dict[str, Dict[str, Any]] = {}


# -----------------------------
# Request models
# -----------------------------

class CaptionRules(BaseModel):
    max_chars_per_line: int = 32
    max_lines: int = 2
    max_cps: int = 17
    min_duration_ms: int = 1000
    max_duration_ms: int = 7000
    min_gap_ms: int = 80
    prefer_punctuation_breaks: bool = True
    scc_frame_rate: float = 29.97
    start_at_hour_00: bool = True


class CreateJobRequest(BaseModel):
    mediaUrl: HttpUrl
    speakerLabels: bool = True
    languageDetection: bool = True
    allowHttp: bool = True
    captionRules: Optional[CaptionRules] = None


# -----------------------------
# Helpers
# -----------------------------

def utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


async def run_caption_job(job_id: str, payload: CreateJobRequest):
    """
    Full async pipeline:
    1. Send media URL to AssemblyAI
    2. Poll until transcription complete
    3. Build backbone SRT + timestamps JSON
    4. Run caption cleanup engine
    5. Store result
    """
    try:
        JOBS[job_id]["status"] = "transcribing"
        JOBS[job_id]["updated_at"] = utc_now()

        # 1) Submit to AssemblyAI
        transcript_id = await submit_transcription_job(
            media_url=str(payload.mediaUrl),
            speaker_labels=payload.speakerLabels,
            language_detection=payload.languageDetection,
        )

        JOBS[job_id]["assemblyai_transcript_id"] = transcript_id
        JOBS[job_id]["status"] = "processing"
        JOBS[job_id]["updated_at"] = utc_now()

        # 2) Poll AssemblyAI
        assembly_result = await wait_for_transcription_result(transcript_id)

        # 3) Convert AssemblyAI result into the two internal inputs
        #    formatter expects:
        #    - backbone_srt_text
        #    - timestamps (word-level tokens)
        backbone_srt_text, timestamps_json = build_caption_inputs_from_assembly_result(
            assembly_result
        )

        # 4) Caption cleanup engine
        protected_phrases = []  # add defaults if desired
        output_formats = ["srt", "vtt", "scc"]

        caption_result = process_caption_job(
            backbone_srt_text=backbone_srt_text,
            timestamps=timestamps_json,
            protected_phrases=protected_phrases,
            output_formats=output_formats,
        )

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["updated_at"] = utc_now()
        JOBS[job_id]["result"] = caption_result
        JOBS[job_id]["error"] = None

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["updated_at"] = utc_now()
        JOBS[job_id]["result"] = None
        JOBS[job_id]["error"] = {
            "message": str(e),
            "trace": traceback.format_exc(),
        }


# -----------------------------
# Routes
# -----------------------------

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "ott-caption-rules-engine",
        "time": utc_now(),
    }


@app.post("/v1/jobs")
async def create_job(payload: CreateJobRequest):
    """
    Base44 job creation endpoint.
    Accepts mediaUrl and caption config.
    Returns a job id immediately, then processes in background.
    """
    job_id = str(uuid.uuid4())

    JOBS[job_id] = {
        "id": job_id,
        "status": "queued",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "input": payload.model_dump(),
        "assemblyai_transcript_id": None,
        "result": None,
        "error": None,
    }

    asyncio.create_task(run_caption_job(job_id, payload))

    return {
        "id": job_id,
        "status": JOBS[job_id]["status"],
        "created_at": JOBS[job_id]["created_at"],
    }


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
