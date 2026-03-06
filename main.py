from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any
import json
import uuid
from datetime import datetime

from services.formatter import process_caption_job

app = FastAPI(title="OTT Caption Rules Engine", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory job store
# Fine for now. Later, if needed, move to Redis / DB.
JOBS: Dict[str, Dict[str, Any]] = {}


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "ott-caption-rules-engine",
        "time": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/generate")
async def generate(
    backbone_srt: UploadFile = File(...),
    timestamps_json: UploadFile = File(...),
    protected_phrases: Optional[str] = Form(None),
    output_formats: Optional[str] = Form(None),
):
    """
    Direct processing endpoint.
    Useful for manual testing and non-job-based clients.
    """
    try:
        srt_text = (await backbone_srt.read()).decode("utf-8", errors="replace")
        ts_text = (await timestamps_json.read()).decode("utf-8", errors="replace")
        ts_data = json.loads(ts_text)

        protected = json.loads(protected_phrases) if protected_phrases else []
        formats = json.loads(output_formats) if output_formats else ["srt"]

        result = process_caption_job(
            backbone_srt_text=srt_text,
            timestamps=ts_data,
            protected_phrases=protected,
            output_formats=formats,
        )

        return {
            "status": "completed",
            "result": result,
        }

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/jobs")
async def create_job(
    backbone_srt: UploadFile = File(...),
    timestamps_json: UploadFile = File(...),
    protected_phrases: Optional[str] = Form(None),
    output_formats: Optional[str] = Form(None),
):
    """
    Base44-compatible job creation endpoint.
    Creates a synchronous 'job', stores result in memory, and returns a job ID.
    """
    job_id = str(uuid.uuid4())

    JOBS[job_id] = {
        "id": job_id,
        "status": "processing",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "result": None,
        "error": None,
    }

    try:
        srt_text = (await backbone_srt.read()).decode("utf-8", errors="replace")
        ts_text = (await timestamps_json.read()).decode("utf-8", errors="replace")
        ts_data = json.loads(ts_text)

        protected = json.loads(protected_phrases) if protected_phrases else []
        formats = json.loads(output_formats) if output_formats else ["srt"]

        result = process_caption_job(
            backbone_srt_text=srt_text,
            timestamps=ts_data,
            protected_phrases=protected,
            output_formats=formats,
        )

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["result"] = result

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)

    return {
        "id": job_id,
        "status": JOBS[job_id]["status"],
        "created_at": JOBS[job_id]["created_at"],
    }


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str):
    """
    Base44-compatible polling endpoint.
    """
    job = JOBS.get(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job
