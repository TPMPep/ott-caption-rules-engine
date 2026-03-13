from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid
import traceback

from services.assembly import (
    submit_transcription_job,
    wait_for_transcription_result,
    fetch_transcript_result,
    build_caption_inputs_from_assembly_result,
)
from services.formatter import process_caption_job, apply_env_overrides, restore_env_overrides

app = FastAPI(title="OTT Caption Rules Engine", version="3.1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store
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
    mediaUrl: Optional[HttpUrl] = None
    transcript_id: Optional[str] = None
    reformat_only: bool = False
    speakerLabels: bool = True
    languageDetection: bool = True
    allowHttp: bool = True
    captionRules: Optional[CaptionRules] = None
    captionOptions: Optional[Dict[str, Any]] = None
    env: Optional[Dict[str, Any]] = None
    output_formats: Optional[List[str]] = None
    protected_phrases: Optional[List[str]] = None


# -----------------------------
# Helpers
# -----------------------------

def utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def parse_output_formats(payload: Dict[str, Any]) -> Optional[List[str]]:
    if isinstance(payload.get("output_formats"), list) and payload.get("output_formats"):
        return [str(f).strip().lower() for f in payload["output_formats"] if str(f).strip()]
    env = payload.get("env") or {}
    if isinstance(env, dict):
        raw = env.get("OUTPUT_FORMATS")
        if raw:
            return [f.strip().lower() for f in str(raw).split(",") if f.strip()]
    return None


def run_caption_job(job_id: str, payload: Dict[str, Any]):
    """
    Background job runner.

    Flow:
    1. Submit media URL to AssemblyAI
    2. Poll AssemblyAI until complete
    3. Build backbone SRT + timestamps JSON
    4. Run caption formatter / cleanup
    5. Store final result
    """
    env_snapshot = None
    try:
        print(f"[{job_id}] Starting caption job")
        print(f"[{job_id}] Input payload: {payload}")

        env_snapshot = apply_env_overrides(payload.get("env") or {})
        output_formats = parse_output_formats(payload)
        protected_phrases = payload.get("protected_phrases") or []

        if payload.get("reformat_only"):
            transcript_id = payload.get("transcript_id")
            if not transcript_id:
                raise ValueError("transcript_id is required when reformat_only=true")

            JOBS[job_id]["status"] = "processing"
            JOBS[job_id]["updated_at"] = utc_now()
            JOBS[job_id]["assemblyai_transcript_id"] = transcript_id

            print(f"[{job_id}] Reformat-only: fetching AssemblyAI transcript {transcript_id}")
            assembly_result = fetch_transcript_result(transcript_id, require_completed=True)
        else:
            JOBS[job_id]["status"] = "transcribing"
            JOBS[job_id]["updated_at"] = utc_now()

            media_url = str(payload["mediaUrl"])
            speaker_labels = payload.get("speakerLabels", True)
            language_detection = payload.get("languageDetection", True)

            print(f"[{job_id}] Submitting to AssemblyAI: {media_url}")
            transcript_id = submit_transcription_job(
                media_url=media_url,
                speaker_labels=speaker_labels,
                language_detection=language_detection,
            )

            print(f"[{job_id}] AssemblyAI transcript id: {transcript_id}")
            JOBS[job_id]["assemblyai_transcript_id"] = transcript_id
            JOBS[job_id]["status"] = "processing"
            JOBS[job_id]["updated_at"] = utc_now()

            print(f"[{job_id}] Waiting for AssemblyAI result...")
            assembly_result = wait_for_transcription_result(transcript_id)

        print(f"[{job_id}] AssemblyAI transcription completed")
        print(f"[{job_id}] Building formatter inputs from AssemblyAI result")
        backbone_srt_text, timestamps_json = build_caption_inputs_from_assembly_result(
            assembly_result
        )

        print(f"[{job_id}] Running caption formatter")
        caption_result = process_caption_job(
            backbone_srt_text=backbone_srt_text,
            timestamps=timestamps_json,
            protected_phrases=protected_phrases,
            output_formats=output_formats,
        )

        print(f"[{job_id}] Caption job completed successfully")
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["updated_at"] = utc_now()
        JOBS[job_id]["result"] = caption_result
        JOBS[job_id]["error"] = None

    except Exception as e:
        print(f"[{job_id}] Caption job FAILED: {e}")
        print(traceback.format_exc())

        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["updated_at"] = utc_now()
        JOBS[job_id]["result"] = None
        JOBS[job_id]["error"] = {
            "message": str(e),
            "trace": traceback.format_exc(),
        }
    finally:
        if env_snapshot is not None:
            restore_env_overrides(env_snapshot)


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
def create_job(payload: CreateJobRequest, background_tasks: BackgroundTasks):
    """
    Base44-compatible job creation endpoint.
    Accepts JSON payload with mediaUrl and caption settings.
    Processes in the background.
    """
    job_id = str(uuid.uuid4())

    print(f"[{job_id}] Job created")

    payload_data = payload.model_dump(mode="json")

    if payload_data.get("reformat_only"):
        if not payload_data.get("transcript_id"):
            raise HTTPException(status_code=400, detail="transcript_id is required when reformat_only=true")
    else:
        if not payload_data.get("mediaUrl"):
            raise HTTPException(status_code=400, detail="mediaUrl is required for new transcription jobs")

    JOBS[job_id] = {
        "id": job_id,
        "status": "queued",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "input": payload_data,
        "assemblyai_transcript_id": None,
        "result": None,
        "error": None,
    }

    background_tasks.add_task(run_caption_job, job_id, payload_data)
    print(f"[{job_id}] Background task dispatched")

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
