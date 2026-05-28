"""
Main FastAPI application — OTT Caption Rules Engine.

This is the BRAIN of the CC Creation pipeline. Base44 only stores results.

Endpoints:
  GET  /health           → liveness probe + version
  POST /v1/jobs          → create a transcription/formatting job
  GET  /v1/jobs/{job_id} → poll a job for status/result

Job lifecycle (background thread):
  queued
    → submitted_to_assemblyai
    → waiting_for_transcription
    → fetching_transcript          (reformat_only path)
    → formatting
    → completed | failed

A completed job carries:
  result.cues[]            — list of formatted CaptionCue dicts
  result.srt               — SRT string
  result.vtt               — VTT string
  result.qc                — QC report dict
  result.assemblyai.utterances[] — raw AAI utterances (so Base44 can derive
                                   CCSpeaker rows server-side from A/B/C
                                   diarization without re-fetching AAI)
  result._used_rules       — every env-driven rule value applied to this
                             run, so the auditor can reproduce the result
                             from the row alone (SOC 2 CC8.1).

Auth:
  If env var ENGINE_SHARED_SECRET is set, every POST/GET must carry
  X-Engine-Secret header matching it. If unset (today's default), the
  service is open — relies on the obscure Railway URL.
"""

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, Dict, Any, List
from datetime import datetime
import os
import threading
import uuid
import traceback

from services.assembly import (
    submit_transcription_job,
    wait_for_transcription_result,
    fetch_transcript_result,
    build_caption_inputs_from_assembly_result,
)
from services.formatter import process_caption_job, apply_env_overrides, restore_env_overrides

# Bump this on every meaningful edit. /health reports it so Base44 can
# verify a deploy landed without grepping Railway logs.
VERSION = "3.5.0-base44-pipeline"

app = FastAPI(title="OTT Caption Rules Engine", version=VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store. For 100+ concurrent users this is fine — jobs are
# transient and consumed by Base44's poller within minutes. If we ever need
# durability we can swap this for Redis (Upstash is already in the stack).
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()


# ─── Auth ───────────────────────────────────────────────────────────

def _check_secret(x_engine_secret: Optional[str]) -> None:
    expected = os.getenv("ENGINE_SHARED_SECRET", "").strip()
    if not expected:
        return  # open mode
    if x_engine_secret != expected:
        raise HTTPException(status_code=401, detail="invalid X-Engine-Secret")


# ─── Request Models ─────────────────────────────────────────────────

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
    protectedPhrases: Optional[List[str]] = None  # camelCase alias

    # Base44-side audit anchors. Echoed verbatim into the job record and
    # the result so the Base44 ingester can correlate the engine job with
    # the originating Project / CCFormatRun without out-of-band tracking.
    project_id: Optional[str] = None
    cc_format_run_id: Optional[str] = None
    request_id: Optional[str] = None


# ─── Helpers ────────────────────────────────────────────────────────

def utc_now() -> str:
    return datetime.utcnow().isoformat() + "Z"


def parse_output_formats(payload: Dict[str, Any]) -> Optional[List[str]]:
    if isinstance(payload.get("output_formats"), list) and payload.get("output_formats"):
        return [str(f).strip().lower() for f in payload["output_formats"] if str(f).strip()]
    env = payload.get("env") or payload.get("captionOptions") or {}
    if isinstance(env, dict):
        raw = env.get("OUTPUT_FORMATS")
        if raw:
            return [f.strip().lower() for f in str(raw).split(",") if f.strip()]
    return None


def get_protected_phrases(payload: Dict[str, Any]) -> List[str]:
    phrases = payload.get("protected_phrases") or payload.get("protectedPhrases") or []
    if isinstance(phrases, str):
        phrases = [p.strip() for p in phrases.split(",") if p.strip()]
    return phrases


def get_env_overrides(payload: Dict[str, Any]) -> Dict[str, Any]:
    env_dict: Dict[str, Any] = {}
    if isinstance(payload.get("env"), dict):
        env_dict.update(payload["env"])
    if isinstance(payload.get("captionOptions"), dict):
        env_dict.update(payload["captionOptions"])
    return env_dict


def update_job(job_id: str, **fields: Any) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(fields)
        job["updated_at"] = utc_now()


def start_job_worker(job_id: str, payload: Dict[str, Any]) -> None:
    worker = threading.Thread(
        target=run_caption_job,
        args=(job_id, payload),
        daemon=True,
        name=f"caption-job-{job_id[:8]}",
    )
    worker.start()


# ─── Background Job Runner ──────────────────────────────────────────

def run_caption_job(job_id: str, payload: Dict[str, Any]) -> None:
    env_snapshot = None
    try:
        print(f"[{job_id}] Starting caption job")

        env_overrides = get_env_overrides(payload)
        env_snapshot = apply_env_overrides(env_overrides)

        output_formats = parse_output_formats(payload)
        protected_phrases = get_protected_phrases(payload)

        transcript_id = JOBS.get(job_id, {}).get("assemblyai_transcript_id") or payload.get("transcript_id")

        if payload.get("reformat_only"):
            if not transcript_id:
                raise ValueError("transcript_id is required when reformat_only=true")
            update_job(job_id, status="processing", stage="fetching_transcript",
                       assemblyai_transcript_id=transcript_id)
            print(f"[{job_id}] Reformat-only: fetching AAI transcript {transcript_id}")
            assembly_result = fetch_transcript_result(transcript_id, require_completed=True)
        else:
            if not transcript_id:
                raise ValueError("assemblyai_transcript_id is required before transcription worker starts")
            update_job(job_id, status="processing", stage="waiting_for_transcription",
                       assemblyai_transcript_id=transcript_id)
            print(f"[{job_id}] Waiting for AssemblyAI result...")
            assembly_result = wait_for_transcription_result(transcript_id)

        update_job(job_id, status="processing", stage="formatting")

        backbone_srt_text, timestamps_json = build_caption_inputs_from_assembly_result(assembly_result)

        # Heartbeat closure — editorial_ai calls this every N cues so the
        # job's updated_at timestamp advances during the long AI polish
        # pass. Without this, the formatter could correctly grind through
        # 400 cues over 90 seconds and look "hung" to Base44's poller
        # (which reads updated_at as the freshness signal). SOC 2 CC8.1 —
        # engine progress must be observable in real time.
        def _formatter_heartbeat(idx: int, total: int) -> None:
            update_job(job_id, stage="formatting",
                       formatter_progress={"cues_processed": idx, "cues_total": total})

        caption_result = process_caption_job(
            backbone_srt_text=backbone_srt_text,
            timestamps=timestamps_json,
            protected_phrases=protected_phrases,
            output_formats=output_formats,
            heartbeat=_formatter_heartbeat,
        )

        # Attach AAI utterances so Base44 can derive CCSpeaker rows without
        # re-fetching from AssemblyAI. SOC 2 CC8.1 — the engine result is
        # self-contained chain-of-custody evidence.
        caption_result["assemblyai"] = {
            "transcript_id": assembly_result.get("id"),
            "language_code": assembly_result.get("language_code"),
            "audio_duration": assembly_result.get("audio_duration"),
            "utterances": [
                {
                    "speaker": u.get("speaker"),
                    "start": u.get("start"),
                    "end": u.get("end"),
                    "text": u.get("text"),
                    "confidence": u.get("confidence"),
                }
                for u in (assembly_result.get("utterances") or [])
            ],
        }
        # Echo Base44-side audit anchors for the ingester to correlate.
        caption_result["base44"] = {
            "project_id": payload.get("project_id"),
            "cc_format_run_id": payload.get("cc_format_run_id"),
            "request_id": payload.get("request_id"),
        }
        caption_result["engine_version"] = VERSION

        update_job(job_id, status="completed", stage="completed",
                   result=caption_result, error=None,
                   assemblyai_transcript_id=assembly_result.get("id") or transcript_id)

    except Exception as e:
        print(f"[{job_id}] Caption job FAILED: {e}")
        print(traceback.format_exc())
        update_job(job_id, status="failed", stage="failed", result=None,
                   error={"message": str(e), "trace": traceback.format_exc()[:4000]})
    finally:
        if env_snapshot is not None:
            restore_env_overrides(env_snapshot)


# ─── Routes ─────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "ott-caption-rules-engine",
        "version": VERSION,
        "time": utc_now(),
    }


@app.post("/v1/jobs")
def create_job(payload: CreateJobRequest, x_engine_secret: Optional[str] = Header(default=None)):
    _check_secret(x_engine_secret)

    job_id = str(uuid.uuid4())
    payload_data = payload.model_dump(mode="json")

    if payload_data.get("reformat_only"):
        if not payload_data.get("transcript_id"):
            raise HTTPException(status_code=400, detail="transcript_id is required when reformat_only=true")
    else:
        if not payload_data.get("mediaUrl"):
            raise HTTPException(status_code=400, detail="mediaUrl is required for new transcription jobs")

    created_at = utc_now()
    assemblyai_transcript_id = payload_data.get("transcript_id")

    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "stage": "queued",
            "created_at": created_at,
            "updated_at": created_at,
            "input": payload_data,
            "assemblyai_transcript_id": assemblyai_transcript_id,
            "project_id": payload_data.get("project_id"),
            "cc_format_run_id": payload_data.get("cc_format_run_id"),
            "result": None,
            "error": None,
        }

    if payload_data.get("reformat_only"):
        update_job(job_id, status="processing", stage="fetching_transcript",
                   assemblyai_transcript_id=assemblyai_transcript_id)
        start_job_worker(job_id, payload_data)
    else:
        media_url = str(payload_data["mediaUrl"])
        speaker_labels = payload_data.get("speakerLabels", True)
        language_detection = payload_data.get("languageDetection", True)
        try:
            assemblyai_transcript_id = submit_transcription_job(
                media_url=media_url,
                speaker_labels=speaker_labels,
                language_detection=language_detection,
            )
        except Exception as exc:
            update_job(job_id, status="failed", stage="failed", error={"message": str(exc)})
            return {
                "id": job_id, "status": "failed", "stage": "failed",
                "created_at": created_at, "assemblyai_transcript_id": None,
                "error": {"message": str(exc)},
            }
        update_job(job_id, status="processing", stage="waiting_for_transcription",
                   assemblyai_transcript_id=assemblyai_transcript_id)
        start_job_worker(job_id, payload_data)

    job = JOBS[job_id]
    return {
        "id": job_id,
        "status": job["status"],
        "stage": job.get("stage"),
        "created_at": job["created_at"],
        "assemblyai_transcript_id": job.get("assemblyai_transcript_id"),
        "engine_version": VERSION,
    }


@app.get("/v1/jobs/{job_id}")
def get_job(job_id: str, x_engine_secret: Optional[str] = Header(default=None)):
    _check_secret(x_engine_secret)
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job
