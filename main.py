"""
Main FastAPI application — OTT Caption Rules Engine.

This is the BRAIN of the CC Creation pipeline. Base44 only stores results.

Endpoints:
  GET  /health           → liveness probe + version
  POST /v1/jobs          → create a transcription/formatting job
  GET  /v1/jobs/{job_id} → poll a job for status/result

Job lifecycle (background thread):
  queued
    → submitting_to_provider          (transcription_provider: 'elevenlabs' | 'assemblyai')
    → waiting_for_transcription       (assemblyai async)
    → fetching_transcript             (reformat_only path)
    → formatting
    → completed | failed

A completed job carries:
  result.cues[]            — list of formatted CaptionCue dicts
  result.srt               — SRT string
  result.vtt               — VTT string
  result.qc                — QC report dict
  result.assemblyai.utterances[] — diarized utterances (AAI-shape, regardless
                                    of provider — Scribe is normalized to
                                    the same shape so the Base44 ingester is
                                    provider-agnostic).
  result.assemblyai.audio_events[] — structured non-dialogue events
                                      (music, applause, laughter, ...) —
                                      promoted into CaptionCue rows server-
                                      side as cue_type='music'/'sound_effect'.
  result.transcription_provider — 'elevenlabs' | 'assemblyai'
  result.transcription_model    — 'scribe_v2' | 'universal-3-pro' | 'universal-2'
  result._used_rules       — every env-driven rule value applied to this run.

Auth:
  If env var ENGINE_SHARED_SECRET is set, every POST/GET must carry
  X-Engine-Secret header matching it. If unset (today's default), the
  service is open — relies on the obscure Railway URL.

Provider selection (auditor-grade default):
  Scribe v2 is the DEFAULT transcription provider for CC projects because it
  provides native diarization + native audio-event tagging (FCC 47 CFR §79.1
  compliance — non-dialogue coverage is provider-emitted, not prompt-coaxed).
  AssemblyAI is retained as an opt-in fallback for legacy projects and as a
  diagnostic comparison path. The Base44 producer sets `transcription_provider`
  on every POST.
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
    build_caption_inputs,
    extract_audio_events_from_assembly_result,
)
from services.scribe import transcribe as scribe_transcribe
from services.formatter import process_caption_job, apply_env_overrides, restore_env_overrides

import json
import urllib.request

# Bump this on every meaningful edit. /health reports it so Base44 can
# verify a deploy landed without grepping Railway logs.
VERSION = "5.7.0-universal-speaker-labels-sound-density"

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

    # ── Reformat-from-baseline mode (engine 5.0.0) ──────────────────────
    # The auditor-grade re-format path. When `baselineUrl` is set, the engine
    # SKIPS transcription entirely: it fetches the immutable baseline JSON
    # (the same {utterances, words, audio_events} shape produced by Scribe/AAI
    # and persisted by Base44 at cc-baselines/{project_id}/aai-baseline.json),
    # and runs the FULL formatting pipeline against it — including the
    # editorial-AI grammar pass. This is what powers a spec swap (e.g. 32×2 →
    # 42×3): the new spec geometry flows in via captionOptions/env, so the
    # grammar-AI re-decides line breaks for the new width (killing orphan
    # lines), NOT a crude mechanical re-wrap. Provider-agnostic — works for
    # any baseline regardless of which provider originally transcribed it.
    #
    # baselineUrl is a short-TTL signed S3 GET URL (Base44 signs it; the
    # engine fetches it). We use a signed URL instead of an inline body so a
    # feature-length show's multi-MB baseline never bloats the request body —
    # mirrors how mediaUrl is handed to the transcription path.
    baselineUrl: Optional[HttpUrl] = None
    reformat_from_baseline: bool = False

    # Transcription provider selection (auditor-grade default: scribe_v2).
    # 'elevenlabs' → ElevenLabs Scribe v2 (native diarization + audio events)
    # 'assemblyai' → AssemblyAI universal-3-pro / universal-2 fallback chain
    # When omitted, defaults to 'elevenlabs' so a misconfigured producer
    # never silently degrades to a less auditable model.
    transcription_provider: Optional[str] = "elevenlabs"
    # ISO 639-3 (Scribe) or auto-detect (AAI). Producer derives this from the
    # project's source_language. None / "auto" → provider auto-detects.
    source_language_code: Optional[str] = None

    # Base44-side audit anchors. Echoed verbatim into the job record and
    # the result so the Base44 ingester can correlate the engine job with
    # the originating Project / CCFormatRun without out-of-band tracking.
    project_id: Optional[str] = None
    cc_format_run_id: Optional[str] = None
    request_id: Optional[str] = None

    # ── Fast-path completion callback (engine 5.2.0+) ───────────────────
    # When set, the engine POSTs { job_id, engine_job_id, status } to
    # callbackUrl the instant a job reaches a terminal state (completed /
    # failed), carrying X-Callback-Secret: callbackSecret. This collapses
    # Base44's up-to-5-min poll gap to ~1-2s. The callback is BEST-EFFORT —
    # Base44's ccPollRailwayJobs scheduled sweep remains the durable
    # guarantee, so a failed/dropped callback never strands a run. We never
    # retry the callback and never block job completion on it.
    # `base44_job_run_id` is the Base44 JobRun.id (NOT our internal job_id),
    # echoed back so the callback can locate the run with zero external lookup.
    callbackUrl: Optional[HttpUrl] = None
    callbackSecret: Optional[str] = None
    base44_job_run_id: Optional[str] = None


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


def _normalize_provider(raw: Optional[str]) -> str:
    """Normalize the transcription_provider field. Default = 'elevenlabs'."""
    if not raw:
        return "elevenlabs"
    p = str(raw).strip().lower()
    if p in ("elevenlabs", "scribe", "scribe_v2", "el"):
        return "elevenlabs"
    if p in ("assemblyai", "aai"):
        return "assemblyai"
    return "elevenlabs"


def _provider_model_id(provider: str) -> str:
    """For auditor evidence — pin the exact model used by this run."""
    if provider == "elevenlabs":
        return os.getenv("ELEVENLABS_SCRIBE_MODEL", "scribe_v2")
    return "universal-3-pro"  # AAI primary; engine sets a fallback chain


# Max baseline JSON we will fetch — a guard against a malformed / unbounded
# signed URL pointing at something enormous. A feature-length show's baseline
# is typically a few MB; 128MB is a generous safety ceiling.
BASELINE_FETCH_MAX_BYTES = 128 * 1024 * 1024
BASELINE_FETCH_TIMEOUT_SECONDS = int(
    os.getenv("BASELINE_FETCH_TIMEOUT_SECONDS", "120") or 120
)


def fetch_baseline_json(baseline_url: str) -> Dict[str, Any]:
    """
    Fetch + parse the immutable baseline JSON from a signed S3 GET URL.

    The baseline is the {utterances, words, audio_events, ...} object Base44
    persisted at transcription time. This is the EXACT shape the formatter's
    `build_caption_inputs()` already consumes — so reformat-from-baseline runs
    the identical pipeline transcription does, just with the provider call
    skipped. No new formatting logic; the engine's editorial brain is reused
    verbatim. SOC 2 CC8.1 — every reformat traces to the same immutable
    evidence the original transcription produced.
    """
    req = urllib.request.Request(baseline_url, method="GET")
    with urllib.request.urlopen(req, timeout=BASELINE_FETCH_TIMEOUT_SECONDS) as resp:
        raw = resp.read(BASELINE_FETCH_MAX_BYTES + 1)
    if len(raw) > BASELINE_FETCH_MAX_BYTES:
        raise ValueError(
            f"Baseline JSON exceeds {BASELINE_FETCH_MAX_BYTES} byte safety ceiling"
        )
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Baseline JSON is not an object: {type(data).__name__}")
    utterances = data.get("utterances")
    if not isinstance(utterances, list) or len(utterances) == 0:
        raise ValueError("Baseline JSON has no utterances — cannot reformat")
    return data


def baseline_to_assembly_result(baseline: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map a stored baseline dict into the canonical `assembly_result` shape the
    formatter + audio-event extractor expect. The baseline was WRITTEN from
    that same shape by the Base44 ingester, so this is a near-identity mapping
    — we just normalize the keys the formatter reads and carry the native
    audio_events through verbatim (so non-dialogue cues reproduce exactly).
    """
    return {
        "id": baseline.get("transcript_id"),
        "language_code": baseline.get("language_code"),
        "audio_duration": baseline.get("audio_duration_sec") or baseline.get("audio_duration"),
        "utterances": baseline.get("utterances") or [],
        "words": baseline.get("words") or [],
        "audio_events": baseline.get("audio_events") or [],
        # Mark provenance so build_caption_inputs never cross-calls a provider
        # for the backbone SRT (it builds locally from utterances).
        "_provider": baseline.get("transcription_provider") or "",
    }


def update_job(job_id: str, **fields: Any) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        job.update(fields)
        job["updated_at"] = utc_now()


# ── Fast-path completion callback ────────────────────────────────────
# POST { job_id, engine_job_id, status } to the Base44 callback URL the
# instant a job terminates. BEST-EFFORT: a short timeout, no retries, every
# failure swallowed + logged. Base44's ccPollRailwayJobs sweep is the durable
# guarantee — this only ACCELERATES the happy path. Never block job completion.
CALLBACK_TIMEOUT_SECONDS = int(os.getenv("CALLBACK_TIMEOUT_SECONDS", "10") or 10)


def fire_completion_callback(job_id: str, status: str) -> None:
    """Notify Base44 that an engine job reached a terminal state. The callback
    carries the Base44 JobRun.id (base44_job_run_id) so the receiver locates
    the run with no lookup; it then re-fetches the full result via GET
    /v1/jobs/:id. We send only the tiny signal, never the multi-MB payload."""
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            return
        payload_in = job.get("input") or {}
        callback_url = payload_in.get("callbackUrl")
        callback_secret = payload_in.get("callbackSecret")
        base44_job_run_id = payload_in.get("base44_job_run_id")

    if not callback_url or not callback_secret or not base44_job_run_id:
        return  # Callback not configured for this job — poller covers it.

    try:
        body = json.dumps({
            "job_id": base44_job_run_id,   # Base44 JobRun.id (what the receiver keys on)
            "engine_job_id": job_id,       # our internal Railway job id
            "status": status,              # 'completed' | 'failed'
        }).encode("utf-8")
        req = urllib.request.Request(
            str(callback_url), data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Callback-Secret": callback_secret,
            },
        )
        with urllib.request.urlopen(req, timeout=CALLBACK_TIMEOUT_SECONDS) as resp:
            print(f"[{job_id}] Completion callback → HTTP {resp.status} (status={status})")
    except Exception as e:
        # Swallow — the poller is the durable path. Log for observability only.
        print(f"[{job_id}] Completion callback failed (non-fatal, poller will cover): {e}")


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

        provider = _normalize_provider(payload.get("transcription_provider"))
        model_id = _provider_model_id(provider)
        transcript_id = JOBS.get(job_id, {}).get("provider_transcript_id") or payload.get("transcript_id")

        # ── Transcription branch ─────────────────────────────────────────
        # `assembly_result` is intentionally the canonical name for the
        # normalized result regardless of provider — the formatter and the
        # downstream Base44 ingester both consume this shape. Scribe v2 is
        # normalized by services.scribe.normalize_scribe_result to match.
        assembly_result: Dict[str, Any]
        if payload.get("reformat_from_baseline"):
            # ── Reformat-from-baseline (engine 5.0.0) ────────────────────
            # Skip transcription entirely. Fetch the immutable baseline JSON
            # from the signed S3 URL and run the FULL formatting pipeline
            # (formatter → editorial-AI → readability → QC) against it. The
            # new spec geometry arrived via captionOptions/env (already
            # applied above), so the editorial-AI rebalances line breaks for
            # the new width — the orphan-line fix on a spec swap.
            baseline_url = payload.get("baselineUrl")
            if not baseline_url:
                raise ValueError("baselineUrl is required when reformat_from_baseline=true")
            update_job(job_id, status="processing", stage="fetching_transcript",
                       transcription_provider=provider,
                       transcription_model=model_id)
            print(f"[{job_id}] Reformat-from-baseline: fetching baseline JSON")
            baseline = fetch_baseline_json(str(baseline_url))
            assembly_result = baseline_to_assembly_result(baseline)
            print(f"[{job_id}] Baseline loaded: {len(assembly_result.get('utterances') or [])} utterances, "
                  f"{len(assembly_result.get('audio_events') or [])} audio events")
        elif payload.get("reformat_only"):
            if not transcript_id:
                raise ValueError("transcript_id is required when reformat_only=true")
            if provider != "assemblyai":
                # Scribe v2 is synchronous — there is no transcript_id to
                # rehydrate against. Reformat-only is an AssemblyAI-only path
                # (the workflow's original use case was pure re-format runs).
                raise ValueError("reformat_only=true requires transcription_provider='assemblyai'")
            update_job(job_id, status="processing", stage="fetching_transcript",
                       provider_transcript_id=transcript_id,
                       transcription_provider=provider,
                       transcription_model=model_id)
            print(f"[{job_id}] Reformat-only: fetching AAI transcript {transcript_id}")
            assembly_result = fetch_transcript_result(transcript_id, require_completed=True)
        elif provider == "elevenlabs":
            update_job(job_id, status="processing", stage="submitting_to_provider",
                       transcription_provider=provider,
                       transcription_model=model_id)
            media_url = str(payload["mediaUrl"])
            lang = payload.get("source_language_code") or None
            print(f"[{job_id}] Submitting to ElevenLabs Scribe (lang={lang or 'auto'})")
            update_job(job_id, status="processing", stage="waiting_for_transcription")
            assembly_result = scribe_transcribe(media_url, language_code=lang)
        else:
            # AssemblyAI path (legacy fallback)
            update_job(job_id, status="processing", stage="submitting_to_provider",
                       transcription_provider=provider,
                       transcription_model=model_id)
            if not transcript_id:
                # AAI is submitted up front by the request handler — if we
                # got here without one, that's a programmer error.
                raise ValueError("AssemblyAI transcript_id missing before worker start")
            update_job(job_id, status="processing", stage="waiting_for_transcription",
                       provider_transcript_id=transcript_id)
            print(f"[{job_id}] Waiting for AssemblyAI result...")
            assembly_result = wait_for_transcription_result(transcript_id)

        update_job(job_id, status="processing", stage="formatting")

        backbone_srt_text, timestamps_json = build_caption_inputs(assembly_result)

        # ── Audio events — computed BEFORE formatting so the formatter can
        # turn them into real sound cues (rendered per the spec's MUSIC_CUE_
        # FORMAT / SOUND_EFFECT_FORMAT). Scribe v2 / baseline carry them
        # natively; AAI is best-effort extracted from bracket tags. Previously
        # this ran AFTER process_caption_job and was only attached to the
        # result side-channel — so the reformat ingester (which reads
        # result.cues[]) never saw them. That was the "no sound cues" bug.
        if payload.get("reformat_from_baseline") or provider == "elevenlabs":
            audio_events = list(assembly_result.get("audio_events") or [])
        else:
            audio_events = extract_audio_events_from_assembly_result(assembly_result)

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
            audio_events=audio_events,
        )

        # Attach diarization + audio events so Base44 can derive CCSpeaker
        # rows and non-dialogue cues without re-fetching from the provider.
        # SOC 2 CC8.1 — the engine result is self-contained chain-of-custody
        # evidence. The key stays 'assemblyai' to keep the Base44 ingester
        # contract stable; the provider fingerprint lives in
        # caption_result.transcription_provider / transcription_model.
        # CRITICAL: include per-utterance `words` (word-level timings) AND a
        # flat top-level `words` array. The Base44 ingester writes these into
        # the immutable S3 baseline, and every future Apply Spec re-run rebuilds
        # its input cues from them via `_splitUtteranceByPauses`. Dropping them
        # here (the prior bug) produced a baseline with words:[] — which made
        # Apply Spec fail with "AAI baseline contained no utterances" on EVERY
        # project. The normalizer (scribe.py / assembly.py) already populates
        # utterance["words"]; we now carry it through verbatim. SOC 2 CC8.1 —
        # the engine result is self-contained chain-of-custody evidence.
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
                    "words": u.get("words") or [],
                }
                for u in (assembly_result.get("utterances") or [])
            ],
            "audio_events": audio_events,
        }
        # Flat word list — the secondary baseline source. Mirrors the
        # normalizer's `words` output (already provider-agnostic). The ingester
        # prefers this when present; the per-utterance words above are the
        # fallback. Carrying both makes the baseline robust to either reader.
        caption_result["words"] = list(assembly_result.get("words") or [])
        # Echo Base44-side audit anchors for the ingester to correlate.
        caption_result["base44"] = {
            "project_id": payload.get("project_id"),
            "cc_format_run_id": payload.get("cc_format_run_id"),
            "request_id": payload.get("request_id"),
        }
        caption_result["engine_version"] = VERSION
        # Auditor fingerprint — pinned per-run, not derived.
        caption_result["transcription_provider"] = provider
        caption_result["transcription_model"] = model_id

        update_job(job_id, status="completed", stage="completed",
                   result=caption_result, error=None,
                   transcription_provider=provider,
                   transcription_model=model_id,
                   provider_transcript_id=assembly_result.get("id") or transcript_id)

    except Exception as e:
        print(f"[{job_id}] Caption job FAILED: {e}")
        print(traceback.format_exc())
        update_job(job_id, status="failed", stage="failed", result=None,
                   error={"message": str(e), "trace": traceback.format_exc()[:4000]})
    finally:
        if env_snapshot is not None:
            restore_env_overrides(env_snapshot)
        # Fast-path completion callback — fired AFTER the job's terminal state
        # is committed to the store (so the GET the receiver does sees the
        # final result) and AFTER env restore (so a callback hang can never
        # leave env vars dirty). Best-effort, swallows all errors; the poller
        # remains the durable guarantee.
        with JOBS_LOCK:
            terminal_status = (JOBS.get(job_id) or {}).get("status")
        if terminal_status in ("completed", "failed"):
            fire_completion_callback(job_id, terminal_status)


# ─── Routes ─────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "ott-caption-rules-engine",
        "version": VERSION,
        "time": utc_now(),
        "default_transcription_provider": "elevenlabs",
        "scribe_model_id": os.getenv("ELEVENLABS_SCRIBE_MODEL", "scribe_v2"),
    }


@app.post("/v1/jobs")
def create_job(payload: CreateJobRequest, x_engine_secret: Optional[str] = Header(default=None)):
    _check_secret(x_engine_secret)

    job_id = str(uuid.uuid4())
    payload_data = payload.model_dump(mode="json")
    provider = _normalize_provider(payload_data.get("transcription_provider"))
    model_id = _provider_model_id(provider)

    if payload_data.get("reformat_from_baseline"):
        if not payload_data.get("baselineUrl"):
            raise HTTPException(status_code=400, detail="baselineUrl is required when reformat_from_baseline=true")
    elif payload_data.get("reformat_only"):
        if not payload_data.get("transcript_id"):
            raise HTTPException(status_code=400, detail="transcript_id is required when reformat_only=true")
        if provider != "assemblyai":
            raise HTTPException(status_code=400, detail="reformat_only=true requires transcription_provider='assemblyai'")
    else:
        if not payload_data.get("mediaUrl"):
            raise HTTPException(status_code=400, detail="mediaUrl is required for new transcription jobs")

    created_at = utc_now()
    provider_transcript_id = payload_data.get("transcript_id")

    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "queued",
            "stage": "queued",
            "created_at": created_at,
            "updated_at": created_at,
            "input": payload_data,
            "transcription_provider": provider,
            "transcription_model": model_id,
            # Back-compat alias used by older Base44 poller paths.
            "assemblyai_transcript_id": provider_transcript_id if provider == "assemblyai" else None,
            "provider_transcript_id": provider_transcript_id,
            "project_id": payload_data.get("project_id"),
            "cc_format_run_id": payload_data.get("cc_format_run_id"),
            "result": None,
            "error": None,
        }

    # ── AAI path: kick off the async transcription up front so the response
    # carries assemblyai_transcript_id (the poller chains off this).
    # ── Scribe path: synchronous — we just start the worker; no upfront ID.
    if payload_data.get("reformat_from_baseline"):
        # Synchronous-style background worker; no upfront provider ID. The
        # worker fetches the baseline + formats. Poller reads /v1/jobs/:id.
        update_job(job_id, status="processing", stage="fetching_transcript")
        start_job_worker(job_id, payload_data)
    elif payload_data.get("reformat_only"):
        update_job(job_id, status="processing", stage="fetching_transcript",
                   provider_transcript_id=provider_transcript_id,
                   assemblyai_transcript_id=provider_transcript_id)
        start_job_worker(job_id, payload_data)
    elif provider == "assemblyai":
        media_url = str(payload_data["mediaUrl"])
        speaker_labels = payload_data.get("speakerLabels", True)
        language_detection = payload_data.get("languageDetection", True)
        try:
            provider_transcript_id = submit_transcription_job(
                media_url=media_url,
                speaker_labels=speaker_labels,
                language_detection=language_detection,
            )
        except Exception as exc:
            update_job(job_id, status="failed", stage="failed", error={"message": str(exc)})
            return {
                "id": job_id, "status": "failed", "stage": "failed",
                "created_at": created_at,
                "assemblyai_transcript_id": None,
                "transcription_provider": provider,
                "transcription_model": model_id,
                "error": {"message": str(exc)},
            }
        update_job(job_id, status="processing", stage="waiting_for_transcription",
                   provider_transcript_id=provider_transcript_id,
                   assemblyai_transcript_id=provider_transcript_id)
        start_job_worker(job_id, payload_data)
    else:
        # ElevenLabs Scribe path — worker thread does the synchronous POST.
        update_job(job_id, status="processing", stage="submitting_to_provider")
        start_job_worker(job_id, payload_data)

    job = JOBS[job_id]
    return {
        "id": job_id,
        "status": job["status"],
        "stage": job.get("stage"),
        "created_at": job["created_at"],
        # AAI-back-compat field — null for Scribe runs (the AAI poller branch
        # in Base44 looks for non-null before it tries to call AAI directly).
        "assemblyai_transcript_id": job.get("assemblyai_transcript_id"),
        "provider_transcript_id": job.get("provider_transcript_id"),
        "transcription_provider": provider,
        "transcription_model": model_id,
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
