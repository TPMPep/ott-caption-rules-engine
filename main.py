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

from fastapi import FastAPI, HTTPException, Header, Request
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
from services.formatter import (
    process_caption_job, apply_env_overrides, restore_env_overrides,
    FORMATTER_VERSION,
)
from services.sequence_optimizer import SEGMENTATION_POLICY_VERSION, OPTIMIZER_VERSION
from services.canonical_hash import canonical_sha256

import json
import hashlib
import urllib.request

# Bump this on every meaningful edit. /health reports it so Base44 can
# verify a deploy landed without grepping Railway logs.
VERSION = "5.36.0-deterministic-reformat-no-editorial-ai"

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

    # ── Reformat-from-baseline mode (Design A: DETERMINISTIC, no editorial-AI) ──
    # The auditor-grade re-format path. When `baselineUrl` is set, the engine
    # SKIPS transcription entirely: it fetches the immutable baseline JSON
    # (the same {utterances, words, audio_events} shape produced by Scribe/AAI
    # and persisted by Base44 at cc-baselines/{project_id}/aai-baseline.json),
    # and runs the DETERMINISTIC formatting pipeline against it — segmentation →
    # shaping → sequence optimizer → readability → condensation → QC → export —
    # with the non-deterministic editorial-AI (OpenAI) grammar pass EXPLICITLY
    # SKIPPED (run_caption_job sets allow_editorial_ai = not reformat_from_baseline;
    # process_caption_job then never calls editorial_refine_cues). This is what
    # powers a spec swap (e.g. 32×2 → 42×3): the new spec geometry flows in via
    # captionOptions/env, so the DETERMINISTIC line-breaker (linebreak.py)
    # re-decides line breaks for the new width (killing orphan lines) — NOT an
    # LLM and NOT a crude mechanical re-wrap. Because no LLM runs, the path is
    # byte-reproducible: same baseline + same spec version + same overrides →
    # identical canonical_output_hash on every run, across restarts and repeated
    # dispatch, with zero OpenAI calls. Provider-agnostic — works for any
    # baseline regardless of which provider originally transcribed it.
    # SOC 2 CC8.1 — the reformat critical path is provably deterministic.
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


def fetch_baseline_json(baseline_url: str):
    """
    Fetch + parse the immutable baseline JSON from a signed S3 GET URL.
    Returns (data, raw_bytes) — the parsed dict and the exact fetched bytes
    (the raw bytes let the caller pin the baseline artifact hash).

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
    # Return the parsed dict AND the exact fetched bytes so the caller can pin
    # the baseline artifact hash (SHA-256 of the immutable bytes) into the run's
    # deterministic input tuple. SOC 2 CC8.1 — the reformat provably operated on
    # a specific baseline artifact.
    return data, raw


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


def _sha256_hex_bytes(raw: bytes) -> str:
    """Raw SHA-256 hex of arbitrary bytes — used for the baseline artifact hash
    (the exact bytes the engine fetched from S3). Distinct from canonical_sha256
    (which hashes a canonical JSON structure); here we want the byte-identity of
    the fetched artifact itself."""
    return hashlib.sha256(raw).hexdigest()


def baseline_content_hash(baseline: Dict[str, Any]) -> str:
    """CANONICAL CONTENT hash of the immutable baseline — the DETERMINISM-IDENTITY
    hash used in the deterministic input tuple (Design A, acceptance point 1).

    Distinct from the artifact-byte hash (_sha256_hex_bytes): two byte-different
    serializations of the SAME logical baseline (whitespace, key order, an added
    non-content field like a re-sign timestamp) MUST yield the SAME content hash,
    or a cosmetic re-serialization of S3 bytes would falsely read as a different
    deterministic input. We therefore hash ONLY the delivery-relevant baseline
    content in canonical order: every utterance's (source id, speaker, start,
    end, text, ordered word timings) plus every audio event's (type, start, end,
    text). Canonical via canonical_sha256 (sorted keys, NFC, fixed separators,
    version-prefixed) so it is byte-identical in Python + JS. The byte hash is
    ALSO recorded on the tuple (baseline_artifact_hash) for tamper evidence; the
    CONTENT hash is what proves 'same logical baseline'. SOC 2 CC8.1."""
    utterances = []
    for u in (baseline.get("utterances") or []):
        words = []
        for w in (u.get("words") or []):
            words.append({
                "t": w.get("text", ""),
                "s": w.get("start", w.get("start_ms")),
                "e": w.get("end", w.get("end_ms")),
            })
        utterances.append({
            "u": u.get("source_utterance_id", u.get("utterance_index")),
            "sp": u.get("speaker"),
            "s": u.get("start"),
            "e": u.get("end"),
            "t": u.get("text", ""),
            "w": words,
        })
    audio_events = []
    for ev in (baseline.get("audio_events") or []):
        audio_events.append({
            "et": ev.get("event_type") or ev.get("type"),
            "s": ev.get("start", ev.get("start_ms")),
            "e": ev.get("end", ev.get("end_ms")),
            "t": ev.get("text", ""),
        })
    return canonical_sha256({
        "kind": "cc_baseline_content",
        "language_code": baseline.get("language_code"),
        "utterances": utterances,
        "audio_events": audio_events,
    })


def canonical_output_hash(cues: List[Dict[str, Any]], qc: Optional[Dict[str, Any]] = None) -> str:
    """CANONICAL delivery-output hash covering ALL delivery-relevant fields in
    canonical order (Design A, acceptance point 5). This is the reproducibility
    probe: two deterministic runs of the same tuple MUST produce the same value.

    Covers, per cue, in sequence order: cue index, cue type, start_ms, end_ms,
    speaker_label, structured speaker_segments (per-line speaker attribution),
    review-required flag, and the LINE STRUCTURE (the explicit lines[] array —
    line count + each line's exact text, NOT a flattened string), plus the
    relevant per-cue QC disposition (segmentation_qc status/severity/codes/
    review from the engine's QC result when present). Deterministic via
    canonical_sha256 (byte-identical to the JS mirror in ccIngestReformatResult).
    SOC 2 CC8.1 — the delivered bytes are provably reproducible from the tuple."""
    # Build a per-cue QC disposition lookup keyed by 0-based cue index, if the
    # engine QC result is supplied. The engine's segmentation QC cue_summaries
    # carry cue_id '#<0-based>'.
    qc_by_index: Dict[int, Dict[str, Any]] = {}
    if isinstance(qc, dict):
        for summ in (qc.get("cue_summaries") or []):
            cid = summ.get("cue_id")
            if isinstance(cid, str) and cid.startswith("#"):
                rest = cid[1:]
                if rest.isdigit():
                    qc_by_index[int(rest)] = summ

    projected = []
    for i, c in enumerate(cues):
        lines = c.get("lines") or []
        seg = c.get("speaker_segments") or []
        summ = qc_by_index.get(i)
        qc_disp = None
        if summ:
            qc_disp = {
                "st": summ.get("segmentation_qc_status"),
                "sev": summ.get("segmentation_qc_highest_severity"),
                "codes": sorted([str(x) for x in (summ.get("segmentation_qc_issue_codes") or [])]),
                "rr": bool(summ.get("segmentation_qc_review_required")),
            }
        projected.append({
            "i": i,
            "tp": c.get("type", "dialogue"),
            "s": int(c.get("start_ms", 0)),
            "e": int(c.get("end_ms", 0)),
            "sp": c.get("speaker_label", "") or "",
            "seg": [{"sp": str(s.get("speaker")), "t": str(s.get("text", ""))}
                    for s in seg if s and s.get("speaker") is not None],
            "rr": bool(c.get("speaker_review_required", False)),
            # LINE STRUCTURE — the explicit line array (count + exact text),
            # never a flattened join, so a re-line-break with identical joined
            # text is still detected as a different delivery.
            "lines": [str(ln) for ln in lines],
            "qc": qc_disp,
        })
    return canonical_sha256({"kind": "cc_delivery_output", "cues": projected})


def build_deterministic_inputs(
    *,
    baseline_hash: Optional[str],
    baseline_artifact_hash: Optional[str],
    overlay_hash: Optional[str],
    spec_version: Any,
    spec_slug: Optional[str],
    overrides_source: Dict[str, Any],
    editorial_mode: str,
) -> Dict[str, Any]:
    """Assemble the COMPLETE deterministic input tuple pinned on every run
    (Design A). This is the reproducibility contract: two runs with an identical
    tuple MUST produce byte-identical output. The ingester persists it on
    CCFormatRun so an auditor can prove — from the row alone — exactly which
    inputs produced a delivery, and re-derive it.

    Fields (per the Design A spec):
      • baseline_hash              — SHA-256 of the immutable transcription
                                     baseline artifact bytes the engine fetched.
      • overlay_hash               — SHA-256 of the frozen editorial overlay
                                     artifact, or the literal 'NONE' when no
                                     overlay was applied (deterministic-formatter-
                                     only path). Never null → 'NONE' is an
                                     explicit, auditable state.
      • spec_slug / spec_version   — the pinned (slug, version) rule set.
      • overrides_hash             — canonical hash of the per-run operator
                                     overrides (the captionOptions env bundle),
                                     so an override change is provably a different
                                     input tuple.
      • formatter_version          — the deterministic formatter generation.
      • segmentation_policy_version— the optimizer's scoring/veto policy gen.
      • optimizer_version          — the optimizer code generation.
      • engine_version             — the full engine build string.
      • editorial_mode             — 'deterministic' | 'ai_assist'. Records
                                     whether the non-deterministic LLM stage was
                                     permitted on THIS run (always 'deterministic'
                                     for reformat_from_baseline).
    SOC 2 CC8.1 — every run is reproducible from this tuple.
    """
    overrides_hash = canonical_sha256({
        "kind": "cc_run_overrides",
        # captionOptions is a flat string→string bundle; hash it canonically so
        # key order can never change the hash. This is the exact env the engine
        # applied, so it captures every spec-derived + operator-override knob.
        "overrides": {str(k): str(v) for k, v in (overrides_source or {}).items()},
    })
    return {
        # baseline_hash IS the canonical CONTENT hash (the determinism-identity
        # anchor, acceptance point 1). Two logically-identical baselines share it
        # even if their S3 bytes differ cosmetically.
        "baseline_hash": baseline_hash,
        # baseline_artifact_hash is the raw byte hash of the exact fetched S3
        # object — tamper evidence, NOT part of the determinism identity.
        "baseline_artifact_hash": baseline_artifact_hash,
        "overlay_hash": overlay_hash if overlay_hash else "NONE",
        "spec_slug": spec_slug,
        "spec_version": spec_version,
        "overrides_hash": overrides_hash,
        "formatter_version": FORMATTER_VERSION,
        "segmentation_policy_version": SEGMENTATION_POLICY_VERSION,
        "optimizer_version": OPTIMIZER_VERSION,
        "engine_version": VERSION,
        "editorial_mode": editorial_mode,
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
        # ── DESIGN A DETERMINISM CONTRACT ────────────────────────────────
        # allow_editorial_ai is the ONE switch that permits the non-deterministic
        # OpenAI polish. It is TRUE only for transcription paths (initial author-
        # ing) and FALSE for reformat_from_baseline (Apply Spec), so Apply Spec is
        # byte-deterministic and never calls OpenAI. baseline_hash pins the exact
        # immutable baseline artifact the reformat operated on. SOC 2 CC8.1.
        allow_editorial_ai = not payload.get("reformat_from_baseline")
        baseline_hash: Optional[str] = None
        baseline_artifact_hash: Optional[str] = None
        if payload.get("reformat_from_baseline"):
            # ── Reformat-from-baseline (Design A: DETERMINISTIC) ─────────
            # Skip transcription entirely. Fetch the immutable baseline JSON
            # from the signed S3 URL and run the DETERMINISTIC formatting
            # pipeline (formatter → shaping → optimizer → readability → QC)
            # against it — NO editorial-AI (allow_editorial_ai=False). The new
            # spec geometry arrived via captionOptions/env (already applied
            # above); deterministic line-breaking (linebreak.py) rebalances for
            # the new width. Same baseline + same spec version → byte-identical
            # output every run, with no OpenAI call. A frozen editorial overlay
            # (Phase 2) is consumed here when present; when absent (the Phase 1
            # universal case) the deterministic formatter is the sole authority.
            baseline_url = payload.get("baselineUrl")
            if not baseline_url:
                raise ValueError("baselineUrl is required when reformat_from_baseline=true")
            update_job(job_id, status="processing", stage="fetching_transcript",
                       transcription_provider=provider,
                       transcription_model=model_id)
            print(f"[{job_id}] Reformat-from-baseline (deterministic, no editorial-AI): fetching baseline JSON")
            baseline, baseline_raw = fetch_baseline_json(str(baseline_url))
            # Point 1: record BOTH hashes. The CONTENT hash is the determinism
            # identity (goes into the tuple's baseline_hash); the ARTIFACT byte
            # hash is tamper evidence (baseline_artifact_hash).
            baseline_artifact_hash = _sha256_hex_bytes(baseline_raw)
            baseline_hash = baseline_content_hash(baseline)
            assembly_result = baseline_to_assembly_result(baseline)
            print(f"[{job_id}] Baseline loaded: {len(assembly_result.get('utterances') or [])} utterances, "
                  f"{len(assembly_result.get('audio_events') or [])} audio events, "
                  f"baseline_hash={baseline_hash[:12]}…")
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
            # DESIGN A: AI polish permitted ONLY on the transcription/authoring
            # path, NEVER on reformat_from_baseline. This is the executable
            # guarantee that Apply Spec is deterministic + calls no OpenAI.
            allow_editorial_ai=allow_editorial_ai,
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
        # ── DESIGN A: COMPLETE DETERMINISTIC INPUT TUPLE ──────────────────
        # Pinned on EVERY run so the Base44 ingester can persist it on
        # CCFormatRun and an auditor can prove reproducibility from the row.
        # baseline_hash is set on the reformat path (the artifact the engine
        # fetched); on the transcription path there is no pre-existing baseline
        # artifact (the run CREATES it), so baseline_hash is None there.
        # overlay_hash is 'NONE' in Phase 1 (no overlay is ever applied yet);
        # Phase 2 sets it to the frozen overlay artifact hash when one is
        # consumed. editorial_mode records whether the LLM stage was permitted.
        caption_result["deterministic_inputs"] = build_deterministic_inputs(
            baseline_hash=baseline_hash,
            baseline_artifact_hash=baseline_artifact_hash,
            overlay_hash=None,  # Phase 1: no overlay is ever applied → 'NONE'
            spec_version=env_overrides.get("SPEC_VERSION"),
            spec_slug=env_overrides.get("SPEC_SLUG"),
            overrides_source=env_overrides,
            editorial_mode=("ai_assist" if allow_editorial_ai else "deterministic"),
        )
        # Point 5: the engine ALSO computes the canonical delivery-output hash
        # over all delivery-relevant fields (sequence, type, timing, speaker,
        # segments, line structure, per-cue QC disposition) and pins it on the
        # result. The Base44 ingester persists this as the run's output_hash so
        # the reproducibility probe is byte-identical Python↔JS. On a
        # deterministic reformat, an identical tuple MUST reproduce this value.
        caption_result["canonical_output_hash"] = canonical_output_hash(
            caption_result.get("cues") or [], caption_result.get("segmentation_qc"))

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
        # Operator-confirmed source language (ISO-639-1 for AAI). When present,
        # submit_transcription_job pins it and forces detection OFF — the
        # confirmed language is authoritative, no silent auto-detect fallback.
        # None / "auto" → AAI auto-detects under language_detection. SOC 2 CC8.1.
        source_language_code = payload_data.get("source_language_code") or None
        try:
            provider_transcript_id = submit_transcription_job(
                media_url=media_url,
                speaker_labels=speaker_labels,
                language_detection=language_detection,
                language_code=source_language_code,
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


# ─── Bounded diagnostic route (deployed-engine verification) ─────────
# LOCKED DOWN by design (see services/diagnostics.py):
#   • 404 unless ENABLE_ENGINE_DIAGNOSTICS=true — disabling needs NO code deploy,
#     just flip the env var; the route vanishes on next boot.
#   • requires X-Engine-Secret (same as /v1/jobs).
#   • body accepts ONLY { "fixture_id": <allowlisted string> } — ANY extra key
#     (tokens, transcript, env, captionOptions, path, ...) is rejected 400.
#   • strict request-size limit; no provider calls, no S3, no network, no DB,
#     no persistence, no fixture-transcript logging.
#   • runs the REAL process_caption_job over committed fixtures + returns a
#     bounded diagnostic result and a FIXED build manifest (SHA-256 of the ten
#     known runtime files — never a caller-supplied path).
#   • zero effect on /v1/jobs.
# The Pydantic model with model_config forbidding extras enforces the
# "reject arbitrary token/transcript inputs" requirement declaratively.

# Max diagnostic request body — a fixture request is a few dozen bytes. 4KB is a
# generous ceiling that still rejects any attempt to smuggle a payload.
_DIAG_MAX_BODY_BYTES = 4096


def _diagnostics_enabled() -> bool:
    return (os.getenv("ENABLE_ENGINE_DIAGNOSTICS", "") or "").strip().lower() in ("1", "true", "yes")


class DiagnosticFixtureRequest(BaseModel):
    # forbid ANY key other than fixture_id — this is the "reject arbitrary
    # token/transcript inputs" guarantee, enforced by Pydantic, not by hand.
    model_config = {"extra": "forbid"}
    fixture_id: str


@app.post("/v1/diagnostics/caption-fixture")
async def diagnostics_caption_fixture(
    request: Request,
    x_engine_secret: Optional[str] = Header(default=None),
):
    # (1) Availability gate — 404 when diagnostics are disabled (indistinguishable
    # from a nonexistent route; leaks nothing about the engine).
    if not _diagnostics_enabled():
        raise HTTPException(status_code=404, detail="Not Found")
    # (2) Secret — same posture as the production routes.
    _check_secret(x_engine_secret)
    # (3) Strict request-size limit BEFORE parsing.
    raw = await request.body()
    if len(raw) > _DIAG_MAX_BODY_BYTES:
        raise HTTPException(status_code=413, detail="diagnostic request too large")
    # (4) Parse + validate with extra=forbid (rejects any token/transcript input).
    try:
        parsed = json.loads(raw.decode("utf-8")) if raw else {}
        req_model = DiagnosticFixtureRequest(**parsed)
    except Exception:
        # SAFE message only — no paths, no internals, no stack trace.
        raise HTTPException(status_code=400, detail="invalid diagnostic request")
    # (5) Allowlist check + run the real pipeline over the committed fixture.
    from services.diagnostics import run_fixture, DiagnosticsError, ALLOWED_FIXTURE_IDS
    if req_model.fixture_id not in ALLOWED_FIXTURE_IDS:
        raise HTTPException(status_code=400, detail="unknown fixture_id")
    try:
        result = run_fixture(req_model.fixture_id, VERSION)
    except DiagnosticsError:
        raise HTTPException(status_code=400, detail="unknown fixture_id")
    except Exception:
        # Never leak a stack trace / path from a diagnostic run.
        raise HTTPException(status_code=500, detail="diagnostic run failed")
    return {
        "ok": True,
        "engine_version": VERSION,
        "allowed_fixture_ids": list(ALLOWED_FIXTURE_IDS),
        **result,
    }
