import os
import uuid
import re
import time
from typing import Any, Dict, List, Optional, Literal, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="AI CC Creator – Caption Rules Engine + AssemblyAI Orchestrator", version="2.0.0")

# ===========================
# Environment
# ===========================

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")
ALLOWED_ORIGINS_RAW = os.getenv("ALLOWED_ORIGINS", "").strip()

# Default: safe local fallback (only for webhook URL building in dev)
if not PUBLIC_BASE_URL:
    PUBLIC_BASE_URL = "http://localhost:8000"

def _parse_allowed_origins(raw: str) -> List[str]:
    if not raw:
        return []
    raw = raw.strip()
    if raw == "*":
        return ["*"]
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]

_allowed_origins = _parse_allowed_origins(ALLOWED_ORIGINS_RAW)

# CORS for Base44 preview/prod → Railway (browser calls)
# For production: set ALLOWED_ORIGINS to explicit domains instead of "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins if _allowed_origins else [],
    allow_origin_regex=None,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

# ===========================
# Models
# ===========================

class Rules(BaseModel):
    # NBCU-ish defaults
    maxCharsPerLine: int = 32
    maxLines: int = 2
    maxCPS: float = 17.0
    minDurationMs: int = 1000
    maxDurationMs: int = 7000
    # NOTE: minGapMs is no longer used as a hard split trigger (it causes micro-cues)
    minGapMs: int = 80
    preferPunctuationBreaks: bool = True

    # SCC specifics
    sccFrameRate: float = 29.97
    startAtHour00: bool = True

class Word(BaseModel):
    text: str
    start: int
    end: int
    speaker: Optional[str] = "A"

class Event(BaseModel):
    type: Literal["music", "applause", "laughter", "sfx", "foreign_language"]
    start: int
    end: int
    text: Optional[str] = None
    language: Optional[str] = None

class FormatRequest(BaseModel):
    rules: Rules = Field(default_factory=Rules)
    words: Optional[List[Word]] = None
    cues: Optional[List[Dict[str, Any]]] = None
    events: Optional[List[Event]] = None

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

# ===========================
# In-memory job store (MVP)
# ===========================
JOBS: Dict[str, JobStatus] = {}
JOB_RULES: Dict[str, Rules] = {}
JOB_CREATED_AT: Dict[str, float] = {}

# ===========================
# Utilities
# ===========================

PUNCT_BREAK_RE = re.compile(r"[.!?…]+$")
SENTENCE_BREAK_RE = re.compile(r"[.!?…]+$")

def _clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))

def _ms_to_srt_time(ms: int) -> str:
    h = ms // 3600000
    ms -= h * 3600000
    m = ms // 60000
    ms -= m * 60000
    s = ms // 1000
    ms -= s * 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def _ms_to_vtt_time(ms: int) -> str:
    h = ms // 3600000
    ms -= h * 3600000
    m = ms // 60000
    ms -= m * 60000
    s = ms // 1000
    ms -= s * 1000
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"

def _normalize_speaker(s: Optional[str]) -> str:
    if not s:
        return "A"
    s = str(s).strip()
    if not s:
        return "A"
    s = s.replace("Speaker ", "").strip()
    return s or "A"

def _is_bracket_token(t: str) -> bool:
    t = (t or "").strip()
    return len(t) >= 3 and t.startswith("[") and t.endswith("]")

def _strip_bracket_tokens(words: List[Word]) -> List[Word]:
    cleaned: List[Word] = []
    for w in words:
        t = (w.text or "").strip()
        if _is_bracket_token(t):
            continue
        # partial bracket tokens can happen; drop them too
        if t.startswith("[") or t.endswith("]"):
            continue
        cleaned.append(w)
    return cleaned

# ===========================
# SOUND CUE OPTION A (heuristic)
# ===========================

SFX_MAP = [
    (re.compile(r"\b(music)\b", re.I), ("music", "[♪ MUSIC ♪]")),
    (re.compile(r"\b(applause|clapping)\b", re.I), ("applause", "[APPLAUSE]")),
    (re.compile(r"\b(laughter|laughing)\b", re.I), ("laughter", "[LAUGHTER]")),
    (re.compile(r"\b(door|slam|bang|gunshot|explosion|siren|phone|ring|knock)\b", re.I), ("sfx", "[SFX]")),
]

def detect_events_from_transcript_tokens(
    words: List[Word],
    transcript_text: Optional[str] = None
) -> List[Event]:
    """
    Option A: lightweight event detection.
    - If transcript contains bracket tokens like [MUSIC], we convert them into events.
    - Otherwise: heuristic scan over transcript_text for keywords and place a generic event
      spanning the nearest word window (best-effort).
    This is not perfect, but gets SDH cues in the output that humans can audit/edit.
    """
    events: List[Event] = []

    # 1) Use explicit bracket tokens if present in word stream
    # NOTE: AssemblyAI often does NOT emit bracket tokens unless you enable specific features.
    for w in words:
        t = (w.text or "").strip()
        if _is_bracket_token(t):
            upper = t.upper()
            for rx, (etype, out_text) in SFX_MAP:
                if rx.search(upper):
                    events.append(Event(type=etype, start=w.start, end=w.end, text=out_text))
                    break

    if events:
        return events

    # 2) Fallback: keyword scan transcript text (best-effort placement)
    if not transcript_text:
        return events

    # Find first/last timing bounds for placement
    if not words:
        return events
    start_ms = int(words[0].start)
    end_ms = int(words[-1].end)

    # If keywords exist, insert a single early event as a flag (auditable)
    lowered = transcript_text.lower()
    for rx, (etype, out_text) in SFX_MAP:
        if rx.search(lowered):
            events.append(Event(type=etype, start=start_ms, end=_clip(start_ms + 1500, start_ms, end_ms), text=out_text))
    return events

# ===========================
# Caption building (PRO – packs into 32x2 and preserves end-times)
# ===========================

def _wrap_into_lines(tokens: List[str], max_chars: int, max_lines: int) -> List[str]:
    """
    Greedy packing into <= max_lines lines, each <= max_chars where possible.
    """
    lines: List[str] = []
    cur = ""

    def push():
        nonlocal cur
        if cur.strip():
            lines.append(cur.strip())
        cur = ""

    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if not cur:
            cur = tok
        else:
            if len(cur) + 1 + len(tok) <= max_chars:
                cur = f"{cur} {tok}"
            else:
                push()
                cur = tok
                if len(lines) >= max_lines:
                    # overflow: append to last line (will violate; QC flags)
                    lines[-1] = (lines[-1] + " " + cur).strip()
                    cur = ""

    push()

    if len(lines) > max_lines:
        head = lines[:max_lines-1]
        tail = " ".join(lines[max_lines-1:])
        lines = head + [tail]

    return lines

def _cue_char_count(text: str) -> int:
    return len(text.replace("\n", "").strip())

def _cps(chars: int, duration_ms: int) -> float:
    if duration_ms <= 0:
        return 9999.0
    return chars / (duration_ms / 1000.0)

def build_cues_from_words_pro(words: List[Word], rules: Rules) -> List[Dict[str, Any]]:
    """
    PRO strategy:
    - Group contiguous words by speaker (primary)
    - Within speaker run, pack words into cues that:
        * try to fill up to 32x2 (or rule values)
        * keep CPS <= maxCPS by splitting earlier (but NOT into micro-cues)
        * NEVER extend end past the last spoken word end
        * Prefer splitting at punctuation when possible
    - No minGapMs hard splits (micro-cue killer)
    """
    if not words:
        return []

    words = sorted(words, key=lambda w: (w.start, w.end))

    cues: List[Dict[str, Any]] = []

    i = 0
    n = len(words)

    while i < n:
        sp = _normalize_speaker(words[i].speaker)
        run_start = i

        # build a speaker run
        while i < n and _normalize_speaker(words[i].speaker) == sp:
            i += 1
        run_end = i  # exclusive

        run_words = words[run_start:run_end]

        # Now chunk within this run
        j = 0
        while j < len(run_words):
            cue_start = run_words[j].start
            tokens: List[str] = []
            cue_end = run_words[j].end

            # We'll grow until we'd violate CPS badly or duration too long.
            # We also try to fill 2 lines x 32 chars by adding words until near capacity.
            last_good_k = j

            k = j
            while k < len(run_words):
                w = run_words[k]
                tokens.append(w.text)
                cue_end = w.end

                # Tentative wrap
                lines = _wrap_into_lines(tokens, rules.maxCharsPerLine, rules.maxLines)
                text = "\n".join(lines)
                chars = _cue_char_count(text)
                dur = max(1, cue_end - cue_start)
                cps_val = _cps(chars, dur)

                # Duration hard cap
                if dur > rules.maxDurationMs:
                    tokens.pop()
                    cue_end = run_words[k-1].end if k-1 >= j else w.end
                    break

                # If CPS too high, we should stop, but avoid micro-cues:
                # require at least ~0.9s or at least 5 words before splitting.
                if cps_val > rules.maxCPS and (dur >= 900 or (k - j + 1) >= 5):
                    # back off one word if that helps materially
                    tokens.pop()
                    cue_end = run_words[k-1].end if k-1 >= j else w.end
                    break

                # Prefer punctuation boundary: mark last_good_k
                if rules.preferPunctuationBreaks and SENTENCE_BREAK_RE.search((w.text or "").strip()):
                    last_good_k = k

                # If we already nearly filled 2 lines, consider stopping at punctuation boundary.
                # Rough capacity: maxCharsPerLine * maxLines
                capacity = rules.maxCharsPerLine * rules.maxLines
                if chars >= int(capacity * 0.92):
                    # stop, but if we have a recent punctuation split, cut there
                    break

                k += 1

            # If we stopped due to "filled" and we have a punctuation point inside this cue, cut there
            # only if it meaningfully reduces awkward splits.
            if last_good_k > j and last_good_k < (j + len(tokens) - 1):
                # cut tokens to last_good_k
                cut_len = (last_good_k - j + 1)
                tokens = tokens[:cut_len]
                cue_end = run_words[last_good_k].end

            # Final text
            lines = _wrap_into_lines(tokens, rules.maxCharsPerLine, rules.maxLines)
            text = "\n".join(lines).strip()

            cues.append({
                "start": int(cue_start),
                "end": int(cue_end),
                "text": text,
                "speaker": sp,
            })

            # Advance j to next word after this cue
            # Find how many words we used
            used = len(tokens)
            j = j + used

    return cues

# ===========================
# Speaker formatting + “dash” mode (Option A)
# ===========================

def apply_speaker_style_dash(cues: List[Dict[str, Any]], rules: Rules) -> List[Dict[str, Any]]:
    """
    Option A:
    If consecutive cues overlap or are very close and speaker changes,
    present as 2-line dash format:
      - line from speaker1
      - line from speaker2
    This is a UI-friendly proxy for “two people speaking” and matches common SDH style.

    NOTE: True overlap detection requires word-level alignment; this is best-effort.
    """
    if not cues:
        return cues

    out: List[Dict[str, Any]] = []
    i = 0

    def text_lines(t: str) -> List[str]:
        return (t or "").split("\n")

    while i < len(cues):
        cur = cues[i]
        if i + 1 < len(cues):
            nxt = cues[i + 1]

            # speaker change and near-adjacent timing
            if cur.get("speaker") != nxt.get("speaker"):
                gap = int(nxt["start"]) - int(cur["end"])
                # treat small gap/overlap as “exchange” candidate
                if gap <= 120:
                    # Merge into one 2-line cue if total duration is sane
                    start = int(cur["start"])
                    end = int(max(int(cur["end"]), int(nxt["end"])))

                    a_text = (cur.get("text") or "").replace("\n", " ")
                    b_text = (nxt.get("text") or "").replace("\n", " ")

                    # Build two lines with dashes (cap to 2 lines)
                    line1 = f"- {a_text}".strip()
                    line2 = f"- {b_text}".strip()

                    # Soft clamp to max chars per line by trimming (QC will still flag if too long)
                    line1 = line1[: max(1, rules.maxCharsPerLine)]
                    line2 = line2[: max(1, rules.maxCharsPerLine)]

                    out.append({
                        "start": start,
                        "end": end,
                        "text": f"{line1}\n{line2}",
                        "speaker": "A",  # merged display cue
                        "speakerMerged": True,
                    })
                    i += 2
                    continue

        out.append(cur)
        i += 1

    return out

# ===========================
# QC
# ===========================

def qc_cues(cues: List[Dict[str, Any]], rules: Rules) -> Dict[str, Any]:
    issues: List[Dict[str, Any]] = []

    for idx, c in enumerate(cues, start=1):
        start = int(c["start"])
        end = int(c["end"])
        text = str(c.get("text") or "")
        duration = max(1, end - start)

        lines = text.split("\n")
        for ln in lines:
            if len(ln) > rules.maxCharsPerLine:
                issues.append({"cue": idx, "type": "line_too_long", "value": len(ln)})

        if len(lines) > rules.maxLines:
            issues.append({"cue": idx, "type": "too_many_lines", "value": len(lines)})

        chars = _cue_char_count(text)
        cps_val = _cps(chars, duration)
        if cps_val > rules.maxCPS:
            issues.append({"cue": idx, "type": "cps_high", "value": round(cps_val, 2)})

        if duration < rules.minDurationMs:
            issues.append({"cue": idx, "type": "too_short_ms", "value": duration})

        if duration > rules.maxDurationMs:
            issues.append({"cue": idx, "type": "too_long_ms", "value": duration})

    return {"issuesCount": len(issues), "issues": issues}

# ===========================
# Exports
# ===========================

def to_srt(cues: List[Dict[str, Any]]) -> str:
    out: List[str] = []
    for i, c in enumerate(cues, start=1):
        out.append(str(i))
        out.append(f"{_ms_to_srt_time(int(c['start']))} --> {_ms_to_srt_time(int(c['end']))}")
        out.append(str(c.get("text") or ""))
        out.append("")
    return "\n".join(out).strip() + "\n"

def to_vtt(cues: List[Dict[str, Any]]) -> str:
    out: List[str] = ["WEBVTT", ""]
    for c in cues:
        out.append(f"{_ms_to_vtt_time(int(c['start']))} --> {_ms_to_vtt_time(int(c['end']))}")
        out.append(str(c.get("text") or ""))
        out.append("")
    return "\n".join(out).strip() + "\n"

# SCC (still MVP – not full 608 packing)
def _ms_to_scc_timecode(ms: int, fps: float) -> str:
    fps_i = 30 if abs(fps - 29.97) < 0.05 else int(round(fps))
    total_seconds = ms / 1000.0
    h = int(total_seconds // 3600)
    m = int((total_seconds % 3600) // 60)
    s = int(total_seconds % 60)
    frac = total_seconds - int(total_seconds)
    ff = int(round(frac * fps_i))
    ff = _clip(ff, 0, fps_i - 1)
    return f"{h:02}:{m:02}:{s:02}:{ff:02}"

def _text_to_fake_scc_hex(text: str) -> str:
    safe = text.encode("latin-1", errors="replace")
    parts = [f"{b:02x}" for b in safe[:120]]
    grouped = []
    for i in range(0, len(parts), 2):
        if i + 1 < len(parts):
            grouped.append(parts[i] + parts[i + 1])
        else:
            grouped.append(parts[i] + "20")
    return " ".join(grouped)

def to_scc(cues: List[Dict[str, Any]], rules: Rules) -> str:
    lines = ["Scenarist_SCC V1.0", ""]
    for c in cues:
        tc = _ms_to_scc_timecode(int(c["start"]), rules.sccFrameRate)
        text = str(c.get("text") or "").replace("\n", " ")
        hex_payload = "94ae 94ae 9420 9420 9470 9470 " + _text_to_fake_scc_hex(text) + " 942c 942c 942f 942f"
        lines.append(f"{tc}\t{hex_payload}")
        lines.append("")
    final_end = int(cues[-1]["end"]) if cues else 0
    lines.append(_ms_to_scc_timecode(max(0, final_end), rules.sccFrameRate) + "\t942c 942c")
    lines.append("")
    return "\n".join(lines)

# ===========================
# Core formatter
# ===========================

def format_payload_pro(req: FormatRequest, transcript_text: Optional[str] = None) -> Dict[str, Any]:
    rules = req.rules

    if req.cues is not None:
        cues = req.cues
    else:
        if not req.words:
            raise HTTPException(status_code=422, detail="You must provide either 'words' or 'cues'.")
        words = [Word(**w.model_dump()) for w in req.words]

        # detect SDH events (Option A)
        events = detect_events_from_transcript_tokens(words, transcript_text=transcript_text)

        # build dialog cues (strip bracket tokens if any)
        dialog_words = _strip_bracket_tokens(words)
        cues = build_cues_from_words_pro(dialog_words, rules)

        # Inject SDH events as cues
        if events:
            for e in events:
                cues.append({"start": e.start, "end": e.end, "text": e.text or "[SFX]", "speaker": "A", "eventType": e.type})

        cues = sorted(cues, key=lambda c: (int(c["start"]), int(c["end"])))

        # Option A dash merge for speaker exchanges
        cues = apply_speaker_style_dash(cues, rules)

    qc = qc_cues(cues, rules)

    return {
        "rules": rules.model_dump(),
        "cues": cues,
        "srt": to_srt(cues),
        "vtt": to_vtt(cues),
        "scc": to_scc(cues, rules),
        "qc": qc,
    }

# ===========================
# AssemblyAI helpers
# ===========================

ASSEMBLY_BASE = "https://api.assemblyai.com/v2"

async def assemblyai_submit(media_url: str, speaker_labels: bool, language_detection: bool, webhook_url: str) -> Dict[str, Any]:
    payload = {
        "audio_url": media_url,
        "speaker_labels": speaker_labels,
        "language_detection": language_detection,
        # REQUIRED (per your earlier 400 error)
        "speech_models": ["universal-3-pro"],
        "webhook_url": webhook_url,
        "webhook_auth_header_name": "X-Webhook-Token",
        "webhook_auth_header_value": WEBHOOK_SECRET,
    }
    headers = {"Authorization": ASSEMBLYAI_API_KEY, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(f"{ASSEMBLY_BASE}/transcript", json=payload, headers=headers)
        r.raise_for_status()
        return r.json()

async def assemblyai_get_transcript(assembly_id: str) -> Dict[str, Any]:
    headers = {"Authorization": ASSEMBLYAI_API_KEY}
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(f"{ASSEMBLY_BASE}/transcript/{assembly_id}", headers=headers)
        r.raise_for_status()
        return r.json()

def _extract_words_from_assembly(t: Dict[str, Any]) -> List[Word]:
    words_raw = t.get("words") or []
    words: List[Word] = []
    for w in words_raw:
        words.append(Word(
            text=w.get("text", ""),
            start=int(w.get("start", 0)),
            end=int(w.get("end", 0)),
            speaker=_normalize_speaker(w.get("speaker") or "A"),
        ))
    return words

# ===========================
# Routes
# ===========================

@app.get("/")
def root():
    return {"ok": True, "service": "ai-cc-creator"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/v1/health")
def health_v1():
    return {"ok": True}

@app.post("/v1/format")
def format_endpoint(req: FormatRequest):
    return format_payload_pro(req)

@app.post("/v1/jobs", response_model=JobStatus)
async def create_job(req: JobCreateRequest):
    if not ASSEMBLYAI_API_KEY:
        raise HTTPException(status_code=500, detail="ASSEMBLYAI_API_KEY is not set in Railway Variables.")
    if not WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="WEBHOOK_SECRET is not set in Railway Variables.")
    if not PUBLIC_BASE_URL:
        raise HTTPException(status_code=500, detail="PUBLIC_BASE_URL is not set in Railway Variables.")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = JobStatus(id=job_id, status="queued")
    JOB_RULES[job_id] = req.rules
    JOB_CREATED_AT[job_id] = time.time()

    webhook_url = f"{PUBLIC_BASE_URL}/v1/webhooks/assemblyai"

    try:
        data = await assemblyai_submit(
            media_url=req.mediaUrl,
            speaker_labels=req.speaker_labels,
            language_detection=req.language_detection,
            webhook_url=webhook_url,
        )
    except httpx.HTTPStatusError as e:
        JOBS[job_id].status = "error"
        JOBS[job_id].error = f"AssemblyAI submit failed: {e.response.status_code} {e.response.text}"
        return JOBS[job_id]
    except Exception as e:
        JOBS[job_id].status = "error"
        JOBS[job_id].error = f"AssemblyAI submit failed: {str(e)}"
        return JOBS[job_id]

    assembly_id = data.get("id")
    JOBS[job_id].status = "processing"
    JOBS[job_id].assembly_id = assembly_id
    return JOBS[job_id]

def _finalize_job_from_transcript(job_id: str, t: Dict[str, Any]) -> None:
    words = _extract_words_from_assembly(t)
    rules = JOB_RULES.get(job_id, Rules())

    transcript_text = t.get("text") or ""
    result = format_payload_pro(FormatRequest(words=words, rules=rules), transcript_text=transcript_text)

    JOBS[job_id].status = "done"
    JOBS[job_id].result = result
    JOBS[job_id].error = None

@app.get("/v1/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    js = JOBS.get(job_id)
    if not js:
        raise HTTPException(status_code=404, detail="Job not found")

    # PRO fallback: if webhook missed, polling can still complete
    if js.status == "processing" and js.assembly_id:
        try:
            t = await assemblyai_get_transcript(js.assembly_id)
            if t.get("status") == "completed":
                _finalize_job_from_transcript(job_id, t)
                return JOBS[job_id]
            if t.get("status") in ("error", "failed"):
                js.status = "error"
                js.error = t.get("error") or "AssemblyAI transcript failed"
                return js
        except Exception:
            # swallow transient failures; caller will poll again
            return js

    return js

@app.post("/v1/webhooks/assemblyai")
async def assemblyai_webhook(request: Request):
    token = request.headers.get("X-Webhook-Token", "")
    if not WEBHOOK_SECRET or token != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid webhook token")

    body = await request.json()
    assembly_id = body.get("transcript_id") or body.get("id")
    status = body.get("status")

    # Find our job
    job_id = None
    for jid, js in JOBS.items():
        if js.assembly_id == assembly_id:
            job_id = jid
            break

    if not job_id:
        return JSONResponse({"ok": True, "ignored": True})

    if status in ("error", "failed"):
        JOBS[job_id].status = "error"
        JOBS[job_id].error = body.get("error") or "AssemblyAI transcript failed"
        return JSONResponse({"ok": True})

    if status != "completed":
        return JSONResponse({"ok": True})

    # Completed: fetch transcript and finalize
    try:
        t = await assemblyai_get_transcript(assembly_id)
        if t.get("status") == "completed":
            _finalize_job_from_transcript(job_id, t)
        else:
            # if completed webhook but transcript still not ready, polling fallback will handle
            pass
    except Exception as e:
        # polling fallback will handle
        JOBS[job_id].error = f"Finalize warning: {str(e)}"

    return JSONResponse({"ok": True})
