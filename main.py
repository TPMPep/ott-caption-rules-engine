import os
import re
from typing import Any, Dict, List, Optional, Literal

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="OTT Caption Rules Engine + AssemblyAI Orchestrator", version="2.0.0")

# ---------------------------
# Environment
# ---------------------------

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")

ALLOWED_ORIGINS_RAW = os.getenv("ALLOWED_ORIGINS", "").strip()
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
    type: Literal["music", "foreign_language"]
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

    # SDH prompt (Option A)
    # If provided, we pass it into AssemblyAI to encourage non-speech cues.
    prompt: Optional[str] = None


class JobStatus(BaseModel):
    id: str  # AssemblyAI transcript id
    status: Literal["queued", "processing", "done", "error"]
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


# ---------------------------
# In-memory rules cache (best-effort)
# - This is only to preserve per-job rules.
# - If Railway restarts, we can still return captions using default Rules.
# ---------------------------

JOB_RULES: Dict[str, Rules] = {}

# ---------------------------
# Utilities
# ---------------------------

PUNCT_BREAK_RE = re.compile(r"[.!?…]+$")


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


def _clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _normalize_speaker(s: Optional[str]) -> str:
    if not s:
        return "A"
    s = str(s).strip()
    if not s:
        return "A"
    s = s.replace("Speaker ", "").strip()
    return s or "A"


def _strip_bracket_tokens(words: List[Word]) -> List[Word]:
    cleaned: List[Word] = []
    for w in words:
        t = (w.text or "").strip()
        if t.startswith("[") and t.endswith("]"):
            continue
        if t.startswith("[") or t.endswith("]"):
            continue
        cleaned.append(w)
    return cleaned


def _has_special_events(events: Optional[List[Event]]) -> bool:
    if not events:
        return False
    return any(e.type in ("music", "foreign_language") for e in events)


# ---------------------------
# Caption cue building
# ---------------------------

def wrap_text(text: str, max_chars: int, max_lines: int, prefer_punct: bool) -> str:
    words = text.split()
    if not words:
        return ""

    lines: List[str] = []
    cur = ""

    def push_line(s: str):
        s = s.strip()
        if s:
            lines.append(s)

    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur = f"{cur} {w}"
        else:
            push_line(cur)
            cur = w
            if len(lines) >= max_lines:
                lines[-1] = (lines[-1] + " " + cur).strip()
                cur = ""

    if cur:
        push_line(cur)

    if len(lines) > max_lines:
        head = lines[: max_lines - 1]
        tail = " ".join(lines[max_lines - 1 :])
        lines = head + [tail]

    return "\n".join(lines)


def build_cues_from_words(words: List[Word], rules: Rules) -> List[Dict[str, Any]]:
    if not words:
        return []

    words = sorted(words, key=lambda w: (w.start, w.end))

    cues: List[Dict[str, Any]] = []
    cur_words: List[Word] = []
    cur_start = words[0].start
    cur_end = words[0].end
    cur_speaker = _normalize_speaker(words[0].speaker)

    def flush():
        nonlocal cur_words, cur_start, cur_end, cur_speaker
        if not cur_words:
            return

        text = " ".join((w.text or "").strip() for w in cur_words).strip()
        if not text:
            cur_words = []
            return

        wrapped = wrap_text(text, rules.maxCharsPerLine, rules.maxLines, rules.preferPunctuationBreaks)

        # IMPORTANT: Do NOT extend beyond the last word end time.
        # We preserve AssemblyAI word timing to prevent “words after they were spoken”.
        duration = cur_end - cur_start
        if duration < rules.minDurationMs:
            # We DO NOT extend end time; QC will flag instead.
            pass

        cues.append(
            {
                "start": int(cur_start),
                "end": int(cur_end),
                "text": wrapped,
                "speaker": cur_speaker,
            }
        )
        cur_words = []

    for w in words:
        sp = _normalize_speaker(w.speaker)

        # speaker change split
        if sp != cur_speaker:
            flush()
            cur_start = w.start
            cur_end = w.end
            cur_speaker = sp
            cur_words = [w]
            continue

        gap = w.start - cur_end
        if gap >= rules.minGapMs:
            flush()
            cur_start = w.start
            cur_end = w.end
            cur_speaker = sp
            cur_words = [w]
            continue

        proposed_end = max(cur_end, w.end)
        if (proposed_end - cur_start) > rules.maxDurationMs:
            flush()
            cur_start = w.start
            cur_end = w.end
            cur_speaker = sp
            cur_words = [w]
            continue

        cur_words.append(w)
        cur_end = max(cur_end, w.end)

        # punctuation split encouragement (only if we already meet min duration)
        if rules.preferPunctuationBreaks and PUNCT_BREAK_RE.search((w.text or "").strip()):
            if (cur_end - cur_start) >= rules.minDurationMs:
                flush()

    flush()
    return cues


# ---------------------------
# QC
# ---------------------------

def qc_cues(cues: List[Dict[str, Any]], rules: Rules) -> Dict[str, Any]:
    issues = []

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

        chars = len(text.replace("\n", ""))
        cps = (chars / (duration / 1000.0)) if duration > 0 else 9999
        if cps > rules.maxCPS:
            issues.append({"cue": idx, "type": "cps_high", "value": round(cps, 2)})

        if duration < rules.minDurationMs:
            issues.append({"cue": idx, "type": "too_short_ms", "value": duration})

        if duration > rules.maxDurationMs:
            issues.append({"cue": idx, "type": "too_long_ms", "value": duration})

    return {"issuesCount": len(issues), "issues": issues}


# ---------------------------
# Exports: SRT/VTT/SCC
# ---------------------------

def to_srt(cues: List[Dict[str, Any]]) -> str:
    out = []
    for i, c in enumerate(cues, start=1):
        out.append(str(i))
        out.append(f"{_ms_to_srt_time(int(c['start']))} --> {_ms_to_srt_time(int(c['end']))}")
        out.append(str(c.get("text") or ""))
        out.append("")
    return "\n".join(out).strip() + "\n"


def to_vtt(cues: List[Dict[str, Any]]) -> str:
    out = ["WEBVTT", ""]
    for c in cues:
        out.append(f"{_ms_to_vtt_time(int(c['start']))} --> {_ms_to_vtt_time(int(c['end']))}")
        out.append(str(c.get("text") or ""))
        out.append("")
    return "\n".join(out).strip() + "\n"


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
    base_offset = 0

    for c in cues:
        start_ms = int(c["start"]) - base_offset
        if start_ms < 0:
            start_ms = 0
        tc = _ms_to_scc_timecode(start_ms, rules.sccFrameRate)
        text = str(c.get("text") or "").replace("\n", " ")
        hex_payload = "94ae 94ae 9420 9420 9470 9470 " + _text_to_fake_scc_hex(text) + " 942c 942c 942f 942f"
        lines.append(f"{tc}\t{hex_payload}")
        lines.append("")

    final_end = int(cues[-1]["end"]) if cues else 0
    lines.append(_ms_to_scc_timecode(max(0, final_end), rules.sccFrameRate) + "\t942c 942c")
    lines.append("")
    return "\n".join(lines)


# ---------------------------
# Core formatter
# ---------------------------

def format_payload(req: FormatRequest) -> Dict[str, Any]:
    rules = req.rules

    if req.cues is not None:
        cues = req.cues
    else:
        if not req.words:
            raise HTTPException(status_code=422, detail="You must provide either 'words' or 'cues'.")

        words = [Word(**w.model_dump()) for w in req.words]

        if _has_special_events(req.events):
            words = _strip_bracket_tokens(words)

        cues = build_cues_from_words(words, rules)

    if req.events:
        for e in req.events:
            if e.type == "music":
                txt = e.text or "[♪ MUSIC ♪]"
                cues.append({"start": e.start, "end": e.end, "text": txt, "speaker": "A"})
            elif e.type == "foreign_language":
                lang = e.language or "Unknown"
                cues.append({"start": e.start, "end": e.end, "text": f"[Speaking {lang}]", "speaker": "A"})
        cues = sorted(cues, key=lambda c: (int(c["start"]), int(c["end"])))

    qc = qc_cues(cues, rules)

    return {
        "rules": rules.model_dump(),
        "cues": cues,
        "srt": to_srt(cues),
        "vtt": to_vtt(cues),
        "scc": to_scc(cues, rules),
        "qc": qc,
    }


# ---------------------------
# Routes
# ---------------------------

@app.get("/")
def root():
    return {"ok": True}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/v1/format")
def format_endpoint(req: FormatRequest):
    return format_payload(req)


@app.post("/v1/jobs", response_model=JobStatus)
async def create_job(req: JobCreateRequest):
    if not ASSEMBLYAI_API_KEY:
        raise HTTPException(status_code=500, detail="ASSEMBLYAI_API_KEY is not set in Railway Variables.")

    webhook_url = f"{PUBLIC_BASE_URL}/v1/webhooks/assemblyai" if WEBHOOK_SECRET else None

    payload: Dict[str, Any] = {
        "audio_url": req.mediaUrl,
        "speaker_labels": req.speaker_labels,
        "language_detection": req.language_detection,
        "speech_models": ["universal-3-pro"],
    }

    if req.prompt:
        payload["prompt"] = req.prompt

    if webhook_url:
        payload["webhook_url"] = webhook_url
        payload["webhook_auth_header_name"] = "X-Webhook-Token"
        payload["webhook_auth_header_value"] = WEBHOOK_SECRET

    headers = {"Authorization": ASSEMBLYAI_API_KEY, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post("https://api.assemblyai.com/v2/transcript", json=payload, headers=headers)

    if r.status_code >= 300:
        return JobStatus(id="(none)", status="error", error=f"AssemblyAI submit failed: {r.status_code} {r.text}")

    data = r.json()
    assembly_id = data.get("id")
    if not assembly_id:
        return JobStatus(id="(none)", status="error", error="AssemblyAI did not return transcript id")

    # Cache rules best-effort
    JOB_RULES[assembly_id] = req.rules

    return JobStatus(id=assembly_id, status="processing", error=None, result=None)


@app.get("/v1/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    if not ASSEMBLYAI_API_KEY:
        return JobStatus(id=job_id, status="error", error="ASSEMBLYAI_API_KEY missing", result=None)

    headers = {"Authorization": ASSEMBLYAI_API_KEY}

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            r = await client.get(f"https://api.assemblyai.com/v2/transcript/{job_id}", headers=headers)
        except httpx.RequestError as e:
            return JobStatus(id=job_id, status="processing", error=f"AssemblyAI network error: {str(e)}", result=None)

    if r.status_code >= 300:
        # This is the exact error you saw; return it cleanly so Base44 stops spinning blindly.
        return JobStatus(id=job_id, status="error", error=f"AssemblyAI get transcript failed: {r.status_code} {r.text}", result=None)

    t = r.json()
    status = t.get("status")

    if status in ("queued", "processing"):
        return JobStatus(id=job_id, status="processing", error=None, result=None)

    if status in ("error", "failed"):
        return JobStatus(id=job_id, status="error", error=t.get("error") or "AssemblyAI transcript failed", result=None)

    if status != "completed":
        return JobStatus(id=job_id, status="processing", error=None, result=None)

    # Completed → format
    words_raw = t.get("words") or []
    words: List[Word] = []
    for w in words_raw:
        words.append(
            Word(
                text=w.get("text", ""),
                start=int(w.get("start", 0)),
                end=int(w.get("end", 0)),
                speaker=_normalize_speaker(w.get("speaker") or "A"),
            )
        )

    rules = JOB_RULES.get(job_id, Rules())
    result = format_payload(FormatRequest(words=words, events=[], rules=rules))
    return JobStatus(id=job_id, status="done", error=None, result=result)


@app.post("/v1/webhooks/assemblyai")
async def assemblyai_webhook(request: Request):
    # Optional webhook mode; polling still works without this.
    if WEBHOOK_SECRET:
        token = request.headers.get("X-Webhook-Token", "")
        if token != WEBHOOK_SECRET:
            raise HTTPException(status_code=401, detail="Invalid webhook token")

    return JSONResponse({"ok": True})
