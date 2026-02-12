import os
import uuid
import re
from typing import Any, Dict, List, Optional, Literal, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(
    title="OTT Caption Rules Engine + AssemblyAI Orchestrator",
    version="2.0.0-pro",
)

# ---------------------------
# Environment
# ---------------------------

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")

# Comma-separated list of allowed origins:
# Example:
# ALLOWED_ORIGINS="https://preview-sandbox--xxxx.base44.app,https://ai-cc-creator.yourdomain.com"
ALLOWED_ORIGINS_RAW = os.getenv("ALLOWED_ORIGINS", "").strip()

if not PUBLIC_BASE_URL:
    # Only used to build webhook URLs; set PUBLIC_BASE_URL in Railway to your service URL
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

# CORS (browser calls in preview)
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
    # “Broadcast-ish” defaults; Base44 can override per-job
    maxCharsPerLine: int = 32
    maxLines: int = 2
    maxCPS: float = 17.0
    minDurationMs: int = 1000
    maxDurationMs: int = 7000
    minGapMs: int = 80
    preferPunctuationBreaks: bool = True

    # “Packing” behavior (pro)
    targetFillRatio: float = 0.80  # try to fill ~80% of capacity before cutting
    allowVeryShortIfForced: bool = True  # if we can't merge, allow short (but QC flags)

    # SCC specifics
    sccFrameRate: float = 29.97
    startAtHour00: bool = True


class Word(BaseModel):
    text: str
    start: int
    end: int
    speaker: Optional[str] = "A"


class Event(BaseModel):
    type: Literal["music", "foreign_language", "sfx"]
    start: int
    end: int
    text: Optional[str] = None       # for music/sfx
    language: Optional[str] = None   # for foreign_language


class FormatRequest(BaseModel):
    rules: Rules = Field(default_factory=Rules)
    words: Optional[List[Word]] = None
    cues: Optional[List[Dict[str, Any]]] = None  # {start,end,text,speaker,kind?}
    events: Optional[List[Event]] = None


class JobCreateRequest(BaseModel):
    mediaUrl: str
    rules: Rules = Field(default_factory=Rules)

    speaker_labels: bool = True
    language_detection: bool = True

    # Pro knobs (optional UI toggles later)
    enable_sdh_prompt: bool = True     # ask AssemblyAI to emit [LAUGHTER], etc.
    speakers_expected: Optional[int] = None  # if you *know* the count, improves diarization


class JobStatus(BaseModel):
    id: str
    status: Literal["queued", "processing", "done", "error"]
    error: Optional[str] = None
    assembly_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


# ---------------------------
# In-memory job store (MVP)
# NOTE: Production: move to Redis/Postgres. In-memory resets on deploy.
# ---------------------------

JOBS: Dict[str, JobStatus] = {}
JOB_RULES: Dict[str, Rules] = {}

# ---------------------------
# Utilities
# ---------------------------

PUNCT_BREAK_RE = re.compile(r"[.!?…]+$")
BRACKET_TOKEN_RE = re.compile(r"^\[.*\]$")

DEFAULT_SDH_PROMPT = (
    "Transcribe dialogue accurately. "
    "Also include SDH-style non-speech sound cues in ALL CAPS inside brackets as standalone tokens, "
    "with timing aligned to the audio, e.g. [♪ MUSIC ♪], [APPLAUSE], [LAUGHTER], [DOOR SLAMS], [ENGINE REVVING]. "
    "Do not paraphrase sound cues; keep them short and standard. "
)

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

def _is_bracket_token(t: str) -> bool:
    t = (t or "").strip()
    if not t:
        return False
    if BRACKET_TOKEN_RE.match(t):
        return True
    # treat musical note tokens as SDH as well
    if "♪" in t:
        return True
    return False

def _normalize_sdh_token(t: str) -> str:
    t = (t or "").strip()
    if not t:
        return ""
    # ensure bracketed
    if not (t.startswith("[") and t.endswith("]")):
        t = f"[{t}]"
    # normalize common spacing
    t = re.sub(r"\s+", " ", t)
    return t

# ---------------------------
# Wrapping / measuring
# ---------------------------

def wrap_text(text: str, max_chars: int, max_lines: int) -> str:
    """
    Greedy wrap within max_lines and max_chars per line.
    If too long, it'll spill into last line (QC will flag).
    """
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

    # hard clamp each line to max_chars visually (QC still flags)
    lines = [ln[:max_chars] for ln in lines[:max_lines]]
    return "\n".join(lines)

def _text_capacity(rules: Rules) -> int:
    # “approx capacity” ignoring newline — used for packing heuristics
    return rules.maxCharsPerLine * rules.maxLines

def _cps_ok(chars: int, duration_ms: int, max_cps: float) -> bool:
    if duration_ms <= 0:
        return True
    cps = chars / (duration_ms / 1000.0)
    return cps <= max_cps

def _cue_chars(text: str) -> int:
    return len(text.replace("\n", "").strip())

# ---------------------------
# SDH extraction from bracket tokens
# ---------------------------

def extract_sdh_events_and_strip(words: List[Word]) -> Tuple[List[Event], List[Word]]:
    """
    If AssemblyAI emits bracket tokens like [LAUGHTER], we:
    - convert them to Events with real timings
    - remove them from dialogue words so they don't pollute packing
    """
    events: List[Event] = []
    dialog: List[Word] = []

    i = 0
    n = len(words)
    while i < n:
        w = words[i]
        t = (w.text or "").strip()

        if _is_bracket_token(t):
            start = int(w.start)
            end = int(w.end)
            tokens = [_normalize_sdh_token(t)]

            j = i + 1
            # merge contiguous bracket tokens into one SDH cue
            while j < n:
                wt = (words[j].text or "").strip()
                if not _is_bracket_token(wt):
                    break
                tokens.append(_normalize_sdh_token(wt))
                end = int(words[j].end)
                j += 1

            # choose best label (first token) unless multiple meaningful tokens
            label = tokens[0]
            # If multiple different ones, keep the first to avoid spam
            events.append(Event(type="sfx", start=start, end=end, text=label))
            i = j
            continue

        dialog.append(w)
        i += 1

    return events, dialog

# ---------------------------
# Pro cue builder (pack-to-fill, no end-time extension)
# ---------------------------

def build_cues_from_words_pro(words: List[Word], rules: Rules) -> List[Dict[str, Any]]:
    """
    Goals:
    - Do NOT extend end times beyond AssemblyAI (no drift).
    - Pack text to fill lines reasonably (avoid 1-3 word flashes).
    - Respect max duration, max cps, max chars/line.
    - Merge short cues with neighbors when safe.
    """
    if not words:
        return []

    words = sorted(words, key=lambda w: (w.start, w.end))

    cues: List[Dict[str, Any]] = []
    cap = _text_capacity(rules)
    target_chars = int(cap * max(0.3, min(0.98, rules.targetFillRatio)))

    cur_words: List[Word] = []
    cur_speaker = _normalize_speaker(words[0].speaker)
    cur_start = int(words[0].start)
    cur_end = int(words[0].end)

    def cur_text_preview(ws: List[Word]) -> str:
        return " ".join((x.text or "").strip() for x in ws).strip()

    def can_add_word(ws: List[Word], w: Word, start_ms: int, end_ms: int) -> bool:
        text = cur_text_preview(ws + [w])
        wrapped = wrap_text(text, rules.maxCharsPerLine, rules.maxLines)
        # line checks
        lines = wrapped.split("\n")
        if len(lines) > rules.maxLines:
            return False
        if any(len(ln) > rules.maxCharsPerLine for ln in lines):
            return False

        duration = max(1, end_ms - start_ms)
        chars = _cue_chars(wrapped)
        if duration > rules.maxDurationMs:
            return False
        if not _cps_ok(chars, duration, rules.maxCPS):
            return False

        return True

    def flush():
        nonlocal cur_words, cur_speaker, cur_start, cur_end
        if not cur_words:
            return

        text = cur_text_preview(cur_words)
        if not text:
            cur_words = []
            return

        wrapped = wrap_text(text, rules.maxCharsPerLine, rules.maxLines)

        cues.append({
            "start": int(cur_start),
            "end": int(cur_end),          # CRITICAL: never extend beyond last word end
            "text": wrapped,
            "speaker": cur_speaker,
            "kind": "dialogue",
        })
        cur_words = []

    for w in words:
        sp = _normalize_speaker(w.speaker)
        w_start = int(w.start)
        w_end = int(w.end)

        # Speaker change: usually flush, but allow merge if it helps and is “tight”
        if sp != cur_speaker:
            flush()
            cur_speaker = sp
            cur_words = [w]
            cur_start = w_start
            cur_end = w_end
            continue

        # If currently empty, start
        if not cur_words:
            cur_words = [w]
            cur_speaker = sp
            cur_start = w_start
            cur_end = w_end
            continue

        # If there is a big gap, consider flushing (but don't over-split on tiny gaps)
        gap = w_start - cur_end
        if gap >= max(rules.minGapMs, 250):
            # only flush if we already have decent fill OR punctuation boundary OR long enough
            preview = wrap_text(cur_text_preview(cur_words), rules.maxCharsPerLine, rules.maxLines)
            filled = _cue_chars(preview)
            if filled >= target_chars or (rules.preferPunctuationBreaks and PUNCT_BREAK_RE.search((cur_words[-1].text or "").strip())):
                flush()
                cur_words = [w]
                cur_speaker = sp
                cur_start = w_start
                cur_end = w_end
                continue

        # Try to add word; if not possible, flush and start new cue
        proposed_end = max(cur_end, w_end)
        if can_add_word(cur_words, w, cur_start, proposed_end):
            cur_words.append(w)
            cur_end = proposed_end

            # If we've hit target fill and we end on punctuation, flush
            if rules.preferPunctuationBreaks and PUNCT_BREAK_RE.search((w.text or "").strip()):
                preview = wrap_text(cur_text_preview(cur_words), rules.maxCharsPerLine, rules.maxLines)
                if _cue_chars(preview) >= target_chars:
                    flush()
        else:
            flush()
            cur_words = [w]
            cur_speaker = sp
            cur_start = w_start
            cur_end = w_end

    flush()

    # Post-pass: merge too-short cues with next if safe (same speaker, small gap, constraints ok)
    merged: List[Dict[str, Any]] = []
    i = 0
    while i < len(cues):
        c = cues[i]
        duration = int(c["end"]) - int(c["start"])
        if duration >= rules.minDurationMs:
            merged.append(c)
            i += 1
            continue

        if not rules.allowVeryShortIfForced and duration < rules.minDurationMs:
            # keep but QC will flag
            merged.append(c)
            i += 1
            continue

        if i + 1 < len(cues):
            n = cues[i + 1]
            if n.get("kind") == "dialogue" and c.get("kind") == "dialogue" and n.get("speaker") == c.get("speaker"):
                gap = int(n["start"]) - int(c["end"])
                if gap <= max(rules.minGapMs, 250):
                    combined_text = (c["text"].replace("\n", " ") + " " + n["text"].replace("\n", " ")).strip()
                    wrapped = wrap_text(combined_text, rules.maxCharsPerLine, rules.maxLines)
                    new_start = int(c["start"])
                    new_end = int(n["end"])  # still from words; no extension beyond next cue end
                    chars = _cue_chars(wrapped)
                    dur = max(1, new_end - new_start)
                    if dur <= rules.maxDurationMs and _cps_ok(chars, dur, rules.maxCPS):
                        merged.append({
                            "start": new_start,
                            "end": new_end,
                            "text": wrapped,
                            "speaker": c["speaker"],
                            "kind": "dialogue",
                        })
                        i += 2
                        continue

        merged.append(c)
        i += 1

    return merged

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
# Exports: SRT/VTT/SCC/JSON
# ---------------------------

def to_srt(cues: List[Dict[str, Any]]) -> str:
    out = []
    i = 1
    for c in cues:
        out.append(str(i))
        out.append(f"{_ms_to_srt_time(int(c['start']))} --> {_ms_to_srt_time(int(c['end']))}")
        out.append(str(c.get("text") or ""))
        out.append("")
        i += 1
    return "\n".join(out).strip() + "\n"

def to_vtt(cues: List[Dict[str, Any]]) -> str:
    out = ["WEBVTT", ""]
    for c in cues:
        out.append(f"{_ms_to_vtt_time(int(c['start']))} --> {_ms_to_vtt_time(int(c['end']))}")
        out.append(str(c.get("text") or ""))
        out.append("")
    return "\n".join(out).strip() + "\n"

# VERY simplified SCC encoder (MVP, not spec-perfect EIA-608 packing)
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
    base_offset = 0 if rules.startAtHour00 else 0

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

        # SDH extraction (if bracket tokens exist)
        sdh_events, dialog_words = extract_sdh_events_and_strip(words)

        # Build dialogue cues (pro packer)
        cues = build_cues_from_words_pro(dialog_words, rules)

        # Add SDH events as cues (do NOT merge into dialogue)
        for e in sdh_events:
            cues.append({
                "start": int(e.start),
                "end": int(e.end),
                "text": _normalize_sdh_token(e.text or "[SFX]"),
                "speaker": "None",
                "kind": "sdh",
            })

    # If explicit events were passed in, add them too
    if req.events:
        for e in req.events:
            if e.type in ("music", "sfx"):
                txt = _normalize_sdh_token(e.text or "[♪ MUSIC ♪]")
                cues.append({"start": e.start, "end": e.end, "text": txt, "speaker": "None", "kind": "sdh"})
            elif e.type == "foreign_language":
                lang = e.language or "Unknown"
                cues.append({"start": e.start, "end": e.end, "text": f"[Speaking {lang}]", "speaker": "None", "kind": "sdh"})

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
    return {"ok": True, "service": "ai-cc-creator"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/v1/health")
def health_v1():
    return {"ok": True}

@app.post("/v1/format")
def format_endpoint(req: FormatRequest):
    return format_payload(req)

@app.post("/v1/jobs", response_model=JobStatus)
async def create_job(req: JobCreateRequest):
    if not ASSEMBLYAI_API_KEY:
        raise HTTPException(status_code=500, detail="ASSEMBLYAI_API_KEY is not set in Railway Variables.")
    if not WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="WEBHOOK_SECRET is not set in Railway Variables.")
    if not PUBLIC_BASE_URL:
        raise HTTPException(status_code=500, detail="PUBLIC_BASE_URL is not set in Railway Variables.")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = JobStatus(id=job_id, status="queued", assembly_id=None)
    JOB_RULES[job_id] = req.rules

    webhook_url = f"{PUBLIC_BASE_URL}/v1/webhooks/assemblyai"

    payload: Dict[str, Any] = {
        "audio_url": req.mediaUrl,                   # MP4 is OK as long as it's publicly accessible
        "speaker_labels": req.speaker_labels,
        "language_detection": req.language_detection,

        # AssemblyAI now expects speech_models as a non-empty list
        "speech_models": ["universal-3-pro"],

        "webhook_url": webhook_url,
        "webhook_auth_header_name": "X-Webhook-Token",
        "webhook_auth_header_value": WEBHOOK_SECRET,

        # Better readability from source:
        "punctuate": True,
        "format_text": True,
    }

    if req.speakers_expected is not None:
        payload["speakers_expected"] = int(req.speakers_expected)

    # SDH “Option A”: ask the model to emit bracket tokens we can time-align.
    if req.enable_sdh_prompt:
        payload["prompt"] = DEFAULT_SDH_PROMPT

    headers = {"Authorization": ASSEMBLYAI_API_KEY, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post("https://api.assemblyai.com/v2/transcript", json=payload, headers=headers)
        if r.status_code >= 300:
            JOBS[job_id].status = "error"
            JOBS[job_id].error = f"AssemblyAI submit failed: {r.status_code} {r.text}"
            return JOBS[job_id]

        data = r.json()
        assembly_id = data.get("id")
        JOBS[job_id].status = "processing"
        JOBS[job_id].assembly_id = assembly_id

    return JOBS[job_id]

@app.get("/v1/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str):
    js = JOBS.get(job_id)
    if not js:
        # IMPORTANT: prevent Base44 infinite spinner if Railway restarted (in-memory store cleared)
        return JobStatus(
            id=job_id,
            status="error",
            error="Job not found. The service may have restarted (in-memory jobs were cleared). Please create a new job.",
            assembly_id=None,
            result=None,
        )
    return js

@app.post("/v1/webhooks/assemblyai")
async def assemblyai_webhook(request: Request):
    token = request.headers.get("X-Webhook-Token", "")
    if not WEBHOOK_SECRET or token != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid webhook token")

    body = await request.json()
    assembly_id = body.get("transcript_id") or body.get("id")
    status = body.get("status")

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

    headers = {"Authorization": ASSEMBLYAI_API_KEY}
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(f"https://api.assemblyai.com/v2/transcript/{assembly_id}", headers=headers)
        if r.status_code >= 300:
            JOBS[job_id].status = "error"
            JOBS[job_id].error = f"AssemblyAI get transcript failed: {r.status_code} {r.text}"
            return JSONResponse({"ok": True})
        t = r.json()

    # Prefer utterances (speaker diarization segments) if present.
    # Fallback to words only.
    words_raw = t.get("words") or []
    utterances = t.get("utterances") or []

    words: List[Word] = []

    if utterances:
        # Many responses include words per utterance; if not, we still can use global words list.
        for u in utterances:
            sp = _normalize_speaker(u.get("speaker"))
            u_words = u.get("words")
            if isinstance(u_words, list) and u_words:
                for w in u_words:
                    words.append(Word(
                        text=w.get("text", ""),
                        start=int(w.get("start", 0)),
                        end=int(w.get("end", 0)),
                        speaker=sp,
                    ))
            else:
                # fallback: just use global words, will still include speakers if provided
                break

    if not words:
        for w in words_raw:
            words.append(Word(
                text=w.get("text", ""),
                start=int(w.get("start", 0)),
                end=int(w.get("end", 0)),
                speaker=_normalize_speaker(w.get("speaker") or "A"),
            ))

    rules = JOB_RULES.get(job_id, Rules())

    result = format_payload(FormatRequest(words=words, events=[], rules=rules))

    JOBS[job_id].status = "done"
    JOBS[job_id].result = result
    JOBS[job_id].error = None

    return JSONResponse({"ok": True})
