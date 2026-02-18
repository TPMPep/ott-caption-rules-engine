import os
import re
from typing import Any, Dict, List, Optional, Literal, Tuple

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

# CORS: allow Base44 preview/prod to call Railway
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
    # NBCU-ish defaults (you can tune)
    maxCharsPerLine: int = 32
    maxLines: int = 2
    maxCPS: float = 17.0
    minDurationMs: int = 1000          # do NOT force timing expansion; QC only
    maxDurationMs: int = 7000
    minGapMs: int = 80                 # used to detect real breaks
    preferPunctuationBreaks: bool = True

    # “don’t make micro-cues” preference (merging heuristic)
    avoidVeryShortMs: int = 700         # try hard not to emit cues shorter than this

    # Two-speaker “dash” combo behavior (Option A)
    enableDashTwoSpeaker: bool = True
    dashMergeMaxGapMs: int = 250        # if speaker flips within this gap, allow "- line\n- line"
    dashMergeMaxTotalMs: int = 4000

    # SCC specifics (still MVP encoder below)
    sccFrameRate: float = 29.97
    startAtHour00: bool = True


class Word(BaseModel):
    text: str
    start: int
    end: int
    speaker: Optional[str] = "A"


class Event(BaseModel):
    type: Literal["music", "sfx", "foreign_language"]
    start: int
    end: int
    text: str


class FormatRequest(BaseModel):
    rules: Rules = Field(default_factory=Rules)
    words: Optional[List[Word]] = None
    cues: Optional[List[Dict[str, Any]]] = None  # {start,end,text,speaker}
    events: Optional[List[Event]] = None


class JobCreateRequest(BaseModel):
    mediaUrl: str
    rules: Rules = Field(default_factory=Rules)

    speaker_labels: bool = True
    language_detection: bool = True

    # Option A: ask AssemblyAI to include SDH bracket tokens in transcript
    enable_sdh_prompt: bool = True


class JobStatus(BaseModel):
    id: str
    status: Literal["queued", "processing", "done", "error"]
    error: Optional[str] = None
    assembly_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


# ---------------------------
# Utilities
# ---------------------------

PUNCT_END_RE = re.compile(r"[.!?…]+$")
BRACKET_TOKEN_RE = re.compile(r"^\[(.+)\]$")  # [APPLAUSE], [♪ MUSIC ♪], [DOOR SLAMS], etc.

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

def _flatten_text(lines: List[str]) -> str:
    return "\n".join([ln.strip() for ln in lines if ln.strip()]).strip()

def wrap_text_greedy_maxfill(text: str, max_chars: int, max_lines: int) -> str:
    """
    Greedy word wrap with a “fill lines as much as possible” bias.
    Always returns <= max_lines (by forcing overflow into last line).
    """
    words = [w for w in text.split() if w]
    if not words:
        return ""

    lines: List[str] = [""]
    for w in words:
        cur = lines[-1]
        if not cur:
            lines[-1] = w
            continue

        if len(cur) + 1 + len(w) <= max_chars:
            lines[-1] = cur + " " + w
        else:
            # need new line
            if len(lines) < max_lines:
                lines.append(w)
            else:
                # force into last line even if it exceeds; QC will flag
                lines[-1] = lines[-1] + " " + w

    # if we somehow have more, merge
    if len(lines) > max_lines:
        head = lines[:max_lines - 1]
        tail = " ".join(lines[max_lines - 1:])
        lines = head + [tail]

    return _flatten_text(lines)

def _char_count_for_cps(text: str) -> int:
    return len(text.replace("\n", "").replace("\r", ""))

def _would_exceed_limits(
    start_ms: int,
    end_ms: int,
    text: str,
    rules: Rules,
) -> bool:
    dur = max(1, end_ms - start_ms)
    chars = _char_count_for_cps(text)
    cps = chars / (dur / 1000.0)

    # hard-ish constraints
    if dur > rules.maxDurationMs:
        return True
    if cps > (rules.maxCPS * 1.10):  # small tolerance while building; QC still flags
        return True
    return False

def _extract_sdh_events_from_words(words: List[Word]) -> Tuple[List[Word], List[Event]]:
    """
    Option A SDH: if AssemblyAI returns bracket tokens as individual "words",
    pull them out as Events and remove from dialogue stream.
    """
    cleaned: List[Word] = []
    events: List[Event] = []

    for w in words:
        t = (w.text or "").strip()
        m = BRACKET_TOKEN_RE.match(t)
        if m:
            label = m.group(1).strip()

            # Normalize common music style
            normalized = f"[{label.upper()}]"
            if "MUSIC" in normalized and "♪" not in normalized:
                normalized = "[♪ MUSIC ♪]"

            # Heuristic classify
            typ: Literal["music", "sfx", "foreign_language"] = "sfx"
            if "MUSIC" in normalized:
                typ = "music"
            if "SPEAKING" in normalized or "SPANISH" in normalized or "FRENCH" in normalized:
                typ = "foreign_language"

            events.append(Event(type=typ, start=w.start, end=w.end, text=normalized))
            continue

        cleaned.append(w)

    return cleaned, events

# ---------------------------
# Cue building (broadcast-safe)
# - NEVER expand OUT times beyond last spoken word
# - Max-fill lines up to 32 chars x 2 lines
# - Split only when needed (speaker change, real gap, limits)
# ---------------------------

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

        raw_text = " ".join((w.text or "").strip() for w in cur_words).strip()
        if not raw_text:
            cur_words = []
            return

        wrapped = wrap_text_greedy_maxfill(raw_text, rules.maxCharsPerLine, rules.maxLines)

        # CRITICAL: do NOT extend end time. Preserve AssemblyAI word OUT.
        cues.append({
            "start": int(cur_start),
            "end": int(cur_end),
            "text": wrapped,
            "speaker": cur_speaker,
        })

        cur_words = []

    for w in words:
        sp = _normalize_speaker(w.speaker)

        # Speaker change: usually split (we may dash-merge later)
        if sp != cur_speaker:
            flush()
            cur_words = [w]
            cur_start = w.start
            cur_end = w.end
            cur_speaker = sp
            continue

        gap = w.start - cur_end

        # Real gap split (someone stopped talking)
        if gap >= max(rules.minGapMs, 250):
            flush()
            cur_words = [w]
            cur_start = w.start
            cur_end = w.end
            cur_speaker = sp
            continue

        # Try to add word and see if we'd violate limits
        candidate_words = cur_words + [w]
        raw_text = " ".join((x.text or "").strip() for x in candidate_words).strip()
        wrapped = wrap_text_greedy_maxfill(raw_text, rules.maxCharsPerLine, rules.maxLines)
        candidate_end = max(cur_end, w.end)

        if _would_exceed_limits(cur_start, candidate_end, wrapped, rules):
            # If we would exceed, flush current cue and start new with this word
            flush()
            cur_words = [w]
            cur_start = w.start
            cur_end = w.end
            cur_speaker = sp
            continue

        # Otherwise accept
        cur_words = candidate_words
        cur_end = candidate_end

        # Punctuation is ONLY a soft preference:
        # flush only if we're already in a “good” duration window and close to limits.
        if rules.preferPunctuationBreaks and PUNCT_END_RE.search((w.text or "").strip()):
            dur = cur_end - cur_start
            if dur >= rules.minDurationMs:
                # “close to full” heuristic
                if len(wrapped.split("\n")) == rules.maxLines or len(wrapped) >= int(rules.maxCharsPerLine * rules.maxLines * 0.85):
                    flush()

    flush()
    return cues

def dash_merge_two_speakers(cues: List[Dict[str, Any]], rules: Rules) -> List[Dict[str, Any]]:
    """
    Option A: If two speakers fire back-to-back quickly, represent as:
      - First speaker line.
      - Second speaker line.
    Must remain max 2 lines.
    """
    if not rules.enableDashTwoSpeaker or not cues:
        return cues

    merged: List[Dict[str, Any]] = []
    i = 0
    while i < len(cues):
        a = cues[i]
        if i + 1 >= len(cues):
            merged.append(a)
            break

        b = cues[i + 1]
        gap = int(b["start"]) - int(a["end"])
        if gap < 0:
            gap = 0

        # eligible only if different speakers and tight timing
        if a.get("speaker") != b.get("speaker") and gap <= rules.dashMergeMaxGapMs:
            total_start = int(a["start"])
            total_end = int(b["end"])
            total_dur = total_end - total_start

            if total_dur <= rules.dashMergeMaxTotalMs:
                # Build two-line dash cue
                a_line = str(a.get("text") or "").replace("\n", " ").strip()
                b_line = str(b.get("text") or "").replace("\n", " ").strip()

                dash_text = _flatten_text([f"- {a_line}", f"- {b_line}"])
                # ensure <=2 lines and each line <= maxCharsPerLine where possible:
                # (we don't hard-truncate; QC flags)
                merged.append({
                    "start": total_start,
                    "end": total_end,
                    "text": dash_text,
                    "speaker": "A/B",
                })
                i += 2
                continue

        merged.append(a)
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

        # IMPORTANT: minDuration is QC only (do NOT shift timing)
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

# SCC: still simplified (your UI can download SCC for now; true 608 packing is a later step)
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

    # Use provided cues if already grouped
    if req.cues is not None:
        cues = req.cues
        events = req.events or []
    else:
        if not req.words:
            raise HTTPException(status_code=422, detail="You must provide either 'words' or 'cues'.")

        words = [Word(**w.model_dump()) for w in req.words]

        # Pull SDH bracket tokens into events (Option A)
        words, extracted_events = _extract_sdh_events_from_words(words)
        events = (req.events or []) + extracted_events

        cues = build_cues_from_words(words, rules)
        cues = dash_merge_two_speakers(cues, rules)

    # Inject events as their own cues (do NOT join into dialogue)
    if events:
        for e in events:
            cues.append({
                "start": int(e.start),
                "end": int(e.end),
                "text": e.text,
                "speaker": "SDH",
                "kind": e.type,
            })
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
    return {"ok": True, "service": "caption-engine"}

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
    if not WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="WEBHOOK_SECRET is not set in Railway Variables.")
    if not PUBLIC_BASE_URL:
        raise HTTPException(status_code=500, detail="PUBLIC_BASE_URL is not set in Railway Variables.")

    webhook_url = f"{PUBLIC_BASE_URL}/v1/webhooks/assemblyai"

    # Option A SDH: ask AssemblyAI to emit bracket tokens as standalone items
    prompt = None
    if req.enable_sdh_prompt:
        prompt = (
            "Transcribe dialogue accurately. Also include SDH-style non-speech sound cues in ALL CAPS inside brackets "
            "as standalone tokens with timing aligned to the audio, e.g., [♪ MUSIC ♪], [APPLAUSE], [LAUGHTER], [DOOR SLAMS], "
            "[ENGINE REVVING]. Do not paraphrase sound cues; keep them short and standard."
        )

    payload: Dict[str, Any] = {
        "audio_url": req.mediaUrl,
        "speaker_labels": req.speaker_labels,
        "language_detection": req.language_detection,

        # AssemblyAI expects speech_models as a non-empty list
        "speech_models": ["universal-3-pro"],

        "webhook_url": webhook_url,
        "webhook_auth_header_name": "X-Webhook-Token",
        "webhook_auth_header_value": WEBHOOK_SECRET,
    }
    if prompt:
        payload["prompt"] = prompt

    headers = {"Authorization": ASSEMBLYAI_API_KEY, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post("https://api.assemblyai.com/v2/transcript", json=payload, headers=headers)
        if r.status_code >= 300:
            return JobStatus(
                id="",
                status="error",
                error=f"AssemblyAI submit failed: {r.status_code} {r.text}",
                assembly_id=None,
                result=None,
            )
        data = r.json()

    assembly_id = data.get("id")
    if not assembly_id:
        return JobStatus(
            id="",
            status="error",
            error="AssemblyAI submit succeeded but did not return an id.",
            assembly_id=None,
            result=None,
        )

    # CRITICAL: job id IS the assembly transcript id (survives Railway restarts)
    return JobStatus(id=assembly_id, status="processing", error=None, assembly_id=assembly_id, result=None)

@app.get("/v1/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str):
    """
    job_id is the AssemblyAI transcript id.
    This avoids “Processing forever” caused by Railway restarts wiping in-memory state.
    """
    if not ASSEMBLYAI_API_KEY:
        raise HTTPException(status_code=500, detail="ASSEMBLYAI_API_KEY is not set in Railway Variables.")

    headers = {"Authorization": ASSEMBLYAI_API_KEY}
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(f"https://api.assemblyai.com/v2/transcript/{job_id}", headers=headers)
        if r.status_code >= 300:
            raise HTTPException(status_code=502, detail=f"AssemblyAI get transcript failed: {r.status_code} {r.text}")
        t = r.json()

    status = t.get("status")
    if status in ("error", "failed"):
        return JobStatus(
            id=job_id,
            status="error",
            error=t.get("error") or "AssemblyAI transcript failed",
            assembly_id=job_id,
            result=None,
        )

    if status != "completed":
        # still processing/queued
        return JobStatus(
            id=job_id,
            status="processing",
            error=None,
            assembly_id=job_id,
            result=None,
        )

    # Completed: build our broadcast-safe cues from word timings
    words_raw = t.get("words") or []
    words: List[Word] = []
    for w in words_raw:
        words.append(Word(
            text=w.get("text", ""),
            start=int(w.get("start", 0)),
            end=int(w.get("end", 0)),
            speaker=_normalize_speaker(w.get("speaker") or "A"),
        ))

    # NOTE: rules are not persisted across restarts yet.
    # Use defaults for now; later we’ll persist per-job rules in DB.
    rules = Rules()

    result = format_payload(FormatRequest(words=words, events=[], rules=rules))

    return JobStatus(
        id=job_id,
        status="done",
        error=None,
        assembly_id=job_id,
        result=result,
    )

@app.post("/v1/webhooks/assemblyai")
async def assemblyai_webhook(request: Request):
    """
    We keep this route for completeness (and for future DB persistence / push updates),
    but the app no longer depends on webhooks to complete jobs.
    Base44 can simply poll /v1/jobs/{assembly_id}.
    """
    token = request.headers.get("X-Webhook-Token", "")
    if not WEBHOOK_SECRET or token != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid webhook token")

    return JSONResponse({"ok": True})
