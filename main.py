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
    title="AI CC Creator — Rules Engine + AssemblyAI Orchestrator",
    version="2.0.0",
)

# =============================================================================
# Environment
# =============================================================================

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

# CORS
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

# =============================================================================
# Models
# =============================================================================


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
    type: Literal["music", "foreign_language", "sound_effect"]
    start: int
    end: int
    text: Optional[str] = None
    language: Optional[str] = None


class FormatRequest(BaseModel):
    rules: Rules = Field(default_factory=Rules)

    # You can send EITHER:
    # - words (recommended)
    # - cues (if you already grouped them)
    words: Optional[List[Word]] = None
    cues: Optional[List[Dict[str, Any]]] = None  # {start,end,text,speaker?}

    # optional events layer
    events: Optional[List[Event]] = None


class JobCreateRequest(BaseModel):
    mediaUrl: str
    rules: Rules = Field(default_factory=Rules)

    # AssemblyAI toggles
    speaker_labels: bool = True
    language_detection: bool = True


class JobStatus(BaseModel):
    id: str
    status: Literal["queued", "processing", "done", "error"]
    error: Optional[str] = None
    assembly_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


# =============================================================================
# In-memory job store (MVP)
# =============================================================================

JOBS: Dict[str, JobStatus] = {}
JOB_RULES: Dict[str, Rules] = {}

# =============================================================================
# Constants / Helpers
# =============================================================================

PUNCT_BREAK_RE = re.compile(r"[.!?…]+$")

# SDH sound cues dictionary (expand anytime)
SOUND_CUE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(applause|clapping)\b", re.I), "[APPLAUSE]"),
    (re.compile(r"\b(laughter|laughing)\b", re.I), "[LAUGHTER]"),
    (re.compile(r"\b(music)\b", re.I), "[♪ MUSIC ♪]"),
    (re.compile(r"\b(door slam|door slams)\b", re.I), "[DOOR SLAMS]"),
    (re.compile(r"\b(gunshot|gunshots)\b", re.I), "[GUNSHOTS]"),
    (re.compile(r"\b(siren|sirens)\b", re.I), "[SIRENS]"),
    (re.compile(r"\b(explosion|explosions)\b", re.I), "[EXPLOSION]"),
]


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
    """
    Removes tokens like [Music], [Speaking Spanish], etc from "words" stream.
    This prevents bracket tokens from being treated as spoken dialog.
    """
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
    return any(e.type in ("music", "foreign_language", "sound_effect") for e in events)


# =============================================================================
# Wrapping / Layout
# =============================================================================

def _wrap_two_lines_best_effort(text: str, max_chars: int) -> str:
    """
    Returns a 1-2 line wrapped string trying to fill lines toward max_chars.
    """
    words = text.split()
    if not words:
        return ""

    # If it already fits on one line, keep it
    if len(text) <= max_chars:
        return text.strip()

    # Try to split into 2 lines as balanced as possible
    best = None
    best_score = None

    for i in range(1, len(words)):
        l1 = " ".join(words[:i]).strip()
        l2 = " ".join(words[i:]).strip()
        if not l1 or not l2:
            continue

        if len(l1) > max_chars or len(l2) > max_chars:
            continue

        # Score: prefer both lines filled close to max_chars
        score = abs(max_chars - len(l1)) + abs(max_chars - len(l2))
        if best is None or score < best_score:
            best = (l1, l2)
            best_score = score

    if best:
        return best[0] + "\n" + best[1]

    # If we couldn't fit into 2 clean lines, do greedy
    lines: List[str] = []
    cur = ""
    for w in words:
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_chars:
            cur = cur + " " + w
        else:
            lines.append(cur)
            cur = w
            if len(lines) >= 2:
                # Stuff the rest into line 2 (will QC flag)
                lines[-1] = (lines[-1] + " " + cur).strip()
                cur = ""

    if cur:
        lines.append(cur)

    if len(lines) == 1:
        return lines[0]
    return "\n".join(lines[:2])


# =============================================================================
# Cue building (timing-safe, broadcast-oriented)
# =============================================================================

def build_cues_from_words(words: List[Word], rules: Rules) -> List[Dict[str, Any]]:
    """
    This is the key fix:
    - We build cues that fill 32 chars where possible
    - We NEVER extend OUT time beyond AssemblyAI (prevents lag)
    - We avoid 1-3 word cues unless timing forces it
    - Speaker changes inside a cue are represented with NBCU dash format
    """
    if not words:
        return []

    words = sorted(words, key=lambda w: (w.start, w.end))

    cues: List[Dict[str, Any]] = []

    # Current cue buffer
    cur_words: List[Word] = []
    cur_start = words[0].start
    cur_end = words[0].end
    cur_speaker = _normalize_speaker(words[0].speaker)

    # For dash-style speaker changes inside a cue
    cur_segments: List[Tuple[str, List[str]]] = [(cur_speaker, [])]

    def flush():
        nonlocal cur_words, cur_start, cur_end, cur_speaker, cur_segments

        if not cur_words:
            return

        # Build text with NBCU dash formatting if speaker changes occurred
        # segments = [(speaker, ["word", "word"...]), ...]
        seg_lines: List[str] = []
        for sp, toks in cur_segments:
            if not toks:
                continue
            line = " ".join(toks).strip()
            if line:
                seg_lines.append(f"- {line}")

        if not seg_lines:
            cur_words = []
            cur_segments = [(cur_speaker, [])]
            return

        # Now wrap the segment output into 1-2 lines max if possible.
        # If there are 2 speakers, we WANT 2 lines:
        # - speaker1...
        # - speaker2...
        if len(seg_lines) == 1:
            wrapped = _wrap_two_lines_best_effort(seg_lines[0].replace("- ", ""), rules.maxCharsPerLine)
            # If it wrapped, it becomes multi-line, but no dash needed in continuation
            wrapped_lines = wrapped.split("\n")
            if len(wrapped_lines) == 1:
                final_text = wrapped_lines[0]
            else:
                final_text = wrapped_lines[0] + "\n" + wrapped_lines[1]
        else:
            # We have at least 2 speaker segments
            # Hard cap to 2 lines:
            final_text = seg_lines[0]
            if len(seg_lines) >= 2:
                final_text += "\n" + seg_lines[1]

        # IMPORTANT:
        # We DO NOT extend end time. Ever.
        # This prevents delayed captions.
        cues.append({
            "start": int(cur_start),
            "end": int(cur_end),
            "text": final_text.strip(),
            "speaker": None,  # we do not expose speaker IDs
        })

        # reset
        cur_words = []
        cur_segments = [(cur_speaker, [])]

    # Heuristic targets: try to fill line space
    TARGET_CHARS = int(rules.maxCharsPerLine * rules.maxLines * 0.85)

    for w in words:
        sp = _normalize_speaker(w.speaker)
        txt = (w.text or "").strip()
        if not txt:
            continue

        # Determine if we should start a new cue
        gap = w.start - cur_end

        # If speaker changes, we allow it INSIDE cue via dash format,
        # unless we are already at 2 lines full.
        speaker_changed = sp != cur_speaker

        # Proposed new end (always real word end)
        proposed_end = max(cur_end, w.end)
        proposed_duration = proposed_end - cur_start

        # Estimate text length if we add this word
        flat_words = [cw.text.strip() for cw in cur_words] + [txt]
        flat_text = " ".join(flat_words).strip()

        # Split rules:
        must_split = False

        # 1) gap split (hard)
        if gap >= rules.minGapMs:
            must_split = True

        # 2) max duration split
        if proposed_duration > rules.maxDurationMs:
            must_split = True

        # 3) punctuation break encouragement (soft)
        punct_break = rules.preferPunctuationBreaks and PUNCT_BREAK_RE.search(txt)

        # 4) If we already have a lot of text, split
        if len(flat_text) >= TARGET_CHARS:
            must_split = True

        if must_split:
            flush()
            cur_start = w.start
            cur_end = w.end
            cur_speaker = sp
            cur_words = [w]
            cur_segments = [(sp, [txt])]
            continue

        # Add word to cue
        cur_words.append(w)
        cur_end = max(cur_end, w.end)

        if speaker_changed:
            cur_speaker = sp
            cur_segments.append((sp, []))

        cur_segments[-1][1].append(txt)

        # Soft punctuation split: flush if we already have decent duration and enough text
        if punct_break:
            if (cur_end - cur_start) >= rules.minDurationMs and len(flat_text) >= int(rules.maxCharsPerLine * 0.6):
                flush()
                # next cue starts naturally

    flush()
    return cues


# =============================================================================
# Hard Enforcement Layer
# =============================================================================

def _split_cue_text_to_fit(text: str, rules: Rules) -> List[str]:
    """
    Returns 1+ caption text blocks that fit maxLines/maxCharsPerLine.
    """
    # If it already fits, return as-is
    lines = text.split("\n")
    if len(lines) <= rules.maxLines and all(len(ln) <= rules.maxCharsPerLine for ln in lines):
        return [text]

    # Remove speaker dash prefix for wrapping decisions
    dash_mode = False
    if text.strip().startswith("- "):
        dash_mode = True

    clean = text.replace("\n", " ").strip()
    tokens = clean.split()
    if not tokens:
        return [text]

    blocks: List[str] = []
    cur_tokens: List[str] = []

    def flush_tokens():
        nonlocal cur_tokens
        if not cur_tokens:
            return
        t = " ".join(cur_tokens).strip()
        wrapped = _wrap_two_lines_best_effort(t, rules.maxCharsPerLine)
        blocks.append(wrapped)
        cur_tokens = []

    for tok in tokens:
        test = (" ".join(cur_tokens + [tok])).strip()
        wrapped = _wrap_two_lines_best_effort(test, rules.maxCharsPerLine)
        wlines = wrapped.split("\n")
        if len(wlines) <= rules.maxLines and all(len(ln) <= rules.maxCharsPerLine for ln in wlines):
            cur_tokens.append(tok)
        else:
            flush_tokens()
            cur_tokens.append(tok)

    flush_tokens()

    # Re-apply dash mode if it was a dash cue
    if dash_mode:
        fixed = []
        for b in blocks:
            bl = b.split("\n")
            if len(bl) == 1:
                fixed.append("- " + bl[0])
            else:
                fixed.append("- " + bl[0] + "\n- " + bl[1])
        return fixed

    return blocks


def enforce_rules_hard(cues: List[Dict[str, Any]], rules: Rules) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Hard enforcement:
    1) Try re-wrap (already done)
    2) If still violates -> split text into smaller chunks
    3) If still violates -> QC issue, but NEVER change timing

    NOTE: Splitting text DOES NOT create new time ranges.
    We keep the same start/end and split into sub-cues inside the same time.
    This is not ideal for broadcast, but it avoids lag and preserves sync.
    """
    qc_extra: List[Dict[str, Any]] = []
    out: List[Dict[str, Any]] = []

    for idx, c in enumerate(cues, start=1):
        start = int(c["start"])
        end = int(c["end"])
        text = str(c.get("text") or "")

        # If it fits, keep it
        lines = text.split("\n")
        ok_lines = len(lines) <= rules.maxLines and all(len(ln) <= rules.maxCharsPerLine for ln in lines)

        duration = max(1, end - start)
        chars = len(text.replace("\n", ""))
        cps = (chars / (duration / 1000.0)) if duration > 0 else 9999

        if ok_lines and cps <= rules.maxCPS:
            out.append(c)
            continue

        # Attempt split into multiple blocks
        blocks = _split_cue_text_to_fit(text, rules)

        # If splitting created >1 blocks, we emit them as sequential "sub cues"
        # but we DO NOT change end time beyond original.
        if len(blocks) > 1:
            # Divide time evenly among blocks
            total = len(blocks)
            block_dur = max(1, (end - start) // total)

            for i, b in enumerate(blocks):
                b_start = start + (i * block_dur)
                b_end = start + ((i + 1) * block_dur) if i < total - 1 else end

                out.append({
                    "start": int(b_start),
                    "end": int(b_end),
                    "text": b.strip(),
                    "speaker": None,
                })

            qc_extra.append({
                "cue": idx,
                "type": "hard_split_applied",
                "value": total,
            })
            continue

        # If still bad after split, keep original but flag QC
        out.append(c)

        if not ok_lines:
            qc_extra.append({"cue": idx, "type": "hard_enforce_failed_lines", "value": len(lines)})
        if cps > rules.maxCPS:
            qc_extra.append({"cue": idx, "type": "hard_enforce_failed_cps", "value": round(cps, 2)})

    return out, qc_extra


# =============================================================================
# QC
# =============================================================================

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


# =============================================================================
# Exports
# =============================================================================

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


# VERY simplified SCC encoder (MVP)
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


# =============================================================================
# SDH / Sound cue extraction (basic but useful)
# =============================================================================

def extract_sdh_events_from_transcript_text(text: str, words: List[Word]) -> List[Event]:
    """
    AssemblyAI itself may expose sound events in the UI,
    but they are NOT returned as structured events in the transcript API.
    So we do a lightweight heuristic here.

    This is NOT perfect — but it is better than returning nothing.
    """
    events: List[Event] = []
    if not text:
        return events
    if not words:
        return events

    # Look for keywords in the full transcript text
    # and inject a generic cue near the start of the transcript.
    # (In v3 you can do true per-time detection.)
    for pat, label in SOUND_CUE_PATTERNS:
        if pat.search(text):
            # Put it at the beginning as a placeholder
            events.append(Event(type="sound_effect", start=words[0].start, end=min(words[0].start + 1500, words[-1].end), text=label))
    return events


# =============================================================================
# Core formatter
# =============================================================================

def format_payload(req: FormatRequest) -> Dict[str, Any]:
    rules = req.rules

    # If cues already provided, trust them
    if req.cues is not None:
        cues = req.cues
    else:
        if not req.words:
            raise HTTPException(status_code=422, detail="You must provide either 'words' or 'cues'.")

        words = [Word(**w.model_dump()) for w in req.words]

        # PRO GUARD:
        if _has_special_events(req.events):
            words = _strip_bracket_tokens(words)

        cues = build_cues_from_words(words, rules)

    # Inject events as cues
    if req.events:
        for e in req.events:
            if e.type == "music":
                txt = e.text or "[♪ MUSIC ♪]"
                cues.append({"start": e.start, "end": e.end, "text": txt, "speaker": None})
            elif e.type == "foreign_language":
                lang = e.language or "Unknown"
                cues.append({"start": e.start, "end": e.end, "text": f"[Speaking {lang}]", "speaker": None})
            elif e.type == "sound_effect":
                txt = e.text or "[SFX]"
                cues.append({"start": e.start, "end": e.end, "text": txt, "speaker": None})

        cues = sorted(cues, key=lambda c: (int(c["start"]), int(c["end"])))

    # Hard enforcement pass
    cues_enforced, qc_extra = enforce_rules_hard(cues, rules)

    qc = qc_cues(cues_enforced, rules)
    if qc_extra:
        qc["issues"].extend(qc_extra)
        qc["issuesCount"] = len(qc["issues"])

    return {
        "rules": rules.model_dump(),
        "cues": cues_enforced,
        "srt": to_srt(cues_enforced),
        "vtt": to_vtt(cues_enforced),
        "scc": to_scc(cues_enforced, rules),
        "qc": qc,
    }


# =============================================================================
# Routes
# =============================================================================

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
    if not WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="WEBHOOK_SECRET is not set in Railway Variables.")
    if not PUBLIC_BASE_URL:
        raise HTTPException(status_code=500, detail="PUBLIC_BASE_URL is not set in Railway Variables.")

    job_id = str(uuid.uuid4())
    JOBS[job_id] = JobStatus(id=job_id, status="queued", assembly_id=None)
    JOB_RULES[job_id] = req.rules

    webhook_url = f"{PUBLIC_BASE_URL}/v1/webhooks/assemblyai"

    payload = {
        "audio_url": req.mediaUrl,

        # diarization + language
        "speaker_labels": req.speaker_labels,
        "language_detection": req.language_detection,

        # REQUIRED by AssemblyAI now
        "speech_models": ["universal-3-pro"],

        # webhook
        "webhook_url": webhook_url,
        "webhook_auth_header_name": "X-Webhook-Token",
        "webhook_auth_header_value": WEBHOOK_SECRET,
    }

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
        raise HTTPException(status_code=404, detail="Job not found")
    return js

@app.post("/v1/webhooks/assemblyai")
async def assemblyai_webhook(request: Request):
    # Verify webhook secret
    token = request.headers.get("X-Webhook-Token", "")
    if not WEBHOOK_SECRET or token != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Invalid webhook token")

    body = await request.json()
    assembly_id = body.get("transcript_id") or body.get("id")
    status = body.get("status")

    # Find our job by assembly_id
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

    # Completed: fetch transcript details including words
    headers = {"Authorization": ASSEMBLYAI_API_KEY}
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(f"https://api.assemblyai.com/v2/transcript/{assembly_id}", headers=headers)
        if r.status_code >= 300:
            JOBS[job_id].status = "error"
            JOBS[job_id].error = f"AssemblyAI get transcript failed: {r.status_code} {r.text}"
            return JSONResponse({"ok": True})
        t = r.json()

    # Extract words
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

    # SDH events (basic heuristic)
    transcript_text = t.get("text") or ""
    events: List[Event] = extract_sdh_events_from_transcript_text(transcript_text, words)

    rules = JOB_RULES.get(job_id, Rules())

    result = format_payload(FormatRequest(words=words, events=events, rules=rules))

    JOBS[job_id].status = "done"
    JOBS[job_id].result = result
    JOBS[job_id].error = None

    return JSONResponse({"ok": True})
