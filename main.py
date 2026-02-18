import os
import re
import json
import uuid
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Literal, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request, Path
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="AI CC Creator – Broadcast Rules Engine + AssemblyAI Orchestrator", version="2.0.0")

# =============================================================================
# Environment
# =============================================================================

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "").strip()  # optional; still supported
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip().rstrip("/")  # optional; still supported
ALLOWED_ORIGINS_RAW = os.getenv("ALLOWED_ORIGINS", "").strip()
SQLITE_PATH = os.getenv("SQLITE_PATH", "/tmp/ai_cc_creator.db").strip()

ASSEMBLYAI_TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"

# If PUBLIC_BASE_URL not set, webhook still works for local dev; polling will still work even without webhook
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
# Persistence (SQLite)
# =============================================================================

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            rules_json TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn

def _db_put_job(job_id: str, rules_json: str) -> None:
    conn = _db()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO jobs (id, created_at, rules_json) VALUES (?, ?, ?)",
            (job_id, datetime.now(timezone.utc).isoformat(), rules_json),
        )
        conn.commit()
    finally:
        conn.close()

def _db_get_rules(job_id: str) -> Optional[Dict[str, Any]]:
    conn = _db()
    try:
        cur = conn.execute("SELECT rules_json FROM jobs WHERE id = ?", (job_id,))
        row = cur.fetchone()
        if not row:
            return None
        return json.loads(row[0])
    finally:
        conn.close()

# =============================================================================
# Models
# =============================================================================

class Rules(BaseModel):
    # NBCU-ish defaults (tweakable)
    maxCharsPerLine: int = 32
    maxLines: int = 2
    maxCPS: float = 17.0

    # Timing constraints
    minDurationMs: int = 900         # do NOT enforce by extending OUT time; used for merge preference + QC
    maxDurationMs: int = 6500
    hardMaxDurationMs: int = 8000    # absolute cap; if exceeded we must split

    # Gap logic
    gapSplitMs: int = 450            # pauses >= this cause a split
    mergeGapMs: int = 250            # small gaps <= this can be merged

    preferPunctuationBreaks: bool = True

    # Speaker formatting
    speakerStyle: Literal["labels", "dash", "none"] = "labels"
    # "labels" => [A] prefix (or [SPEAKER A] if you want later)
    # "dash"   => "- line" style for 2 speakers in one cue (Option A)
    # "none"   => no speaker prefix

    # SDH / sound cues
    sdhBracketsUppercase: bool = True

    # SCC export
    sccFrameRate: float = 29.97
    startAtHour00: bool = True


class Word(BaseModel):
    text: str
    start: int
    end: int
    speaker: Optional[str] = None


class Event(BaseModel):
    type: Literal["sound", "music", "foreign_language"]
    start: int
    end: int
    text: str


class JobCreateRequest(BaseModel):
    mediaUrl: str
    rules: Rules = Field(default_factory=Rules)

    # AssemblyAI switches
    speaker_labels: bool = True
    language_detection: bool = True


class JobStatus(BaseModel):
    id: str
    status: Literal["queued", "processing", "done", "error"]
    error: Optional[str] = None
    assembly_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


# =============================================================================
# AssemblyAI Prompt (Option A for SDH cues)
# =============================================================================

SDH_PROMPT = (
    "Transcribe dialogue accurately. "
    "Also include SDH-style non-speech sound cues in ALL CAPS inside square brackets as standalone tokens, "
    "with timing aligned to the audio, e.g. [MUSIC], [APPLAUSE], [LAUGHTER], [DOOR SLAMS], [ENGINE REVVING]. "
    "Do not paraphrase sound cues; keep them short and standard. "
    "When multiple speakers are present, label speakers consistently."
)

# =============================================================================
# Utilities
# =============================================================================

PUNCT_END_RE = re.compile(r"[.!?…]+$")

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
    if len(t) < 3:
        return False
    return t.startswith("[") and t.endswith("]")

def _clean_sdh_token(t: str, uppercase: bool) -> str:
    t = (t or "").strip()
    if not (t.startswith("[") and t.endswith("]")):
        t = f"[{t}]"
    inner = t[1:-1].strip()
    inner = re.sub(r"\s+", " ", inner)
    if uppercase:
        inner = inner.upper()
    return f"[{inner}]"

def wrap_text_to_lines(text: str, max_chars: int, max_lines: int) -> List[str]:
    """
    Greedy wrap that tries to fill lines close to max_chars without exceeding.
    Returns list of lines (len <= max_lines, may be longer if forced).
    """
    words = text.split()
    if not words:
        return [""]

    lines: List[str] = []
    cur = ""

    def push():
        nonlocal cur
        if cur.strip():
            lines.append(cur.strip())
        cur = ""

    for w in words:
        if not cur:
            cur = w
            continue
        if len(cur) + 1 + len(w) <= max_chars:
            cur = f"{cur} {w}"
        else:
            push()
            cur = w
            if len(lines) >= max_lines:
                # force remainder into last line
                lines[-1] = (lines[-1] + " " + cur).strip()
                cur = ""

    if cur:
        push()

    # If still too many lines, compress
    if len(lines) > max_lines:
        head = lines[:max_lines - 1]
        tail = " ".join(lines[max_lines - 1:])
        lines = head + [tail]

    return lines[:max_lines]

def _text_cps(text: str, duration_ms: int) -> float:
    dur = max(1, duration_ms) / 1000.0
    chars = len(text.replace("\n", ""))
    return chars / dur

def _format_speaker_prefix(speaker: str, rules: Rules) -> str:
    if rules.speakerStyle == "none":
        return ""
    if rules.speakerStyle == "labels":
        return f"[{speaker}] "
    # dash style handled elsewhere
    return ""

# =============================================================================
# Broadcast Cue Builder (PRO)
# =============================================================================

def build_cues_pro(words: List[Word], rules: Rules) -> Tuple[List[Dict[str, Any]], List[Event]]:
    """
    PRO principles:
    - NEVER extend cue OUT past last word end (prevents drift).
    - Prefer filling lines toward maxCharsPerLine.
    - Merge short cues forward when possible.
    - Split only when required by max duration / long pause / speaker change constraints.
    - Extract SDH bracket tokens into standalone cues with their own timestamps.
    """

    if not words:
        return [], []

    # Sort by time
    words = sorted(words, key=lambda w: (w.start, w.end))

    # Pass 1: extract SDH bracket tokens as Events and remove them from dialogue stream
    events: List[Event] = []
    dialogue: List[Word] = []
    for w in words:
        t = (w.text or "").strip()
        if _is_bracket_token(t):
            events.append(
                Event(
                    type="sound",
                    start=int(w.start),
                    end=int(w.end),
                    text=_clean_sdh_token(t, rules.sdhBracketsUppercase),
                )
            )
        else:
            dialogue.append(w)

    if not dialogue and events:
        # Only SDH tokens; return them as cues
        cues = [{"start": e.start, "end": e.end, "text": e.text, "speaker": "A"} for e in events]
        cues = sorted(cues, key=lambda c: (c["start"], c["end"]))
        return cues, events

    # Helper to build cue text with wrapping and speaker formatting
    def make_cue(words_chunk: List[Word], speaker: str) -> Dict[str, Any]:
        raw = " ".join((w.text or "").strip() for w in words_chunk).strip()
        prefix = _format_speaker_prefix(speaker, rules)
        # fill lines
        lines = wrap_text_to_lines(prefix + raw, rules.maxCharsPerLine, rules.maxLines)
        text = "\n".join(lines).strip()
        return {
            "start": int(words_chunk[0].start),
            "end": int(words_chunk[-1].end),
            "text": text,
            "speaker": speaker,
            "wordCount": len(words_chunk),
        }

    # Decide if adding a word would break constraints
    def would_break(cur_words: List[Word], next_word: Word, speaker: str) -> bool:
        test_words = cur_words + [next_word]
        start = int(test_words[0].start)
        end = int(test_words[-1].end)
        dur = max(1, end - start)

        # hard duration cap
        if dur > rules.hardMaxDurationMs:
            return True

        cue = make_cue(test_words, speaker)
        text = cue["text"]
        # line/char limits are enforced by wrapping, but CPS might break
        cps = _text_cps(text, dur)
        if cps > rules.maxCPS:
            return True

        # soft max duration
        if dur > rules.maxDurationMs:
            return True

        return False

    # Pass 2: build initial cues with conservative splitting
    cues: List[Dict[str, Any]] = []
    cur_words: List[Word] = []
    cur_speaker = _normalize_speaker(dialogue[0].speaker)

    for w in dialogue:
        sp = _normalize_speaker(w.speaker)

        if not cur_words:
            cur_words = [w]
            cur_speaker = sp
            continue

        prev = cur_words[-1]
        gap = int(w.start) - int(prev.end)

        # speaker change: split (we may later merge with dash style if appropriate)
        if sp != cur_speaker:
            cues.append(make_cue(cur_words, cur_speaker))
            cur_words = [w]
            cur_speaker = sp
            continue

        # large pause: split
        if gap >= rules.gapSplitMs:
            cues.append(make_cue(cur_words, cur_speaker))
            cur_words = [w]
            cur_speaker = sp
            continue

        # constraint-based split
        if would_break(cur_words, w, cur_speaker):
            cues.append(make_cue(cur_words, cur_speaker))
            cur_words = [w]
            cur_speaker = sp
            continue

        # otherwise add
        cur_words.append(w)

        # punctuation encouragement (only if we already satisfy minDuration)
        if rules.preferPunctuationBreaks and PUNCT_END_RE.search((w.text or "").strip()):
            dur = int(cur_words[-1].end) - int(cur_words[0].start)
            if dur >= rules.minDurationMs:
                cues.append(make_cue(cur_words, cur_speaker))
                cur_words = []

    if cur_words:
        cues.append(make_cue(cur_words, cur_speaker))

    # Pass 3: merge adjacent cues to eliminate flashing (WITHOUT changing end beyond last word)
    merged: List[Dict[str, Any]] = []
    i = 0
    while i < len(cues):
        c = cues[i]
        if not merged:
            merged.append(c)
            i += 1
            continue

        prev = merged[-1]
        same_speaker = prev["speaker"] == c["speaker"]
        gap = int(c["start"]) - int(prev["end"])

        if same_speaker and gap <= rules.mergeGapMs:
            # attempt merge
            # reconstruct "words" is hard once flattened; so merge by text is risky.
            # Instead: only merge if both are "tiny" and merge will NOT exceed caps by estimate.
            prev_dur = max(1, int(prev["end"]) - int(prev["start"]))
            c_dur = max(1, int(c["end"]) - int(c["start"]))
            new_start = int(prev["start"])
            new_end = int(c["end"])
            new_dur = max(1, new_end - new_start)
            if new_dur <= rules.hardMaxDurationMs:
                # remove speaker prefix duplication if present
                ptxt = str(prev["text"])
                ctxt = str(c["text"])
                # If labels style, both already include prefix; keep first, strip second prefix
                if rules.speakerStyle == "labels":
                    ctxt = re.sub(r"^\[[A-Z0-9]+\]\s+", "", ctxt)
                combined_raw = (ptxt + " " + ctxt).replace("\n", " ").strip()
                # rewrap to lines
                lines = wrap_text_to_lines(combined_raw, rules.maxCharsPerLine, rules.maxLines)
                combined_text = "\n".join(lines).strip()
                cps = _text_cps(combined_text, new_dur)
                if cps <= rules.maxCPS and new_dur <= rules.maxDurationMs:
                    merged[-1] = {
                        **prev,
                        "start": new_start,
                        "end": new_end,
                        "text": combined_text,
                        "wordCount": int(prev.get("wordCount", 0)) + int(c.get("wordCount", 0)),
                    }
                    i += 1
                    continue

        merged.append(c)
        i += 1

    cues = merged

    # Pass 4: dash-style 2-speaker cue (Option A)
    # If speakerStyle == "dash", we combine very short alternating speaker cues into a single 2-line cue:
    if rules.speakerStyle == "dash":
        dashed: List[Dict[str, Any]] = []
        i = 0
        while i < len(cues):
            if i + 1 < len(cues):
                a = cues[i]
                b = cues[i + 1]
                gap = int(b["start"]) - int(a["end"])
                if gap <= rules.mergeGapMs and a["speaker"] != b["speaker"]:
                    # Build two-line cue: "- line" / "- line"
                    # Strip any speaker labels that may exist
                    at = re.sub(r"^\[[A-Z0-9]+\]\s+", "", str(a["text"]).replace("\n", " ")).strip()
                    bt = re.sub(r"^\[[A-Z0-9]+\]\s+", "", str(b["text"]).replace("\n", " ")).strip()
                    two = f"- {at}\n- {bt}"
                    # Ensure <=2 lines and <=32 chars each by wrapping each line independently
                    a_lines = wrap_text_to_lines(f"- {at}", rules.maxCharsPerLine, 1)
                    b_lines = wrap_text_to_lines(f"- {bt}", rules.maxCharsPerLine, 1)
                    text = (a_lines[0] + "\n" + b_lines[0]).strip()
                    start = int(a["start"])
                    end = int(b["end"])
                    dur = max(1, end - start)
                    cps = _text_cps(text, dur)
                    if cps <= rules.maxCPS and dur <= rules.maxDurationMs:
                        dashed.append({
                            "start": start,
                            "end": end,
                            "text": text,
                            "speaker": f"{a['speaker']}/{b['speaker']}",
                            "wordCount": int(a.get("wordCount", 0)) + int(b.get("wordCount", 0)),
                        })
                        i += 2
                        continue
            dashed.append(cues[i])
            i += 1
        cues = dashed

    # Pass 5: inject SDH events as dedicated cues
    for e in events:
        cues.append({"start": e.start, "end": e.end, "text": e.text, "speaker": "A", "wordCount": 0})

    cues = sorted(cues, key=lambda c: (int(c["start"]), int(c["end"])))

    # Remove helper key
    for c in cues:
        if "wordCount" in c:
            del c["wordCount"]

    return cues, events

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

        cps = _text_cps(text, duration)
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

# Minimal SCC placeholder (kept from your MVP; can be upgraded later)
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
        hex_payload = "94ae 94ae 94ae 94ae " + _text_to_fake_scc_hex(text) + " 942c 942c"
        lines.append(f"{tc}\t{hex_payload}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"

# =============================================================================
# Format payload builder
# =============================================================================

def format_result_from_words(words: List[Word], rules: Rules) -> Dict[str, Any]:
    cues, events = build_cues_pro(words, rules)
    qc = qc_cues(cues, rules)
    return {
        "rules": rules.model_dump(),
        "cues": cues,
        "events": [e.model_dump() for e in events],
        "srt": to_srt(cues),
        "vtt": to_vtt(cues),
        "scc": to_scc(cues, rules),
        "qc": qc,
    }

# =============================================================================
# Routes
# =============================================================================

@app.get("/")
def root():
    return {"ok": True, "service": "ai-cc-creator", "version": app.version}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/v1/jobs", response_model=JobStatus)
async def create_job(req: JobCreateRequest):
    if not ASSEMBLYAI_API_KEY:
        raise HTTPException(status_code=500, detail="ASSEMBLYAI_API_KEY is not set in Railway Variables.")

    # Submit to AssemblyAI
    payload = {
        "audio_url": req.mediaUrl,
        "speaker_labels": req.speaker_labels,
        "language_detection": req.language_detection,
        "speech_models": ["universal-3-pro"],
        "prompt": SDH_PROMPT,
    }

    # Optional webhook (nice-to-have; polling still works even if webhook fails)
    if WEBHOOK_SECRET and PUBLIC_BASE_URL:
        payload["webhook_url"] = f"{PUBLIC_BASE_URL}/v1/webhooks/assemblyai"
        payload["webhook_auth_header_name"] = "X-Webhook-Token"
        payload["webhook_auth_header_value"] = WEBHOOK_SECRET

    headers = {"Authorization": ASSEMBLYAI_API_KEY, "Content-Type": "application/json"}

    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.post(ASSEMBLYAI_TRANSCRIPT_URL, json=payload, headers=headers)
        if r.status_code >= 300:
            return JobStatus(
                id=str(uuid.uuid4()),
                status="error",
                error=f"AssemblyAI submit failed: {r.status_code} {r.text}",
                assembly_id=None,
                result=None,
            )

        data = r.json()
        assembly_id = data.get("id")
        if not assembly_id:
            return JobStatus(
                id=str(uuid.uuid4()),
                status="error",
                error="AssemblyAI response missing transcript id.",
                assembly_id=None,
                result=None,
            )

    # Persist rules keyed by transcript id so polling survives restarts
    _db_put_job(assembly_id, json.dumps(req.rules.model_dump()))

    # IMPORTANT: Job id returned to Base44 == Assembly transcript id (no more 404 after restarts)
    return JobStatus(id=assembly_id, status="processing", error=None, assembly_id=assembly_id, result=None)

@app.get("/v1/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str = Path(..., description="AssemblyAI transcript id")):
    if not ASSEMBLYAI_API_KEY:
        raise HTTPException(status_code=500, detail="ASSEMBLYAI_API_KEY is not set in Railway Variables.")

    rules_dict = _db_get_rules(job_id) or Rules().model_dump()
    rules = Rules(**rules_dict)

    headers = {"Authorization": ASSEMBLYAI_API_KEY}
    async with httpx.AsyncClient(timeout=60.0) as client:
        r = await client.get(f"{ASSEMBLYAI_TRANSCRIPT_URL}/{job_id}", headers=headers)
        if r.status_code >= 300:
            return JobStatus(
                id=job_id,
                status="error",
                error=f"AssemblyAI get transcript failed: {r.status_code} {r.text}",
                assembly_id=job_id,
                result=None,
            )
        t = r.json()

    status = (t.get("status") or "").lower()
    if status in ("queued", "processing"):
        return JobStatus(id=job_id, status="processing", error=None, assembly_id=job_id, result=None)

    if status in ("error", "failed"):
        return JobStatus(
            id=job_id,
            status="error",
            error=t.get("error") or "AssemblyAI transcript failed",
            assembly_id=job_id,
            result=None,
        )

    if status != "completed":
        return JobStatus(id=job_id, status="processing", error=None, assembly_id=job_id, result=None)

    # Completed: build broadcast-safe cues from word timestamps
    words_raw = t.get("words") or []
    words: List[Word] = []
    for w in words_raw:
        words.append(
            Word(
                text=w.get("text", ""),
                start=int(w.get("start", 0)),
                end=int(w.get("end", 0)),
                speaker=_normalize_speaker(w.get("speaker")),
            )
        )

    result = format_result_from_words(words, rules)
    return JobStatus(id=job_id, status="done", error=None, assembly_id=job_id, result=result)

@app.post("/v1/webhooks/assemblyai")
async def assemblyai_webhook(request: Request):
    # Webhook is optional in this architecture; we accept it for completeness.
    if WEBHOOK_SECRET:
        token = request.headers.get("X-Webhook-Token", "")
        if token != WEBHOOK_SECRET:
            raise HTTPException(status_code=401, detail="Invalid webhook token")

    # We don't need to do anything here because polling queries AssemblyAI directly.
    # But returning 200 prevents AssemblyAI retries.
    return JSONResponse({"ok": True})
