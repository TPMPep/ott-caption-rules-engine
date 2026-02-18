import os
import re
import json
import time
import uuid
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================================
# Config
# ============================================================

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
if not ASSEMBLYAI_API_KEY:
    # Don't crash on import; just fail on job creation calls.
    pass

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").strip()
BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()  # optional; set to your Railway public URL
DB_PATH = os.getenv("SQLITE_PATH", "jobs.db")

# Caption rules (NBCU-ish defaults; adjust if needed)
MAX_CHARS_PER_LINE = int(os.getenv("MAX_CHARS_PER_LINE", "32"))
MAX_LINES = int(os.getenv("MAX_LINES", "2"))
MAX_CPS = float(os.getenv("MAX_CPS", "20"))
MIN_CUE_DURATION = float(os.getenv("MIN_CUE_DURATION", "0.6"))  # for injected SDH cues
POLL_HINT_SECONDS = int(os.getenv("POLL_HINT_SECONDS", "3"))  # Base44 polling interval hint only

# ============================================================
# FastAPI setup
# ============================================================

app = FastAPI(title="AI CC Creator API", version="1.0")

origins = ["*"] if ALLOWED_ORIGINS == "*" else [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DB helpers (SQLite)
# ============================================================

def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db() -> None:
    with db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                title TEXT,
                media_url TEXT,
                status TEXT NOT NULL,
                error TEXT,
                assembly_transcript_id TEXT,
                result_json TEXT,
                srt TEXT,
                vtt TEXT
            )
            """
        )
        conn.commit()

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

init_db()

# ============================================================
# Models
# ============================================================

class CreateJobRequest(BaseModel):
    title: str
    mediaUrl: str

class JobResponse(BaseModel):
    id: str
    createdAt: str
    updatedAt: str
    title: Optional[str] = None
    mediaUrl: Optional[str] = None
    status: str
    error: Optional[str] = None
    pollHintSeconds: int = POLL_HINT_SECONDS
    exports: Optional[Dict[str, Any]] = None

# ============================================================
# AssemblyAI client
# ============================================================

AA_BASE = "https://api.assemblyai.com/v2"

def aa_headers() -> Dict[str, str]:
    return {
        "authorization": ASSEMBLYAI_API_KEY,
        "content-type": "application/json",
    }

def aa_create_transcript(audio_url: str) -> str:
    """
    Creates transcript and returns transcript_id.
    """
    if not ASSEMBLYAI_API_KEY:
        raise HTTPException(status_code=500, detail="ASSEMBLYAI_API_KEY is not configured")

    webhook_url = ""
    if BASE_URL:
        webhook_url = f"{BASE_URL.rstrip('/')}/v1/webhooks/assemblyai"

    payload = {
        "audio_url": audio_url,
        "speaker_labels": True,
        "dual_channel": False,
        # You can keep your prompt; it helps generate SDH tokens in transcript text.
        "prompt": (
            "Transcribe dialogue accurately. Also include SDH-style non-speech sound cues in ALL CAPS "
            "inside brackets as standalone tokens, e.g. [♪ MUSIC ♪], [APPLAUSE], [LAUGHTER], [DOOR SLAMS]. "
            "Do not paraphrase sound cues; keep them short and standard."
        ),
    }
    if webhook_url:
        payload["webhook_url"] = webhook_url
        payload["webhook_auth_header_name"] = "x-webhook-token"
        payload["webhook_auth_header_value"] = os.getenv("ASSEMBLYAI_WEBHOOK_TOKEN", "").strip()

    r = requests.post(f"{AA_BASE}/transcript", headers=aa_headers(), data=json.dumps(payload), timeout=60)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"AssemblyAI create transcript failed: {r.status_code} {r.text}")

    transcript_id = r.json().get("id")
    if not transcript_id:
        raise HTTPException(status_code=502, detail="AssemblyAI create transcript failed: missing transcript id")

    return transcript_id

def aa_get_transcript(transcript_id: str) -> Dict[str, Any]:
    r = requests.get(f"{AA_BASE}/transcript/{transcript_id}", headers=aa_headers(), timeout=60)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"AssemblyAI get transcript failed: {r.status_code} {r.text}")
    return r.json()

def aa_get_srt(transcript_id: str) -> str:
    r = requests.get(f"{AA_BASE}/transcript/{transcript_id}/srt", headers=aa_headers(), timeout=60)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"AssemblyAI get SRT failed: {r.status_code} {r.text}")
    return r.text

def aa_get_words(transcript_id: str) -> List[Dict[str, Any]]:
    """
    Returns word list with timings + speaker labels if available.
    """
    r = requests.get(f"{AA_BASE}/transcript/{transcript_id}/words", headers=aa_headers(), timeout=60)
    if r.status_code >= 400:
        # Not fatal; we can still produce SRT-based output without speakers/cues.
        return []
    data = r.json()
    # AssemblyAI returns {"words":[...]} on some accounts; on others it may return list directly.
    if isinstance(data, dict) and "words" in data and isinstance(data["words"], list):
        return data["words"]
    if isinstance(data, list):
        return data
    return []

# ============================================================
# Time helpers
# ============================================================

def srt_time_to_seconds(t: str) -> float:
    # "HH:MM:SS,mmm" or "HH:MM:SS.mmm"
    t = t.replace(",", ".")
    hh, mm, rest = t.split(":")
    ss, ms = rest.split(".")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + (int(ms) / 1000.0)

def seconds_to_srt_time(x: float) -> str:
    if x < 0:
        x = 0
    hh = int(x // 3600)
    x -= hh * 3600
    mm = int(x // 60)
    x -= mm * 60
    ss = int(x)
    ms = int(round((x - ss) * 1000.0))
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

# ============================================================
# SRT parsing/formatting
# ============================================================

def parse_srt(srt_text: str) -> List[Dict[str, Any]]:
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    cues: List[Dict[str, Any]] = []
    for b in blocks:
        lines = b.strip().splitlines()
        if len(lines) < 3:
            continue
        # index line may exist; ignore if weird
        times = lines[1].strip()
        m = re.match(r"(\d\d:\d\d:\d\d[,.]\d\d\d)\s*-->\s*(\d\d:\d\d:\d\d[,.]\d\d\d)", times)
        if not m:
            continue
        start = srt_time_to_seconds(m.group(1))
        end = srt_time_to_seconds(m.group(2))
        text = "\n".join(lines[2:]).strip()
        cues.append({"start": start, "end": end, "text": text})
    return cues

def cues_to_srt(cues: List[Dict[str, Any]]) -> str:
    out = []
    for i, c in enumerate(cues, 1):
        out.append(str(i))
        out.append(f"{seconds_to_srt_time(c['start'])} --> {seconds_to_srt_time(c['end'])}")
        out.append(c["text"].rstrip())
        out.append("")
    return "\n".join(out).rstrip() + "\n"

def cues_to_vtt(cues: List[Dict[str, Any]]) -> str:
    def vtt_time(x: float) -> str:
        # HH:MM:SS.mmm
        hh = int(x // 3600)
        x -= hh * 3600
        mm = int(x // 60)
        x -= mm * 60
        ss = int(x)
        ms = int(round((x - ss) * 1000.0))
        return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"

    out = ["WEBVTT", ""]
    for c in cues:
        out.append(f"{vtt_time(c['start'])} --> {vtt_time(c['end'])}")
        out.append(c["text"].rstrip())
        out.append("")
    return "\n".join(out).rstrip() + "\n"

# ============================================================
# Line wrap (fills up to 32 chars/line where possible)
# ============================================================

_WORD_SPLIT = re.compile(r"\s+")

def normalize_for_match(w: str) -> str:
    w = w.strip().lower()
    w = re.sub(r"^[\W_]+|[\W_]+$", "", w)  # trim punctuation ends
    return w

def wrap_to_lines(text: str, max_chars: int, max_lines: int) -> str:
    """
    Wrap text into <=max_lines lines, each <=max_chars when possible,
    using greedy packing to *fill* the line.
    """
    raw = text.replace("\n", " ").strip()
    raw = re.sub(r"\s+", " ", raw)
    if not raw:
        return ""

    words = raw.split(" ")
    lines: List[str] = []
    cur = ""

    def flush():
        nonlocal cur
        if cur:
            lines.append(cur)
            cur = ""

    for w in words:
        if not cur:
            cur = w
        else:
            candidate = f"{cur} {w}"
            if len(candidate) <= max_chars:
                cur = candidate
            else:
                flush()
                cur = w

        # If we've already used max_lines-1 lines, push everything else into last line (truncate wrap)
        if len(lines) == max_lines - 1:
            # pack remaining words into final line respecting max_chars as much as possible
            remaining = [cur] + [x for x in words[words.index(w) + 1:]]
            final = remaining[0]
            for rw in remaining[1:]:
                cand = f"{final} {rw}"
                if len(cand) <= max_chars:
                    final = cand
                else:
                    break
            lines.append(final)
            return "\n".join(lines)

    flush()
    return "\n".join(lines[:max_lines])

def cps_for_cue(text: str, start: float, end: float) -> float:
    dur = max(0.001, end - start)
    chars = len(text.replace("\n", ""))
    return chars / dur

# ============================================================
# Speaker + SDH overlay logic
# ============================================================

SOUND_CUE_PAT = re.compile(r"\[(.*?)\]")
MUSIC_NOTE_PAT = re.compile(r"^\s*[♪♫].*[♪♫]\s*$")

def standardize_sound_cue(token: str) -> str:
    t = token.strip()
    # Normalize music tokens
    if MUSIC_NOTE_PAT.match(t) or "music" in t.lower():
        return "[♪ MUSIC ♪]"
    t = t.upper()
    t = re.sub(r"\s+", " ", t)
    if not t.startswith("["):
        t = "[" + t
    if not t.endswith("]"):
        t = t + "]"
    return t

def align_sound_cues_to_words(transcript_text: str, words: List[Dict[str, Any]]) -> List[Tuple[float, str]]:
    """
    Finds bracket tokens in transcript text and assigns a timestamp by walking
    transcript tokens and matching spoken words to the AssemblyAI word list.
    If a bracket token occurs, we timestamp it at the next matched spoken word start,
    otherwise fallback to previous word end.
    """
    if not transcript_text:
        return []
    if not words:
        # No timing source to align to.
        return []

    # Tokenize transcript text into either bracket tokens or word-ish tokens
    raw = transcript_text.replace("\n", " ")
    raw = re.sub(r"\s+", " ", raw).strip()

    tokens: List[str] = []
    i = 0
    while i < len(raw):
        if raw[i] == "[":
            j = raw.find("]", i + 1)
            if j != -1:
                tokens.append(raw[i:j + 1])
                i = j + 1
                continue
        # normal token
        # consume until space or bracket
        j = i
        while j < len(raw) and raw[j] not in [" ", "["]:
            j += 1
        tokens.append(raw[i:j])
        i = j + 1 if j < len(raw) and raw[j] == " " else j

    # Walk words list and transcript tokens in lockstep
    w_idx = 0
    cues: List[Tuple[float, str]] = []
    last_end = float(words[0].get("end", 0)) / 1000.0

    def w_norm(idx: int) -> str:
        return normalize_for_match(str(words[idx].get("text", "")))

    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("[") and tok.endswith("]"):
            # timestamp at next word start if possible
            if w_idx < len(words):
                ts = float(words[w_idx].get("start", 0)) / 1000.0
            else:
                ts = last_end
            cues.append((ts, standardize_sound_cue(tok)))
            continue

        # try to match this token to the current word (robust-ish)
        tn = normalize_for_match(tok)
        if not tn:
            continue

        # advance word pointer until match or give up a little
        # (keeps alignment without exploding on punctuation)
        for _ in range(6):
            if w_idx >= len(words):
                break
            wn = w_norm(w_idx)
            if wn == tn:
                last_end = float(words[w_idx].get("end", 0)) / 1000.0
                w_idx += 1
                break
            # allow partial match cases (e.g., "car's" vs "cars")
            if wn and tn and (wn.startswith(tn) or tn.startswith(wn)):
                last_end = float(words[w_idx].get("end", 0)) / 1000.0
                w_idx += 1
                break
            w_idx += 1

    # de-dupe cues that are the same and very close in time
    deduped: List[Tuple[float, str]] = []
    for ts, txt in cues:
        if not deduped:
            deduped.append((ts, txt))
            continue
        pts, ptxt = deduped[-1]
        if txt == ptxt and abs(ts - pts) < 0.25:
            continue
        deduped.append((ts, txt))

    return deduped

def words_in_window(words: List[Dict[str, Any]], start: float, end: float) -> List[Dict[str, Any]]:
    if not words:
        return []
    s_ms = int(start * 1000)
    e_ms = int(end * 1000)
    out = []
    for w in words:
        ws = int(w.get("start", 0))
        we = int(w.get("end", 0))
        # overlap test
        if we <= s_ms:
            continue
        if ws >= e_ms:
            break
        out.append(w)
    return out

def apply_nbcu_dash_if_two_speakers(cue_text: str, cue_words: List[Dict[str, Any]]) -> str:
    """
    NBCU style you described:
    - No speaker names.
    - ONLY use dashes when two people speak in the same caption instance.
    """
    if not cue_words:
        return cue_text

    speakers = [w.get("speaker") for w in cue_words if w.get("speaker") is not None]
    uniq = []
    for s in speakers:
        if s not in uniq:
            uniq.append(s)

    # Single speaker => NO dash
    if len(uniq) <= 1:
        return cue_text

    # Two+ speakers in one cue => split by speaker runs into up to 2 lines
    # We create a simplified two-line representation:
    # - <speaker1 run text>
    # - <speaker2 run text>
    runs: List[Tuple[Any, List[str]]] = []
    cur_spk = None
    cur_words: List[str] = []

    for w in cue_words:
        spk = w.get("speaker")
        txt = str(w.get("text", "")).strip()
        if not txt:
            continue
        if cur_spk is None:
            cur_spk = spk
            cur_words = [txt]
            continue
        if spk == cur_spk:
            cur_words.append(txt)
        else:
            runs.append((cur_spk, cur_words))
            cur_spk = spk
            cur_words = [txt]
    if cur_spk is not None and cur_words:
        runs.append((cur_spk, cur_words))

    # Take first two speaker runs (broadcast 2-line max)
    line1 = " ".join(runs[0][1]) if len(runs) >= 1 else cue_text.replace("\n", " ")
    line2 = " ".join(runs[1][1]) if len(runs) >= 2 else ""

    line1 = wrap_to_lines(line1, MAX_CHARS_PER_LINE, 1)
    line2 = wrap_to_lines(line2, MAX_CHARS_PER_LINE, 1) if line2 else ""

    # Prefix dash on BOTH lines (this is the only time we do it)
    if line2:
        return f"- {line1}\n- {line2}"
    else:
        return f"- {line1}"

def insert_sound_cue_captions(
    cues: List[Dict[str, Any]],
    sound_events: List[Tuple[float, str]],
) -> List[Dict[str, Any]]:
    """
    Insert SDH cues as standalone captions at best-effort timestamps.
    We try to fit them into gaps; if no gap, we still insert with minimal duration
    and clamp to avoid negative or inverted times.
    """
    if not sound_events:
        return cues

    cues_sorted = sorted(cues, key=lambda c: (c["start"], c["end"]))
    out: List[Dict[str, Any]] = []

    # Build quick access to next cue start for clamping
    idx = 0
    for c in cues_sorted:
        out.append(c)

    # Insert and re-sort (safe because we clamp times)
    for ts, sdh in sound_events:
        # Find the nearest cue index where this timestamp belongs
        # We want to place it at ts with duration MIN_CUE_DURATION, but prefer gaps.
        insert_i = 0
        while insert_i < len(out) and out[insert_i]["start"] <= ts:
            insert_i += 1

        prev_end = out[insert_i - 1]["end"] if insert_i - 1 >= 0 else 0.0
        next_start = out[insert_i]["start"] if insert_i < len(out) else None

        start = max(ts, prev_end)
        end = start + MIN_CUE_DURATION

        if next_start is not None:
            end = min(end, max(start + 0.2, next_start))  # ensure it shows briefly
            # If there is a real gap, fill it more naturally
            gap = next_start - start
            if gap >= 0.7:
                end = min(next_start, start + min(1.5, gap))

        if end <= start:
            continue

        out.append({"start": start, "end": end, "text": sdh, "type": "sdh"})

    out = sorted(out, key=lambda c: (c["start"], c["end"]))

    # Remove duplicates that land almost same time with same text
    cleaned: List[Dict[str, Any]] = []
    for c in out:
        if not cleaned:
            cleaned.append(c)
            continue
        p = cleaned[-1]
        if c["text"] == p["text"] and abs(c["start"] - p["start"]) < 0.2:
            continue
        cleaned.append(c)

    return cleaned

# ============================================================
# Core pipeline: SRT-first + overlays
# ============================================================

def build_pro_captions_from_assembly(transcript_id: str) -> Dict[str, Any]:
    """
    1) Fetch AssemblyAI transcript JSON (to grab SDH bracket tokens in `text`)
    2) Fetch AssemblyAI SRT (best timing/segmentation)
    3) Fetch word timestamps (speaker labels) if available
    4) Wrap lines to 32 chars (max 2 lines) WITHOUT changing cue times
    5) Only apply "-" when multiple speakers truly occur in same cue window
    6) Insert SDH cues as standalone captions (best effort alignment)
    """
    transcript = aa_get_transcript(transcript_id)
    transcript_text = transcript.get("text", "") or ""

    srt_raw = aa_get_srt(transcript_id)
    cues = parse_srt(srt_raw)

    words = aa_get_words(transcript_id)
    # Ensure words are sorted by start for window selection
    words = sorted(words, key=lambda w: int(w.get("start", 0))) if words else []

    # SDH sound cues: align bracket tokens from transcript text back to words
    sound_events = align_sound_cues_to_words(transcript_text, words)

    pro_cues: List[Dict[str, Any]] = []
    for c in cues:
        start = float(c["start"])
        end = float(c["end"])
        txt = c["text"].strip()

        # Normalize whitespace; keep original meaning
        txt = txt.replace("\r", "").strip()
        txt = re.sub(r"[ \t]+", " ", txt.replace("\n", " ")).strip()

        # Wrap to max lines/chars
        wrapped = wrap_to_lines(txt, MAX_CHARS_PER_LINE, MAX_LINES)

        # Speaker dash only when 2 speakers occur in this SAME cue window
        cw = words_in_window(words, start, end)
        wrapped = apply_nbcu_dash_if_two_speakers(wrapped, cw)

        # Enforce max chars/line again after dash insertion
        if wrapped.startswith("- "):
            parts = wrapped.split("\n")
            fixed_parts = []
            for p in parts:
                if p.startswith("- "):
                    content = p[2:].strip()
                    content = wrap_to_lines(content, MAX_CHARS_PER_LINE - 2, 1)
                    fixed_parts.append(f"- {content}")
                else:
                    fixed_parts.append(wrap_to_lines(p, MAX_CHARS_PER_LINE, 1))
            wrapped = "\n".join(fixed_parts[:MAX_LINES])

        # CPS issue detection (do not stretch times)
        cps = cps_for_cue(wrapped, start, end)
        issues = []
        if cps > MAX_CPS:
            issues.append({"rule": "cps_high", "value": round(cps, 2)})

        # If cue is weirdly short with tiny text, we still keep it because we are NOT allowed
        # to merge cues (merging is what caused timing drift).
        pro_cues.append({
            "start": start,
            "end": end,
            "text": wrapped,
            "cps": round(cps, 2),
            "issues": issues,
            "type": "dialogue",
        })

    # Insert SDH cues as standalone captions
    pro_cues = insert_sound_cue_captions(pro_cues, sound_events)

    # Produce exports
    srt_out = cues_to_srt(pro_cues)
    vtt_out = cues_to_vtt(pro_cues)

    result_json = {
        "transcriptId": transcript_id,
        "rules": {
            "maxCharsPerLine": MAX_CHARS_PER_LINE,
            "maxLines": MAX_LINES,
            "maxCPS": MAX_CPS,
        },
        "cues": pro_cues,
    }

    return {
        "result_json": result_json,
        "srt": srt_out,
        "vtt": vtt_out,
    }

# ============================================================
# API endpoints
# ============================================================

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "ts": now_iso()}

@app.post("/v1/jobs", response_model=JobResponse)
def create_job(req: CreateJobRequest) -> JobResponse:
    transcript_id = aa_create_transcript(req.mediaUrl)

    created = now_iso()
    with db() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO jobs
            (id, created_at, updated_at, title, media_url, status, error, assembly_transcript_id, result_json, srt, vtt)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transcript_id,  # IMPORTANT: job id == Assembly transcript id
                created,
                created,
                req.title,
                req.mediaUrl,
                "processing",
                None,
                transcript_id,
                None,
                None,
                None,
            ),
        )
        conn.commit()

    return JobResponse(
        id=transcript_id,
        createdAt=created,
        updatedAt=created,
        title=req.title,
        mediaUrl=req.mediaUrl,
        status="processing",
        pollHintSeconds=POLL_HINT_SECONDS,
        exports=None,
    )

@app.get("/v1/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str) -> JobResponse:
    with db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")

    exports = None
    if row["status"] == "completed":
        exports = {"srt": True, "vtt": True, "json": True}

    return JobResponse(
        id=row["id"],
        createdAt=row["created_at"],
        updatedAt=row["updated_at"],
        title=row["title"],
        mediaUrl=row["media_url"],
        status=row["status"],
        error=row["error"],
        pollHintSeconds=POLL_HINT_SECONDS,
        exports=exports,
    )

@app.get("/v1/jobs/{job_id}/export/srt")
def export_srt(job_id: str) -> str:
    with db() as conn:
        row = conn.execute("SELECT srt, status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        if row["status"] != "completed" or not row["srt"]:
            raise HTTPException(status_code=409, detail="Job not completed")
        return row["srt"]

@app.get("/v1/jobs/{job_id}/export/vtt")
def export_vtt(job_id: str) -> str:
    with db() as conn:
        row = conn.execute("SELECT vtt, status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        if row["status"] != "completed" or not row["vtt"]:
            raise HTTPException(status_code=409, detail="Job not completed")
        return row["vtt"]

@app.get("/v1/jobs/{job_id}/export/json")
def export_json(job_id: str) -> Dict[str, Any]:
    with db() as conn:
        row = conn.execute("SELECT result_json, status FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Job not found")
        if row["status"] != "completed" or not row["result_json"]:
            raise HTTPException(status_code=409, detail="Job not completed")
        return json.loads(row["result_json"])

@app.post("/v1/webhooks/assemblyai")
async def assemblyai_webhook(request: Request) -> Dict[str, Any]:
    """
    AssemblyAI calls this when transcript completes/fails (if webhook_url was set).
    We update the SAME id that Base44 is polling (job_id == transcript_id).
    """
    payload = await request.json()

    # Optional auth
    expected = os.getenv("ASSEMBLYAI_WEBHOOK_TOKEN", "").strip()
    if expected:
        got = request.headers.get("x-webhook-token", "")
        if got != expected:
            raise HTTPException(status_code=401, detail="Invalid webhook token")

    transcript_id = payload.get("transcript_id") or payload.get("id")
    status = payload.get("status")

    if not transcript_id:
        raise HTTPException(status_code=400, detail="Missing transcript_id")

    updated = now_iso()

    # We always upsert because sometimes jobs are created before webhook and vice versa.
    with db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (transcript_id,)).fetchone()
        if not row:
            conn.execute(
                """
                INSERT OR REPLACE INTO jobs
                (id, created_at, updated_at, title, media_url, status, error, assembly_transcript_id, result_json, srt, vtt)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    transcript_id,
                    updated,
                    updated,
                    None,
                    None,
                    "processing",
                    None,
                    transcript_id,
                    None,
                    None,
                    None,
                ),
            )
            conn.commit()

    if status == "completed":
        try:
            built = build_pro_captions_from_assembly(transcript_id)
            with db() as conn:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = ?, error = ?, updated_at = ?, result_json = ?, srt = ?, vtt = ?
                    WHERE id = ?
                    """,
                    (
                        "completed",
                        None,
                        updated,
                        json.dumps(built["result_json"]),
                        built["srt"],
                        built["vtt"],
                        transcript_id,
                    ),
                )
                conn.commit()
        except Exception as e:
            with db() as conn:
                conn.execute(
                    "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                    ("error", f"{type(e).__name__}: {str(e)}", updated, transcript_id),
                )
                conn.commit()

    elif status == "error" or status == "failed":
        err = payload.get("error") or "AssemblyAI transcript failed"
        with db() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                ("error", err, updated, transcript_id),
            )
            conn.commit()

    else:
        # still processing
        with db() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
                ("processing", updated, transcript_id),
            )
            conn.commit()

    return {"ok": True}

# ============================================================
# Notes:
# - We DO NOT merge SRT cues (merging caused timing drift).
# - We ONLY add '-' when >1 speaker appears inside the same cue window.
# - SDH cues are inserted as standalone captions using transcript text alignment.
# ============================================================
