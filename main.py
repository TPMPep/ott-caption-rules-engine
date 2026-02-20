# ============================================================
# AI CC CREATOR – NBCU-STYLE BROADCAST CAPTION ENGINE
# AssemblyAI words -> rule-driven cue segmentation (CPS/lines/duration/gap)
# SDH (sound cues) handled as standalone cues
# Speaker changes rendered with "-" only when needed (no names)
# ============================================================

import os
import re
import json
import sqlite3
from uuid import uuid4
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ============================================================
# CONFIG
# ============================================================

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()
DB_PATH = os.getenv("SQLITE_PATH", "jobs.db")
POLL_HINT_SECONDS = int(os.getenv("POLL_HINT_SECONDS", "3"))

AA_BASE = "https://api.assemblyai.com/v2"

# ============================================================
# FASTAPI + CORS
# ============================================================

app = FastAPI(title="AI CC Creator API", version="NBCU-Broadcast")

allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()
allow_origins = ["*"] if not allowed_origins_env else [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
allow_origin_regex = r"^https://.*\.(base44\.app|base44\.com)$"

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_origin_regex=allow_origin_regex,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

@app.on_event("startup")
async def _startup():
    print("CORS allow_origins:", allow_origins)
    print("CORS allow_origin_regex:", allow_origin_regex)

@app.options("/{full_path:path}")
def _options_preflight(full_path: str, request: Request):
    return Response(status_code=200)

_ALLOWED_ORIGINS_LIST = [o for o in allow_origins if o != "*"]
_ALLOWED_ORIGIN_REGEX = re.compile(allow_origin_regex) if allow_origin_regex else None

def _maybe_add_cors_headers(response: Response, request: Request) -> Response:
    origin = request.headers.get("origin")
    if not origin:
        return response
    ok = False
    if "*" in allow_origins:
        ok = True
    elif origin in _ALLOWED_ORIGINS_LIST:
        ok = True
    elif _ALLOWED_ORIGIN_REGEX and _ALLOWED_ORIGIN_REGEX.match(origin):
        ok = True
    if ok:
        response.headers["Access-Control-Allow-Origin"] = "*" if "*" in allow_origins else origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,PATCH,DELETE,OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = request.headers.get("access-control-request-headers", "*")
    return response

@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    resp = JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {str(exc)}"})
    return _maybe_add_cors_headers(resp, request)

# ============================================================
# DB
# ============================================================

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def init_db():
    with db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            transcript_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            title TEXT,
            media_url TEXT,
            status TEXT NOT NULL,
            error TEXT,
            rules_json TEXT,
            result_json TEXT,
            srt TEXT,
            vtt TEXT
        )""")
        conn.commit()

def ensure_migrations():
    with db() as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(jobs)").fetchall()]
        if "transcript_id" not in cols:
            conn.execute("ALTER TABLE jobs ADD COLUMN transcript_id TEXT")
            conn.commit()

init_db()
ensure_migrations()

# ============================================================
# MODELS
# ============================================================

class CreateJobRequest(BaseModel):
    title: Optional[str] = None
    mediaUrl: str
    speaker_labels: Optional[bool] = True
    language_detection: Optional[bool] = True
    rules: Optional[Dict[str, Any]] = None

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
# ASSEMBLYAI
# ============================================================

def aa_headers() -> Dict[str, str]:
    return {"authorization": ASSEMBLYAI_API_KEY, "content-type": "application/json"}

def aa_create_transcript(audio_url: str, speaker_labels: bool = True, **kwargs) -> str:
    api_key = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="ASSEMBLYAI_API_KEY is not configured")

    language_detection = bool(kwargs.get("language_detection", True))
    webhook_url = (kwargs.get("webhook_url") or "").strip()
    webhook_secret = (kwargs.get("webhook_secret") or "").strip()

    if not webhook_url and BASE_URL:
        webhook_url = f"{BASE_URL.rstrip('/')}/v1/webhooks/assemblyai"
    if not webhook_secret:
        webhook_secret = os.getenv("ASSEMBLYAI_WEBHOOK_TOKEN", "").strip()

    speech_models = kwargs.get("speech_models") or ["universal-3-pro", "universal-2"]

    payload: Dict[str, Any] = {
        "audio_url": audio_url,
        "speaker_labels": bool(speaker_labels),
        "language_detection": language_detection,
        "speech_models": speech_models,
    }

    if "universal-3-pro" in speech_models:
        payload["prompt"] = (
            "Transcribe dialogue accurately. Include SDH-style non-speech cues in ALL CAPS in brackets, "
            "e.g. [♪ MUSIC ♪], [APPLAUSE], [LAUGHTER], [DOOR SLAMS]. Keep cues short and standard."
        )

    if webhook_url:
        payload["webhook_url"] = webhook_url
        if webhook_secret:
            payload["webhook_auth_header_name"] = "x-webhook-token"
            payload["webhook_auth_header_value"] = webhook_secret

    r = requests.post(f"{AA_BASE}/transcript", headers=aa_headers(), json=payload, timeout=60)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"AssemblyAI create transcript failed: {r.status_code} {r.text}")

    tid = r.json().get("id")
    if not tid:
        raise HTTPException(status_code=502, detail="AssemblyAI create transcript failed: missing transcript id")
    return tid

def aa_get_transcript(transcript_id: str) -> Dict[str, Any]:
    r = requests.get(f"{AA_BASE}/transcript/{transcript_id}", headers=aa_headers(), timeout=60)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"AssemblyAI get transcript failed: {r.status_code} {r.text}")
    return r.json()

def aa_get_words(transcript_id: str) -> List[Dict[str, Any]]:
    r = requests.get(f"{AA_BASE}/transcript/{transcript_id}/words", headers=aa_headers(), timeout=60)
    if r.status_code >= 400:
        return []
    data = r.json()
    if isinstance(data, dict) and isinstance(data.get("words"), list):
        return data["words"]
    if isinstance(data, list):
        return data
    return []

# ============================================================
# NBCU RULES DEFAULTS
# ============================================================

def get_rules(req_rules: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    r = dict(req_rules or {})
    # NBCU-ish defaults
    r.setdefault("maxCharsPerLine", 32)
    r.setdefault("maxLines", 2)
    r.setdefault("maxCPS", 17)
    r.setdefault("minDurationMs", 1000)
    r.setdefault("maxDurationMs", 7000)
    r.setdefault("minGapMs", 80)
    r.setdefault("preferPunctuationBreaks", True)

    # Speaker handling
    r.setdefault("minWordsPerRun", 2)          # ignore speaker jitter
    r.setdefault("maxSpeakersInCue", 2)

    # SDH handling
    r.setdefault("sdhMinCueDurationMs", 600)
    r.setdefault("sdhMergeWindowMs", 350)     # merge consecutive SDH tokens
    r.setdefault("sdhDedupWindowMs", 250)

    # Small timing pads (helps CPS without breaking sync)
    r.setdefault("tailPadMs", 120)

    return r

# ============================================================
# TEXT WRAP
# ============================================================

_PUNCT_BREAK = re.compile(r"[,\.;:\!\?]$")
_WS = re.compile(r"\s+")
def wrap_text(text: str, max_chars: int, max_lines: int, prefer_punct: bool = True) -> str:
    raw = _WS.sub(" ", (text or "").replace("\n", " ").strip())
    if not raw:
        return ""
    words = raw.split(" ")
    lines: List[str] = []
    cur: List[str] = []

    def cur_len_with(w: str) -> int:
        if not cur:
            return len(w)
        return len(" ".join(cur)) + 1 + len(w)

    def flush():
        nonlocal cur
        if cur:
            lines.append(" ".join(cur))
            cur = []

    i = 0
    while i < len(words):
        w = words[i]
        if cur and cur_len_with(w) > max_chars:
            flush()
            if len(lines) >= max_lines:
                break
            continue
        cur.append(w)
        if prefer_punct and _PUNCT_BREAK.search(w) and (i + 1) < len(words):
            if len(" ".join(cur)) >= max_chars * 0.55:
                flush()
                if len(lines) >= max_lines:
                    break
        i += 1

    if cur and len(lines) < max_lines:
        flush()

    return "\n".join(lines[:max_lines])

def calc_cps(text: str, dur_ms: int) -> float:
    dur_s = max(0.001, dur_ms / 1000.0)
    chars = len((text or "").replace("\n", ""))
    return chars / dur_s

# ============================================================
# SDH + SPEAKER UTILITIES
# ============================================================

SDH_TOKEN = re.compile(r"^\[.*\]$")

def is_sdh_word(w: Dict[str, Any]) -> bool:
    t = str(w.get("text", "")).strip()
    return bool(t) and bool(SDH_TOKEN.match(t))

def normalize_sdh_token(t: str) -> str:
    t = str(t or "").strip()
    # Normalize MUSIC variants
    if re.search(r"music", t, re.IGNORECASE):
        return "[♪ MUSIC ♪]"
    # Uppercase + squeeze spaces
    inner = t.strip()[1:-1] if t.startswith("[") and t.endswith("]") else t
    inner = _WS.sub(" ", inner).upper().strip()
    return f"[{inner}]"

def build_sdh_events(words: List[Dict[str, Any]], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create standalone SDH cues from bracket tokens in the word stream."""
    if not words:
        return []
    min_dur = int(rules["sdhMinCueDurationMs"])
    merge_win = int(rules["sdhMergeWindowMs"])
    dedup_win = int(rules["sdhDedupWindowMs"])

    # Collect raw SDH events at their word start
    raw: List[Tuple[int, str]] = []
    for w in words:
        if is_sdh_word(w):
            raw.append((int(w.get("start", 0)), normalize_sdh_token(w.get("text", ""))))

    if not raw:
        return []

    # Dedup identical tokens in a short window
    raw.sort(key=lambda x: x[0])
    deduped: List[Tuple[int, str]] = []
    for ts, tok in raw:
        if not deduped:
            deduped.append((ts, tok))
            continue
        pts, ptok = deduped[-1]
        if tok == ptok and abs(ts - pts) <= dedup_win:
            continue
        deduped.append((ts, tok))

    # Merge tokens that are near each other into one cue (NBCU-friendly)
    merged: List[Dict[str, Any]] = []
    cur_start = deduped[0][0]
    cur_end = cur_start + min_dur
    cur_tokens = [deduped[0][1]]

    for ts, tok in deduped[1:]:
        if ts <= cur_end + merge_win:
            cur_tokens.append(tok)
            cur_end = max(cur_end, ts + min_dur)
        else:
            merged.append({
                "start": cur_start,
                "end": cur_end,
                "text": " ".join(cur_tokens),
                "speaker": None,
                "type": "sdh",
            })
            cur_start = ts
            cur_end = ts + min_dur
            cur_tokens = [tok]

    merged.append({
        "start": cur_start,
        "end": cur_end,
        "text": " ".join(cur_tokens),
        "speaker": None,
        "type": "sdh",
    })

    return merged

def split_into_speaker_runs(words: List[Dict[str, Any]], min_words: int) -> List[Dict[str, Any]]:
    """Speaker runs for dash formatting (we do NOT make separate timecodes per run)."""
    def is_real_token(t: str) -> bool:
        # ignore SDH tokens + pure punctuation
        if not t:
            return False
        if SDH_TOKEN.match(t.strip()):
            return False
        return bool(re.sub(r"[\W_]+", "", t))

    runs: List[Dict[str, Any]] = []
    cur_spk = None
    cur_words: List[Dict[str, Any]] = []

    def flush():
        nonlocal cur_words, cur_spk
        if not cur_words:
            return
        wc = sum(1 for w in cur_words if is_real_token(str(w.get("text", "")).strip()))
        txt = " ".join(str(w.get("text", "")).strip() for w in cur_words).strip()
        runs.append({"speaker": cur_spk, "text": txt, "word_count": wc})
        cur_words = []

    for w in words:
        t = str(w.get("text", "")).strip()
        if not t or SDH_TOKEN.match(t):
            continue
        spk = w.get("speaker")
        if cur_spk is None and not cur_words:
            cur_spk = spk
            cur_words = [w]
            continue
        if spk == cur_spk:
            cur_words.append(w)
        else:
            flush()
            cur_spk = spk
            cur_words = [w]
    flush()

    # Merge tiny jitter runs
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(runs):
        r = runs[i]
        if r["word_count"] >= min_words:
            out.append(r); i += 1; continue
        if out:
            out[-1]["text"] = (out[-1]["text"] + " " + r["text"]).strip()
            out[-1]["word_count"] += r["word_count"]
            i += 1
            continue
        if i + 1 < len(runs):
            runs[i + 1]["text"] = (r["text"] + " " + runs[i + 1]["text"]).strip()
            runs[i + 1]["word_count"] += r["word_count"]
            i += 1
            continue
        out.append(r); i += 1
    return out

def dash_format_if_needed(text: str, cue_words: List[Dict[str, Any]], rules: Dict[str, Any]) -> str:
    """If two meaningful speakers are inside one cue, show dashes on each line (no names)."""
    runs = split_into_speaker_runs(cue_words, int(rules["minWordsPerRun"]))
    # Need 2 distinct speakers
    speakers = [r.get("speaker") for r in runs if r.get("speaker") is not None]
    distinct = []
    for s in speakers:
        if s not in distinct:
            distinct.append(s)
    if len(distinct) < 2:
        return text

    # Use first two runs only
    r1 = runs[0]["text"]
    r2 = runs[1]["text"] if len(runs) > 1 else ""

    max_chars = int(rules["maxCharsPerLine"])
    l1 = wrap_text(r1, max_chars - 2, 1, bool(rules["preferPunctuationBreaks"]))
    l2 = wrap_text(r2, max_chars - 2, 1, bool(rules["preferPunctuationBreaks"]))
    if l2:
        return f"- {l1}\n- {l2}"
    return f"- {l1}"

# ============================================================
# TIME FORMATTERS (ms-based, since AssemblyAI words are ms)
# ============================================================

def ms_to_srt_time(ms: int) -> str:
    if ms < 0:
        ms = 0
    hh = ms // 3600000; ms -= hh * 3600000
    mm = ms // 60000; ms -= mm * 60000
    ss = ms // 1000; ms -= ss * 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

def ms_to_vtt_time(ms: int) -> str:
    if ms < 0:
        ms = 0
    hh = ms // 3600000; ms -= hh * 3600000
    mm = ms // 60000; ms -= mm * 60000
    ss = ms // 1000; ms -= ss * 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"

def cues_to_srt(cues: List[Dict[str, Any]]) -> str:
    out: List[str] = []
    for i, c in enumerate(cues, 1):
        out.append(str(i))
        out.append(f"{ms_to_srt_time(int(c['start']))} --> {ms_to_srt_time(int(c['end']))}")
        out.append(c["text"])
        out.append("")
    return "\n".join(out).rstrip() + "\n"

def cues_to_vtt(cues: List[Dict[str, Any]]) -> str:
    out = ["WEBVTT", ""]
    for c in cues:
        out.append(f"{ms_to_vtt_time(int(c['start']))} --> {ms_to_vtt_time(int(c['end']))}")
        out.append(c["text"])
        out.append("")
    return "\n".join(out).rstrip() + "\n"

# ============================================================
# CORE SEGMENTATION ENGINE (words -> cues)
# ============================================================

def build_dialogue_stream(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop SDH bracket tokens; keep spoken tokens."""
    out = []
    for w in words:
        t = str(w.get("text", "")).strip()
        if not t:
            continue
        if is_sdh_word(w):
            continue
        out.append(w)
    return out

def try_build_cue_from_words(chunk: List[Dict[str, Any]], next_start_ms: Optional[int], rules: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not chunk:
        return None

    max_chars = int(rules["maxCharsPerLine"])
    max_lines = int(rules["maxLines"])
    max_cps = float(rules["maxCPS"])
    min_dur = int(rules["minDurationMs"])
    max_dur = int(rules["maxDurationMs"])
    min_gap = int(rules["minGapMs"])
    tail_pad = int(rules.get("tailPadMs", 0))
    prefer_punct = bool(rules["preferPunctuationBreaks"])

    start = int(chunk[0].get("start", 0))
    end_raw = int(chunk[-1].get("end", start))

    # Allow small tail pad, but never violate min gap to next cue start
    end = end_raw + tail_pad
    if next_start_ms is not None:
        end = min(end, max(start, next_start_ms - min_gap))

    # Duration clamps
    dur = end - start
    if dur > max_dur:
        # too long: caller should split earlier
        return None
    if dur < min_dur and next_start_ms is not None:
        # try extending to min duration if possible
        end = min(start + min_dur, max(start, next_start_ms - min_gap))
        dur = end - start

    # Text
    text_raw = " ".join(str(w.get("text", "")).strip() for w in chunk).strip()
    wrapped = wrap_text(text_raw, max_chars, max_lines, prefer_punct=prefer_punct)
    wrapped = dash_format_if_needed(wrapped, chunk, rules)

    # Enforce line lengths strictly
    for line in wrapped.split("\n"):
        if len(line) > max_chars:
            return None

    # CPS check
    cps = calc_cps(wrapped, max(dur, 1))
    if cps > max_cps:
        return None

    return {
        "start": start,
        "end": max(end, start + 1),
        "text": wrapped,
        "speaker": None,
        "type": "caption",
    }

def segment_dialogue(words: List[Dict[str, Any]], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Greedy segmentation with backoff to satisfy CPS/lines/duration/gap."""
    if not words:
        return []

    max_dur = int(rules["maxDurationMs"])
    min_gap = int(rules["minGapMs"])

    out: List[Dict[str, Any]] = []

    i = 0
    while i < len(words):
        # Start a chunk at i
        chunk: List[Dict[str, Any]] = []
        j = i
        # Greedily add words; if we fail constraints, back off.
        while j < len(words):
            chunk.append(words[j])
            start = int(chunk[0].get("start", 0))
            end = int(chunk[-1].get("end", start))
            if end - start > max_dur:
                chunk.pop()
                break
            # We'll decide with next_start when we finalize
            j += 1

        # Now we have a max-duration chunk candidate ending at j-1.
        # We need the *largest* prefix that passes constraints.
        best = None
        best_k = None

        # next start for timing caps (word after the chunk we test)
        for k in range(len(chunk), 0, -1):
            cand = chunk[:k]
            next_start = int(words[i + k].get("start", 0)) if (i + k) < len(words) else None
            cue = try_build_cue_from_words(cand, next_start, rules)
            if cue:
                best = cue
                best_k = k
                break

        if not best:
            # As a last resort, force a 1-word cue with minimal formatting
            w = words[i]
            next_start = int(words[i + 1].get("start", 0)) if (i + 1) < len(words) else None
            cue = try_build_cue_from_words([w], next_start, rules)
            if cue:
                best = cue
                best_k = 1
            else:
                # If even that fails, emit raw (should be very rare)
                start = int(w.get("start", 0)); end = int(w.get("end", start + 1))
                best = {"start": start, "end": end, "text": str(w.get("text","")).strip(), "speaker": None, "type": "caption"}
                best_k = 1

        # Enforce min gap with previous by trimming previous end if needed
        if out:
            prev = out[-1]
            gap = int(best["start"]) - int(prev["end"])
            if gap < min_gap:
                prev["end"] = max(int(prev["start"]) + 1, int(best["start"]) - min_gap)

        out.append(best)
        i += int(best_k)

    return out

def insert_sdh(dialogue: List[Dict[str, Any]], sdh: List[Dict[str, Any]], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not sdh:
        return dialogue
    min_gap = int(rules["minGapMs"])
    merged = sorted(dialogue, key=lambda c: (int(c["start"]), int(c["end"])))

    for s in sdh:
        s_start = int(s["start"]); s_end = int(s["end"])
        # Find insertion spot
        idx = 0
        while idx < len(merged) and int(merged[idx]["start"]) <= s_start:
            idx += 1

        prev_end = int(merged[idx-1]["end"]) if idx-1 >= 0 else -10**9
        next_start = int(merged[idx]["start"]) if idx < len(merged) else None

        s_start = max(s_start, prev_end + min_gap)
        if next_start is not None:
            s_end = min(s_end, next_start - min_gap)
        if s_end <= s_start:
            continue

        merged.append({**s, "start": s_start, "end": s_end, "type": "caption"})
        merged.sort(key=lambda c: (int(c["start"]), int(c["end"])))

    return merged

def build_nbcu_captions(transcript_id: str, request_rules: Dict[str, Any]) -> Dict[str, Any]:
    words = sorted(aa_get_words(transcript_id), key=lambda w: int(w.get("start", 0)))
    rules = get_rules(request_rules)

    # SDH from bracket tokens
    sdh_events = build_sdh_events(words, rules)

    # Dialogue stream (excluding SDH tokens)
    dialogue_words = build_dialogue_stream(words)

    # Build rule-compliant dialogue cues
    dialogue_cues = segment_dialogue(dialogue_words, rules)

    # Insert SDH cues into timeline safely
    merged = insert_sdh(dialogue_cues, sdh_events, rules)

    # Final sort + cleanup
    merged.sort(key=lambda c: (int(c["start"]), int(c["end"])))
    cleaned: List[Dict[str, Any]] = []
    for c in merged:
        s = int(c["start"]); e = int(c["end"])
        if e <= s:
            continue
        cleaned.append(c)

    exports = {
        "srt": cues_to_srt(cleaned),
        "vtt": cues_to_vtt(cleaned),
    }

    return {
        "transcriptId": transcript_id,
        "rules": rules,
        "cues": cleaned,
        "exports": exports,
    }

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return {"ok": True, "timestamp": now_iso()}

@app.get("/cors-test")
def cors_test():
    return {"ok": True}

@app.post("/v1/jobs", response_model=JobResponse)
def create_job(req: CreateJobRequest):
    title = (req.title or "Untitled").strip()
    rules = get_rules(req.rules)
    rules_json = json.dumps(rules)

    transcript_id = aa_create_transcript(
        req.mediaUrl,
        speaker_labels=bool(req.speaker_labels),
        language_detection=bool(req.language_detection),
    )

    job_id = str(uuid4())
    created = now_iso()

    with db() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO jobs
            (id, transcript_id, created_at, updated_at, title, media_url, status, error,
             rules_json, result_json, srt, vtt)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, transcript_id, created, created, title, req.mediaUrl, "queued", None, rules_json, None, None, None),
        )
        conn.commit()

    return JobResponse(
        id=job_id,
        createdAt=created,
        updatedAt=created,
        title=title,
        mediaUrl=req.mediaUrl,
        status="queued",
        error=None,
    )

@app.get("/v1/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str):
    now = now_iso()

    with db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    status = (row["status"] or "queued").lower()
    title = row["title"] or "Untitled"
    created_at = row["created_at"] or now
    updated_at = row["updated_at"] or row["created_at"] or now
    media_url = row["media_url"]
    transcript_id = row["transcript_id"] or job_id
    error = row["error"]

    terminal = {"completed", "failed", "error"}

    live_status = None
    live = None
    if status not in terminal:
        try:
            live = aa_get_transcript(transcript_id)
            live_status = (live.get("status") or "").lower() or None
        except Exception as e:
            print(f"[get_job] AssemblyAI refresh failed for {job_id}: {e}")
            live_status = None

    if live_status == "queued":
        status = "queued"; error = None
    elif live_status == "processing":
        status = "processing"; error = None
    elif live_status == "completed":
        status = "completed"; error = None
    elif live_status == "error":
        status = "failed"; error = live.get("error") if isinstance(live, dict) else "AssemblyAI error"

    exports = None
    if status == "completed":
        if row["srt"] and row["vtt"] and row["result_json"]:
            exports = {
                "srt": row["srt"],
                "vtt": row["vtt"],
                "result": json.loads(row["result_json"]) if row["result_json"] else None,
            }
        else:
            try:
                rules = json.loads(row["rules_json"] or "{}")
            except Exception:
                rules = get_rules(None)

            result_obj = build_nbcu_captions(transcript_id, rules)

            with db() as conn:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = ?, error = ?, updated_at = ?, srt = ?, vtt = ?, result_json = ?
                    WHERE id = ?
                    """,
                    ("completed", None, now, result_obj["exports"]["srt"], result_obj["exports"]["vtt"], json.dumps(result_obj), job_id),
                )
                conn.commit()

            exports = {"srt": result_obj["exports"]["srt"], "vtt": result_obj["exports"]["vtt"], "result": result_obj}

    if live_status is not None:
        if row["status"] != status or row["error"] != error:
            with db() as conn:
                conn.execute(
                    "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                    (status, error, now, job_id),
                )
                conn.commit()
            updated_at = now

    return JobResponse(
        id=job_id,
        createdAt=created_at,
        updatedAt=updated_at,
        title=title,
        mediaUrl=media_url,
        status=status,
        error=error,
        exports=exports,
    )
