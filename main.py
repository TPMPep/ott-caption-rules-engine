# ============================================================
# AI CC CREATOR – NBCU-STYLE BROADCAST ENGINE (FIXED)
# SRT-first timing + NBCU rules + SDH + speaker dash logic
# ============================================================

import os
import re
import json
import math
import sqlite3
from uuid import uuid4
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, PlainTextResponse
from pydantic import BaseModel

# ============================================================
# CONFIG
# ============================================================

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").strip()
BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()
DB_PATH = os.getenv("SQLITE_PATH", "jobs.db")
POLL_HINT_SECONDS = int(os.getenv("POLL_HINT_SECONDS", "3"))

AA_BASE = "https://api.assemblyai.com/v2"

# ============================================================
# FASTAPI SETUP
# ============================================================

app = FastAPI(title="AI CC Creator API", version="NBCU-Fixed")

# CORS
allow_origins = ["*"] if not ALLOWED_ORIGINS else [o.strip() for o in ALLOWED_ORIGINS.split(",") if o.strip()]
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
# DATABASE
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
# REQUEST / RESPONSE MODELS
# ============================================================

class CreateJobRequest(BaseModel):
    title: Optional[str] = None
    mediaUrl: str
    speaker_labels: Optional[bool] = True
    language_detection: Optional[bool] = True
    allow_http: Optional[bool] = False
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
# ASSEMBLYAI CLIENT
# ============================================================

def aa_headers() -> Dict[str, str]:
    api_key = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
    if not api_key:
        # We'll fail on create_job, but keep server running.
        api_key = ASSEMBLYAI_API_KEY
    return {"authorization": api_key, "content-type": "application/json"}

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

    # Encourage bracketed SDH tokens in transcript text (best-effort)
    if "universal-3-pro" in speech_models:
        payload["prompt"] = (
            "Transcribe dialogue accurately. Include SDH-style non-speech sound cues in ALL CAPS "
            "inside brackets as standalone tokens, e.g. [♪ MUSIC ♪], [APPLAUSE], [LAUGHTER], [DOOR SLAMS]."
        )

    if webhook_url:
        payload["webhook_url"] = webhook_url
        if webhook_secret:
            payload["webhook_auth_header_name"] = "x-webhook-token"
            payload["webhook_auth_header_value"] = webhook_secret

    r = requests.post(f"{AA_BASE}/transcript", headers=aa_headers(), json=payload, timeout=60)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"AssemblyAI create transcript failed: {r.status_code} {r.text}")

    transcript_id = (r.json() or {}).get("id")
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
# NBCU RULES (DEFAULTS)
# ============================================================

def get_rules(req_rules: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    NBCU-ish defaults (as per your stated target):
      - 32 chars/line
      - 2 lines max
      - 17 CPS max
      - min duration 1.0s
      - max duration 6.0s
      - min gap 80ms
    """
    r = dict(req_rules or {})

    r.setdefault("maxCharsPerLine", 32)
    r.setdefault("maxLines", 2)
    r.setdefault("maxCPS", 17)

    r.setdefault("minDurationMs", 1000)
    r.setdefault("maxDurationMs", 6000)
    r.setdefault("minGapMs", 80)

    r.setdefault("preferPunctuationBreaks", True)
    r.setdefault("startAtHour00", True)

    # speaker/run + SDH tuning
    r.setdefault("minWordsPerRun", 2)
    r.setdefault("sdhMinCueDurationMs", 600)
    r.setdefault("sdhDedupWindowMs", 250)

    return r

def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))

# ============================================================
# TIME / FORMAT UTILS
# ============================================================

SRT_BLOCK_SPLIT = re.compile(r"\n\s*\n", re.MULTILINE)
SRT_TIME_PATTERN = re.compile(r"(\d\d:\d\d:\d\d[,.]\d\d\d)\s*-->\s*(\d\d:\d\d:\d\d[,.]\d\d\d)")

def srt_time_to_seconds(t: str) -> float:
    t = t.replace(",", ".")
    hh, mm, rest = t.split(":")
    ss, ms = rest.split(".")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0

def seconds_to_srt_time(x: float) -> str:
    if x < 0:
        x = 0.0
    hh = int(x // 3600); x -= hh * 3600
    mm = int(x // 60); x -= mm * 60
    ss = int(x)
    ms = int(round((x - ss) * 1000.0))
    if ms == 1000:
        ss += 1; ms = 0
    if ss == 60:
        mm += 1; ss = 0
    if mm == 60:
        hh += 1; mm = 0
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

def parse_srt(srt_text: str) -> List[Dict[str, Any]]:
    blocks = SRT_BLOCK_SPLIT.split((srt_text or "").strip())
    cues: List[Dict[str, Any]] = []
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue
        time_line = lines[1].strip()
        m = SRT_TIME_PATTERN.match(time_line)
        if not m:
            continue
        start = srt_time_to_seconds(m.group(1))
        end = srt_time_to_seconds(m.group(2))
        text = "\n".join(lines[2:]).strip()
        cues.append({"start": start, "end": end, "text": text, "type": "dialogue"})
    return cues

def cues_to_srt(cues: List[Dict[str, Any]]) -> str:
    out: List[str] = []
    for i, c in enumerate(cues, 1):
        out.append(str(i))
        out.append(f"{seconds_to_srt_time(float(c['start']))} --> {seconds_to_srt_time(float(c['end']))}")
        out.append(str(c["text"]))
        out.append("")
    return "\n".join(out).rstrip() + "\n"

def cues_to_vtt(cues: List[Dict[str, Any]]) -> str:
    def vtt_time(x: float) -> str:
        hh = int(x // 3600); x -= hh * 3600
        mm = int(x // 60); x -= mm * 60
        ss = int(x)
        ms = int(round((x - ss) * 1000.0))
        return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"

    lines = ["WEBVTT", ""]
    for c in cues:
        lines.append(f"{vtt_time(float(c['start']))} --> {vtt_time(float(c['end']))}")
        lines.append(str(c["text"]))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

# ============================================================
# WRAP ENGINE (32 chars/line, 2 lines)
# ============================================================

_PUNCT_BREAK = re.compile(r"[,\.;:\!\?]$")

def wrap_text_to_lines(text: str, max_chars: int, max_lines: int, prefer_punct: bool = True) -> str:
    raw = re.sub(r"\s+", " ", (text or "").replace("\n", " ").strip())
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
            lines.append(" ".join(cur).strip())
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

    if len(lines) < max_lines and cur:
        flush()

    lines = lines[:max_lines]
    return "\n".join(lines)

def calc_cps(text: str, start: float, end: float) -> float:
    dur = max(0.001, float(end) - float(start))
    chars = len((text or "").replace("\n", ""))
    return chars / dur

# ============================================================
# WORD WINDOW + SPEAKER RUNS
# ============================================================

_WORD_CLEAN = re.compile(r"^[\W_]+|[\W_]+$")
def normalize_word(word: str) -> str:
    word = (word or "").strip().lower()
    return _WORD_CLEAN.sub("", word)

def words_in_window(words: List[Dict[str, Any]], start: float, end: float) -> List[Dict[str, Any]]:
    if not words:
        return []
    s_ms = int(float(start) * 1000)
    e_ms = int(float(end) * 1000)
    out: List[Dict[str, Any]] = []
    for w in words:
        ws = int(w.get("start", 0))
        we = int(w.get("end", 0))
        if we <= s_ms:
            continue
        if ws >= e_ms:
            break
        out.append(w)
    return out

def _is_real_word_token(txt: str) -> bool:
    if not txt:
        return False
    return bool(normalize_word(txt))

def build_speaker_runs(cue_words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    cur_speaker = None
    cur_words: List[Dict[str, Any]] = []

    def flush():
        nonlocal cur_words, cur_speaker
        if not cur_words:
            return
        rs = float(cur_words[0].get("start", 0)) / 1000.0
        re_ = float(cur_words[-1].get("end", 0)) / 1000.0
        txt = " ".join([str(w.get("text", "")).strip() for w in cur_words]).strip()
        wc = sum(1 for w in cur_words if _is_real_word_token(str(w.get("text", ""))))
        runs.append({"speaker": cur_speaker, "words": cur_words, "text": txt, "start": rs, "end": re_, "word_count": wc})
        cur_words = []

    for w in cue_words:
        spk = w.get("speaker", None)
        txt = str(w.get("text", "")).strip()
        if not txt:
            continue
        if cur_speaker is None and not cur_words:
            cur_speaker = spk
            cur_words = [w]
            continue
        if spk == cur_speaker:
            cur_words.append(w)
        else:
            flush()
            cur_speaker = spk
            cur_words = [w]

    flush()
    return runs

def split_runs_meaningfully(runs: List[Dict[str, Any]], min_words_per_run: int) -> List[Dict[str, Any]]:
    if not runs:
        return []
    merged: List[Dict[str, Any]] = []
    i = 0
    while i < len(runs):
        r = runs[i]
        if r["word_count"] >= min_words_per_run:
            merged.append(r)
            i += 1
            continue

        if merged:
            prev = merged[-1]
            prev["words"].extend(r["words"])
            prev["text"] = (prev["text"] + " " + r["text"]).strip()
            prev["end"] = max(prev["end"], r["end"])
            prev["word_count"] = prev["word_count"] + r["word_count"]
            i += 1
            continue

        if i + 1 < len(runs):
            nxt = runs[i + 1]
            nxt["words"] = r["words"] + nxt["words"]
            nxt["text"] = (r["text"] + " " + nxt["text"]).strip()
            nxt["start"] = min(r["start"], nxt["start"])
            nxt["word_count"] = nxt["word_count"] + r["word_count"]
            i += 1
            continue

        merged.append(r)
        i += 1
    return merged

def count_distinct_speakers(runs: List[Dict[str, Any]]) -> int:
    seen = set()
    for r in runs:
        s = r.get("speaker", None)
        if s is None:
            continue
        seen.add(s)
    return len(seen)

def format_dash_two_lines(line1: str, line2: str) -> str:
    line1 = (line1 or "").strip()
    line2 = (line2 or "").strip()
    if line2:
        return f"- {line1}\n- {line2}"
    return f"- {line1}"

def apply_speaker_dash_if_needed(
    cue_words: List[Dict[str, Any]],
    original_text: str,
    rules: Dict[str, Any],
) -> str:
    """
    NBCU-style:
      - ONLY apply dashes if there are 2+ distinct speakers within the cue window.
      - Render as ONE cue with two dashed lines.
    """
    max_chars = int(rules.get("maxCharsPerLine", 32))
    prefer_punct = bool(rules.get("preferPunctuationBreaks", True))
    min_words_per_run = int(rules.get("minWordsPerRun", 2))

    runs = build_speaker_runs(cue_words)
    runs = split_runs_meaningfully(runs, min_words_per_run)

    if len(runs) >= 2 and count_distinct_speakers(runs) >= 2:
        r1 = runs[0]["text"]
        r2 = runs[1]["text"]
        # wrap each run to ONE line (because dash format already uses two lines)
        r1w = wrap_text_to_lines(r1, max_chars - 2, 1, prefer_punct=prefer_punct)
        r2w = wrap_text_to_lines(r2, max_chars - 2, 1, prefer_punct=prefer_punct)
        return format_dash_two_lines(r1w, r2w)

    # single speaker (or no speaker labels) => normal wrap
    return wrap_text_to_lines(original_text, max_chars, int(rules.get("maxLines", 2)), prefer_punct=prefer_punct)

# ============================================================
# DURATION / GAP ENFORCEMENT
# ============================================================

def enforce_duration_and_gap(cues: List[Dict[str, Any]], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not cues:
        return cues

    min_dur = float(rules.get("minDurationMs", 1000)) / 1000.0
    max_dur = float(rules.get("maxDurationMs", 6000)) / 1000.0
    min_gap = float(rules.get("minGapMs", 80)) / 1000.0

    out = [dict(c) for c in cues]
    out.sort(key=lambda x: (float(x["start"]), float(x["end"])))

    for i in range(len(out)):
        c = out[i]
        start = float(c["start"])
        end = float(c["end"])
        if end < start:
            end = start

        dur = end - start

        if dur > max_dur:
            end = start + max_dur
            dur = end - start

        if dur < min_dur:
            desired_end = start + min_dur
            if i + 1 < len(out):
                next_start = float(out[i + 1]["start"])
                cap_end = max(start, next_start - min_gap)
                end = min(desired_end, cap_end)
            else:
                end = desired_end

        if i + 1 < len(out):
            next_start = float(out[i + 1]["start"])
            if end > next_start - min_gap:
                end = max(start, next_start - min_gap)

        c["start"] = start
        c["end"] = end

    cleaned = []
    for c in out:
        if float(c["end"]) - float(c["start"]) <= 0.001:
            continue
        cleaned.append(c)
    return cleaned

# ============================================================
# CPS ENFORCEMENT (SPLIT CUES WHEN TOO FAST)
# ============================================================

def _fits_wrap(words: List[str], max_chars: int, max_lines: int, prefer_punct: bool) -> bool:
    txt = " ".join(words).strip()
    wrapped = wrap_text_to_lines(txt, max_chars, max_lines, prefer_punct=prefer_punct)
    # If wrap truncates, it will drop words; detect by comparing word sets roughly
    # Simple heuristic: ensure last word is present.
    if not words:
        return True
    return (words[-1] in wrapped) or (normalize_word(words[-1]) and normalize_word(words[-1]) in normalize_word(wrapped.split()[-1] if wrapped.split() else ""))

def split_text_to_wrap_chunks(
    text: str,
    rules: Dict[str, Any],
) -> List[str]:
    """
    Split text into chunks that each can wrap within maxCharsPerLine/maxLines
    WITHOUT truncating (best-effort).
    """
    max_chars = int(rules.get("maxCharsPerLine", 32))
    max_lines = int(rules.get("maxLines", 2))
    prefer_punct = bool(rules.get("preferPunctuationBreaks", True))

    raw = re.sub(r"\s+", " ", (text or "").replace("\n", " ").strip())
    if not raw:
        return []

    words = raw.split(" ")
    chunks: List[List[str]] = []
    cur: List[str] = []

    for w in words:
        test = cur + [w]
        if not cur:
            cur = [w]
            continue

        # Try to see if adding would likely break wrap constraints.
        if _fits_wrap(test, max_chars, max_lines, prefer_punct):
            cur = test
        else:
            chunks.append(cur)
            cur = [w]

    if cur:
        chunks.append(cur)

    # Convert to strings and wrap each
    out = []
    for ch in chunks:
        out.append(wrap_text_to_lines(" ".join(ch), max_chars, max_lines, prefer_punct=prefer_punct))
    return out

def enforce_cps_by_splitting(
    cue: Dict[str, Any],
    rules: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    If cps > maxCPS, split cue into multiple cues within the same time window.
    We keep timing window (start/end) but allocate sub-windows respecting minGap/minDuration as possible.
    """
    max_cps = float(rules.get("maxCPS", 17))
    min_gap = float(rules.get("minGapMs", 80)) / 1000.0
    min_dur = float(rules.get("minDurationMs", 1000)) / 1000.0

    start = float(cue["start"])
    end = float(cue["end"])
    text = str(cue.get("text", "") or "").strip()

    if not text:
        return [cue]

    cps_val = calc_cps(text, start, end)
    if cps_val <= max_cps:
        return [cue]

    # Split by wrap constraints first
    chunks = split_text_to_wrap_chunks(text, rules)
    if len(chunks) <= 1:
        # fallback: split by words evenly
        raw = re.sub(r"\s+", " ", text.replace("\n", " ").strip())
        words = raw.split(" ")
        if len(words) <= 3:
            return [cue]
        mid = len(words) // 2
        chunks = [
            wrap_text_to_lines(" ".join(words[:mid]), int(rules.get("maxCharsPerLine", 32)), int(rules.get("maxLines", 2)), bool(rules.get("preferPunctuationBreaks", True))),
            wrap_text_to_lines(" ".join(words[mid:]), int(rules.get("maxCharsPerLine", 32)), int(rules.get("maxLines", 2)), bool(rules.get("preferPunctuationBreaks", True))),
        ]

    # Determine how many cues we can fit time-wise
    n = len(chunks)
    total_window = max(0.0, end - start)
    available = total_window - min_gap * (n - 1)

    if available <= 0:
        # no room for gaps; just return original
        return [cue]

    # Allocate durations proportional to character counts, but not below min_dur
    char_counts = [max(1, len(c.replace("\n", ""))) for c in chunks]
    total_chars = sum(char_counts)
    durations = [available * (cc / total_chars) for cc in char_counts]

    # Enforce min duration where possible
    # If enforcing min makes sum exceed available, we'll allow some to be under min (but try hard)
    if sum(durations) < n * min_dur and (n * min_dur) <= available:
        durations = [min_dur] * n
        leftover = available - n * min_dur
        # distribute leftover proportional to char counts
        for i in range(n):
            durations[i] += leftover * (char_counts[i] / total_chars)

    # Build subcues sequentially
    out: List[Dict[str, Any]] = []
    t = start
    for i in range(n):
        d = durations[i]
        s = t
        e = min(end, s + d)
        # Ensure last cue ends at end (if rounding drift)
        if i == n - 1:
            e = end

        # If too short, squeeze but keep monotonic
        if e - s < 0.001:
            continue

        out.append({
            "start": s,
            "end": e,
            "text": chunks[i],
            "type": cue.get("type", "dialogue"),
        })
        t = e + min_gap

    # Final sanity: if any still violate CPS badly, we keep them but report cps in result
    return out if out else [cue]

# ============================================================
# SDH EXTRACTION + INSERTION
# ============================================================

SDH_BRACKET_PATTERN = re.compile(r"\[(.*?)\]")
MUSIC_PATTERN = re.compile(r"music", re.IGNORECASE)

def normalize_sdh_token(raw: str) -> str:
    txt = (raw or "").strip()
    if MUSIC_PATTERN.search(txt):
        return "[♪ MUSIC ♪]"
    txt = txt.upper()
    txt = re.sub(r"\s+", " ", txt)
    if not txt.startswith("["):
        txt = "[" + txt
    if not txt.endswith("]"):
        txt = txt + "]"
    return txt

def align_sdh_to_words(transcript_text: str, words: List[Dict[str, Any]]) -> List[Tuple[float, str]]:
    if not transcript_text or not words:
        return []

    raw = re.sub(r"\s+", " ", transcript_text.replace("\n", " ").strip())

    tokens: List[str] = []
    i = 0
    while i < len(raw):
        if raw[i] == "[":
            j = raw.find("]", i + 1)
            if j != -1:
                tokens.append(raw[i:j+1])
                i = j + 1
                continue
        j = i
        while j < len(raw) and raw[j] not in [" ", "["]:
            j += 1
        tokens.append(raw[i:j])
        i = j + 1 if j < len(raw) and raw[j] == " " else j

    word_idx = 0
    aligned: List[Tuple[float, str]] = []

    def next_word_time() -> float:
        if word_idx < len(words):
            return float(words[word_idx].get("start", 0)) / 1000.0
        if words:
            return float(words[-1].get("end", 0)) / 1000.0
        return 0.0

    for tok in tokens:
        if not tok:
            continue

        if tok.startswith("[") and tok.endswith("]"):
            ts = next_word_time()
            aligned.append((ts, normalize_sdh_token(tok)))
            continue

        tn = normalize_word(tok)
        if not tn:
            continue

        for _ in range(6):
            if word_idx >= len(words):
                break
            wn = normalize_word(str(words[word_idx].get("text", "")))
            if wn == tn or (wn and tn and (wn.startswith(tn) or tn.startswith(wn))):
                word_idx += 1
                break
            word_idx += 1

    return aligned

def deduplicate_sdh_events(events: List[Tuple[float, str]], rules: Dict[str, Any]) -> List[Tuple[float, str]]:
    if not events:
        return []
    window = float(rules.get("sdhDedupWindowMs", 250)) / 1000.0
    events_sorted = sorted(events, key=lambda x: x[0])
    cleaned: List[Tuple[float, str]] = []
    for ts, txt in events_sorted:
        if not cleaned:
            cleaned.append((ts, txt))
            continue
        prev_ts, prev_txt = cleaned[-1]
        if txt == prev_txt and abs(ts - prev_ts) < window:
            continue
        cleaned.append((ts, txt))
    return cleaned

def build_sdh_candidates(transcript_text: str, words: List[Dict[str, Any]], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    aligned = align_sdh_to_words(transcript_text, words)
    deduped = deduplicate_sdh_events(aligned, rules)
    min_dur = float(rules.get("sdhMinCueDurationMs", 600)) / 1000.0
    out = []
    for ts, token in deduped:
        out.append({"start": float(ts), "end": float(ts) + min_dur, "text": token, "type": "sdh"})
    return out

def insert_sdh_cues(dialogue_cues: List[Dict[str, Any]], sdh: List[Dict[str, Any]], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not sdh:
        return dialogue_cues

    min_gap = float(rules.get("minGapMs", 80)) / 1000.0
    merged = sorted([dict(c) for c in dialogue_cues], key=lambda c: (float(c["start"]), float(c["end"])))

    for s in sdh:
        s_start = float(s["start"])
        s_end = float(s["end"])

        idx = 0
        while idx < len(merged) and float(merged[idx]["start"]) <= s_start:
            idx += 1

        prev_end = float(merged[idx - 1]["end"]) if idx - 1 >= 0 else 0.0
        next_start = float(merged[idx]["start"]) if idx < len(merged) else None

        s_start = max(s_start, prev_end + min_gap)
        if next_start is not None:
            s_end = min(s_end, next_start - min_gap)

        if s_end <= s_start:
            continue

        merged.append({"start": s_start, "end": s_end, "text": s["text"], "type": "sdh"})
        merged.sort(key=lambda c: (float(c["start"]), float(c["end"])))

    return merged

# ============================================================
# FULL NBCU CAPTION PIPELINE (THIS IS WHAT YOU WERE MISSING)
# ============================================================

def build_pro_captions(transcript_id: str, request_rules: Dict[str, Any]) -> Dict[str, Any]:
    transcript_json = aa_get_transcript(transcript_id)
    transcript_text = transcript_json.get("text", "") or ""
    srt_raw = aa_get_srt(transcript_id)
    words = sorted(aa_get_words(transcript_id), key=lambda w: w.get("start", 0))

    rules = get_rules(request_rules)

    base_cues = parse_srt(srt_raw)

    dialogue_cues: List[Dict[str, Any]] = []
    for cue in base_cues:
        start = float(cue["start"])
        end = float(cue["end"])
        original_text = str(cue["text"] or "")

        cue_words = words_in_window(words, start, end)

        # Speaker dash logic (one cue, two dashed lines only when needed)
        wrapped = apply_speaker_dash_if_needed(cue_words, original_text, rules)

        # Build cue
        dialogue_cues.append({"start": start, "end": end, "text": wrapped, "type": "dialogue"})

    # Sort + duration/gap
    dialogue_cues.sort(key=lambda c: (float(c["start"]), float(c["end"])))
    dialogue_cues = enforce_duration_and_gap(dialogue_cues, rules)

    # CPS enforcement (split too-fast cues)
    cps_fixed: List[Dict[str, Any]] = []
    for c in dialogue_cues:
        cps_fixed.extend(enforce_cps_by_splitting(c, rules))
    cps_fixed.sort(key=lambda c: (float(c["start"]), float(c["end"])))

    # Re-apply duration/gap after splitting
    cps_fixed = enforce_duration_and_gap(cps_fixed, rules)

    # SDH insertion
    sdh_candidates = build_sdh_candidates(transcript_text, words, rules)
    merged = insert_sdh_cues(cps_fixed, sdh_candidates, rules)
    merged.sort(key=lambda c: (float(c["start"]), float(c["end"])))

    # Final duration/gap (SDH included)
    merged = enforce_duration_and_gap(merged, rules)

    # Tag cps for reporting
    for c in merged:
        c["cps"] = round(calc_cps(str(c.get("text", "")), float(c["start"]), float(c["end"])), 2)

    srt_out = cues_to_srt(merged)
    vtt_out = cues_to_vtt(merged)

    result_json = {
        "transcriptId": transcript_id,
        "rules": rules,
        "cues": merged,
    }

    return {"result_json": result_json, "srt": srt_out, "vtt": vtt_out}

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return {"ok": True, "timestamp": now_iso()}

@app.get("/cors-test")
def cors_test():
    return {"ok": True}

# ------------------------------------------------------------
# CREATE JOB
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# JOB STATUS (POLL)
# ------------------------------------------------------------

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
    updated_at = row["updated_at"] or created_at or now
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
        status = "failed"
        error = live.get("error") if isinstance(live, dict) else "AssemblyAI error"

    exports = None

    # If completed: generate NBCU-grade exports (THIS IS THE KEY FIX)
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
                rules = {}

            built = build_pro_captions(transcript_id, rules)
            srt_text = built["srt"]
            vtt_text = built["vtt"]
            result_obj = built["result_json"]

            with db() as conn:
                conn.execute(
                    """
                    UPDATE jobs
                    SET status = ?, error = ?, updated_at = ?, srt = ?, vtt = ?, result_json = ?
                    WHERE id = ?
                    """,
                    ("completed", None, now, srt_text, vtt_text, json.dumps(result_obj), job_id),
                )
                conn.commit()

            exports = {"srt": srt_text, "vtt": vtt_text, "result": result_obj}

    # Persist status changes (even while processing) so polling is stable
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

# ------------------------------------------------------------
# DOWNLOAD ENDPOINTS
# ------------------------------------------------------------

@app.get("/v1/jobs/{job_id}/srt")
def download_srt(job_id: str):
    with db() as conn:
        row = conn.execute("SELECT srt FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row or not row["srt"]:
        raise HTTPException(status_code=404, detail="SRT not available (job not completed yet?)")
    return PlainTextResponse(
        content=row["srt"],
        media_type="application/x-subrip",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.srt"'},
    )

@app.get("/v1/jobs/{job_id}/vtt")
def download_vtt(job_id: str):
    with db() as conn:
        row = conn.execute("SELECT vtt FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if not row or not row["vtt"]:
        raise HTTPException(status_code=404, detail="VTT not available (job not completed yet?)")
    return PlainTextResponse(
        content=row["vtt"],
        media_type="text/vtt",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.vtt"'},
    )
