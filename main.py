# ============================================================
# AI CC CREATOR – FULL BROADCAST ENGINE
# SRT-first timing + Speaker segmentation + SDH + Rule-driven
# ============================================================

import os
import re
import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi.responses import Response, JSONResponse
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

# ============================================================
# CONFIG
# ============================================================

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").strip()
BASE_URL = os.getenv("PUBLIC_BASE_URL", "").strip()
DB_PATH = os.getenv("SQLITE_PATH", "jobs.db")
POLL_HINT_SECONDS = int(os.getenv("POLL_HINT_SECONDS", "3"))

if not ASSEMBLYAI_API_KEY:
    raise RuntimeError("ASSEMBLYAI_API_KEY not configured")

# ============================================================
# FASTAPI SETUP
# ============================================================

app = FastAPI(title="AI CC Creator API", version="Broadcast-Full")

# --- CORS (MUST be configured BEFORE endpoints) ---
allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "").strip()

# If ALLOWED_ORIGINS is empty, allow all (prevents "Failed to fetch" while debugging)
allow_origins = (
    ["*"]
    if not allowed_origins_env
    else [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
)

# Base44 preview/production domains can vary, so regex helps a lot.
# This does NOT open you up to random sites unless you set ALLOWED_ORIGINS="*".
allow_origin_regex = r"^https:\/\/.*\.base44\.app$"

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

# Keep a sanitized list for manual CORS headers (useful on 500s/crashes)
_ALLOWED_ORIGINS_LIST = [o for o in allow_origins if o != "*"]
_ALLOWED_ORIGIN_REGEX = re.compile(allow_origin_regex) if allow_origin_regex else None

def _maybe_add_cors_headers(response: Response, request: Request) -> Response:
    """Ensure CORS headers are present even on error responses."""
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
        response.headers["Access-Control-Allow-Headers"] = request.headers.get(
            "access-control-request-headers", "*"
        )
    return response


# ============================================================
# DATABASE
# ============================================================

def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
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
        )
        """)
        conn.commit()

def now_iso():
    return datetime.now(timezone.utc).isoformat()

init_db()

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
def aa_create_transcript(audio_url: str, speaker_labels: bool = True, **kwargs) -> str:
    """
    Creates an AssemblyAI transcript and returns transcript_id.

    Accepts extra kwargs (language_detection, allow_http, webhook_url, webhook_secret, speech_models)
    so the API won't crash if the frontend sends new fields.
    """
    api_key = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="ASSEMBLYAI_API_KEY is not configured")

    language_detection = bool(kwargs.get("language_detection", True))
    allow_http = bool(kwargs.get("allow_http", False))
    webhook_url = kwargs.get("webhook_url") or ""
    webhook_secret = kwargs.get("webhook_secret") or ""
    speech_models = kwargs.get("speech_models") or ["universal-2"]  # required by AssemblyAI now

    payload: Dict[str, Any] = {
        "audio_url": audio_url,
        "speaker_labels": bool(speaker_labels),
        "language_detection": language_detection,
        "allow_http": allow_http,
        "speech_models": speech_models,
        "prompt": (
            "Transcribe dialogue accurately. Also include SDH-style non-speech sound cues in ALL CAPS "
            "inside brackets as standalone tokens, e.g. [♪ MUSIC ♪], [APPLAUSE], [LAUGHTER], [DOOR SLAMS]. "
            "Do not paraphrase sound cues; keep them short and standard."
        ),
    }

    # Optional webhook support
    if webhook_url:
        payload["webhook_url"] = webhook_url
        if webhook_secret:
            payload["webhook_auth_header_name"] = "x-webhook-token"
            payload["webhook_auth_header_value"] = webhook_secret

    r = requests.post(f"{AA_BASE}/transcript", headers=aa_headers(), json=payload, timeout=60)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"AssemblyAI create transcript failed: {r.status_code} {r.text}")

    transcript_id = r.json().get("id")
    if not transcript_id:
        raise HTTPException(status_code=502, detail="AssemblyAI create transcript failed: missing transcript id")

    return transcript_id

# ============================================================
# TIME UTILITIES
# ============================================================

def srt_time_to_seconds(t: str) -> float:
    """
    Converts SRT time format HH:MM:SS,mmm or HH:MM:SS.mmm to seconds.
    """
    t = t.replace(",", ".")
    hh, mm, rest = t.split(":")
    ss, ms = rest.split(".")
    return (
        int(hh) * 3600
        + int(mm) * 60
        + int(ss)
        + int(ms) / 1000.0
    )


def seconds_to_srt_time(x: float) -> str:
    """
    Converts seconds to SRT time format HH:MM:SS,mmm.
    """
    if x < 0:
        x = 0.0

    hh = int(x // 3600)
    x -= hh * 3600

    mm = int(x // 60)
    x -= mm * 60

    ss = int(x)
    ms = int(round((x - ss) * 1000.0))

    # prevent overflow like 1000ms
    if ms == 1000:
        ss += 1
        ms = 0

    if ss == 60:
        mm += 1
        ss = 0

    if mm == 60:
        hh += 1
        mm = 0

    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


# ============================================================
# SRT PARSING
# ============================================================

SRT_BLOCK_SPLIT = re.compile(r"\n\s*\n", re.MULTILINE)
SRT_TIME_PATTERN = re.compile(
    r"(\d\d:\d\d:\d\d[,.]\d\d\d)\s*-->\s*(\d\d:\d\d:\d\d[,.]\d\d\d)"
)

def parse_srt(srt_text: str) -> List[Dict[str, Any]]:
    """
    Parses AssemblyAI SRT output into cue list.
    We treat SRT as timing authority.
    """
    blocks = SRT_BLOCK_SPLIT.split(srt_text.strip())
    cues: List[Dict[str, Any]] = []

    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 3:
            continue

        time_line = lines[1].strip()
        match = SRT_TIME_PATTERN.match(time_line)
        if not match:
            continue

        start = srt_time_to_seconds(match.group(1))
        end = srt_time_to_seconds(match.group(2))

        text = "\n".join(lines[2:]).strip()

        cues.append({
            "start": start,
            "end": end,
            "text": text,
            "type": "dialogue"
        })

    return cues


def cues_to_srt(cues: List[Dict[str, Any]]) -> str:
    """
    Serializes cues back to SRT.
    """
    output_lines: List[str] = []

    for i, cue in enumerate(cues, 1):
        output_lines.append(str(i))
        output_lines.append(
            f"{seconds_to_srt_time(cue['start'])} --> {seconds_to_srt_time(cue['end'])}"
        )
        output_lines.append(cue["text"])
        output_lines.append("")

    return "\n".join(output_lines).rstrip() + "\n"


def cues_to_vtt(cues: List[Dict[str, Any]]) -> str:
    """
    Converts cues to WEBVTT format.
    """
    def vtt_time(x: float) -> str:
        hh = int(x // 3600)
        x -= hh * 3600
        mm = int(x // 60)
        x -= mm * 60
        ss = int(x)
        ms = int(round((x - ss) * 1000.0))
        return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"

    lines = ["WEBVTT", ""]

    for cue in cues:
        lines.append(f"{vtt_time(cue['start'])} --> {vtt_time(cue['end'])}")
        lines.append(cue["text"])
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ============================================================
# TEXT NORMALIZATION
# ============================================================

_WORD_CLEAN = re.compile(r"^[\W_]+|[\W_]+$")

def normalize_word(word: str) -> str:
    """
    Normalizes words for alignment comparison.
    """
    word = word.strip().lower()
    word = _WORD_CLEAN.sub("", word)
    return word
# ============================================================
# WORD WINDOW SELECTION (OVERLAP-BASED)
# ============================================================

def words_in_window(words: List[Dict[str, Any]], start: float, end: float) -> List[Dict[str, Any]]:
    """
    Returns words that overlap [start, end] (in seconds).
    Uses overlap test rather than strict containment to avoid missing boundary words.
    AssemblyAI word times are ms ints in "start"/"end".
    """
    if not words:
        return []

    s_ms = int(start * 1000)
    e_ms = int(end * 1000)

    out: List[Dict[str, Any]] = []
    # words are expected sorted by start time
    for w in words:
        ws = int(w.get("start", 0))
        we = int(w.get("end", 0))

        if we <= s_ms:
            continue
        if ws >= e_ms:
            break

        out.append(w)

    return out


# ============================================================
# SPEAKER RUNS + MEANINGFUL CHANGE DETECTION
# ============================================================

def _is_real_word_token(txt: str) -> bool:
    """
    Filters out empty tokens; AssemblyAI sometimes includes punctuation-only tokens.
    """
    if not txt:
        return False
    n = normalize_word(txt)
    return bool(n)


def build_speaker_runs(cue_words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Builds contiguous speaker runs from words in a cue window.

    Returns list of runs:
      {
        "speaker": <speaker_id or None>,
        "words": [word_dicts...],
        "text": "joined text",
        "start": seconds,
        "end": seconds,
        "word_count": int
      }
    """
    runs: List[Dict[str, Any]] = []
    cur_speaker = None
    cur_words: List[Dict[str, Any]] = []

    def flush():
        nonlocal cur_speaker, cur_words, runs
        if not cur_words:
            return
        # compute boundaries from words
        rs = float(cur_words[0].get("start", 0)) / 1000.0
        re_ = float(cur_words[-1].get("end", 0)) / 1000.0
        txt = " ".join([str(w.get("text", "")).strip() for w in cur_words]).strip()
        wc = sum(1 for w in cur_words if _is_real_word_token(str(w.get("text", ""))))
        runs.append({
            "speaker": cur_speaker,
            "words": cur_words,
            "text": txt,
            "start": rs,
            "end": re_,
            "word_count": wc,
        })
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


def split_runs_meaningfully(
    runs: List[Dict[str, Any]],
    min_words_per_run: int,
) -> List[Dict[str, Any]]:
    """
    Collapses tiny "blips" so we don't over-split when labels jitter.
    Strategy:
      - If a run has fewer than min_words_per_run real words,
        merge it into the previous run if possible; otherwise merge into next.
    """
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

        # tiny run: merge
        if merged:
            # merge into previous
            prev = merged[-1]
            prev["words"].extend(r["words"])
            prev["text"] = (prev["text"] + " " + r["text"]).strip()
            prev["end"] = max(prev["end"], r["end"])
            prev["word_count"] = prev["word_count"] + r["word_count"]
            i += 1
            continue

        # no previous: merge into next if exists
        if i + 1 < len(runs):
            nxt = runs[i + 1]
            nxt["words"] = r["words"] + nxt["words"]
            nxt["text"] = (r["text"] + " " + nxt["text"]).strip()
            nxt["start"] = min(r["start"], nxt["start"])
            nxt["word_count"] = nxt["word_count"] + r["word_count"]
            i += 1
            continue

        # only run
        merged.append(r)
        i += 1

    return merged


def count_distinct_speakers(runs: List[Dict[str, Any]]) -> int:
    seen = []
    for r in runs:
        s = r.get("speaker", None)
        if s is None:
            continue
        if s not in seen:
            seen.append(s)
    return len(seen)


# ============================================================
# DASH FORMATTING (NBCU STYLE)
# - No names.
# - Only add '-' when there are 2 meaningful speaker runs in same cue instance.
# - If 3+ speakers, still only show first 2 runs (broadcast constraint).
# ============================================================

def format_dash_two_runs(line1: str, line2: str) -> str:
    line1 = line1.strip()
    line2 = line2.strip()
    if line2:
        return f"- {line1}\n- {line2}"
    return f"- {line1}"


def apply_dash_if_needed(
    wrapped_text: str,
    runs: List[Dict[str, Any]],
    rules: Dict[str, Any],
) -> str:
    """
    If two meaningful speaker runs exist inside this cue, represent as:
      - <run1>
      - <run2>

    Otherwise, return wrapped_text unchanged (NO dash spam).
    """
    max_chars = int(rules.get("maxCharsPerLine", 32))

    # must have at least 2 runs with meaningful word_count already enforced
    if len(runs) < 2:
        return wrapped_text

    # If speaker labels missing, don't dash
    if count_distinct_speakers(runs) < 2:
        return wrapped_text

    r1 = runs[0]["text"]
    r2 = runs[1]["text"]

    # wrap each run to 1 line to preserve 2-line dash format
    r1w = wrap_text_to_lines(r1, max_chars - 2, 1, prefer_punct=bool(rules.get("preferPunctuationBreaks", True)))
    r2w = wrap_text_to_lines(r2, max_chars - 2, 1, prefer_punct=bool(rules.get("preferPunctuationBreaks", True)))

    return format_dash_two_runs(r1w, r2w)
# ============================================================
# RULE HELPERS + SAFE CLAMPS
# ============================================================

def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def get_rules(req_rules: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Returns a sanitized rules dict with defaults (NBCU-ish).
    These are *defaults only* — Base44 can override per job.
    """
    r = dict(req_rules or {})

    # Base44 fields (per your UI)
    r.setdefault("maxCharsPerLine", 32)
    r.setdefault("maxLines", 2)
    r.setdefault("maxCPS", 20)

    r.setdefault("minDurationMs", 1000)
    r.setdefault("maxDurationMs", 7000)
    r.setdefault("minGapMs", 80)

    r.setdefault("preferPunctuationBreaks", True)
    r.setdefault("sccFrameRate", 29.97)
    r.setdefault("startAtHour00", True)

    # Additional internal knobs (safe defaults)
    r.setdefault("minWordsPerRun", 2)          # meaningful run threshold
    r.setdefault("sdhMinCueDurationMs", 600)   # standalone SDH cue length
    r.setdefault("sdhDedupWindowMs", 250)      # dedup SDH events close together

    return r


# ============================================================
# PUNCTUATION-AWARE WRAP ENGINE
# ============================================================

_PUNCT_BREAK = re.compile(r"[,\.;:\!\?]$")

def wrap_text_to_lines(
    text: str,
    max_chars: int,
    max_lines: int,
    prefer_punct: bool = True
) -> str:
    """
    Wraps text into <= max_lines lines, each ideally <= max_chars.
    Greedy packing but will try to break on punctuation when prefer_punct=True.
    """
    raw = re.sub(r"\s+", " ", (text or "").replace("\n", " ").strip())
    if not raw:
        return ""

    words = raw.split(" ")
    lines: List[str] = []
    cur: List[str] = []

    def cur_len_with(word: str) -> int:
        if not cur:
            return len(word)
        return len(" ".join(cur)) + 1 + len(word)

    def flush_line():
        nonlocal cur
        if cur:
            lines.append(" ".join(cur).strip())
            cur = []

    i = 0
    while i < len(words):
        w = words[i]

        # If adding word would exceed max, flush.
        if cur and cur_len_with(w) > max_chars:
            if prefer_punct:
                # Try to move a trailing punctuation word to end of the line if present
                # (We already packed greedily; punctuation preference mostly helps by
                # not overfilling when punctuation appears.)
                pass
            flush_line()
            if len(lines) >= max_lines:
                break
            continue

        cur.append(w)

        # If punctuation at end of current word, and we still have words ahead,
        # we prefer flushing here when it doesn't create a very short line.
        if prefer_punct and _PUNCT_BREAK.search(w) and (i + 1) < len(words):
            # only flush if it creates a reasonably filled line
            if len(" ".join(cur)) >= max_chars * 0.55:
                flush_line()
                if len(lines) >= max_lines:
                    break

        i += 1

    if len(lines) < max_lines and cur:
        flush_line()

    # If we somehow created > max_lines, truncate
    lines = lines[:max_lines]

    return "\n".join(lines)


# ============================================================
# CPS + DURATION RULES
# ============================================================

def calc_cps(text: str, start: float, end: float) -> float:
    dur = max(0.001, end - start)
    chars = len((text or "").replace("\n", ""))
    return chars / dur


def enforce_duration_and_gap(
    cues: List[Dict[str, Any]],
    rules: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Enforces:
      - minDurationMs (by extending end, without overlapping next)
      - maxDurationMs (by shrinking end)
      - minGapMs (by shortening end to leave gap before next cue)
    NOTE: We do NOT shift start earlier (to preserve SRT authority),
          except SDH inserts handled separately.
    """
    if not cues:
        return cues

    min_dur = float(rules.get("minDurationMs", 1000)) / 1000.0
    max_dur = float(rules.get("maxDurationMs", 7000)) / 1000.0
    min_gap = float(rules.get("minGapMs", 80)) / 1000.0

    out = [dict(c) for c in cues]

    for i in range(len(out)):
        c = out[i]
        start = float(c["start"])
        end = float(c["end"])

        # clamp inverted
        if end < start:
            end = start

        dur = end - start

        # enforce max duration by shrinking end (safe)
        if dur > max_dur:
            end = start + max_dur
            dur = end - start

        # enforce min duration by extending end, but do not overlap next-start - min_gap
        if dur < min_dur:
            desired_end = start + min_dur
            if i + 1 < len(out):
                next_start = float(out[i + 1]["start"])
                cap_end = max(start, next_start - min_gap)
                end = min(desired_end, cap_end)
            else:
                end = desired_end

        # enforce min gap vs next cue
        if i + 1 < len(out):
            next_start = float(out[i + 1]["start"])
            if end > next_start - min_gap:
                end = max(start, next_start - min_gap)

        c["start"] = start
        c["end"] = end

    # Remove any cues that became zero-length
    cleaned: List[Dict[str, Any]] = []
    for c in out:
        if float(c["end"]) - float(c["start"]) <= 0.001:
            continue
        cleaned.append(c)

    return cleaned


def apply_start_at_hour00(cues: List[Dict[str, Any]], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    If startAtHour00 is False, we leave as-is.
    If True, we ensure time starts from 00:00:00 (it already does in our seconds model).
    This exists for SCC offset workflows and future expansion; kept as no-op for SRT/VTT.
    """
    # For SRT/VTT we already output starting at 00:00:00.
    # If in the future you need source timecode offsets, implement here.
    return cues
# ============================================================
# SDH EXTRACTION + ALIGNMENT
# ============================================================

SDH_BRACKET_PATTERN = re.compile(r"\[(.*?)\]")
MUSIC_PATTERN = re.compile(r"music", re.IGNORECASE)

def normalize_sdh_token(raw: str) -> str:
    """
    Normalizes SDH tokens to broadcast style.
    """
    txt = raw.strip()

    if MUSIC_PATTERN.search(txt):
        return "[♪ MUSIC ♪]"

    txt = txt.upper()
    txt = re.sub(r"\s+", " ", txt)

    if not txt.startswith("["):
        txt = "[" + txt
    if not txt.endswith("]"):
        txt = txt + "]"

    return txt


def extract_sdh_tokens_from_transcript(transcript_text: str) -> List[str]:
    """
    Returns list of normalized SDH tokens found in transcript.
    """
    if not transcript_text:
        return []

    matches = SDH_BRACKET_PATTERN.findall(transcript_text)
    tokens = []

    for m in matches:
        normalized = normalize_sdh_token(m)
        tokens.append(normalized)

    return tokens


def align_sdh_to_words(
    transcript_text: str,
    words: List[Dict[str, Any]],
) -> List[Tuple[float, str]]:
    """
    Align bracket tokens to nearest word start time.
    Walks transcript tokens and word list in parallel.
    Returns list of (timestamp_seconds, token).
    """
    if not transcript_text or not words:
        return []

    raw = re.sub(r"\s+", " ", transcript_text.replace("\n", " ").strip())

    # Tokenize preserving bracket tokens
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

    def next_word_time():
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

        # advance word pointer trying to match spoken word
        tn = normalize_word(tok)
        if not tn:
            continue

        for _ in range(6):  # small forward scan tolerance
            if word_idx >= len(words):
                break

            wn = normalize_word(str(words[word_idx].get("text", "")))
            if wn == tn or (wn and tn and (wn.startswith(tn) or tn.startswith(wn))):
                word_idx += 1
                break
            word_idx += 1

    return aligned


def deduplicate_sdh_events(
    events: List[Tuple[float, str]],
    rules: Dict[str, Any]
) -> List[Tuple[float, str]]:
    """
    Removes duplicate SDH events that occur within dedup window.
    """
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


def build_sdh_cue_candidates(
    transcript_text: str,
    words: List[Dict[str, Any]],
    rules: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Builds standalone SDH cue candidates with timestamp only.
    Duration applied later.
    """
    aligned = align_sdh_to_words(transcript_text, words)
    deduped = deduplicate_sdh_events(aligned, rules)

    min_dur = float(rules.get("sdhMinCueDurationMs", 600)) / 1000.0

    candidates: List[Dict[str, Any]] = []

    for ts, token in deduped:
        candidates.append({
            "start": ts,
            "end": ts + min_dur,
            "text": token,
            "type": "sdh"
        })

    return candidates
# ============================================================
# SDH INSERTION INTO TIMELINE
# ============================================================

def insert_sdh_cues_into_timeline(
    dialogue_cues: List[Dict[str, Any]],
    sdh_candidates: List[Dict[str, Any]],
    rules: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Inserts SDH cues into dialogue timeline safely.
    We:
      - Respect minGapMs
      - Avoid overlap with dialogue
      - Clamp SDH duration if needed
      - Keep deterministic ordering
    """
    if not sdh_candidates:
        return dialogue_cues

    min_gap = float(rules.get("minGapMs", 80)) / 1000.0

    merged = sorted(dialogue_cues, key=lambda c: (c["start"], c["end"]))

    for sdh in sdh_candidates:
        s_start = float(sdh["start"])
        s_end = float(sdh["end"])

        # Find insertion index
        idx = 0
        while idx < len(merged) and merged[idx]["start"] <= s_start:
            idx += 1

        prev_end = merged[idx - 1]["end"] if idx - 1 >= 0 else 0.0
        next_start = merged[idx]["start"] if idx < len(merged) else None

        # Clamp start so we don't overlap previous cue
        s_start = max(s_start, prev_end + min_gap)

        # Clamp end so we don't overlap next cue
        if next_start is not None:
            max_allowed_end = next_start - min_gap
            if s_end > max_allowed_end:
                s_end = max_allowed_end

        if s_end <= s_start:
            continue  # no room to insert safely

        merged.append({
            "start": s_start,
            "end": s_end,
            "text": sdh["text"],
            "type": "sdh"
        })

        merged = sorted(merged, key=lambda c: (c["start"], c["end"]))

    return merged


# ============================================================
# FINAL SORT + SAFETY NORMALIZATION
# ============================================================

def normalize_and_sort_cues(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensures:
      - start <= end
      - sorted by start time
      - stable ordering
    """
    cleaned: List[Dict[str, Any]] = []

    for c in cues:
        start = float(c["start"])
        end = float(c["end"])

        if end < start:
            end = start

        cleaned.append({
            "start": start,
            "end": end,
            "text": c["text"],
            "type": c.get("type", "dialogue")
        })

    cleaned.sort(key=lambda x: (x["start"], x["end"]))
    return cleaned
# ============================================================
# FULL CAPTION BUILD PIPELINE (SRT-FIRST)
# ============================================================

def build_pro_captions(transcript_id: str, request_rules: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full broadcast-grade caption build pipeline.
    """

    # ---- Load transcript data
    transcript_json = aa_get_transcript(transcript_id)
    transcript_text = transcript_json.get("text", "") or ""
    srt_raw = aa_get_srt(transcript_id)
    words = sorted(aa_get_words(transcript_id), key=lambda w: w.get("start", 0))

    # ---- Sanitize rules
    rules = get_rules(request_rules)

    max_chars = int(rules["maxCharsPerLine"])
    max_lines = int(rules["maxLines"])
    max_cps = float(rules["maxCPS"])
    min_words_per_run = int(rules["minWordsPerRun"])
    prefer_punct = bool(rules["preferPunctuationBreaks"])

    # ---- Parse SRT (timing authority)
    base_cues = parse_srt(srt_raw)

    dialogue_cues: List[Dict[str, Any]] = []

    # ========================================================
    # DIALOGUE PROCESSING (SRT-FIRST)
    # ========================================================

    for cue in base_cues:
        start = float(cue["start"])
        end = float(cue["end"])
        original_text = cue["text"]

        # Words overlapping this cue
        cue_words = words_in_window(words, start, end)

        # Build speaker runs
        runs = build_speaker_runs(cue_words)

        # Collapse tiny jitter runs
        runs = split_runs_meaningfully(runs, min_words_per_run)

        # Decide if meaningful multi-speaker split
        if len(runs) >= 2 and count_distinct_speakers(runs) >= 2:
            # Split cue into speaker-run-based subcues
            for r in runs[:2]:  # broadcast constraint: max 2 lines
                r_start = clamp(r["start"], start, end)
                r_end = clamp(r["end"], start, end)

                wrapped = wrap_text_to_lines(
                    r["text"],
                    max_chars,
                    max_lines,
                    prefer_punct=prefer_punct
                )

                cps_val = calc_cps(wrapped, r_start, r_end)

                dialogue_cues.append({
                    "start": r_start,
                    "end": r_end,
                    "text": wrapped,
                    "cps": round(cps_val, 2),
                    "type": "dialogue"
                })

        else:
            wrapped = wrap_text_to_lines(
                original_text,
                max_chars,
                max_lines,
                prefer_punct=prefer_punct
            )

            # Apply dash formatting only if required
            wrapped = apply_dash_if_needed(wrapped, runs, rules)

            cps_val = calc_cps(wrapped, start, end)

            dialogue_cues.append({
                "start": start,
                "end": end,
                "text": wrapped,
                "cps": round(cps_val, 2),
                "type": "dialogue"
            })

    # ========================================================
    # ENFORCE DURATION + GAP RULES (Dialogue Only)
    # ========================================================

    dialogue_cues = normalize_and_sort_cues(dialogue_cues)
    dialogue_cues = enforce_duration_and_gap(dialogue_cues, rules)

    # ========================================================
    # SDH EXTRACTION + INSERTION
    # ========================================================

    sdh_candidates = build_sdh_cue_candidates(transcript_text, words, rules)

    merged_cues = insert_sdh_cues_into_timeline(
        dialogue_cues,
        sdh_candidates,
        rules
    )

    merged_cues = normalize_and_sort_cues(merged_cues)

    # ========================================================
    # FINAL PASS: CPS TAGGING (for reporting)
    # ========================================================

    for cue in merged_cues:
        cue["cps"] = round(calc_cps(cue["text"], cue["start"], cue["end"]), 2)

    # Apply optional start-at-hour-00 handling
    merged_cues = apply_start_at_hour00(merged_cues, rules)

    # ========================================================
    # EXPORTS
    # ========================================================

    srt_out = cues_to_srt(merged_cues)
    vtt_out = cues_to_vtt(merged_cues)

    result_json = {
        "transcriptId": transcript_id,
        "rules": rules,
        "cues": merged_cues
    }

    return {
        "result_json": result_json,
        "srt": srt_out,
        "vtt": vtt_out
    }
# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/health")
def health():
    return {"ok": True, "timestamp": now_iso()}
@app.get("/cors-test")
def cors_test():
    return {"ok": True}

@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    # Make sure the browser ALWAYS gets a readable JSON error + CORS headers
    resp = JSONResponse(
        status_code=500,
        content={"detail": f"{type(exc).__name__}: {str(exc)}"},
    )
    return _maybe_add_cors_headers(resp, request)

# ------------------------------------------------------------
# CREATE JOB
# ------------------------------------------------------------

@app.post("/v1/jobs", response_model=JobResponse)
def create_job(req: CreateJobRequest):
    title = (req.title or "Untitled").strip()

    rules = get_rules(req.rules)  # rules come from Base44 UI (dynamic)
    rules_json = json.dumps(rules)

    transcript_id = aa_create_transcript(
        req.mediaUrl,
        speaker_labels=bool(req.speaker_labels),
        language_detection=bool(req.language_detection),
        allow_http=bool(getattr(req, "allow_http", False)),
        # if you later wire webhook fields, add them here
    )

    created = now_iso()

    with db() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO jobs
            (id, created_at, updated_at, title, media_url, status, error,
             rules_json, result_json, srt, vtt)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                transcript_id,
                created,
                created,
                title,
                req.mediaUrl,
                "processing",
                None,         # error column (db) is null at creation
                rules_json,
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
        title=title,
        mediaUrl=req.mediaUrl,
        status="processing",
        error=None,  # IMPORTANT: include this so response_model validation never fails
    )

# ------------------------------------------------------------
# JOB STATUS
# ------------------------------------------------------------

@app.get("/v1/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str):

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
        exports=exports,
    )


# ------------------------------------------------------------
# EXPORT SRT
# ------------------------------------------------------------

@app.get("/v1/jobs/{job_id}/export/srt")
def export_srt(job_id: str):

    with db() as conn:
        row = conn.execute("SELECT srt, status FROM jobs WHERE id = ?", (job_id,)).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    if row["status"] != "completed" or not row["srt"]:
        raise HTTPException(status_code=409, detail="Job not completed")

    return row["srt"]


# ------------------------------------------------------------
# EXPORT VTT
# ------------------------------------------------------------

@app.get("/v1/jobs/{job_id}/export/vtt")
def export_vtt(job_id: str):

    with db() as conn:
        row = conn.execute("SELECT vtt, status FROM jobs WHERE id = ?", (job_id,)).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    if row["status"] != "completed" or not row["vtt"]:
        raise HTTPException(status_code=409, detail="Job not completed")

    return row["vtt"]


# ------------------------------------------------------------
# EXPORT JSON
# ------------------------------------------------------------

@app.get("/v1/jobs/{job_id}/export/json")
def export_json(job_id: str):

    with db() as conn:
        row = conn.execute("SELECT result_json, status FROM jobs WHERE id = ?", (job_id,)).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    if row["status"] != "completed" or not row["result_json"]:
        raise HTTPException(status_code=409, detail="Job not completed")

    return json.loads(row["result_json"])


# ------------------------------------------------------------
# WEBHOOK HANDLER
# ------------------------------------------------------------

@app.post("/v1/webhooks/assemblyai")
async def assemblyai_webhook(request: Request):

    payload = await request.json()

    transcript_id = payload.get("transcript_id") or payload.get("id")
    status = payload.get("status")
    updated = now_iso()

    if not transcript_id:
        return {"ok": True}

    with db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (transcript_id,)).fetchone()

    if not row:
        return {"ok": True}

    rules = json.loads(row["rules_json"] or "{}")

    if status == "completed":

        try:
            built = build_pro_captions(transcript_id, rules)

            with db() as conn:
                conn.execute("""
                UPDATE jobs
                SET status=?, updated_at=?, result_json=?, srt=?, vtt=?
                WHERE id=?
                """, (
                    "completed",
                    updated,
                    json.dumps(built["result_json"]),
                    built["srt"],
                    built["vtt"],
                    transcript_id,
                ))
                conn.commit()

        except Exception as e:
            with db() as conn:
                conn.execute("""
                UPDATE jobs
                SET status=?, error=?, updated_at=?
                WHERE id=?
                """, (
                    "error",
                    str(e),
                    updated,
                    transcript_id
                ))
                conn.commit()

    elif status in ("error", "failed"):
        with db() as conn:
            conn.execute("""
            UPDATE jobs
            SET status=?, error=?, updated_at=?
            WHERE id=?
            """, (
                "error",
                payload.get("error", "AssemblyAI failure"),
                updated,
                transcript_id
            ))
            conn.commit()

    else:
        with db() as conn:
            conn.execute("""
            UPDATE jobs
            SET status=?, updated_at=?
            WHERE id=?
            """, (
                "processing",
                updated,
                transcript_id
            ))
            conn.commit()

    return {"ok": True}

@app.post("/debug-test")
def debug_test():
    return {"ok": True}
