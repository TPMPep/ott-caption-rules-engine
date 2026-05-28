"""
ElevenLabs Scribe v2 client + result normalizer.

PURPOSE
-------
Scribe v2 is the AUDITOR-GRADE default for Closed Caption transcription in
this app. Distinct from AssemblyAI in three load-bearing ways:

  1. Native diarization. Every word carries a stable `speaker_id` ("speaker_0",
     "speaker_1", ...) — no separate "speaker_labels" toggle, no probabilistic
     post-hoc clustering. FCC 47 CFR §79.1 — speaker identification is a
     provider-native feature, not coaxed from a prompt.

  2. Native audio-event tagging. Scribe emits non-dialogue events
     (music, laughter, applause, footsteps, gunshot, ...) inline in the
     transcript as parenthetical tokens, e.g.
       {"type": "audio_event", "text": "(music)", "start": 12.34, "end": 18.90}
     We promote these into a structured `audio_events[]` array so the
     downstream worker can render them as proper non-dialogue caption rows.
     This is the difference between "we asked the model nicely" (AAI) and
     "the provider's specified feature gave us a structured list" (Scribe).

  3. 90+ languages with one model. No fallback chain, no per-language code
     swap; one API path covers everything from English to Norwegian Bokmål
     to Tamil.

OUTPUT SHAPE
------------
After `submit_and_wait()` and `normalize_scribe_result()` we return the EXACT
same shape the formatter / Base44 ingester already consume from AssemblyAI:

    {
      "id": "<scribe transcription_id>",
      "language_code": "eng",
      "audio_duration": 1234.5,           # seconds
      "text": "full plain-text transcript",
      "utterances": [                     # diarized utterances
        {
          "speaker": "A",                 # mapped from speaker_0 → "A"
          "start": 1234,                  # ms
          "end":   5678,                  # ms
          "text":  "What's up?",
          "confidence": 0.93,
          "words": [
            {"text": "What's", "start": 1234, "end": 1389, "confidence": 0.97},
            ...
          ]
        }
      ],
      "audio_events": [                   # NEW — native non-dialogue tags
        {"event_type": "music", "start": 12340, "end": 18900},
        {"event_type": "applause", "start": 56000, "end": 58000}
      ],
      "words": [...]                      # flat word list (for backbone SRT)
    }

This is the contract the worker's `ccFormatRunWorkerStep` already expects —
its `EVENT_TO_TEXT_WORKER` mapping consumes `audio_events[]` directly.

SECURITY
--------
ELEVENLABS_API_KEY is read once at module-import time from env. If unset,
every call throws — by design. No silent fallback to AssemblyAI from here;
that decision lives in `main.py`.

DOCS
----
https://elevenlabs.io/docs/api-reference/speech-to-text/convert
"""

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"

# Scribe v2 is a SYNCHRONOUS API (no transcript_id polling like AAI).
# A single POST returns the full transcript. We still apply a generous
# read timeout because long files can take a few minutes. The engine's
# background-thread model already covers concurrency from the user's
# perspective — we are not blocking the request that created the job.
SCRIBE_REQUEST_TIMEOUT_SECONDS = int(
    os.getenv("ELEVENLABS_SCRIBE_TIMEOUT_SECONDS", str(60 * 30)) or (60 * 30)
)
SCRIBE_MODEL_ID = os.getenv("ELEVENLABS_SCRIBE_MODEL", "scribe_v2")

# Map common Scribe audio-event tag text → normalized event_type.
# Scribe emits these with parentheses: "(music)", "(applause)", etc.
# Source: ElevenLabs Scribe documentation + production observation.
# Order matters — we longest-match so "(crowd cheering)" beats "(cheering)".
AUDIO_EVENT_TAG_MAP: List[Tuple[str, str]] = [
    # Music
    ("music playing", "music_playing"),
    ("music",         "music"),
    # Crowd / audience reactions
    ("crowd cheering", "crowd_noise"),
    ("cheering",       "crowd_noise"),
    ("crowd noise",    "crowd_noise"),
    ("crowd",          "crowd_noise"),
    ("applause",       "applause"),
    ("laughter",       "laughter"),
    ("laughing",       "laughter"),
    ("gasps",          "gasping"),
    ("gasping",        "gasping"),
    # Vocalizations
    ("screaming", "screaming"),
    ("shouting",  "shouting"),
    ("crying",    "crying"),
    ("sobbing",   "crying"),
    ("whispering","whispering"),
    # Phones / alarms
    ("phone ringing", "ringtone"),
    ("phone rings",   "ringtone"),
    ("ringtone",      "ringtone"),
    ("doorbell",      "ringtone"),
    ("alarm",         "alarm"),
    ("siren",         "siren"),
    ("buzzer",        "alarm"),
    ("whistle blows", "whistle"),
    ("whistle",       "whistle"),
    # Impacts / vehicles / weapons
    ("gunshot",        "gunshot"),
    ("gunfire",        "gunshot"),
    ("explosion",      "explosion"),
    ("engine revving", "engine_noise"),
    ("engine noise",   "engine_noise"),
    ("engine",         "engine_noise"),
    ("car horn",       "engine_noise"),
    # Ambient
    ("thunder",   "thunder"),
    ("rain",      "rain"),
    ("wind",      "rain"),
    ("footsteps", "footsteps"),
    ("knocking",  "knocking"),
    ("knock",     "knocking"),
]
# Compile a single regex matching any of the parenthetical / bracketed
# audio-event tokens. Built once at import-time.
_AUDIO_EVENT_TOKEN_RE = re.compile(r"[\(\[]\s*([^\)\]]+?)\s*[\)\]]")


# ─── HTTP ──────────────────────────────────────────────────────────────────

def _headers() -> Dict[str, str]:
    if not ELEVENLABS_API_KEY:
        raise ValueError("Missing ELEVENLABS_API_KEY environment variable")
    return {"xi-api-key": ELEVENLABS_API_KEY}


def submit_and_wait(media_url: str, language_code: Optional[str] = None) -> Dict[str, Any]:
    """
    POST media URL to /v1/speech-to-text and return the raw Scribe response.

    Scribe v2 is synchronous: this single call blocks until transcription is
    complete and returns the full payload. The engine's background-thread
    model already isolates concurrency from the user-facing request — this
    function is called from inside `run_caption_job` thread.

    Parameters
    ----------
    media_url : str
        Signed S3 URL of the source media.
    language_code : str | None
        ISO-639-3 (3-letter) code for the source language. If None or "auto",
        Scribe auto-detects. The Base44 producer derives this from the
        project's `source_language` via `_lib_scribeLanguageCodes`.
    """
    payload: Dict[str, Any] = {
        "model_id": SCRIBE_MODEL_ID,
        "cloud_storage_url": media_url,
        # Native diarization — every word gets a stable speaker_id.
        "diarize": True,
        # Native non-dialogue tagging — emits "(music)", "(applause)", etc.
        # in the transcript output. Promoted to a structured audio_events[]
        # array by `normalize_scribe_result` below.
        "tag_audio_events": True,
        # Word-level timestamps are required for backbone SRT + accurate
        # rules-engine line-wrapping. Scribe returns these by default;
        # explicit for safety.
        "timestamps_granularity": "word",
    }
    if language_code and language_code != "auto":
        payload["language_code"] = language_code

    response = requests.post(
        f"{ELEVENLABS_BASE_URL}/speech-to-text",
        data=payload,
        headers=_headers(),
        timeout=SCRIBE_REQUEST_TIMEOUT_SECONDS,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"ElevenLabs Scribe submit failed ({response.status_code}): "
            f"{response.text[:500]}"
        )
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected Scribe response shape: {type(data).__name__}")
    return data


# ─── Normalization ────────────────────────────────────────────────────────

def _seconds_to_ms(v: Any) -> int:
    """Scribe returns floating-point seconds; convert to integer milliseconds."""
    try:
        return int(round(float(v) * 1000.0))
    except (TypeError, ValueError):
        return 0


def _classify_audio_event_tag(raw: str) -> Optional[str]:
    """
    Map a raw parenthetical/bracketed audio-event token (e.g. "music playing",
    "crowd cheering") to a normalized event_type the worker's
    EVENT_TO_TEXT_WORKER map knows how to render.

    Returns None if the token is not a recognized audio event — caller should
    fall through and treat it as inline dialogue text. This conservatism is
    deliberate: an auditor-defensible pipeline must not invent non-dialogue
    cues from arbitrary parenthetical text the model produced.
    """
    text = re.sub(r"\s+", " ", raw or "").strip().lower()
    if not text:
        return None
    for needle, event_type in AUDIO_EVENT_TAG_MAP:
        if needle in text:
            return event_type
    return None


def _extract_audio_events_from_words(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Scribe v2 emits audio events as words with `type='audio_event'` (newer
    response shape) OR as parenthetical/bracketed text inside regular word
    tokens (older shape). We handle BOTH so a Scribe response-shape revision
    never silently drops our non-dialogue coverage.
    """
    out: List[Dict[str, Any]] = []
    for w in words or []:
        word_type = (w.get("type") or "").lower()
        raw_text = (w.get("text") or "").strip()
        if not raw_text:
            continue

        # Shape A — explicit audio_event word type.
        if word_type == "audio_event":
            # Strip any wrapping punctuation: "(music)" → "music"
            stripped = raw_text.strip("()[]{}").strip()
            event_type = _classify_audio_event_tag(stripped)
            if event_type is None:
                continue
            out.append({
                "event_type": event_type,
                "start": _seconds_to_ms(w.get("start", 0)),
                "end":   _seconds_to_ms(w.get("end", w.get("start", 0))),
            })
            continue

        # Shape B — parenthetical token inside a regular word.
        m = _AUDIO_EVENT_TOKEN_RE.search(raw_text)
        if not m:
            continue
        event_type = _classify_audio_event_tag(m.group(1))
        if event_type is None:
            continue
        out.append({
            "event_type": event_type,
            "start": _seconds_to_ms(w.get("start", 0)),
            "end":   _seconds_to_ms(w.get("end", w.get("start", 0))),
        })

    # Merge adjacent same-type events (Scribe can emit multiple consecutive
    # "(music)" tokens through a song — we want one cue, not 20).
    if not out:
        return out
    out.sort(key=lambda e: (e["start"], e["end"]))
    merged: List[Dict[str, Any]] = [dict(out[0])]
    for ev in out[1:]:
        prev = merged[-1]
        if ev["event_type"] == prev["event_type"] and ev["start"] - prev["end"] <= 2000:
            prev["end"] = max(prev["end"], ev["end"])
        else:
            merged.append(dict(ev))
    # Enforce a minimum visible duration so 100ms music ticks don't render.
    return [e for e in merged if (e["end"] - e["start"]) >= 300]


def _map_speaker_id_to_letter(speaker_id: Any, mapping: Dict[str, str]) -> str:
    """
    Convert Scribe's "speaker_0", "speaker_1" → "A", "B", ... so the rest
    of the pipeline (Base44 worker + ingester) can treat Scribe output
    identically to AAI's diarization labels. The mapping is built lazily
    on first appearance to preserve ORDER ("A" = first speaker we saw).
    """
    if speaker_id is None:
        return ""
    key = str(speaker_id)
    if key in mapping:
        return mapping[key]
    letter = chr(ord("A") + len(mapping))
    mapping[key] = letter
    return letter


def _is_audio_event_word(w: Dict[str, Any]) -> bool:
    """True if this word token is actually an audio-event marker (shape A
    or shape B). Used to STRIP them from utterance text/words — they belong
    on the audio_events[] array, never inside dialogue.
    """
    if (w.get("type") or "").lower() == "audio_event":
        return True
    raw_text = (w.get("text") or "").strip()
    if not raw_text:
        return False
    m = _AUDIO_EVENT_TOKEN_RE.fullmatch(raw_text)
    return bool(m and _classify_audio_event_tag(m.group(1)) is not None)


def normalize_scribe_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a raw Scribe v2 response into the AAI-compatible shape the
    formatter and Base44 ingester already consume.

    Scribe v2 response (relevant fields):
        {
          "language_code": "eng",
          "language_probability": 0.99,
          "text": "...",
          "words": [
            {"text":"Hello", "start":0.12, "end":0.40, "type":"word",
             "speaker_id":"speaker_0", ...},
            {"text":"(music)", "start":1.20, "end":3.40, "type":"audio_event"},
            ...
          ]
        }

    We:
      • Build a stable speaker_id → "A"/"B" mapping (in first-appearance order).
      • Strip audio-event words from dialogue; promote them to audio_events[].
      • Group dialogue words into utterances by (speaker, ≤300ms gap) — same
        contract the worker's `_splitUtteranceByPauses` uses for AAI.
      • Compute audio_duration from the max word.end (Scribe doesn't always
        return a top-level duration).
    """
    raw_words = list(raw.get("words") or [])

    # 1) Extract audio events FIRST (off the full word list).
    audio_events = _extract_audio_events_from_words(raw_words)

    # 2) Filter out audio-event tokens from the dialogue word stream.
    dialogue_words = [w for w in raw_words if not _is_audio_event_word(w)]

    # 3) Build speaker mapping in first-appearance order.
    speaker_letter_map: Dict[str, str] = {}
    for w in dialogue_words:
        sid = w.get("speaker_id")
        if sid is not None:
            _map_speaker_id_to_letter(sid, speaker_letter_map)

    # 4) Flat word array (AAI-compatible shape: text/start/end/speaker/confidence)
    flat_words: List[Dict[str, Any]] = []
    for w in dialogue_words:
        text = (w.get("text") or "").strip()
        if not text:
            continue
        flat_words.append({
            "text": text,
            "start": _seconds_to_ms(w.get("start", 0)),
            "end":   _seconds_to_ms(w.get("end", w.get("start", 0))),
            "speaker": speaker_letter_map.get(str(w.get("speaker_id")), None),
            "confidence": float(w.get("logprob") and 1.0 or w.get("confidence") or 0.0),
        })

    # 5) Group words into utterances by (speaker, ≤300ms gap, ≤7s duration).
    UTT_GAP_MS = 300
    UTT_MAX_DUR_MS = 7000
    utterances: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for fw in flat_words:
        if current is None:
            current = {
                "speaker": fw["speaker"],
                "start": fw["start"],
                "end":   fw["end"],
                "_words": [fw],
            }
            continue
        gap = fw["start"] - current["end"]
        new_dur = fw["end"] - current["start"]
        same_speaker = fw["speaker"] == current["speaker"]
        if same_speaker and gap <= UTT_GAP_MS and new_dur <= UTT_MAX_DUR_MS:
            current["end"] = fw["end"]
            current["_words"].append(fw)
        else:
            utterances.append(current)
            current = {
                "speaker": fw["speaker"],
                "start": fw["start"],
                "end":   fw["end"],
                "_words": [fw],
            }
    if current is not None:
        utterances.append(current)

    # 6) Finalize utterance shape — AAI-compatible.
    finalized: List[Dict[str, Any]] = []
    for u in utterances:
        words_in_utt = u["_words"]
        text = " ".join(w["text"] for w in words_in_utt).strip()
        if not text:
            continue
        confs = [w["confidence"] for w in words_in_utt if w["confidence"]]
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        finalized.append({
            "speaker": u["speaker"],
            "start": u["start"],
            "end":   u["end"],
            "text":  text,
            "confidence": round(avg_conf, 4),
            "words": [
                {"text": w["text"], "start": w["start"], "end": w["end"], "confidence": w["confidence"]}
                for w in words_in_utt
            ],
        })

    # 7) Top-level audio_duration. Scribe doesn't always return it; derive
    # from max word.end so CostLog ALWAYS gets a real number.
    duration_ms = 0
    if raw_words:
        duration_ms = max((_seconds_to_ms(w.get("end", 0)) for w in raw_words), default=0)
    audio_duration_sec = round(duration_ms / 1000.0, 3) if duration_ms else 0.0

    # 8) Plain text — strip audio-event tokens so the full-text record is
    # editorial-clean.
    full_text = (raw.get("text") or "").strip()
    if full_text:
        full_text = _AUDIO_EVENT_TOKEN_RE.sub(" ", full_text)
        full_text = re.sub(r"\s+", " ", full_text).strip()

    return {
        "id": raw.get("transcription_id") or raw.get("id") or "",
        "language_code": raw.get("language_code") or "",
        "language_probability": raw.get("language_probability"),
        "audio_duration": audio_duration_sec,
        "text": full_text,
        "utterances": finalized,
        "words": flat_words,
        "audio_events": audio_events,
        # Provider fingerprint — survives all the way to the auditor row.
        "_provider": "elevenlabs",
        "_model_id": SCRIBE_MODEL_ID,
    }


# ─── Public submit-and-normalize convenience ───────────────────────────────

def transcribe(media_url: str, language_code: Optional[str] = None) -> Dict[str, Any]:
    """One-shot: POST to Scribe, normalize the result for the formatter."""
    raw = submit_and_wait(media_url, language_code=language_code)
    return normalize_scribe_result(raw)
