import os
import time
from typing import Any, Dict, List, Tuple

import requests

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
ASSEMBLYAI_BASE_URL = "https://api.assemblyai.com/v2"

POLL_INTERVAL_SECONDS = 3
POLL_TIMEOUT_SECONDS = 60 * 15  # 15 minutes


def _headers() -> Dict[str, str]:
    if not ASSEMBLYAI_API_KEY:
        raise ValueError("Missing ASSEMBLYAI_API_KEY environment variable")
    return {
        "authorization": ASSEMBLYAI_API_KEY,
        "content-type": "application/json",
    }


def submit_transcription_job(
    media_url: str,
    speaker_labels: bool = True,
    language_detection: bool = True,
) -> str:
    """
    Submit a transcription request to AssemblyAI.

    Returns:
        transcript_id
    """
    payload: Dict[str, Any] = {
        "audio_url": media_url,
        "speech_models": ["universal-3-pro"],
        "speaker_labels": speaker_labels,
        "format_text": True,
        "language_detection": language_detection,
        "prompt": (
            "Transcribe speech with accurate punctuation and formatting. "
            "Preserve non-speech audio in tags to indicate when the audio occurred. "
            "Include audio event markers for [music], [laughter], [applause], [noise], [pause], [inaudible], [cheering], and [sound effect] when clearly present. "
            "Preserve proper nouns and show titles accurately. "
            "If speech is in a language other than English, preserve it in the original language."
        ),
    }

    # Leave language unset so AssemblyAI can auto-detect
    if language_detection:
        pass

    response = requests.post(
        f"{ASSEMBLYAI_BASE_URL}/transcript",
        json=payload,
        headers=_headers(),
        timeout=120,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"AssemblyAI submit failed ({response.status_code}): {response.text}"
        )

    data = response.json()
    transcript_id = data.get("id")
    if not transcript_id:
        raise RuntimeError(f"AssemblyAI submit did not return transcript id: {data}")

    return transcript_id


def wait_for_transcription_result(transcript_id: str) -> Dict[str, Any]:
    """
    Poll AssemblyAI until transcript is completed or failed.

    Returns:
        Full transcript JSON
    """
    elapsed = 0

    while elapsed < POLL_TIMEOUT_SECONDS:
        response = requests.get(
            f"{ASSEMBLYAI_BASE_URL}/transcript/{transcript_id}",
            headers=_headers(),
            timeout=120,
        )

        if response.status_code >= 400:
            raise RuntimeError(
                f"AssemblyAI polling failed ({response.status_code}): {response.text}"
            )

        data = response.json()
        status = data.get("status")

        if status == "completed":
            return data

        if status == "error":
            raise RuntimeError(
                f"AssemblyAI transcription failed: {data.get('error', 'Unknown error')}"
            )

        time.sleep(POLL_INTERVAL_SECONDS)
        elapsed += POLL_INTERVAL_SECONDS

    raise TimeoutError(
        f"AssemblyAI transcription polling timed out after {POLL_TIMEOUT_SECONDS} seconds"
    )


def build_caption_inputs_from_assembly_result(
    assembly_result: Dict[str, Any]
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Convert AssemblyAI transcript result into the two internal inputs expected
    by the caption formatter:

    1) backbone_srt_text
    2) timestamps_json (list of tokens)
    """
    transcript_id = assembly_result.get("id")
    if not transcript_id:
        raise ValueError("AssemblyAI result missing transcript id")

    backbone_srt_text = fetch_srt(transcript_id)
    timestamps_json = build_word_timestamps_from_result(assembly_result)

    return backbone_srt_text, timestamps_json


def fetch_srt(transcript_id: str) -> str:
    """
    Fetch SRT captions from AssemblyAI.
    This becomes the canonical timing backbone.
    """
    response = requests.get(
        f"{ASSEMBLYAI_BASE_URL}/transcript/{transcript_id}/srt",
        headers={"authorization": ASSEMBLYAI_API_KEY},
        timeout=120,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"AssemblyAI SRT fetch failed ({response.status_code}): {response.text}"
        )

    return response.text


def build_word_timestamps_from_result(assembly_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build normalized word-level timestamps list from AssemblyAI result.

    Output format:
    [
      {
        "text": "Hello",
        "start_ms": 1000,
        "end_ms": 1200,
        "speaker": "A"
      }
    ]
    """
    words = assembly_result.get("words") or []
    utterances = assembly_result.get("utterances") or []

    if utterances:
        return _build_tokens_from_utterances(utterances)

    return _build_tokens_from_words(words)


def _build_tokens_from_words(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tokens: List[Dict[str, Any]] = []

    for word in words:
        text = (word.get("text") or "").strip()
        if not text:
            continue

        start_ms = int(word.get("start", 0))
        end_ms = int(word.get("end", start_ms))
        speaker = word.get("speaker")

        tokens.append(
            {
                "text": text,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "speaker": speaker,
            }
        )

    return tokens


def _build_tokens_from_utterances(utterances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tokens: List[Dict[str, Any]] = []

    for utterance in utterances:
        speaker = utterance.get("speaker")
        words = utterance.get("words") or []

        for word in words:
            text = (word.get("text") or "").strip()
            if not text:
                continue

            start_ms = int(word.get("start", 0))
            end_ms = int(word.get("end", start_ms))

            tokens.append(
                {
                    "text": text,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "speaker": speaker,
                }
            )

    return tokens


def normalize_tokens(ts: Any) -> List[Dict[str, Any]]:
    """
    Backward-compatible normalizer.

    Accept either:
      - list[{"text","start","end","speaker"}]
      - list[{"text","start_ms","end_ms","speaker"}]
      - {"words":[...]}
    """
    if isinstance(ts, dict) and "words" in ts:
        items = ts["words"]
    elif isinstance(ts, list):
        items = ts
    else:
        raise ValueError("Unsupported timestamps JSON format. Expect list or dict with 'words'.")

    out: List[Dict[str, Any]] = []

    for item in items:
        text = (item.get("text") or item.get("word") or "").strip()
        start = int(item.get("start_ms", item.get("start", 0)))
        end = int(item.get("end_ms", item.get("end", start)))
        speaker = item.get("speaker")

        out.append(
            {
                "text": text,
                "start_ms": start,
                "end_ms": end,
                "speaker": speaker,
            }
        )

    return out


def is_sound_token(text: str) -> bool:
    if not text:
        return False
    text = text.strip()
    return text.startswith("[") and text.endswith("]")
