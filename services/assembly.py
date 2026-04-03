import os
import re
import time
from typing import Any, Dict, List, Tuple

import requests

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
ASSEMBLYAI_BASE_URL = "https://api.assemblyai.com/v2"

POLL_INTERVAL_SECONDS = int(os.getenv("ASSEMBLY_POLL_INTERVAL_SECONDS", "5") or 5)
POLL_TIMEOUT_SECONDS = int(os.getenv("ASSEMBLY_POLL_TIMEOUT_SECONDS", str(60 * 60 * 8)) or (60 * 60 * 8))

BRACKET_TAG_RE = re.compile(r"\[[^\]]+\]")


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
    payload: Dict[str, Any] = {
        "audio_url": media_url,
        "speech_models": ["universal-3-pro"],
        "speaker_labels": speaker_labels,
        "format_text": True,
        "language_detection": language_detection,
        "prompt": (
            "Transcribe speech with accurate punctuation and formatting. "
            "Preserve non-speech audio in tags to indicate when the audio occurred. "
            "Include audio event markers for [music], [laughter], [applause], [noise], "
            "[pause], [inaudible], [cheering], and [sound effect] when clearly present. "
            "Preserve proper nouns and show titles accurately. "
            "If speech is in a language other than English, preserve it in the original language."
        ),
    }

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


def fetch_transcript_result(transcript_id: str, require_completed: bool = True) -> Dict[str, Any]:
    response = requests.get(
        f"{ASSEMBLYAI_BASE_URL}/transcript/{transcript_id}",
        headers=_headers(),
        timeout=120,
    )

    if response.status_code >= 400:
        raise RuntimeError(
            f"AssemblyAI fetch failed ({response.status_code}): {response.text}"
        )

    data = response.json()
    status = data.get("status")

    if status == "error":
        raise RuntimeError(
            f"AssemblyAI transcription failed: {data.get('error', 'Unknown error')}"
        )

    if require_completed and status != "completed":
        raise RuntimeError(
            f"AssemblyAI transcript not completed (status={status})"
        )

    return data


def build_caption_inputs_from_assembly_result(
    assembly_result: Dict[str, Any]
) -> Tuple[str, List[Dict[str, Any]]]:

    transcript_id = assembly_result.get("id")
    if not transcript_id:
        raise ValueError("AssemblyAI result missing transcript id")

    print("[ASSEMBLY] NEW VERSION ACTIVE - building caption inputs")

    backbone_srt_text = fetch_srt(transcript_id)

    spoken_tokens = build_word_timestamps_from_result(assembly_result)
    sound_tokens = extract_sound_tokens_from_json(assembly_result)

    if not sound_tokens:
        print("[ASSEMBLY] No sound tags found in JSON; trying SRT fallback")
        sound_tokens = extract_sound_tokens_from_srt(backbone_srt_text)

    print(f"[ASSEMBLY] Spoken tokens: {len(spoken_tokens)}")
    print(f"[ASSEMBLY] Sound tokens extracted: {len(sound_tokens)}")

    merged_tokens = merge_and_dedup_tokens(spoken_tokens, sound_tokens)

    print(f"[ASSEMBLY] Total merged tokens: {len(merged_tokens)}")

    return backbone_srt_text, merged_tokens


def fetch_srt(transcript_id: str) -> str:
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


def extract_sound_tokens_from_json(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    tokens: List[Dict[str, Any]] = []

    utterances = result.get("utterances") or []
    full_text = (result.get("text") or "").strip()

    for utt in utterances:
        text = (utt.get("text") or "").strip()
        if not text:
            continue

        start_ms = int(utt.get("start", 0))
        end_ms = int(utt.get("end", start_ms))

        matches = BRACKET_TAG_RE.findall(text)

        if not matches:
            continue

        total_dur = max(1, end_ms - start_ms)
        per_tag = max(1, total_dur // max(1, len(matches)))

        for idx, tag in enumerate(matches):
            tag_start = start_ms + (idx * per_tag)
            tag_end = min(end_ms, tag_start + per_tag)

            tokens.append(
                {
                    "text": tag.strip().upper(),
                    "start_ms": tag_start,
                    "end_ms": max(tag_start + 1, tag_end),
                    "speaker": None,
                }
            )

    if tokens:
        return tokens

    matches = BRACKET_TAG_RE.findall(full_text)

    for tag in matches:
        tokens.append(
            {
                "text": tag.strip().upper(),
                "start_ms": 0,
                "end_ms": 1,
                "speaker": None,
            }
        )

    return tokens


def extract_sound_tokens_from_srt(srt_text: str) -> List[Dict[str, Any]]:
    tokens: List[Dict[str, Any]] = []

    cues = parse_srt_blocks(srt_text)

    for cue in cues:
        text = " ".join(cue["lines"]).strip()

        tags = BRACKET_TAG_RE.findall(text)

        if not tags:
            continue

        start_ms = cue["start_ms"]
        end_ms = cue["end_ms"]
        total_dur = max(1, end_ms - start_ms)
        per_tag = max(1, total_dur // max(1, len(tags)))

        for idx, tag in enumerate(tags):
            tag_start = start_ms + (idx * per_tag)
            tag_end = min(end_ms, tag_start + per_tag)

            tokens.append(
                {
                    "text": tag.strip().upper(),
                    "start_ms": tag_start,
                    "end_ms": max(tag_start + 1, tag_end),
                    "speaker": None,
                }
            )

    return tokens


def parse_srt_blocks(srt_text: str) -> List[Dict[str, Any]]:
    blocks = re.split(r"\n\s*\n", srt_text.strip(), flags=re.MULTILINE)

    cues: List[Dict[str, Any]] = []

    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]

        if len(lines) < 2:
            continue

        try:
            int(lines[0])
            timing = lines[1]
            text_lines = lines[2:]
        except ValueError:
            timing = lines[0]
            text_lines = lines[1:]

        if "-->" not in timing:
            continue

        start_tc, end_tc = [part.strip() for part in timing.split("-->")]

        start_ms = tc_to_ms(start_tc)
        end_ms = tc_to_ms(end_tc)

        cues.append(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "lines": text_lines,
            }
        )

    return cues


def tc_to_ms(tc: str) -> int:
    m = re.match(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", tc.strip())

    if not m:
        raise ValueError(f"Invalid SRT timecode: {tc}")

    hh, mm, ss, ms = map(int, m.groups())

    return ((hh * 60 + mm) * 60 + ss) * 1000 + ms


def merge_and_dedup_tokens(
    spoken_tokens: List[Dict[str, Any]],
    sound_tokens: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:

    merged = spoken_tokens + sound_tokens

    merged.sort(key=lambda x: (x["start_ms"], x["end_ms"], x["text"]))

    deduped: List[Dict[str, Any]] = []
    seen = set()

    for tok in merged:
        key = (tok["text"], tok["start_ms"], tok["end_ms"], tok.get("speaker"))

        if key in seen:
            continue

        seen.add(key)
        deduped.append(tok)

    return deduped


def normalize_tokens(ts: Any) -> List[Dict[str, Any]]:
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
