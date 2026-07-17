"""
AssemblyAI client + caption-input builder.

Owns:
  - Submitting media URLs to AssemblyAI with broadcast-grade prompt
  - Polling until completion
  - Building backbone SRT + word-level timestamps for the formatter
  - Extracting non-dialogue bracketed audio tags ([MUSIC], [APPLAUSE], etc.)

NOTE: This file is a verbatim mirror of the upstream service. Edits here
must be pushed to https://github.com/TPMPep/ott-caption-rules-engine .
"""

import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY", "")
ASSEMBLYAI_BASE_URL = "https://api.assemblyai.com/v2"

POLL_INTERVAL_SECONDS = int(os.getenv("ASSEMBLY_POLL_INTERVAL_SECONDS", "5") or 5)
POLL_TIMEOUT_SECONDS = int(os.getenv("ASSEMBLY_POLL_TIMEOUT_SECONDS", str(60 * 60 * 8)) or (60 * 60 * 8))

BRACKET_TAG_RE = re.compile(r"\[[^\]]+\]")

DEFAULT_TRANSCRIPTION_PROMPT = (
    # Broadcast-grade prompt aligned with FCC 47 CFR §79.1, which REQUIRES that
    # non-speech information necessary for comprehension be identified in
    # closed captions. Auditor framing: a CC track for a sports / drama /
    # comedy asset that emits zero audio-event tags is presumptively
    # non-compliant. We instruct AAI to emit common, editorially-useful tags
    # generously — the rules engine downstream applies a SOUND_DENSITY filter
    # ('conservative' / 'standard' / 'verbose') that trims to spec.
    "Transcribe spoken dialogue accurately with punctuation and formatting. "
    "Dialogue is the priority. Preserve proper nouns and show titles accurately. "
    "If speech is in a language other than English, preserve it in the original language. "
    "Emit bracketed non-speech audio-event tags whenever a sound is editorially "
    "meaningful for a Deaf or hard-of-hearing viewer — music, audience reactions, "
    "and prominent ambient sounds are routinely required for broadcast compliance. "
    "Permitted tags: [music], [laughter], [applause], [cheering], [crowd cheering], "
    "[gasps], [crying], [screaming], [phone rings], [doorbell], [knocking], "
    "[footsteps], [gunshot], [explosion], [siren], [car horn], [engine revving], "
    "[whistle blows], [whistle], [buzzer], [thunder], [wind], [whispering]. "
    "Use [music] for any musical passage (score, song, jingle, theme). "
    "Use [crowd cheering] or [cheering] for sports crowds. Use [whistle blows] for "
    "referee whistles in sports. Use [applause] for audible audience applause. "
    "Do not emit [inaudible], [noise], [pause], [sound effect], or [silence]. "
    "Do not stack multiple audio tags. Do not insert audio tags inside dialogue sentences — "
    "place them on their own line between sentences. "
    "When a sound is clearly present and editorially useful, emit the tag. "
    "Reserve omission for sounds that are ambiguous or genuinely imperceptible."
)


def _headers() -> Dict[str, str]:
    if not ASSEMBLYAI_API_KEY:
        raise ValueError("Missing ASSEMBLYAI_API_KEY environment variable")
    return {"authorization": ASSEMBLYAI_API_KEY, "content-type": "application/json"}


def _transcription_prompt() -> Optional[str]:
    override = os.getenv("ASSEMBLYAI_PROMPT")
    if override is not None:
        prompt = override.strip()
        return prompt or None
    return DEFAULT_TRANSCRIPTION_PROMPT


def submit_transcription_job(media_url: str, speaker_labels: bool = True,
                             language_detection: bool = True,
                             language_code: Optional[str] = None) -> str:
    """Submit a media URL to AssemblyAI for transcription.

    OPERATOR-CONFIRMED LANGUAGE (2026-06-09):
      When `language_code` is a non-empty, non-"auto" ISO-639-1 code (e.g.
      "ja", "no", "es"), the captioner has CONFIRMED the source language in the
      CC preflight modal. AssemblyAI requires that a pinned `language_code` and
      `language_detection` are MUTUALLY EXCLUSIVE — you cannot pin a language
      and also ask AAI to detect one. So when a code is pinned we set
      `language_code` and FORCE `language_detection` OFF, regardless of what the
      caller passed. The Base44 dispatch layer already sends
      language_detection=false in this case; forcing it here is defense in depth
      so a stray true can never reach AAI alongside a pinned code.

      When `language_code` is None/"auto", behaviour is unchanged: AAI
      auto-detects under the `language_detection` flag (the operator's explicit
      auto opt-out). SOC 2 CC8.1 — the source language AAI uses is the
      attributed operator decision, never a silent re-detect.
    """
    pinned = (language_code or "").strip().lower()
    if pinned in ("", "auto"):
        pinned = None

    payload: Dict[str, Any] = {
        "audio_url": media_url,
        "speech_models": ["universal-3-pro", "universal-2"],
        "speaker_labels": speaker_labels,
        "format_text": True,
    }
    if pinned:
        # Pinned language is authoritative — detection OFF (AAI rejects both).
        payload["language_code"] = pinned
        payload["language_detection"] = False
    else:
        payload["language_detection"] = language_detection

    prompt = _transcription_prompt()
    if prompt:
        payload["prompt"] = prompt

    response = requests.post(f"{ASSEMBLYAI_BASE_URL}/transcript",
                             json=payload, headers=_headers(), timeout=120)
    if response.status_code >= 400:
        raise RuntimeError(f"AssemblyAI submit failed ({response.status_code}): {response.text}")
    data = response.json()
    transcript_id = data.get("id")
    if not transcript_id:
        raise RuntimeError(f"AssemblyAI submit did not return transcript id: {data}")
    return transcript_id


def wait_for_transcription_result(transcript_id: str) -> Dict[str, Any]:
    elapsed = 0
    while elapsed < POLL_TIMEOUT_SECONDS:
        response = requests.get(f"{ASSEMBLYAI_BASE_URL}/transcript/{transcript_id}",
                                headers=_headers(), timeout=120)
        if response.status_code >= 400:
            raise RuntimeError(f"AssemblyAI polling failed ({response.status_code}): {response.text}")
        data = response.json()
        status = data.get("status")
        if status == "completed":
            return data
        if status == "error":
            raise RuntimeError(f"AssemblyAI transcription failed: {data.get('error', 'Unknown error')}")
        time.sleep(POLL_INTERVAL_SECONDS)
        elapsed += POLL_INTERVAL_SECONDS
    raise TimeoutError(f"AssemblyAI polling timed out after {POLL_TIMEOUT_SECONDS} seconds")


def fetch_transcript_result(transcript_id: str, require_completed: bool = True) -> Dict[str, Any]:
    response = requests.get(f"{ASSEMBLYAI_BASE_URL}/transcript/{transcript_id}",
                            headers=_headers(), timeout=120)
    if response.status_code >= 400:
        raise RuntimeError(f"AssemblyAI fetch failed ({response.status_code}): {response.text}")
    data = response.json()
    status = data.get("status")
    if status == "error":
        raise RuntimeError(f"AssemblyAI transcription failed: {data.get('error', 'Unknown error')}")
    if require_completed and status != "completed":
        raise RuntimeError(f"AssemblyAI transcript not completed (status={status})")
    return data


def _ms_to_srt_tc(ms: int) -> str:
    """Milliseconds → SRT timecode HH:MM:SS,mmm."""
    ms = max(0, int(ms))
    hh, rem = divmod(ms, 3600000)
    mm, rem = divmod(rem, 60000)
    ss, rem = divmod(rem, 1000)
    return f"{hh:02d}:{mm:02d}:{ss:02d},{rem:03d}"


def build_backbone_srt_from_utterances(assembly_result: Dict[str, Any]) -> str:
    """
    Build the backbone SRT LOCALLY from the normalized utterances the engine
    already holds — NO provider REST call.

    This is the provider-agnostic source of truth: every transcription
    provider (AssemblyAI, ElevenLabs Scribe) is normalized to the same
    `utterances[]` shape (speaker/start/end/text) before this runs, so the
    backbone is built identically regardless of who transcribed. The previous
    code fetched the SRT from AssemblyAI's /transcript/{id}/srt endpoint, which
    400'd for any non-AAI provider (the Scribe transcript id is unknown to AAI).
    Building locally removes the provider coupling entirely — there is nothing
    provider-specific left to get wrong. SOC 2 CC8.1 — the backbone is
    deterministically reproducible from the in-memory result.
    """
    utterances = assembly_result.get("utterances") or []
    blocks: List[str] = []
    idx = 0
    for utt in utterances:
        text = (utt.get("text") or "").strip()
        if not text:
            continue
        idx += 1
        start_ms = int(utt.get("start", 0) or 0)
        end_ms = int(utt.get("end", start_ms) or start_ms)
        if end_ms <= start_ms:
            end_ms = start_ms + 1
        blocks.append(
            f"{idx}\n{_ms_to_srt_tc(start_ms)} --> {_ms_to_srt_tc(end_ms)}\n{text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def build_caption_inputs(assembly_result: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Provider-agnostic caption-input builder. Works identically for AssemblyAI
    and ElevenLabs Scribe because both are normalized to the same result shape
    upstream (services.scribe.normalize_scribe_result / AAI's native shape).

    Backbone SRT is built LOCALLY from utterances — no REST call to any
    provider. The legacy AAI `/srt` endpoint fetch is retained ONLY as a
    fallback for the reformat_only path where a result may arrive without
    utterances (pure AAI re-format of a historic transcript).
    """
    backbone_srt_text = build_backbone_srt_from_utterances(assembly_result)

    # Fallback: if (and only if) there were no utterances to build from AND we
    # have a real AAI transcript id, pull the SRT from AAI. This never fires
    # for a Scribe run (Scribe always has utterances) and never cross-calls the
    # wrong provider.
    if not backbone_srt_text.strip():
        provider = (assembly_result.get("_provider") or "").lower()
        transcript_id = assembly_result.get("id")
        if provider in ("", "assemblyai") and transcript_id:
            backbone_srt_text = fetch_srt(transcript_id)

    spoken_tokens = build_word_timestamps_from_result(assembly_result)
    sound_tokens = extract_sound_tokens_from_json(assembly_result)
    if not sound_tokens:
        sound_tokens = extract_sound_tokens_from_srt(backbone_srt_text)
    merged_tokens = merge_and_dedup_tokens(spoken_tokens, sound_tokens)
    return backbone_srt_text, merged_tokens


def build_caption_inputs_from_assembly_result(assembly_result: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """Back-compat alias — delegates to the provider-agnostic builder."""
    return build_caption_inputs(assembly_result)


def fetch_srt(transcript_id: str) -> str:
    response = requests.get(f"{ASSEMBLYAI_BASE_URL}/transcript/{transcript_id}/srt",
                            headers={"authorization": ASSEMBLYAI_API_KEY}, timeout=120)
    if response.status_code >= 400:
        raise RuntimeError(f"AssemblyAI SRT fetch failed ({response.status_code}): {response.text}")
    return response.text


def build_word_timestamps_from_result(assembly_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    words = assembly_result.get("words") or []
    utterances = assembly_result.get("utterances") or []
    if utterances:
        # Preferred path: utterances carry the diarized speaker; each word
        # inherits its parent utterance's speaker. This is the speaker-correct
        # source of truth (FCC 47 CFR §79.1 — speaker identification).
        return _build_tokens_from_utterances(utterances)
    # Fallback path: only a flat word array is available. Some providers /
    # stored baselines leave per-word `speaker` null even when utterance-level
    # diarization existed. Reconcile word speakers from the utterance time
    # windows so the downstream segmenter still sees real speaker boundaries
    # (the dash / one-speaker-per-line rules depend on this). SOC 2 CC8.1 —
    # speaker identity is never silently lost on the fallback path.
    return _build_tokens_from_words(words, utterances)


def _reconcile_word_speaker(
    start_ms: int,
    end_ms: int,
    utterance_windows: List[Dict[str, Any]],
) -> Optional[str]:
    """Return the speaker whose utterance time-window contains this word.

    Used only on the flat-words fallback path when a word carries no speaker.
    A word belongs to the utterance window it overlaps most (by midpoint).
    Deterministic and pure — identical inputs always resolve identically."""
    if not utterance_windows:
        return None
    mid = (start_ms + end_ms) // 2 if end_ms >= start_ms else start_ms
    for win in utterance_windows:
        if win["start"] <= mid <= win["end"]:
            return win["speaker"]
    return None


def _build_tokens_from_words(
    words: List[Dict[str, Any]],
    utterances: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    # Pre-build utterance time-windows once so per-word reconciliation is O(1)
    # amortised — never re-scans utterances per word. Each window carries its
    # utterance INDEX so a reconciled word can also recover its source-utterance
    # identity (the pause-boundary invariant's anchor on the flat-words path).
    utterance_windows: List[Dict[str, Any]] = []
    for u_index, u in enumerate(utterances or []):
        sp = u.get("speaker")
        if sp is None:
            continue
        try:
            us = int(u.get("start", 0))
            ue = int(u.get("end", us))
        except (TypeError, ValueError):
            continue
        utterance_windows.append({"speaker": sp, "start": us, "end": ue, "index": u_index})

    def _reconcile_word_utterance_id(s_ms: int, e_ms: int):
        """Source-utterance index whose time window contains this word's midpoint
        (flat-words path). None when no window matches. Mirrors
        _reconcile_word_speaker so speaker + utterance id come from the same
        window — keeping the pause boundary and speaker ownership consistent."""
        if not utterance_windows:
            return None
        mid = (s_ms + e_ms) // 2 if e_ms >= s_ms else s_ms
        for win in utterance_windows:
            if win["start"] <= mid <= win["end"]:
                return win["index"]
        return None

    tokens: List[Dict[str, Any]] = []
    for word in words:
        text = (word.get("text") or "").strip()
        if not text:
            continue
        start_ms = int(word.get("start", 0))
        end_ms = int(word.get("end", start_ms))
        speaker = word.get("speaker")
        # Backfill a missing word-level speaker from the utterance windows so
        # the segmenter sees real speaker boundaries even when the provider /
        # stored baseline left per-word speaker null.
        if speaker is None and utterance_windows:
            speaker = _reconcile_word_speaker(start_ms, end_ms, utterance_windows)
        tokens.append({"text": text, "start_ms": start_ms, "end_ms": end_ms,
                       "speaker": speaker,
                       "source_utterance_id": _reconcile_word_utterance_id(start_ms, end_ms)})
    return tokens


def _build_tokens_from_utterances(utterances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tokens: List[Dict[str, Any]] = []
    for utt_index, utterance in enumerate(utterances):
        utt_speaker = utterance.get("speaker")
        # SOURCE-UTTERANCE IDENTITY (the pause-boundary invariant's anchor).
        # Every word carries the index of the provider utterance it came from so
        # segmentation can enforce the hard inter-utterance pause boundary
        # (segmentation.segment_into_sentence_groups): a cue may never span a
        # ≥pause_boundary_ms silence BETWEEN two distinct source utterances. A
        # long hesitation WITHIN one utterance shares one id, so it never
        # over-fragments dramatic speech. SOC 2 CC8.1 — the boundary is provably
        # a function of source-utterance identity, not a blind gap heuristic.
        for word in (utterance.get("words") or []):
            text = (word.get("text") or "").strip()
            if not text:
                continue
            start_ms = int(word.get("start", 0))
            end_ms = int(word.get("end", start_ms))
            # Speaker precedence: the word's OWN speaker wins when present
            # (post-fix Scribe baselines carry it on every word); otherwise
            # inherit the utterance's authoritative speaker. A word can NEVER
            # arrive null when its utterance has a speaker — this is what
            # stops the packer fusing adjacent A/B utterances into one
            # mislabeled cue on the reformat-from-baseline path. SOC 2 CC8.1.
            speaker = word.get("speaker")
            if speaker is None:
                speaker = utt_speaker
            tokens.append({"text": text, "start_ms": start_ms, "end_ms": end_ms,
                           "speaker": speaker, "source_utterance_id": utt_index})
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
            tokens.append({
                "text": tag.strip().upper(),
                "start_ms": tag_start,
                "end_ms": max(tag_start + 1, tag_end),
                "speaker": None,
            })

    if tokens:
        return tokens

    matches = BRACKET_TAG_RE.findall(full_text)
    for tag in matches:
        tokens.append({"text": tag.strip().upper(), "start_ms": 0, "end_ms": 1, "speaker": None})
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
            tokens.append({
                "text": tag.strip().upper(),
                "start_ms": tag_start,
                "end_ms": max(tag_start + 1, tag_end),
                "speaker": None,
            })
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
        cues.append({"start_ms": tc_to_ms(start_tc), "end_ms": tc_to_ms(end_tc), "lines": text_lines})
    return cues


def tc_to_ms(tc: str) -> int:
    m = re.match(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", tc.strip())
    if not m:
        raise ValueError(f"Invalid SRT timecode: {tc}")
    hh, mm, ss, ms = map(int, m.groups())
    return ((hh * 60 + mm) * 60 + ss) * 1000 + ms


# =============================================================================
# extract_audio_events_from_assembly_result — promote bracketed tags from an
# AAI transcript into the SAME structured `audio_events[]` shape Scribe v2
# emits natively. Lets the Base44 ingester consume one provider-agnostic
# array (FCC 47 CFR §79.1 non-dialogue coverage on both transcription paths).
# Returns: [{event_type, start, end}, ...] in milliseconds.
# =============================================================================
_AAI_EVENT_TAG_MAP: List[Tuple[str, str]] = [
    ("music playing", "music_playing"),
    ("music",         "music"),
    ("crowd cheering","crowd_noise"),
    ("cheering",      "crowd_noise"),
    ("crowd noise",   "crowd_noise"),
    ("crowd",         "crowd_noise"),
    ("applause",      "applause"),
    ("laughter",      "laughter"),
    ("laughing",      "laughter"),
    ("gasps",         "gasping"),
    ("gasping",       "gasping"),
    ("screaming",     "screaming"),
    ("shouting",      "shouting"),
    ("crying",        "crying"),
    ("sobbing",       "crying"),
    ("whispering",    "whispering"),
    ("phone ringing", "ringtone"),
    ("phone rings",   "ringtone"),
    ("doorbell",      "ringtone"),
    ("alarm",         "alarm"),
    ("siren",         "siren"),
    ("buzzer",        "alarm"),
    ("whistle blows", "whistle"),
    ("whistle",       "whistle"),
    ("gunshot",       "gunshot"),
    ("gunfire",       "gunshot"),
    ("explosion",     "explosion"),
    ("engine revving","engine_noise"),
    ("engine",        "engine_noise"),
    ("car horn",      "engine_noise"),
    ("thunder",       "thunder"),
    ("rain",          "rain"),
    ("wind",          "rain"),
    ("footsteps",     "footsteps"),
    ("knocking",      "knocking"),
    ("knock",         "knocking"),
]


def _classify_aai_audio_tag(inner: str) -> Optional[str]:
    s = re.sub(r"\s+", " ", (inner or "")).strip().lower()
    if not s:
        return None
    for needle, event_type in _AAI_EVENT_TAG_MAP:
        if needle in s:
            return event_type
    return None


def extract_audio_events_from_assembly_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    utterances = result.get("utterances") or []
    for utt in utterances:
        text = (utt.get("text") or "").strip()
        if not text:
            continue
        start_ms = int(utt.get("start", 0))
        end_ms = int(utt.get("end", start_ms))
        matches = list(BRACKET_TAG_RE.finditer(text))
        if not matches:
            continue
        total_dur = max(1, end_ms - start_ms)
        per_tag = max(1, total_dur // max(1, len(matches)))
        for idx, m in enumerate(matches):
            inner = m.group(0).strip("[]")
            event_type = _classify_aai_audio_tag(inner)
            if not event_type:
                continue
            tag_start = start_ms + (idx * per_tag)
            tag_end = min(end_ms, tag_start + per_tag)
            out.append({
                "event_type": event_type,
                "start": tag_start,
                "end": max(tag_start + 300, tag_end),
            })
    # Merge adjacent same-type events (≤2s gap).
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
    return [e for e in merged if (e["end"] - e["start"]) >= 300]


def merge_and_dedup_tokens(spoken: List[Dict[str, Any]], sound: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged = spoken + sound
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


def is_sound_token(text: str) -> bool:
    text = (text or "").strip().upper()
    return text.startswith("[") and text.endswith("]")


def normalize_tokens(ts: Any) -> List[Dict[str, Any]]:
    if isinstance(ts, dict):
        if isinstance(ts.get("words"), list):
            return list(ts["words"])
        if isinstance(ts.get("utterances"), list):
            out: List[Dict[str, Any]] = []
            for utt in ts["utterances"]:
                if isinstance(utt, dict) and isinstance(utt.get("words"), list):
                    out.extend(utt["words"])
            return out
    if isinstance(ts, list):
        return ts
    return []
