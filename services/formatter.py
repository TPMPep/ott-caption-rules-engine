from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import re

@dataclass
class Caption:
    start_ms: int
    end_ms: int
    lines: List[str]
    speaker: str | None = None
    kind: str = "caption"  # "caption" or "sfx"

DEFAULT_RULES = {
    "maxLines": 2,
    "maxCharsPerLine": 32,
    "minCaptionMs": 900,
    "maxCaptionMs": 6000,
    "pauseBreakMs": 650,
    "maxCPS": 17.0,  # characters per second
    "forceUpperSfx": True,
    "foreignLanguageTag": "[FOREIGN LANGUAGE]",
    "dashSecondSpeakerLine": True,
}

PUNCT_END_RE = re.compile(r"[.!?]$")

def _merge_rules(rules: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(DEFAULT_RULES)
    merged.update({k: v for k, v in (rules or {}).items() if v is not None})
    return merged

def build_captions_from_assembly(transcript: Dict[str, Any], rules: Dict[str, Any]) -> Tuple[List[Caption], Dict[str, Any]]:
    r = _merge_rules(rules)

    lang = transcript.get("language_code") or transcript.get("language") or "en"
    words = transcript.get("words") or []

    meta = {"language_code": lang, "wordCount": len(words)}

    # If it’s entirely non-English: single tag across duration (simple and CC-correct)
    if lang and not str(lang).lower().startswith("en"):
        audio_dur = transcript.get("audio_duration") or 0
        end_ms = int(float(audio_dur) * 1000) if audio_dur else (words[-1]["end"] if words else 1000)
        return [Caption(0, end_ms, [r["foreignLanguageTag"]])], meta

    # AssemblyAI can also provide "utterances" w/ speaker; prefer those if present
    utterances = transcript.get("utterances") or []
    if utterances:
        return _captions_from_utterances(utterances, r), meta

    # Fallback: build from words (speaker may be absent)
    return _captions_from_words(words, r), meta

def _captions_from_utterances(utterances: List[Dict[str, Any]], r: Dict[str, Any]) -> List[Caption]:
    captions: List[Caption] = []
    for utt in utterances:
        speaker = str(utt.get("speaker")) if utt.get("speaker") is not None else None
        start_ms = int(utt.get("start", 0))
        end_ms = int(utt.get("end", 0))
        text = (utt.get("text") or "").strip()
        if not text:
            continue

        # Segment utterance into caption-sized chunks
        pieces = _segment_text_with_timing(text, start_ms, end_ms, r)
        for p_start, p_end, p_text in pieces:
            lines = _wrap_to_lines(p_text, r)
            captions.append(Caption(p_start, p_end, lines, speaker=speaker))
    return _post_process_speakers(captions, r)

def _captions_from_words(words: List[Dict[str, Any]], r: Dict[str, Any]) -> List[Caption]:
    captions: List[Caption] = []
    if not words:
        return captions

    buf = []
    seg_start = int(words[0]["start"])
    last_end = int(words[0]["end"])
    last_text = str(words[0].get("text",""))

    def flush(seg_end: int):
        nonlocal buf, seg_start
        text = " ".join(buf).strip()
        if text:
            lines = _wrap_to_lines(text, r)
            captions.append(Caption(seg_start, seg_end, lines))
        buf = []

    for w in words:
        t = str(w.get("text",""))
        s = int(w.get("start", last_end))
        e = int(w.get("end", s))
        pause = s - last_end

        # Sentence / pause breaks
        if buf and (pause >= r["pauseBreakMs"] or PUNCT_END_RE.search(last_text)):
            flush(last_end)
            seg_start = s

        buf.append(t)
        last_end = e
        last_text = t

    if buf:
        flush(last_end)

    # Now enforce timing + CPS + max duration by splitting captions that are too dense
    captions = _enforce_density_and_duration(captions, r)
    return captions

def _segment_text_with_timing(text: str, start_ms: int, end_ms: int, r: Dict[str, Any]):
    # Basic clause segmentation by punctuation; then enforce duration and CPS.
    duration = max(1, end_ms - start_ms)
    clauses = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(clauses) == 1:
        return [(start_ms, end_ms, text.strip())]

    pieces = []
    cur = []
    cur_len = 0
    for c in clauses:
        c = c.strip()
        if not c:
            continue
        if cur and (cur_len + len(c) > 64):  # rough pre-chunking
            pieces.append(" ".join(cur))
            cur = [c]
            cur_len = len(c)
        else:
            cur.append(c)
            cur_len += len(c) + 1
    if cur:
        pieces.append(" ".join(cur))

    # Map chunks across the time range proportionally
    total_chars = sum(len(p) for p in pieces) or 1
    out = []
    cursor = start_ms
    for p in pieces:
        frac = len(p) / total_chars
        seg_dur = int(duration * frac)
        seg_end = min(end_ms, cursor + max(r["minCaptionMs"], seg_dur))
        out.append((cursor, seg_end, p))
        cursor = seg_end
    # Ensure last ends at end_ms
    if out:
        out[-1] = (out[-1][0], end_ms, out[-1][2])
    return out

def _enforce_density_and_duration(caps: List[Caption], r: Dict[str, Any]) -> List[Caption]:
    out: List[Caption] = []
    for c in caps:
        dur = max(1, c.end_ms - c.start_ms)
        text = " ".join(c.lines).replace("\n", " ").strip()
        max_chars = int((dur / 1000.0) * float(r["maxCPS"]))
        # absolute cap: 2*32=64 chars visible, but we may have slightly more across timing
        hard_max = max(64, max_chars)

        if dur > r["maxCaptionMs"] or len(text) > hard_max:
            # Split at best break (punct, then space nearest middle)
            parts = _split_text_smart(text)
            if len(parts) == 1:
                parts = _split_text_mid(text)
            # Split timing evenly
            step = dur // len(parts)
            s = c.start_ms
            for i, p in enumerate(parts):
                e = c.end_ms if i == len(parts)-1 else s + max(r["minCaptionMs"], step)
                out.append(Caption(s, e, _wrap_to_lines(p, r), speaker=c.speaker))
                s = e
        else:
            # enforce min duration
            if dur < r["minCaptionMs"]:
                c.end_ms = c.start_ms + r["minCaptionMs"]
            out.append(Caption(c.start_ms, c.end_ms, _wrap_to_lines(text, r), speaker=c.speaker))
    return out

def _split_text_smart(text: str) -> List[str]:
    # Prefer punctuation boundaries
    parts = re.split(r"(?<=[,:;.!?])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        return [text]
    # Merge tiny parts
    merged = []
    buf = ""
    for p in parts:
        if not buf:
            buf = p
        elif len(buf) < 18:
            buf = f"{buf} {p}"
        else:
            merged.append(buf)
            buf = p
    if buf:
        merged.append(buf)
    return merged

def _split_text_mid(text: str) -> List[str]:
    words = text.split()
    if len(words) < 6:
        return [text]
    mid = len(words)//2
    return [" ".join(words[:mid]), " ".join(words[mid:])]

def _wrap_to_lines(text: str, r: Dict[str, Any]) -> List[str]:
    max_len = int(r["maxCharsPerLine"])
    max_lines = int(r["maxLines"])

    text = re.sub(r"\s+", " ", text).strip()

    # If fits in one line
    if len(text) <= max_len:
        return [text]

    # Greedy build lines but balance (2 lines)
    if max_lines == 2:
        # Find best split near middle, but <= max_len for line 1
        best = None
        target = len(text) // 2
        for i in range(max(0, target-20), min(len(text), target+20)):
            if text[i] == " ":
                left = text[:i].strip()
                right = text[i+1:].strip()
                if len(left) <= max_len and len(right) <= max_len:
                    score = abs(len(left) - len(right))
                    if best is None or score < best[0]:
                        best = (score, left, right)
        if best:
            l1, l2 = best[1], best[2]
        else:
            # fallback: split at last space <= max_len
            cut = text.rfind(" ", 0, max_len+1)
            if cut == -1:
                cut = max_len
            l1 = text[:cut].strip()
            l2 = text[cut:].strip()

        # Orphan fix: if second line is 1 word, rebalance
        if len(l2.split()) == 1 and len(l1.split()) >= 3:
            words1 = l1.split()
            l2 = f"{words1[-1]} {l2}".strip()
            l1 = " ".join(words1[:-1]).strip()

        return [l1, l2]

    # Generic wrap for >2 lines (rare for CC)
    lines = []
    cur = ""
    for w in text.split():
        if not cur:
            cur = w
        elif len(cur) + 1 + len(w) <= max_len:
            cur = f"{cur} {w}"
        else:
            lines.append(cur)
            cur = w
            if len(lines) >= max_lines:
                break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    return lines

def _post_process_speakers(caps: List[Caption], r: Dict[str, Any]) -> List[Caption]:
    # If two different speakers occur in the same time window, convert to dashed two-line style.
    # (Simple heuristic: if adjacent captions overlap or are very close and speakers differ)
    out: List[Caption] = []
    i = 0
    while i < len(caps):
        c = caps[i]
        if i+1 < len(caps):
            n = caps[i+1]
            close = abs(n.start_ms - c.end_ms) <= 120
            if close and c.speaker and n.speaker and c.speaker != n.speaker:
                # Merge into one dashed caption block if timing overlaps/adjacent and both fit
                merged_start = c.start_ms
                merged_end = n.end_ms
                l1 = " ".join(c.lines).replace("\n", " ")
                l2 = " ".join(n.lines).replace("\n", " ")
                if r.get("dashSecondSpeakerLine", True):
                    l2 = f"- {l2}" if not l2.startswith("-") else l2
                out.append(Caption(merged_start, merged_end, _wrap_to_lines(f"{l1} {l2}".replace("  "," "), r)))
                i += 2
                continue
        out.append(c)
        i += 1
    return out
