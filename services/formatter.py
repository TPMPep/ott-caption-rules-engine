import re
from typing import Any, Dict, List, Tuple

from services.assembly import normalize_tokens, is_sound_token
from services.exporters import parse_srt, export_srt, export_vtt, export_scc
from services.qc import qc_report, violates_line_limits, count_function_word_endings
from services.readability import apply_readability_rules

MAX_LINES = 2
MAX_CHARS = 32
MIN_DIALOGUE_MS = 800
MIN_SOUND_MS = 1000
SOUND_CLAMP_MS = 1400
MUSIC_MIN_DURATION_MS = 5000

PROTECTED_DEFAULTS = [
    "Watch What Happens Live",
]

FUNCTION_WORDS = {
    "a", "an", "the", "of", "to", "and", "or", "but",
    "with", "from", "in", "on", "at", "for", "that",
    "this", "these", "those", "is", "are", "was", "were",
}


def process_caption_job(
    backbone_srt_text: str,
    timestamps: Any,
    protected_phrases: List[str] | None = None,
    output_formats: List[str] | None = None,
) -> Dict[str, Any]:
    protected_phrases = list(dict.fromkeys((protected_phrases or []) + PROTECTED_DEFAULTS))
    output_formats = output_formats or ["srt"]

    print("[FORMATTER] Parsing backbone SRT")
    backbone = parse_srt(backbone_srt_text)
    cues_in = len(backbone)
    print(f"[FORMATTER] Backbone cues: {cues_in}")

    print("[FORMATTER] Normalizing timestamps")
    tokens = normalize_tokens(timestamps)
    tokens.sort(key=lambda w: (w["start_ms"], w["end_ms"]))
    print(f"[FORMATTER] Tokens: {len(tokens)}")

    print("[FORMATTER] Building speaker runs")
    for cue in backbone:
        cue["meta"]["runs"] = build_speaker_runs_for_cue(cue, tokens)
        cue["type"] = "dialogue"

    print("[FORMATTER] Building sound cues")
    sound_cues = build_sound_cues(tokens)
    print(f"[FORMATTER] Sound cues: {len(sound_cues)}")

    print("[FORMATTER] Merging timeline")
    cues = merge_timeline(backbone, sound_cues)
    print(f"[FORMATTER] Timeline cues after merge: {len(cues)}")

    print("[FORMATTER] Pre-splitting multi-speaker cues")
    cues = presplit_multispeaker(cues)
    print(f"[FORMATTER] Timeline cues after pre-split: {len(cues)}")

    print("[FORMATTER] Formatting cues with retry loops")
    cues = format_with_retry_loops(cues, protected_phrases)
    print(f"[FORMATTER] Cues after formatting: {len(cues)}")

    print("[FORMATTER] Applying readability rules")
    cues = apply_readability_rules(cues)

    print("[FORMATTER] Resolving overlaps")
    cues = resolve_overlaps(cues)
    print("[FORMATTER] Overlaps resolved")

    cues.sort(key=lambda c: (c["start_ms"], c["end_ms"]))
    for i, cue in enumerate(cues, start=1):
        cue["idx"] = i

    print("[FORMATTER] Exporting outputs")
    srt_out = export_srt(cues)
    print("[FORMATTER] SRT export complete")
    vtt_out = export_vtt(cues) if "vtt" in output_formats else None
    scc_out = export_scc(cues) if "scc" in output_formats else None

    print("[FORMATTER] Running QC")
    qc = qc_report(cues_in, cues, protected_phrases)
    print("[FORMATTER] QC complete")

    print("[FORMATTER] Formatter completed")
    return {
        "srt": srt_out,
        "vtt": vtt_out,
        "scc": scc_out,
        "qc": qc,
    }


def build_speaker_runs_for_cue(cue: Dict[str, Any], tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    current_run = None

    for token in tokens:
        if token["end_ms"] < cue["start_ms"]:
            continue
        if token["start_ms"] > cue["end_ms"]:
            break

        text = (token["text"] or "").strip()
        if not text or is_sound_token(text):
            continue

        speaker = token.get("speaker") or "A"

        if current_run is None or current_run["speaker"] != speaker:
            current_run = {"speaker": speaker, "text_parts": []}
            runs.append(current_run)

        current_run["text_parts"].append(text)

    for run in runs:
        run["text"] = clean_dialogue_text(" ".join(run["text_parts"]))
        del run["text_parts"]

    return runs


def build_sound_cues(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    raw_events: List[Dict[str, Any]] = []

    for token in tokens:
        text = (token["text"] or "").strip()
        if not is_sound_token(text):
            continue

        label = normalize_sound_label(text)
        if not label:
            continue

        start = int(token["start_ms"])
        end = int(token["end_ms"] if token["end_ms"] > token["start_ms"] else token["start_ms"] + 1)

        raw_events.append({"label": label, "start_ms": start, "end_ms": end})

    merged = merge_sound_events(raw_events)
    filtered = filter_sound_events(merged)

    out: List[Dict[str, Any]] = []
    for ev in filtered:
        start = ev["start_ms"]
        end = max(ev["end_ms"], start + MIN_SOUND_MS)

        out.append(
            {
                "idx": 0,
                "start_ms": start,
                "end_ms": end,
                "lines": [ev["label"]],
                "type": "sound",
                "meta": {
                    "sound_label": ev["label"],
                    "sound_priority": sound_priority(ev["label"]),
                },
            }
        )

    return out


def normalize_sound_label(text: str) -> str:
    label = text.strip().upper()

    if not label.startswith("[") or not label.endswith("]"):
        return ""

    inner = label[1:-1].strip()
    if not inner:
        return ""

    parts = [p.strip() for p in re.split(r"\s+AND\s+|,\s*", inner) if p.strip()]
    deduped: List[str] = []
    for p in parts:
        if p not in deduped:
            deduped.append(p)

    if deduped == ["NOISE"] or deduped == ["SOUND"]:
        return ""

    if "NOISE" in deduped and any(x in deduped for x in ["APPLAUSE", "LAUGHTER", "CHEERING", "MUSIC"]):
        deduped = [p for p in deduped if p != "NOISE"]

    if "MUSIC" in deduped and any(x in deduped for x in ["APPLAUSE", "LAUGHTER", "CHEERING"]):
        deduped = [p for p in deduped if p != "MUSIC"]

    cleaned = " AND ".join(deduped).strip()
    if not cleaned:
        return ""

    if cleaned == "MUSIC":
        return "[♪]"
    return f"[{cleaned}]"


def merge_sound_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not events:
        return []

    events = sorted(events, key=lambda e: (e["start_ms"], e["end_ms"]))
    merged: List[Dict[str, Any]] = [events[0].copy()]

    for ev in events[1:]:
        prev = merged[-1]

        same_label = ev["label"] == prev["label"]
        close_gap = ev["start_ms"] - prev["end_ms"] <= 700
        overlap = ev["start_ms"] <= prev["end_ms"]

        if same_label and (close_gap or overlap):
            prev["end_ms"] = max(prev["end_ms"], ev["end_ms"])
            continue

        if ev["start_ms"] - prev["end_ms"] <= 150 and abs(ev["start_ms"] - prev["start_ms"]) <= 500:
            combined = combine_sound_labels(prev["label"], ev["label"])
            prev["label"] = combined
            prev["end_ms"] = max(prev["end_ms"], ev["end_ms"])
            continue

        merged.append(ev.copy())

    return merged


def filter_sound_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    last_music_end = -999999

    for ev in events:
        label = ev["label"]
        duration = ev["end_ms"] - ev["start_ms"]

        if label == "[♪]" and duration < MUSIC_MIN_DURATION_MS:
            continue

        if label == "[♪]" and ev["start_ms"] - last_music_end < 5000:
            continue

        if label == "[♪]":
            last_music_end = ev["end_ms"]

        filtered.append(ev)

    return filtered


def sound_priority(label: str) -> int:
    if label in ("[LAUGHTER]", "[APPLAUSE]", "[CHEERING]"):
        return 3
    if label == "[♪]":
        return 1
    return 2


def combine_sound_labels(a: str, b: str) -> str:
    if a == "[♪]" and b == "[♪]":
        return "[♪]"

    def split_label(lbl: str) -> List[str]:
        inner = lbl.strip()[1:-1]
        if not inner:
            return []
        if inner == "♪":
            return ["♪"]
        return [p.strip() for p in inner.split(" AND ") if p.strip()]

    parts: List[str] = []
    for piece in split_label(a) + split_label(b):
        if piece and piece not in parts:
            parts.append(piece)

    if parts == ["♪"]:
        return "[♪]"

    return f"[{' AND '.join(parts)}]"


def merge_timeline(dialogue_cues: List[Dict[str, Any]], sound_cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    combined = dialogue_cues + sound_cues
    combined.sort(key=lambda c: (c["start_ms"], c["end_ms"], 0 if c["type"] == "dialogue" else 1))
    return combined


def presplit_multispeaker(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for cue in cues:
        if cue["type"] == "sound":
            out.append(cue)
            continue

        runs = cue["meta"].get("runs", [])

        if len(runs) <= 1:
            out.append(cue)
            continue

        if len(runs) == 2:
            cue["meta"]["two_speaker"] = True
            out.append(cue)
            continue

        total_duration = max(1, cue["end_ms"] - cue["start_ms"])
        lengths = [max(1, len(r["text"])) for r in runs]
        total_length = max(1, sum(lengths))
        current_start = cue["start_ms"]

        for run, length in zip(runs, lengths):
            dur = int(total_duration * (length / total_length))
            new_end = current_start + max(1, dur)

            out.append(
                {
                    "idx": 0,
                    "start_ms": current_start,
                    "end_ms": new_end,
                    "lines": [],
                    "type": "dialogue",
                    "meta": {"runs": [run], "two_speaker": False},
                }
            )
            current_start = new_end

    return out


def format_with_retry_loops(
    cues: List[Dict[str, Any]],
    protected_phrases: List[str],
    max_rounds: int = 3,
) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []

    for idx, cue in enumerate(cues):
        if idx % 50 == 0:
            print(f"[FORMATTER] Processing cue {idx + 1}/{len(cues)}")

        if cue["type"] == "sound":
            cue["lines"] = [cue["lines"][0][:MAX_CHARS]]
            formatted.append(cue)
            continue

        runs = cue["meta"].get("runs", [])
        is_two_speaker = bool(cue["meta"].get("two_speaker")) and len(runs) == 2

        if is_two_speaker:
            lines: List[str] = []
            for run in runs:
                broken = heuristic_split(
                    text=run["text"],
                    max_chars=MAX_CHARS - 2,
                    max_lines=1,
                    protected_phrases=protected_phrases,
                )
                line = broken[0][: MAX_CHARS - 2] if broken else run["text"][: MAX_CHARS - 2]
                lines.append(f"- {line}")
            cue["lines"] = lines[:2]
            formatted.append(cue)
            continue

        text = clean_dialogue_text(" ".join([r["text"] for r in runs]).strip())
        broken = heuristic_split(
            text=text,
            max_chars=MAX_CHARS,
            max_lines=MAX_LINES,
            protected_phrases=protected_phrases,
        )
        cue["lines"] = broken[:MAX_LINES]

        if violates_line_limits(cue["lines"], MAX_LINES, MAX_CHARS):
            if max_rounds <= 0:
                cue["lines"] = [line[:MAX_CHARS] for line in cue["lines"][:MAX_LINES]]
                formatted.append(cue)
            else:
                children = split_cue(cue)
                child_results = format_with_retry_loops(children, protected_phrases, max_rounds=max_rounds - 1)
                formatted.extend(child_results)
            continue

        if count_function_word_endings(cue["lines"]) > 0 and max_rounds > 0:
            retry_broken = heuristic_split(
                text=text,
                max_chars=MAX_CHARS,
                max_lines=MAX_LINES,
                protected_phrases=protected_phrases,
            )
            if retry_broken:
                cue["lines"] = retry_broken[:MAX_LINES]

        formatted.append(cue)

    return formatted


def split_cue(cue: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = " ".join(cue["lines"]).strip()
    if not text:
        return [cue]

    words = text.split()
    if len(words) < 2:
        return [cue]

    midpoint = len(words) // 2
    left_text = " ".join(words[:midpoint]).strip()
    right_text = " ".join(words[midpoint:]).strip()

    total_duration = max(1, cue["end_ms"] - cue["start_ms"])
    denominator = max(1, len(left_text) + len(right_text))
    split_time = cue["start_ms"] + int(total_duration * (len(left_text) / denominator))

    left_cue = {
        **cue,
        "start_ms": cue["start_ms"],
        "end_ms": split_time,
        "lines": [left_text],
        "meta": {"runs": [{"speaker": "A", "text": left_text}], "two_speaker": False},
    }

    right_cue = {
        **cue,
        "start_ms": split_time,
        "end_ms": cue["end_ms"],
        "lines": [right_text],
        "meta": {"runs": [{"speaker": "A", "text": right_text}], "two_speaker": False},
    }

    return [left_cue, right_cue]


def heuristic_split(
    text: str,
    max_chars: int = 32,
    max_lines: int = 2,
    protected_phrases: List[str] | None = None,
) -> List[str]:
    text = (text or "").strip()
    if not text:
        return [""]

    protected_phrases = protected_phrases or []

    if len(text) <= max_chars:
        return [text]

    words = text.split()
    best: Tuple[int, List[str]] | None = None

    if max_lines == 1:
        return [fit_single_line(text, max_chars, protected_phrases)]

    for i in range(1, len(words)):
        left = " ".join(words[:i]).strip()
        right = " ".join(words[i:]).strip()

        if len(left) > max_chars or len(right) > max_chars:
            continue

        score = split_score(left, right, protected_phrases)
        candidate = [left, right]

        if best is None or score < best[0]:
            best = (score, candidate)

    if best:
        return best[1]

    return [fit_single_line(text, max_chars, protected_phrases), text[max_chars:max_chars * 2].strip()]


def fit_single_line(text: str, max_chars: int, protected_phrases: List[str]) -> str:
    if len(text) <= max_chars:
        return text

    trimmed = text[:max_chars].rstrip()

    if len(trimmed) < len(text) and not text[len(trimmed)].isspace():
        trimmed = " ".join(trimmed.split()[:-1]).strip() or trimmed[:max_chars]

    for phrase in protected_phrases:
        normalized = phrase.strip()
        if not normalized:
            continue
        if normalized in text and normalized.replace(" ", "\n") in (trimmed + "\n" + text[len(trimmed):]):
            trimmed = " ".join(trimmed.split()[:-1]).strip()

    return trimmed[:max_chars].rstrip()


def split_score(left: str, right: str, protected_phrases: List[str]) -> int:
    score = 0
    score += abs(len(left) - len(right))

    if not re.search(r"[,\.\?!:;]$", left):
        score += 8

    left_last = sanitize_last_word(left)
    if left_last in FUNCTION_WORDS:
        score += 15

    combined = left + "\n" + right
    for phrase in protected_phrases:
        if phrase and phrase in combined.replace("\n", " ") and phrase.replace(" ", "\n") in combined:
            score += 100

    right_words = len(right.split())
    if right_words <= 2:
        score += 12

    return score


def sanitize_last_word(text: str) -> str:
    text = re.sub(r"[^\w']+$", "", text.strip().lower())
    parts = text.split()
    return parts[-1] if parts else ""


def clean_dialogue_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    return text


def resolve_overlaps(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cues.sort(key=lambda c: (c["start_ms"], c["end_ms"], 0 if c["type"] == "dialogue" else 1))

    resolved: List[Dict[str, Any]] = []

    for cue in cues:
        if not resolved:
            resolved.append(cue)
            continue

        prev = resolved[-1]

        if cue["type"] == "sound" and prev["type"] == "dialogue":
            if cue["start_ms"] < prev["end_ms"]:
                cue["start_ms"] = prev["end_ms"]
            if cue["end_ms"] - cue["start_ms"] < MIN_SOUND_MS:
                continue
            resolved.append(cue)
            continue

        if cue["type"] == "dialogue" and prev["type"] == "sound":
            if prev["end_ms"] > cue["start_ms"]:
                if prev["meta"].get("sound_priority", 1) <= 1:
                    prev["end_ms"] = min(prev["end_ms"], cue["start_ms"])
                    if prev["end_ms"] - prev["start_ms"] < MIN_SOUND_MS:
                        resolved.pop()
                else:
                    cue["start_ms"] = max(cue["start_ms"], prev["end_ms"])
                    if cue["end_ms"] <= cue["start_ms"]:
                        cue["end_ms"] = cue["start_ms"] + 1
            resolved.append(cue)
            continue

        if prev["end_ms"] > cue["start_ms"]:
            cue["start_ms"] = prev["end_ms"]
            if cue["end_ms"] <= cue["start_ms"]:
                cue["end_ms"] = cue["start_ms"] + 1

        resolved.append(cue)

    return resolved
