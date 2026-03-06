import json
from typing import Any, Dict, List

from services.assembly import normalize_tokens, is_sound_token
from services.exporters import parse_srt, export_srt, export_vtt, export_scc
from services.qc import qc_report, violates_line_limits, count_function_word_endings
from services.readability import apply_readability_rules

MAX_LINES = 2
MAX_CHARS = 32
MIN_DIALOGUE_MS = 800
MIN_SOUND_MS = 800
SOUND_CLAMP_MS = 1200


def process_caption_job(
    backbone_srt_text: str,
    timestamps: Any,
    protected_phrases: List[str] | None = None,
    output_formats: List[str] | None = None,
) -> Dict[str, Any]:
    protected_phrases = protected_phrases or []
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
    print("[FORMATTER] Exporting SRT")

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

        text = token["text"].strip()
        if not text or is_sound_token(text):
            continue

        speaker = token.get("speaker") or "A"

        if current_run is None or current_run["speaker"] != speaker:
            current_run = {
                "speaker": speaker,
                "text_parts": [],
            }
            runs.append(current_run)

        current_run["text_parts"].append(text)

    for run in runs:
        run["text"] = " ".join(run["text_parts"]).replace(" ,", ",").replace(" .", ".")
        del run["text_parts"]

    return runs


def build_sound_cues(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for token in tokens:
        text = token["text"].strip()
        if is_sound_token(text):
            start = int(token["start_ms"])
            out.append(
                {
                    "idx": 0,
                    "start_ms": start,
                    "end_ms": start + SOUND_CLAMP_MS,
                    "lines": [text],
                    "type": "sound",
                    "meta": {},
                }
            )

    return out


def merge_timeline(dialogue_cues: List[Dict[str, Any]], sound_cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    combined = dialogue_cues + sound_cues
    combined.sort(key=lambda c: (c["start_ms"], c["end_ms"], 0 if c["type"] == "sound" else 1))
    return resolve_overlaps(combined)


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
        total_length = sum(lengths)
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
                    "meta": {
                        "runs": [run],
                        "two_speaker": False,
                    },
                }
            )
            current_start = new_end

    return resolve_overlaps(out)


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
                )
                line = broken[0][: MAX_CHARS - 2] if broken else run["text"][: MAX_CHARS - 2]
                lines.append(f"- {line}")
            cue["lines"] = lines[:2]
            formatted.append(cue)
            continue

        text = " ".join([r["text"] for r in runs]).strip()
        broken = heuristic_split(
            text=text,
            max_chars=MAX_CHARS,
            max_lines=MAX_LINES,
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
            )
            if retry_broken:
                cue["lines"] = retry_broken[:MAX_LINES]

        formatted.append(cue)

    return resolve_overlaps(formatted)


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
        "meta": {
            "runs": [{"speaker": "A", "text": left_text}],
            "two_speaker": False,
        },
    }

    right_cue = {
        **cue,
        "start_ms": split_time,
        "end_ms": cue["end_ms"],
        "lines": [right_text],
        "meta": {
            "runs": [{"speaker": "A", "text": right_text}],
            "two_speaker": False,
        },
    }

    return [left_cue, right_cue]


def heuristic_split(text: str, max_chars: int = 32, max_lines: int = 2) -> List[str]:
    text = (text or "").strip()
    if not text:
        return [""]

    words = text.split()
    if len(text) <= max_chars:
        return [text]

    best = None

    for i in range(1, len(words)):
        left = " ".join(words[:i])
        right = " ".join(words[i:])

        if len(left) <= max_chars and len(right) <= max_chars:
            score = abs(len(left) - len(right))
            if best is None or score < best[0]:
                best = (score, left, right)

    if best and max_lines == 2:
        return [best[1], best[2]]

    if max_lines == 1:
        return [text[:max_chars]]

    return [text[:max_chars], text[max_chars:max_chars * 2]]


def resolve_overlaps(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cues.sort(key=lambda c: (c["start_ms"], c["end_ms"]))

    for i in range(len(cues) - 1):
        current_cue = cues[i]
        next_cue = cues[i + 1]

        if current_cue["end_ms"] > next_cue["start_ms"]:
            shift = current_cue["end_ms"] - next_cue["start_ms"]
            next_cue["start_ms"] += shift

            if next_cue["end_ms"] <= next_cue["start_ms"]:
                next_cue["end_ms"] = next_cue["start_ms"] + 1

    return cues
