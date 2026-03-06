import os
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
    """
    Main production pipeline.

    Order:
    1) Parse AssemblyAI SRT backbone (canonical timing)
    2) Normalize timestamps JSON
    3) Build speaker runs per backbone cue
    4) Extract and insert sound cues as standalone cues
    5) Pre-split multi-speaker cues
    6) Run AI line breaking
    7) Retry failed splits
    8) Apply readability rules
    9) Resolve overlaps
    10) Export final files
    """
    protected_phrases = protected_phrases or []
    output_formats = output_formats or ["srt"]

    backbone = parse_srt(backbone_srt_text)
    cues_in = len(backbone)

    tokens = normalize_tokens(timestamps)
    tokens.sort(key=lambda w: (w["start_ms"], w["end_ms"]))

    # Build speaker runs aligned to backbone
    for cue in backbone:
        cue["meta"]["runs"] = build_speaker_runs_for_cue(cue, tokens)

    # Build standalone sound cues
    sound_cues = build_sound_cues(tokens)

    # Merge timeline and guarantee no overlap
    cues = merge_timeline(backbone, sound_cues)

    # Handle multi-speaker before AI
    cues = presplit_multispeaker(cues)

    # AI formatting + retry loop
    cues = format_with_retry_loops(cues, protected_phrases)

    # Readability layer (this is the piece you asked about)
    cues = apply_readability_rules(cues)

    # Hard safety net
    cues = resolve_overlaps(cues)

    # Reindex
    cues.sort(key=lambda c: (c["start_ms"], c["end_ms"]))
    for i, cue in enumerate(cues, start=1):
        cue["idx"] = i

    srt_out = export_srt(cues)
    vtt_out = export_vtt(cues) if "vtt" in output_formats else None
    scc_out = export_scc(cues) if "scc" in output_formats else None

    qc = qc_report(cues_in, cues, protected_phrases)

    return {
        "srt": srt_out,
        "vtt": vtt_out,
        "scc": scc_out,
        "qc": qc,
    }


def build_speaker_runs_for_cue(cue: Dict[str, Any], tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Align word-level AssemblyAI tokens to a single backbone cue and group by speaker.
    Excludes sound tokens from dialogue reconstruction.
    """
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
    """
    Extract bracketed sound tokens and convert them into standalone cues.
    IMPORTANT: We do NOT trust their durations from AssemblyAI.
    We clamp them to a readable duration.
    """
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
    """
    Merge dialogue cues and sound cues into one timeline.
    Then run overlap resolution.
    """
    combined = dialogue_cues + sound_cues
    combined.sort(key=lambda c: (c["start_ms"], c["end_ms"], 0 if c["type"] == "sound" else 1))
    return resolve_overlaps(combined)


def presplit_multispeaker(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    If a cue contains:
    - 1 speaker -> leave it
    - 2 speaker runs -> keep as one cue, later rendered as two dash lines
    - >2 speaker runs -> split into multiple cues before AI
    """
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

        # More than 2 speaker runs: split cue
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
    """
    AI line breaking with retry loops.

    Hard rules:
    - Max 2 lines
    - Max 32 chars per line

    If a cue fails validation:
    - split cue
    - re-run AI on child cues
    """
    formatted: List[Dict[str, Any]] = []

    for cue in cues:
        if cue["type"] == "sound":
            cue["lines"] = [cue["lines"][0][:MAX_CHARS]]
            formatted.append(cue)
            continue

        runs = cue["meta"].get("runs", [])
        is_two_speaker = bool(cue["meta"].get("two_speaker")) and len(runs) == 2

        if is_two_speaker:
            # one speaker per line, each line prefixed with dash
            lines: List[str] = []
            for run in runs:
                broken = linebreak_ai(
                    text=run["text"],
                    protected_phrases=protected_phrases,
                    max_lines=1,
                    max_chars=MAX_CHARS - 2,
                )
                line = broken[0][: MAX_CHARS - 2] if broken else run["text"][: MAX_CHARS - 2]
                lines.append(f"- {line}")
            cue["lines"] = lines[:2]
            formatted.append(cue)
            continue

        # Single-speaker
        text = " ".join([r["text"] for r in runs]).strip()
        broken = linebreak_ai(
            text=text,
            protected_phrases=protected_phrases,
            max_lines=MAX_LINES,
            max_chars=MAX_CHARS,
        )
        cue["lines"] = broken[:MAX_LINES]

        # HARD validation
        if violates_line_limits(cue["lines"], MAX_LINES, MAX_CHARS):
            if max_rounds <= 0:
                cue["lines"] = [line[:MAX_CHARS] for line in cue["lines"][:MAX_LINES]]
                formatted.append(cue)
            else:
                children = split_cue(cue)
                child_results = format_with_retry_loops(children, protected_phrases, max_rounds=max_rounds - 1)
                formatted.extend(child_results)
            continue

        # SOFT retry: function word endings
        if count_function_word_endings(cue["lines"]) > 0 and max_rounds > 0:
            retry_broken = linebreak_ai(
                text=text,
                protected_phrases=protected_phrases,
                max_lines=MAX_LINES,
                max_chars=MAX_CHARS,
                retry_hint="Avoid ending lines with function words. Choose a cleaner break.",
            )
            if retry_broken:
                cue["lines"] = retry_broken[:MAX_LINES]

        formatted.append(cue)

    return resolve_overlaps(formatted)


def split_cue(cue: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Deterministic split for a cue when validation fails.
    Split text by midpoint and divide timing proportionally.
    """
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


def linebreak_ai(
    text: str,
    protected_phrases: List[str],
    max_lines: int = 2,
    max_chars: int = 32,
    retry_hint: str | None = None,
) -> List[str]:
    """
    Use OpenAI if available, otherwise fallback to deterministic heuristic.
    """
    text = (text or "").strip()
    if not text:
        return [""]

    if not os.getenv("OPENAI_API_KEY"):
        return heuristic_split(text, max_chars=max_chars, max_lines=max_lines)

    try:
        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        system_prompt = (
            "You are a professional broadcast closed-caption line breaker.\n"
            f"Maximum {max_lines} lines.\n"
            f"Maximum {max_chars} characters per line.\n"
            "Do not split protected phrases or named entities.\n"
            "Avoid ending a line with function words.\n"
            "Prefer punctuation and phrase boundaries.\n"
            'Return JSON only in the form: {"lines":["...","..."],"needs_split":false}\n'
            'If impossible, return: {"lines":[],"needs_split":true}\n'
        )

        if retry_hint:
            system_prompt += f"\nExtra instruction: {retry_hint}\n"

        payload = {
            "text": text,
            "protected_phrases": protected_phrases or [],
        }

        response = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload)},
            ],
            temperature=0.2,
        )

        parsed = json.loads(response.output_text.strip())
        lines = parsed.get("lines") or []

        if not lines:
            return heuristic_split(text, max_chars=max_chars, max_lines=max_lines)

        return [line[:max_chars] for line in lines[:max_lines]]

    except Exception:
        return heuristic_split(text, max_chars=max_chars, max_lines=max_lines)


def heuristic_split(text: str, max_chars: int = 32, max_lines: int = 2) -> List[str]:
    """
    Fallback deterministic splitter if OpenAI is unavailable.
    """
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

    return [text[:max_chars], text[max_chars : max_chars * 2]]


def resolve_overlaps(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Hard safety net:
    guarantee no overlapping cues in final timeline.
    """
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
