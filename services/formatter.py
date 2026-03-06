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
MUSIC_MIN_MS = 5000
SOUND_GAP_MIN_MS = 350
SOUND_PAD_MS = 80
NEARBY_SOUND_MERGE_MS = 500
SIMULTANEOUS_SOUND_MERGE_MS = 150

PROTECTED_DEFAULTS = [
    "Watch What Happens Live",
]

FUNCTION_WORDS = {
    "a", "an", "the", "of", "to", "and", "or", "but",
    "with", "from", "in", "on", "at", "for", "that",
    "this", "these", "those", "is", "are", "was", "were",
}

HIGH_VALUE_SOUND_LABELS = {
    "LAUGHTER",
    "APPLAUSE",
    "CHEERING",
    "GASP",
    "SIGHS",
    "SIGHS",
    "CRYING",
    "INAUDIBLE",
}

LOW_VALUE_SOUND_LABELS = {
    "NOISE",
    "SOUND",
    "SOUND EFFECT",
    "MUSIC",
}


# =========================
# Main entrypoint
# =========================
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
    print(f"[FORMATTER] Sound cues after policy filter: {len(sound_cues)}")

    print("[FORMATTER] Merging timeline with dialogue priority")
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

    cues = [cue for cue in cues if cue.get("lines") and any((line or "").strip() for line in cue["lines"])]
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


# =========================
# Dialogue helpers
# =========================
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
            current_run = {
                "speaker": speaker,
                "text_parts": [],
            }
            runs.append(current_run)

        current_run["text_parts"].append(text)

    for run in runs:
        run["text"] = clean_dialogue_text(" ".join(run["text_parts"]))
        del run["text_parts"]

    return runs


# =========================
# Sound cue policy
# =========================
def build_sound_cues(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    raw_events: List[Dict[str, Any]] = []

    for token in tokens:
        text = (token.get("text") or "").strip()
        if not is_sound_token(text):
            continue

        event = parse_sound_event(text, int(token["start_ms"]), int(token["end_ms"]))
        if not event:
            continue
        raw_events.append(event)

    merged = merge_sound_events(raw_events)
    filtered = filter_sound_events(merged)

    out: List[Dict[str, Any]] = []
    for ev in filtered:
        label_text = render_sound_label(ev)
        if not label_text:
            continue

        start = int(ev["start_ms"])
        end = int(max(ev["end_ms"], start + MIN_SOUND_MS))

        out.append(
            {
                "idx": 0,
                "start_ms": start,
                "end_ms": end,
                "lines": [label_text],
                "type": "sound",
                "meta": {
                    "sound_label": label_text,
                    "sound_kind": ev["kind"],
                    "sound_priority": ev["priority"],
                    "sound_upper_right": ev["kind"] == "music",
                    "raw_labels": ev["labels"],
                },
            }
        )

    return out


def parse_sound_event(text: str, start_ms: int, end_ms: int) -> Dict[str, Any] | None:
    label = text.strip().upper()
    if not label.startswith("[") or not label.endswith("]"):
        return None

    inner = label[1:-1].strip()
    if not inner:
        return None

    parts = [p.strip() for p in re.split(r"\s+AND\s+|,\s*", inner) if p.strip()]
    labels: List[str] = []
    for part in parts:
        normalized = normalize_sound_atom(part)
        if normalized and normalized not in labels:
            labels.append(normalized)

    if not labels:
        return None

    kind, priority = classify_sound_labels(labels)
    safe_end = end_ms if end_ms > start_ms else start_ms + 1

    return {
        "labels": labels,
        "kind": kind,
        "priority": priority,
        "start_ms": start_ms,
        "end_ms": safe_end,
    }


def normalize_sound_atom(label: str) -> str:
    label = re.sub(r"\s+", " ", label.strip().upper())
    replacements = {
        "NOISE": "NOISE",
        "SOUND": "SOUND",
        "SOUNDS": "SOUND",
        "SOUND EFFECT": "SOUND EFFECT",
        "SOUND EFFECTS": "SOUND EFFECT",
        "LAUGH": "LAUGHTER",
        "LAUGHS": "LAUGHTER",
        "LAUGHTER": "LAUGHTER",
        "APPLAUSE": "APPLAUSE",
        "CHEER": "CHEERING",
        "CHEERS": "CHEERING",
        "CHEERING": "CHEERING",
        "MUSIC": "MUSIC",
        "MUSICAL": "MUSIC",
        "INAUDIBLE": "INAUDIBLE",
        "SPEAKER": "",
    }
    return replacements.get(label, label)


def classify_sound_labels(labels: List[str]) -> Tuple[str, int]:
    label_set = set(labels)
    if "MUSIC" in label_set and len(label_set) == 1:
        return "music", 1
    if label_set & {"APPLAUSE", "CHEERING", "LAUGHTER"}:
        return "audience", 3
    if "INAUDIBLE" in label_set:
        return "inaudible", 3
    if label_set <= {"NOISE", "SOUND", "SOUND EFFECT"}:
        return "generic", 0
    if "MUSIC" in label_set:
        return "mixed", 2
    return "effect", 1


def merge_sound_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not events:
        return []

    events = sorted(events, key=lambda e: (e["start_ms"], e["end_ms"]))
    merged: List[Dict[str, Any]] = [events[0].copy()]

    for ev in events[1:]:
        prev = merged[-1]
        same_kind = ev["kind"] == prev["kind"]
        gap = ev["start_ms"] - prev["end_ms"]
        overlap = ev["start_ms"] <= prev["end_ms"]

        same_label_set = set(ev["labels"]) == set(prev["labels"])
        if same_label_set and (gap <= NEARBY_SOUND_MERGE_MS or overlap):
            prev["end_ms"] = max(prev["end_ms"], ev["end_ms"])
            prev["priority"] = max(prev["priority"], ev["priority"])
            continue

        if same_kind and gap <= NEARBY_SOUND_MERGE_MS:
            prev["end_ms"] = max(prev["end_ms"], ev["end_ms"])
            prev["labels"] = merge_label_lists(prev["labels"], ev["labels"])
            prev["priority"] = max(prev["priority"], ev["priority"])
            prev["kind"] = dominant_sound_kind(prev["labels"])
            continue

        if gap <= SIMULTANEOUS_SOUND_MERGE_MS and abs(ev["start_ms"] - prev["start_ms"]) <= 500:
            prev["end_ms"] = max(prev["end_ms"], ev["end_ms"])
            prev["labels"] = merge_label_lists(prev["labels"], ev["labels"])
            prev["priority"] = max(prev["priority"], ev["priority"])
            prev["kind"] = dominant_sound_kind(prev["labels"])
            continue

        merged.append(ev.copy())

    return merged


def merge_label_lists(a: List[str], b: List[str]) -> List[str]:
    out: List[str] = []
    for item in a + b:
        if item and item not in out:
            out.append(item)
    return out


def dominant_sound_kind(labels: List[str]) -> str:
    kind, _ = classify_sound_labels(labels)
    return kind


def filter_sound_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []

    for ev in events:
        duration = ev["end_ms"] - ev["start_ms"]
        label_set = set(ev["labels"])

        # Pure low-information generic noise usually harms more than helps.
        if ev["kind"] == "generic":
            continue

        # Pure background music under 5s should not be captioned.
        if ev["kind"] == "music" and duration < MUSIC_MIN_MS:
            continue

        # Mixed music/noise/audience labels: prefer the meaningful audience cue.
        if "MUSIC" in label_set and label_set & {"APPLAUSE", "CHEERING", "LAUGHTER"}:
            kept = [lbl for lbl in ev["labels"] if lbl in {"APPLAUSE", "CHEERING", "LAUGHTER"}]
            if kept:
                ev = {**ev, "labels": kept, "kind": dominant_sound_kind(kept), "priority": 3}
                duration = ev["end_ms"] - ev["start_ms"]

        # Music mixed only with generic labels still needs to clear 5s.
        if "MUSIC" in label_set and not (label_set & {"APPLAUSE", "CHEERING", "LAUGHTER"}) and duration < MUSIC_MIN_MS:
            kept = [lbl for lbl in ev["labels"] if lbl != "MUSIC" and lbl not in {"NOISE", "SOUND", "SOUND EFFECT"}]
            if not kept:
                continue
            ev = {**ev, "labels": kept, "kind": dominant_sound_kind(kept), "priority": max(1, ev["priority"])}

        filtered.append(ev)

    return collapse_repeated_sound_events(filtered)


def collapse_repeated_sound_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not events:
        return []

    out: List[Dict[str, Any]] = []
    for ev in sorted(events, key=lambda e: (e["start_ms"], e["end_ms"])):
        if out:
            prev = out[-1]
            same_render = render_sound_label(prev) == render_sound_label(ev)
            if same_render and ev["start_ms"] - prev["end_ms"] <= 2500:
                prev["end_ms"] = max(prev["end_ms"], ev["end_ms"])
                prev["labels"] = merge_label_lists(prev["labels"], ev["labels"])
                prev["kind"] = dominant_sound_kind(prev["labels"])
                prev["priority"] = max(prev["priority"], ev["priority"])
                continue
        out.append(ev.copy())
    return out


def render_sound_label(event: Dict[str, Any]) -> str:
    labels = event.get("labels", [])
    if not labels:
        return ""

    if event["kind"] == "music":
        return "♪ MUSIC ♪"

    preferred_order = ["APPLAUSE", "CHEERING", "LAUGHTER", "INAUDIBLE"]
    kept: List[str] = []
    for item in preferred_order:
        if item in labels and item not in kept:
            kept.append(item)

    for item in labels:
        if item in HIGH_VALUE_SOUND_LABELS and item not in kept:
            kept.append(item)

    if not kept:
        for item in labels:
            if item not in LOW_VALUE_SOUND_LABELS and item not in kept:
                kept.append(item)

    if not kept:
        return ""

    return f"[{' AND '.join(kept)}]"


# =========================
# Timeline merging
# =========================
def merge_timeline(dialogue_cues: List[Dict[str, Any]], sound_cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    dialogue = sorted(dialogue_cues, key=lambda c: (c["start_ms"], c["end_ms"]))
    sounds = sorted(sound_cues, key=lambda c: (c["start_ms"], c["end_ms"]))

    placed_sounds: List[Dict[str, Any]] = []
    for sound in sounds:
        placed = place_sound_without_displacing_dialogue(sound, dialogue)
        if placed is not None:
            placed_sounds.append(placed)

    combined = dialogue + placed_sounds
    combined.sort(key=lambda c: (c["start_ms"], c["end_ms"], 0 if c["type"] == "dialogue" else 1))
    return resolve_overlaps(combined)


def place_sound_without_displacing_dialogue(sound: Dict[str, Any], dialogue_cues: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    sound_start = sound["start_ms"]
    sound_end = sound["end_ms"]
    natural_duration = max(MIN_SOUND_MS, sound_end - sound_start)

    prev_dialogue = None
    next_dialogue = None

    for cue in dialogue_cues:
        if cue["end_ms"] <= sound_start:
            prev_dialogue = cue
            continue
        if cue["start_ms"] >= sound_end:
            next_dialogue = cue
            break
        # overlapping dialogue exists; keep scanning to find nearest surrounding cues
        if cue["start_ms"] <= sound_end and cue["end_ms"] >= sound_start:
            if cue["start_ms"] < sound_start:
                prev_dialogue = cue
            else:
                next_dialogue = cue
                break

    gap_start = 0 if prev_dialogue is None else prev_dialogue["end_ms"] + SOUND_PAD_MS
    gap_end = sound_end if next_dialogue is None else next_dialogue["start_ms"] - SOUND_PAD_MS

    candidate_start = max(sound_start, gap_start)
    candidate_end = min(sound_end, gap_end)

    if candidate_end - candidate_start >= SOUND_GAP_MIN_MS:
        placed = {**sound, "start_ms": candidate_start, "end_ms": max(candidate_end, candidate_start + SOUND_GAP_MIN_MS)}
        return clamp_sound_to_available_gap(placed, prev_dialogue, next_dialogue)

    # Try pre-dialogue pocket.
    if prev_dialogue and sound_start < prev_dialogue["end_ms"] and next_dialogue:
        pocket_start = prev_dialogue["end_ms"] + SOUND_PAD_MS
        pocket_end = next_dialogue["start_ms"] - SOUND_PAD_MS
        if pocket_end - pocket_start >= SOUND_GAP_MIN_MS:
            return {**sound, "start_ms": pocket_start, "end_ms": min(pocket_end, pocket_start + natural_duration)}

    # No safe room. Dialogue wins.
    return None


def clamp_sound_to_available_gap(sound: Dict[str, Any], prev_dialogue: Dict[str, Any] | None, next_dialogue: Dict[str, Any] | None) -> Dict[str, Any] | None:
    gap_start = 0 if prev_dialogue is None else prev_dialogue["end_ms"] + SOUND_PAD_MS
    gap_end = sound["end_ms"] if next_dialogue is None else next_dialogue["start_ms"] - SOUND_PAD_MS

    start = max(sound["start_ms"], gap_start)
    end = min(sound["end_ms"], gap_end)
    if end - start < SOUND_GAP_MIN_MS:
        return None

    return {**sound, "start_ms": start, "end_ms": end}


# =========================
# Formatting
# =========================
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
            line = (cue["lines"][0] if cue.get("lines") else "").strip()
            if not line:
                continue
            cue["lines"] = [line[:MAX_CHARS]]
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
        if not text:
            continue

        broken = heuristic_split(
            text=text,
            max_chars=MAX_CHARS,
            max_lines=MAX_LINES,
            protected_phrases=protected_phrases,
        )
        cue["lines"] = [line for line in broken[:MAX_LINES] if line.strip()]

        if violates_line_limits(cue["lines"], MAX_LINES, MAX_CHARS):
            if max_rounds <= 0:
                cue["lines"] = [line[:MAX_CHARS] for line in cue["lines"][:MAX_LINES]]
                formatted.append(cue)
            else:
                children = split_cue_from_text(cue, text)
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
                cue["lines"] = [line for line in retry_broken[:MAX_LINES] if line.strip()]

        formatted.append(cue)

    return resolve_overlaps(formatted)


def split_cue_from_text(cue: Dict[str, Any], text: str) -> List[Dict[str, Any]]:
    text = clean_dialogue_text(text)
    if not text:
        return [cue]

    words = text.split()
    if len(words) < 2:
        return [cue]

    best_index = None
    best_score = None
    for i in range(1, len(words)):
        left = " ".join(words[:i]).strip()
        right = " ".join(words[i:]).strip()
        score = split_score(left, right, PROTECTED_DEFAULTS)
        if best_score is None or score < best_score:
            best_index = i
            best_score = score

    midpoint = best_index if best_index is not None else len(words) // 2
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


def heuristic_split(
    text: str,
    max_chars: int = 32,
    max_lines: int = 2,
    protected_phrases: List[str] | None = None,
) -> List[str]:
    text = clean_dialogue_text(text or "")
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

    fitted = fit_single_line(text, max_chars, protected_phrases)
    remainder = text[len(fitted):].strip()
    if remainder:
        remainder = fit_single_line(remainder, max_chars, protected_phrases)
    return [fitted, remainder] if remainder else [fitted]


def fit_single_line(text: str, max_chars: int, protected_phrases: List[str]) -> str:
    text = clean_dialogue_text(text)
    if len(text) <= max_chars:
        return text

    words = text.split()
    current = []
    for word in words:
        candidate = " ".join(current + [word]).strip()
        if len(candidate) <= max_chars:
            current.append(word)
            continue
        break

    trimmed = " ".join(current).strip()
    if not trimmed:
        return text[:max_chars].rstrip()

    # Avoid ending on a function word when there is a better option.
    while len(trimmed.split()) > 1 and sanitize_last_word(trimmed) in FUNCTION_WORDS:
        trimmed = " ".join(trimmed.split()[:-1]).strip()

    # Avoid cutting through a protected phrase if the last token is a fragment.
    for phrase in protected_phrases:
        if phrase in text and phrase not in trimmed and any(part.startswith(trimmed.split()[-1]) for part in phrase.split() if trimmed.split()):
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

    if re.search(r"\b(?:Mr|Mrs|Ms|Dr|St)\.?$", left):
        score += 20

    combined = left + "\n" + right
    for phrase in protected_phrases:
        if phrase and phrase in combined.replace("\n", " ") and phrase.replace(" ", "\n") in combined:
            score += 100

    right_words = len(right.split())
    if right_words <= 2:
        score += 12

    if len(left.split()) == 1 or len(right.split()) == 1:
        score += 8

    return score


def sanitize_last_word(text: str) -> str:
    text = re.sub(r"[^\w']+$", "", text.strip().lower())
    parts = text.split()
    return parts[-1] if parts else ""


def clean_dialogue_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    text = re.sub(r"\b([A-Za-z])\s+([A-Za-z]{1,2})\b", lambda m: m.group(0), text)
    return text


# =========================
# Overlap resolution
# =========================
def resolve_overlaps(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cues = [c for c in cues if c.get("end_ms", 0) > c.get("start_ms", 0)]
    cues.sort(key=lambda c: (c["start_ms"], c["end_ms"], 0 if c["type"] == "dialogue" else 1))

    out: List[Dict[str, Any]] = []
    for cue in cues:
        if not out:
            out.append(cue)
            continue

        prev = out[-1]
        if prev["end_ms"] <= cue["start_ms"]:
            out.append(cue)
            continue

        # Dialogue always wins.
        if prev["type"] == "dialogue" and cue["type"] == "sound":
            cue["start_ms"] = prev["end_ms"] + SOUND_PAD_MS
            if cue["end_ms"] - cue["start_ms"] < SOUND_GAP_MIN_MS:
                continue
            out.append(cue)
            continue

        if prev["type"] == "sound" and cue["type"] == "dialogue":
            prev["end_ms"] = min(prev["end_ms"], cue["start_ms"] - SOUND_PAD_MS)
            if prev["end_ms"] - prev["start_ms"] < SOUND_GAP_MIN_MS:
                out.pop()
            out.append(cue)
            continue

        if prev["type"] == "sound" and cue["type"] == "sound":
            if render_sound_label(prev) == render_sound_label(cue):
                prev["end_ms"] = max(prev["end_ms"], cue["end_ms"])
            elif cue["start_ms"] < prev["end_ms"]:
                cue["start_ms"] = prev["end_ms"] + SOUND_PAD_MS
                if cue["end_ms"] - cue["start_ms"] >= SOUND_GAP_MIN_MS:
                    out.append(cue)
            else:
                out.append(cue)
            continue

        # Dialogue-dialogue overlap: trim the later cue rather than mangling prior text.
        cue["start_ms"] = prev["end_ms"]
        if cue["end_ms"] <= cue["start_ms"]:
            cue["end_ms"] = cue["start_ms"] + 1
        out.append(cue)

    return [c for c in out if c["end_ms"] > c["start_ms"] and c.get("lines")]
