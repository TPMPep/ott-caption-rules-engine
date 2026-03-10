import re
from typing import Any, Dict, List, Tuple

from services.assembly import normalize_tokens, is_sound_token
from services.exporters import parse_srt, export_srt, export_vtt, export_scc
from services.qc import qc_report, violates_line_limits, count_function_word_endings
try:
    from services.editorial_ai import editorial_refine_cues
except Exception:
    def editorial_refine_cues(cues, protected_phrases):
        return cues

MAX_LINES = 2
MAX_CHARS = 32
MIN_DIALOGUE_MS = 800
MIN_SOUND_MS = 1000
MUSIC_MIN_DURATION_MS = 2500
MIN_GAP_FOR_SOUND_MS = 400
SOUND_MERGE_GAP_MS = 700
STRONG_SOUND_MIN_MS = 650
PREFER_TWO_LINE_MIN_CHARS = 18
PREFER_TWO_LINE_MAX_CHARS = 28

PROTECTED_DEFAULTS: List[str] = []

FUNCTION_WORDS = {
    "a", "an", "the", "of", "to", "and", "or", "but",
    "with", "from", "in", "on", "at", "for", "that",
    "this", "these", "those", "is", "are", "was", "were",
}

WEAK_FRAGMENT_STARTS = {"and", "but", "or", "to", "of", "for", "with", "because", "that", "this", "these", "those"}
WEAK_FRAGMENT_ENDS = FUNCTION_WORDS | {"i", "it", "who", "what", "this", "that"}
BRACKET_TAG_RE = re.compile(r"\[[^\]]+\]")


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

    protected_phrases = build_runtime_protected_phrases(backbone, protected_phrases)
    print(f"[FORMATTER] Protected phrases detected: {len(protected_phrases)}")

    print("[FORMATTER] Normalizing timestamps")
    tokens = normalize_tokens(timestamps)
    tokens.sort(key=lambda w: (w["start_ms"], w["end_ms"]))
    print(f"[FORMATTER] Tokens: {len(tokens)}")

    print("[FORMATTER] Building dialogue cues from SRT text")
    dialogue_cues = build_dialogue_cues_from_srt(backbone, tokens)
    print(f"[FORMATTER] Dialogue cues: {len(dialogue_cues)}")

    print("[FORMATTER] Building sound events from tokens")
    raw_sound_events = build_sound_events(tokens)
    print(f"[FORMATTER] Raw sound events: {len(raw_sound_events)}")

    print("[FORMATTER] Inserting sound cues using bridge placement")
    sound_cues = place_sound_cues_in_gaps(raw_sound_events, dialogue_cues)
    print(f"[FORMATTER] Sound cues placed: {len(sound_cues)}")

    print("[FORMATTER] Combining dialogue + sound timeline")
    cues = dialogue_cues + sound_cues
    cues.sort(key=lambda c: (c["start_ms"], c["end_ms"], 0 if c["type"] == "dialogue" else 1))
    print(f"[FORMATTER] Combined cues before speaker split: {len(cues)}")

    print("[FORMATTER] Pre-splitting multi-speaker dialogue cues")
    cues = presplit_multispeaker(cues)
    print(f"[FORMATTER] Cues after pre-split: {len(cues)}")

    print("[FORMATTER] Formatting dialogue lines")
    cues = format_cues(cues, protected_phrases)
    print(f"[FORMATTER] Cues after formatting: {len(cues)}")

    print("[FORMATTER] Applying conservative readability pass")
    cues = conservative_readability_pass(cues)

    print("[FORMATTER] Running post-format repair")
    cues = post_format_repair(cues, protected_phrases)
    print(f"[FORMATTER] Cues after repair: {len(cues)}")

    print("[FORMATTER] Running AI editorial pass")
    cues = editorial_refine_cues(cues, protected_phrases)
    print(f"[FORMATTER] Cues after AI editorial: {len(cues)}")

    print("[FORMATTER] Resolving overlaps")
    cues = resolve_overlaps_dialogue_first(cues)
    print("[FORMATTER] Overlaps resolved")

    print("[FORMATTER] Running final hard validation")
    cues = final_hard_validate(cues, protected_phrases)

    cues = remove_empty_or_tiny_cues(cues)
    cues.sort(key=lambda c: (c["start_ms"], c["end_ms"], 0 if c["type"] == "dialogue" else 1))

    for i, cue in enumerate(cues, start=1):
        cue["idx"] = i

    print("[FORMATTER] Exporting outputs")
    srt_out = export_srt(cues)
    vtt_out = export_vtt(cues) if "vtt" in output_formats else None
    scc_out = export_scc(cues) if "scc" in output_formats else None

    print("[FORMATTER] Running QC")
    qc = qc_report(cues_in, cues, protected_phrases)
    print("[FORMATTER] QC complete")

    return {
        "srt": srt_out,
        "vtt": vtt_out,
        "scc": scc_out,
        "qc": qc,
    }


def detect_protected_phrases_from_backbone(backbone: List[Dict[str, Any]]) -> List[str]:
    phrases: List[str] = []
    seen = set()

    for cue in backbone:
        raw_text = " ".join(cue.get("lines", [])).strip()
        text = strip_sound_tags(raw_text)
        matches = re.findall(
            r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}|Med)){1,5}\b",
            text,
        )
        for phrase in matches:
            phrase = phrase.strip()
            words = phrase.split()
            if len(words) < 2 or len(words) > 6:
                continue
            lowered = [w.lower() for w in words]
            if sum(1 for w in lowered if w in FUNCTION_WORDS) >= len(words) - 1:
                continue
            if phrase not in seen:
                seen.add(phrase)
                phrases.append(phrase)

    if "Below Deck Med" not in seen:
        for cue in backbone:
            raw_text = " ".join(cue.get("lines", [])).strip()
            if "Below Deck Med" in raw_text:
                phrases.append("Below Deck Med")
                seen.add("Below Deck Med")
                break

    return phrases


def build_runtime_protected_phrases(backbone: List[Dict[str, Any]], protected_phrases: List[str] | None = None) -> List[str]:
    explicit = protected_phrases or []
    detected = detect_protected_phrases_from_backbone(backbone)
    return list(dict.fromkeys(PROTECTED_DEFAULTS + explicit + detected))


def build_dialogue_cues_from_srt(backbone: List[Dict[str, Any]], tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cues: List[Dict[str, Any]] = []

    for cue in backbone:
        raw_text = " ".join(cue.get("lines", [])).strip()
        dialogue_text = strip_sound_tags(raw_text)
        dialogue_text = clean_dialogue_text(dialogue_text)

        runs = build_speaker_runs_for_cue(cue, tokens)
        if not runs:
            runs = [{"speaker": "A", "text": dialogue_text}]
        elif len(runs) == 1:
            runs = [{"speaker": runs[0]["speaker"], "text": dialogue_text}]
        else:
            runs = align_runs_to_text(runs, dialogue_text)

        cues.append(
            {
                "idx": cue.get("idx", 0),
                "start_ms": cue["start_ms"],
                "end_ms": cue["end_ms"],
                "lines": [],
                "type": "dialogue",
                "meta": {
                    "runs": runs,
                    "two_speaker": False,
                    "raw_text": raw_text,
                    "dialogue_text": dialogue_text,
                },
            }
        )

    return cues


def strip_sound_tags(text: str) -> str:
    if not text:
        return ""

    def _replace(match: re.Match) -> str:
        tag = match.group(0).upper()
        if tag == "[INAUDIBLE]":
            return " [INAUDIBLE] "
        return " "

    text = BRACKET_TAG_RE.sub(_replace, text)
    return re.sub(r"\s+", " ", text).strip()


def build_speaker_runs_for_cue(cue: Dict[str, Any], tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    current_run = None

    for token in tokens:
        if token["end_ms"] < cue["start_ms"]:
            continue
        if token["start_ms"] > cue["end_ms"]:
            break

        text = (token.get("text") or "").strip()
        if not text or is_sound_token(text):
            continue

        speaker = token.get("speaker") or "A"
        if current_run is None or current_run["speaker"] != speaker:
            current_run = {"speaker": speaker, "text_parts": []}
            runs.append(current_run)
        current_run["text_parts"].append(text)

    collapsed: List[Dict[str, Any]] = []
    for run in runs:
        run_text = clean_dialogue_text(" ".join(run["text_parts"]))
        if not run_text:
            continue
        words = run_text.split()
        if collapsed and len(words) == 1 and len(run_text) <= 6 and sanitize_last_word(run_text) in {"uh", "um", "yeah", "yes", "no"}:
            collapsed[-1]["text"] = clean_dialogue_text(collapsed[-1]["text"] + " " + run_text)
            continue
        collapsed.append({"speaker": run["speaker"], "text": run_text})

    return collapsed[:6]


def align_runs_to_text(runs: List[Dict[str, Any]], full_text: str) -> List[Dict[str, Any]]:
    full_text = clean_dialogue_text(full_text)
    if not full_text:
        return runs

    cleaned_runs = []
    for run in runs:
        run_text = clean_dialogue_text(run.get("text", ""))
        if run_text:
            cleaned_runs.append({"speaker": run.get("speaker") or "A", "text": run_text})
    runs = cleaned_runs

    if len(runs) <= 1:
        return [{"speaker": runs[0]["speaker"] if runs else "A", "text": full_text}]

    full_words = full_text.split()
    if not full_words:
        return [{"speaker": runs[0]["speaker"], "text": full_text}]

    assigned = []
    cursor = 0
    for i, run in enumerate(runs):
        target = run["text"].split()
        if not target:
            continue
        if i == len(runs) - 1:
            piece_words = full_words[cursor:]
        else:
            n = len(target)
            piece_words = None
            best_end = None
            lower_target = [w.lower() for w in target]
            for end in range(cursor + max(1, n - 2), min(len(full_words), cursor + n + 4) + 1):
                candidate = full_words[cursor:end]
                lower_candidate = [w.lower() for w in candidate]
                if lower_candidate == lower_target[:len(candidate)]:
                    piece_words = candidate
                    best_end = end
            if piece_words is None:
                return [{"speaker": runs[0]["speaker"], "text": full_text}]
            cursor = best_end
        assigned.append({"speaker": run["speaker"], "text": clean_dialogue_text(" ".join(piece_words))})

    assigned = [r for r in assigned if r["text"]]
    if len(assigned) < 2:
        return [{"speaker": assigned[0]["speaker"] if assigned else "A", "text": full_text}]

    reconstructed = clean_dialogue_text(" ".join(r["text"] for r in assigned))
    if [w.lower() for w in reconstructed.split()] != [w.lower() for w in full_words]:
        return [{"speaker": assigned[0]["speaker"], "text": full_text}]

    return assigned


def build_sound_events(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    raw_events: List[Dict[str, Any]] = []
    for token in tokens:
        text = (token.get("text") or "").strip()
        if not is_sound_token(text):
            continue
        label = normalize_sound_label(text)
        if not label:
            continue
        start = int(token["start_ms"])
        end = int(token["end_ms"] if token["end_ms"] > token["start_ms"] else token["start_ms"] + 1)
        raw_events.append({"label": label, "start_ms": start, "end_ms": end})
    return filter_sound_events(merge_sound_events(raw_events))


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

    junk = {"NOISE", "SOUND", "SOUND EFFECT", "SPEAKER", "VOICE", "TALKING", "INAUDIBLE"}
    deduped = [p for p in deduped if p not in junk]
    if not deduped:
        return ""

    if "NOISE" in deduped and any(x in deduped for x in ["APPLAUSE", "LAUGHTER", "CHEERING", "MUSIC"]):
        deduped = [p for p in deduped if p != "NOISE"]

    cleaned = " AND ".join(deduped).strip()
    if not cleaned:
        return ""
    if cleaned == "MUSIC":
        return "[♪]"
    if cleaned in ("APPLAUSE AND MUSIC", "MUSIC AND APPLAUSE"):
        return "[APPLAUSE AND ♪]"
    if cleaned in ("LAUGHTER AND MUSIC", "MUSIC AND LAUGHTER"):
        return "[LAUGHTER AND ♪]"
    if cleaned in ("CHEERING AND MUSIC", "MUSIC AND CHEERING"):
        return "[CHEERING AND ♪]"
    return f"[{cleaned}]"


def merge_sound_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not events:
        return []
    events = sorted(events, key=lambda e: (e["start_ms"], e["end_ms"]))
    merged: List[Dict[str, Any]] = [events[0].copy()]

    for ev in events[1:]:
        prev = merged[-1]
        same_label = ev["label"] == prev["label"]
        close_gap = ev["start_ms"] - prev["end_ms"] <= SOUND_MERGE_GAP_MS
        overlap = ev["start_ms"] <= prev["end_ms"]

        if same_label and (close_gap or overlap):
            prev["end_ms"] = max(prev["end_ms"], ev["end_ms"])
            continue

        if ev["start_ms"] - prev["end_ms"] <= 220:
            combined = combine_sound_labels(prev["label"], ev["label"])
            if combined != prev["label"]:
                prev["label"] = combined
                prev["end_ms"] = max(prev["end_ms"], ev["end_ms"])
                continue

        merged.append(ev.copy())
    return merged


def filter_sound_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    last_music_end = -999999
    last_label = None
    last_label_end = -999999

    for ev in events:
        label = ev["label"]
        duration = ev["end_ms"] - ev["start_ms"]
        if label == "[♪]" and duration < MUSIC_MIN_DURATION_MS:
            continue
        if label == "[♪]" and ev["start_ms"] - last_music_end < 3500:
            continue
        if label == last_label and ev["start_ms"] - last_label_end < 1200:
            continue
        if label == "[♪]":
            last_music_end = ev["end_ms"]
        last_label = label
        last_label_end = ev["end_ms"]
        filtered.append(ev)
    return filtered


def place_sound_cues_in_gaps(sound_events: List[Dict[str, Any]], dialogue_cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not sound_events:
        return []

    dialogue_sorted = sorted(dialogue_cues, key=lambda c: (c["start_ms"], c["end_ms"]))
    sound_cues: List[Dict[str, Any]] = []
    occupied: List[Tuple[int, int]] = []

    for ev in sound_events:
        placed = fit_sound_into_dialogue_gap(ev, dialogue_sorted, occupied)
        if placed:
            sound_cues.append(placed)
            occupied.append((placed["start_ms"], placed["end_ms"]))

    return sound_cues


def fit_sound_into_dialogue_gap(ev: Dict[str, Any], dialogue_sorted: List[Dict[str, Any]], occupied: List[Tuple[int, int]]) -> Dict[str, Any] | None:
    label = ev.get("label", "")
    min_required = STRONG_SOUND_MIN_MS if any(x in label for x in ("APPLAUSE", "LAUGHTER", "CHEERING", "♪")) else MIN_SOUND_MS
    target_ms = 2200 if any(x in label for x in ("APPLAUSE", "LAUGHTER", "CHEERING")) else 2800
    max_ms = 4200 if label == "[♪]" else 3200

    candidates: List[Tuple[int, int, int]] = []
    n = len(dialogue_sorted)

    for i in range(n + 1):
        prev = dialogue_sorted[i - 1] if i > 0 else None
        nxt = dialogue_sorted[i] if i < n else None
        gap_start = prev["end_ms"] if prev else max(0, ev["start_ms"] - 1200)
        gap_end = nxt["start_ms"] if nxt else ev["end_ms"] + 2200
        if gap_end <= gap_start:
            continue

        center = (ev["start_ms"] + ev["end_ms"]) // 2
        distance = 0
        if center < gap_start:
            distance = gap_start - center
        elif center > gap_end:
            distance = center - gap_end

        start = max(gap_start, min(ev["start_ms"], gap_end - min_required))
        end = min(gap_end, max(start + min_required, min(start + target_ms, gap_end)))

        if end - start < min_required:
            if gap_end - gap_start >= min_required:
                start = gap_start
                end = min(gap_end, start + max(min_required, min(target_ms, gap_end - gap_start)))
            else:
                continue

        if overlaps_any(start, end, occupied):
            continue

        candidates.append((distance, start, min(end, start + max_ms)))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], abs((x[1] + x[2]) // 2 - (ev["start_ms"] + ev["end_ms"]) // 2)))
    _, start, end = candidates[0]
    if end - start < min_required:
        return None

    return {
        "idx": 0,
        "start_ms": start,
        "end_ms": end,
        "lines": [label[:MAX_CHARS]],
        "type": "sound",
        "meta": {"sound_label": label, "sound_priority": sound_priority(label)},
    }


def overlaps_any(start: int, end: int, occupied: List[Tuple[int, int]]) -> bool:
    for a, b in occupied:
        if start < b and a < end:
            return True
    return False


def sound_priority(label: str) -> int:
    if label in ("[LAUGHTER]", "[APPLAUSE]", "[CHEERING]", "[APPLAUSE AND ♪]"):
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


def presplit_multispeaker(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for cue in cues:
        if cue["type"] == "sound":
            out.append(cue)
            continue

        runs = cue["meta"].get("runs", [])
        if len(runs) == 2:
            left = clean_dialogue_text(runs[0].get("text", ""))
            right = clean_dialogue_text(runs[1].get("text", ""))
            if left and right and len(left) <= MAX_CHARS - 2 and len(right) <= MAX_CHARS - 2:
                cue["meta"]["two_speaker"] = True
                out.append(cue)
                continue

        if len(runs) <= 1:
            out.append(cue)
            continue

        total_duration = max(1, cue["end_ms"] - cue["start_ms"])
        lengths = [max(1, len(r["text"])) for r in runs]
        total_length = max(1, sum(lengths))
        current_start = cue["start_ms"]

        for j, (run, length) in enumerate(zip(runs, lengths)):
            dur = max(350, int(total_duration * (length / total_length)))
            new_end = cue["end_ms"] if j == len(runs) - 1 else min(cue["end_ms"], current_start + dur)
            out.append(
                {
                    "idx": 0,
                    "start_ms": current_start,
                    "end_ms": new_end,
                    "lines": [],
                    "type": "dialogue",
                    "meta": {"runs": [run], "two_speaker": False, "dialogue_text": run["text"]},
                }
            )
            current_start = new_end

    return out


def format_cues(cues: List[Dict[str, Any]], protected_phrases: List[str], max_rounds: int = 3) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []

    for cue in cues:
        if cue["type"] == "sound":
            cue["lines"] = [cue["lines"][0][:MAX_CHARS]]
            formatted.append(cue)
            continue

        runs = cue["meta"].get("runs", [])
        is_two_speaker = bool(cue["meta"].get("two_speaker")) and len(runs) == 2

        if is_two_speaker:
            left_text = clean_dialogue_text(runs[0]["text"])
            right_text = clean_dialogue_text(runs[1]["text"])
            if len(left_text) <= MAX_CHARS - 2 and len(right_text) <= MAX_CHARS - 2:
                cue["lines"] = [f"- {left_text}", f"- {right_text}"]
                cue["meta"]["dialogue_text"] = f"{left_text} {right_text}".strip()
                formatted.append(cue)
                continue
            formatted.extend(format_cues(split_two_speaker_cue(cue), protected_phrases, max_rounds=max_rounds))
            continue

        text = clean_dialogue_text(cue["meta"].get("dialogue_text", ""))
        lines, needs_event_split = smart_split(text, protected_phrases, prefer_balance=True)

        if needs_event_split and max_rounds > 0:
            formatted.extend(format_cues(split_cue(cue), protected_phrases, max_rounds=max_rounds - 1))
            continue

        cue["lines"] = tidy_lines_for_readability(lines[:MAX_LINES], protected_phrases)
        if violates_line_limits(cue["lines"], MAX_LINES, MAX_CHARS):
            if max_rounds > 0:
                formatted.extend(format_cues(split_cue(cue), protected_phrases, max_rounds=max_rounds - 1))
            else:
                cue["lines"] = [fit_single_line(line, MAX_CHARS, protected_phrases) for line in cue["lines"][:MAX_LINES]]
                formatted.append(cue)
            continue

        if count_function_word_endings(cue["lines"]) > 0 and max_rounds > 0:
            improved, needs_split = smart_split(text, protected_phrases, prefer_balance=False)
            if needs_split:
                formatted.extend(format_cues(split_cue(cue), protected_phrases, max_rounds=max_rounds - 1))
                continue
            cue["lines"] = tidy_lines_for_readability(improved[:MAX_LINES], protected_phrases)

        formatted.append(cue)

    return formatted


def split_two_speaker_cue(cue: Dict[str, Any]) -> List[Dict[str, Any]]:
    runs = cue["meta"].get("runs", [])
    if len(runs) != 2:
        return [cue]

    total_duration = max(1, cue["end_ms"] - cue["start_ms"])
    left_len = max(1, len(runs[0]["text"]))
    right_len = max(1, len(runs[1]["text"]))
    split_time = cue["start_ms"] + int(total_duration * (left_len / (left_len + right_len)))

    return [
        {
            **cue,
            "start_ms": cue["start_ms"],
            "end_ms": split_time,
            "lines": [],
            "type": "dialogue",
            "meta": {"runs": [runs[0]], "two_speaker": False, "dialogue_text": runs[0]["text"]},
        },
        {
            **cue,
            "start_ms": split_time,
            "end_ms": cue["end_ms"],
            "lines": [],
            "type": "dialogue",
            "meta": {"runs": [runs[1]], "two_speaker": False, "dialogue_text": runs[1]["text"]},
        },
    ]


def split_cue(cue: Dict[str, Any]) -> List[Dict[str, Any]]:
    text = cue["meta"].get("dialogue_text", "").strip()
    if not text:
        return [cue]

    words = text.split()
    if len(words) < 2:
        return [cue]

    midpoint = choose_split_midpoint(words)
    left_text = " ".join(words[:midpoint]).strip()
    right_text = " ".join(words[midpoint:]).strip()

    total_duration = max(1, cue["end_ms"] - cue["start_ms"])
    denominator = max(1, len(left_text) + len(right_text))
    split_time = cue["start_ms"] + int(total_duration * (len(left_text) / denominator))
    first_speaker = cue["meta"].get("runs", [{"speaker": "A"}])[0]["speaker"]

    return [
        {
            **cue,
            "start_ms": cue["start_ms"],
            "end_ms": split_time,
            "lines": [],
            "meta": {**cue["meta"], "dialogue_text": left_text, "runs": [{"speaker": first_speaker, "text": left_text}], "two_speaker": False},
        },
        {
            **cue,
            "start_ms": split_time,
            "end_ms": cue["end_ms"],
            "lines": [],
            "meta": {**cue["meta"], "dialogue_text": right_text, "runs": [{"speaker": first_speaker, "text": right_text}], "two_speaker": False},
        },
    ]


def choose_split_midpoint(words: List[str]) -> int:
    midpoint = len(words) // 2
    best = midpoint
    best_score = 999999

    for i in range(max(1, midpoint - 8), min(len(words), midpoint + 9)):
        left = words[:i]
        right = words[i:]
        if not left or not right:
            continue
        left_text = " ".join(left)
        right_text = " ".join(right)
        score = abs(len(left_text) - len(right_text))
        left_last = re.sub(r"[^\w',;:.?!-]+$", "", left[-1]).lower()
        right_first = re.sub(r"^[^\w']+", "", right[0]).lower()
        if left_last in FUNCTION_WORDS:
            score += 35
        if right_first in FUNCTION_WORDS:
            score += 14
        if re.search(r"[,;:]$", left[-1]):
            score -= 30
        elif re.search(r"[\.!?]$", left[-1]):
            score -= 36
        if len(right) <= 2:
            score += 25
        if len(left) <= 2:
            score += 20
        if len(left) == 1 and left[0].lower() in {"and", "but", "so", "because"}:
            score += 50
        if score < best_score:
            best_score = score
            best = i
    return best


def smart_split(text: str, protected_phrases: List[str], prefer_balance: bool = True) -> Tuple[List[str], bool]:
    text = clean_dialogue_text(text)
    if not text:
        return [""], False

    words = text.split()
    candidates: List[Tuple[int, List[str]]] = []
    for i in range(1, len(words)):
        left = " ".join(words[:i]).strip()
        right = " ".join(words[i:]).strip()
        if len(left) > MAX_CHARS or len(right) > MAX_CHARS:
            continue
        score = split_score(left, right, protected_phrases, prefer_balance=prefer_balance)
        candidates.append((score, [left, right]))

    if len(text) <= MAX_CHARS:
        if PREFER_TWO_LINE_MIN_CHARS <= len(text) <= PREFER_TWO_LINE_MAX_CHARS and candidates:
            candidates.sort(key=lambda x: x[0])
            best_score, best_lines = candidates[0]
            if best_score < 48:
                return best_lines, False
        return [text], False

    if candidates:
        candidates.sort(key=lambda x: x[0])
        best_score, best_lines = candidates[0]
        if best_score < 120:
            return best_lines, False
    return [text], True


def fit_single_line(text: str, max_chars: int, protected_phrases: List[str]) -> str:
    text = clean_dialogue_text(text)
    if len(text) <= max_chars:
        return text

    words = text.split()
    out: List[str] = []
    for word in words:
        trial = " ".join(out + [word]).strip()
        if len(trial) <= max_chars:
            out.append(word)
        else:
            break
    if not out:
        return text[:max_chars].rstrip()

    candidate = " ".join(out).strip()
    while len(out) > 1 and out[-1].lower() in FUNCTION_WORDS:
        out.pop()
        candidate = " ".join(out).strip()

    for phrase in protected_phrases:
        pw = phrase.split()
        cw = candidate.split()
        if len(cw) < len(pw) and cw == pw[:len(cw)]:
            if len(out) > 1:
                out.pop()
                candidate = " ".join(out).strip()
            break
    return candidate or " ".join(words[:1])[:max_chars].rstrip()


def split_score(left: str, right: str, protected_phrases: List[str], prefer_balance: bool = True) -> int:
    score = 0
    if prefer_balance:
        score += abs(len(left) - len(right))
    if re.search(r"[,;:]$", left):
        score -= 24
    elif re.search(r"[\.!?]$", left):
        score -= 34
    else:
        score += 24

    left_last = sanitize_last_word(left)
    right_first = re.sub(r"^[^\w']+", "", right.strip().split()[0]).lower() if right.strip().split() else ""
    if left_last in FUNCTION_WORDS:
        score += 40
    if right_first in FUNCTION_WORDS:
        score += 18

    combined = left + "\n" + right
    for phrase in protected_phrases:
        if phrase and phrase in combined.replace("\n", " ") and phrase.replace(" ", "\n") in combined:
            score += 250

    left_words = left.split()
    right_words = right.split()
    if len(right_words) <= 2:
        score += 28
    if len(right_words) == 1:
        score += 60
    if len(left_words) <= 2:
        score += 28
    if left_words and left_words[-1].lower() in WEAK_FRAGMENT_ENDS:
        score += 45
    if right_words and right_words[0].lower() in WEAK_FRAGMENT_STARTS:
        score += 26
    awkward_heads = {"and i", "it's", "who made", "what was", "because i", "just go", "little bit"}
    if " ".join(left_words[-2:]).lower() in awkward_heads:
        score += 70
    if " ".join(right_words[:2]).lower() in awkward_heads:
        score += 60
    return score


def conservative_readability_pass(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cues = sorted(cues, key=lambda c: (c["start_ms"], c["end_ms"], 0 if c["type"] == "dialogue" else 1))
    out: List[Dict[str, Any]] = []
    for i, cue in enumerate(cues):
        cue = cue.copy()
        if cue["type"] == "sound":
            target_end = cue["start_ms"] + 2800
            min_end = cue["start_ms"] + STRONG_SOUND_MIN_MS
            next_start = cues[i + 1]["start_ms"] if i < len(cues) - 1 else target_end
            cue["end_ms"] = min(max(min_end, cue["end_ms"], min(target_end, cue["start_ms"] + 4000)), max(min_end, next_start - 200))
        out.append(cue)
    return out


def post_format_repair(cues: List[Dict[str, Any]], protected_phrases: List[str]) -> List[Dict[str, Any]]:
    cues = sorted(cues, key=lambda c: (c["start_ms"], c["end_ms"], 0 if c["type"] == "dialogue" else 1))
    repaired: List[Dict[str, Any]] = []

    for cue in cues:
        if cue["type"] == "sound":
            label = cue["lines"][0].strip() if cue.get("lines") else ""
            label = normalize_sound_label(label)
            if not label:
                continue
            cue["lines"] = [label[:MAX_CHARS]]
            repaired.append(cue)
            continue

        runs = cue["meta"].get("runs", [])
        is_two_speaker = bool(cue["meta"].get("two_speaker")) and len(runs) == 2
        if is_two_speaker:
            left = clean_dialogue_text(runs[0]["text"])
            right = clean_dialogue_text(runs[1]["text"])
            if left and right and len(left) <= MAX_CHARS - 2 and len(right) <= MAX_CHARS - 2:
                cue["lines"] = [f"- {left}", f"- {right}"]
                cue["meta"]["dialogue_text"] = f"{left} {right}".strip()
                repaired.append(cue)
                continue
            rendered: List[Dict[str, Any]] = []
            for child in split_two_speaker_cue(cue):
                rendered.extend(format_cues([child], protected_phrases, max_rounds=1))
            repaired.extend(rendered)
            continue

        text = clean_dialogue_text(cue["meta"].get("dialogue_text", " ".join(cue.get("lines", []))).strip())
        cue["meta"]["dialogue_text"] = text
        lines, needs_event_split = smart_split(text, protected_phrases, prefer_balance=False)
        if needs_event_split:
            split_children = split_cue(cue)
            if split_children != [cue]:
                rendered = []
                for child in split_children:
                    rendered.extend(format_cues([child], protected_phrases, max_rounds=1))
                repaired.extend(rendered)
                continue
        cue["lines"] = tidy_lines_for_readability(lines[:MAX_LINES], protected_phrases)
        repaired.append(cue)

    merged: List[Dict[str, Any]] = []
    i = 0
    while i < len(repaired):
        cue = repaired[i]
        if (
            cue["type"] == "dialogue"
            and i < len(repaired) - 1
            and repaired[i + 1]["type"] == "dialogue"
            and not cue["meta"].get("two_speaker")
            and not repaired[i + 1]["meta"].get("two_speaker")
        ):
            curr_text = clean_dialogue_text(cue["meta"].get("dialogue_text", " ".join(cue.get("lines", []))))
            next_text = clean_dialogue_text(repaired[i + 1]["meta"].get("dialogue_text", " ".join(repaired[i + 1].get("lines", []))))
            curr_words = curr_text.split()
            fragment_like = len(curr_words) <= 3 or curr_text.lower() in {"and i", "it's", "who made", "what was your", "because i always said to", "just go home", "little bit about his"} or sanitize_last_word(curr_text) in WEAK_FRAGMENT_ENDS
            if fragment_like:
                combined = clean_dialogue_text((curr_text + " " + next_text).strip())
                lines, needs_split = smart_split(combined, protected_phrases, prefer_balance=False)
                if not needs_split:
                    new_cue = cue.copy()
                    new_cue["end_ms"] = repaired[i + 1]["end_ms"]
                    new_cue["meta"] = dict(cue["meta"])
                    new_cue["meta"]["dialogue_text"] = combined
                    new_cue["meta"]["runs"] = [{"speaker": cue["meta"].get("runs", [{"speaker": "A"}])[0]["speaker"], "text": combined}]
                    new_cue["meta"]["two_speaker"] = False
                    new_cue["lines"] = tidy_lines_for_readability(lines[:MAX_LINES], protected_phrases)
                    merged.append(new_cue)
                    i += 2
                    continue
        merged.append(cue)
        i += 1

    for i in range(1, len(merged)):
        prev = merged[i - 1]
        curr = merged[i]
        if prev["type"] == "dialogue" and curr["type"] == "dialogue":
            prev_text = clean_dialogue_text(prev["meta"].get("dialogue_text", " ".join(prev.get("lines", []))))
            curr_text = clean_dialogue_text(curr["meta"].get("dialogue_text", " ".join(curr.get("lines", []))))
            pw = prev_text.split()
            cw = curr_text.split()
            if pw and cw and pw[-1].lower() == cw[0].lower():
                trimmed = clean_dialogue_text(" ".join(cw[1:]))
                if trimmed:
                    lines, needs_split = smart_split(trimmed, protected_phrases, prefer_balance=False)
                    if not needs_split:
                        curr["meta"]["dialogue_text"] = trimmed
                        curr["meta"]["runs"] = [{"speaker": curr["meta"].get("runs", [{"speaker": "A"}])[0]["speaker"], "text": trimmed}]
                        curr["lines"] = tidy_lines_for_readability(lines[:MAX_LINES], protected_phrases)
    return merged


def resolve_overlaps_dialogue_first(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cues.sort(key=lambda c: (c["start_ms"], c["end_ms"], 0 if c["type"] == "dialogue" else 1))
    resolved: List[Dict[str, Any]] = []

    for cue in cues:
        if not resolved:
            resolved.append(cue)
            continue
        prev = resolved[-1]
        if prev["end_ms"] > cue["start_ms"]:
            if cue["type"] == "sound":
                cue["start_ms"] = prev["end_ms"]
                if cue["end_ms"] - cue["start_ms"] < STRONG_SOUND_MIN_MS:
                    continue
            else:
                cue["start_ms"] = prev["end_ms"]
                if cue["end_ms"] <= cue["start_ms"]:
                    cue["end_ms"] = cue["start_ms"] + 1
        resolved.append(cue)
    return resolved


def clean_dialogue_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    text = re.sub(r"\b(yes|no|yeah|yep)\.\s+(yes|no|yeah|yep)\b", lambda m: f"{m.group(1)}, {m.group(2)}", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(tonight|today|now|here|there|okay|ok|well|look|show|dress|thing|way|mean|guess)\.\s+(right\?)", r"\1, \2", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<=[a-z]),(?=[A-Z])", ", ", text)
    text = re.sub(r"(?<=[a-z])\.(?= [a-z])", ",", text)
    return text.strip()


def tidy_lines_for_readability(lines: List[str], protected_phrases: List[str]) -> List[str]:
    if not lines:
        return lines
    if len(lines) == 2:
        left, right = clean_dialogue_text(lines[0]), clean_dialogue_text(lines[1])
        weak_single = {"is", "to", "of", "and", "or", "but", "for", "with"}
        if len(right.split()) == 1 and right.lower() in weak_single:
            joined = clean_dialogue_text(f"{left} {right}")
            better, needs_split = smart_split(joined, protected_phrases, prefer_balance=False)
            if not needs_split:
                return better[:2]
        if sanitize_last_word(left) in WEAK_FRAGMENT_ENDS or (right.split() and right.split()[0].lower() in WEAK_FRAGMENT_STARTS):
            joined = clean_dialogue_text(f"{left} {right}")
            better, needs_split = smart_split(joined, protected_phrases, prefer_balance=False)
            if not needs_split:
                return better[:2]
        return [left, right]
    return [clean_dialogue_text(x) for x in lines]


def sanitize_last_word(text: str) -> str:
    text = re.sub(r"[^\w']+$", "", text.strip().lower())
    parts = text.split()
    return parts[-1] if parts else ""


def final_hard_validate(cues: List[Dict[str, Any]], protected_phrases: List[str]) -> List[Dict[str, Any]]:
    validated: List[Dict[str, Any]] = []
    bad_fragment_patterns = {
        "and i", "it's", "who made", "what was your", "because i always said to",
        "just go home", "little bit about his", "the shower drain. of"
    }

    for cue in cues:
        if cue.get("type") == "sound":
            lines = cue.get("lines", [])[:1]
            if not lines:
                continue
            cue["lines"] = [clean_dialogue_text(lines[0])[:MAX_CHARS]]
            validated.append(cue)
            continue

        text = clean_dialogue_text(cue.get("meta", {}).get("dialogue_text", " ".join(cue.get("lines", []))).strip())
        runs = cue.get("meta", {}).get("runs", [])
        is_two = bool(cue.get("meta", {}).get("two_speaker")) and len(runs) == 2
        lines = [clean_dialogue_text(x) for x in cue.get("lines", []) if clean_dialogue_text(x)]
        if not lines and text:
            lines = [text]

        if is_two:
            good = len(lines) == 2 and all(line.startswith("- ") and len(line) <= MAX_CHARS for line in lines)
            if good:
                cue["lines"] = lines
                validated.append(cue)
                continue
            left = clean_dialogue_text(runs[0].get("text", ""))
            right = clean_dialogue_text(runs[1].get("text", ""))
            if left and right and len(left) <= MAX_CHARS - 2 and len(right) <= MAX_CHARS - 2:
                cue["lines"] = [f"- {left}", f"- {right}"]
                cue["meta"]["dialogue_text"] = f"{left} {right}".strip()
                validated.append(cue)
                continue
            validated.extend(format_cues(split_two_speaker_cue(cue), protected_phrases, max_rounds=1))
            continue

        def looks_bad(ls: List[str]) -> bool:
            if not ls or len(ls) > MAX_LINES or any(len(x) > MAX_CHARS for x in ls):
                return True
            joined = clean_dialogue_text(" ".join(ls)).lower()
            if joined in bad_fragment_patterns:
                return True
            if len(ls) == 1 and len(ls[0]) >= PREFER_TWO_LINE_MIN_CHARS:
                better, needs_split = smart_split(ls[0], protected_phrases, prefer_balance=False)
                if not needs_split and len(better) == 2:
                    return True
            if len(ls) == 2:
                if sanitize_last_word(ls[0]) in WEAK_FRAGMENT_ENDS:
                    return True
                if ls[1].split() and ls[1].split()[0].lower() in WEAK_FRAGMENT_STARTS:
                    return True
            return False

        if lines and not looks_bad(lines):
            cue["lines"] = lines
            validated.append(cue)
            continue

        broken, needs_split = smart_split(text, protected_phrases, prefer_balance=False)
        if not needs_split and not looks_bad(broken[:MAX_LINES]):
            cue["lines"] = tidy_lines_for_readability(broken[:MAX_LINES], protected_phrases)
            validated.append(cue)
            continue

        children = split_cue(cue)
        if children != [cue]:
            validated.extend(format_cues(children, protected_phrases, max_rounds=1))
            continue

        first = fit_single_line(text, MAX_CHARS, protected_phrases)
        remainder = clean_dialogue_text(text[len(first):].strip())
        second = fit_single_line(remainder, MAX_CHARS, protected_phrases) if remainder else ""
        cue["lines"] = [x for x in [first, second] if x][:MAX_LINES]
        validated.append(cue)
    return validated


def remove_empty_or_tiny_cues(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for cue in cues:
        text = " ".join(cue.get("lines", [])).strip()
        dur = cue["end_ms"] - cue["start_ms"]
        if not text:
            continue
        if cue["type"] == "dialogue" and dur < 200:
            continue
        if cue["type"] == "sound" and dur < STRONG_SOUND_MIN_MS:
            continue
        cleaned.append(cue)
    return cleaned
