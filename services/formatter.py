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
MUSIC_MIN_DURATION_MS = 5000
MIN_GAP_FOR_SOUND_MS = 400
SOUND_MERGE_GAP_MS = 700

PROTECTED_DEFAULTS: List[str] = []

FUNCTION_WORDS = {
    "a", "an", "the", "of", "to", "and", "or", "but",
    "with", "from", "in", "on", "at", "for", "that",
    "this", "these", "those", "is", "are", "was", "were",
}

BRACKET_TAG_RE = re.compile(r"\[[^\]]+\]")


def process_caption_job(
    backbone_srt_text: str,
    timestamps: Any,
    protected_phrases: List[str] | None = None,
    output_formats: List[str] | None = None,
) -> Dict[str, Any]:
    """
    Safer dialogue-first formatter based on the existing baseline.

    Design goals:
    - dialogue text comes from AssemblyAI SRT backbone
    - never raw-char-slice dialogue when splitting
    - if a cue cannot fit safely, split into additional cue events instead of dropping text
    - sound cues never displace dialogue
    - two-speaker cues are one speaker per line or split into separate events
    """
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

    print("[FORMATTER] Inserting sound cues into real gaps only")
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


def detect_protected_phrases_from_backbone(backbone: List[Dict[str, Any]]) -> List[str]:
    """
    Auto-detect likely proper nouns/titles from transcript text.
    """
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

    # common short entertainment title alias handling
    if "Below Deck Med" not in seen:
        for cue in backbone:
            raw_text = " ".join(cue.get("lines", [])).strip()
            if "Below Deck Med" in raw_text:
                phrases.append("Below Deck Med")
                seen.add("Below Deck Med")
                break

    return phrases


def build_runtime_protected_phrases(
    backbone: List[Dict[str, Any]],
    protected_phrases: List[str] | None = None,
) -> List[str]:
    explicit = protected_phrases or []
    detected = detect_protected_phrases_from_backbone(backbone)
    combined = list(dict.fromkeys(PROTECTED_DEFAULTS + explicit + detected))
    return combined


def build_dialogue_cues_from_srt(
    backbone: List[Dict[str, Any]],
    tokens: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
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
    text = BRACKET_TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


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


def align_runs_to_text(runs: List[Dict[str, Any]], full_text: str) -> List[Dict[str, Any]]:
    """
    Preserve multi-speaker structure, but keep the full SRT dialogue text intact.
    """
    if not full_text:
        return runs

    if len(runs) <= 1:
        return [{"speaker": runs[0]["speaker"] if runs else "A", "text": full_text}]

    total_chars = sum(max(1, len(r.get("text", ""))) for r in runs)
    assigned: List[Dict[str, Any]] = []
    cursor = 0

    for i, run in enumerate(runs):
        if i == len(runs) - 1:
            piece = full_text[cursor:].strip()
        else:
            share = max(1, int(len(full_text) * (max(1, len(run.get("text", ""))) / total_chars)))
            cut = find_nearest_space(full_text, cursor + share)
            piece = full_text[cursor:cut].strip()
            cursor = cut

        assigned.append({"speaker": run["speaker"], "text": piece})

    cleaned = [r for r in assigned if r["text"]]
    return cleaned or [{"speaker": "A", "text": full_text}]


def find_nearest_space(text: str, target: int) -> int:
    target = max(0, min(len(text), target))
    if target >= len(text):
        return len(text)
    if text[target].isspace():
        return target

    left = target
    right = target
    while left > 0 or right < len(text):
        if left > 0:
            left -= 1
            if text[left].isspace():
                return left
        if right < len(text):
            if text[right].isspace():
                return right
            right += 1

    return target


def build_sound_events(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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

        raw_events.append(
            {
                "label": label,
                "start_ms": start,
                "end_ms": end,
            }
        )

    merged = merge_sound_events(raw_events)
    filtered = filter_sound_events(merged)
    return filtered


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
        close_gap = ev["start_ms"] - prev["end_ms"] <= SOUND_MERGE_GAP_MS
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
    last_label = None
    last_label_end = -999999

    for ev in events:
        label = ev["label"]
        duration = ev["end_ms"] - ev["start_ms"]

        if label == "[♪]" and duration < MUSIC_MIN_DURATION_MS:
            continue

        if label == "[♪]" and ev["start_ms"] - last_music_end < 5000:
            continue

        if label == last_label and ev["start_ms"] - last_label_end < 1500:
            continue

        if label == "[♪]":
            last_music_end = ev["end_ms"]

        last_label = label
        last_label_end = ev["end_ms"]
        filtered.append(ev)

    return filtered


def place_sound_cues_in_gaps(
    sound_events: List[Dict[str, Any]],
    dialogue_cues: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not sound_events:
        return []

    dialogue_sorted = sorted(dialogue_cues, key=lambda c: (c["start_ms"], c["end_ms"]))
    sound_cues: List[Dict[str, Any]] = []

    for ev in sound_events:
        placed = fit_sound_into_dialogue_gap(ev, dialogue_sorted)
        if placed:
            sound_cues.append(placed)

    return sound_cues


def fit_sound_into_dialogue_gap(
    ev: Dict[str, Any],
    dialogue_sorted: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    desired_start = ev["start_ms"]
    desired_end = max(ev["end_ms"], desired_start + MIN_SOUND_MS)

    prev_dialogue = None
    next_dialogue = None

    for cue in dialogue_sorted:
        if cue["end_ms"] <= desired_start:
            prev_dialogue = cue
            continue

        if cue["start_ms"] >= desired_start:
            next_dialogue = cue
            break

        if intervals_overlap(desired_start, desired_end, cue["start_ms"], cue["end_ms"]):
            return None

    gap_start = prev_dialogue["end_ms"] if prev_dialogue else desired_start
    gap_end = next_dialogue["start_ms"] if next_dialogue else desired_end

    if gap_end - gap_start < MIN_GAP_FOR_SOUND_MS:
        return None

    start = max(desired_start, gap_start)
    end = min(desired_end, gap_end)

    # Do not let sound cues consume the entire lead-in before dialogue.
    if next_dialogue is not None and gap_end - gap_start > MIN_SOUND_MS + 250:
        end = min(end, gap_end - 500)

    if end - start < MIN_SOUND_MS:
        if gap_end - gap_start >= MIN_SOUND_MS:
            start = gap_start
            end = start + MIN_SOUND_MS
        else:
            return None

    return {
        "idx": 0,
        "start_ms": start,
        "end_ms": end,
        "lines": [ev["label"][:MAX_CHARS]],
        "type": "sound",
        "meta": {
            "sound_label": ev["label"],
            "sound_priority": sound_priority(ev["label"]),
        },
    }

def intervals_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


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
                    "meta": {"runs": [run], "two_speaker": False, "dialogue_text": run["text"]},
                }
            )
            current_start = new_end

    return out


def format_cues(cues: List[Dict[str, Any]], protected_phrases: List[str], max_rounds: int = 3) -> List[Dict[str, Any]]:
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
            left_text = clean_dialogue_text(runs[0]["text"])
            right_text = clean_dialogue_text(runs[1]["text"])

            if len(left_text) <= MAX_CHARS - 2 and len(right_text) <= MAX_CHARS - 2:
                cue["lines"] = [f"- {left_text}", f"- {right_text}"]
                cue["meta"]["dialogue_text"] = f"{left_text} {right_text}".strip()
                formatted.append(cue)
                continue

            child_results = format_cues(split_two_speaker_cue(cue), protected_phrases, max_rounds=max_rounds)
            formatted.extend(child_results)
            continue

        text = cue["meta"].get("dialogue_text", "")
        text = clean_dialogue_text(text)

        broken, needs_event_split = smart_split(text, protected_phrases)

        if needs_event_split and max_rounds > 0:
            children = split_cue(cue)
            child_results = format_cues(children, protected_phrases, max_rounds=max_rounds - 1)
            formatted.extend(child_results)
            continue

        cue["lines"] = broken[:MAX_LINES]

        if violates_line_limits(cue["lines"], MAX_LINES, MAX_CHARS):
            if max_rounds > 0:
                children = split_cue(cue)
                child_results = format_cues(children, protected_phrases, max_rounds=max_rounds - 1)
                formatted.extend(child_results)
            else:
                cue["lines"] = [fit_single_line(line, MAX_CHARS, protected_phrases) for line in cue["lines"][:MAX_LINES]]
                formatted.append(cue)
            continue

        if count_function_word_endings(cue["lines"]) > 0 and max_rounds > 0:
            improved, needs_split = smart_split(text, protected_phrases, prefer_balance=False)
            if needs_split:
                children = split_cue(cue)
                child_results = format_cues(children, protected_phrases, max_rounds=max_rounds - 1)
                formatted.extend(child_results)
                continue
            cue["lines"] = improved[:MAX_LINES]

        formatted.append(cue)

    return formatted

def split_two_speaker_cue(cue: Dict[str, Any]) -> List[Dict[str, Any]]:
    runs = cue["meta"].get("runs", [])
    if len(runs) != 2:
        return [cue]

    total_duration = max(1, cue["end_ms"] - cue["start_ms"])
    left_len = max(1, len(runs[0]["text"]))
    right_len = max(1, len(runs[1]["text"]))
    denom = left_len + right_len
    split_time = cue["start_ms"] + int(total_duration * (left_len / denom))

    first = {
        **cue,
        "start_ms": cue["start_ms"],
        "end_ms": split_time,
        "lines": [],
        "type": "dialogue",
        "meta": {"runs": [runs[0]], "two_speaker": False, "dialogue_text": runs[0]["text"]},
    }
    second = {
        **cue,
        "start_ms": split_time,
        "end_ms": cue["end_ms"],
        "lines": [],
        "type": "dialogue",
        "meta": {"runs": [runs[1]], "two_speaker": False, "dialogue_text": runs[1]["text"]},
    }
    return [first, second]


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

    left_cue = {
        **cue,
        "start_ms": cue["start_ms"],
        "end_ms": split_time,
        "lines": [],
        "meta": {
            **cue["meta"],
            "dialogue_text": left_text,
            "runs": [{"speaker": first_speaker, "text": left_text}],
            "two_speaker": False,
        },
    }

    right_cue = {
        **cue,
        "start_ms": split_time,
        "end_ms": cue["end_ms"],
        "lines": [],
        "meta": {
            **cue["meta"],
            "dialogue_text": right_text,
            "runs": [{"speaker": first_speaker, "text": right_text}],
            "two_speaker": False,
        },
    }

    return [left_cue, right_cue]


def choose_split_midpoint(words: List[str]) -> int:
    midpoint = len(words) // 2
    best = midpoint
    best_score = 999999

    for i in range(max(1, midpoint - 5), min(len(words), midpoint + 6)):
        left = words[:i]
        right = words[i:]
        if not left or not right:
            continue

        left_text = " ".join(left)
        right_text = " ".join(right)

        score = abs(len(left_text) - len(right_text))

        left_last = re.sub(r"[^\w',;:.?!-]+$", "", left[-1]).lower()
        if left_last in FUNCTION_WORDS:
            score += 25

        if re.search(r"[,.;:?!]$", left[-1]):
            score -= 25
        if len(right) <= 2:
            score += 18
        if len(left) <= 2:
            score += 10

        if score < best_score:
            best_score = score
            best = i

    return best

def smart_split(
    text: str,
    protected_phrases: List[str],
    prefer_balance: bool = True,
) -> Tuple[List[str], bool]:
    """
    Never raw-character-slice a word.
    Prefer punctuation / full phrase boundaries over visual balance.
    If it cannot fit safely into 2 lines, signal caller to split into additional events.
    """
    text = clean_dialogue_text(text)
    if not text:
        return [""], False

    if len(text) <= MAX_CHARS:
        return [text], False

    words = text.split()
    candidates: List[Tuple[int, List[str]]] = []

    for i in range(1, len(words)):
        left = " ".join(words[:i]).strip()
        right = " ".join(words[i:]).strip()

        if len(left) > MAX_CHARS or len(right) > MAX_CHARS:
            continue

        score = split_score(left, right, protected_phrases, prefer_balance=prefer_balance)
        candidates.append((score, [left, right]))

    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1], False

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

    if not re.search(r"[,\.\?!:;]$", left):
        score += 8

    left_last = sanitize_last_word(left)
    if left_last in FUNCTION_WORDS:
        score += 15

    combined = left + "\n" + right
    for phrase in protected_phrases:
        if phrase and phrase in combined.replace("\n", " ") and phrase.replace(" ", "\n") in combined:
            score += 100

    if len(right.split()) <= 2:
        score += 12

    return score




def conservative_readability_pass(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Very conservative readability pass.
    Do NOT merge/split dialogue text here.
    Only extend short sound cues slightly when there is clear room.
    """
    cues = sorted(cues, key=lambda c: (c["start_ms"], c["end_ms"], 0 if c["type"] == "dialogue" else 1))
    out: List[Dict[str, Any]] = []

    for i, cue in enumerate(cues):
        cue = cue.copy()
        if cue["type"] == "sound":
            min_end = cue["start_ms"] + MIN_SOUND_MS
            if cue["end_ms"] < min_end:
                next_start = cues[i + 1]["start_ms"] if i < len(cues) - 1 else min_end
                cue["end_ms"] = min(max(min_end, cue["end_ms"]), next_start)
        out.append(cue)

    return out

def post_format_repair(cues: List[Dict[str, Any]], protected_phrases: List[str]) -> List[Dict[str, Any]]:
    """
    Lightweight, non-destructive repair pass:
    - preserve two-speaker dash formatting
    - normalize sound labels again
    - re-split invalid dialogue safely
    - merge obvious fragment cues with neighbors
    - de-duplicate repeated boundary words
    """
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
            if len(left) <= MAX_CHARS - 2 and len(right) <= MAX_CHARS - 2:
                cue["lines"] = [f"- {left}", f"- {right}"]
                cue["meta"]["dialogue_text"] = f"{left} {right}".strip()
                repaired.append(cue)
                continue

        text = cue["meta"].get("dialogue_text", " ".join(cue.get("lines", []))).strip()
        text = clean_dialogue_text(text)
        lines, needs_event_split = smart_split(text, protected_phrases)

        if needs_event_split:
            split_children = split_cue(cue)
            if split_children != [cue]:
                rendered: List[Dict[str, Any]] = []
                for child in split_children:
                    rendered.extend(format_cues([child], protected_phrases, max_rounds=0))
                repaired.extend(rendered)
                continue

        cue["meta"]["dialogue_text"] = text
        cue["lines"] = lines[:MAX_LINES]
        repaired.append(cue)

    merged: List[Dict[str, Any]] = []
    i = 0
    while i < len(repaired):
        cue = repaired[i]
        if (
            cue["type"] == "dialogue"
            and not (cue["meta"].get("two_speaker") and len(cue["meta"].get("runs", [])) == 2)
            and i < len(repaired) - 1
            and repaired[i + 1]["type"] == "dialogue"
            and not (repaired[i + 1]["meta"].get("two_speaker") and len(repaired[i + 1]["meta"].get("runs", [])) == 2)
        ):
            curr_text = cue["meta"].get("dialogue_text", " ".join(cue.get("lines", [])).strip())
            next_text = repaired[i + 1]["meta"].get("dialogue_text", " ".join(repaired[i + 1].get("lines", [])).strip())

            curr_words = curr_text.split()
            is_fragment = (
                len(curr_words) <= 2
                or curr_text in {"It's.", "And I", "Who made"}
                or (curr_words and curr_words[-1].lower() in FUNCTION_WORDS)
            )

            if is_fragment:
                combined = clean_dialogue_text((curr_text + " " + next_text).strip())
                lines, needs_split = smart_split(combined, protected_phrases)
                if not needs_split:
                    new_cue = cue.copy()
                    new_cue["end_ms"] = repaired[i + 1]["end_ms"]
                    new_cue["meta"] = dict(cue["meta"])
                    new_cue["meta"]["dialogue_text"] = combined
                    new_cue["meta"]["runs"] = [{"speaker": cue["meta"].get("runs", [{"speaker":"A"}])[0]["speaker"], "text": combined}]
                    new_cue["meta"]["two_speaker"] = False
                    new_cue["lines"] = lines[:MAX_LINES]
                    merged.append(new_cue)
                    i += 2
                    continue

        merged.append(cue)
        i += 1

    for i in range(1, len(merged)):
        prev = merged[i - 1]
        curr = merged[i]
        if prev["type"] == "dialogue" and curr["type"] == "dialogue":
            prev_text = prev["meta"].get("dialogue_text", " ".join(prev.get("lines", [])).strip())
            curr_text = curr["meta"].get("dialogue_text", " ".join(curr.get("lines", [])).strip())
            pw = prev_text.split()
            cw = curr_text.split()
            if pw and cw and pw[-1].lower() == cw[0].lower():
                curr_text = " ".join(cw[1:]).strip()
                if curr_text:
                    lines, needs_split = smart_split(curr_text, protected_phrases)
                    if not needs_split:
                        curr["meta"]["dialogue_text"] = curr_text
                        curr["lines"] = lines[:MAX_LINES]
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
                if cue["end_ms"] - cue["start_ms"] < MIN_SOUND_MS:
                    continue
            elif prev["type"] == "sound" and cue["type"] == "dialogue":
                cue["start_ms"] = max(cue["start_ms"], prev["end_ms"])
                if cue["end_ms"] <= cue["start_ms"]:
                    cue["end_ms"] = cue["start_ms"] + 1
            else:
                cue["start_ms"] = prev["end_ms"]
                if cue["end_ms"] <= cue["start_ms"]:
                    cue["end_ms"] = cue["start_ms"] + 1

        resolved.append(cue)

    return resolved


def clean_dialogue_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" ,", ",").replace(" .", ".").replace(" !", "!").replace(" ?", "?")
    return text


def sanitize_last_word(text: str) -> str:
    text = re.sub(r"[^\w']+$", "", text.strip().lower())
    parts = text.split()
    return parts[-1] if parts else ""




def final_hard_validate(cues: List[Dict[str, Any]], protected_phrases: List[str]) -> List[Dict[str, Any]]:
    """
    Final safety pass after AI:
    - preserve cue count/timing unless cue is empty
    - ensure max 2 lines, 32 chars
    - if invalid, fall back to deterministic splitter using stored dialogue_text
    """
    validated: List[Dict[str, Any]] = []
    for cue in cues:
        if cue.get("type") == "sound":
            lines = cue.get("lines", [])[:1]
            if not lines:
                continue
            cue["lines"] = [clean_dialogue_text(lines[0])[:MAX_CHARS]]
            validated.append(cue)
            continue

        text = cue.get("meta", {}).get("dialogue_text", " ".join(cue.get("lines", []))).strip()
        text = clean_dialogue_text(text)
        lines = [clean_dialogue_text(x) for x in cue.get("lines", []) if clean_dialogue_text(x)]
        if not lines:
            lines = [text] if text else []

        # keep valid two-speaker dash format
        runs = cue.get("meta", {}).get("runs", [])
        is_two = bool(cue.get("meta", {}).get("two_speaker")) and len(runs) == 2
        if is_two:
            if len(lines) == 2 and all(line.startswith("- ") and len(line) <= MAX_CHARS for line in lines):
                validated.append(cue)
                continue
            # fallback deterministic two-speaker
            left = clean_dialogue_text(runs[0].get("text", ""))
            right = clean_dialogue_text(runs[1].get("text", ""))
            if left and right and len(left) <= MAX_CHARS - 2 and len(right) <= MAX_CHARS - 2:
                cue["lines"] = [f"- {left}", f"- {right}"]
                validated.append(cue)
                continue

        if len(lines) <= MAX_LINES and all(len(line) <= MAX_CHARS for line in lines):
            cue["lines"] = lines
            validated.append(cue)
            continue

        broken, needs_split = smart_split(text, protected_phrases)
        if not needs_split:
            cue["lines"] = broken[:MAX_LINES]
            validated.append(cue)
            continue

        # keep best-effort deterministic fit
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

        if cue["type"] == "sound" and dur < MIN_SOUND_MS:
            continue

        cleaned.append(cue)

    return cleaned
