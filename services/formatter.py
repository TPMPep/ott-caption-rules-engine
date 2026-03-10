import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from services.assembly import normalize_tokens, is_sound_token
from services.exporters import parse_srt, export_srt, export_vtt, export_scc
from services.qc import qc_report

try:
    from services.editorial_ai import editorial_refine_cues
except Exception:
    def editorial_refine_cues(cues, protected_phrases):
        return cues

MAX_LINES = 2
MAX_CHARS = 32
MIN_DIALOGUE_MS = 800
MIN_SOUND_MS = 1000
MAX_SOUND_MS = 3500
LONG_GAP_MS = 1400
MAX_CPS = 20.0
MERGE_SAME_SPEAKER_GAP_MS = 220
TWO_SPEAKER_GAP_MS = 220
PAUSE_BOUNDARY_MS = 550
CLAUSE_BOUNDARY_MS = 420
ENABLE_EDITORIAL_AI = False

ALLOWED_SOUND_LABELS = {"[APPLAUSE]", "[LAUGHTER]", "[MUSIC]", "[CHEERING]"}
DEFAULT_PROTECTED_PHRASES = [
    "Watch What Happens Live",
    "Below Deck Med",
    "Below Deck Mediterranean",
    "Andy Cohen",
]

FUNCTION_WORDS = {
    "a", "an", "the", "of", "to", "and", "or", "but", "with", "from", "in", "on",
    "at", "for", "that", "this", "these", "those", "is", "are", "was", "were", "be",
    "been", "being", "because", "if", "then", "than", "as", "by", "it", "i", "you",
}

WEAK_LINE_STARTS = {"and", "or", "but", "to", "of", "for", "with", "because", "that", "this", "these", "those"}
WEAK_LINE_ENDS = FUNCTION_WORDS | {"who", "what", "when", "where", "why", "how"}
BRACKET_TAG_RE = re.compile(r"\[[^\]]+\]")
WORD_RE = re.compile(r"[A-Za-z0-9']+")


def process_caption_job(
    backbone_srt_text: str,
    timestamps: Any,
    protected_phrases: Optional[List[str]] = None,
    output_formats: Optional[List[str]] = None,
) -> Dict[str, Any]:
    protected_phrases = protected_phrases or []
    output_formats = output_formats or ["srt"]

    backbone = parse_srt(backbone_srt_text)
    cues_in = len(backbone)

    runtime_protected = build_runtime_protected_phrases(backbone, timestamps, protected_phrases)
    tokens = normalize_tokens(timestamps)
    tokens.sort(key=lambda w: (int(w.get("start_ms", 0)), int(w.get("end_ms", 0))))

    utterances = extract_utterances(timestamps, tokens)
    raw_events = build_raw_events_from_utterances(utterances)
    raw_events = merge_adjacent_same_speaker_dialogue(raw_events)
    raw_events = bridge_sound_events(raw_events)

    cues = materialize_cues(raw_events, runtime_protected)
    cues = maybe_pack_two_speaker_cues(cues)
    cues = resolve_overlaps(cues)
    cues = drop_bad_cues(cues)

    if ENABLE_EDITORIAL_AI:
        cues = editorial_refine_cues(cues, runtime_protected)

    for i, cue in enumerate(cues, start=1):
        cue["idx"] = i

    srt_out = export_srt(cues)
    vtt_out = export_vtt(cues) if "vtt" in output_formats else None
    scc_out = export_scc(cues) if "scc" in output_formats else None
    qc = qc_report(cues_in, cues, runtime_protected)

    return {
        "srt": srt_out,
        "vtt": vtt_out,
        "scc": scc_out,
        "qc": qc,
    }


# -----------------------------------------------------------------------------
# Phrase protection and normalization
# -----------------------------------------------------------------------------

def build_runtime_protected_phrases(
    backbone: List[Dict[str, Any]],
    timestamps: Any,
    explicit: Optional[List[str]] = None,
) -> List[str]:
    phrases: List[str] = []
    seen = set()

    for phrase in DEFAULT_PROTECTED_PHRASES + (explicit or []):
        key = phrase.strip().lower()
        if key and key not in seen:
            seen.add(key)
            phrases.append(phrase.strip())

    for cue in backbone:
        text = " ".join(cue.get("lines", [])).strip()
        for phrase in detect_protected_phrases_in_text(text):
            key = phrase.lower()
            if key not in seen:
                seen.add(key)
                phrases.append(phrase)

    if isinstance(timestamps, dict):
        for utter in timestamps.get("utterances") or []:
            for phrase in detect_protected_phrases_in_text(utter.get("text", "")):
                key = phrase.lower()
                if key not in seen:
                    seen.add(key)
                    phrases.append(phrase)

    return phrases


def detect_protected_phrases_in_text(text: str) -> List[str]:
    text = preserve_inline_uncertainty(text)
    found: List[str] = []
    if not text:
        return found

    patterns = re.findall(r"\b(?:[A-Z][a-z]+|[A-Z]{2,})(?:\s+(?:[A-Z][a-z]+|[A-Z]{2,}|Med)){1,5}\b", text)
    for phrase in patterns:
        words = phrase.split()
        if 2 <= len(words) <= 6:
            lowered = [w.lower() for w in words]
            if sum(1 for w in lowered if w in FUNCTION_WORDS) >= len(words) - 1:
                continue
            found.append(phrase.strip())
    return list(dict.fromkeys(found))


def clean_dialogue_text(text: str) -> str:
    text = preserve_inline_uncertainty(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    text = re.sub(r"([\[\(])\s+", r"\1", text)
    text = re.sub(r"\s+([\]\)])", r"\1", text)
    # very conservative repairs for specific ASR split damage
    text = re.sub(r"\btonight\.\s+right\?", "tonight, right?", text, flags=re.IGNORECASE)
    text = re.sub(r"\b([Yy])es\.\s+yes\.", r"\1es, yes.", text)
    text = re.sub(r"\bI\'m your host\.\s+Andy Cohen\.", "I'm your host, Andy Cohen.", text)
    text = re.sub(r"\bI haven\'t\.\s+but\b", "I haven't, but", text)
    return text.strip()


def preserve_inline_uncertainty(text: str) -> str:
    def repl(match: re.Match) -> str:
        token = match.group(0).upper()
        return " [INAUDIBLE] " if token.startswith("[INAUDIBLE") else " " + token + " "
    return BRACKET_TAG_RE.sub(repl, text or "")


def strip_punctuation_token(word: str) -> str:
    match = WORD_RE.search(word or "")
    return match.group(0).lower() if match else ""


def tokenize_words_for_phrase_checks(words: Sequence[str]) -> List[str]:
    return [strip_punctuation_token(w) for w in words]


def split_crosses_protected_phrase(words: Sequence[str], split_idx: int, protected_phrases: Sequence[str]) -> bool:
    lowered = tokenize_words_for_phrase_checks(words)
    if split_idx <= 0 or split_idx >= len(lowered):
        return False
    for phrase in protected_phrases:
        phrase_words = [strip_punctuation_token(x) for x in phrase.split() if strip_punctuation_token(x)]
        if len(phrase_words) < 2:
            continue
        n = len(phrase_words)
        for i in range(0, len(lowered) - n + 1):
            if lowered[i:i + n] == phrase_words and i < split_idx < i + n:
                return True
    return False


# -----------------------------------------------------------------------------
# Utterance and token extraction
# -----------------------------------------------------------------------------

def extract_utterances(timestamps: Any, tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if isinstance(timestamps, dict) and isinstance(timestamps.get("utterances"), list) and timestamps.get("utterances"):
        out = []
        for utter in timestamps["utterances"]:
            words = []
            for w in utter.get("words") or []:
                start = int(w.get("start_ms", w.get("start", 0)))
                end = int(w.get("end_ms", w.get("end", start)))
                text = (w.get("text") or "").strip()
                if not text:
                    continue
                words.append({
                    "text": text,
                    "start_ms": start,
                    "end_ms": max(end, start + 1),
                    "speaker": w.get("speaker") or utter.get("speaker") or "A",
                })
            if not words:
                continue
            out.append({
                "speaker": utter.get("speaker") or words[0].get("speaker") or "A",
                "start_ms": int(utter.get("start", utter.get("start_ms", words[0]["start_ms"]))),
                "end_ms": int(utter.get("end", utter.get("end_ms", words[-1]["end_ms"]))),
                "text": utter.get("text") or join_token_texts(words),
                "words": words,
            })
        if out:
            return out

    # fallback: derive utterances from normalized tokens by speaker and longer pauses
    utterances: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for token in tokens:
        text = (token.get("text") or "").strip()
        if not text:
            continue
        speaker = token.get("speaker") or "A"
        start = int(token.get("start_ms", 0))
        end = int(token.get("end_ms", start + 1))
        gap = start - current["words"][-1]["end_ms"] if current and current.get("words") else 0
        new_utt = (
            current is None
            or speaker != current["speaker"]
            or gap > 1200
            or (current["words"] and ends_sentence(current["words"][-1]["text"]) and gap > 400)
        )
        word_obj = {"text": text, "start_ms": start, "end_ms": max(end, start + 1), "speaker": speaker}
        if new_utt:
            current = {"speaker": speaker, "start_ms": start, "end_ms": max(end, start + 1), "text": text, "words": [word_obj]}
            utterances.append(current)
        else:
            current["words"].append(word_obj)
            current["end_ms"] = max(current["end_ms"], end)
            current["text"] = join_token_texts(current["words"])
    return utterances


# -----------------------------------------------------------------------------
# Raw event construction
# -----------------------------------------------------------------------------

def build_raw_events_from_utterances(utterances: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for utter in utterances:
        speaker = utter.get("speaker") or "A"
        words = utter.get("words") or []
        if not words:
            continue

        current_kind = None
        bucket: List[Dict[str, Any]] = []
        for w in words:
            text = (w.get("text") or "").strip()
            if not text:
                continue
            if BRACKET_TAG_RE.fullmatch(text) and not is_effective_sound_token(text):
                continue
            kind = "sound" if is_effective_sound_token(text) else "dialogue"
            if current_kind is None:
                current_kind = kind
                bucket = [w]
                continue
            if kind == current_kind:
                bucket.append(w)
                continue
            events.append(build_event_from_word_bucket(current_kind, bucket, speaker))
            current_kind = kind
            bucket = [w]
        if bucket:
            events.append(build_event_from_word_bucket(current_kind or "dialogue", bucket, speaker))

    events = [e for e in events if e]
    events.sort(key=lambda e: (e["start_ms"], e["end_ms"], 0 if e["type"] == "dialogue" else 1))
    return events


def build_event_from_word_bucket(kind: str, words: Sequence[Dict[str, Any]], speaker: str) -> Optional[Dict[str, Any]]:
    if not words:
        return None
    start_ms = int(words[0]["start_ms"])
    end_ms = int(words[-1]["end_ms"])
    if kind == "sound":
        label = choose_sound_label([w["text"] for w in words])
        if not label:
            return None
        return {
            "type": "sound",
            "speaker": None,
            "start_ms": start_ms,
            "end_ms": max(end_ms, start_ms + 1),
            "text": label,
            "tokens": list(words),
        }

    text = clean_dialogue_text(join_token_texts(words))
    if not text:
        return None
    return {
        "type": "dialogue",
        "speaker": speaker,
        "start_ms": start_ms,
        "end_ms": max(end_ms, start_ms + 1),
        "text": text,
        "tokens": list(words),
    }


def is_effective_sound_token(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    if not is_sound_token(text):
        return False
    label = choose_sound_label([text])
    return bool(label)


def choose_sound_label(texts: Sequence[str]) -> Optional[str]:
    counts: Dict[str, int] = {}
    for raw in texts:
        inner = raw.strip().upper()
        if inner.startswith("[") and inner.endswith("]"):
            inner = inner[1:-1].strip()
        if not inner:
            continue
        pieces = [p.strip() for p in re.split(r"\s+AND\s+|,", inner) if p.strip()]
        for piece in pieces:
            if piece in {"NOISE", "SOUND", "SOUND EFFECT", "SPEAKER", "VOICE", "PAUSE", "TALKING"}:
                continue
            if piece.startswith("INAUDIBLE"):
                continue
            if piece == "MUSIC":
                counts["[MUSIC]"] = counts.get("[MUSIC]", 0) + 3
            elif piece == "APPLAUSE":
                counts["[APPLAUSE]"] = counts.get("[APPLAUSE]", 0) + 2
            elif piece == "LAUGHTER":
                counts["[LAUGHTER]"] = counts.get("[LAUGHTER]", 0) + 2
            elif piece == "CHEERING":
                counts["[CHEERING]"] = counts.get("[CHEERING]", 0) + 2

    if not counts:
        return None

    for preferred in ("[MUSIC]", "[APPLAUSE]", "[LAUGHTER]", "[CHEERING]"):
        if preferred in counts and counts[preferred] == max(counts.values()):
            return preferred
    return max(counts.items(), key=lambda kv: kv[1])[0]


def merge_adjacent_same_speaker_dialogue(events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not events:
        return []
    merged: List[Dict[str, Any]] = [copy_event(events[0])]
    for ev in events[1:]:
        prev = merged[-1]
        if (
            prev["type"] == "dialogue"
            and ev["type"] == "dialogue"
            and prev.get("speaker") == ev.get("speaker")
            and ev["start_ms"] - prev["end_ms"] <= MERGE_SAME_SPEAKER_GAP_MS
            and ends_sentence(prev["text"])
            and len(prev["text"]) + 1 + len(ev["text"]) <= 180
        ):
            prev["end_ms"] = ev["end_ms"]
            prev["tokens"].extend(ev.get("tokens") or [])
            prev["text"] = clean_dialogue_text(join_token_texts(prev["tokens"]))
        else:
            merged.append(copy_event(ev))
    return merged


def bridge_sound_events(events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not events:
        return []
    bridged: List[Dict[str, Any]] = []
    for i, ev in enumerate(events):
        cue = copy_event(ev)
        if cue["type"] != "sound":
            bridged.append(cue)
            continue

        prev_dialogue = None
        next_dialogue = None
        for j in range(i - 1, -1, -1):
            if events[j]["type"] == "dialogue":
                prev_dialogue = events[j]
                break
        for j in range(i + 1, len(events)):
            if events[j]["type"] == "dialogue":
                next_dialogue = events[j]
                break

        prev_end = prev_dialogue["end_ms"] if prev_dialogue else cue["start_ms"]
        next_start = next_dialogue["start_ms"] if next_dialogue else cue["end_ms"]
        available_before = max(0, cue["start_ms"] - prev_end)
        available_after = max(0, next_start - cue["end_ms"])
        current_dur = cue["end_ms"] - cue["start_ms"]
        target = 1400 if cue["text"] == "[MUSIC]" else 1200

        if available_before >= LONG_GAP_MS:
            cue["start_ms"] = max(prev_end + 80, cue["end_ms"] - min(target, available_before))
        elif current_dur < MIN_SOUND_MS and available_after >= 600:
            cue["end_ms"] = min(next_start - 80, cue["start_ms"] + target)
        elif current_dur < MIN_SOUND_MS and available_before >= 600:
            cue["start_ms"] = max(prev_end + 80, cue["end_ms"] - target)

        cue["end_ms"] = min(cue["end_ms"], next_start - 80 if next_dialogue else cue["end_ms"])
        cue["start_ms"] = max(cue["start_ms"], prev_end + 80 if prev_dialogue else cue["start_ms"])
        if cue["end_ms"] - cue["start_ms"] < MIN_SOUND_MS and cue["text"] != "[MUSIC]":
            # short non-music reactions are often more distracting than useful.
            continue
        if cue["end_ms"] - cue["start_ms"] < 700:
            continue
        if cue["end_ms"] - cue["start_ms"] > MAX_SOUND_MS:
            cue["end_ms"] = cue["start_ms"] + MAX_SOUND_MS
        bridged.append(cue)
    return bridged


# -----------------------------------------------------------------------------
# Materialization into caption cues
# -----------------------------------------------------------------------------

def materialize_cues(events: Sequence[Dict[str, Any]], protected_phrases: Sequence[str]) -> List[Dict[str, Any]]:
    cues: List[Dict[str, Any]] = []
    for ev in events:
        if ev["type"] == "sound":
            cues.append({
                "idx": 0,
                "start_ms": int(ev["start_ms"]),
                "end_ms": int(ev["end_ms"]),
                "lines": [ev["text"]],
                "type": "sound",
                "meta": {"label": ev["text"]},
            })
            continue

        pieces = split_dialogue_tokens_into_events(ev.get("tokens") or [], protected_phrases)
        if not pieces:
            text = clean_dialogue_text(ev.get("text", ""))
            lines, ok = format_dialogue_text(text, protected_phrases)
            if ok:
                cues.append(make_dialogue_cue(ev["start_ms"], ev["end_ms"], ev.get("speaker"), text, lines))
            continue
        for piece in pieces:
            text = clean_dialogue_text(join_token_texts(piece))
            lines, ok = format_dialogue_text(text, protected_phrases)
            if not ok:
                subpieces = split_long_piece(piece, protected_phrases)
                for sub in subpieces:
                    subtext = clean_dialogue_text(join_token_texts(sub))
                    sublines, subok = format_dialogue_text(subtext, protected_phrases)
                    if not subok:
                        sublines = hard_wrap_dialogue_text(subtext, protected_phrases)
                    cues.append(make_dialogue_cue(sub[0]["start_ms"], sub[-1]["end_ms"], ev.get("speaker"), subtext, sublines))
            else:
                cues.append(make_dialogue_cue(piece[0]["start_ms"], piece[-1]["end_ms"], ev.get("speaker"), text, lines))

    cues.sort(key=lambda c: (c["start_ms"], c["end_ms"], 0 if c["type"] == "dialogue" else 1))
    cues = massage_dialogue_timing(cues)
    return cues


def make_dialogue_cue(start_ms: int, end_ms: int, speaker: Optional[str], text: str, lines: Sequence[str]) -> Dict[str, Any]:
    return {
        "idx": 0,
        "start_ms": int(start_ms),
        "end_ms": int(max(end_ms, start_ms + 1)),
        "lines": list(lines),
        "type": "dialogue",
        "meta": {
            "speaker": speaker or "A",
            "runs": [{"speaker": speaker or "A", "text": text}],
            "two_speaker": False,
            "dialogue_text": text,
        },
    }


def split_dialogue_tokens_into_events(tokens: Sequence[Dict[str, Any]], protected_phrases: Sequence[str]) -> List[List[Dict[str, Any]]]:
    if not tokens:
        return []
    clauses = split_tokens_into_clauses(tokens, protected_phrases)
    if not clauses:
        return [list(tokens)]

    pieces: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for clause in clauses:
        trial = current + clause
        trial_text = clean_dialogue_text(join_token_texts(trial))
        trial_lines, ok = format_dialogue_text(trial_text, protected_phrases)
        dur = max(1, trial[-1]["end_ms"] - trial[0]["start_ms"])
        cps = cps_value(trial_text, dur)

        if current and (not ok or (cps > MAX_CPS and len(trial_text) > 18)):
            pieces.append(current)
            current = list(clause)
            continue

        if not current and (not ok or (cps > MAX_CPS and len(trial_text) > 18)):
            pieces.extend(split_long_piece(clause, protected_phrases))
            current = []
            continue

        current = trial

    if current:
        pieces.append(current)
    return pieces


def split_tokens_into_clauses(tokens: Sequence[Dict[str, Any]], protected_phrases: Sequence[str]) -> List[List[Dict[str, Any]]]:
    clauses: List[List[Dict[str, Any]]] = []
    start = 0
    words = [t.get("text", "") for t in tokens]
    for i in range(len(tokens) - 1):
        token_text = words[i]
        next_gap = int(tokens[i + 1]["start_ms"]) - int(tokens[i]["end_ms"])
        boundary = False
        if ends_sentence(token_text):
            boundary = True
        elif ends_clause(token_text) and len(join_token_texts(tokens[start:i + 1])) >= 12:
            boundary = True
        elif next_gap >= PAUSE_BOUNDARY_MS and len(tokens[start:i + 1]) >= 4 and len(join_token_texts(tokens[start:i + 1])) >= 14:
            boundary = True
        elif next_gap >= CLAUSE_BOUNDARY_MS and len(tokens[start:i + 1]) >= 5 and len(join_token_texts(tokens[start:i + 1])) >= 22:
            boundary = True

        if boundary and not split_crosses_protected_phrase(words, i + 1, protected_phrases):
            clauses.append(list(tokens[start:i + 1]))
            start = i + 1
    if start < len(tokens):
        clauses.append(list(tokens[start:]))
    return [c for c in clauses if c]


def split_long_piece(tokens: Sequence[Dict[str, Any]], protected_phrases: Sequence[str]) -> List[List[Dict[str, Any]]]:
    text = clean_dialogue_text(join_token_texts(tokens))
    lines, ok = format_dialogue_text(text, protected_phrases)
    if ok:
        return [list(tokens)]
    if len(tokens) <= 2:
        return [list(tokens)]

    split_idx = choose_event_split_index(tokens, protected_phrases)
    if split_idx <= 0 or split_idx >= len(tokens):
        return [list(tokens)]

    left = list(tokens[:split_idx])
    right = list(tokens[split_idx:])
    out: List[List[Dict[str, Any]]] = []
    for part in (left, right):
        part_text = clean_dialogue_text(join_token_texts(part))
        _, part_ok = format_dialogue_text(part_text, protected_phrases)
        if part_ok or len(part) <= 2:
            out.append(part)
        else:
            out.extend(split_long_piece(part, protected_phrases))
    return out


def choose_event_split_index(tokens: Sequence[Dict[str, Any]], protected_phrases: Sequence[str]) -> int:
    words = [t.get("text", "") for t in tokens]
    mid = max(1, len(tokens) // 2)
    best_idx = mid
    best_score = 10 ** 9
    for i in range(1, len(tokens)):
        if split_crosses_protected_phrase(words, i, protected_phrases):
            continue
        left_text = clean_dialogue_text(join_token_texts(tokens[:i]))
        right_text = clean_dialogue_text(join_token_texts(tokens[i:]))
        if not left_text or not right_text:
            continue
        score = abs(len(left_text) - len(right_text))
        left_last = strip_punctuation_token(words[i - 1])
        right_first = strip_punctuation_token(words[i])
        if ends_sentence(words[i - 1]):
            score -= 80
        elif ends_clause(words[i - 1]):
            score -= 35
        if is_protected_phrase_edge(words, i, protected_phrases):
            score -= 55
        if left_last in WEAK_LINE_ENDS:
            score += 70
        if right_first in WEAK_LINE_STARTS:
            score += 65
        if len(left_text) > 64 or len(right_text) > 64:
            score += 120
        if score < best_score:
            best_score = score
            best_idx = i
    return best_idx


def is_protected_phrase_edge(words: Sequence[str], split_idx: int, protected_phrases: Sequence[str]) -> bool:
    lowered = tokenize_words_for_phrase_checks(words)
    for phrase in protected_phrases:
        phrase_words = [strip_punctuation_token(x) for x in phrase.split() if strip_punctuation_token(x)]
        n = len(phrase_words)
        if n < 2:
            continue
        for i in range(0, len(lowered) - n + 1):
            if lowered[i:i + n] == phrase_words and split_idx in {i, i + n}:
                return True
    return False


def maybe_pack_two_speaker_cues(cues: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    packed: List[Dict[str, Any]] = []
    i = 0
    while i < len(cues):
        cue = copy_cue(cues[i])
        if cue["type"] != "dialogue" or i >= len(cues) - 1:
            packed.append(cue)
            i += 1
            continue
        nxt = cues[i + 1]
        if nxt["type"] != "dialogue":
            packed.append(cue)
            i += 1
            continue
        if cue["meta"].get("speaker") == nxt["meta"].get("speaker"):
            packed.append(cue)
            i += 1
            continue
        gap = nxt["start_ms"] - cue["end_ms"]
        left_text = cue["meta"].get("dialogue_text", "")
        right_text = nxt["meta"].get("dialogue_text", "")
        if gap <= TWO_SPEAKER_GAP_MS and len(left_text) <= MAX_CHARS - 2 and len(right_text) <= MAX_CHARS - 2:
            packed.append({
                "idx": 0,
                "start_ms": cue["start_ms"],
                "end_ms": nxt["end_ms"],
                "lines": [f"- {left_text}", f"- {right_text}"],
                "type": "dialogue",
                "meta": {
                    "speaker": cue["meta"].get("speaker"),
                    "runs": [
                        {"speaker": cue["meta"].get("speaker"), "text": left_text},
                        {"speaker": nxt["meta"].get("speaker"), "text": right_text},
                    ],
                    "two_speaker": True,
                    "dialogue_text": f"{left_text} {right_text}",
                },
            })
            i += 2
            continue
        packed.append(cue)
        i += 1
    return packed


def massage_dialogue_timing(cues: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = [copy_cue(c) for c in cues]
    out.sort(key=lambda c: (c["start_ms"], c["end_ms"], 0 if c["type"] == "dialogue" else 1))
    for i, cue in enumerate(out):
        prev_end = out[i - 1]["end_ms"] if i > 0 else cue["start_ms"]
        next_start = out[i + 1]["start_ms"] if i < len(out) - 1 else cue["end_ms"]
        if cue["type"] == "dialogue":
            min_end = cue["start_ms"] + MIN_DIALOGUE_MS
            if cue["end_ms"] < min_end and next_start - cue["start_ms"] >= MIN_DIALOGUE_MS:
                cue["end_ms"] = min(next_start - 80, min_end)
        else:
            if cue["end_ms"] - cue["start_ms"] > MAX_SOUND_MS:
                cue["end_ms"] = cue["start_ms"] + MAX_SOUND_MS
        cue["start_ms"] = max(cue["start_ms"], prev_end if i > 0 else cue["start_ms"])
        if i < len(out) - 1 and cue["end_ms"] > next_start:
            cue["end_ms"] = max(cue["start_ms"] + 1, next_start)
    return out


# -----------------------------------------------------------------------------
# Formatting
# -----------------------------------------------------------------------------

def format_dialogue_text(text: str, protected_phrases: Sequence[str]) -> Tuple[List[str], bool]:
    text = clean_dialogue_text(text)
    if not text:
        return [], False
    if len(text) <= 14:
        return [text], True
    words = text.split()

    if len(text) <= MAX_CHARS:
        if len(text) >= 24:
            two_line = best_two_line_split(words, protected_phrases, allow_short=False)
            if two_line:
                return two_line, True
        return [text], True

    if len(text) <= MAX_CHARS * 2:
        two_line = best_two_line_split(words, protected_phrases, allow_short=True)
        if two_line:
            return two_line, True
    return [], False


def hard_wrap_dialogue_text(text: str, protected_phrases: Sequence[str]) -> List[str]:
    text = clean_dialogue_text(text)
    words = text.split()
    two = best_two_line_split(words, protected_phrases, allow_short=True)
    if two:
        return two
    if len(text) <= MAX_CHARS:
        return [text]
    left_words: List[str] = []
    for word in words:
        trial = " ".join(left_words + [word]).strip()
        if len(trial) <= MAX_CHARS:
            left_words.append(word)
        else:
            break
    if not left_words:
        left = text[:MAX_CHARS].rstrip()
        right = text[MAX_CHARS:MAX_CHARS * 2].strip()
        return [left] + ([right] if right else [])
    left = " ".join(left_words)
    right = clean_dialogue_text(text[len(left):].strip())
    if len(right) > MAX_CHARS:
        right = right[:MAX_CHARS].rstrip()
    return [left] + ([right] if right else [])


def best_two_line_split(words: Sequence[str], protected_phrases: Sequence[str], allow_short: bool) -> Optional[List[str]]:
    if len(words) < 2:
        return None
    best: Optional[List[str]] = None
    best_score = 10 ** 9
    for i in range(1, len(words)):
        if split_crosses_protected_phrase(words, i, protected_phrases):
            continue
        left = clean_dialogue_text(" ".join(words[:i]))
        right = clean_dialogue_text(" ".join(words[i:]))
        if len(left) > MAX_CHARS or len(right) > MAX_CHARS:
            continue
        if not allow_short and (len(left) < 8 or len(right) < 8):
            continue
        score = split_score(words, i, left, right)
        if score < best_score:
            best_score = score
            best = [left, right]
    return best


def split_score(words: Sequence[str], idx: int, left: str, right: str) -> int:
    score = abs(len(left) - len(right))
    left_last_raw = words[idx - 1]
    right_first_raw = words[idx]
    left_last = strip_punctuation_token(left_last_raw)
    right_first = strip_punctuation_token(right_first_raw)

    if ends_sentence(left_last_raw):
        score -= 42
    elif ends_clause(left_last_raw):
        score -= 18
    else:
        score += 20

    if left_last in WEAK_LINE_ENDS:
        score += 55
    if right_first in WEAK_LINE_STARTS:
        score += 35

    if len(right.split()) == 1:
        score += 60
    if len(left.split()) == 1:
        score += 45
    if len(left) < 10 or len(right) < 10:
        score += 35
    if len(left.split()) <= 2 and left.split()[0].lower() in {"and", "but", "so"}:
        score += 65
    if len(right.split()) <= 2 and right.split()[0].lower() in {"and", "but", "so", "with", "from"}:
        score += 55

    awkward_heads = {"and i", "and thank", "it's", "who made", "what was", "because i"}
    if " ".join(left.split()[-2:]).lower() in awkward_heads:
        score += 55
    if " ".join(right.split()[:2]).lower() in awkward_heads:
        score += 45
    return score


def ends_sentence(text: str) -> bool:
    return bool(re.search(r"[.!?][\]\)]*$", text or ""))


def ends_clause(text: str) -> bool:
    return bool(re.search(r"[,;:][\]\)]*$", text or ""))


def cps_value(text: str, duration_ms: int) -> float:
    stripped = re.sub(r"\[[^\]]+\]", "", text or "")
    chars = len(stripped.replace("\n", "").strip())
    if duration_ms <= 0:
        return float(chars)
    return chars / max(0.001, duration_ms / 1000.0)


# -----------------------------------------------------------------------------
# Joining and final validation helpers
# -----------------------------------------------------------------------------

def join_token_texts(tokens: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for t in tokens:
        txt = (t.get("text") or "").strip()
        if not txt:
            continue
        parts.append(txt)
    return clean_dialogue_text(" ".join(parts))


def copy_event(ev: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(ev)
    out["tokens"] = list(ev.get("tokens") or [])
    return out


def copy_cue(cue: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(cue)
    out["lines"] = list(cue.get("lines") or [])
    out["meta"] = dict(cue.get("meta") or {})
    if "runs" in out["meta"]:
        out["meta"]["runs"] = [dict(r) for r in out["meta"].get("runs") or []]
    return out


def resolve_overlaps(cues: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered = [copy_cue(c) for c in cues]
    ordered.sort(key=lambda c: (c["start_ms"], c["end_ms"], 0 if c["type"] == "dialogue" else 1))
    out: List[Dict[str, Any]] = []
    for cue in ordered:
        if not out:
            out.append(cue)
            continue
        prev = out[-1]
        if cue["start_ms"] < prev["end_ms"]:
            cue["start_ms"] = prev["end_ms"]
            if cue["end_ms"] <= cue["start_ms"]:
                cue["end_ms"] = cue["start_ms"] + 1
        out.append(cue)
    return out


def drop_bad_cues(cues: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for cue in cues:
        if cue["type"] == "sound":
            label = (cue.get("lines") or [""])[0].strip().upper()
            if label not in ALLOWED_SOUND_LABELS:
                continue
            dur = cue["end_ms"] - cue["start_ms"]
            if dur < 700:
                continue
            cue = copy_cue(cue)
            cue["lines"] = [label]
            cleaned.append(cue)
            continue

        text = clean_dialogue_text(cue.get("meta", {}).get("dialogue_text", " ".join(cue.get("lines") or [])))
        if not text:
            continue
        if cue.get("meta", {}).get("two_speaker"):
            lines = cue.get("lines") or []
            if len(lines) == 2 and all(line.startswith("- ") and len(line) <= MAX_CHARS for line in lines):
                cleaned.append(cue)
            continue
        lines = [clean_dialogue_text(x) for x in (cue.get("lines") or []) if clean_dialogue_text(x)]
        if not lines:
            lines, ok = format_dialogue_text(text, DEFAULT_PROTECTED_PHRASES)
            if not ok:
                lines = hard_wrap_dialogue_text(text, DEFAULT_PROTECTED_PHRASES)
            cue = copy_cue(cue)
            cue["lines"] = lines
        if len(cue["lines"]) > MAX_LINES or any(len(x) > MAX_CHARS for x in cue["lines"]):
            continue
        cleaned.append(cue)
    return cleaned
