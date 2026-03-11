import json
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Project imports with local fallbacks for robustness/testability.
try:
    from services.assembly import normalize_tokens as _normalize_tokens, is_sound_token as _is_sound_token
except Exception:
    def _normalize_tokens(timestamps: Any) -> List[Dict[str, Any]]:
        source = []
        if isinstance(timestamps, dict):
            source = timestamps.get("words") or []
        elif isinstance(timestamps, list):
            source = timestamps
        out = []
        for item in source:
            out.append(
                {
                    "text": item.get("text", ""),
                    "start_ms": int(item.get("start_ms", item.get("start", 0) or 0)),
                    "end_ms": int(item.get("end_ms", item.get("end", 0) or 0)),
                    "speaker": item.get("speaker") or "A",
                }
            )
        return out

    def _is_sound_token(text: str) -> bool:
        text = (text or "").strip().upper()
        return text.startswith("[") and text.endswith("]")

try:
    from services.exporters import parse_srt as _parse_srt, export_srt as _export_srt, export_vtt as _export_vtt, export_scc as _export_scc
except Exception:
    def _ms_to_srt(ms: int) -> str:
        ms = max(0, int(ms))
        hh, rem = divmod(ms, 3600000)
        mm, rem = divmod(rem, 60000)
        ss, rem = divmod(rem, 1000)
        return f"{hh:02d}:{mm:02d}:{ss:02d},{rem:03d}"

    def _parse_srt(text: str) -> List[Dict[str, Any]]:
        text = (text or "").replace("\r\n", "\n").strip()
        blocks = re.split(r"\n\s*\n", text)
        cues = []
        for block in blocks:
            lines = [ln for ln in block.split("\n") if ln.strip() != ""]
            if len(lines) < 2:
                continue
            if re.match(r"^\d+$", lines[0].strip()):
                idx = int(lines[0].strip())
                timing = lines[1]
                body = lines[2:]
            else:
                idx = len(cues) + 1
                timing = lines[0]
                body = lines[1:]
            m = re.match(r"(\d\d:\d\d:\d\d,\d\d\d)\s+-->\s+(\d\d:\d\d:\d\d,\d\d\d)", timing)
            if not m:
                continue
            cues.append({"idx": idx, "start_ms": _parse_tc(m.group(1)), "end_ms": _parse_tc(m.group(2)), "lines": body})
        return cues

    def _parse_tc(tc: str) -> int:
        hh, mm, ssms = tc.split(":")
        ss, ms = ssms.split(",")
        return ((int(hh) * 60 + int(mm)) * 60 + int(ss)) * 1000 + int(ms)

    def _export_srt(cues: List[Dict[str, Any]]) -> str:
        out = []
        for i, cue in enumerate(cues, 1):
            out.append(str(i))
            out.append(f"{_ms_to_srt(cue['start_ms'])} --> {_ms_to_srt(cue['end_ms'])}")
            out.extend(cue.get("lines") or [""])
            out.append("")
        return "\n".join(out).strip() + "\n"

    def _export_vtt(cues: List[Dict[str, Any]]) -> str:
        body = _export_srt(cues).replace(",", ".")
        return "WEBVTT\n\n" + body

    def _export_scc(cues: List[Dict[str, Any]]) -> Optional[str]:
        return None

try:
    from services.qc import qc_report as _qc_report
except Exception:
    def _qc_report(cues_in: int, cues_out: List[Dict[str, Any]], protected_phrases: List[str]) -> Dict[str, Any]:
        return {"input_cues": cues_in, "output_cues": len(cues_out), "protected_phrases": protected_phrases}

MAX_LINES = 2
MAX_CHARS = 32
MAX_CUE_CHARS = MAX_LINES * MAX_CHARS
TARGET_CPS = 18.0
MAX_CPS = 20.0
MIN_DIALOGUE_MS = 800
MIN_SOUND_MS = 700
TARGET_SOUND_MS = 1800
MAX_SOUND_MS = 3500
MERGE_GAP_MS = 650
WINDOW_PAD_MS = 700
SAME_SPEAKER_HARD_GAP_MS = 3200
SOUND_CLUSTER_GAP_MS = 700
TWO_SPEAKER_GAP_MS = 900
MAX_TWO_SPEAKER_WINDOW_MS = 4500
INLINE_DIALOGUE_TAGS = {"[INAUDIBLE]", "[UNINTELLIGIBLE]"}
FUNCTION_WORDS = {
    "a", "an", "the", "of", "to", "and", "or", "but", "with", "from", "in", "on", "at", "for",
    "that", "this", "these", "those", "is", "are", "was", "were", "be", "been", "being", "it", "i",
}
WEAK_ENDS = FUNCTION_WORDS | {"who", "what", "which", "when", "where", "why", "how"}
WEAK_STARTS = {"and", "or", "but", "to", "of", "for", "with", "because", "that", "this", "these", "those"}
CONNECTORS = {"of", "the", "and", "a", "an", "to", "for", "with", "vs", "v", "de", "du", "la", "le", "von"}
ALLOWED_SOUND = {"[APPLAUSE]", "[LAUGHTER]", "[MUSIC]", "[CHEERING]"}
SOUND_PRIORITY = {"[MUSIC]": 1, "[CHEERING]": 2, "[LAUGHTER]": 3, "[APPLAUSE]": 4}
# Single-word dialogue cues allowed to stay as their own cue (not merged with previous)
ALLOWED_STANDALONE_WORDS = {"yes", "no", "ok", "okay", "right", "correct", "wrong", "maybe", "sure", "absolutely", "exactly", "never"}
MIN_FRAGMENT_WORDS = 3
MIN_FRAGMENT_CHARS = 10
LONG_GAP_BRIDGE_MS = 2200
BRIDGE_PAD_MS = 120
TAG_RE = re.compile(r"\[[^\]]+\]")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")
TITLEISH_RE = re.compile(r"^[A-Z][A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?$|^[A-Z0-9]{2,}$")


def process_caption_job(
    backbone_srt_text: str,
    timestamps: Any,
    protected_phrases: Optional[List[str]] = None,
    output_formats: Optional[List[str]] = None,
) -> Dict[str, Any]:
    protected_phrases = protected_phrases or []
    output_formats = output_formats or ["srt"]

    backbone = _parse_srt(backbone_srt_text)
    tokens = normalize_tokens(timestamps)
    protected = build_runtime_protected_phrases(backbone, tokens, protected_phrases)

    dialogue_windows, sound_events = build_windows_from_backbone(backbone)
    dialogue_windows = merge_dialogue_windows(dialogue_windows, protected)

    for win in dialogue_windows:
        candidate_tokens = [t for t in tokens if t["start_ms"] < win["end_ms"] + WINDOW_PAD_MS and t["end_ms"] > win["start_ms"] - WINDOW_PAD_MS and not token_is_sound(t["text"])]
        aligned = slice_tokens_to_dialogue(candidate_tokens, win["dialogue_text"])
        if aligned:
            win["tokens"] = aligned
        else:
            win["tokens"] = [t for t in tokens if t["start_ms"] < win["end_ms"] and t["end_ms"] > win["start_ms"] and not token_is_sound(t["text"])]
        win["runs"] = build_runs_from_tokens(win["tokens"], win["dialogue_text"])

    leading_sound_events = build_leading_sound_events(dialogue_windows)
    dialogue_atoms = explode_windows_to_atoms(dialogue_windows)
    dialogue_atoms = merge_same_speaker_atoms(dialogue_atoms, protected)
    dialogue_atoms = pack_adjacent_two_speaker(dialogue_atoms)

    formatted_dialogue: List[Dict[str, Any]] = []
    for atom in dialogue_atoms:
        formatted_dialogue.extend(format_dialogue_atom(atom, protected))
    formatted_dialogue = merge_fragment_dialogue_cues(formatted_dialogue, protected)

    token_sound_events = build_sound_events_from_tokens(tokens)
    merged_sound = merge_sound_events(sound_events + leading_sound_events + token_sound_events)
    sound_cues = place_sound_events(merged_sound, formatted_dialogue)

    cues = formatted_dialogue + sound_cues
    cues = resolve_overlaps(cues)
    cues = final_qc_cleanup(cues, protected)
    cues.sort(key=lambda c: (c["start_ms"], c["end_ms"], 0 if c["type"] == "dialogue" else 1))
    for i, cue in enumerate(cues, 1):
        cue["idx"] = i

    srt_out = _export_srt(cues)
    vtt_out = _export_vtt(cues) if "vtt" in output_formats else None
    scc_out = _export_scc(cues) if "scc" in output_formats else None
    qc = _qc_report(len(backbone), cues, protected)
    return {"srt": srt_out, "vtt": vtt_out, "scc": scc_out, "qc": qc}


# ------------ text helpers ------------

def normalize_tokens(timestamps: Any) -> List[Dict[str, Any]]:
    raw = _normalize_tokens(timestamps)
    out = []
    for item in raw:
        text = str(item.get("text", "") or "").strip()
        if not text:
            continue
        start = int(item.get("start_ms", item.get("start", 0) or 0))
        end = int(item.get("end_ms", item.get("end", 0) or 0))
        if end <= start:
            end = start + 1
        out.append({"text": text, "start_ms": start, "end_ms": end, "speaker": str(item.get("speaker") or "A")})
    out.sort(key=lambda x: (x["start_ms"], x["end_ms"]))
    return out


def token_is_sound(text: str) -> bool:
    tag = (text or "").strip().upper()
    if tag in INLINE_DIALOGUE_TAGS:
        return False
    try:
        return bool(_is_sound_token(text)) and tag not in INLINE_DIALOGUE_TAGS
    except Exception:
        return tag.startswith("[") and tag.endswith("]") and tag not in INLINE_DIALOGUE_TAGS


def normalize_space(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    return text.strip()


def flatten_words(text: str) -> List[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text or "")]


def normalize_token_word(text: str) -> str:
    text = normalize_space(text)
    text = re.sub(r"^[^A-Za-z0-9\[]+", "", text)
    text = re.sub(r"[^A-Za-z0-9\]'.’+-]+$", "", text)
    return text


def token_match_word(text: str) -> str:
    text = normalize_token_word(text)
    parts = flatten_words(text)
    return parts[0] if parts else ""


def join_tokens(tokens: Sequence[Dict[str, Any]]) -> str:
    return normalize_space(" ".join(t["text"] for t in tokens))


def strip_non_dialogue_tags(text: str) -> str:
    def repl(match: re.Match) -> str:
        tag = match.group(0).upper()
        return tag if tag in INLINE_DIALOGUE_TAGS else " "
    return normalize_space(TAG_RE.sub(repl, text or ""))


def sanitize_last_word(text: str) -> str:
    words = flatten_words(text)
    return words[-1] if words else ""


def text_word_count(text: str) -> int:
    return len(flatten_words(text))


def is_ultra_fragment(text: str) -> bool:
    text = normalize_space(text)
    return text_word_count(text) <= 1 or len(text) <= 6


def repair_continuing_punctuation(text: str) -> str:
    text = normalize_space(text)
    if not text:
        return text
    # Preserve clear appositives / continuing phrases instead of inventing sentence stops.
    text = re.sub(r"\b([Ii]'?m your host)\. ([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+){0,2})\b", r"\1, \2", text)
    text = re.sub(r"\. (?=(?:right|okay|ok|speaking|please|because|which|who|where|when|while|though|that|if|but|and|or|so|then|now)\b)", ", ", text, flags=re.I)
    text = re.sub(r"\b(Yes|No)\. (?=(?:yes|no)\b)", r"\1, ", text)
    text = re.sub(r"\b([A-Z][A-Za-z]+)\. (?=(?:welcome|speaking|wearing|thank|please|because|and|but|or|so|then|now)\b)", r"\1, ", text)
    text = re.sub(r"\b([A-Z][A-Za-z]+), (?=(?:[a-z]))", r"\1, ", text)
    # If a period is followed by a lowercase word, it is usually a broken continuation.
    text = re.sub(r"(?<![A-Z])\. (?=[a-z])", ", ", text)
    text = re.sub(r",, +", ", ", text)
    return normalize_space(text)


def apply_asr_corrections(text: str) -> str:
    """
    Fix common ASR (speech-to-text) errors for broadcast captions.
    Only whole-word/phrase replacements that apply to any content; no show-specific rules.
    Preserves timing and word count.
    """
    if not text or not text.strip():
        return text
    text = normalize_space(text)
    # Universal mishears (any content)
    text = re.sub(r"\bdickhand\b", "deckhand", text, flags=re.I)
    # Duplicate word
    text = re.sub(r"\bdown\s+down\b", "down", text, flags=re.I)
    # Capitalization: comma then capitalized mid-sentence word
    text = re.sub(r",\s*But\s+", ", but ", text)
    text = re.sub(r",\s*So\s+", ", so ", text)
    text = re.sub(r",\s*Because\s+", ", because ", text)
    text = re.sub(r",\s*And\s+", ", and ", text)
    # Grammar: "in I would" → "and I would"
    text = re.sub(r"\s+in\s+I\s+would\s+", " and I would ", text, flags=re.I)
    return normalize_space(text)


# ------------ protected phrases ------------

def build_runtime_protected_phrases(backbone: List[Dict[str, Any]], tokens: List[Dict[str, Any]], explicit: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for phrase in explicit + detect_phrases_from_backbone(backbone) + detect_phrases_from_tokens(tokens):
        phrase = normalize_space(phrase)
        if len(phrase.split()) < 2:
            continue
        key = phrase.lower()
        if key not in seen:
            seen.add(key)
            out.append(phrase)
    return out


def detect_phrases_from_backbone(backbone: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for cue in backbone:
        text = strip_non_dialogue_tags(normalize_space(" ".join(cue.get("lines", []))))
        out.extend(extract_titleish_phrases(text.split()))
    return out


def detect_phrases_from_tokens(tokens: List[Dict[str, Any]]) -> List[str]:
    words = [t["text"] for t in tokens if not token_is_sound(t["text"])]
    return extract_titleish_phrases(words)


def extract_titleish_phrases(words: Sequence[str]) -> List[str]:
    cleaned = [normalize_token_word(w) for w in words]
    out: List[str] = []
    i = 0
    while i < len(cleaned):
        w = cleaned[i]
        if not w or not is_titleish(w):
            i += 1
            continue
        phrase = [w]
        j = i + 1
        while j < len(cleaned) and len(phrase) < 6:
            nxt = cleaned[j]
            if not nxt:
                break
            if is_titleish(nxt) or nxt.lower() in CONNECTORS:
                phrase.append(nxt)
                j += 1
            else:
                break
        if len(phrase) >= 2 and sum(1 for part in phrase if is_titleish(part)) >= 2:
            out.append(normalize_space(" ".join(phrase)))
        i = j if len(phrase) > 1 else i + 1
    return out


def is_titleish(word: str) -> bool:
    return bool(TITLEISH_RE.match(word or ""))


# ------------ sound and dialogue windows from backbone ------------

def build_windows_from_backbone(backbone: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dialogue: List[Dict[str, Any]] = []
    sound: List[Dict[str, Any]] = []
    for idx, cue in enumerate(backbone):
        raw_text = normalize_space(" ".join(cue.get("lines", [])))
        leading, dialogue_text, all_labels = split_sound_and_dialogue(raw_text)
        start_ms = int(cue["start_ms"])
        end_ms = int(cue["end_ms"])
        if dialogue_text:
            dialogue.append({
                "start_ms": start_ms,
                "end_ms": end_ms,
                "raw_text": raw_text,
                "dialogue_text": apply_asr_corrections(repair_continuing_punctuation(dialogue_text)),
                "leading_sound": dominant_sound_label(leading, prefer_first_reaction=True),
                "tokens": [],
                "runs": [],
            })
        else:
            label = dominant_sound_label(all_labels, prefer_first_reaction=True)
            if label:
                # extend a pure sound cue to the next cue start if the next cue begins with the same sound label.
                next_start = end_ms
                if idx + 1 < len(backbone):
                    next_raw = normalize_space(" ".join(backbone[idx + 1].get("lines", [])))
                    next_lead, next_dialogue, _ = split_sound_and_dialogue(next_raw)
                    next_label = dominant_sound_label(next_lead)
                    if next_dialogue and next_label == label:
                        next_start = int(backbone[idx + 1]["start_ms"])
                sound.append({"label": label, "start_ms": start_ms, "end_ms": max(end_ms, next_start)})
    return dialogue, sound


def split_sound_and_dialogue(raw_text: str) -> Tuple[List[str], str, List[str]]:
    raw_text = normalize_space(raw_text)
    if not raw_text:
        return [], "", []
    matches = list(TAG_RE.finditer(raw_text))
    leading: List[str] = []
    all_labels: List[str] = []
    pos = 0
    leading_phase = True
    for m in matches:
        between = raw_text[pos:m.start()].strip()
        if between:
            leading_phase = False
        label = normalize_sound_label(m.group(0))
        if label:
            all_labels.append(label)
            if leading_phase:
                leading.append(label)
        pos = m.end()
    dialogue = strip_non_dialogue_tags(raw_text)
    return leading, dialogue, all_labels


def normalize_sound_label(text: str) -> str:
    tag = normalize_space(text).upper()
    if not (tag.startswith("[") and tag.endswith("]")):
        return ""
    inner = tag[1:-1].strip().replace("♪", "MUSIC")
    parts = [p.strip() for p in re.split(r"\s+AND\s+|\s*,\s*|\s*/\s*", inner) if p.strip()]
    labels: List[str] = []
    for part in parts:
        if part in {"NOISE", "SOUND", "VOICE", "SPEAKER", "PAUSE", "TALKING", "CROSSTALK"}:
            continue
        if part.startswith("MUSIC") or part == "SONG":
            label = "[MUSIC]"
        elif part.startswith("APPLAUSE") or part in {"CLAPPING", "CLAPS"}:
            label = "[APPLAUSE]"
        elif part.startswith("LAUGHTER") or part in {"LAUGHS", "LAUGHING"}:
            label = "[LAUGHTER]"
        elif part.startswith("CHEER"):
            label = "[CHEERING]"
        else:
            label = ""
        if label and label not in labels:
            labels.append(label)
    return dominant_sound_label(labels)


def dominant_sound_label(labels: Sequence[str], prefer_first_reaction: bool = False) -> str:
    ordered = [label for label in labels if label in ALLOWED_SOUND]
    if not ordered:
        return ""
    if prefer_first_reaction:
        for label in ordered:
            if label in {"[APPLAUSE]", "[LAUGHTER]", "[CHEERING]"}:
                return label
    counts: Dict[str, int] = {}
    for label in ordered:
        counts[label] = counts.get(label, 0) + 1
    return sorted(counts.items(), key=lambda kv: (kv[1], SOUND_PRIORITY.get(kv[0], 0)), reverse=True)[0][0]


def build_leading_sound_events(windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for win in windows:
        label = win.get("leading_sound")
        if label not in ALLOWED_SOUND:
            continue
        toks = win.get("tokens") or []
        if toks:
            end_ms = toks[0]["start_ms"]
        else:
            end_ms = min(win["end_ms"], win["start_ms"] + TARGET_SOUND_MS)
        start_ms = win["start_ms"]
        if end_ms - start_ms < MIN_SOUND_MS:
            continue
        events.append({"label": label, "start_ms": start_ms, "end_ms": end_ms})
    return events


# ------------ dialogue window merging ------------

def merge_dialogue_windows(windows: List[Dict[str, Any]], protected: List[str]) -> List[Dict[str, Any]]:
    if not windows:
        return []
    merged = [windows[0]]
    for win in windows[1:]:
        prev = merged[-1]
        if should_merge_windows(prev, win, protected):
            prev["end_ms"] = win["end_ms"]
            prev["raw_text"] = normalize_space(f"{prev['raw_text']} {win['raw_text']}")
            prev["dialogue_text"] = apply_asr_corrections(repair_continuing_punctuation(normalize_space(f"{prev['dialogue_text']} {win['dialogue_text']}")))
        else:
            merged.append(win)
    return merged


def should_merge_windows(a: Dict[str, Any], b: Dict[str, Any], protected: List[str]) -> bool:
    gap = b["start_ms"] - a["end_ms"]
    if gap > 1100:
        return False
    combined = normalize_space(f"{a['dialogue_text']} {b['dialogue_text']}")
    if len(combined) <= 64 and gap <= MERGE_GAP_MS:
        return True
    if is_fragment(a["dialogue_text"]) or is_fragment(b["dialogue_text"]):
        return True
    if continues_sentence(a["dialogue_text"], b["dialogue_text"]):
        return True
    if boundary_splits_protected(a["dialogue_text"], b["dialogue_text"], protected):
        return True
    return False


def is_fragment(text: str) -> bool:
    text = normalize_space(text)
    words = text.split()
    if len(words) <= 2 or len(text) <= 14:
        return True
    if sanitize_last_word(text) in WEAK_ENDS:
        return True
    if words and words[0].lower() in WEAK_STARTS:
        return True
    return False


def continues_sentence(a: str, b: str) -> bool:
    a = normalize_space(a)
    b = normalize_space(b)
    if not a or not b:
        return False
    if re.search(r"[,:;]$", a):
        return True
    if not re.search(r"[.!?]$", a):
        return True
    if b.split() and b.split()[0][:1].islower():
        return True
    return False


def boundary_splits_protected(a: str, b: str, protected: List[str]) -> bool:
    aw = flatten_words(a)
    bw = flatten_words(b)
    for phrase in protected:
        pw = flatten_words(phrase)
        if len(pw) < 2:
            continue
        max_i = min(len(aw), len(pw) - 1)
        for i in range(1, max_i + 1):
            if aw[-i:] == pw[:i] and bw[:len(pw)-i] == pw[i:]:
                return True
    return False


# ------------ token alignment and speaker runs ------------

def slice_tokens_to_dialogue(candidate_tokens: List[Dict[str, Any]], dialogue_text: str) -> List[Dict[str, Any]]:
    target = flatten_words(dialogue_text)
    if not target or not candidate_tokens:
        return []
    cand = [token_match_word(t["text"]) for t in candidate_tokens]
    n = len(target)
    for i in range(0, len(cand) - n + 1):
        if cand[i:i+n] == target:
            return candidate_tokens[i:i+n]
    # fuzzy prefix fallback for near-exact windows
    for i in range(len(cand)):
        if cand[i] != target[0]:
            continue
        j = 0
        while i + j < len(cand) and j < n and cand[i + j] == target[j]:
            j += 1
        if j >= max(3, n - 1):
            return candidate_tokens[i:i+j]
    return []


def build_runs_from_tokens(tokens: List[Dict[str, Any]], fallback_text: str) -> List[Dict[str, Any]]:
    text = normalize_space(fallback_text)
    if not tokens:
        if not text:
            return []
        return [{"speaker": "A", "text": text, "start_ms": 0, "end_ms": 0, "tokens": []}]
    if len({(t.get("speaker") or "A") for t in tokens}) == 1:
        return [{
            "speaker": tokens[0].get("speaker") or "A",
            "text": apply_asr_corrections(repair_continuing_punctuation(text or join_tokens(tokens))),
            "start_ms": tokens[0]["start_ms"],
            "end_ms": tokens[-1]["end_ms"],
            "tokens": tokens,
        }]
    runs: List[Dict[str, Any]] = []
    current = [tokens[0]]
    for tok in tokens[1:]:
        prev = current[-1]
        if tok["speaker"] != prev["speaker"] or tok["start_ms"] - prev["end_ms"] > SAME_SPEAKER_HARD_GAP_MS:
            runs.append(run_from_tokens(current))
            current = [tok]
        else:
            current.append(tok)
    runs.append(run_from_tokens(current))
    return [r for r in runs if r["text"]]


def run_from_tokens(tokens: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "speaker": tokens[0]["speaker"] or "A",
        "text": apply_asr_corrections(repair_continuing_punctuation(join_tokens(tokens))),
        "start_ms": tokens[0]["start_ms"],
        "end_ms": tokens[-1]["end_ms"],
        "tokens": tokens,
    }


# ------------ dialogue atoms ------------

def explode_windows_to_atoms(windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    atoms: List[Dict[str, Any]] = []
    for win in windows:
        runs = win.get("runs") or [{"speaker": "A", "text": win["dialogue_text"], "start_ms": win["start_ms"], "end_ms": win["end_ms"], "tokens": win.get("tokens", [])}]
        if len(runs) == 1:
            run = dict(runs[0])
            run["start_ms"] = win["start_ms"]
            run["end_ms"] = win["end_ms"]
            atoms.append(make_atom([run], False, win))
            continue
        if len(runs) == 2 and can_two_speaker(runs):
            atoms.append(make_atom(runs, True, win))
            continue
        for run in runs:
            atoms.append(make_atom([run], False, win))
    return atoms


def can_two_speaker(runs: List[Dict[str, Any]]) -> bool:
    if len(runs) != 2:
        return False
    if runs[1]["start_ms"] - runs[0]["end_ms"] > TWO_SPEAKER_GAP_MS:
        return False
    if runs[1]["end_ms"] - runs[0]["start_ms"] > MAX_TWO_SPEAKER_WINDOW_MS:
        return False
    return all(len(normalize_space(r["text"])) <= MAX_CHARS - 2 and text_word_count(normalize_space(r["text"])) <= 8 for r in runs)


def make_atom(runs: List[Dict[str, Any]], two_speaker: bool, win: Dict[str, Any]) -> Dict[str, Any]:
    start = runs[0].get("start_ms") or win["start_ms"]
    end = runs[-1].get("end_ms") or win["end_ms"]
    return {
        "idx": 0,
        "start_ms": start,
        "end_ms": max(end, start + 1),
        "lines": [],
        "type": "dialogue",
        "meta": {"dialogue_text": repair_continuing_punctuation(normalize_space(" ".join(r["text"] for r in runs))), "runs": runs, "two_speaker": two_speaker},
    }


def merge_same_speaker_atoms(atoms: List[Dict[str, Any]], protected: List[str]) -> List[Dict[str, Any]]:
    if not atoms:
        return []
    merged = [atoms[0]]
    for atom in atoms[1:]:
        prev = merged[-1]
        if prev["meta"].get("two_speaker") or atom["meta"].get("two_speaker"):
            merged.append(atom)
            continue
        pruns = prev["meta"].get("runs", [])
        aruns = atom["meta"].get("runs", [])
        if len(pruns) != 1 or len(aruns) != 1 or pruns[0]["speaker"] != aruns[0]["speaker"]:
            merged.append(atom)
            continue
        gap = atom["start_ms"] - prev["end_ms"]
        combined = normalize_space(f"{pruns[0]['text']} {aruns[0]['text']}")
        if gap <= MERGE_GAP_MS and (continues_sentence(pruns[0]["text"], aruns[0]["text"]) or boundary_splits_protected(pruns[0]["text"], aruns[0]["text"], protected) or len(combined) <= 96):
            prev["end_ms"] = atom["end_ms"]
            pruns[0]["text"] = repair_continuing_punctuation(combined)
            pruns[0]["end_ms"] = atom["end_ms"]
            pruns[0]["tokens"] = pruns[0].get("tokens", []) + aruns[0].get("tokens", [])
            prev["meta"]["dialogue_text"] = repair_continuing_punctuation(combined)
        else:
            merged.append(atom)
    return merged


def pack_adjacent_two_speaker(atoms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    packed: List[Dict[str, Any]] = []
    i = 0
    while i < len(atoms):
        atom = atoms[i]
        if i + 1 < len(atoms):
            nxt = atoms[i + 1]
            ar = atom["meta"].get("runs", [])
            nr = nxt["meta"].get("runs", [])
            if not atom["meta"].get("two_speaker") and not nxt["meta"].get("two_speaker") and len(ar) == len(nr) == 1 and ar[0]["speaker"] != nr[0]["speaker"]:
                left = normalize_space(ar[0]["text"])
                right = normalize_space(nr[0]["text"])
                if len(left) <= MAX_CHARS - 2 and len(right) <= MAX_CHARS - 2 and text_word_count(left) <= 8 and text_word_count(right) <= 8 and nxt["start_ms"] - atom["end_ms"] <= TWO_SPEAKER_GAP_MS and nxt["end_ms"] - atom["start_ms"] <= MAX_TWO_SPEAKER_WINDOW_MS:
                    packed.append({
                        "idx": 0,
                        "start_ms": atom["start_ms"],
                        "end_ms": nxt["end_ms"],
                        "lines": [],
                        "type": "dialogue",
                        "meta": {"dialogue_text": repair_continuing_punctuation(normalize_space(f"{left} {right}")), "runs": ar + nr, "two_speaker": True},
                    })
                    i += 2
                    continue
        packed.append(atom)
        i += 1
    return packed


# ------------ formatting ------------

def format_dialogue_atom(atom: Dict[str, Any], protected: List[str]) -> List[Dict[str, Any]]:
    runs = atom["meta"].get("runs", [])
    if atom["meta"].get("two_speaker") and len(runs) == 2:
        left = normalize_space(runs[0]["text"])
        right = normalize_space(runs[1]["text"])
        if len(left) <= MAX_CHARS - 2 and len(right) <= MAX_CHARS - 2:
            out = atom.copy()
            out["lines"] = [f"- {left}", f"- {right}"]
            return [out]
        # split back into single-speaker atoms if it won't fit cleanly.
        return [format_dialogue_atom(make_atom([run], False, atom), protected)[0] for run in runs]

    if len(runs) == 1 and runs[0].get("tokens"):
        return segment_single_speaker_atom(atom, runs[0], protected)

    text = normalize_space(atom["meta"].get("dialogue_text", ""))
    if not text:
        return []
    out = atom.copy()
    out["lines"] = best_layout(text, protected)
    return [out]


def segment_single_speaker_atom(atom: Dict[str, Any], run: Dict[str, Any], protected: List[str]) -> List[Dict[str, Any]]:
    tokens = run.get("tokens") or []
    if not tokens:
        out = atom.copy()
        out["lines"] = best_layout(normalize_space(run["text"]), protected)
        return [out]
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(tokens):
        end = choose_chunk_end(tokens, i, protected)
        while end < len(tokens):
            chunk = tokens[i:end]
            text = apply_asr_corrections(repair_continuing_punctuation(join_tokens(chunk)))
            if (text_word_count(text) >= MIN_FRAGMENT_WORDS and len(text) >= MIN_FRAGMENT_CHARS) or re.search(r"[.!?]$", text):
                break
            end += 1
        chunk = tokens[i:end]
        text = apply_asr_corrections(repair_continuing_punctuation(join_tokens(chunk)))
        lines = best_layout(text, protected)
        start_ms = atom["start_ms"] if i == 0 else chunk[0]["start_ms"]
        end_ms = atom["end_ms"] if end >= len(tokens) else chunk[-1]["end_ms"]
        out.append({
            "idx": 0,
            "start_ms": start_ms,
            "end_ms": max(end_ms, start_ms + 1),
            "lines": lines,
            "type": "dialogue",
            "meta": {"dialogue_text": text, "runs": [{"speaker": run["speaker"], "text": text, "tokens": chunk, "start_ms": start_ms, "end_ms": end_ms}], "two_speaker": False},
        })
        i = end

    # Merge any leftover micro-fragments created by timing gaps.
    merged: List[Dict[str, Any]] = []
    for cue in out:
        text = cue["meta"]["dialogue_text"]
        if merged and cue["start_ms"] - merged[-1]["end_ms"] <= 900:
            prev = merged[-1]
            prev_text = prev["meta"]["dialogue_text"]
            combined = repair_continuing_punctuation(normalize_space(f"{prev_text} {text}"))
            if (is_ultra_fragment(text) or is_fragment(text) or is_fragment(prev_text)) and len(combined) <= MAX_CUE_CHARS and maybe_best_layout(combined, protected) is not None:
                prev["end_ms"] = cue["end_ms"]
                prev["meta"]["dialogue_text"] = combined
                prev["meta"]["runs"][0]["text"] = combined
                prev["meta"]["runs"][0]["tokens"] = prev["meta"]["runs"][0].get("tokens", []) + cue["meta"]["runs"][0].get("tokens", [])
                prev["lines"] = best_layout(combined, protected)
                continue
        merged.append(cue)
    return merged


def choose_chunk_end(tokens: List[Dict[str, Any]], start_idx: int, protected: List[str]) -> int:
    best_end = start_idx + 1
    best_score: Optional[float] = None
    max_end = min(len(tokens), start_idx + 28)
    for end in range(start_idx + 1, max_end + 1):
        chunk = tokens[start_idx:end]
        text = apply_asr_corrections(repair_continuing_punctuation(join_tokens(chunk)))
        if len(text) > MAX_CUE_CHARS + 18:
            break
        lines = maybe_best_layout(text, protected)
        if lines is None:
            continue
        next_word = token_match_word(tokens[end]["text"]).lower() if end < len(tokens) else ""
        score = chunk_score(chunk, text, lines, next_word, protected)
        if best_score is None or score < best_score:
            best_score = score
            best_end = end
        if score < 8 and re.search(r"[.!?]$", text):
            break
    return best_end


def chunk_score(chunk: List[Dict[str, Any]], text: str, lines: List[str], next_word: str, protected: List[str]) -> float:
    score = 0.0
    total_len = len(text)
    duration_ms = chunk[-1]["end_ms"] - chunk[0]["start_ms"]
    cps = total_len / max(duration_ms / 1000.0, 0.001)
    if len(lines) == 2:
        score += abs(len(lines[0]) - len(lines[1])) * 0.6
        if sanitize_last_word(lines[0]) in WEAK_ENDS:
            score += 18
        if lines[1].split() and lines[1].split()[0].lower() in WEAK_STARTS:
            score += 10
    else:
        if total_len > 24:
            score += 8
    if re.search(r"[.!?]$", text):
        score -= 18
    elif re.search(r"[,;:]$", text):
        score -= 10
    else:
        score += 5
    if sanitize_last_word(text) in WEAK_ENDS:
        score += 16
    if next_word in WEAK_STARTS:
        score += 10
    if cps > MAX_CPS:
        score += (cps - MAX_CPS) * 25
    elif cps > TARGET_CPS:
        score += (cps - TARGET_CPS) * 8
    if total_len < 8:
        score += 80
    elif total_len < 14:
        score += 42
    elif total_len < 20:
        score += 18
    elif total_len < 26 and not re.search(r"[.!?,;:]$", text):
        score += 14
    if len(chunk) < MIN_FRAGMENT_WORDS:
        score += 28
    elif len(chunk) < 5 and not re.search(r"[.!?,;:]$", text):
        score += 12
    if boundary_splits_protected(text, " ".join(next_word.split()), protected):
        score += 20
    return score


def maybe_best_layout(text: str, protected: List[str]) -> Optional[List[str]]:
    text = normalize_space(text)
    if not text:
        return None
    if len(text) <= MAX_CHARS:
        two = best_two_line_split(text, protected)
        if two and len(text) >= 24 and split_layout_score(two, protected) < 10:
            return two
        return [text]
    if len(text) > MAX_CUE_CHARS:
        return None
    return best_two_line_split(text, protected)


def best_layout(text: str, protected: List[str]) -> List[str]:
    layout = maybe_best_layout(text, protected)
    if layout is not None:
        return layout
    words = text.split()
    if not words:
        return []
    best = [words[0][:MAX_CHARS]]
    if len(words) > 1:
        remainder = normalize_space(" ".join(words[1:]))
        if remainder:
            best.append(remainder[:MAX_CHARS].rstrip())
    return best[:MAX_LINES]


def best_two_line_split(text: str, protected: List[str]) -> Optional[List[str]]:
    words = text.split()
    candidates: List[Tuple[float, List[str]]] = []
    for i in range(1, len(words)):
        left = normalize_space(" ".join(words[:i]))
        right = normalize_space(" ".join(words[i:]))
        if len(left) > MAX_CHARS or len(right) > MAX_CHARS:
            continue
        lines = [left, right]
        candidates.append((split_layout_score(lines, protected), lines))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def split_layout_score(lines: List[str], protected: List[str]) -> float:
    left, right = lines
    score = abs(len(left) - len(right)) * 0.7
    if re.search(r"[.!?]$", left):
        score -= 8
    elif re.search(r"[,;:]$", left):
        score -= 6
    else:
        score += 6
    if sanitize_last_word(left) in WEAK_ENDS:
        score += 18
    if right.split() and right.split()[0].lower() in WEAK_STARTS:
        score += 12
    if len(right.split()) == 1:
        score += 50
    if len(left.split()) == 1:
        score += 28
    if len(right) < 8:
        score += 12
    if len(left) < 8:
        score += 10
    if boundary_splits_protected(left, right, protected):
        score += 40
    return score


# ------------ sound placement ------------



def merge_fragment_dialogue_cues(cues: List[Dict[str, Any]], protected: List[str]) -> List[Dict[str, Any]]:
    if not cues:
        return []
    cues = sorted(cues, key=lambda c: (c["start_ms"], c["end_ms"]))

    def cue_text(c: Dict[str, Any]) -> str:
        return repair_continuing_punctuation(normalize_space(c.get("meta", {}).get("dialogue_text", " ".join(c.get("lines", [])))))

    def can_merge(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        if a["type"] != "dialogue" or b["type"] != "dialogue":
            return False
        if a.get("meta", {}).get("two_speaker") or b.get("meta", {}).get("two_speaker"):
            return False
        aruns = a.get("meta", {}).get("runs", [])
        bruns = b.get("meta", {}).get("runs", [])
        if len(aruns) != 1 or len(bruns) != 1 or aruns[0].get("speaker") != bruns[0].get("speaker"):
            return False
        at = cue_text(a)
        bt = cue_text(b)
        words_bt = bt.split()
        # Single-word cue that is allowed to stand alone (e.g. "Yes." "No.") — do not merge
        if len(words_bt) == 1 and sanitize_last_word(bt).lower() in ALLOWED_STANDALONE_WORDS:
            return False
        gap = b["start_ms"] - a["end_ms"]
        max_gap = max(MERGE_GAP_MS, 900)
        if len(words_bt) <= 2 and not re.search(r"[.!?]$", bt):
            max_gap = 1200  # Allow larger gap when merging short fragments
        if (b["end_ms"] - b["start_ms"]) < MIN_DIALOGUE_MS:
            max_gap = max(max_gap, 1200)  # Merge very short display-time cues
        if gap > max_gap:
            return False
        combined = repair_continuing_punctuation(normalize_space(f"{at} {bt}"))
        if len(combined) > MAX_CUE_CHARS:
            return False
        if maybe_best_layout(combined, protected) is None:
            return False
        if is_fragment(at) or is_fragment(bt):
            return True
        if continues_sentence(at, bt) or boundary_splits_protected(at, bt, protected):
            return True
        return False

    def merge_pair(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        arun = a["meta"]["runs"][0]
        brun = b["meta"]["runs"][0]
        text = repair_continuing_punctuation(normalize_space(f"{cue_text(a)} {cue_text(b)}"))
        tokens = list(arun.get("tokens", [])) + list(brun.get("tokens", []))
        return {
            "idx": 0,
            "start_ms": a["start_ms"],
            "end_ms": b["end_ms"],
            "lines": best_layout(text, protected),
            "type": "dialogue",
            "meta": {
                "dialogue_text": text,
                "runs": [{
                    "speaker": arun.get("speaker", "A"),
                    "text": text,
                    "tokens": tokens,
                    "start_ms": a["start_ms"],
                    "end_ms": b["end_ms"],
                }],
                "two_speaker": False,
            },
        }

    changed = True
    working = list(cues)
    while changed:
        changed = False
        out: List[Dict[str, Any]] = []
        i = 0
        while i < len(working):
            cur = working[i]
            if i + 1 < len(working) and can_merge(cur, working[i + 1]):
                out.append(merge_pair(cur, working[i + 1]))
                i += 2
                changed = True
                continue
            out.append(cur)
            i += 1
        working = out
    return working

def build_sound_events_from_tokens(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None
    for tok in tokens:
        if not token_is_sound(tok.get("text", "")):
            current = None
            continue
        label = normalize_sound_label(tok.get("text", ""))
        if label not in ALLOWED_SOUND:
            current = None
            continue
        start_ms = int(tok["start_ms"])
        end_ms = int(tok["end_ms"])
        if current and current["label"] == label and start_ms - current["end_ms"] <= SOUND_CLUSTER_GAP_MS:
            current["end_ms"] = end_ms
        else:
            current = {"label": label, "start_ms": start_ms, "end_ms": end_ms}
            events.append(current)
    return events


def merge_sound_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events = [e for e in events if e.get("label") in ALLOWED_SOUND]
    if not events:
        return []
    events.sort(key=lambda e: (e["start_ms"], e["end_ms"]))
    merged: List[Dict[str, Any]] = []
    cluster: List[Dict[str, Any]] = [events[0].copy()]

    def flush_cluster(cluster_events: List[Dict[str, Any]]) -> None:
        if not cluster_events:
            return
        if len(cluster_events) == 1:
            merged.append(cluster_events[0].copy())
            return
        labels = [e["label"] for e in cluster_events]
        first = labels[0]
        start = cluster_events[0]["start_ms"]
        end = max(e["end_ms"] for e in cluster_events)
        if first in {"[APPLAUSE]", "[LAUGHTER]", "[CHEERING]"}:
            music_events = [e for e in cluster_events if e["label"] == "[MUSIC]" and e["start_ms"] > cluster_events[0]["end_ms"]]
            if music_events:
                reaction_end = max(cluster_events[0]["end_ms"], min(end, start + 1200))
                merged.append({"label": first, "start_ms": start, "end_ms": reaction_end})
                music_start = music_events[0]["start_ms"]
                if end - music_start >= 250:
                    merged.append({"label": "[MUSIC]", "start_ms": music_start, "end_ms": end})
                return
            merged.append({"label": first, "start_ms": start, "end_ms": end})
            return
        counts: Dict[str, int] = {}
        for lab in labels:
            counts[lab] = counts.get(lab, 0) + 1
        label = sorted(counts.items(), key=lambda kv: (kv[1], SOUND_PRIORITY.get(kv[0], 0)), reverse=True)[0][0]
        merged.append({"label": label, "start_ms": start, "end_ms": end})

    for ev in events[1:]:
        prev = cluster[-1]
        if ev["start_ms"] - prev["end_ms"] <= SOUND_CLUSTER_GAP_MS:
            cluster.append(ev.copy())
        else:
            flush_cluster(cluster)
            cluster = [ev.copy()]
    flush_cluster(cluster)
    return merged


def place_sound_events(events: List[Dict[str, Any]], dialogue_cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not events:
        return []
    dialogue = sorted([c for c in dialogue_cues if c["type"] == "dialogue"], key=lambda c: (c["start_ms"], c["end_ms"]))
    placed: List[Dict[str, Any]] = []
    for ev in events:
        cue = place_sound_event(ev, dialogue)
        if cue:
            placed.append(cue)
    return placed


def place_sound_event(ev: Dict[str, Any], dialogue: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    prev = None
    nxt = None
    for cue in dialogue:
        if cue["end_ms"] <= ev["start_ms"]:
            prev = cue
            continue
        if cue["start_ms"] >= ev["start_ms"]:
            nxt = cue
            break
        if cue["start_ms"] < ev["end_ms"] and ev["start_ms"] < cue["end_ms"]:
            nxt = cue
            break
    gap_start = prev["end_ms"] if prev else ev["start_ms"]
    gap_end = nxt["start_ms"] if nxt else max(ev["end_ms"], ev["start_ms"] + TARGET_SOUND_MS)
    if gap_end - gap_start < MIN_SOUND_MS:
        return None
    # In long dead-air gaps, bridge earlier instead of pinning the cue right on top of dialogue.
    if ev["start_ms"] - gap_start >= 1000 and gap_end - gap_start >= LONG_GAP_BRIDGE_MS:
        start = gap_start + BRIDGE_PAD_MS
    else:
        start = max(gap_start, ev["start_ms"])
    lead_out = BRIDGE_PAD_MS if nxt else 0
    end_cap = gap_end - lead_out
    if end_cap <= start:
        return None
    desired_len = TARGET_SOUND_MS if ev["label"] == "[MUSIC]" else max(MIN_SOUND_MS, 1200)
    desired_end = min(end_cap, start + desired_len, max(ev["end_ms"], start + MIN_SOUND_MS))
    if desired_end - start < MIN_SOUND_MS:
        desired_end = min(end_cap, start + MIN_SOUND_MS)
    if desired_end - start < 500:
        return None
    return {"idx": 0, "start_ms": start, "end_ms": min(desired_end, start + MAX_SOUND_MS), "lines": [ev["label"]], "type": "sound", "meta": {"sound_label": ev["label"]}}


def repair_global_fragments(cues: List[Dict[str, Any]], protected: List[str]) -> List[Dict[str, Any]]:
    if not cues:
        return []
    cues = sorted(cues, key=lambda c: (c["start_ms"], c["end_ms"], 0 if c["type"] == "dialogue" else 1))

    def cue_text(c: Dict[str, Any]) -> str:
        return repair_continuing_punctuation(normalize_space(c.get("meta", {}).get("dialogue_text", " ".join(c.get("lines", [])))))

    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(cues):
        cur = cues[i]
        if cur["type"] != "dialogue":
            out.append(cur)
            i += 1
            continue
        cur_runs = cur.get("meta", {}).get("runs", [])
        cur_text = cue_text(cur)
        frag = is_ultra_fragment(cur_text) or is_fragment(cur_text)
        merged = False
        if frag and len(cur_runs) == 1:
            # try next first
            if i + 1 < len(cues):
                nxt = cues[i + 1]
                nxt_runs = nxt.get("meta", {}).get("runs", [])
                if nxt["type"] == "dialogue" and len(nxt_runs) == 1 and nxt_runs[0].get("speaker") == cur_runs[0].get("speaker") and nxt["start_ms"] - cur["end_ms"] <= 1000:
                    combined = repair_continuing_punctuation(normalize_space(f"{cur_text} {cue_text(nxt)}"))
                    if len(combined) <= MAX_CUE_CHARS and maybe_best_layout(combined, protected) is not None:
                        new = dict(cur)
                        new["end_ms"] = nxt["end_ms"]
                        new["meta"] = dict(cur.get("meta", {}))
                        new["meta"]["dialogue_text"] = combined
                        new["meta"]["runs"] = [{**cur_runs[0], "text": combined, "end_ms": nxt["end_ms"], "tokens": cur_runs[0].get("tokens", []) + nxt_runs[0].get("tokens", [])}]
                        new["lines"] = best_layout(combined, protected)
                        out.append(new)
                        i += 2
                        merged = True
            if not merged and out:
                prev = out[-1]
                prev_runs = prev.get("meta", {}).get("runs", []) if isinstance(prev.get("meta"), dict) else []
                if prev.get("type") == "dialogue" and len(prev_runs) == 1 and prev_runs[0].get("speaker") == cur_runs[0].get("speaker") and cur["start_ms"] - prev["end_ms"] <= 1000:
                    combined = repair_continuing_punctuation(normalize_space(f"{cue_text(prev)} {cur_text}"))
                    if len(combined) <= MAX_CUE_CHARS and maybe_best_layout(combined, protected) is not None:
                        prev["end_ms"] = cur["end_ms"]
                        prev["meta"]["dialogue_text"] = combined
                        prev["meta"]["runs"] = [{**prev_runs[0], "text": combined, "end_ms": cur["end_ms"], "tokens": prev_runs[0].get("tokens", []) + cur_runs[0].get("tokens", [])}]
                        prev["lines"] = best_layout(combined, protected)
                        i += 1
                        merged = True
        if not merged:
            out.append(cur)
            i += 1
    return out


# ------------ final cleanup ------------

def resolve_overlaps(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cues = sorted(cues, key=lambda c: (c["start_ms"], c["end_ms"], 0 if c["type"] == "dialogue" else 1))
    out: List[Dict[str, Any]] = []
    for cue in cues:
        cue = dict(cue)
        cue["lines"] = [normalize_space(line)[:MAX_CHARS] for line in cue.get("lines", []) if normalize_space(line)]
        if not cue["lines"]:
            continue
        if not out:
            out.append(cue)
            continue
        prev = out[-1]
        if cue["start_ms"] < prev["end_ms"]:
            if cue["type"] == "sound":
                cue["start_ms"] = prev["end_ms"]
            elif prev["type"] == "sound":
                prev["end_ms"] = min(prev["end_ms"], cue["start_ms"])
            else:
                cue["start_ms"] = prev["end_ms"]
        if cue["end_ms"] <= cue["start_ms"]:
            cue["end_ms"] = cue["start_ms"] + 1
        if cue["type"] == "sound" and cue["end_ms"] - cue["start_ms"] < 500:
            continue
        out.append(cue)
    return [c for c in out if c["end_ms"] > c["start_ms"]]


def final_qc_cleanup(cues: List[Dict[str, Any]], protected: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for cue in cues:
        if cue["type"] == "sound":
            label = normalize_sound_label(cue["lines"][0])
            if label:
                cue["lines"] = [label]
                out.append(cue)
            continue
        runs = cue.get("meta", {}).get("runs", [])
        if cue.get("meta", {}).get("two_speaker") and len(runs) == 2:
            left = repair_continuing_punctuation(normalize_space(runs[0]["text"]))
            right = repair_continuing_punctuation(normalize_space(runs[1]["text"]))
            if len(left) <= MAX_CHARS - 2 and len(right) <= MAX_CHARS - 2:
                cue["lines"] = [f"- {left}", f"- {right}"]
            else:
                out.extend(format_dialogue_atom(make_atom([runs[0]], False, cue), protected))
                out.extend(format_dialogue_atom(make_atom([runs[1]], False, cue), protected))
                continue
        else:
            text = repair_continuing_punctuation(normalize_space(cue.get("meta", {}).get("dialogue_text", " ".join(cue.get("lines", [])))))
            cue["meta"]["dialogue_text"] = text
            cue["lines"] = best_layout(text, protected)
        cue["lines"] = [repair_continuing_punctuation(normalize_space(x))[:MAX_CHARS] for x in cue.get("lines", []) if repair_continuing_punctuation(normalize_space(x))]
        if cue["lines"]:
            out.append(cue)

    # final anti-fragment pass on dialogue only
    dialogue = [c for c in out if c["type"] == "dialogue"]
    sounds = [c for c in out if c["type"] == "sound"]
    dialogue = merge_fragment_dialogue_cues(dialogue, protected)
    final = repair_global_fragments(dialogue + sounds, protected)
    final.sort(key=lambda c: (c["start_ms"], c["end_ms"], 0 if c["type"] == "dialogue" else 1))
    return final

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        with open(sys.argv[1]) as f:
            srt_text = f.read()
        with open(sys.argv[2]) as f:
            raw = json.load(f)
        result = process_caption_job(srt_text, raw, [])
        print(result["srt"])
