import os
import json
from typing import List, Dict, Any

from services.assembly import normalize_tokens, is_sound_token
from services.exporters import parse_srt, export_srt, export_vtt, export_scc
from services.qc import qc_report, violates_line_limits, count_function_word_endings

MAX_LINES = 2
MAX_CHARS = 32
MIN_DIALOGUE_MS = 800
MIN_SOUND_MS = 800
SOUND_CLAMP_MS = 1200

def process_caption_job(backbone_srt_text: str, timestamps: Any, protected_phrases=None, output_formats=None):
    protected_phrases = protected_phrases or []
    output_formats = output_formats or ["srt"]

    backbone = parse_srt(backbone_srt_text)
    cues_in = len(backbone)

    tokens = normalize_tokens(timestamps)
    tokens.sort(key=lambda w: (w["start_ms"], w["end_ms"]))

    # 1) align tokens -> backbone cues, build speaker runs
    for c in backbone:
        c["meta"]["runs"] = build_speaker_runs_for_cue(c, tokens)

    # 2) extract sound events (don’t trust their durations)
    sound_cues = build_sound_cues(tokens)

    # 3) merge timeline + enforce non-overlap
    cues = merge_timeline(backbone, sound_cues)

    # 4) multi-speaker presplit / dash formatting rules
    cues = presplit_multispeaker(cues)

    # 5) AI formatting with required retry loop on splits
    cues = format_with_retry_loops(cues, protected_phrases)

    # 6) readability gate: min duration + merge micro-cues
    cues = readability_gate(cues)

    # 7) final overlap safety net
    cues = resolve_overlaps(cues)

    # export
    cues.sort(key=lambda c: (c["start_ms"], c["end_ms"]))
    for i, c in enumerate(cues, start=1):
        c["idx"] = i

    srt_out = export_srt(cues)
    vtt_out = export_vtt(cues) if "vtt" in output_formats else None
    scc_out = export_scc(cues) if "scc" in output_formats else None

    qc = qc_report(cues_in, cues, protected_phrases)

    return {"srt": srt_out, "vtt": vtt_out, "scc": scc_out, "qc": qc}

# ------------------------
# Speaker runs + sound cues
# ------------------------

def build_speaker_runs_for_cue(cue, tokens):
    runs = []
    cur = None
    for w in tokens:
        if w["end_ms"] < cue["start_ms"]:
            continue
        if w["start_ms"] > cue["end_ms"]:
            break
        t = w["text"].strip()
        if not t or is_sound_token(t):
            continue
        sp = w.get("speaker") or "A"
        if cur is None or cur["speaker"] != sp:
            cur = {"speaker": sp, "text_parts": []}
            runs.append(cur)
        cur["text_parts"].append(t)
    for r in runs:
        r["text"] = " ".join(r["text_parts"]).replace(" ,", ",").replace(" .", ".")
        del r["text_parts"]
    return runs

def build_sound_cues(tokens):
    out = []
    for w in tokens:
        t = w["text"].strip()
        if is_sound_token(t):
            start = int(w["start_ms"])
            out.append({
                "idx": 0,
                "start_ms": start,
                "end_ms": start + SOUND_CLAMP_MS,   # clamp so it never flashes
                "lines": [t],
                "type": "sound",
                "meta": {}
            })
    return out

def merge_timeline(dialogue, sound):
    combined = dialogue + sound
    combined.sort(key=lambda c: (c["start_ms"], c["end_ms"], 0 if c["type"] == "sound" else 1))
    return resolve_overlaps(combined)

# ------------------------
# Multi-speaker rules
# ------------------------

def presplit_multispeaker(cues):
    out = []
    for c in cues:
        if c["type"] == "sound":
            out.append(c)
            continue
        runs = c["meta"].get("runs", [])
        if len(runs) <= 1:
            out.append(c)
            continue

        # if exactly 2 speakers: force dash-per-line
        if len(runs) == 2:
            c["meta"]["two_speaker"] = True
            out.append(c)
            continue

        # more than 2 speaker runs: split into multiple cues (proportional)
        total = max(1, c["end_ms"] - c["start_ms"])
        lens = [max(1, len(r["text"])) for r in runs]
        denom = sum(lens)
        cur_start = c["start_ms"]
        for r, ln in zip(runs, lens):
            dur = int(total * (ln / denom))
            new_end = cur_start + max(1, dur)
            out.append({
                "idx": 0,
                "start_ms": cur_start,
                "end_ms": new_end,
                "lines": [],
                "type": "dialogue",
                "meta": {"runs": [r], "two_speaker": False}
            })
            cur_start = new_end
    return resolve_overlaps(out)

# ------------------------
# AI formatting (with retry loops)
# ------------------------

def format_with_retry_loops(cues, protected_phrases, max_rounds=3):
    formatted = []
    for c in cues:
        if c["type"] == "sound":
            c["lines"] = [c["lines"][0][:MAX_CHARS]]
            formatted.append(c)
            continue

        runs = c["meta"].get("runs", [])
        two = bool(c["meta"].get("two_speaker")) and len(runs) == 2

        if two:
            # one speaker per line with dash, each line max 30 chars (because "- ")
            lines = []
            for r in runs:
                one = linebreak_ai(r["text"], protected_phrases, max_lines=1, max_chars=MAX_CHARS - 2)
                lines.append(f"- {one[0][:MAX_CHARS - 2]}")
            c["lines"] = lines[:2]
            formatted.append(c)
            continue

        # single-speaker cue: AI linebreak
        text = " ".join([r["text"] for r in runs]).strip()
        lines = linebreak_ai(text, protected_phrases, max_lines=2, max_chars=MAX_CHARS)
        c["lines"] = lines[:2]

        # HARD validation: if fail, split and RE-RUN AI on children (this is the key missing piece)
        if violates_line_limits(c["lines"], MAX_LINES, MAX_CHARS):
            if max_rounds <= 0:
                c["lines"] = [l[:MAX_CHARS] for l in c["lines"][:MAX_LINES]]
                formatted.append(c)
            else:
                kids = split_cue(c)
                formatted.extend(format_with_retry_loops(kids, protected_phrases, max_rounds=max_rounds-1))
        else:
            # SOFT failure retry (function-word endings)
            if count_function_word_endings(c["lines"]) > 0 and max_rounds > 0:
                lines2 = linebreak_ai(text, protected_phrases, max_lines=2, max_chars=MAX_CHARS, retry_hint="Avoid function-word endings.")
                c["lines"] = lines2[:2]
            formatted.append(c)

    return resolve_overlaps(formatted)

def split_cue(cue):
    # split by words midpoint, timing proportional
    txt = " ".join(cue["lines"]).strip()
    if not txt:
        return [cue]
    words = txt.split()
    if len(words) < 2:
        return [cue]
    mid = len(words) // 2
    a = " ".join(words[:mid]).strip()
    b = " ".join(words[mid:]).strip()
    total = max(1, cue["end_ms"] - cue["start_ms"])
    cut = cue["start_ms"] + int(total * (len(a) / max(1, (len(a) + len(b)))))
    c1 = {**cue, "start_ms": cue["start_ms"], "end_ms": cut, "lines": [a], "meta": {"runs": [{"speaker":"A","text":a}]}}
    c2 = {**cue, "start_ms": cut, "end_ms": cue["end_ms"], "lines": [b], "meta": {"runs": [{"speaker":"A","text":b}]}}
    return [c1, c2]

def linebreak_ai(text, protected_phrases, max_lines=2, max_chars=32, retry_hint=None):
    """
    Uses OpenAI if OPENAI_API_KEY is set; otherwise falls back to a deterministic splitter.
    """
    text = (text or "").strip()
    if not text:
        return [""]

    if not os.getenv("OPENAI_API_KEY"):
        return heuristic_split(text, max_chars=max_chars, max_lines=max_lines)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        system = (
            "You are a broadcast closed-caption line breaker.\n"
            f"Rules: max {max_lines} lines, max {max_chars} chars per line.\n"
            "Never split protected phrases or named entities.\n"
            "Avoid ending a line with function words.\n"
            "Prefer punctuation and phrase boundaries.\n"
            "Return JSON only: {\"lines\":[\"...\",\"...\"],\"needs_split\":false}.\n"
            "If impossible: {\"lines\":[],\"needs_split\":true}.\n"
        )
        if retry_hint:
            system += f"\nExtra instruction: {retry_hint}\n"

        payload = {"text": text, "protected_phrases": protected_phrases}
        resp = client.responses.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            input=[
                {"role":"system","content":system},
                {"role":"user","content":json.dumps(payload)}
            ],
            temperature=0.2
        )
        out = json.loads(resp.output_text.strip())
        lines = out.get("lines") or []
        if not lines:
            return heuristic_split(text, max_chars=max_chars, max_lines=max_lines)
        return [l[:max_chars] for l in lines[:max_lines]]
    except Exception:
        return heuristic_split(text, max_chars=max_chars, max_lines=max_lines)

def heuristic_split(text, max_chars=32, max_lines=2):
    words = text.split()
    if len(text) <= max_chars:
        return [text]
    best = None
    for i in range(1, len(words)):
        a = " ".join(words[:i]); b = " ".join(words[i:])
        if len(a) <= max_chars and len(b) <= max_chars:
            score = abs(len(a)-len(b))
            best = (score, a, b) if best is None or score < best[0] else best
    if best and max_lines == 2:
        return [best[1], best[2]]
    # fallback hard wrap
    return [text[:max_chars], text[max_chars:max_chars*2]]

# ------------------------
# Readability gate + overlap resolver
# ------------------------

def readability_gate(cues):
    cues.sort(key=lambda c: (c["start_ms"], c["end_ms"]))
    merged = []
    i = 0
    while i < len(cues):
        c = cues[i]
        dur = c["end_ms"] - c["start_ms"]
        min_ms = MIN_SOUND_MS if c["type"] == "sound" else MIN_DIALOGUE_MS

        # enforce min duration if there is room before next cue
        if dur < min_ms and i < len(cues)-1:
            gap = cues[i+1]["start_ms"] - c["end_ms"]
            need = min_ms - dur
            if gap >= need:
                c["end_ms"] += need

        # merge micro dialogue cues (<=2 words and very short)
        if c["type"] == "dialogue" and dur < 1000:
            wc = len(" ".join(c["lines"]).split())
            if wc <= 2 and i < len(cues)-1 and cues[i+1]["type"] == "dialogue":
                nxt = cues[i+1]
                if nxt["start_ms"] - c["end_ms"] <= 200:
                    c["lines"] = [(" ".join(c["lines"]) + " " + " ".join(nxt["lines"])).strip()]
                    c["end_ms"] = nxt["end_ms"]
                    i += 2
                    merged.append(c)
                    continue

        # merge consecutive sound cues
        if c["type"] == "sound" and i < len(cues)-1 and cues[i+1]["type"] == "sound":
            nxt = cues[i+1]
            a = " ".join(c["lines"]).strip()[1:-1]
            b = " ".join(nxt["lines"]).strip()[1:-1]
            c["lines"] = [f"[{a} AND {b}]".replace("  ", " ").strip()]
            c["end_ms"] = max(c["end_ms"], nxt["end_ms"])
            i += 2
            merged.append(c)
            continue

        merged.append(c)
        i += 1

    return merged

def resolve_overlaps(cues):
    cues.sort(key=lambda c: (c["start_ms"], c["end_ms"]))
    for i in range(len(cues)-1):
        a = cues[i]; b = cues[i+1]
        if a["end_ms"] > b["start_ms"]:
            shift = a["end_ms"] - b["start_ms"]
            b["start_ms"] += shift
            if b["end_ms"] <= b["start_ms"]:
                b["end_ms"] = b["start_ms"] + 1
    return cues
