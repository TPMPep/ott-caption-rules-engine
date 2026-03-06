import re

SOUND_TOKEN_RE = re.compile(r"^\[[A-Z0-9 ,.'\-]+\]$")

def normalize_tokens(ts):
    """
    Accept either:
      - list[{"text","start","end","speaker"}]
      - {"words":[...]}
    Normalizes to: {"text","start_ms","end_ms","speaker"}
    """
    if isinstance(ts, dict) and "words" in ts:
        items = ts["words"]
    elif isinstance(ts, list):
        items = ts
    else:
        raise ValueError("Unsupported timestamps JSON format. Expect list or dict with 'words'.")

    out = []
    for w in items:
        text = (w.get("text") or w.get("word") or "").strip()
        start = int(w.get("start_ms", w.get("start", 0)))
        end = int(w.get("end_ms", w.get("end", start)))
        speaker = w.get("speaker")  # may be None
        out.append({"text": text, "start_ms": start, "end_ms": end, "speaker": speaker})
    return out

def is_sound_token(text: str) -> bool:
    return bool(SOUND_TOKEN_RE.match(text.strip()))
