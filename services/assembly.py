import re
from typing import Any, Dict, List

SOUND_TOKEN_RE = re.compile(r"^\[[A-Z0-9 ,.'\-]+\]$")


def normalize_tokens(ts: Any) -> List[Dict[str, Any]]:
    """
    Accept either:
      - list[{"text","start","end","speaker"}]
      - {"words":[...]}
    Normalize to:
      {"text","start_ms","end_ms","speaker"}
    """
    if isinstance(ts, dict) and "words" in ts:
        items = ts["words"]
    elif isinstance(ts, list):
        items = ts
    else:
        raise ValueError("Unsupported timestamps JSON format. Expect list or dict with 'words'.")

    out: List[Dict[str, Any]] = []

    for item in items:
        text = (item.get("text") or item.get("word") or "").strip()
        start = int(item.get("start_ms", item.get("start", 0)))
        end = int(item.get("end_ms", item.get("end", start)))
        speaker = item.get("speaker")

        out.append(
            {
                "text": text,
                "start_ms": start,
                "end_ms": end,
                "speaker": speaker,
            }
        )

    return out


def is_sound_token(text: str) -> bool:
    return bool(SOUND_TOKEN_RE.match((text or "").strip()))
