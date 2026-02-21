from typing import List
from services.formatter import Caption

def _ms_to_tc_srt(ms: int) -> str:
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def _ms_to_tc_vtt(ms: int) -> str:
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"

def captions_to_srt(caps: List[Caption]) -> str:
    out = []
    for idx, c in enumerate(caps, start=1):
        out.append(str(idx))
        out.append(f"{_ms_to_tc_srt(c.start_ms)} --> {_ms_to_tc_srt(c.end_ms)}")
        out.extend(c.lines)
        out.append("")
    return "\n".join(out).strip() + "\n"

def captions_to_vtt(caps: List[Caption]) -> str:
    out = ["WEBVTT", ""]
    for c in caps:
        out.append(f"{_ms_to_tc_vtt(c.start_ms)} --> {_ms_to_tc_vtt(c.end_ms)}")
        out.extend(c.lines)
        out.append("")
    return "\n".join(out).strip() + "\n"

def vtt_to_scc_safe(vtt: str):
    # Optional: requires pycaption
    try:
        from pycaption import WebVTTReader, SCCWriter
        r = WebVTTReader().read(vtt)
        return SCCWriter().write(r)
    except Exception:
        return None
