import re
from typing import Any, Dict, List

TIME_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")


def ms_to_tc(ms: int) -> str:
    if ms < 0:
        ms = 0
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def ms_to_vtt(ms: int) -> str:
    return ms_to_tc(ms).replace(",", ".")


def tc_to_ms(tc: str) -> int:
    match = TIME_RE.match(tc.strip())
    if not match:
        raise ValueError(f"Invalid SRT timecode: {tc}")
    h, m, s, ms = map(int, match.groups())
    return ((h * 60 + m) * 60 + s) * 1000 + ms


def parse_srt(srt_text: str) -> List[Dict[str, Any]]:
    """
    Parse SRT into a normalized cue list.
    """
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    cues: List[Dict[str, Any]] = []

    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        try:
            idx = int(lines[0])
            timing = lines[1]
            text_lines = lines[2:]
        except ValueError:
            idx = len(cues) + 1
            timing = lines[0]
            text_lines = lines[1:]

        if "-->" not in timing:
            continue

        start_tc, end_tc = [part.strip() for part in timing.split("-->")]
        start_ms = tc_to_ms(start_tc)
        end_ms = tc_to_ms(end_tc)

        cues.append(
            {
                "idx": idx,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "lines": text_lines,
                "type": "dialogue",
                "meta": {},
            }
        )

    for i, cue in enumerate(cues, start=1):
        cue["idx"] = i

    return cues


def export_srt(cues: List[Dict[str, Any]]) -> str:
    out: List[str] = []

    for i, cue in enumerate(cues, start=1):
        out.append(str(i))
        out.append(f"{ms_to_tc(cue['start_ms'])} --> {ms_to_tc(cue['end_ms'])}")
        out.extend(cue["lines"] if cue["lines"] else [""])
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def export_vtt(cues: List[Dict[str, Any]]) -> str:
    out: List[str] = ["WEBVTT", ""]

    for cue in cues:
        out.append(f"{ms_to_vtt(cue['start_ms'])} --> {ms_to_vtt(cue['end_ms'])}")
        out.extend(cue["lines"] if cue["lines"] else [""])
        out.append("")

    return "\n".join(out).rstrip() + "\n"


def export_scc(cues: List[Dict[str, Any]]) -> str:
    """
    Placeholder SCC exporter.
    Keep your existing SCC logic if you already have a real one.
    """
    out: List[str] = ["Scenarist_SCC V1.0", ""]

    for cue in cues:
        out.append(f"{cue['start_ms']}:{cue['end_ms']}  {' '.join(cue['lines'])}")

    return "\n".join(out).rstrip() + "\n"
