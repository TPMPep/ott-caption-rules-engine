import re

TIME_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")

def ms_to_tc(ms: int) -> str:
    if ms < 0: ms = 0
    h = ms // 3600000; ms %= 3600000
    m = ms // 60000;   ms %= 60000
    s = ms // 1000;    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def ms_to_vtt(ms: int) -> str:
    return ms_to_tc(ms).replace(",", ".")

def parse_srt(srt_text: str):
    blocks = re.split(r"\n\s*\n", srt_text.strip())
    cues = []
    for b in blocks:
        lines = [l.rstrip() for l in b.splitlines() if l.strip()]
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
        a, b2 = [x.strip() for x in timing.split("-->")]
        sh, sm, ss, sms = map(int, TIME_RE.match(a).groups())
        eh, em, es, ems = map(int, TIME_RE.match(b2).groups())
        start_ms = ((sh*60+sm)*60+ss)*1000 + sms
        end_ms   = ((eh*60+em)*60+es)*1000 + ems

        cues.append({
            "idx": idx,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "lines": text_lines,
            "type": "dialogue",
            "meta": {}
        })
    # normalize idx
    for i, c in enumerate(cues, start=1):
        c["idx"] = i
    return cues

def export_srt(cues):
    out = []
    for i, c in enumerate(cues, start=1):
        out.append(str(i))
        out.append(f"{ms_to_tc(c['start_ms'])} --> {ms_to_tc(c['end_ms'])}")
        out.extend(c["lines"] if c["lines"] else [""])
        out.append("")
    return "\n".join(out).rstrip() + "\n"

def export_vtt(cues):
    out = ["WEBVTT", ""]
    for c in cues:
        out.append(f"{ms_to_vtt(c['start_ms'])} --> {ms_to_vtt(c['end_ms'])}")
        out.extend(c["lines"] if c["lines"] else [""])
        out.append("")
    return "\n".join(out).rstrip() + "\n"

def export_scc(cues):
    # SCC is frame/timecode + control codes; keep your existing SCC exporter if you have one.
    # This placeholder is for debugging only.
    out = ["Scenarist_SCC V1.0", ""]
    for c in cues:
        out.append(f"{c['start_ms']}:{c['end_ms']}  {' '.join(c['lines'])}")
    return "\n".join(out).rstrip() + "\n"
