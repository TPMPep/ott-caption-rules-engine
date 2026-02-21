from typing import Any, Dict, List
from services.formatter import Caption

def run_qc(caps: List[Caption], rules: Dict[str, Any]) -> Dict[str, Any]:
    max_lines = int(rules.get("maxLines", 2))
    max_chars = int(rules.get("maxCharsPerLine", 32))

    issues = []
    for i, c in enumerate(caps, start=1):
        if len(c.lines) > max_lines:
            issues.append({"cue": i, "type": "too_many_lines", "detail": f"{len(c.lines)} lines"})
        for ln in c.lines:
            if len(ln) > max_chars:
                issues.append({"cue": i, "type": "line_too_long", "detail": f"{len(ln)} chars: {ln}"})

        dur = c.end_ms - c.start_ms
        if dur < int(rules.get("minCaptionMs", 900)):
            issues.append({"cue": i, "type": "too_short", "detail": f"{dur}ms"})
        if dur > int(rules.get("maxCaptionMs", 6000)):
            issues.append({"cue": i, "type": "too_long", "detail": f"{dur}ms"})

    return {"issuesCount": len(issues), "issues": issues}
