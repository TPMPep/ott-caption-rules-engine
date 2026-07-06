"""
Editorial AI — GPT-powered caption refinement.

OPTIONAL polish pass that runs AFTER the rules-engine formatter has done
its work. Improves punctuation, capitalization, phrase-aware line breaks,
and speaker formatting WITHOUT changing the actual words. If the AI
returns a rewrite that changes the underlying words (ignoring case +
punctuation), the original is kept — auditor-defensible word fidelity.

Skipped entirely if OPENAI_API_KEY is unset.

╔══ Resilience contract (2026-05-28) ════════════════════════════════════╗
║ This pass is the #1 source of formatter hangs in production. Root      ║
║ cause: OpenAI's responses.create() has NO default timeout, and we call ║
║ it once per cue serially. A single stalled connection (OpenAI's        ║
║ responses endpoint occasionally just stops streaming — observed twice  ║
║ on the same media) blocks the formatter thread forever, leaving the    ║
║ engine job at stage='formatting' until Base44's 15-min stale guard     ║
║ kills it. Auditor-grade defenses applied here:                         ║
║                                                                        ║
║   1. PER-CALL TIMEOUT — 30s on every OpenAI call. The SDK raises       ║
║      APITimeoutError → the cue falls through to the "original kept"    ║
║      branch → next cue starts immediately. One slow call ≠ stuck run.  ║
║   2. RUN BUDGET — total wall-clock cap on the AI pass (default 120s).  ║
║      Once exceeded, remaining cues are appended as-is. The rules       ║
║      engine already produced spec-correct output; AI is polish only.   ║
║   3. ERROR-RATE BAILOUT — if >20% of attempted cues errored, the AI    ║
║      provider is having a bad day → abort and use rules-engine output. ║
║   4. HEARTBEAT HOOK — every N cues, invoke an optional progress        ║
║      callback so the engine can bump JOBS[job_id].updated_at, which    ║
║      lets Base44's poller distinguish "still working" from "hung".     ║
║                                                                        ║
║ SOC 2 CC8.1 — graceful degradation never silently changes output;      ║
║ every fallback path is the original rules-engine cue, byte-identical.  ║
║ The audit row records cues_ai_applied vs cues_ai_skipped explicitly.   ║
╚════════════════════════════════════════════════════════════════════════╝
"""

import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

_WORD_RE = re.compile(r"\b[\w']+\b")
_STYLE_TAG_RE = re.compile(r"\{\\+an\d\}")
_ITALIC_TAG_RE = re.compile(r"</?i>")
# Strips a leading speaker label the AI may have emitted so we can re-apply the
# deterministic one. Matches '- ', '[NAME:] ', '[SPEAKER A:] ', 'A: ', 'NAME: '.
_LEADING_LABEL_RE = re.compile(r"^\s*(?:-\s+|\[[^\]]*\]:?\s*|[^\s:]{1,24}:\s+)")

# SINGLE SOURCE OF TRUTH for the speaker label/dash. The AI polishes TEXT only;
# the label is re-applied deterministically by render_lines so the AI can never
# strip or mangle it (the "no dash / no bracket on a different speaker" defect).
try:
    from .rendering import render_lines as _render_lines
except Exception:
    _render_lines = None


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _max_lines() -> int:
    return _env_int("CUSTOM_MAX_LINES", 2)


def _max_chars() -> int:
    return _env_int("CUSTOM_MAX_CHARS", 32)


def _speaker_label_mode() -> str:
    return (os.getenv("SPEAKER_LABEL_MODE", "") or "dash").strip().lower()


def _build_system_prompt(max_lines: int, max_chars: int, speaker_mode: str) -> str:
    parts = [
        "You are a broadcast closed-caption editorial assistant. "
        "Output must be suitable for any media (TV, streaming).",
        "Do not add, remove, replace, or reorder words.",
        "You may only change capitalization, punctuation, and line breaks.",
        "Capitalization: capitalize only the first word of a true sentence and proper nouns "
        "(names, titles, I). Do not capitalize a word just because it follows a comma or "
        "starts a new caption line.",
        "Punctuation (critical): use commas where the sentence or thought continues; "
        "use periods only at a real sentence stop. "
        "Only change a period to a comma when the next dialogue clearly continues the same "
        "thought (e.g. next starts with a lowercase continuation word).",
        "If this caption starts with a word that continues prev_dialogue "
        "(e.g. it's, well, and, but, so, then, where, really), output that word lowercased.",
        "When splitting into two lines, avoid a single word on the second line unless it is "
        "a brief response (Yes, No, OK, Yeah, Right).",
        "Prefer splitting at phrase or clause boundaries.",
        "Do not split protected phrases across lines.",
        "Avoid ending a line with weak function words "
        "(a, an, the, of, to, and, or, but, with, from, in, on, at, for, that) unless unavoidable.",
    ]

    if speaker_mode == "dash":
        parts.append("If there are exactly two speaker runs, you MUST output exactly two lines "
                     "and begin each line with '- ' (dash space).")
    elif speaker_mode == "alpha":
        parts.append("If there are multiple speaker runs, prefix each speaker's line with their "
                     "letter label followed by a colon (e.g. 'A: Hello' / 'B: Hi there').")
    elif speaker_mode == "generic":
        gp = os.getenv("SPEAKER_GENERIC_PREFIX", "SPEAKER") or "SPEAKER"
        parts.append(f"If there are multiple speaker runs, prefix each speaker's line with "
                     f"'{gp} N:' where N is the speaker number.")
    elif speaker_mode == "named":
        parts.append("If there are multiple speaker runs, prefix each speaker's line with "
                     "the speaker's name followed by a colon.")

    parts.append(f"If text fits in one line (≤{max_chars} characters), output one line only; "
                 f"{max_lines} lines is the max, not required.")
    parts.append('Return JSON only with the shape {"lines":["...","..."]}.')

    return " ".join(parts)


def editorial_refine_cues(
    cues: List[Dict[str, Any]],
    protected_phrases: List[str],
    heartbeat: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Any]]:
    """Apply the optional editorial-AI polish pass to dialogue cues.

    Args:
        cues: Rules-engine output cues. Non-dialogue cues pass through.
        protected_phrases: Phrases the AI is forbidden from splitting.
        heartbeat: Optional callable `(idx, total) -> None` invoked every
            HEARTBEAT_EVERY cues so the caller (main.py) can bump the job's
            `updated_at` timestamp. Lets Base44's poller distinguish a
            slow-but-progressing run from a hung formatter. SOC 2 CC8.1 —
            engine state must be observable in real time.

    Returns:
        List of cues. AI-improved where successful, original where not.
        Length always equals len(cues) — no cues are dropped.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return cues
    try:
        from openai import OpenAI, APITimeoutError, APIError
        # Per-call timeout pinned at the client level. The OpenAI SDK
        # raises APITimeoutError after this wall-clock budget — the cue
        # then falls through to "original kept" instead of blocking the
        # entire formatter thread forever (the production hang root cause).
        per_call_timeout = _env_int("OPENAI_REQUEST_TIMEOUT_SECONDS", 30)
        client = OpenAI(api_key=api_key, timeout=float(per_call_timeout))
    except Exception:
        return cues

    # Resilience budgets — all env-tunable so we can ratchet them under
    # real-world load without a redeploy.
    run_budget_seconds = _env_int("EDITORIAL_AI_RUN_BUDGET_SECONDS", 120)
    error_rate_bailout_pct = _env_int("EDITORIAL_AI_ERROR_RATE_BAILOUT_PCT", 20)
    min_attempts_before_bailout = _env_int("EDITORIAL_AI_MIN_ATTEMPTS_BEFORE_BAILOUT", 25)
    heartbeat_every = _env_int("EDITORIAL_AI_HEARTBEAT_EVERY", 25)

    max_lines = _max_lines()
    max_chars = _max_chars()
    speaker_mode = _speaker_label_mode()
    system_prompt = _build_system_prompt(max_lines, max_chars, speaker_mode)

    # ── Parallel fan-out (engine 5.22.0) ─────────────────────────────────
    # The polish pass was previously ONE OpenAI call per cue, SERIALLY —
    # ~1.5s × N cues, so the 120s run budget covered only ~80 cues of a
    # 1000-cue feature and the rest shipped unpolished. Every cue's payload
    # reads only the ORIGINAL neighbor texts (never another cue's polished
    # result), so the calls are fully independent and now fan out through a
    # bounded thread pool. Same budgets, same fidelity guards, deterministic
    # assembly by index — just N-way concurrent. SOC 2 CC8.1: outcome per
    # cue is identical to the serial pass; only wall-clock changes.
    concurrency = _env_int("EDITORIAL_AI_CONCURRENCY", 8)
    total = len(cues)
    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    stats = {
        "dialogue_attempted": 0,
        "ai_applied": 0,
        "ai_rejected_word_drift": 0,
        "ai_rejected_overflow": 0,
        "ai_error_timeout": 0,
        "ai_error_other": 0,
        "skipped_run_budget": 0,
        "skipped_error_rate_bailout": 0,
    }

    # Build every cue's payload up front (serial, cheap, reads originals only).
    jobs = []  # (idx, cue, dialogue_text, runs, payload)
    for idx, cue in enumerate(cues):
        if cue.get("type") != "dialogue":
            continue
        runs = cue.get("meta", {}).get("runs", [])
        dialogue_text = cue.get("meta", {}).get(
            "dialogue_text", " ".join(cue.get("lines", []))
        ).strip()
        if not dialogue_text:
            continue
        prev_text = ""
        next_text = ""
        if idx > 0 and cues[idx - 1].get("type") == "dialogue":
            prev_text = cues[idx - 1].get("meta", {}).get(
                "dialogue_text", " ".join(cues[idx - 1].get("lines", []))
            ).strip()
        if idx < total - 1 and cues[idx + 1].get("type") == "dialogue":
            next_text = cues[idx + 1].get("meta", {}).get(
                "dialogue_text", " ".join(cues[idx + 1].get("lines", []))
            ).strip()
        payload = {
            "dialogue_text": dialogue_text,
            "current_lines": cue.get("lines", []),
            "speaker_runs": runs,
            "prev_dialogue": prev_text,
            "next_dialogue": next_text,
            "protected_phrases": (protected_phrases or [])[:50],
            "rules": {
                "max_lines": max_lines,
                "max_chars_per_line": max_chars,
                "speaker_mode": speaker_mode,
                "two_speaker_lines_must_start_with_dash": speaker_mode == "dash",
                "preserve_words_exactly": True,
                "fix_punctuation_and_capitalization_only": True,
                "prefer_phrase_and_punctuation_boundaries": True,
                "avoid_weak_function_word_line_endings": True,
            },
        }
        jobs.append((idx, cue, dialogue_text, runs, payload))

    def _polish_one(idx, cue, dialogue_text, runs, payload):
        """One cue's full polish → (idx, outcome, new_cue|None). Pure over its
        inputs; validation + deterministic re-render happen in-worker so the
        harvest loop only aggregates. Thread-safe (OpenAI client is httpx-based)."""
        try:
            response = client.responses.create(
                model=model_name,
                temperature=0,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
            )
            raw = response.output_text.strip()
            data = json.loads(raw)
            ai_lines = [str(x).strip() for x in data.get("lines", []) if str(x).strip()]
        except APITimeoutError:
            return idx, "timeout", None
        except Exception:
            return idx, "error", None

        ai_lines = _normalize_lines(ai_lines)
        if not ai_lines:
            return idx, "noop", None
        # Strip ANY speaker label the AI emitted before the fidelity check —
        # the label is NOT dialogue; the deterministic renderer re-applies it.
        ai_text_lines = [_strip_leading_label(l) for l in ai_lines]
        if _word_fingerprint(dialogue_text) != _word_fingerprint(" ".join(ai_text_lines)):
            return idx, "word_drift", None
        # Re-apply the speaker label/dash DETERMINISTICALLY via render_lines
        # (single source of truth) — the AI can never drop or mangle it.
        if _render_lines is not None:
            polished_text = " ".join(ai_text_lines).strip()
            final_lines = _render_lines(
                polished_text.split(), runs, max_lines, max_chars, polished_text,
            )
        else:
            final_lines = ai_lines
        if len(final_lines) > max_lines or any(_visible_len(l) > max_chars for l in final_lines):
            return idx, "overflow", None
        new_cue = dict(cue)
        new_cue["lines"] = final_lines
        # KEEP META IN SYNC (2026-07-06): the AI's casing/punctuation fixes must
        # land on meta.dialogue_text too, not just the rendered lines. Leaving
        # dialogue_text stale was the 'This place' defect: the AI capitalized
        # the LINE, downstream deterministic passes (sentence capitalization,
        # condensation) read dialogue_text, saw lowercase 'this', concluded
        # nothing to fix — and the delivered line shipped the wrong capital.
        # Lines and dialogue_text must never diverge. SOC 2 CC8.1.
        new_meta = dict(cue.get("meta") or {})
        new_meta["dialogue_text"] = " ".join(ai_text_lines).strip()
        new_cue["meta"] = new_meta
        return idx, "applied", new_cue

    results: Dict[int, Dict[str, Any]] = {}
    bailed_out = False
    start_time = time.monotonic()
    completed = 0

    if jobs:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_polish_one, *job): job[0] for job in jobs}
            for fut in as_completed(futures):
                completed += 1
                if heartbeat and completed % heartbeat_every == 0:
                    try:
                        heartbeat(completed, len(jobs))
                    except Exception:
                        pass  # heartbeat is best-effort; never block
                stats["dialogue_attempted"] += 1
                try:
                    idx, outcome, new_cue = fut.result()
                except Exception:
                    idx, outcome, new_cue = -1, "error", None
                if outcome == "applied" and new_cue is not None:
                    results[idx] = new_cue
                    stats["ai_applied"] += 1
                elif outcome == "timeout":
                    stats["ai_error_timeout"] += 1
                elif outcome == "error":
                    stats["ai_error_other"] += 1
                elif outcome == "word_drift":
                    stats["ai_rejected_word_drift"] += 1
                elif outcome == "overflow":
                    stats["ai_rejected_overflow"] += 1

                # Error-rate bailout — unchanged semantics, parallel-safe:
                # if the provider is having a bad day, cancel what hasn't
                # started and ship rules-engine output for the remainder.
                errored = stats["ai_error_timeout"] + stats["ai_error_other"]
                if completed >= min_attempts_before_bailout and \
                        (errored * 100) // max(1, completed) >= error_rate_bailout_pct:
                    stats["skipped_error_rate_bailout"] = len(jobs) - completed
                    bailed_out = True
                    pool.shutdown(wait=False, cancel_futures=True)
                    break
                # Run wall-clock budget — unchanged.
                if time.monotonic() - start_time > run_budget_seconds:
                    stats["skipped_run_budget"] = len(jobs) - completed
                    bailed_out = True
                    pool.shutdown(wait=False, cancel_futures=True)
                    break

    # Deterministic assembly: AI-improved where accepted, original otherwise.
    # Length always equals len(cues) — no cue is ever dropped.
    refined = [results.get(i, cue) for i, cue in enumerate(cues)]

    elapsed_s = round(time.monotonic() - start_time, 1)
    print(
        f"[EDITORIAL_AI] elapsed={elapsed_s}s concurrency={concurrency} "
        f"applied={stats['ai_applied']}/{stats['dialogue_attempted']} "
        f"timeouts={stats['ai_error_timeout']} other_errors={stats['ai_error_other']} "
        f"bailed_out={bailed_out} "
        f"skipped_budget={stats['skipped_run_budget']} "
        f"skipped_error_rate={stats['skipped_error_rate_bailout']}"
    )

    return refined


def _word_fingerprint(text: str) -> List[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def _strip_dashes(lines: List[str]) -> List[str]:
    out: List[str] = []
    for line in lines:
        if line.startswith("- "):
            out.append(line[2:])
        else:
            out.append(line)
    return out


def _strip_leading_label(line: str) -> str:
    """Remove a leading speaker label the AI may have emitted so the spoken
    text can be fidelity-checked and re-rendered with the deterministic label.
    Handles '- ', '[NAME:] ', '[SPEAKER A:] ', 'A: ', 'NAME: '. Idempotent on
    lines that carry no label."""
    return _LEADING_LABEL_RE.sub("", line or "", count=1).strip()


def _normalize_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    for line in lines:
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            out.append(line)
    return out[:_max_lines()]


def _visible_len(text: str) -> int:
    text = _STYLE_TAG_RE.sub("", text or "")
    text = _ITALIC_TAG_RE.sub("", text or "")
    return len(text)
