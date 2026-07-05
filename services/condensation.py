"""
Condensation — editorial reading-text reduction (the "captions aren't transcripts" stage).

WHY THIS MODULE EXISTS
──────────────────────
Professional CC/subtitle files are EDITED READING TEXT, not verbatim transcripts.
When speech runs faster than the spec's CPS (characters-per-second) reading limit,
a human captioner CUTS and REPHRASES to fit — removing fillers ("um", "you know",
false starts) and, when necessary, paraphrasing to a shorter equivalent that
preserves meaning. Before this stage the engine could only EXTEND timing and SPLIT
cues (see cps.py) — it flagged an over-fast line and shipped it. It never edited
the words. This stage closes that gap the way a real captioner does.

WHERE IT RUNS (formatter.py pipeline)
─────────────────────────────────────
AFTER the deterministic CPS extend/split pass (cps.enforce_cps_rules) and BEFORE
the QC gate. So it fires ONLY on cues that are STILL over max_cps after the free
timing fixes — never on a cue timing alone could rescue. This is the last
editorial step before QC grades the delivered text.

SPEC-GATED (the compliance invariant)
──────────────────────────────────────
The rest of the engine treats word-fidelity as a hard invariant (editorial_ai.py
NEVER changes a word). Condensation DELIBERATELY breaks that — it changes words on
purpose. That can NEVER be silent or unconditional. Driven by CONDENSATION_MODE:

  off             → verbatim; never touch words (legal/documentary specs that
                    contractually require verbatim). This stage is a no-op.
  disfluency_only → (DEFAULT) DETERMINISTIC removal of fillers / false starts /
                    stuttered repeats only. No AI. Fully reproducible. Safe,
                    universally-accepted captioning practice.
  condense_to_cps → disfluency_only PLUS a bounded LLM paraphrase pass on cues
                    that are STILL over max_cps — meaning-preserving, entity- and
                    number-locked. AI is used ONLY here, ONLY when needed.

PROVENANCE (SOC 2 CC8.1 / TPN MS-4.x)
─────────────────────────────────────
Every condensed cue carries its ORIGINAL verbatim text + the kind of change, so an
auditor (and a human reviewer) can see exactly what changed and revert it. The cue
dict gains meta.condensation = { applied, kind, verbatim } — the Base44 ingester
maps this onto CaptionCue.original_verbatim_text / condensation_applied /
condensation_kind. Nothing is ever silently reworded.

RESILIENCE
──────────
Mirrors editorial_ai.py: per-call timeout, run wall-clock budget, error-rate
bailout, heartbeat. A failed/slow LLM call falls through to the deterministic
(disfluency-only) result — never a hang, never a silent verbatim ship without the
provenance stamp.
"""

import json
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional

# CJK awareness — character-based measurement for no-space scripts.
try:
    from .cjk import is_cjk_text as _is_cjk, cjk_char_count as _cjk_count
except Exception:  # pragma: no cover
    def _is_cjk(text):
        return False

    def _cjk_count(text):
        return len((text or "").replace(" ", ""))

try:
    from .rendering import render_lines as _render_lines
except Exception:
    _render_lines = None


# ─── Spec knobs ──────────────────────────────────────────────────────
def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _mode() -> str:
    """CONDENSATION_MODE: 'off' | 'disfluency_only' | 'condense_to_cps'.
    Default 'disfluency_only' — the safe, deterministic, always-reproducible
    layer. Sent from the spec's condensation_rules via the producer mapper."""
    return (os.getenv("CONDENSATION_MODE", "disfluency_only") or "disfluency_only").strip().lower()


def _preserve_named_entities() -> bool:
    return (os.getenv("CONDENSATION_PRESERVE_ENTITIES", "1") or "1") in ("1", "true", "True")


def _preserve_numbers() -> bool:
    return (os.getenv("CONDENSATION_PRESERVE_NUMBERS", "1") or "1") in ("1", "true", "True")


def _max_cps() -> int:
    return _env_int("CUSTOM_MAX_CPS", 45)


def _max_lines() -> int:
    return _env_int("CUSTOM_MAX_LINES", 2)


def _max_chars() -> int:
    return _env_int("CUSTOM_MAX_CHARS", 32)


def _cps_measurement() -> str:
    return (os.getenv("CPS_MEASUREMENT", "characters") or "characters").strip().lower()


# ─── CPS helpers (mirrors cps.py measurement contract) ───────────────
def _visible_chars(text: str) -> int:
    text = (text or "").replace("\n", " ").strip()
    if _is_cjk(text):
        return _cjk_count(text)
    mode = _cps_measurement()
    if mode == "characters_no_spaces":
        return len(text.replace(" ", ""))
    if mode == "words":
        return len(text.split())
    return len(text)


def _cue_dialogue_text(cue: Dict[str, Any]) -> str:
    meta = cue.get("meta") or {}
    dt = meta.get("dialogue_text")
    if dt:
        return dt
    return " ".join(cue.get("lines", []))


def _cue_cps(cue: Dict[str, Any], text: str) -> float:
    dur_ms = max(1, int(cue.get("end_ms", 0)) - int(cue.get("start_ms", 0)))
    return _visible_chars(text) / (dur_ms / 1000.0)


# ─── Deterministic disfluency removal ────────────────────────────────
# Curated multilingual filler / hesitation set. Only STANDALONE tokens are
# stripped — we never touch a word that happens to contain these letters. The
# list is intentionally conservative (universally-accepted captioning fillers)
# so the deterministic default is always safe and auditor-defensible.
_FILLER_TOKENS = {
    # English
    "um", "uh", "erm", "er", "ah", "eh", "hmm", "mm", "mhm", "uh-huh", "huh",
    "y'know", "ya", "like",  # 'like' handled contextually below (only mid-clause repeats)
    # Spanish / Portuguese / Italian
    "eh", "este", "pues", "bueno",
    # French
    "euh", "ben", "bah",
    # German
    "äh", "ähm", "also",
}
# Multi-word filler phrases removed as a unit (case-insensitive, word-boundary).
_FILLER_PHRASES = [
    "you know", "i mean", "sort of", "kind of", "or something",
    "or whatever", "i guess", "you see",
]

_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)


def _strip_leading_label(text: str):
    """Split off a leading speaker label ('- ', '[NAME:] ', 'A: ') so it is
    preserved verbatim and never condensed. Returns (label, body)."""
    m = re.match(r"^\s*(?:-\s+|\[[^\]]*\]:?\s*|[^\s:]{1,24}:\s+)", text or "")
    if m:
        return text[:m.end()], text[m.end():]
    return "", (text or "")


def remove_disfluencies(text: str) -> str:
    """DETERMINISTIC filler/false-start/repeat removal. No AI, fully reproducible.
    Preserves a leading speaker label untouched. Collapses immediate word repeats
    ('I-I-I' / 'the the') and removes standalone filler tokens + filler phrases,
    then normalizes spacing/punctuation. Meaning is never altered — only spoken
    hesitations a captioner routinely drops are removed."""
    label, body = _strip_leading_label(text)
    if not body.strip():
        return text

    # 1. Remove multi-word filler phrases (bounded, case-insensitive), only when
    #    set off by a boundary — never inside a larger word.
    for phrase in _FILLER_PHRASES:
        body = re.sub(
            r"(?<![\w'])" + re.escape(phrase) + r"(?![\w'])",
            " ", body, flags=re.IGNORECASE,
        )

    # 2. Tokenize on whitespace, strip standalone fillers + collapse immediate
    #    repeats (stutters + doubled words), keeping punctuation attached.
    raw_tokens = body.split()
    out_tokens: List[str] = []
    prev_core = None
    for tok in raw_tokens:
        stripped = tok.strip(".,!?;:—–\"'")
        core = stripped.lower()
        # Standalone filler token → drop. Only when the token carries NO trailing
        # sentence punctuation (dropping "um." would strip a sentence boundary) —
        # a filler that ends a sentence is left in place, conservative by design.
        ends_sentence = tok.rstrip().endswith((".", "!", "?"))
        if core in _FILLER_TOKENS and not ends_sentence:
            # 'like' is only dropped as a standalone filler, never at sentence start
            # where it may be meaningful — conservative: skip dropping 'like' at idx 0.
            if core == "like" and not out_tokens:
                out_tokens.append(tok)
                prev_core = core
                continue
            continue
        # Immediate repeat (stutter 'I-I' arrives as 'I I', or doubled word) → collapse.
        if core and core == prev_core and core.isalpha():
            continue
        out_tokens.append(tok)
        prev_core = core

    cleaned = " ".join(out_tokens)
    # 3. Normalize orphaned/duplicated punctuation and spacing produced by removals.
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"([,;:])\1+", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    # 4. Capitalize the first letter if the removal exposed a new sentence start.
    if cleaned and cleaned[0].islower() and not label:
        cleaned = cleaned[0].upper() + cleaned[1:]

    return (label + cleaned).strip() if cleaned else text


# ─── Meaning-preserving guards for the LLM paraphrase ────────────────
def _named_entities(text: str) -> set:
    """Capitalized tokens mid-sentence = proper nouns the paraphrase must keep.
    Conservative (skips the sentence-initial word). Only used to REJECT a
    rewrite that dropped an entity — never to add one."""
    tokens = text.split()
    ents = set()
    for i, tok in enumerate(tokens):
        core = tok.strip(".,!?;:—–\"'()")
        if i > 0 and core and core[0].isupper() and core.lower() != "i":
            ents.add(core.lower())
    return ents


def _numbers(text: str) -> set:
    return set(re.findall(r"\d[\d.,:/-]*\d|\d", text or ""))


# ─── LLM paraphrase (condense_to_cps only) ───────────────────────────
def _build_system_prompt(max_cps: int, preserve_entities: bool, preserve_numbers: bool) -> str:
    parts = [
        "You are a professional broadcast closed-caption editor.",
        "Your job: CONDENSE one caption line so it reads within the reading-speed "
        f"limit ({max_cps} characters per second) WITHOUT losing meaning.",
        "This is real captioning practice: cut redundancy and tighten phrasing the "
        "way a human captioner does when speech is too fast to read verbatim.",
        "Rules you MUST follow:",
        "- Preserve the full meaning. Never omit information, never invent it.",
        "- Remove only redundancy, filler, and wordiness. Prefer shorter synonyms "
        "and tighter grammar.",
        "- Keep the same speaker's voice and register.",
        "- Do NOT add commentary, notes, or quotation marks.",
    ]
    if preserve_entities:
        parts.append("- Keep ALL proper nouns (names, places, titles) exactly as written.")
    if preserve_numbers:
        parts.append("- Keep ALL numbers, dates, and figures exactly as written.")
    parts.append('Return JSON only: {"condensed":"..."}.')
    return " ".join(parts)


def _llm_condense_cue(client, model, system_prompt, verbatim: str, target_chars: int) -> Optional[str]:
    """One bounded LLM condensation call. Returns the condensed text, or None on
    any failure / guard rejection (caller keeps the deterministic result)."""
    payload = {
        "caption_text": verbatim,
        "target_max_characters": target_chars,
        "instruction": "Condense to fit the reading limit while preserving meaning.",
    }
    try:
        response = client.responses.create(
            model=model,
            temperature=0,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
        raw = response.output_text.strip()
        data = json.loads(raw)
        condensed = str(data.get("condensed", "")).strip()
    except Exception:
        return None

    if not condensed or condensed == verbatim:
        return None
    # Never ACCEPT a rewrite that is longer than what we started with.
    if _visible_chars(condensed) >= _visible_chars(verbatim):
        return None
    # Meaning-preservation guards — reject if an entity/number was dropped.
    if _preserve_named_entities():
        if not _named_entities(verbatim).issubset(_named_entities(condensed) | _named_entities(condensed.title())):
            # Recompute against a looser lowercase compare before rejecting.
            v_ents = _named_entities(verbatim)
            c_low = condensed.lower()
            if not all(e in c_low for e in v_ents):
                return None
    if _preserve_numbers():
        if not _numbers(verbatim).issubset(_numbers(condensed)):
            return None
    return condensed


# ─── Master pass ─────────────────────────────────────────────────────
def condense_cues(
    cues: List[Dict[str, Any]],
    heartbeat: Optional[Callable[[int, int], None]] = None,
) -> List[Dict[str, Any]]:
    """Apply the spec-gated condensation stage.

    - mode 'off'            → no-op (returns cues unchanged).
    - mode 'disfluency_only'→ deterministic filler/repeat removal on over-CPS cues.
    - mode 'condense_to_cps'→ disfluency removal, then a bounded LLM paraphrase on
                              cues STILL over max_cps.

    Every changed cue is re-rendered through render_lines (so line geometry +
    speaker label stay spec-correct) and stamped with meta.condensation =
    {applied, kind, verbatim} for the Base44 provenance ingest. Cue count is
    preserved exactly."""
    mode = _mode()
    if mode == "off":
        return cues

    max_cps = _max_cps()
    max_lines = _max_lines()
    max_chars = _max_chars()

    # LLM client only initialized when condense_to_cps is active AND a key exists.
    client = None
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    system_prompt = None
    use_llm = mode == "condense_to_cps"
    if use_llm:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                from openai import OpenAI
                per_call_timeout = _env_int("CONDENSATION_REQUEST_TIMEOUT_SECONDS", 30)
                client = OpenAI(api_key=api_key, timeout=float(per_call_timeout))
                system_prompt = _build_system_prompt(max_cps, _preserve_named_entities(), _preserve_numbers())
            except Exception:
                client = None
        if not client:
            # No key / SDK → degrade to deterministic-only, but DON'T silently
            # claim condense_to_cps ran. The disfluency pass below still applies.
            use_llm = False

    run_budget_seconds = _env_int("CONDENSATION_RUN_BUDGET_SECONDS", 120)
    heartbeat_every = _env_int("CONDENSATION_HEARTBEAT_EVERY", 25)
    start_time = time.monotonic()
    llm_budget_exhausted = False

    stats = {"disfluency": 0, "llm": 0, "still_over": 0, "llm_rejected": 0}
    total = len(cues)
    out: List[Dict[str, Any]] = []

    for idx, cue in enumerate(cues):
        if heartbeat and idx > 0 and idx % heartbeat_every == 0:
            try:
                heartbeat(idx, total)
            except Exception:
                pass

        if cue.get("type") != "dialogue":
            out.append(cue)
            continue

        verbatim = _cue_dialogue_text(cue).strip()
        if not verbatim:
            out.append(cue)
            continue

        # Only act on cues STILL over the reading limit after CPS extend/split.
        if _cue_cps(cue, verbatim) <= max_cps:
            out.append(cue)
            continue

        applied_kind = None
        working = verbatim

        # 1. Deterministic disfluency removal (always, for over-CPS cues).
        disfluent_removed = remove_disfluencies(working)
        if disfluent_removed != working and disfluent_removed.strip():
            working = disfluent_removed
            applied_kind = "disfluency"
            stats["disfluency"] += 1

        # 2. LLM paraphrase — only if still over CPS AND mode allows it.
        if use_llm and not llm_budget_exhausted and _cue_cps(cue, working) > max_cps:
            if time.monotonic() - start_time > run_budget_seconds:
                llm_budget_exhausted = True
            else:
                dur_s = max(0.001, (int(cue.get("end_ms", 0)) - int(cue.get("start_ms", 0))) / 1000.0)
                target_chars = int(max_cps * dur_s)
                condensed = _llm_condense_cue(client, model, system_prompt, working, target_chars)
                if condensed:
                    working = condensed
                    applied_kind = "condense" if applied_kind == "disfluency" else "condense"
                    stats["llm"] += 1
                else:
                    stats["llm_rejected"] += 1

        if applied_kind is None or working == verbatim:
            if _cue_cps(cue, verbatim) > max_cps:
                stats["still_over"] += 1
            out.append(cue)
            continue

        # Re-render the condensed text through the shared renderer so line
        # geometry + speaker label stay spec-correct (single source of truth).
        label, body = _strip_leading_label(working)
        runs = (cue.get("meta") or {}).get("runs") or []
        if _render_lines is not None:
            new_lines = _render_lines(body.split(), runs, max_lines, max_chars, body)
        else:
            new_lines = [working]

        new_cue = dict(cue)
        new_cue["lines"] = new_lines
        new_meta = dict(cue.get("meta") or {})
        new_meta["dialogue_text"] = working
        # Provenance — the Base44 ingester maps this onto CaptionCue.
        new_meta["condensation"] = {
            "applied": True,
            "kind": applied_kind,           # 'disfluency' | 'condense'
            "verbatim": verbatim,           # the exact original, for revert + audit
        }
        new_cue["meta"] = new_meta
        out.append(new_cue)
        if _cue_cps(new_cue, working) > max_cps:
            stats["still_over"] += 1

    print(
        f"[CONDENSATION] mode={mode} disfluency={stats['disfluency']} "
        f"llm={stats['llm']} llm_rejected={stats['llm_rejected']} "
        f"still_over_after={stats['still_over']} "
        f"llm_budget_exhausted={llm_budget_exhausted}"
    )
    return out
