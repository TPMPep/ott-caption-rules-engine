"""
Deployed-engine diagnostics — bounded, allowlisted fixture verification.

WHY THIS EXISTS
───────────────
The production engine exposes only /health + async POST /v1/jobs (audio
transcription). There was no way to prove the DEPLOYED Railway image can import
and execute the full services.* pipeline against a known input and produce the
expected structured-speaker / pause-boundary / decision-identity output — i.e.
no way to verify a deploy landed beyond trusting the /health version string.

This module is that proof, WITHOUT opening a generic transcript-processing
surface. It exposes NOTHING to the caller except:

  • an ALLOWLISTED fixture id (no caller-supplied tokens / transcript / options)
  • the real process_caption_job run over deterministic, committed fixtures
  • a bounded diagnostic result (rendered lines, times, structured speaker
    attribution, pause provenance, decision identity) — never raw word arrays
  • a FIXED build manifest (engine version + SHA-256 of the ten known runtime
    files) — enumerated internally, never a caller-supplied path.

HARD SECURITY POSTURE (enforced by the route in main.py, mirrored here):
  • The route is 404 unless ENABLE_ENGINE_DIAGNOSTICS=true (disable = no code
    deploy needed; flip the env var and the route vanishes on next boot).
  • The route requires the existing X-Engine-Secret.
  • The request body accepts ONLY { "fixture_id": <allowlisted string> }. Any
    extra key (tokens, transcript, env, captionOptions, path, ...) is REJECTED.
  • No provider calls, no S3, no network, no DB, no persistence, no logging of
    fixture transcript text.
  • Zero effect on /v1/jobs.

The fixtures below are the SOURCE OF TRUTH for the speaker/pause verification —
they encode the exact tokens, timings, source-utterance ids, and caption options
the acceptance criteria require, committed with the engine so the deployed image
runs the identical input every time. SOC 2 CC8.1 — deterministic + reproducible.
"""

import hashlib
import os
from typing import Any, Dict, List, Optional, Tuple

# The real pipeline — the whole point is to exercise the DEPLOYED code paths.
from .formatter import (
    process_caption_job,
    apply_env_overrides,
    restore_env_overrides,
)

# ─── Allowlisted fixtures ────────────────────────────────────────────
# Each fixture is a deterministic (tokens, audio_events, caption_options) triple.
# Tokens carry realistic timings + distinct source_utterance_id values so the
# segmentation pause-boundary + speaker-ownership invariants are genuinely
# exercised. NOTHING here is caller-supplied.

# Pluto-style dash-speaker caption options (the applicable policy for the
# cross-speaker fixture). Bounded, engine-native env knobs only.
_PLUTO_DASH_OPTIONS: Dict[str, Any] = {
    "CAPTION_PROFILE": "pluto",
    "CUSTOM_MAX_CHARS": "32",
    "CUSTOM_MAX_LINES": "2",
    "CUSTOM_MAX_CPS": "17",
    "SPEAKER_LABEL_MODE": "dash",
    "CUSTOM_PAUSE_BOUNDARY_MS": "1200",
    "SEQ_OPTIMIZER_ENABLED": "1",
    "OUTPUT_FORMATS": "srt",
}

_DEFAULT_OPTIONS: Dict[str, Any] = {
    "CUSTOM_MAX_CHARS": "42",
    "CUSTOM_MAX_LINES": "2",
    "CUSTOM_MAX_CPS": "17",
    "CUSTOM_PAUSE_BOUNDARY_MS": "1200",
    "SEQ_OPTIMIZER_ENABLED": "1",
    "OUTPUT_FORMATS": "srt",
}


def _tok(text: str, start: int, end: int, speaker: str, utt: int) -> Dict[str, Any]:
    return {
        "text": text,
        "start_ms": start,
        "end_ms": end,
        "speaker": speaker,
        "source_utterance_id": utt,
    }


def _words(text: str, start: int, end: int, speaker: str, utt: int) -> List[Dict[str, Any]]:
    """Split a phrase into per-word tokens with proportional timings inside
    [start, end], all sharing one source_utterance_id + speaker."""
    parts = text.split()
    if not parts:
        return []
    span = max(1, end - start)
    total = sum(len(p) for p in parts) or 1
    out: List[Dict[str, Any]] = []
    cursor = 0
    for p in parts:
        w_start = start + (span * cursor) // total
        cursor += len(p)
        w_end = start + (span * cursor) // total
        out.append(_tok(p, int(w_start), int(max(w_end, w_start + 1)), speaker, utt))
    return out


def _fixture_cross_speaker_a_g() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Speaker A: 'Oh, honey, you scared me. You okay?' immediately followed by
    Speaker G: 'I just wanted some milk.' — distinct source-utterance ids. Under
    the Pluto dash policy this must NOT fuse both speakers into one flat cue."""
    tokens: List[Dict[str, Any]] = []
    tokens += _words("Oh, honey, you scared me. You okay?", 0, 2600, "A", 1)
    tokens += _words("I just wanted some milk.", 2750, 3950, "G", 2)
    return tokens, [], dict(_PLUTO_DASH_OPTIONS)


def _fixture_pause_1460ms() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Same speaker A: 'Not as much, but I'm easier to handle.' then a DISTINCT
    source utterance 'Cookie?' beginning after a measured 1,460 ms gap. With
    CUSTOM_PAUSE_BOUNDARY_MS=1200 the gap (1460 ≥ 1200) is an immutable boundary
    — 'Cookie?' must begin its own cue and never rejoin the prior line."""
    tokens: List[Dict[str, Any]] = []
    tokens += _words("Not as much, but I'm easier to handle.", 60000, 62400, "A", 10)
    # Gap = 63860 - 62400 = 1460 ms (≥ 1200 pause boundary), new utterance id.
    tokens += _words("Cookie?", 63860, 64660, "A", 11)
    return tokens, [], dict(_PLUTO_DASH_OPTIONS)


def _fixture_passthrough_identity() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Three UNRELATED single-speaker passthrough windows, each a distinct
    source utterance separated by hard pauses, so each must receive its OWN
    decision identity (no shared seq:undefined collapse). Plus the bounded
    contract cases: one known speaker, a legitimate two-speaker dash cue, an
    unknown speaker, and a non-dialogue/music cue."""
    tokens: List[Dict[str, Any]] = []
    # Window 1 — speaker A, utterance 20.
    tokens += _words("This is the first separate line.", 0, 2400, "A", 20)
    # Window 2 — speaker B, utterance 21, after a hard pause.
    tokens += _words("And this is a second unrelated line.", 4000, 6600, "B", 21)
    # Window 3 — speaker C, utterance 22, after another hard pause.
    tokens += _words("Finally a third independent line.", 8000, 10400, "C", 22)
    # Two-speaker dash cue candidate — tight A/D back-and-forth, distinct utts.
    tokens += _words("Quick question here.", 12000, 12900, "A", 23)
    tokens += _words("Quick answer there.", 13050, 13950, "D", 24)
    # Unknown-speaker line — speaker is None so it is review-required.
    tokens += _words("Who is even speaking now?", 15500, 17000, None, 25)
    audio_events = [
        {"event_type": "music", "start": 19000, "end": 21000, "text": "music"},
    ]
    return tokens, audio_events, dict(_PLUTO_DASH_OPTIONS)


# fixture_id → builder. The ONLY inputs the route accepts.
_FIXTURES = {
    "cross_speaker_a_g": _fixture_cross_speaker_a_g,
    "pause_1460ms": _fixture_pause_1460ms,
    "passthrough_identity": _fixture_passthrough_identity,
}

ALLOWED_FIXTURE_IDS = tuple(sorted(_FIXTURES.keys()))


# ─── Build manifest ──────────────────────────────────────────────────
# FIXED enumeration of the ten runtime files — NEVER a caller-supplied path,
# NEVER arbitrary file hashing. The route returns engine version + a SHA-256 per
# file so a checksum (not just the version string) proves what is deployed.
_MANIFEST_FILES = (
    "main.py",
    "services/boundaries.py",
    "services/segmentation.py",
    "services/assembly.py",
    "services/formatter.py",
    "services/shaping.py",
    "services/readability.py",
    "services/cps.py",
    "services/condensation.py",
    "services/sequence_optimizer.py",
)


def _engine_root() -> str:
    # services/diagnostics.py → parent of services/ is the engine root.
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_manifest(version: str) -> Dict[str, Any]:
    """Return { engine_version, build_id, files: {relpath: sha256|null} } for the
    ten allowlisted files. Missing/unreadable files report null (never an error
    that leaks a path or stack trace)."""
    root = _engine_root()
    files: Dict[str, Optional[str]] = {}
    for rel in _MANIFEST_FILES:
        abs_path = os.path.join(root, rel)
        try:
            with open(abs_path, "rb") as fh:
                files[rel] = hashlib.sha256(fh.read()).hexdigest()
        except Exception:
            files[rel] = None
    return {
        "engine_version": version,
        "build_id": os.getenv("RAILWAY_GIT_COMMIT_SHA")
        or os.getenv("BUILD_ID")
        or None,
        "files": files,
    }


# ─── Bounded result projection ───────────────────────────────────────
def _bounded_cue(c: Dict[str, Any]) -> Dict[str, Any]:
    """Project ONE engine result cue into the bounded diagnostic shape. Carries
    rendered lines, times, structured speaker attribution, the bounded pause
    provenance, and the seq_opt DECISION IDENTITY — but NEVER raw word arrays,
    candidate summaries, or original window text."""
    meta = c.get("meta") or {}
    so = meta.get("seq_opt") or {}
    pause = meta.get("pause_provenance") or None
    decision_identity = {
        "operation": so.get("operation"),
        "transformation_sequence": so.get("transformation_sequence"),
        "decision_key_unbound": so.get("decision_key_unbound"),
        "input_hash": so.get("input_hash"),
        "candidate_set_hash": so.get("candidate_set_hash"),
        "output_hash": so.get("output_hash"),
        "source_cue_ids": (so.get("source_cue_ids") or [])[:20],
    } if so else None
    return {
        "idx": c.get("idx"),
        "start_ms": c.get("start_ms"),
        "end_ms": c.get("end_ms"),
        "type": c.get("type", "dialogue"),
        "lines": c.get("lines") or [],
        "speaker_label": c.get("speaker_label", ""),
        "speaker_segments": c.get("speaker_segments") or [],
        "speaker_review_required": bool(c.get("speaker_review_required", False)),
        "pause_provenance": pause,
        "decision_identity": decision_identity,
    }


class DiagnosticsError(Exception):
    """Raised for a bad fixture request. The route maps this to a 400 with a
    SAFE message (no paths, no internals)."""


def run_fixture(fixture_id: str, version: str) -> Dict[str, Any]:
    """Run ONE allowlisted fixture through the REAL process_caption_job and
    return the bounded diagnostic result + build manifest. Raises
    DiagnosticsError on an unknown fixture id.

    No provider calls, no S3, no network, no persistence — process_caption_job
    over in-memory tokens only. Env overrides are applied + fully restored so a
    diagnostic run can never leak env state into a concurrent /v1/jobs run."""
    builder = _FIXTURES.get(fixture_id)
    if builder is None:
        raise DiagnosticsError("unknown fixture_id")

    tokens, audio_events, options = builder()

    # Build a backbone SRT locally from the tokens (one cue per utterance) so the
    # formatter's normal input contract is satisfied without any provider call.
    env_snapshot = None
    try:
        env_snapshot = apply_env_overrides(options)
        result = process_caption_job(
            backbone_srt_text="",
            timestamps={"words": tokens},
            protected_phrases=[],
            output_formats=["srt"],
            heartbeat=None,
            audio_events=audio_events,
        )
    finally:
        if env_snapshot is not None:
            restore_env_overrides(env_snapshot)

    cues = result.get("cues") or []
    bounded = [_bounded_cue(c) for c in cues]

    # Bounded pause-provenance rollup for the caller: which cues opened at a hard
    # pause + the effective threshold. Enumerated from the cue projection only.
    pause_boundaries = [
        {
            "idx": bc["idx"],
            "lines": bc["lines"],
            "pause_provenance": bc["pause_provenance"],
        }
        for bc in bounded
        if bc.get("pause_provenance")
    ]

    # Distinct decision identities (the passthrough-linkage evidence).
    decision_keys = []
    for bc in bounded:
        di = bc.get("decision_identity")
        if di and di.get("decision_key_unbound"):
            decision_keys.append(di["decision_key_unbound"])
    distinct_keys = sorted(set(decision_keys))

    return {
        "fixture_id": fixture_id,
        "caption_options": options,
        "cue_count": len(bounded),
        "cues": bounded,
        "pause_boundaries": pause_boundaries,
        "decision_identity_summary": {
            "decision_key_count": len(decision_keys),
            "distinct_decision_key_count": len(distinct_keys),
            "all_keys_unique": len(decision_keys) == len(distinct_keys),
        },
        "manifest": build_manifest(version),
    }
