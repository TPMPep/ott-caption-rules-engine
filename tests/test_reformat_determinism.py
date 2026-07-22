"""
Phase 1 DETERMINISM ACCEPTANCE HARNESS (engine side).
=====================================================
Proves the Design A contract for reformat_from_baseline (Apply Spec): the
DETERMINISTIC formatting pipeline, run against the immutable baseline with the
editorial-AI stage forbidden, produces BYTE-IDENTICAL delivered output on every
run — sequentially, concurrently, and in a clean process — and never calls
OpenAI.

WHAT THIS FILE IS / IS NOT
──────────────────────────
• It exercises the REAL pipeline (main.process_caption_job via the committed
  reformat_baseline.json fixture) — not a re-implementation.
• The "clean interpreter/process run" (test_clean_process_run_matches) spawns a
  FRESH python -c subprocess so module-level state (rules context, caches) can
  never carry across — a genuine cold-start reproducibility proof.
• Point 6: the OpenAI stage is blocked TWO ways and any invocation FAILS the run:
    (a) services.editorial_ai.FORBID_EDITORIAL_AI = True (raises if reached);
    (b) openai.OpenAI is monkeypatched to a sentinel that raises on construction.
  Neither relies on a log line. If the reformat path calls OpenAI, the test errors.
• Cross-language hash parity (engine↔JS) is asserted in the JS suite
  (src/lib/__tests__/cc-reformat-determinism-parity.test.js) against the SAME
  committed parity vector (fixtures/reformat_output_parity.json). This file
  WRITES/asserts the engine side of that vector so the two stay locked.

Run:  pytest src/cc-rules-engine/tests/test_reformat_determinism.py -v
"""

import json
import os
import subprocess
import sys
import concurrent.futures

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import services.editorial_ai as editorial_ai  # noqa: E402
from services.formatter import process_caption_job  # noqa: E402
from services.canonical_hash import canonical_sha256  # noqa: E402
import main  # noqa: E402

_FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
_BASELINE_PATH = os.path.join(_FIXTURE_DIR, "reformat_baseline.json")
_PARITY_PATH = os.path.join(_FIXTURE_DIR, "reformat_output_parity.json")

# The pinned spec geometry the reformat is graded against — a fixed 42×2 / 17cps
# posture. This IS the captionOptions/env bundle Base44's ccApplySpec would send;
# holding it constant makes the run reproducible. NO OpenAI-related knob here.
_ENV_OVERRIDES = {
    "CUSTOM_MAX_CHARS": "42",
    "CUSTOM_MAX_LINES": "2",
    "CUSTOM_TARGET_CPS": "15",
    "CUSTOM_MAX_CPS": "17",
    "CUSTOM_MIN_CPS": "5",
    "CPS_MEASUREMENT": "characters",
    "CUSTOM_MIN_DISPLAY_MS": "833",
    "CUSTOM_MAX_DISPLAY_MS": "7000",
    "CUSTOM_FRAME_RATE": "25",
    "CUSTOM_MIN_GAP_FRAMES": "2",
    "CUSTOM_MERGE_GAP_MS": "80",
    "CUSTOM_PAUSE_BOUNDARY_MS": "1200",
    "CUSTOM_TARGET_DURATION_MS": "3000",
    "CUSTOM_SHAPING_ENABLED": "1",
    "SPEAKER_LABEL_MODE": "dash",
    "MUSIC_CUE_FORMAT": "bracketed_uppercase",
    "SOUND_EFFECT_FORMAT": "bracketed_uppercase",
    "SOUND_DENSITY": "aggressive",
    "CONDENSATION_MODE": "disfluency_only",
    "OUTPUT_FORMATS": "srt,vtt",
    "SEQ_OPTIMIZER_ENABLED": "1",
    "SEQ_OPTIMIZER_VERSION": "1",
    "SPEC_SLUG": "acceptance-42x2",
    "SPEC_VERSION": "1",
}


def _load_baseline():
    with open(_BASELINE_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


class _ExplodingOpenAI:
    """Sentinel that fails LOUD if anything tries to construct an OpenAI client
    on the deterministic reformat path. Point 6: proof-by-failure, not by log."""
    def __init__(self, *a, **k):
        raise AssertionError(
            "OpenAI client was constructed during a deterministic reformat run — "
            "the no-editorial-AI guarantee is violated."
        )


def _run_reformat_once(baseline=None, env=None):
    """Run ONE deterministic reformat exactly as main.run_caption_job does on the
    reformat_from_baseline path: map baseline → assembly_result, build inputs,
    run process_caption_job with allow_editorial_ai=False, and compute the
    canonical output hash the ingester persists. Returns (cues, canonical_hash,
    deterministic_inputs). The editorial-AI stage is blocked two ways for the
    duration of the call."""
    baseline = baseline or _load_baseline()
    env = env if env is not None else dict(_ENV_OVERRIDES)

    from services.assembly import build_caption_inputs

    token = main.apply_env_overrides(env)
    forbid_prev = editorial_ai.FORBID_EDITORIAL_AI
    openai_prev = None
    try:
        # (a) source-level guard: reaching editorial_refine_cues raises.
        editorial_ai.FORBID_EDITORIAL_AI = True
        # (b) construction-level guard: any OpenAI() build raises.
        try:
            import openai
            openai_prev = openai.OpenAI
            openai.OpenAI = _ExplodingOpenAI
        except Exception:
            openai_prev = None  # SDK not installed in CI → nothing to patch

        assembly_result = main.baseline_to_assembly_result(baseline)
        backbone_srt, timestamps = build_caption_inputs(assembly_result)
        audio_events = list(assembly_result.get("audio_events") or [])
        result = process_caption_job(
            backbone_srt_text=backbone_srt,
            timestamps=timestamps,
            protected_phrases=[],
            output_formats=["srt", "vtt"],
            audio_events=audio_events,
            allow_editorial_ai=False,  # DESIGN A: deterministic reformat path
        )
        cues = result.get("cues") or []
        chash = main.canonical_output_hash(cues, result.get("segmentation_qc"))
        det = main.build_deterministic_inputs(
            baseline_hash=main.baseline_content_hash(baseline),
            baseline_artifact_hash=main._sha256_hex_bytes(
                json.dumps(baseline, sort_keys=True).encode("utf-8")),
            overlay_hash=None,
            spec_version=env.get("SPEC_VERSION"),
            spec_slug=env.get("SPEC_SLUG"),
            overrides_source=env,
            editorial_mode="deterministic",
        )
        return cues, chash, det
    finally:
        editorial_ai.FORBID_EDITORIAL_AI = forbid_prev
        if openai_prev is not None:
            import openai
            openai.OpenAI = openai_prev
        main.restore_env_overrides(token)


# =============================================================================
# 1. SEQUENTIAL REPEATED RUNS — identical cue count + canonical output hash.
# =============================================================================
def test_sequential_repeated_runs_are_byte_identical():
    runs = [_run_reformat_once() for _ in range(5)]
    hashes = {r[1] for r in runs}
    counts = {len(r[0]) for r in runs}
    assert len(hashes) == 1, f"non-deterministic output_hash across runs: {hashes}"
    assert len(counts) == 1, f"cue count varied across runs: {counts}"
    # The canonical hash is a real 64-char lowercase SHA-256.
    h = next(iter(hashes))
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)


# =============================================================================
# 2. CONCURRENT REPEATED RUNS — parallelism must not fork the output.
# =============================================================================
def test_concurrent_runs_are_byte_identical():
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: _run_reformat_once(), range(8)))
    hashes = {r[1] for r in results}
    counts = {len(r[0]) for r in results}
    assert len(hashes) == 1, f"concurrent runs diverged: {hashes}"
    assert len(counts) == 1, f"concurrent cue counts diverged: {counts}"


# =============================================================================
# 3. CLEAN PROCESS RUN — a fresh interpreter reproduces the same hash.
# =============================================================================
def test_clean_process_run_matches_in_process():
    _, in_proc_hash, _ = _run_reformat_once()
    engine_dir = os.path.join(os.path.dirname(__file__), "..")
    prog = (
        "import json,os,sys;"
        "sys.path.insert(0, os.getcwd());"
        "import services.editorial_ai as e; e.FORBID_EDITORIAL_AI=True;"
        "import main;"
        "from services.assembly import build_caption_inputs;"
        "from services.formatter import process_caption_job;"
        f"b=json.load(open(os.path.join('tests','fixtures','reformat_baseline.json')));"
        f"env={json.dumps(_ENV_OVERRIDES)};"
        "t=main.apply_env_overrides(env);"
        "ar=main.baseline_to_assembly_result(b);"
        "bs,ts=build_caption_inputs(ar);"
        "r=process_caption_job(backbone_srt_text=bs,timestamps=ts,protected_phrases=[],"
        "output_formats=['srt','vtt'],audio_events=list(ar.get('audio_events') or []),"
        "allow_editorial_ai=False);"
        "main.restore_env_overrides(t);"
        "print(main.canonical_output_hash(r.get('cues') or [], r.get('segmentation_qc')))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", prog],
        cwd=engine_dir, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"clean-process run failed:\n{proc.stderr}"
    clean_hash = proc.stdout.strip().splitlines()[-1].strip()
    assert clean_hash == in_proc_hash, (
        f"clean-process hash {clean_hash} != in-process hash {in_proc_hash}")


# =============================================================================
# 6. NO OPENAI CALL — proof by failure, not by log.
# =============================================================================
def test_reformat_never_calls_openai():
    # If the reformat path reaches editorial_refine_cues, FORBID_EDITORIAL_AI
    # raises RuntimeError inside process_caption_job. If it constructs an OpenAI
    # client, _ExplodingOpenAI raises AssertionError. Either surfaces here.
    cues, chash, _ = _run_reformat_once()
    assert cues, "reformat produced no cues"
    assert chash


def test_forbid_flag_would_catch_a_direct_call():
    # Directly asserts the guard fires — so the block in test above is proven
    # to be a real trap, not a no-op that happened to never be reached.
    prev = editorial_ai.FORBID_EDITORIAL_AI
    editorial_ai.FORBID_EDITORIAL_AI = True
    try:
        with pytest.raises(RuntimeError):
            editorial_ai.editorial_refine_cues([{"type": "dialogue", "lines": ["hi"]}], [])
    finally:
        editorial_ai.FORBID_EDITORIAL_AI = prev


# =============================================================================
# 7a. OVERRIDES CANONICALIZATION — reordered keys ≡ same overrides_hash.
# =============================================================================
def test_reordered_override_keys_same_overrides_hash():
    _, _, det_a = _run_reformat_once(env=dict(_ENV_OVERRIDES))
    reordered = dict(reversed(list(_ENV_OVERRIDES.items())))
    _, _, det_b = _run_reformat_once(env=reordered)
    assert det_a["overrides_hash"] == det_b["overrides_hash"], \
        "override key order changed the overrides_hash — canonicalization broken"


# =============================================================================
# 7b. SEMANTIC OVERRIDE CHANGE — a real value change IS a different tuple.
# =============================================================================
def test_semantic_override_change_changes_overrides_hash():
    _, _, det_a = _run_reformat_once(env=dict(_ENV_OVERRIDES))
    changed = dict(_ENV_OVERRIDES)
    changed["CUSTOM_MAX_CHARS"] = "32"  # 42 → 32 is a genuinely different spec
    _, _, det_b = _run_reformat_once(env=changed)
    assert det_a["overrides_hash"] != det_b["overrides_hash"], \
        "a real override change did not change the overrides_hash"


# =============================================================================
# 7c. UNCHANGED BASELINE ARTIFACT + CONTENT HASHES across runs.
# =============================================================================
def test_baseline_hashes_stable_across_runs():
    b = _load_baseline()
    c1 = main.baseline_content_hash(b)
    c2 = main.baseline_content_hash(_load_baseline())
    assert c1 == c2 and len(c1) == 64
    # Content hash is invariant to cosmetic re-serialization (key order),
    # while artifact byte hash tracks the exact bytes.
    reserialized = json.loads(json.dumps(b, sort_keys=True))
    assert main.baseline_content_hash(reserialized) == c1


# =============================================================================
# 7d. ENGINE↔JS PARITY VECTOR — freeze the engine side of the shared vector.
# =============================================================================
def test_emit_and_assert_parity_vector():
    """Writes/verifies fixtures/reformat_output_parity.json: a fixed cue+QC
    payload and the engine's canonical_output_hash of it. The JS suite hashes
    the SAME payload with canonicalDeliveryOutputHash and asserts the same 64-hex
    value — the byte-for-byte cross-language lock (acceptance point 1)."""
    cues = [
        {"type": "dialogue", "start_ms": 0, "end_ms": 1400,
         "lines": ["- Now, today's story", "is about a father"],
         "speaker_label": "A", "speaker_segments": [], "speaker_review_required": False},
        {"type": "dialogue", "start_ms": 9200, "end_ms": 11500,
         "lines": ["That's not fair."],
         "speaker_label": "B", "speaker_segments": [], "speaker_review_required": False},
        {"type": "music", "start_ms": 18000, "end_ms": 21000,
         "lines": ["[GENTLE MUSIC]"], "speaker_label": "", "speaker_segments": [],
         "speaker_review_required": False},
    ]
    seg_qc = {"cue_summaries": [
        {"cue_id": "#0", "segmentation_qc_status": "pass",
         "segmentation_qc_highest_severity": "info",
         "segmentation_qc_issue_codes": [], "segmentation_qc_review_required": False},
        {"cue_id": "#1", "segmentation_qc_status": "warn",
         "segmentation_qc_highest_severity": "warn",
         "segmentation_qc_issue_codes": ["MICRO_CUE"],
         "segmentation_qc_review_required": False},
    ]}
    engine_hash = main.canonical_output_hash(cues, seg_qc)
    vector = {"_readme": "SHARED engine↔JS parity vector. Frozen contract. "
                         "test_reformat_determinism.py (Python) and "
                         "cc-reformat-determinism-parity.test.js (JS) both hash "
                         "`cues`+`seg_qc` and MUST get `expected_canonical_output_hash`.",
              "cues": cues, "seg_qc": seg_qc,
              "expected_canonical_output_hash": engine_hash}

    if os.path.exists(_PARITY_PATH):
        with open(_PARITY_PATH, "r", encoding="utf-8") as fh:
            existing = json.load(fh)
        assert existing.get("expected_canonical_output_hash") == engine_hash, (
            "engine canonical_output_hash changed vs committed parity vector — "
            "if intentional, update fixtures/reformat_output_parity.json AND the "
            "JS parity test in the same change.")
    else:
        with open(_PARITY_PATH, "w", encoding="utf-8") as fh:
            json.dump(vector, fh, indent=2, ensure_ascii=False)
            fh.write("\n")

    assert len(engine_hash) == 64 and all(c in "0123456789abcdef" for c in engine_hash)
