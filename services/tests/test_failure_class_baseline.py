"""
Phase 0 — Failure-Class Baseline Fixtures.

These fixtures encode the EIGHT global failure CLASSES from the Phase 0 root-cause
report (docs/phase0-root-cause-report.md) as CLASS INVARIANTS — never as
cue-number assertions tied to the reviewer's specific examples. The reviewer's
seven examples are evidence; the invariant is the product requirement.

WHY EVERY TEST IS xfail(strict=True)
────────────────────────────────────
Each test asserts the CORRECT (post-fix) behavior and is expected to FAIL against
today's engine. Marking them xfail(strict=True) means:
  • The suite stays GREEN today — the failures are expected, so CI is not broken
    by committing a baseline before the fixes land.
  • The day a phase actually corrects a class, the test becomes an UNEXPECTED
    PASS. Because strict=True, an unexpected pass HARD-FAILS the build — which
    forces whoever shipped the fix to delete the xfail marker and lock the test
    as a permanent, always-green regression guard. The baseline can therefore
    never silently rot: a class is either provably-still-broken (xfail) or
    provably-fixed-and-locked (green, no marker).

NO NUMERIC WEIGHTS ARE ASSERTED. The tests check STRUCTURAL invariants (word
count per cue, boundary quality, word conservation, label repetition, timing
plausibility) that must hold regardless of the final corpus-calibrated constants.

NO ENGINE CODE IS MODIFIED by this file. It is a read-only observation harness.

Each test's docstring names its class letter (A–H) and the report invariant it
guards, so an auditor can map test → report → code path in one hop.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.sequence_optimizer import optimize_cue_sequence  # noqa: E402
from services.segmentation import segment_into_sentence_groups  # noqa: E402


# ─── Env harness (identical posture to the existing corpus test) ────────────
_SPEC = dict(
    CUSTOM_MAX_CHARS=42, CUSTOM_MAX_LINES=2, CUSTOM_MAX_CPS=17,
    CUSTOM_TARGET_CPS=15, CUSTOM_MIN_DISPLAY_MS=833, CUSTOM_MAX_DISPLAY_MS=7000,
    CUSTOM_MERGE_GAP_MS=80, CUSTOM_TARGET_DURATION_MS=3000,
    SEQ_OPTIMIZER_ENABLED=1,
)


def _set_env(**kw):
    prev = {}
    for k, v in kw.items():
        prev[k] = os.environ.get(k)
        os.environ[k] = str(v)
    return prev


def _restore(prev):
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _cue(word_timings, speaker="A", **overrides):
    """Build a dialogue cue from explicit (text, start_ms, end_ms) word timings."""
    text = " ".join(w[0] for w in word_timings)
    cue = {
        "idx": 1, "start_ms": word_timings[0][1], "end_ms": word_timings[-1][2],
        "lines": [text], "type": "dialogue",
        "meta": {"dialogue_text": text,
                 "runs": [{"speaker": speaker, "word_start": 0}],
                 "word_timings": [{"text": w[0], "start_ms": w[1], "end_ms": w[2]}
                                  for w in word_timings]},
    }
    cue.update(overrides)
    return cue


def _text(c):
    return (c.get("meta") or {}).get("dialogue_text") or " ".join(c.get("lines", []))


def _tokens(cues):
    out = []
    for c in cues:
        out.extend(_text(c).split())
    return out


def _dur(c):
    return c["end_ms"] - c["start_ms"]


def _words(c):
    return _text(c).split()


# Forward-binding function words that should not END a cue (Class F). This is the
# report's list; the fixture asserts the CLASS, the engine will own the canonical set.
_FORWARD_BINDING_TAIL = {
    "the", "a", "an", "of", "to", "for", "at", "by", "with", "from", "into", "onto",
}


# ════════════════════════════════════════════════════════════════════════════
# CLASS A — malformed / implausible provider word timing (item D)
# Invariant: no single spoken word may occupy an implausible duration; the engine
# repairs it deterministically before segmentation and preserves the original
# span as evidence. (Report §2 Class A.)
# NOTE: the repair pass does not exist yet — Phase 1. The timings here are
# representative (synthetic) pending the real provider dump; the INVARIANT holds
# under either mechanism (explicit malformed word duration OR synthesized span).
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.xfail(strict=True, reason="Class A: word-timing anomaly repair lands in Phase 1")
def test_class_a_implausible_word_duration_is_repaired_before_segmentation():
    prev = _set_env(**_SPEC)
    try:
        # "There" occupies ~14.3s; the rest of the sentence is normal, contiguous.
        # A physically-plausible word is < ~1.2s; 14s is impossible for one word.
        wt = [
            ("There", 2589518, 2603858),          # ← implausible 14,340 ms span
            ("are", 2603898, 2604020),
            ("no", 2604020, 2604180),
            ("certainties", 2604180, 2604760),
            ("where", 2604760, 2604920),
            ("dreams", 2604920, 2605280),
            ("are", 2605280, 2605420),
            ("concerned.", 2605420, 2605980),
        ]
        cue = _cue(wt)
        out = optimize_cue_sequence([cue])

        # INVARIANT: after the engine processes this window, no emitted cue may
        # carry a single-word span that is physically implausible, and the whole
        # sentence must not have been exploded across many cues BY the corrupt
        # frame. We assert the plausibility invariant on the produced cue timings:
        PLAUSIBLE_WORD_CEILING_MS = 3000  # generous; a real word is < ~1.2s
        for c in out:
            wts = (c.get("meta") or {}).get("word_timings") or []
            for w in wts:
                span = int(w["end_ms"]) - int(w["start_ms"])
                assert span <= PLAUSIBLE_WORD_CEILING_MS, (
                    f"implausible word span survived into output: "
                    f"{w.get('text')!r} = {span}ms"
                )
        # And the sentence must NOT have shattered into >2 cues purely because of
        # the corrupt frame (8 short words easily fit 1–2 compliant cues).
        assert len(out) <= 2, f"corrupt frame exploded the sentence into {len(out)} cues"
    finally:
        _restore(prev)


# ════════════════════════════════════════════════════════════════════════════
# CLASS B — unnecessary fragmentation of complete semantic units (items 1, 7)
# Invariant: when multiple compliant layouts exist, prefer the fewest cue
# boundaries / strongest semantic grouping; never explode a sentence into
# one/two-word cues when a ≥3-word-per-side compliant layout exists. (§2 Class B.)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.xfail(strict=True, reason="Class B: fewest-boundaries scoring lands in Phase 2")
def test_class_b_complete_sentence_not_over_fragmented():
    prev = _set_env(**_SPEC)
    try:
        # "All right, ladies. I can watch all you in slow-mo on the highlights tape."
        # Comfortable timing (~3.4 words/sec) — no CPS/CPL/duration force to split
        # into 3. A professional layout is 1 or 2 cues, never 3.
        wt = [
            ("All", 0, 300), ("right,", 300, 700), ("ladies.", 700, 1300),
            ("I", 1300, 1450), ("can", 1450, 1700), ("watch", 1700, 2100),
            ("all", 2100, 2350), ("you", 2350, 2600), ("in", 2600, 2800),
            ("slow-mo", 2800, 3300), ("on", 3300, 3500), ("the", 3500, 3700),
            ("highlights", 3700, 4300), ("tape.", 4300, 4800),
        ]
        # Present it as an already-over-split 3-cue incoming arrangement to prove
        # the optimizer recombines toward the coarsest compliant grouping.
        cues = [_cue(wt[:3]), _cue(wt[3:10]), _cue(wt[10:])]
        cues[0]["start_ms"], cues[0]["end_ms"] = 0, 1300
        cues[1]["start_ms"], cues[1]["end_ms"] = 1300, 3300
        cues[2]["start_ms"], cues[2]["end_ms"] = 3300, 4800
        before = _tokens(cues)
        out = optimize_cue_sequence(cues)

        assert _tokens(out) == before, "words changed"
        # INVARIANT: fewer cues than the over-split incoming arrangement.
        assert len(out) <= 2, f"sentence stayed fragmented into {len(out)} cues"
        # INVARIANT: no ≤2-word dialogue cue produced.
        for c in out:
            assert len(_words(c)) >= 3, f"tiny fragment cue produced: {_text(c)!r}"
    finally:
        _restore(prev)


# ════════════════════════════════════════════════════════════════════════════
# CLASS C — over-weighting weak pauses / source-segment boundaries (item 7)
# Invariant: a boundary corresponding only to a weak pause / raw source-segment
# edge (no punctuation, no hard wall) must lose to a compliant coarser grouping.
# (§2 Class C — same scoring gap as B, viewed from the "why did the weak boundary
# survive" side.)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.xfail(strict=True, reason="Class C: weak-boundary demotion lands in Phase 2")
def test_class_c_weak_midphrase_boundary_recombined():
    prev = _set_env(**_SPEC)
    try:
        # One sentence, no internal punctuation, split into 3 incoming cues at
        # MID-PHRASE points (weak boundaries). All three are individually
        # compliant, so only the coarser-grouping preference should recombine them.
        wt = [
            ("There", 0, 300), ("are", 300, 500), ("no", 500, 700),
            ("certainties", 700, 1300), ("where", 1300, 1600),
            ("dreams", 1600, 2000), ("are", 2000, 2200),
            ("concerned.", 2200, 2800),
        ]
        cues = [_cue(wt[:3]), _cue(wt[3:5]), _cue(wt[5:])]
        cues[0]["start_ms"], cues[0]["end_ms"] = 0, 700
        cues[1]["start_ms"], cues[1]["end_ms"] = 700, 1600
        cues[2]["start_ms"], cues[2]["end_ms"] = 1600, 2800
        before = _tokens(cues)
        out = optimize_cue_sequence(cues)

        assert _tokens(out) == before
        # INVARIANT: the weak mid-phrase boundaries do not survive as 3 cues.
        assert len(out) <= 2, f"weak boundaries preserved as {len(out)} cues"
        # INVARIANT: no cue ends mid-phrase on a word with no punctuation when a
        # coarser grouping exists (spot-check: first cue should not end on "no").
        assert not _words(out[0])[-1].rstrip(".,;:!?").lower() == "no", \
            "recombination still ends first cue mid-phrase on 'no'"
    finally:
        _restore(prev)


# ════════════════════════════════════════════════════════════════════════════
# CLASS D — condensation before faithful compliant layouts exhausted (item 2)
# Invariant: spoken words are never removed while a faithful compliant layout
# (incl. a safe idle-duration borrow) exists. (§2 Class D.)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.xfail(strict=True, reason="Class D: borrow-duration candidate lands in Phase 3")
def test_class_d_no_condensation_when_borrow_duration_would_fit():
    prev = _set_env(**_SPEC)
    os.environ["CONDENSATION_MODE"] = "condense_to_cps"
    os.environ.pop("OPENAI_API_KEY", None)  # deterministic-only; no live LLM in CI
    try:
        # "Start 'em up, Coach." — short, but packed into a window tight enough to
        # trip CPS, WITH ample idle duration immediately after (next cue far away).
        # A legal borrow of that idle time makes the cue compliant with ALL words.
        wt = [
            ("Start", 0, 220), ("'em", 220, 380), ("up,", 380, 560),
            ("Coach.", 560, 900),
        ]
        cue = _cue(wt)
        cue["start_ms"], cue["end_ms"] = 0, 900   # ~0.9s → over-CPS as-is
        # A neighbor exists but 6s later, so >5s of idle duration is borrowable.
        neighbor = _cue([("Yeah.", 6000, 6600)])
        before = _tokens([cue, neighbor])
        out = optimize_cue_sequence([cue, neighbor])

        # INVARIANT: every spoken word survives — "up" is never dropped.
        assert _tokens(out) == before, f"words changed: {before} -> {_tokens(out)}"
        assert "up," in _tokens(out) or "up" in [w.rstrip(",") for w in _tokens(out)], \
            "'up' was removed despite borrowable idle duration"
    finally:
        _restore(prev)


# ════════════════════════════════════════════════════════════════════════════
# CLASS E — incorrect sentence-start capitalization (item 3)
# Invariant: a cue that starts a new sentence keeps a capital initial across every
# transform path. (§2 Class E — reproduce to disambiguate E1 vs E2 in Phase 3.)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.xfail(strict=True, reason="Class E: reproduced + fixed in Phase 3 (E1/E2 to be pinned)")
def test_class_e_sentence_start_capitalization_preserved():
    prev = _set_env(**_SPEC)
    try:
        from services.capitalization import apply_sentence_capitalization

        # Two cues: the first ends a sentence with a quote-closed terminal; the
        # second STARTS a new sentence. The new-sentence cue must be capitalized.
        c_prev = {
            "idx": 1, "start_ms": 0, "end_ms": 1500, "type": "dialogue",
            "lines": ['He said "go home."'],
            "meta": {"dialogue_text": 'He said "go home."',
                     "runs": [{"speaker": "A", "word_start": 0}]},
        }
        # New sentence, but arrives lowercased ("none of them are.") — the pass
        # must raise it because the prior cue DID end a sentence.
        c_new = {
            "idx": 2, "start_ms": 1600, "end_ms": 3000, "type": "dialogue",
            "lines": ["none of them are."],
            "meta": {"dialogue_text": "none of them are.",
                     "runs": [{"speaker": "A", "word_start": 0}]},
        }
        out = apply_sentence_capitalization([c_prev, c_new])
        new_body = (out[1].get("meta") or {}).get("dialogue_text") or out[1]["lines"][0]
        assert new_body[:1] == "N", f"sentence start not capitalized: {new_body!r}"
    finally:
        _restore(prev)


# ════════════════════════════════════════════════════════════════════════════
# CLASS F — cue endings on forward-binding function words (item 4)
# Invariant: a cue should not end on a determiner/article/preposition when a
# compliant alternative boundary exists (strong SOFT penalty, phrase-aware).
# (§2 Class F.)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.xfail(strict=True, reason="Class F: forward-binding tail penalty lands in Phase 2")
def test_class_f_cue_does_not_end_on_forward_binding_word():
    prev = _set_env(**_SPEC)
    try:
        # "For players and fans alike, the game is a crucible of character."
        # A boundary after "the" is compliant, but so is a boundary after "alike,"
        # (a clause end) — the engine must prefer the clause boundary and keep
        # "the game" bound together.
        wt = [
            ("For", 0, 250), ("players", 250, 700), ("and", 700, 900),
            ("fans", 900, 1250), ("alike,", 1250, 1700), ("the", 1700, 1900),
            ("game", 1900, 2250), ("is", 2250, 2450), ("a", 2450, 2550),
            ("crucible", 2550, 3100), ("of", 3100, 3250),
            ("character.", 3250, 3900),
        ]
        cue = _cue(wt)
        cue["start_ms"], cue["end_ms"] = 0, 3900
        out = optimize_cue_sequence([cue])

        # INVARIANT: no emitted cue (except the last) ends on a forward-binding
        # function word when a clause boundary was available.
        for i, c in enumerate(out[:-1]):
            last = _words(c)[-1].rstrip(".,;:!?").lower()
            assert last not in _FORWARD_BINDING_TAIL, \
                f"cue {i} ends on forward-binding word {last!r}: {_text(c)!r}"
    finally:
        _restore(prev)


# ════════════════════════════════════════════════════════════════════════════
# CLASS G — repeated speaker labels without a valid reintroduction (item 5)
# Invariant: consecutive same-speaker cues do not repeat the label, INCLUDING
# across an intervening unknown/sound cue. (§2 Class G — reproduce G1/G2 in Ph3.)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.xfail(strict=True, reason="Class G: reproduced + hardened in Phase 3 (G1/G2 to be pinned)")
def test_class_g_same_speaker_label_not_repeated_across_unknown_cue():
    prev = _set_env(**_SPEC)
    os.environ["SPEAKER_LABEL_MODE"] = "named"
    try:
        from services.rendering import suppress_repeat_speaker_labels

        def _named(text, speaker, start, end):
            return {
                "idx": 0, "start_ms": start, "end_ms": end, "type": "dialogue",
                "lines": [f"[{speaker.upper()}:] {text}"],
                "meta": {"dialogue_text": text,
                         "runs": [{"speaker": speaker, "word_start": 0}]},
            }

        # A, then an UNKNOWN-speaker cue (no runs speaker), then A again. The
        # second A must NOT repeat its label — the unknown cue must not reset the
        # same-speaker turn run.
        cA1 = _named("The game begins now.", "A", 0, 1500)
        cUnknown = {
            "idx": 0, "start_ms": 1600, "end_ms": 2400, "type": "dialogue",
            "lines": ["(indistinct chatter)"],
            "meta": {"dialogue_text": "(indistinct chatter)",
                     "runs": [{"speaker": None, "word_start": 0}], "review_required": True},
        }
        cA2 = _named("And so it ends.", "A", 2500, 4000)
        out = suppress_repeat_speaker_labels([cA1, cUnknown, cA2])

        # INVARIANT: the returning-A cue does not re-emit the [A:] label.
        a2_line = out[2]["lines"][0]
        assert not a2_line.lstrip().startswith("[A:]"), \
            f"same speaker re-labeled across an unknown cue: {a2_line!r}"
    finally:
        _restore(prev)


# ════════════════════════════════════════════════════════════════════════════
# CLASS H — inappropriate merging of short complete stand-alone utterances (item 6)
# Invariant: a short complete stand-alone utterance that independently clears the
# hard constraints is allowed to remain its own cue. (§2 Class H.)
# ════════════════════════════════════════════════════════════════════════════
@pytest.mark.xfail(strict=True, reason="Class H: stand-alone-unit protection lands in Phase 2")
def test_class_h_short_complete_utterance_not_merged():
    prev = _set_env(**_SPEC)
    try:
        # "Agatha Schnitzler." is a complete stand-alone name address that clears
        # min_display + CPS on its own. It must not be merged with the following
        # DIFFERENT-thought sentence from the same speaker.
        wt_name = [("Agatha", 0, 700), ("Schnitzler.", 700, 1600)]
        wt_next = [
            ("Are", 1700, 1950), ("you", 1950, 2150), ("trying", 2150, 2600),
            ("to", 2600, 2750), ("set", 2750, 3050), ("me", 3050, 3250),
            ("up", 3250, 3500), ("again?", 3500, 4100),
        ]
        cName = _cue(wt_name)
        cName["start_ms"], cName["end_ms"] = 0, 1600   # 1.6s ≥ min_display, clears CPS
        cNext = _cue(wt_next)
        cNext["start_ms"], cNext["end_ms"] = 1700, 4100
        out = optimize_cue_sequence([cName, cNext])

        # INVARIANT: the name utterance is not fused into the question cue.
        first = _text(out[0]).strip()
        assert "Schnitzler." in first and "trying" not in first, \
            f"stand-alone utterance was merged: {first!r}"
        assert len(out) >= 2, "stand-alone utterance collapsed into neighbor"
    finally:
        _restore(prev)
