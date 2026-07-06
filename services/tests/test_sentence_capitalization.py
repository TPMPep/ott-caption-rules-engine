"""
Regression tests for services/capitalization.apply_sentence_capitalization —
the deterministic sentence-boundary casing authority.

Locks the two exact defects the operator reported:
  • Bug 1 (0031→0032): a sentence split across cues must keep the continuation
    cue's first word LOWERCASE ("this place", not "This place").
  • Bug 2 (0034): a cue that starts a new sentence must have its first word
    CAPITALIZED ("You've", not "you've").

Plus the proper-noun-safety invariant (never downcase "I", an acronym, or a
proven proper noun), sound-cue transparency, and idempotency.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.capitalization import apply_sentence_capitalization  # noqa: E402


def _dialogue(text, start=0, end=2000):
    """Build a minimal dialogue cue the way the formatter does."""
    return {
        "idx": 0,
        "start_ms": start,
        "end_ms": end,
        "lines": text.split("\n"),
        "type": "dialogue",
        "meta": {"dialogue_text": text.replace("\n", " ")},
    }


def _sound(text="[MUSIC]", start=0, end=1000):
    return {"idx": 0, "start_ms": start, "end_ms": end, "lines": [text], "type": "music", "meta": {}}


def _body(cue):
    return (cue.get("meta") or {}).get("dialogue_text") or " ".join(cue.get("lines", []))


# ── Bug 1 — continuation cue keeps first word lowercase ─────────────────
def test_continuation_first_word_lowercased():
    cues = [
        _dialogue("Okay, give me 25 laps around", 0, 2300),
        _dialogue("This place, then hit the lockers.", 2300, 3900),
    ]
    apply_sentence_capitalization(cues)
    # First cue does not end a sentence → second cue continues it → "this".
    assert _body(cues[1]).startswith("this place"), _body(cues[1])
    # The line text must also be recased, not just the dialogue_text.
    assert cues[1]["lines"][0].startswith("this place"), cues[1]["lines"]


# ── Bug 2 — new sentence start capitalized ──────────────────────────────
def test_new_sentence_start_capitalized():
    cues = [
        _dialogue("Good work today, Hart.", 0, 2000),
        _dialogue("you've been a big help.", 2000, 4000),
    ]
    apply_sentence_capitalization(cues)
    # First cue ends a sentence → second cue is a new sentence → "You've".
    assert _body(cues[1]).startswith("You've"), _body(cues[1])
    assert cues[1]["lines"][0].startswith("You've"), cues[1]["lines"]


# ── Proper-noun safety — never downcase a proven proper noun ────────────
def test_continuation_preserves_proper_noun():
    # "Hart" appears capitalized mid-sentence in cue 0 → proven proper noun.
    # Even as a continuation lead in cue 1, it must stay capitalized.
    cues = [
        _dialogue("I spoke with Hart and", 0, 2000),
        _dialogue("Hart agreed to help.", 2000, 4000),
    ]
    apply_sentence_capitalization(cues)
    assert _body(cues[1]).startswith("Hart"), _body(cues[1])


def test_continuation_preserves_pronoun_I():
    cues = [
        _dialogue("She said that", 0, 2000),
        _dialogue("I would be fine.", 2000, 4000),
    ]
    apply_sentence_capitalization(cues)
    assert _body(cues[1]).startswith("I would"), _body(cues[1])


def test_continuation_preserves_acronym():
    cues = [
        _dialogue("He works for the", 0, 2000),
        _dialogue("FBI in Washington.", 2000, 4000),
    ]
    apply_sentence_capitalization(cues)
    assert _body(cues[1]).startswith("FBI"), _body(cues[1])


# ── Sound-cue transparency — a music cue between halves is invisible ────
def test_sound_cue_transparent_to_continuation():
    cues = [
        _dialogue("give me 25 laps around", 0, 2000),
        _sound("[MUSIC]", 2000, 2500),
        _dialogue("This place, then hit the lockers.", 2500, 4000),
    ]
    apply_sentence_capitalization(cues)
    # The [MUSIC] cue must NOT reset the sentence — the dialogue after it still
    # continues the sentence, so "this" stays lowercase.
    assert _body(cues[2]).startswith("this place"), _body(cues[2])
    # And the sound cue itself is untouched.
    assert cues[1]["lines"] == ["[MUSIC]"]


# ── Abbreviation awareness — "Mr." is not a sentence end ────────────────
def test_abbreviation_not_treated_as_sentence_end():
    cues = [
        _dialogue("You must be Mr.", 0, 2000),
        _dialogue("Wang, welcome.", 2000, 4000),
    ]
    apply_sentence_capitalization(cues)
    # "Mr." does NOT end a sentence → "Wang" continues it, but "Wang" is a
    # proper noun (capitalized) so it stays capitalized (proper-noun safety wins).
    assert _body(cues[1]).startswith("Wang"), _body(cues[1])


# ── First dialogue cue always starts a sentence ─────────────────────────
def test_first_cue_capitalized():
    cues = [_dialogue("okay, let's begin.", 0, 2000)]
    apply_sentence_capitalization(cues)
    assert _body(cues[0]).startswith("Okay"), _body(cues[0])


# ── Speaker label preserved verbatim ────────────────────────────────────
def test_speaker_label_preserved():
    cues = [
        _dialogue("- Good work today, Hart.", 0, 2000),
        _dialogue("- you've been a big help.", 2000, 4000),
    ]
    apply_sentence_capitalization(cues)
    assert _body(cues[1]).startswith("- You've"), _body(cues[1])
    assert cues[1]["lines"][0].startswith("- You've"), cues[1]["lines"]


# ── Closed-class pronouns are never protected as proper nouns ────────────
def test_common_pronoun_never_poisoned_by_stray_midcue_capital():
    # A stray mid-cue capitalized "They" (after a quote-closed sentence the
    # naive check can't see through) must NOT protect the pronoun globally —
    # the continuation "They" in cue 2 must still be lowercased.
    cues = [
        _dialogue('He said "we win." They believed him.', 0, 3000),
        _dialogue("and for one glorious season,", 3000, 5000),
        _dialogue("They ruled the state.", 5000, 7000),
    ]
    apply_sentence_capitalization(cues)
    assert _body(cues[2]).startswith("they"), _body(cues[2])


# ── Quote-closed sentence end drives the continuation tracker ────────────
def test_quote_closed_cue_starts_new_sentence():
    cues = [
        _dialogue('She shouted "we won the game."', 0, 2000),
        _dialogue("nobody could believe it.", 2000, 4000),
    ]
    apply_sentence_capitalization(cues)
    assert _body(cues[1]).startswith("Nobody"), _body(cues[1])


# ── Conservative ambiguity — unknown capitalized word stays as-is ───────
def test_ambiguous_name_left_capitalized():
    # "Kowalski" appears ONLY at the continuation start — no mid-cue capitalized
    # evidence, no lowercase occurrence, not a function word. Conservative rule:
    # leave it capitalized rather than risk downcasing a name.
    cues = [
        _dialogue("Give the report to", 0, 2000),
        _dialogue("Kowalski before you leave.", 2000, 4000),
    ]
    apply_sentence_capitalization(cues)
    assert _body(cues[1]).startswith("Kowalski"), _body(cues[1])


# ── Lowercase evidence — a word seen lowercase elsewhere is downcased ────
def test_lowercase_evidence_enables_downcase():
    # "Suddenly" isn't in the static common-word set, but it appears lowercase
    # in another cue — positive evidence it's a common word here → downcase.
    cues = [
        _dialogue("It happened so suddenly last night.", 0, 2000),
        _dialogue("Everything went dark and", 2000, 4000),
        _dialogue("Suddenly the lights came back.", 4000, 6000),
    ]
    apply_sentence_capitalization(cues)
    assert _body(cues[2]).startswith("suddenly"), _body(cues[2])


# ── Idempotency — a second pass changes nothing ─────────────────────────
def test_idempotent():
    cues = [
        _dialogue("Okay, give me 25 laps around", 0, 2300),
        _dialogue("This place, then hit the lockers.", 2300, 3900),
    ]
    apply_sentence_capitalization(cues)
    first = [_body(c) for c in cues]
    apply_sentence_capitalization(cues)
    second = [_body(c) for c in cues]
    assert first == second
