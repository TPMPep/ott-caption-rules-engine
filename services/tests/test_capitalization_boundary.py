"""
Regression: sentence-boundary capitalization must honour the immutable-boundary
primitive. A cue that opens across a hard wall (speaker change / inter-utterance
pause) is a NEW SENTENCE START — never a continuation of the previous cue — even
when the previous cue's text lacked terminal punctuation.

Real-world defect this pins (Pluto FAST, "Another Pluto Test"):
  cue 1  [SPEAKER A:] That's it. Chest pass. Chest pass   (no trailing period)
  cue 2  [SPEAKER B:] For the town of Everwood,

Before the fix the tracker saw cue 1 end on "pass" (no terminal punct) and
downcased cue 2's "For" → "for". A different speaker never continues the prior
speaker's sentence, so "For" must stay capitalized. Deterministic, no AI.
"""

from services.capitalization import apply_sentence_capitalization


def _dialogue(text, speaker, pause_before=False):
    return {
        "type": "dialogue",
        "lines": [text],
        "meta": {
            "dialogue_text": text,
            "runs": [{"speaker": speaker, "word_start": 0}],
            **({"pause_boundary_before": True} if pause_before else {}),
        },
    }


def test_speaker_change_forces_sentence_start_capitalization():
    cues = [
        _dialogue("[SPEAKER A:] That's it. Chest pass. Chest pass", "A"),
        _dialogue("[SPEAKER B:] For the town of Everwood,", "B"),
    ]
    out = apply_sentence_capitalization(cues)
    body2 = out[1]["meta"]["dialogue_text"]
    # The new speaker's first word must NOT have been downcased.
    assert "For the town" in body2, body2
    assert "for the town" not in body2, body2


def test_same_speaker_continuation_still_lowercases():
    # Control: within ONE speaker, a genuine continuation (prev cue did not end a
    # sentence) still lowercases a common leading word — the fix must not break it.
    cues = [
        _dialogue("[SPEAKER A:] I ran across", "A"),
        _dialogue("[SPEAKER A:] The street quickly.", "A"),
    ]
    out = apply_sentence_capitalization(cues)
    body2 = out[1]["meta"]["dialogue_text"]
    assert "the street" in body2, body2


def test_pause_boundary_forces_sentence_start():
    # A hard inter-utterance pause (same speaker) is also an immutable boundary →
    # the cue after it starts a new sentence.
    cues = [
        _dialogue("[SPEAKER A:] I ran across", "A"),
        _dialogue("[SPEAKER A:] The street quickly.", "A", pause_before=True),
    ]
    out = apply_sentence_capitalization(cues)
    body2 = out[1]["meta"]["dialogue_text"]
    assert "The street" in body2, body2


def test_idempotent():
    cues = [
        _dialogue("[SPEAKER A:] That's it. Chest pass", "A"),
        _dialogue("[SPEAKER B:] For the town of Everwood,", "B"),
    ]
    once = apply_sentence_capitalization(cues)
    twice = apply_sentence_capitalization(once)
    assert [c["meta"]["dialogue_text"] for c in once] == [c["meta"]["dialogue_text"] for c in twice]
