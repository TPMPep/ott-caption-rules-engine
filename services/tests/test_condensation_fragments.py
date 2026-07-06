"""
Regression tests — fragment-safe condensation (engine 5.21.x).

Locks in the three root-cause fixes from the 'Keep playing like this' defect:
  1. Comparative 'like' is NEVER stripped as filler; only comma-bracketed
     filler 'like' is removed. 'ya' / 'also' are never dropped.
  2. The merge-then-condense pass merges same-speaker sentence-continuation
     fragments into ONE cue, extends into the trailing idle gap (ceiling
     division — no off-by-a-few-ms rejection), and accepts only a fully
     compliant result.
  3. Merge never fires across different speakers or after a completed
     sentence — those are genuinely separate captions.

All tests are deterministic (client=None — no LLM). SOC 2 CC8.1: identical
inputs always produce identical outputs, reproducible by an auditor.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.condensation import (  # noqa: E402
    remove_disfluencies,
    _merge_condense_continuations,
)

ENV_KEYS = [
    "CONDENSATION_MODE", "CUSTOM_MAX_CPS", "CUSTOM_MAX_LINES",
    "CUSTOM_MAX_CHARS", "CUSTOM_MAX_DISPLAY_MS", "CUSTOM_MERGE_GAP_MS",
    "CONDENSATION_MERGE_MAX_GAP_MS", "CPS_MEASUREMENT", "SPEAKER_LABEL_MODE",
]


def _cue(text, start_ms, end_ms, speaker="A"):
    return {
        "idx": 0,
        "start_ms": start_ms,
        "end_ms": end_ms,
        "type": "dialogue",
        "lines": [text],
        "meta": {"dialogue_text": text,
                 "runs": [{"speaker": speaker, "word_start": 0}]},
    }


class CondensationFragmentTests(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in ENV_KEYS}
        os.environ["CUSTOM_MAX_CPS"] = "18"
        os.environ["CUSTOM_MAX_LINES"] = "2"
        os.environ["CUSTOM_MAX_CHARS"] = "32"
        os.environ["CUSTOM_MAX_DISPLAY_MS"] = "7000"
        os.environ["CUSTOM_MERGE_GAP_MS"] = "80"
        os.environ["SPEAKER_LABEL_MODE"] = "none"

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # ── 1. Filler safety ────────────────────────────────────────────
    def test_comparative_like_is_preserved(self):
        self.assertEqual(
            remove_disfluencies("Keep playing like this,"),
            "Keep playing like this,",
        )

    def test_comma_bracketed_like_is_removed(self):
        out = remove_disfluencies("I was, like, so mad.")
        self.assertNotIn("like", out.lower())
        self.assertIn("so mad", out)

    def test_ya_and_also_are_never_dropped(self):
        self.assertIn("ya", remove_disfluencies("See ya later tonight."))
        self.assertIn("also", remove_disfluencies("I also want the ball."))

    # ── 2. Merge-then-condense (deterministic path) ─────────────────
    def test_continuation_fragments_merge_and_extend(self):
        """An over-CPS left fragment + same-speaker continuation must merge
        into ONE cue whose window extends into the trailing idle gap, and the
        merged text must be the full joined sentence (nothing dropped)."""
        left = _cue("Keep playing like this,", 0, 700)          # 34 cps — over
        right = _cue("you might make the team.", 700, 2200)     # under
        follower = _cue("Go again.", 5000, 6000)                # big idle gap
        merged, count = _merge_condense_continuations(
            [left, right, follower], None, "m", None,
        )
        self.assertEqual(count, 1)
        self.assertEqual(len(merged), 2)
        cue = merged[0]
        self.assertEqual(
            cue["meta"]["dialogue_text"],
            "Keep playing like this, you might make the team.",
        )
        self.assertTrue(cue["meta"].get("continuation_merge"))
        # Extended past the original 2200ms end, never into the follower.
        self.assertGreater(cue["end_ms"], 2200)
        self.assertLessEqual(cue["end_ms"], 5000 - 80)
        # Ceiling-division guarantee: the merged cue is AT or UNDER max_cps.
        chars = len(cue["meta"]["dialogue_text"])
        dur_s = (cue["end_ms"] - cue["start_ms"]) / 1000.0
        self.assertLessEqual(chars / dur_s, 18.0 + 1e-9)

    def test_no_merge_across_speakers(self):
        left = _cue("Keep playing like this,", 0, 700, speaker="A")
        right = _cue("you might make the team.", 700, 2200, speaker="B")
        merged, count = _merge_condense_continuations(
            [left, right], None, "m", None,
        )
        self.assertEqual(count, 0)
        self.assertEqual(len(merged), 2)

    def test_no_merge_after_completed_sentence(self):
        left = _cue("This is not baseball.", 0, 700)  # ends a sentence
        right = _cue("You might make the team.", 700, 2200)
        merged, count = _merge_condense_continuations(
            [left, right], None, "m", None,
        )
        self.assertEqual(count, 0)
        self.assertEqual(len(merged), 2)

    def test_rejected_merge_keeps_originals_verbatim(self):
        """When no gap exists and the merged cue can't reach compliance
        deterministically, the ORIGINAL cues ship untouched — never a
        mangled fragment."""
        left = _cue("Keep playing like this and never ever stop moving,", 0, 700)
        right = _cue("because the girls' field hockey team is watching you.", 700, 1400)
        follower = _cue("Go.", 1480, 2400)  # no idle gap to extend into
        merged, count = _merge_condense_continuations(
            [left, right, follower], None, "m", None,
        )
        self.assertEqual(count, 0)
        self.assertEqual(merged[0]["meta"]["dialogue_text"],
                         "Keep playing like this and never ever stop moving,")
        self.assertEqual(merged[1]["meta"]["dialogue_text"],
                         "because the girls' field hockey team is watching you.")


if __name__ == "__main__":
    unittest.main()
