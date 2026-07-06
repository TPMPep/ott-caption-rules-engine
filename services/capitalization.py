"""
Sentence-boundary capitalization — the DETERMINISTIC final-authority pass.

WHY THIS MODULE EXISTS
──────────────────────
Cue text is assembled from raw provider transcripts and then reshaped by the
shaper (which splits a long sentence across cues at clause boundaries). Neither
step guarantees the ONE thing a professional captioner always gets right:

  • A cue that CONTINUES the previous sentence must keep its first word
    lowercase ("...give me 25 laps around" / "this place, then hit the lockers.").
    The shaper's split had left it capitalized ("This place") — a mid-sentence
    capital, which reads as a new sentence that isn't one.

  • A cue that STARTS a new sentence must capitalize its first word
    ("Good work today, Hart." / "You've been a big help."). The raw transcript
    had left "you've" lowercase — an un-capitalized sentence start.

This is DETERMINISTIC GRAMMAR, not an editorial judgment call. It must therefore
NOT depend on the optional, best-effort editorial-AI (which is skipped when no
key is set, bails under load, and runs per-cue in isolation with only raw
neighbor text). Capitalization correctness for an A++++ deliverable cannot ride
on an LLM that can silently bail at 100-user concurrency.

WHERE IT RUNS (formatter.py pipeline)
─────────────────────────────────────
LAST — after shaping, editorial-AI, readability, condensation, and label
suppression, immediately before QC. Running last makes it the FINAL AUTHORITY:
whatever any upstream stage did (or didn't do, when the AI bailed), the
delivered text has correct sentence-boundary casing. Idempotent — a second run
changes nothing.

THE TWO RULES (applied per cue, in output order)
────────────────────────────────────────────────
Walking cues in order, we track whether the PREVIOUS dialogue cue ended a
sentence (abbreviation-aware, via segmentation.is_sentence_end on its last
word — so "...Mr." is NOT treated as a sentence end):

  1. CONTINUATION  (prev cue did NOT end a sentence)  → LOWERCASE this cue's
     first word, but ONLY with POSITIVE evidence it is safe. Protection (never
     downcase): "I"/contractions, ALL-CAPS acronyms ("FBI"), words proven
     proper by a MID-CUE capitalized occurrence elsewhere ("Hart"), and any
     word following an abbreviation/initial ("Mr." → "Wang"). Lowercasing
     requires proof the word is common: membership in the static function-word
     set ("this", "then", "at", …) or a lowercase occurrence elsewhere in the
     same output. An ambiguous word with no evidence either way is LEFT AS-IS
     (a stray capital reads better than a downcased name). Cue-INITIAL capitals
     are never harvested as proper-noun evidence — that's exactly the defect
     class this pass fixes, and it must not launder itself into proof.

  2. SENTENCE START (prev cue DID end a sentence, or this is cue #1) →
     CAPITALIZE this cue's first word (unless it's already a bracketed sound
     cue or starts with punctuation/a number).

Sound / music / non-dialogue cues are never touched and never advance or reset
the sentence-continuation state incorrectly — a bracketed [MUSIC] cue between
two dialogue halves must NOT make the second half look like a new sentence, so
non-dialogue cues are transparent to the continuation tracker.

Speaker labels are preserved verbatim: the rule targets the first REAL WORD of
the cue body, after any leading "- " / "[NAME:]" / "A:" label.

Pure functions only — no env reads, no I/O, deterministic. SOC 2 CC8.1 —
identical cues always yield identical casing, reproducible by an auditor.
"""

import re
from typing import Any, Dict, List, Optional, Set

try:
    from .segmentation import is_sentence_end as _is_sentence_end
except Exception:  # pragma: no cover — defensive for alternate import roots
    def _is_sentence_end(word, next_word=None):
        w = (word or "").strip()
        return bool(w) and w[-1] in (".", "!", "?", "。", "！", "？")

try:
    from .cjk import is_cjk_text as _is_cjk
except Exception:  # pragma: no cover
    def _is_cjk(text):
        return False


# Leading-label matcher — identical shape to condensation._strip_leading_label
# so the two stages agree byte-for-byte on where a label ends and the body
# begins. Matches "- ", "[NAME:] ", "NAME: ".
_LABEL_RE = re.compile(r"^\s*(?:-\s+|\[[^\]]*\]:?\s*|[^\s:]{1,24}:\s+)")

# Pronoun "I" and its contractions — ALWAYS stay capitalized, even mid-sentence.
_ALWAYS_CAP_LOWER = {"i", "i'm", "i've", "i'll", "i'd"}

# Terminal punctuation that closes a sentence (mirrors segmentation).
_TERMINALS = (".", "!", "?", "…", "。", "！", "？")

# Closing quotes/brackets that can trail terminal punctuation ('win."', 'done!)').
# The sentence-end check must see through these — otherwise a mid-cue sentence
# that closes with a quote poisons the proper-noun harvest (the word after it
# looks "mid-sentence capitalized") and the continuation tracker misreads a
# quote-closed cue as unfinished.
_TRAILING_CLOSERS = "\"'»”’)]}"


def _ends_sentence(token: str) -> bool:
    """Quote-aware sentence-end check: strip trailing closing quotes/brackets,
    then defer to the ONE shared abbreviation-aware primitive. 'state."' → True;
    'Mr.' → False; 'ago,' → False. Used by both the evidence harvest and the
    continuation tracker so they can never disagree."""
    w = (token or "").strip().rstrip(_TRAILING_CLOSERS)
    return _is_sentence_end(w)


def _split_label(text: str):
    """Return (label, body). Label is the verbatim leading speaker tag (or '')."""
    m = _LABEL_RE.match(text or "")
    if m:
        return text[:m.end()], text[m.end():]
    return "", (text or "")


def _cue_text(cue: Dict[str, Any]) -> str:
    meta = cue.get("meta") or {}
    return (meta.get("dialogue_text") or " ".join(cue.get("lines", []))).strip()


def _bare(word: str) -> str:
    """Word stripped of surrounding punctuation, for classification."""
    return (word or "").strip(".,!?;:—–…\"'()[]{}«»“”‘’").strip()


def _is_acronym(bare_word: str) -> bool:
    """ALL-CAPS token of length ≥2 with no lowercase letter — 'FBI', 'NASA',
    'U.S' (after bare-strip). Never lowercased (it's not common-word-cased)."""
    stripped = bare_word.replace(".", "")
    return len(stripped) >= 2 and stripped.isupper() and stripped.isalpha()


# Common English words that are SAFE to lowercase at a continuation start —
# function words + demonstratives + pronouns + auxiliaries + frequent adverbs.
# The shaper's clause-boundary splits overwhelmingly begin the right half with
# one of these ("this place", "then hit", "at the", "because of"). A word NOT in
# this set and with no lowercase evidence elsewhere is left as-is (conservative:
# a wrong mid-sentence capital is less damaging than downcasing a name).
_COMMON_WORDS = frozenset({
    # articles / determiners / demonstratives
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "each",
    "every", "no", "all", "both", "either", "neither", "such", "another",
    # pronouns (possessives + objects included; 'i' handled by _ALWAYS_CAP_LOWER)
    "he", "she", "it", "we", "they", "you", "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours",
    "theirs", "who", "whom", "whose", "which", "what", "there", "here",
    "it's", "he's", "she's", "we're", "they're", "you're", "we've", "they've",
    "you've", "we'll", "they'll", "you'll", "he'll", "she'll", "he'd", "she'd",
    "we'd", "they'd", "you'd", "that's", "there's", "here's", "what's", "who's",
    # prepositions / conjunctions
    "and", "but", "or", "nor", "so", "yet", "for", "of", "in", "on", "at", "to",
    "by", "with", "from", "into", "onto", "over", "under", "about", "after",
    "before", "between", "through", "during", "against", "without", "within",
    "because", "although", "though", "while", "when", "where", "if", "unless",
    "until", "since", "as", "than", "like", "unlike", "toward", "towards",
    # auxiliaries / copulas
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does",
    "did", "have", "has", "had", "will", "would", "can", "could", "shall",
    "should", "may", "might", "must", "won't", "don't", "doesn't", "didn't",
    "can't", "couldn't", "wouldn't", "shouldn't", "isn't", "aren't", "wasn't",
    "weren't", "haven't", "hasn't", "hadn't",
    # frequent adverbs / discourse words
    "then", "now", "just", "not", "never", "always", "still", "even", "only",
    "also", "too", "very", "really", "again", "back", "away", "out", "up",
    "down", "off", "how", "why", "maybe", "perhaps", "please", "well", "okay",
})


def _collect_case_evidence(cue_texts: List[str]):
    """Two DETERMINISTIC evidence sets from the finished output:

    proper_nouns  — words seen CAPITALIZED at a MID-CUE position (i>0 within a
        cue body) where the preceding word in the SAME cue did not end a
        sentence. This is trustworthy evidence: mid-cue capitals are never the
        shaper-split defect (which only mis-capitalizes CUE-INITIAL words), so a
        mid-cue capital genuinely marks a proper noun ("with Hart and").
        CRITICALLY we do NOT harvest cue-INITIAL capitals as evidence — a
        continuation cue's wrong capital ("This place") is exactly the defect
        this pass exists to fix, and must never launder itself into proof.

    lowercase_seen — words seen fully lowercase anywhere. A word proven common
        in this content is safe to downcase at a continuation start even when
        it's not in the static _COMMON_WORDS set.

    Both lower-cased for O(1) membership tests. SOC 2 CC8.1 — evidence is a pure
    function of the cue texts, reproducible by an auditor."""
    proper: Set[str] = set()
    lowercase_seen: Set[str] = set()
    for text in cue_texts:
        _, body = _split_label(text)
        tokens = body.split()
        for i, tok in enumerate(tokens):
            core = _bare(tok)
            if not core or _is_cjk(core):
                continue
            low = core.lower()
            if core[0].islower():
                lowercase_seen.add(low)
                continue
            if not core[0].isupper() or low in _ALWAYS_CAP_LOWER:
                continue
            # CLOSED-CLASS GUARD: pronouns/function words ("they", "this", "and")
            # are NEVER proper nouns in English — a capitalized mid-cue occurrence
            # is itself a casing artifact (ellipsis continuation, quote-closed
            # sentence, transcript noise), and one such occurrence must not
            # globally protect the word across the whole file. This closed the
            # real-world bug where a single stray "They" capital protected every
            # continuation "They" in a 1000-cue show.
            if low in _COMMON_WORDS:
                continue
            if i > 0 and not _ends_sentence(tokens[i - 1]):
                proper.add(low)
    return proper, lowercase_seen


def _prev_word_is_abbreviation(prev_last_word: str) -> bool:
    """True when the previous cue's last word ends with '.' but is NOT a genuine
    sentence end (per the abbreviation-aware segmentation rule) — i.e. it is an
    abbreviation or initial ("Mr.", "Dr.", "J."). The word FOLLOWING such a token
    is almost certainly a proper noun ("Mr. / Wang") and must never be downcased,
    even with no other evidence. Derived from the ONE shared sentence-end
    primitive, so it can never disagree with segmentation."""
    w = (prev_last_word or "").strip()
    return w.endswith(".") and not _is_sentence_end(w)


def _lowercase_first_word(
    body: str,
    proper_nouns: Set[str],
    lowercase_seen: Set[str],
    prev_last_word: str,
) -> str:
    """Lowercase the first word of `body` ONLY when it is provably safe.

    PROTECT (never downcase): "I"/contractions, ALL-CAPS acronyms, CJK, proven
    proper nouns (mid-cue capitalized evidence), and any word that follows an
    abbreviation/initial ("Mr." → "Wang").

    LOWERCASE only with POSITIVE evidence the word is common: it is in the
    static function-word set (_COMMON_WORDS) OR it appears lowercase elsewhere
    in this output (lowercase_seen). An ambiguous word with no evidence either
    way is LEFT AS-IS — conservative: a stray mid-sentence capital reads better
    than a downcased name. Deterministic + auditor-reproducible."""
    if not body:
        return body
    m = re.match(r"^(\s*)(\S+)(.*)$", body, re.DOTALL)
    if not m:
        return body
    lead_ws, first, rest = m.group(1), m.group(2), m.group(3)
    core = _bare(first)
    if not core or not core[0].isalpha():
        return body  # starts with punctuation / number — leave it
    if _is_cjk(core):
        return body  # CJK has no case
    if not first[0:1].isupper():
        return body  # already lowercase — nothing to do
    low = core.lower()
    if low in _ALWAYS_CAP_LOWER:
        return body
    if _is_acronym(core):
        return body
    if low in proper_nouns:
        return body  # proven proper noun — never downcase
    if _prev_word_is_abbreviation(prev_last_word):
        return body  # follows "Mr." / "Dr." / an initial — a name
    # Positive-evidence gate: only downcase a word we can PROVE is common.
    if low not in _COMMON_WORDS and low not in lowercase_seen:
        return body  # ambiguous — conservative keep
    return lead_ws + first[0].lower() + first[1:] + rest


def _capitalize_first_word(body: str) -> str:
    """Capitalize the first alphabetic word of `body` (sentence start). Leaves a
    bracketed cue / punctuation / number lead untouched."""
    if not body:
        return body
    m = re.match(r"^(\s*)(\S+)(.*)$", body, re.DOTALL)
    if not m:
        return body
    lead_ws, first, rest = m.group(1), m.group(2), m.group(3)
    # Find the first alphabetic char in `first` (skip an opening quote/bracket).
    for i, ch in enumerate(first):
        if ch.isalpha():
            if ch.isupper() or _is_cjk(ch):
                return body  # already capitalized (or CJK, no case)
            return lead_ws + first[:i] + ch.upper() + first[i + 1:] + rest
        if ch.isdigit():
            return body  # starts with a number — leave it
    return body


def _apply_body(cue: Dict[str, Any], new_body: str, old_body: str) -> None:
    """Write `new_body` back onto the cue's lines + meta.dialogue_text, preserving
    the leading label and the existing line-break structure. Only the FIRST word
    changed, so we re-map the first line's body without re-wrapping (the geometry
    is untouched — casing never changes line length)."""
    if new_body == old_body:
        return
    label, _ = _split_label(_cue_text(cue))
    lines = cue.get("lines") or []
    if lines:
        # The first word lives on line 0. Rebuild line 0 = label + first-line-body
        # with only the leading word recased; every other line is unchanged.
        first_line = lines[0]
        f_label, f_body = _split_label(first_line)
        # Recase the leading word of the first line's body the same way.
        recased_first = _recased_first_line_body(f_body, old_body, new_body)
        new_lines = [f_label + recased_first] + list(lines[1:])
        cue["lines"] = new_lines
    meta = dict(cue.get("meta") or {})
    body_label, _ = _split_label(meta.get("dialogue_text") or "")
    meta["dialogue_text"] = (body_label + new_body).strip() if body_label else new_body
    cue["meta"] = meta


def _recased_first_line_body(first_line_body: str, old_full_body: str, new_full_body: str) -> str:
    """Apply the same first-word case change that turned old_full_body →
    new_full_body onto the first visible line's body. Because only the leading
    word changed and it lives on line 1, we recase the first line's leading word
    directly (robust to the line body differing from the full dialogue text)."""
    if not first_line_body:
        return first_line_body
    # Determine the case operation from the full-body delta.
    old_first = old_full_body.split()[0] if old_full_body.split() else ""
    new_first = new_full_body.split()[0] if new_full_body.split() else ""
    if not old_first or old_first == new_first:
        return first_line_body
    m = re.match(r"^(\s*)(\S+)(.*)$", first_line_body, re.DOTALL)
    if not m:
        return first_line_body
    lead_ws, word, rest = m.group(1), m.group(2), m.group(3)
    # Match on the bare word to be safe against differing trailing punctuation.
    if _bare(word).lower() != _bare(old_first).lower():
        return first_line_body
    # Apply the same first-alpha-char case flip.
    made_upper = new_first[:1].isupper() and not old_first[:1].isupper()
    for i, ch in enumerate(word):
        if ch.isalpha():
            flipped = ch.upper() if made_upper else ch.lower()
            return lead_ws + word[:i] + flipped + word[i + 1:] + rest
        if ch.isdigit():
            return first_line_body
    return first_line_body


def _sync_first_line_case(cue: Dict[str, Any], decided_body: str) -> bool:
    """DEFENSE-IN-DEPTH: force the DELIVERED first line's leading word to carry
    the same case as the decided body. The casing decision above is made on
    meta.dialogue_text, but what ships is lines[] — if any upstream stage
    recased a rendered line without updating dialogue_text (the 'This place'
    defect), the decision would silently not reach the screen. This reconciler
    makes lines[0] authoritative-consistent with the decided body even when the
    body itself needed no change. Returns True when a line was corrected."""
    words = decided_body.split()
    lines = cue.get("lines") or []
    if not words or not lines:
        return False
    decided_first = words[0]
    f_label, f_body = _split_label(lines[0])
    m = re.match(r"^(\s*)(\S+)(.*)$", f_body, re.DOTALL)
    if not m:
        return False
    lead_ws, word, rest = m.group(1), m.group(2), m.group(3)
    # Only reconcile when it IS the same word (bare, case-insensitive) — never
    # touch a line that diverged in content rather than case.
    if _bare(word).lower() != _bare(decided_first).lower():
        return False
    want_upper = None
    for ch in decided_first:
        if ch.isalpha():
            want_upper = ch.isupper()
            break
    if want_upper is None:
        return False
    for i, ch in enumerate(word):
        if ch.isalpha():
            if ch.isupper() == want_upper:
                return False  # already consistent
            flipped = ch.upper() if want_upper else ch.lower()
            cue["lines"] = [f_label + lead_ws + word[:i] + flipped + word[i + 1:] + rest] + list(lines[1:])
            return True
        if ch.isdigit():
            return False
    return False


def apply_sentence_capitalization(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """DETERMINISTIC sentence-boundary capitalization — the final authority.

    For every DIALOGUE cue in output order:
      • if it continues the previous sentence → lowercase its first word
        (proper-noun-safe);
      • if it starts a new sentence (prev cue ended one, or it's the first
        dialogue cue) → capitalize its first word.

    Non-dialogue cues (music/SFX) are transparent: never modified, and they
    neither open nor close a sentence for the continuation tracker. Idempotent.
    Cue count preserved. SOC 2 CC8.1 — reproducible, no AI dependency."""
    if not cues:
        return cues

    dialogue_texts = [_cue_text(c) for c in cues if c.get("type") == "dialogue"]
    proper_nouns, lowercase_seen = _collect_case_evidence(dialogue_texts)

    prev_ended_sentence = True  # first dialogue cue starts a sentence
    prev_last_word = ""         # previous DIALOGUE cue's last word (abbrev guard)
    stats = {"lowered": 0, "raised": 0, "line_synced": 0}

    for cue in cues:
        if cue.get("type") != "dialogue":
            # Transparent — carry the sentence state across a sound cue unchanged.
            continue

        old_body_full = _cue_text(cue)
        _, body = _split_label(old_body_full)
        if not body.strip():
            continue

        if prev_ended_sentence:
            new_body = _capitalize_first_word(body)
            if new_body != body:
                stats["raised"] += 1
        else:
            new_body = _lowercase_first_word(body, proper_nouns, lowercase_seen, prev_last_word)
            if new_body != body:
                stats["lowered"] += 1

        if new_body != body:
            _apply_body(cue, new_body, body)
        # Defense-in-depth: the delivered line must carry the decided casing
        # even when dialogue_text was already correct (catches upstream
        # line/meta divergence — the 'This place' defect class).
        if _sync_first_line_case(cue, new_body):
            stats["line_synced"] += 1

        # Update the continuation tracker from THIS cue's (post-recase) last word.
        final_body = _cue_text(cue)
        _, fb = _split_label(final_body)
        prev_last_word = fb.split()[-1] if fb.split() else ""
        prev_ended_sentence = _ends_sentence(prev_last_word)

    print(f"[CAPITALIZATION] lowered_continuations={stats['lowered']} "
          f"raised_starts={stats['raised']} line_synced={stats['line_synced']}")
    return cues
