"""
Line-break intelligence — the deterministic "Universal Caption Builder" brain.

WHY THIS MODULE EXISTS
──────────────────────
Greedy left-to-right fill (the old wrap_text behaviour) packs line 1 to the
character brim, then dumps the rest on line 2. That satisfies max_chars but
routinely produces breaks a senior caption editor would reject:

    BAD  (greedy):                 BETTER (balanced + clause-aware):
    I went to the store because    I went to the store
    I needed milk.                 because I needed milk.

    The United                     The United States
    States government              government

This module makes the SINGLE break decision for a one-line-too-long string in
the common 2-line case. It is pure, deterministic, and parameterised only by
spec flags the ClosedCaptionSpec already declares:

    CUSTOM_PREFER_BALANCED_LINES   (default 1 / true)  → reward balanced lengths
    CUSTOM_PRESERVE_CLAUSE_INTEGRITY (default 1 / true)→ break on clause/phrase
                                                          boundaries, never mid-phrase

NO AI. NO I/O. Identical (text, max_chars) always returns the identical break.
SOC 2 CC8.1 — every line-break decision is reproducible and auditor-defensible.

CONTRACT: choose_two_line_break(text, max_chars) returns either
  • [line1, line2]  — the chosen split (both within max_chars), or
  • None            — no two-line split fits; caller falls back to greedy.

The caller (rendering.wrap_text) owns max_lines handling, speaker labels, and
the >2-line fallback. This module ONLY answers "where is the best single break
for two lines?" — the highest-frequency, highest-impact editorial decision.
"""

import os
import re

from .rules import get_rule as _rule_get
from typing import List, Optional


# ─── Spec flags (read once per call; cheap) ──────────────────────────
def _flag(name: str, default: bool) -> bool:
    raw = _rule_get(name)
    if raw is None or raw == "":
        return default
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def _prefer_balanced() -> bool:
    return _flag("CUSTOM_PREFER_BALANCED_LINES", True)


def _preserve_clause() -> bool:
    return _flag("CUSTOM_PRESERVE_CLAUSE_INTEGRITY", True)


# ─── Linguistic fact tables (baked-in English; NOT per-job knobs) ────
# These are linguistic constants, identical to how segmentation.py bakes in the
# abbreviation set. A spec NEVER overrides what an article or conjunction is.

# Words that must LEAD the next line, not trail the current one. Breaking right
# BEFORE these is good (they start the continuation); leaving them dangling at
# the end of a line is the classic "weak line ending" defect.
_LEADING_WORDS = {
    # Articles
    "a", "an", "the",
    # Coordinating + common subordinating conjunctions
    "and", "but", "or", "nor", "for", "so", "yet",
    "because", "although", "though", "while", "whereas", "since", "unless",
    "until", "if", "when", "whenever", "where", "wherever", "after", "before",
    "as", "that", "than", "whether",
    # Prepositions (high-frequency)
    "of", "to", "in", "on", "at", "by", "with", "from", "into", "onto",
    "upon", "about", "over", "under", "through", "between", "among",
    "during", "without", "within", "toward", "towards", "against",
    # Prepositions (second tier — previously missing; the '…25 laps around |
    # this place' defect: 'around' wasn't listed, so nothing penalized
    # stranding it at a line/cue end or splitting inside its phrase).
    # Deliberately EXCLUDES phrasal-verb particles (off/up/down/out) — those
    # legitimately end a line ('take off', 'sit down').
    "around", "across", "along", "behind", "beside", "besides", "beyond",
    "near", "past", "despite", "except", "inside", "outside", "underneath",
    "via", "amid", "atop",
    # Relative / possessive that bind tightly to what follows
    "which", "who", "whom", "whose",
}

# Possessive / determiner words that bind tightly to the FOLLOWING noun — never
# strand them at the end of a line either.
_DETERMINERS = {
    "my", "your", "his", "her", "its", "our", "their",
    "this", "that", "these", "those", "some", "any", "no", "every", "each",
}

# ─── CLOSED-CLASS GRAMMATICAL BONDS (progressive-confidence tier 3) ──────────
# These are HIGH-CONFIDENCE, CLOSED lexical classes — NOT a generic content-word
# heuristic and NOT an attempt at POS tagging. They encode two universal English
# bonds that a caption should not sever when a balance-equal alternative exists:
#
#   • FORWARD-BINDING words bind to what FOLLOWS them: a subject pronoun, an
#     auxiliary/copula, a modal, or a subject+aux contraction ("I'm", "Let's",
#     "we're", "he'll"). Stranding one at the END of line 1 severs it from the
#     verb/complement it introduces ("Then we" | "saw …", "Let's" | "get …").
#     Penalize a break whose line-1 LAST word is forward-binding.
#     NOTE — this governs the CLOSED-CLASS seam only (pronoun/aux/modal/
#     contraction). It deliberately does NOT address the verb→object seam
#     ("want" | "Cypress speed") — "want" is a content word, and inferring
#     verb→object cohesion would approximate POS tagging, which is out of scope.
#   • PARTICLES bind BACKWARD to a preceding verb (phrasal verbs: "get up",
#     "speed up", "back off"). Leading line 2 with a bare particle severs it from
#     its verb ("get" | "up …"). Penalize a break whose line-2 FIRST word is a
#     particle.
#
# Both are TIE-BREAKERS ONLY (weight below the sentence/clause rewards and the
# weak-ending/leading-word rules, above raw balance) — they never select an
# infeasible break, never override max_chars, and never beat a real punctuation
# boundary. Deterministic, casing-independent, reproducible. SOC 2 CC8.1.

# Subject pronouns + auxiliaries/copulas + modals that bind forward to a verb.
# The bare-word forms (contraction suffixes are matched separately below).
_FORWARD_BINDING = {
    # Subject pronouns
    "i", "we", "you", "he", "she", "they", "it",
    # Copula / auxiliary "be"
    "am", "is", "are", "was", "were", "be", "been", "being",
    # Auxiliary "do" / "have"
    "do", "does", "did", "have", "has", "had",
    # Modals
    "will", "would", "shall", "should", "can", "could",
    "may", "might", "must", "ought",
    # Common subject+aux / let-us contractions (whole-token forms)
    "let's", "lets", "i'm", "i've", "i'll", "i'd",
    "we're", "we've", "we'll", "we'd",
    "you're", "you've", "you'll", "you'd",
    "he's", "he'll", "he'd", "she's", "she'll", "she'd",
    "they're", "they've", "they'll", "they'd",
    "it's", "it'll", "that's", "there's", "here's",
}

# Phrasal-verb particles that bind BACKWARD to a preceding verb. Kept DISTINCT
# from the prepositions in _LEADING_WORDS: as a PARTICLE ("get up", "speed up")
# it must not lead a line; as a preposition it legitimately leads one ("up the
# hill"). Because a bare particle at a LINE START almost always continues a
# phrasal verb from the prior line in caption text, penalizing it here is the
# safe, high-confidence call.
_PARTICLES = {
    "up", "down", "out", "off", "in", "on", "over",
    "back", "away", "around", "through", "along", "apart",
}


def _is_forward_binding(word: str) -> bool:
    """True when `word` (contraction-aware) is a subject pronoun / auxiliary /
    modal / subject+aux contraction that binds FORWARD to a following verb."""
    b = _bare(word)
    if not b:
        return False
    # Keep the apostrophe for contraction matching (`_bare` preserves it).
    return b in _FORWARD_BINDING


def _is_particle(word: str) -> bool:
    """True when `word` is a phrasal-verb particle that binds BACKWARD to a
    preceding verb (never a good line-2 leader in caption text)."""
    return _bare(word) in _PARTICLES

_SENTENCE_PUNCT = (".", "!", "?")
_CLAUSE_PUNCT = (",", ";", ":", "—", "–")

# Strip trailing punctuation for word-class lookup.
_STRIP_RE = re.compile(r"[^\w']+$")


def _bare(word: str) -> str:
    return _STRIP_RE.sub("", (word or "")).strip().lower()


def _is_capitalized_cohesion_token(word: str) -> bool:
    """DETERMINISTIC CAPITALIZATION HEURISTIC — NOT named-entity recognition.

    This is a *capitalized phrase-cohesion* proxy, not true proper-noun
    recognition. It has NO dictionary, NO NER model, NO I/O — it reasons purely
    about the SHAPE of a token. It returns True when a token:
      • starts with an uppercase letter, AND
      • has at least one lowercase letter after position 0 (so an ALL-CAPS
        acronym like 'US' / 'FBI' / 'NASA' is deliberately EXCLUDED — those are
        single lexical units, not a splittable multi-word phrase), AND
      • is at least two characters, AND
      • is not the pronoun 'I', AND
      • is not a function word (article / conjunction / preposition / determiner
        — 'The' / 'And' at a sentence start must never be read as part of a
        capitalized phrase).

    KNOWN LIMITATIONS (documented on purpose — do NOT overstate this as NER):
      • It cannot tell a genuine proper name from any other capitalized word, so
        a MID-SENTENCE capitalized common word would also count. In practice the
        heuristic only ever fires as a −18 TIE-BREAK and only when line-1's last
        word does NOT close a sentence/clause, so a sentence-initial capital
        (the dominant false-positive source) is handled by the far stronger
        +32/+28 boundary rewards long before this nudge matters.
      • It is CAPITALIZATION-DEPENDENT: lowercased ASR output ('united states')
        carries no capitalization evidence, so the heuristic correctly does
        nothing — it neither helps nor harms.
      • It is ENGLISH/Latin-casing oriented. Scripts without comparable
        casing (CJK) never reach this path (rendering routes CJK to the
        character wrapper), and casing-divergent languages simply get no nudge.
      • A particle-led name ('de Gaulle') only coheres on the capitalized
        tokens; the lowercase particle 'de' is a function-word-shaped token and
        is scored by the ordinary leading/weak-ending rules, not here.

    It ONLY ever adds a tie-breaking PREFERENCE — it never forces an infeasible
    or worse-scoring break, never overrides a hard max_chars limit (an
    over-width split is rejected as None before scoring), and never beats a real
    sentence/clause boundary. Deterministic + reproducible. SOC 2 CC8.1."""
    raw = _STRIP_RE.sub("", (word or "")).strip()
    if len(raw) < 2 or not raw[0].isupper():
        return False
    if not any(c.islower() for c in raw[1:]):
        return False  # acronym (US, FBI) — not a multi-word proper-noun phrase
    b = raw.lower()
    if b == "i":
        return False
    return b not in _LEADING_WORDS and b not in _DETERMINERS


def _line_len(words: List[str], start: int, end: int) -> int:
    """Rendered length of words[start:end] joined by single spaces."""
    if start >= end:
        return 0
    seg = words[start:end]
    return sum(len(w) for w in seg) + (len(seg) - 1)


def _score_break(words: List[str], i: int, max_chars: int) -> Optional[float]:
    """
    Score a candidate break BETWEEN words[i-1] and words[i] (i.e. line 1 =
    words[0:i], line 2 = words[i:]). Higher is better. Returns None when the
    break is infeasible (either line exceeds max_chars).

    Scoring (all additive, deterministic):
      + balance       : reward small |len(line1) - len(line2)|
      + clause punct  : big reward if line 1's last word ends in , ; : — (break
                        AT a clause boundary reads naturally)
      + leading word  : reward if line 2's FIRST word is a conjunction/prep/
                        article (it correctly LEADS the continuation)
      - weak ending   : penalize if line 1's LAST word is an article / prep /
                        conjunction / determiner left dangling (the weak-ending
                        defect) — only when preserve_clause is on
    """
    n = len(words)
    len1 = _line_len(words, 0, i)
    len2 = _line_len(words, i, n)
    if len1 == 0 or len2 == 0:
        return None
    if len1 > max_chars or len2 > max_chars:
        return None

    score = 0.0

    # Balance — the further the two lines are from equal, the worse. Weight is
    # modest so it never overrides a clean clause break, but decisive between
    # otherwise-equal candidates. Disabled when the spec turns balance off.
    if _prefer_balanced():
        score -= abs(len1 - len2) * 1.0

    last_word = words[i - 1]
    first_word = words[i]
    last_bare = _bare(last_word)
    first_bare = _bare(first_word)

    if _preserve_clause():
        # Reward breaking right AFTER sentence/clause punctuation. A SENTENCE
        # boundary (. ! ?) is the STRONGEST possible line ending — a complete
        # grammatical unit — so it must outrank a mid-sentence CLAUSE boundary
        # (, ; :). The previous ordering (clause 30 > sentence 20) inverted this
        # and produced breaks like "Good work today," / "Hart. You've helped a
        # lot." — orphaning "Hart." from its own sentence. Ending a line on a
        # finished sentence and starting the next line with the next sentence is
        # the professional-captioner default. Sentence 32 > clause 28 so a full
        # stop always wins when both are feasible, but a lone clause break is
        # still strongly preferred over a mid-phrase break.
        if last_word.endswith(_SENTENCE_PUNCT):
            score += 32.0
        elif last_word.endswith(_CLAUSE_PUNCT):
            score += 28.0

        # Reward line 2 LEADING with a conjunction/preposition/article — the
        # canonical "I went to the store / because I needed milk" fix.
        if first_bare in _LEADING_WORDS:
            score += 15.0

        # Penalize a weak line-1 ending: a dangling article / prep / conjunction
        # / determiner that should have led the next line instead.
        if last_bare in _LEADING_WORDS or last_bare in _DETERMINERS:
            score -= 25.0

        # ── CLOSED-CLASS GRAMMATICAL-BOND COHESION (tier 3, tie-breaker) ─────
        # Subordinate to the sentence/clause rewards (+32/+28) and the weak-
        # ending/leading rules above — a real punctuation boundary or a dangling
        # function word always dominates. Weight −12 sits BELOW those and only
        # decides between otherwise-acceptable, balance-comparable candidates:
        #   • line-1 ends on a FORWARD-BINDING word (subject pronoun / aux /
        #     modal / "Let's"-type contraction) → it's severed from its verb.
        #     (Closed-class only — the verb→object seam is NOT in scope here.)
        #   • line-2 leads with a phrasal-verb PARTICLE → severed from its verb.
        # A break that would strand a bare function word already earns −25, so a
        # forward-binder that is ALSO a leading/determiner word isn't double-
        # penalized here (the stronger −25 covers it). This term catches the
        # class the weak-ending rule misses: pronouns/auxes/contractions/
        # particles that are NOT prepositions/articles.
        if last_bare not in _LEADING_WORDS and last_bare not in _DETERMINERS:
            if _is_forward_binding(last_word):
                score -= 12.0
        if first_bare not in _LEADING_WORDS and _is_particle(first_word):
            score -= 12.0

    # CAPITALIZED PHRASE-COHESION PREFERENCE (deterministic capitalization
    # heuristic — NOT NER; always on, not gated on the clause flag, since
    # severing a capitalized multi-word phrase reads wrong regardless of the
    # balance/clause posture): penalize a break that splits a run of consecutive
    # capitalized cohesion tokens ('United | States', 'New | York', 'Mr. | Wang')
    # when the word ending line 1 does NOT itself close a sentence/clause. Only a
    # TIE-BREAKING nudge — it never selects an infeasible break, never overrides
    # a hard max_chars limit, and is easily overridden by a real sentence/clause
    # boundary (+32/+28) — so it keeps a capitalized phrase whole only when
    # nothing better competes. Closes the 'The United States | government' vs
    # 'The United | States government' tie in favour of keeping the phrase intact.
    if (not last_word.endswith(_SENTENCE_PUNCT)
            and not last_word.endswith(_CLAUSE_PUNCT)
            and _is_capitalized_cohesion_token(last_word)
            and _is_capitalized_cohesion_token(first_word)):
        score -= 18.0

    return score


def choose_two_line_break(text: str, max_chars: int) -> Optional[List[str]]:
    """
    Decide the best single break that splits `text` into TWO lines, each within
    max_chars, honoring clause integrity + balance. Returns [line1, line2] or
    None if no feasible two-line split exists (caller falls back to greedy).

    Deterministic: scans every inter-word gap, scores each feasible one, and
    returns the highest-scoring split. Ties broken by the EARLIER break (stable,
    reproducible). For a string that fits on one line this returns None (no
    split needed — the caller short-circuits before calling us).
    """
    text = (text or "").strip()
    if not text:
        return None

    words = text.split()
    if len(words) < 2:
        return None  # can't split a single token across two lines

    best_i: Optional[int] = None
    best_score = float("-inf")

    # Candidate break i means line1=words[:i], line2=words[i:], for i in 1..n-1.
    for i in range(1, len(words)):
        s = _score_break(words, i, max_chars)
        if s is None:
            continue
        # Strict > keeps the EARLIEST best break on ties → stable output.
        if s > best_score:
            best_score = s
            best_i = i

    if best_i is None:
        return None

    line1 = " ".join(words[:best_i])
    line2 = " ".join(words[best_i:])
    return [line1, line2]
