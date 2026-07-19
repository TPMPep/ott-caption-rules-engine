# CC Rules Engine — Phase 0 Root-Cause Report & Failure-Class Baseline

**Status:** Investigation artifact. **No production engine behavior is changed by this report or its
companion fixtures.** The fixtures in `tests/test_failure_class_baseline.py` are written to **FAIL
against the current engine** — that failure IS the evidence baseline. Phases 1–3 flip each class to
green without regressing the corpus.

**Scope:** the eight reviewer-supplied examples are treated as *evidence of global decision-failure
classes*, never as cue-number targets. Every fixture asserts a **class invariant**, not "cue 0013 now
reads X".

---

## 0. Evidence-integrity disclaimer (read first)

The historical engine runs that produced the reviewer's seven examples are **not forensically
reconstructable**. This is a documented fact of the current architecture, not an evasion:

- The engine's job store is in-memory / transient (see `tests/test_segmentation_regression_corpus.py`
  header — same limitation, independently recorded).
- The formatter persists **no per-stage transformation history** — the intermediate token stream,
  word timings, candidate scores, and `CONDENSATION_MODE` for those exact runs are gone.
- `CCRawTranscriptSegment` stores the immutable *baseline* rows, and `CCSegmentationDecision` stores
  *run-level* audit for runs made **after** decision persistence shipped — but the specific runs the
  reviewer screenshotted predate a captured decision trail for these windows, and the raw provider
  JSON was never persisted to S3 beyond the AAI/Scribe baseline snapshot.

**Consequence for item D (the ~14s "There"):** I therefore CANNOT assert, from stored data, that the
provider emitted `"There"` with an explicit 14,340 ms word duration. The 14s span could have arisen by
**either** mechanism below; the report marks this as an open hypothesis and the fixture asserts the
**invariant** that holds under either mechanism.

Every statement below is tagged:

- **[FACT]** — directly observed in the current committed code or in a stored entity schema.
- **[CODE-PATH]** — a conclusion derived by tracing the committed code paths (deterministic, but not a
  replay of the exact historical run).
- **[HYPOTHESIS]** — a plausible mechanism that REQUIRES the raw provider JSON to confirm/refute.

To convert any **[HYPOTHESIS]** to **[FACT]** we need the raw AssemblyAI/Scribe transcript JSON for
the affected project (the `words[]` / `utterances[]` arrays with `start`/`end`). If that JSON still
exists in the project's S3 baseline (`Project.cc_aai_baseline_key`), `_diagnoseCCBaselineWordSpeakers`
/ `_diagnoseCCBaselineTimings` can dump it — Phase 1 begins by running those against the affected
project and pasting the real word record into the fixture, replacing the synthetic-but-representative
timings used here.

---

## 1. Pipeline lineage (the map every trace uses)

Confirmed stage order **[FACT]**, from `formatter.process_caption_job` + `_build_dialogue_cues`:

```
provider JSON
  → assembly.build_word_timestamps_from_result        (word tokens; speaker + source_utterance_id)
  → formatter._normalize_tokens                        (shape normalization; NO timing sanity guard)
  → segmentation.segment_into_sentence_groups          (sentence groups; pause/speaker boundaries)
  → formatter._build_dialogue_cues                     (pack groups → cues; _word_timings_in_window)
      → _split_sentence_into_cue_chunks                (over-budget sentence → chunks; interpolated tc)
  → shaping.shape_caption_rhythm                       (rhythm split; _split_time_at honors word end_ms)
  → sequence_optimizer.optimize_cue_sequence           (windowed resegmentation; _score; cond. gate)
  → editorial_ai.editorial_refine_cues                 (optional LLM; skips under load)
  → readability.apply_readability_rules                (orphan reflow, micro-merge, CPS, CPL safety net)
  → condensation.condense_cues                         (disfluency + LLM paraphrase; gated)
  → rendering.suppress_repeat_speaker_labels           (turn-based label suppression)
  → capitalization.apply_sentence_capitalization       (final deterministic casing authority)
  → segmentation_qc.run_segmentation_qc                (QC verdict; export gate)
  → export
```

**Key structural facts that recur in the traces below:**

- **[FACT]** No stage between the provider and segmentation validates that a single word's duration is
  physically plausible. `_normalize_tokens` (formatter.py) and `_build_tokens_from_utterances`
  (assembly.py) both emit `int(word["end"]) - int(word["start"])` verbatim.
- **[FACT]** `formatter._word_timings_in_window` assigns a word to a cue by **midpoint**
  (`mid = (ts+te)//2`). A 14s word has a midpoint ~7s into the program, so it lands alone in a cue
  whose window is dictated by that single corrupt frame.
- **[FACT]** `shaping._split_time_at` resolves a split boundary from `word["end_ms"]` — frame-faithful
  to whatever the word record says, including a corrupt frame.
- **[FACT]** `sequence_optimizer._score` current weights: sentence-end cut `+14`, clause-end `+8`,
  mid-phrase `−8`, dangling-tail-marker (coordinators only) `−16`, flash `−20`, fragmentation
  `−2/part`, `no_change` stability `+5`. There is **no** term for "fewest unnecessary boundaries", **no**
  whole-semantic-unit reward, and the orphan-tail penalty list is coordinators only — it does **not**
  include determiners/articles/prepositions (`the/a/of/to/for/at/by/with/from/into/onto`).
- **[FACT]** `sequence_optimizer` condensation gate (`condensation_allowed`) is set True only when NO
  compliant *resegmentation* exists. It never attempts a "borrow idle duration" remedy before
  conceding to condensation.
- **[FACT]** `rendering.suppress_repeat_speaker_labels` is turn-based and correct in isolation, BUT it
  early-returns unless `SPEAKER_LABEL_MODE ∈ {first_occurrence_per_scene, named, every_change, alpha,
  generic}`, and it skips any cue with no resolvable speaker (`speaker is None` → does not advance the
  turn run).

---

## 2. Failure-class traces

Each class below fills the reviewer's 12-point template as far as stored evidence allows. Where a point
depends on the lost historical run, it is marked **[HYPOTHESIS]** and the fixture asserts the invariant.

### Class A — Malformed / implausible provider word timing (item D: the ~14s "There")

1. **Raw transcript / provider timing input** — reviewer report: `"There"` ≈ `00:43:09.518 → 00:43:23.858`
   (≈14,340 ms), next tokens `"are no certainties where dreams are concerned"` begin ≈ `00:43:23.898`.
   **[HYPOTHESIS]** on the exact numbers — not yet dumped from the baseline.
2. **Provider utterance vs word timing** — **[HYPOTHESIS]**, two candidate mechanisms:
   - **(A1)** The provider explicitly emitted `"There"` with a 14s `end`. (Rare but seen with ASR
     hangovers on trailing silence / music beds.)
   - **(A2)** `"There"` had **absent/zero word timing** and inherited the utterance window, or a
     downstream builder synthesized the span. Per your instruction I explicitly do **not** assume A1.
3. **Normalized token values** — **[CODE-PATH]** whichever of A1/A2 holds, `_normalize_tokens` /
   `_build_tokens_from_utterances` pass the span through unmodified (no sanity guard). **[FACT]**
4. **Sentence-group / segmentation decisions** — **[CODE-PATH]** `"There"` and the continuation are
   the same source utterance (no `≥pause_boundary_ms` inter-*utterance* gap is required; the gap here
   is *intra*-token), so segmentation does NOT hard-split — the corruption is carried as one group with
   a 14s internal gap.
5. **Pause / speaker provenance** — **[CODE-PATH]** no `pause_boundary_before` is set (intra-utterance),
   so this is NOT a pause-boundary artifact. Confirms it is a timing-integrity fault, not a boundary
   fault.
6. **Optimizer candidates** — **[CODE-PATH]** `_window_word_stream` builds per-word timings with a 14s
   gap; `_needs_shaping`/rhythm fire because the cue duration is enormous; candidate splits land on the
   corrupt frame.
7. **Candidate scores / rejections** — **[CODE-PATH]** every candidate is scored against corrupt
   durations, so a "flash" penalty and duration-balance terms are computed from garbage — the scores
   are meaningless, not wrong-per-policy.
8. **Condensation eligibility / transform** — n/a for this class (the defect is timing, upstream of
   words-changing).
9. **Capitalization before/after** — n/a (casing is correct; the defect is timecodes).
10. **Speaker-label before/after** — n/a.
11. **Final cue text / timing** — sentence explodes into multiple cues with visibly wrong timecodes
    (the reviewer's symptom).
12. **Responsible stage / code path** — **[FACT]** **token normalization has no timing-plausibility
    guard**: `formatter._normalize_tokens` + `assembly._build_tokens_from_utterances`; the corruption
    is then honored by `formatter._word_timings_in_window` (midpoint assignment) and
    `shaping._split_time_at` (word-`end_ms` boundary). **Not an optimizer scoring fault.**

- **Proposed invariant (Class A):** *No single spoken word token may occupy a duration outside a
  physically plausible band. When a token's measured duration is implausible, the engine deterministically
  repairs it (re-interpolating within the enclosing utterance's real span) and preserves the original
  measurement as evidence (`meta.timing_anomaly = {original_start_ms, original_end_ms, repair_reason,
  repair_version}`). Plausible timings are never touched. The repair runs BEFORE segmentation so no
  downstream stage ever sees the corrupt frame.*
- **Acceptance test (Class A):** given an utterance whose first word carries an implausible span, the
  normalized token stream contains no word whose duration exceeds the plausibility ceiling; the original
  span is recoverable from `meta.timing_anomaly`; and a plausible control stream is returned byte-identical
  (no false repair). **Numeric ceiling NOT locked here** — Phase 1 calibrates it against the corpus (a
  word is empirically ~80–1200 ms; the ceiling will be set with margin and cross-checked so no real long
  word is ever "repaired").

### Class B — Unnecessary fragmentation of complete semantic units (items 1 & 7)

Examples: *"All right, ladies. I can watch all you in slow-mo on the highlights tape."* → 3 cues;
*"There are no certainties where dreams are concerned."* → one/two-word cues.

- 6–7. **[FACT]** `_score` rewards a sentence-end cut `+14` but has **no** term rewarding the layout with
  the *fewest unnecessary boundaries*, and `no_change` (which may itself be an over-split incoming
  arrangement) gets `+5` stability regardless of whether a *coarser compliant* arrangement exists. A
  3-way split of one sentence and a 2-way split score close enough that fragmentation (`−2/part`) is the
  only thing separating them — too weak to dominate.
- 12. **Responsible code path:** `sequence_optimizer._score` (weight hierarchy) + candidate selection.
- **Proposed invariant (Class B):** *When multiple compliant layouts exist for a same-speaker window,
  prefer the one with the fewest cue boundaries and the strongest semantic grouping. A complete sentence
  is kept in one cue when it is naturally readable within all hard constraints (CPS/CPL/duration/lines),
  and is divided only at the strongest available clause/phrase boundary when a hard constraint requires
  it — never exploded into one/two-word cues.* Note (your clarification, honored): this does **not** force
  every sentence into one cue.
- **Acceptance test (Class B):** for a sentence that fits two compliant layouts (e.g. 1-cue and 3-cue),
  the engine selects the layout with fewer boundaries; and no compliant window ever emits a dialogue cue
  of ≤2 words when a ≥3-word-per-side compliant arrangement exists. **Weights NOT locked** — corpus-derived
  in Phase 2.

### Class C — Over-weighting of weak pauses / source-segment boundaries (item 7)

- 5. **[FACT]** genuine hard boundaries require `≥ pause_boundary_ms` (default 1200) **between distinct
  source utterances** — a strong, correct gate. So weak *sub-second* pauses are NOT hard walls today.
- 6–7. **[CODE-PATH]** the over-fragmentation the reviewer sees from "minor pauses" is therefore **not**
  a pause-wall fault; it is the **incoming cue arrangement** (source segments / shaper splits) being
  treated as near-sacred by the optimizer's `+5 no_change` stability with no counter-pressure toward
  coarser semantic grouping. i.e. Class C ≡ the same scoring gap as Class B, viewed from the "why did a
  weak boundary survive" side.
- **Proposed invariant (Class C):** *A cue boundary that corresponds only to a weak/minor pause or a raw
  source-segment edge (not a sentence/clause/semantic boundary and not a hard pause wall) is treated as
  weak evidence; it must lose to a compliant coarser grouping unless a hard constraint requires the split.*
- **Acceptance test (Class C):** an incoming 3-cue arrangement of one sentence, where the two internal
  boundaries are mid-phrase (no punctuation) and all three cues are individually compliant, must be
  recombined toward the coarsest compliant grouping rather than preserved by stability bias.

### Class D — Condensation before faithful compliant layouts are exhausted (item 2: "Start 'em up" → "start 'em")

- 8. **[CODE-PATH]** the optimizer set `condensation_allowed=True` because it found no compliant
  *resegmentation*, and condensation then dropped "up". **[FACT]** the optimizer never tries a bounded
  **borrow-idle-duration** remedy before conceding.
- 12. **Responsible code path:** `sequence_optimizer` candidate generation (missing borrow-duration
  candidate) → `condensation.condense_cues` performing the reword.
- **Proposed invariant (Class D):** *Spoken words are never removed or reworded while any faithful
  compliant layout exists. The ordered remedy hierarchy is: (1) rebalance line breaks, (2) rebalance cue
  boundaries, (3) borrow safe available idle duration, (4) merge/redistribute neighboring material,
  (5) condense — only when 1–4 cannot yield a faithful compliant layout. The engine records why
  condensation became eligible and which non-destructive classes were attempted and rejected.*
- **Borrow-duration safety constraints (your 7, locked as invariant, not yet implemented):** must not
  create cue overlap; cross a speaker change; cross a shot/scene/SFX/hard boundary; consume timing an
  adjacent spoken cue needs; detach text materially from its speech; exceed configured
  lead-in/lead-out/duration/CPS/sync tolerances; or alter immutable source timing provenance.
- **Acceptance test (Class D):** given a cue that is over-CPS by a margin that a legal borrow of trailing
  idle duration would resolve, the engine borrows and preserves all words; condensation is NOT invoked;
  the audit records `condensation_eligible_reason` + attempted/rejected non-destructive classes.

### Class E — Incorrect sentence-start capitalization (item 3: "Start 'em up" → "start 'em", "None of them are." → "none…")

- 9. **[CODE-PATH]** `capitalization._lowercase_first_word` downcases a cue's first word only when the
  continuation tracker believes the previous cue did **not** end a sentence AND the word is provably
  common. **[HYPOTHESIS]** two candidate causes, to be disambiguated by reproduction:
  - **(E1)** The "up" drop (Class D) turned "Start 'em up." into "start 'em." — i.e. capitalization is
    a *victim* of Class D, not itself broken.
  - **(E2)** The continuation tracker mis-set `prev_ended_sentence=False` (e.g. a prior cue ended on a
    quote/closer or an abbreviation the quote-aware check missed), causing a legitimate sentence start
    to be downcased.
- 12. **Responsible code path:** to be pinned by reproduction — `condensation` (if E1) vs
  `capitalization.apply_sentence_capitalization` continuation tracker (if E2). **Per your instruction I
  do NOT assume it is only the tracker.**
- **Proposed invariant (Class E):** *A cue that starts a new sentence has its first word capitalized;
  sentence-initial capitalization is preserved across every transformation path (segmentation,
  resegmentation, condensation, merge, split), unless the transcript itself intentionally used lowercase.
  Capitalization is deterministic and idempotent.*
- **Acceptance test (Class E):** a cue that is the first cue of a new sentence renders with a capital
  initial after the full pipeline, including when the preceding cue ended on a quote-closed sentence and
  including after a condensation/merge transform on the preceding cue.

### Class F — Cue endings on forward-binding function words (item 4: "...fans alike, the")

- 6–7. **[FACT]** `_score` penalizes a cue ending on a **coordinator/discourse marker** (`_DANGLING_TAIL_MARKERS`)
  at `−16`, but does **not** penalize a cue ending on a **determiner/article/preposition**
  (`the/a/an/of/to/for/at/by/with/from/into/onto`). "the" at a cue end is unpenalized today.
- 12. **Responsible code path:** `sequence_optimizer._score` (missing forward-binding tail penalty);
  also relevant in `linebreak._score_break` (line level) and `shaping._pick_word_phrase_boundary`
  (already has a partial `−12`) — Phase 2 unifies these so cue-level and line-level agree.
- **Proposed invariant (Class F):** *A cue should not end on a forward-binding function word
  (determiner/article/preposition/aux/subject-pronoun) when a compliant alternative boundary exists.
  This is a strong SOFT penalty, phrase-aware (it preserves binding units like "the game", "to leave",
  "of the house", "with her"), not a hard prohibition — a hard timing/readability constraint may leave
  no alternative.*
- **Acceptance test (Class F):** for a window with two compliant boundaries where one strands a
  determiner/preposition at the cue end and the other does not, the engine selects the non-stranding
  boundary; when only the stranding boundary is compliant, it is still allowed (soft, not hard).

### Class G — Repeated speaker labels without a valid reintroduction condition (item 5)

- 10. **[FACT]** `suppress_repeat_speaker_labels` is turn-based and correct in isolation, BUT it
  early-returns for modes outside its suppress set and skips cues with `speaker is None` (does not
  advance the turn run). **[HYPOTHESIS]** the repeated `[SPEAKER A:]` arose because either
  (G1) the active `SPEAKER_LABEL_MODE` was outside the suppress set, or (G2) an intervening cue lost its
  resolvable speaker id (diarization drop), so the run tracker did not treat the second A cue as a
  continuation. Requires reproduction to pin which.
- 12. **Responsible code path:** `rendering.suppress_repeat_speaker_labels` behavior at mode/`None`-speaker
  edges (to be confirmed by reproduction).
- **Proposed invariant (Class G):** *A speaker label is (re)introduced only on a genuine speaker change,
  after a sufficient silence/scene break, or when SDH rules require re-identification. Consecutive
  same-speaker cues with no intervening change do not repeat the label — including when an intervening
  cue has an unresolved speaker (an unknown-speaker cue must not reset the turn run for a provably
  same-speaker pair around it).*
- **Acceptance test (Class G):** two consecutive same-speaker cues emit the label once; a same-speaker
  pair separated by a music/unknown cue still suppresses the second label; the label re-appears when a
  different known speaker intervenes.

### Class H — Inappropriate merging of short, complete stand-alone utterances (item 6: "Agatha Schnitzler.")

- 6–7. **[CODE-PATH]** two possible mergers: the optimizer's `merge_all` candidate winning on a short
  window, or `readability.merge_micro_cues` fusing a sub-`min_display` complete utterance into a
  neighbor. **[FACT]** neither path has a "this short cue is a complete stand-alone discourse unit
  (name/greeting/acknowledgement) — prefer to let it stand when timing permits" signal.
- 12. **Responsible code path:** `sequence_optimizer` merge scoring and/or `readability.merge_micro_cues`.
- **Proposed invariant (Class H):** *A short but complete stand-alone utterance (proper-name address,
  greeting, acknowledgement, complete one-clause response) is allowed to remain its own cue when it
  independently satisfies the hard constraints (min duration, CPS, CPL). It is merged with a neighbor
  only when it cannot stand alone compliantly or when a merge is unambiguously more readable.*
- **Acceptance test (Class H):** a complete short utterance that independently clears min_display + CPS
  is not merged into an adjacent different-thought cue; a genuinely sub-min_display fragment still merges
  (Class H does not disable the real micro-cue fix).

---

## 3. Phase plan (unchanged from agreed sequence; recorded here for the commit)

- **Phase 1 — Timing-data integrity.** Class A. Begin by dumping the real provider word record for the
  affected project (`_diagnoseCCBaselineWordSpeakers` / `_diagnoseCCBaselineTimings`) to convert the
  Class-A **[HYPOTHESIS]** to **[FACT]** and replace the synthetic timings in the fixture. Then implement
  the deterministic anomaly-repair pass (Python + JS parity), corpus before/after, Railway/GitHub sync
  list.
- **Phase 2 — Semantic segmentation + scoring hierarchy.** Classes B, C, F, H. Corpus-calibrated weights
  (never hand-picked), before/after, parity, sync list.
- **Phase 3 — Faithful layout-before-condensation, then reproduced capitalization + speaker-label fixes.**
  Class D (borrow-duration candidate + ordered remedy audit), then reproduce E and G through the real
  pipeline to pin E1/E2 and G1/G2 before fixing. Parity, corpus, sync list.

**Acceptance criterion for every phase:** the class invariant holds across a representative corpus (short
dialogue, long dialogue, narration, rapid exchanges, pauses, speaker changes, SDH labels, sound effects,
malformed timing, long-form) with **no regressions**; identical input → identical output; Python and JS in
parity; all new + existing tests pass. **No final numeric weights are locked in Phase 0.**

---

## 4. What is committed in Phase 0

- This report (`docs/phase0-root-cause-report.md`).
- `tests/test_failure_class_baseline.py` — eight permanent, class-invariant fixtures, **currently
  expected to fail** (marked `xfail(strict=True)` so the suite stays green today AND so the day a class is
  fixed, the corresponding xfail flips to an unexpected pass and forces us to remove the marker — the
  baseline can never silently rot). Each fixture references the class + invariant in this report.

No engine module is modified in Phase 0.
