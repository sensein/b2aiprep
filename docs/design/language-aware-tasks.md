# Language-aware task metadata

**Status**: implemented (en + es-419) · **Created**: 2026-08-17

## Implemented

- **`language` field** on every sidecar, from the participant's `selected_language`
  (`_language_from_selected` → BCP-47; default `en`), threaded through
  `convert_response_to_bids_metadata` and `dataset.py`. Documented in
  `metadata.json`.
- **Language-aware stimulus** via `_bank_name_for_language` (bank `→ <bank>_<lang>`
  when it exists, else the English bank) and a per-language static bank
  (`static_stimulus_<lang>.json`) for the passage/recall reference tasks.
- **es-419 banks** harvested from the redcap docs by `scripts/harvest_es419_banks.py`
  (Harvard 70×10, CAPE-V v2 ×6, Caterpillar + Story-Recall v2). Spanish read/recall
  recordings now carry the Spanish reference; English output is unchanged apart from
  the added `language` field.

**Simplifications** (Spanish only received the *current* protocol):
- No special "unavailable-in-language" handling. A Spanish recording whose task has
  no es-419 stimulus (Rainbow Passage, story-recall v1 — both retired) falls back to
  the English text but is still tagged `language: es-419`, so it is filterable. These
  are edge/retired cases, not current-protocol Spanish tasks.

- **Spanish instructions** — `task_instructions_es_419.json` (family-keyed),
  harvested from the es-419 acoustic-task description pages, override the English
  instruction for non-English recordings of that family. Covers all current-protocol
  adult families Spanish sessions record (Harvard, CAPE-V, Caterpillar, Story Recall,
  Free Speech, Picture Description, Glides, Diadochokinesis, MPT, Prolonged Vowel,
  Loudness, Respiration & Cough). Family-keyed, so a v1-labelled recording in a
  Spanish session gets the current Spanish instruction (Spanish has one per family);
  the CAPE-V stimulus bank likewise falls back to its single current version.

**Robustness (from code review)**:
- `selected_language` is a RedCap radio with three choices (data dictionary:
  `1, English | 2, French | 3, Spanish`; `selected_language_2` carries the same as
  BCP-47 `en-US/fr-CA/es-419` but is absent from the exports). `_language_from_selected`
  maps every form an export can carry — text labels, the integer codes `1/2/3`, and
  the BCP-47 codes — to `en` / `fr-CA` / `es-419`, so a *coded* export never
  silently collapses Spanish (`3`) or French (`2`) to English. Anything still
  unrecognized **warns** and defaults to `en` rather than silently mislabeling.
- **French** is a valid choice with no fr-CA acoustic-task content in the repo, so a
  French session is tagged `fr-CA` (filterable) and its read tasks trip the
  missing-language-bank warning below — never silently presented as English.
- A non-English **read/recall** task whose bank has no `<bank>_<lang>` variant
  (reading-passage / repeating-sentences / repeating-words — all peds, so never
  es-419 in practice) **warns** rather than silently shipping the English reference
  tagged as another language.
- The flat-file fallback path now threads `language` into `_resolve_prompt_ref`.

**Random Item Generation — category-driven, variant-aware instruction**:
- One recording per session (verified: 339 sessions, all count 1); the drawn
  category is stored in `random_item_generation_category` and the variant
  (repeatable letters/numbers vs non-repeatable category) is chosen from it.
- A category outside {Numbers, Letters} appears **only** in Category_2, so it is
  unambiguously the **non-repeatable** variant -> the category instruction with the
  category named.
- **Numbers / Letters appear in BOTH** Category_1 and Category_2, and nothing in
  the data disambiguates them (no session records both, no combined value), so
  those keep the **general** instruction (which describes both variants) rather
  than asserting a repeatability the data can't confirm.
- Curated `random_item_instructions.json` (en + es-419); category lines verbatim
  from source, es-419 `general` composed from the es-419 source phrases (the
  Spanish page has no single combined line). Both variants carry the
  "selection appears / auto-stops" procedural line.
- **Time limit removed** from the random-item instruction (flat + registry + the
  new resource). The source states v1 1 min / v2 2 min (English) and the es-419
  "- v2" page contradicts it ("1 minuto"), but more decisively `random_duration`
  shows recordings routinely blow past any limit (v1 median 98 s, 95/237 over
  2 min; v2 max 1554 s), so it was not an enforced cap and stating one would
  misrepresent the data. `random_item_generation_category` is a RedCap radio with
  10 fixed choices (numbers, letters, firstNames, cityNames, countryNames, jobs,
  fruits, drinks, animals, wordStartingWithT); the drawn category is surfaced on
  every random-item instruction regardless of variant.

- **Free Speech cue** — the numbered current (v2) free-speech carries a
  per-recording Spanish cue (`free_speech_bank_es_419.json`, keyed by index),
  wired via `_free_speech_cue` and gated on the `(v2)` marker + trailing index.
  The **voice** variant (unnumbered `free-speech`) and **v1** are both Retired and
  have no es-419 source anywhere (no doc, no MLM translation), so they keep the
  English cue and are never given the v2 questions -- the numbered/voice
  distinction is preserved.

**Complete for all meaningful content** (every WER reference, every current-protocol
instruction, the language field). The items below are not functional gaps:
- **Grouped-task instruction granularity** (minor, not a gap) — Spanish non-lexical
  grouped tasks (e.g. diadochokinesis per syllable) get one family-level Spanish
  instruction rather than per-recording prose. Nothing is lost: the discriminating
  target (the syllable) is in the recording name; only the prose is general.
- **Generator integration** (optional cleanup, not a gap) — the es-419 harvest is a
  standalone script that is already `--check` drift-guarded; folding it into
  `build_task_registry.py` would just consolidate the two generators.
- **Time-limit discrepancy** (upstream data bug, not fixable here) — the es-419 "- v2"
  page says "1 minuto" vs the English "2 minutes"; report to bridge2ai-redcap.
  Already handled downstream by removing the time limit from the instruction.
- ~~**vocab / random-item** stimulus from the questionnaire join~~ — **verified
  moot**: Spanish sessions (07_01) recorded zero Productive-Vocabulary,
  Random-Item, and Stroop tasks (0 recordings and 0 questionnaire rows). Their
  only elicited recordings are Free Speech (no reference text) and Picture
  Description (language-neutral image), so the join has nothing language-dependent
  to resolve for Spanish. No change needed.

## Original proposal

**Status**: proposed · **Created**: 2026-08-17

## Why

The protocol is administered in more than one language, but the task resolver is
English-only, so non-English recordings inherit English instructions and reference
text. In `registered_adult_07_01.csv`: `selected_language` = 1078 English / **16
Spanish** (one value per participant, no mixed sessions), producing 657 Spanish
recordings. **316 of them (300 read + 16 recall) ship an English `stimulus_text`
as the WER ground truth for Spanish speech**, and no sidecar field records the
language, so they can't even be filtered out. Affects every prior release; the
PhysioNet feature dataset is immutable, so only future versions can be corrected.

The correct Spanish content already exists in `bridge2ai-redcap` (`es-419` layer):
parallel-indexed Spanish stimulus for Harvard Sentences
(`HarvardSentences-es-419.md`), CAPE-V, Caterpillar Passage, Story Recall v2,
Productive Vocabulary, Random-Item categories, plus Spanish task descriptions. Only
Rainbow Passage (4 recs) has no `es-419` version.

Not this change: transcription is *also* English-forced (`prepare.py:342` hardcodes
`language_code: "en"`) — tracked as a separate tooling issue.

## The change

Add a `language` axis to task resolution, mirroring the existing `population`
(peds/adult) axis — the same routing shape, one more discriminator.

1. **Thread `language`** from the session's `selected_language` into
   `convert_response_to_bids_metadata` (`fhir_utils.py`), defaulting to `en` when
   absent — exactly as `population` is threaded today in `dataset.py`.
2. **Add a `language` field to every sidecar** (BCP-47: `en`, `es-419`); document it
   in `resources/metadata.json`. *(This alone is the correctness floor — makes the
   316 filterable even before Spanish stimulus is wired in.)*
3. **Route to `(population, language)`** in `_resolve_task_registry` / the alias
   index; add `es_419` stimulus banks parallel to the English banks (reuse the
   existing `_resolve_prompt_ref` index logic — same list/index keys).
4. **Missing-in-language** read/recall (Rainbow, 4 recs): `stimulus_text = ""`,
   `stimulus_source` marks it, warn once per `(task, language)` — never substitute
   English.
5. **Generator**: harvest the `es-419` docs into the banks/registry so it stays
   drift-guarded by `--check`, not hand-maintained.

## Invariants / checks

- English sessions: byte-identical output vs today (parity on a real build).
- Spanish read tasks: `stimulus_text` = the `es-419` reference for that
  family/index; instructions Spanish.
- Sidecar-schema change: bundle metadata table + C2M2/RO-Crate projections must
  tolerate the new `language` field.

## Open questions

- Rainbow Passage (4) has no `es-419` — did those sessions present English Rainbow
  (English ref arguably fine) or nothing? Confirm before choosing English vs empty.
- Story Recall v1 Spanish reference: exists, or mark unavailable-in-language?
- `es-419` vs plain `es` for the field value (recommend `es-419`, matches source).

## Sizing

Small–medium: (1)+(2) are a few lines through `dataset.py`/`fhir_utils.py` +
`metadata.json`; (3)–(5) are the es-419 banks and one more index key — the
population axis is the working precedent. Do (1)+(2) first as the immutable-safe
floor even if (3)–(5) slip a release.
