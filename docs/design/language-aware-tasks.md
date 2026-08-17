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

**Deferred / follow-up**:
- **Spanish instructions** — the es-419 docs carry them (and the static bank
  harvests them), but `instructions` are still English for now; stimulus_text (the
  WER reference) was the priority. Wire `instructions` next.
- **Generator integration** — the es-419 harvest is a standalone script; fold it
  into `build_task_registry.py` so `--check` drift-guards the Spanish banks too.
- **vocab / random-item** stimulus is per-participant from the questionnaire join,
  so it should already reflect the Spanish values the participant saw — verify
  against a real Spanish session rather than assume.

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
