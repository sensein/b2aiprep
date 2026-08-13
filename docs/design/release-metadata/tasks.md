---

description: "Task list for feature 001: pipeline-generated release metadata"
---

# Tasks: Generated Release Metadata

**Input**: Design documents from `/docs/design/release-metadata/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/cli.md, quickstart.md

**Tests**: Included. The spec does not request them, but `.specify/memory/constitution.md` v1.0.0
Quality Gates 1, 4, 5, and 6 require them for any change on a publishing path — the empty case, the
failure path, idempotency, and the identifying-content scan. Each test task cites its gate.

**Organization**: Grouped by user story so each is independently implementable and testable.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: parallelizable — different file, no dependency on an incomplete task
- **[Story]**: US1–US5, matching spec.md priorities P1–P5

## Path Conventions

Paths are relative to the `b2aiprep` repository root. References to `senselab/…` are in the separate
`sensein/senselab` repository. Release data and scripts live outside both, under
`/orcd/data/satra/002/datasets/b2aivoice/post_3.0/` on the cluster.

Work in the `b2aiprep_test` conda env (Python 3.12.13) — the env the release scripts activate.

---

## Phase 1: Setup

**Purpose**: make the new dependencies and package available without changing any behaviour.

- [ ] T001 Add a `metadata` optional-dependency group (`fairscape-cli>=1.2.9`, `fairscape-models>=1.2.1`) to `pyproject.toml`
- [ ] T002 Create the package skeleton `src/b2aiprep/metadata/__init__.py` and register it under `[tool.setuptools.packages.find]` in `pyproject.toml`
- [ ] T003 Install the extra into the release env (`conda activate b2aiprep_test; pip install -e 'b2aiprep[metadata]'`) and record the resolved fairscape versions in `docs/design/release-metadata/research.md` under R3

---

## Phase 2: Foundational (blocking prerequisites)

**Purpose**: the record substrate every story reads or writes. No story can start before this.

- [ ] T004 Define `RunRecord` and `UnitRecord` schemas with field-level validation in `src/b2aiprep/metadata/records.py`, matching the tables in `docs/design/release-metadata/data-model.md`
- [ ] T005 Implement the closed reason vocabulary as an enum in `src/b2aiprep/metadata/records.py`, one member per row of the reason table in data-model.md
- [ ] T006 Implement record write/read, `run_id` derivation from `SLURM_JOB_ID`/`SLURM_ARRAY_TASK_ID`, and `partial`→`complete` status promotion on clean exit in `src/b2aiprep/metadata/provenance.py`
- [ ] T007 [P] Implement environment capture — software and backend versions via `importlib.metadata`, host and SLURM context — in `src/b2aiprep/metadata/environment.py`
- [ ] T008 [P] Implement model-identity resolution using `HFModel.get_model_info().sha` in `src/b2aiprep/metadata/models.py`, returning `{role, provider, repo, requested_revision, resolved_sha}`
- [ ] T009 [P] Implement a release-version guard that rejects a local/dirty version segment in `src/b2aiprep/metadata/environment.py` (constitution III; the release env currently installs `3.1.0+51.g34174e7.dirty`)
- [ ] T010 Add an optional `--provenance-dir` to `redcap2bids`, `generate-audio-features`, `run-quality-control-on-audios`, `deidentify-bids-dataset`, and `create-bundled-dataset` in `src/b2aiprep/commands.py`, defaulting to `<dataset_root>/provenance`, with no behaviour change yet
- [ ] T011 Record the decision on whether `provenance/` is uploaded to Sage and enumerated by C2M2, in `docs/design/release-metadata/research.md` under R6 (FR-028 requires the choice be recorded, not implicit)
- [ ] T012 [P] Test record round-trip, `run_id` stability across a resumed array task, and that an aborted run leaves `status: partial`, in `tests/test_provenance_records.py` (constitution gate 5)
- [ ] T013 [P] Test that the release-version guard rejects `3.1.0+51.g34174e7.dirty` and accepts `3.1.0`, in `tests/test_provenance_records.py` (constitution gate 3)

**Checkpoint**: records can be written and read; nothing writes them yet.

---

## Phase 3: User Story 1 — PhysioNet crate from one command (Priority: P1) 🎯 MVP

**Goal**: an engineer runs one command against a built bundle and gets a complete, valid RO-Crate with
every file described, every schema correct, and no typed values.

**Independent test**: build the fixture bundle per `quickstart.md` steps 1–2, run
`generate-release-metadata`, and confirm every bundle file has an entry with its own size, digest, and
schema; zero empty property values; and byte-identical output across two runs.

### Data corrections this story depends on

- [ ] T014 [P] [US1] Fix the field name in `src/b2aiprep/prepare/resources/feature_schemas/ppgs.json` from `ppg` to `ppgs` to match the parquet column
- [ ] T015 [P] [US1] Fix the field name in `src/b2aiprep/prepare/resources/feature_schemas/torchaudio_spectrogram.json` from `spectrograms` to `spectrogram`
- [ ] T016 [US1] Remove the hard-coded backend version strings and the unset `extraction_date` from all nine files in `src/b2aiprep/prepare/resources/feature_schemas/`, so versions come from the run record instead
- [ ] T017 [P] [US1] Resolve the duplicated arXiv identifier between `ppgs.json:87` and `sparc_ema.json:66-67` in `src/b2aiprep/prepare/resources/feature_schemas/`, or mark the uncertain one as unverified
- [ ] T018 [P] [US1] Create `src/b2aiprep/prepare/resources/release_metadata.yaml` holding the single editorial source — authorship with ORCIDs, rights, ethics, collection method, limitations, intended uses, citation, structured funder award — plus `approved_by`, `approved_on`, and a digest of the approved text

### Bundle-time record

- [ ] T019 [US1] Emit a bundle `RunRecord` from `create_bundled_dataset` in `src/b2aiprep/commands.py`, promoting the `bundle_output_stats` dict built at `commands.py:333` from a log line into the record, including each feature family's own participant, session, and recording counts, since these differ between families within one release (FR-012)
- [ ] T020 [US1] Capture per-output size and sha256 for every bundle file into the bundle record in `src/b2aiprep/commands.py`, so nothing downstream recomputes digests
- [ ] T021 [US1] Emit bundle `UnitRecord` rows for the skip branches currently logged and discarded at `src/b2aiprep/prepare/bundle_data.py:49`, `:52`, `:145`, `:150`, mapping each to its reason-vocabulary member
- [ ] T022 [US1] Record the `skipped(no-data)` whole-family outcome from `commands.py:532-544` and `:569-581` into the bundle record, so an absent parquet is documented rather than omitted (FR-019)

### Crate generation

- [ ] T023 [US1] Implement RO-Crate graph construction with `fairscape_models` in `src/b2aiprep/metadata/rocrate.py` — root, one dataset entry per file, schema per file from its shipped dictionary, software and model entities, one activity per distinct (software, config) set
- [ ] T024 [US1] Emit `contentSize` as a string in `src/b2aiprep/metadata/rocrate.py` (research R3: `fairscape-models>=1.2.1` rejects `int`)
- [ ] T025 [US1] Invoke `fairscape-cli build subcrate` and `build datasheet` on the crate directory from `src/b2aiprep/metadata/rocrate.py` to produce the datasheet, preview, croissant, prov-graph, merkle tree, and AI-ready score
- [ ] T026 [US1] Implement the pre-emit gate checks in `src/b2aiprep/metadata/rocrate.py` — no empty or placeholder value, no duplicate identifier, declared name agrees with location, aggregates equal the sum over items, dates ISO 8601 and inside the run→publication interval — failing before anything is written (FR-004, FR-021)
- [ ] T027 [US1] Add the `generate-release-metadata` click command to `src/b2aiprep/commands.py` with the signature and exit codes in `docs/design/release-metadata/contracts/cli.md`
- [ ] T028 [US1] Register `generate_release_metadata` in `src/b2aiprep/cli.py`

### Tests for User Story 1

- [ ] T029 [P] [US1] Test that generation from the fixture bundle describes every file with size, digest, and schema, and yields zero empty property values, in `tests/test_rocrate_generation.py`
- [ ] T030 [P] [US1] Test that a bundle with a missing record, an incomplete record, or an unaccounted file causes a non-zero exit and writes nothing, in `tests/test_rocrate_generation.py` (constitution gate 4)
- [ ] T031 [P] [US1] Test that two generations from identical inputs differ only in the generation timestamp, in `tests/test_rocrate_generation.py` (constitution gate 5)
- [ ] T032 [P] [US1] Test that every shipped feature dictionary's field names equal its parquet's column names, parameterized over all nine, in `tests/test_resources.py` (would have caught T014/T015)
- [ ] T033 [P] [US1] Test the empty case — a feature family with no rows — produces a documented absence rather than an omitted entry, in `tests/test_rocrate_generation.py` (constitution gate 1)

**Checkpoint**: US1 delivers independently. The notebook can be retired.

---

## Phase 4: User Story 2 — provenance captured while it still exists (Priority: P2)

**Goal**: extraction, QC, deidentification, and RedCap conversion each record what only they can know,
so a published file traces to the exact software, models, and configuration that produced it.

**Independent test**: run extraction over a few subjects, then again with `--update` and a changed
config; the records name both invocations, their versions, and which family each produced. No cluster
run needed.

- [ ] T034 [US2] Pin the diarization model explicitly in `src/b2aiprep/prepare/prepare.py` by restoring the commented-out `HFModel` at `prepare.py:298-300`, so all three models are caller-chosen and reportable (research R2)
- [ ] T035 [US2] Emit an extraction `RunRecord` per invocation from `extract_features_workflow` in `src/b2aiprep/prepare/prepare.py` carrying software versions, the three resolved model shas, and the effective configuration
- [ ] T036 [US2] Emit extraction `UnitRecord` rows per `(recording, family)` from `extract_single` in `src/b2aiprep/prepare/prepare.py`, distinguishing `computed`, `preserved`, `skipped`, and `failed` — including the `--update` preservation branch at `prepare.py:189-244` and the failure branches at `:157-160`, `:311-314`, `:330-335`, `:351-354`
- [ ] T037 [US2] Record the requested configuration plus per-family computed-versus-preserved counts in the extraction record, since under `--update` a single flat config claim is false (FR-010)
- [ ] T038 [P] [US2] Emit a QC `RunRecord` from `quality_control_wrapper` in `src/b2aiprep/prepare/quality_control.py` with senselab metric versions, thresholds, the `deep_checks`/`skip_windowing` flags, and the row count written at `quality_control.py:90`
- [ ] T039 [P] [US2] Emit a deidentification `RunRecord` and `UnitRecord` rows from `BIDSDataset.deidentify` in `src/b2aiprep/prepare/dataset.py`, using deidentified identifiers only and recording config digests plus per-rule affected counts from `dataset.py:1963-1967` and `:1987-1991`
- [ ] T040 [P] [US2] Record per-recording withheld feature families in the deidentification unit rows in `src/b2aiprep/prepare/dataset.py`, from the sensitivity list at `dataset.py:71-75` applied at `:2263-2268`
- [ ] T041 [P] [US2] Emit a `redcap2bids` `RunRecord` in `src/b2aiprep/prepare/redcap.py` carrying the source export path, sha256, row count, declared access tier, and the `b2ai-redcap2rs` commit via `get_commit_sha` (`prepare/utils.py:52`)
- [ ] T042 [US2] Implement cross-invocation homogeneity verification in `src/b2aiprep/metadata/rocrate.py` — group invocations by (software set, configuration), attribute each published family to exactly one, and fail naming the competing `run_id`s when it cannot (FR-010, research R5)
- [ ] T043 [US2] Extend the crate's software and model entities in `src/b2aiprep/metadata/rocrate.py` to cover senselab and all six feature backends with the versions actually recorded, replacing the single hand-typed software entity
- [ ] T044 [P] [US2] Test that a recording whose family was preserved by `--update` is attributed to the invocation that computed it, not to the resuming invocation, in `tests/test_provenance_records.py`
- [ ] T045 [P] [US2] Test that two invocations with differing versions over the same release cause the generator to fail and name both `run_id`s, in `tests/test_rocrate_generation.py` (constitution gate 4)
- [ ] T046 [P] [US2] Test that no model appears in a record with a branch name rather than a resolved sha, in `tests/test_provenance_records.py` (constitution gate 3)

**Checkpoint**: published derived files trace to real software and model versions.

---

## Phase 5: User Story 3 — a release cannot ship with a gap (Priority: P3)

**Goal**: one command either passes or names every missing, empty, inconsistent, or unaccounted fact —
and can audit an already-published release without regenerating anything.

**Independent test**: run it against freshly generated fixture metadata (passes), against the same
metadata with one property emptied (fails, naming it), and against the published 3.0.0 crate (reports
the catalogued defects).

- [ ] T047 [US3] Implement the check suite in `src/b2aiprep/metadata/validate.py`, one function per row of the check table in `docs/design/release-metadata/contracts/cli.md`, each returning findings rather than raising
- [ ] T048 [US3] Implement the FR-011 accounting identity in `src/b2aiprep/metadata/validate.py` — published rows plus non-`computed` unit rows must equal total (recording × family) combinations — reporting both sides on failure
- [ ] T049 [US3] Implement the records-optional mode in `src/b2aiprep/metadata/validate.py`, so a published crate plus a file inventory can be audited with no `provenance/` present (FR-022)
- [ ] T050 [US3] Implement the identifying-content scan in `src/b2aiprep/metadata/validate.py` — reject participant identifiers in the source space, free-text response values, and transcripts in any record or artifact (FR-025, constitution VII)
- [ ] T051 [US3] Implement the crosswalk-absence check in `src/b2aiprep/metadata/validate.py`, failing if an identifier-map or crosswalk file is reachable inside a dataset root (FR-026)
- [ ] T052 [US3] Add the `validate-release-metadata` click command to `src/b2aiprep/commands.py` per `contracts/cli.md`, with exit codes 0/1/2 and one-pass reporting of all findings
- [ ] T053 [US3] Register `validate_release_metadata` in `src/b2aiprep/cli.py`
- [ ] T054 [P] [US3] Test that emptying one property in generated metadata causes exit 1 naming that property, in `tests/test_metadata_validation.py` (constitution gate 4)
- [ ] T055 [P] [US3] Test the accounting identity fails when a unit row is removed, in `tests/test_metadata_validation.py`
- [ ] T056 [P] [US3] Test the identifying-content scan detects a planted source-space participant id and a planted transcript, in `tests/test_metadata_validation.py` (constitution gate 6)
- [ ] T057 [US3] Document the 3.0.0 audit as a local procedure in `docs/design/release-metadata/quickstart.md` step 4, asserting the expected findings — 286 empty values, the duplicated `ark:59853/b2ai-voice-schema-phenotype-confounders`, `sparc-periodicity` named `sparc_loudness.parquet`, `torchaudio-pitch` named `torchaudio_spectrogram.parquet`, prose `contentSize`, the `01/29/2026` computation date, and four datasets with `contentUrl: []` — and confirm no release data enters the repository (constitution VIII)

**Checkpoint**: nothing ships with a gap, and what already shipped can be audited.

---

## Phase 6: User Story 4 — both datasets described and reconciled (Priority: P4)

**Goal**: the Sage audio dataset gets its own crate, each dataset names the other, and their coverage
relationship is stated as measured counts rather than an assumed rule.

**Independent test**: run against `deid_bids_registered_04_14_26` and `deid_bids_controlled_04_14_26`;
participant overlap must report 767 shared / 66 registered-only / 0 controlled-only, matching a direct
count, with unmatched recordings counted rather than dropped.

- [ ] T058 [US4] Implement participant-set reconciliation in `src/b2aiprep/metadata/reconcile.py` as a direct set operation on published identifiers, reporting all three directions per cohort with no containment rule asserted (FR-014, research R1)
- [ ] T059 [US4] Implement recording-level matching in `src/b2aiprep/metadata/reconcile.py` on `(participant, session_prefix_8, task)`, flagging pairs that agree on the 8-character prefix but differ in length, and reporting the unmatched count (research R1: session ids can be re-derived at 16 characters per tree)
- [ ] T060 [US4] Extend `src/b2aiprep/metadata/rocrate.py` to emit a crate for the deidentified-audio distribution at its own root, with its own host, access tier, licence, and identifier, inheriting nothing from the PhysioNet distribution (FR-013)
- [ ] T061 [US4] Implement sibling cross-references in `src/b2aiprep/metadata/rocrate.py` — the PhysioNet crate references Sage by synId, which already resolves at upload time; the Sage crate may reference the PhysioNet DOI once minted (FR-013, FR-023)
- [ ] T062 [US4] Implement the pending-fact representation in `src/b2aiprep/metadata/rocrate.py` so an unminted DOI is explicitly pending rather than empty or invented, and the validator treats it as known-pending (FR-015)
- [ ] T063 [US4] Represent the common deidentified source as a graph entity with no retrieval URL in `src/b2aiprep/metadata/rocrate.py`, with both distributions deriving from it rather than from each other (FR-022 of spec, research R1)
- [ ] T064 [P] [US4] Apply the T011 decision in `external_scripts/sage_upload_scripts/sage_generate_manifest.py` — include or exclude `provenance/` explicitly
- [ ] T065 [P] [US4] Make `external_scripts/sage_upload_scripts/verify_sage_contents.py` verify an uploaded copy against the release record file by file — every recorded output present, every digest matching — and exit non-zero on any missing file or mismatch; today it raises only on setup failures at `:255`, `:261`, `:281`, `:353` and never on a digest mismatch. Honour the same `provenance/` decision as T064 (FR-020, constitution V)
- [ ] T066 [P] [US4] Test reconciliation counts against the two v3.1 deid trees as a documented local procedure, and against synthetic trees in CI, in `tests/test_cross_dataset_reconcile.py`
- [ ] T067 [P] [US4] Test that a session id of 8 characters in one tree and 16 in the other still matches, and that a genuinely unmatched recording is counted rather than dropped, in `tests/test_cross_dataset_reconcile.py` (constitution gate 1)

**Checkpoint**: both published datasets are described and truthfully related.

---

## Phase 7: User Story 5 — catalogue submission from the same facts (Priority: P5)

**Goal**: the C2M2 submission comes from the same records, with nothing re-entered and nothing
re-hashed, and every fact shared with the RO-Crate is identical in both.

**Independent test**: generate both artifacts for the fixture release and cross-compare every shared
fact; `cfde-c2m2 validate` passes with no operator-supplied arguments.

- [ ] T068 [US5] Move the C2M2 library into `src/b2aiprep/metadata/c2m2/` as `mappings.py`, `constants.py`, `bundle.py`, `controlled.py`, and `subjects.py`, converting the top-level imports to relative ones
- [ ] T069 [US5] Reduce `external_scripts/c2m2/*.py` to thin argparse wrappers importing `b2aiprep.metadata.c2m2`, preserving every interface documented in `external_scripts/c2m2/README.md`
- [ ] T070 [US5] De-duplicate the appending writes in `src/b2aiprep/metadata/c2m2/bundle.py` and `subjects.py` (was `bundle_to_c2m2.py:497-533`, `fill_subject_files.py:146-152`) so a re-run replaces rather than doubles (FR-019, constitution VI)
- [ ] T071 [US5] Read file sizes and digests from the bundle record instead of re-walking and re-hashing in `src/b2aiprep/metadata/c2m2/bundle.py`, replacing the `rglob('*')` plus `calculate_sha256_file_digest` at `bundle_to_c2m2.py:25`, `:34` (FR-016)
- [ ] T072 [US5] Take the PhysioNet version from the record rather than `--physionet_version` when templating access URLs in `src/b2aiprep/metadata/c2m2/bundle.py`, and fix the adult-slug-for-pediatric bug at `bundle_to_c2m2.py:468-469`
- [ ] T073 [US5] Close the silent-empty vocabulary paths in `src/b2aiprep/metadata/c2m2/mappings.py` — the `''` fallbacks for `.wav`/`.parquet` at `:158-172`, the empty `peds_condition_to_DOID` at `:153`, the seven adult conditions mapping to `[]` at `:132-145`, and unmapped task names — so an absent mapping fails and names the unmapped value (FR-017, constitution I)
- [ ] T074 [US5] Apply the T011 decision to C2M2 enumeration in `src/b2aiprep/metadata/c2m2/bundle.py` and `controlled.py` — describe `provenance/` with correct types or exclude it explicitly (FR-028)
- [ ] T075 [US5] Wire `--c2m2-out` into `generate-release-metadata` in `src/b2aiprep/commands.py` so both artifacts come from one invocation
- [ ] T076 [P] [US5] Implement the cross-format agreement check in `src/b2aiprep/metadata/validate.py` — every fact present in both the RO-Crate and the C2M2 submission must be identical, and the file inventories must cover the same set (FR-016)
- [ ] T077 [P] [US5] Test that running C2M2 generation twice produces identical output rather than doubled rows, in `tests/test_c2m2_generation.py` (constitution gate 5)
- [ ] T078 [P] [US5] Test that an unmapped condition or file type fails and names the value rather than emitting an empty term, in `tests/test_c2m2_generation.py` (constitution gate 1)
- [ ] T079 [P] [US5] Test that the existing documented `external_scripts/c2m2` invocations still work after relocation, in `tests/test_c2m2_generation.py`

**Checkpoint**: the last manual metadata path is gone.

---

## Phase 8: Polish & cross-cutting

- [ ] T080 Add the two new steps to `RELEASE.md` and reconcile it with the release scripts — the documented example passes `--update` and a `tiny` transcription model while `post_3.0/v3.1/scripts/adult_feature_extraction.sh` passes no `--update` and uses `large-v3-turbo` (FR-024, constitution IV)
- [ ] T081 [P] Add `GeneratedBy`, `SourceDatasets`, and `DatasetDOI` to `src/b2aiprep/template/dataset_description.json`, so the BIDS trees carry provenance natively
- [ ] T082 [P] Add `.zenodo.json` and update `CITATION.cff` in `b2aiprep/`, and add both to `senselab/` which has neither, so FR-008's durable identifiers exist before the release citing them
- [ ] T083 Document the release ordering in `RELEASE.md` — tag and release senselab, then b2aiprep, mint both DOIs, then run the pipeline, then generate and validate, then upload — since PhysioNet cannot be corrected afterwards
- [ ] T084 [P] Add the missing `sage_upload_array.sbatch` referenced at `external_scripts/sage_upload_scripts/README.md:54,66-67`, or remove the instruction (constitution IV)
- [ ] T085 [P] Wire the identifying-content scan into the test suite so it runs on every change to a publishing path, in `tests/test_metadata_validation.py` (constitution gate 6)
- [ ] T086 Run the full quickstart end to end on the fixtures in `data/` and correct any divergence in `docs/design/release-metadata/quickstart.md`
- [ ] T087 Perform the 3.0.0 audit locally per quickstart step 4 and record the findings in `docs/design/release-metadata/research.md`, confirming the validator detects each catalogued defect (SC-007)

---

## Dependencies & Execution Order

### Phase dependencies

- **Setup (T001–T003)**: no dependencies
- **Foundational (T004–T013)**: needs Setup; **blocks every story**
- **US1 (T014–T033)**: needs Foundational
- **US2 (T034–T046)**: needs Foundational; T042–T043 extend US1's generator, so US1 first
- **US3 (T047–T057)**: needs Foundational; reuses US1's gate checks, so US1 first
- **US4 (T058–T067)**: needs US1 and US3
- **US5 (T068–T079)**: needs US1 and US2 (reads the bundle record's digests)
- **Polish (T080–T087)**: needs the stories it documents

### Story dependencies

US1 is standalone and is the MVP. US2, US3 build on US1's generator but are independently testable and
deliverable. US4 needs US3's validator for its reconciliation checks. US5 is last because the catalogue
is amendable, unlike PhysioNet.

### Within-phase parallelism

```
Foundational:  T007, T008, T009 in parallel (separate modules); T012, T013 in parallel
US1 fixes:     T014, T015, T017, T018 in parallel (separate files); T016 touches all nine
US1 tests:     T029–T033 in parallel
US2 records:   T038, T039, T040, T041 in parallel (separate modules)
US2 tests:     T044, T045, T046 in parallel
US3 tests:     T054, T055, T056 in parallel
US4:           T064, T065 in parallel with T058–T063; T066, T067 in parallel
US5 tests:     T077, T078, T079 in parallel
Polish:        T081, T082, T084, T085 in parallel
```

Sequential within US1's generator chain: T023 → T024 → T025 → T026 → T027 → T028, all touching
`rocrate.py` and `commands.py`.

## Implementation Strategy

**MVP is US1 alone** (T001–T033, 33 tasks). It retires the notebook, fixes the two wrong data
dictionaries, and converts the 3.0.0 defects that are recoverable from a built bundle — file coverage,
identifiers, schemas, sizes, digests, placeholder text — without touching the extraction path at all.

Deliver in story order thereafter. Each checkpoint is a usable state: US2 makes provenance real, US3
makes gaps fatal, US4 connects the two published datasets, US5 removes the last manual path.

Two things gate a real release independently of task order: the `b2aiprep_test` env must be installed
from a tagged commit rather than the current `3.1.0+51.g34174e7.dirty` (T009 will refuse otherwise), and
the Zenodo registrations in T082 need an administrative action on the `sensein` organization.

## Task Summary

| Phase | Tasks | Count |
|---|---|---|
| Setup | T001–T003 | 3 |
| Foundational | T004–T013 | 10 |
| US1 (P1, MVP) | T014–T033 | 20 |
| US2 (P2) | T034–T046 | 13 |
| US3 (P3) | T047–T057 | 11 |
| US4 (P4) | T058–T067 | 10 |
| US5 (P5) | T068–T079 | 12 |
| Polish | T080–T087 | 8 |
| **Total** | | **87** |

Test tasks: 20 of 87, each citing the constitution Quality Gate that requires it.
