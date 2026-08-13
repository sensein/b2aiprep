# Implementation Plan: Generated Release Metadata

**Date**: 2026-08-13 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `docs/design/release-metadata/spec.md`

## Summary

Each pipeline step that produces published files writes a record of what it did — versions, resolved
model commits, effective configuration, inputs, outputs, and the outcome for every recording × feature
family. Two new commands then turn those records plus the built dataset into the publication metadata:
`generate-release-metadata` emits the RO-Crate and the C2M2 submission, and `validate-release-metadata`
refuses anything with a gap and can be pointed at an already-published release to audit it.

The rule the design enforces: every published value is computed from the dataset or read from a run
record. Nothing is authored per release except the narrative text, which is written once and approved
once. That removes the class of defect the 3.0.0 crate exhibits — 286 empty values, 15 of ~98 files
described, two mislabelled files, two duplicated identifiers, one computation dated after its own
outputs, and one software entry with a hand-typed version.

## Technical Context

**Language/Version**: Python 3.12.13 — the interpreter in the release environment
(activated by every script in `post_3.0/v3.1/scripts/`)

**Primary Dependencies**: existing — `senselab 1.3.0`, `torch`/`torchaudio 2.8.0`, `opensmile 2.6.0`,
`praat-parselmouth 0.4.7`, `speech-articulatory-coding 0.1.0`, `ppgs 0.0.9`, `pyannote.audio 4.0.4`,
`pyarrow 23.0.1`, `synapseclient 4.12.0`, `click`, `pandas`. New, as a `metadata` optional-dependency
group — `fairscape-cli>=1.2.9`, `fairscape-models>=1.2.1` (neither present in the env today). External
tool: `cfde-c2m2` CLI, invoked, not imported.

**Storage**: files only. Run records under `<dataset_root>/provenance/`; metadata artifacts at each
dataset root, per RO-Crate's relative-path requirement.

**Testing**: `pytest` (existing `tests/`, `conftest.py`), using the synthetic fixtures already in
`data/` — `sdv_redcap_synthetic_data_1000_rows.csv`, `test_audio_16k_single.wav`. The 3.0.0 comparison
is a documented local procedure against `post_3.0/.../bundled_12_21_25`, not a committed test; no release
data enters the repository.

**Target Platform**: Linux, MIT ORCD/engaging SLURM cluster. Feature extraction runs as an array over
subject-split files (`adult_feature_extraction.sh`, `--array=0-33`, `-n 1`, GPU per task).

**Project Type**: CLI additions to an installed Python package, plus a small upstream change.

**Performance Goals**: none. Metadata generation is a single pass over ~100 files plus ~70 records.
Record writing must not measurably slow extraction, whose per-recording cost is dominated by model
inference.

**Constraints**: PhysioNet is immutable after upload, so generation and validation both run before it.
Records must be writable by 34 concurrent array tasks without coordination. New files must not alter
the behaviour of any existing step, validator, or upload tool — several enumerate files with bare
`rglob`. No participant-identifying content in any record or artifact.

**Scale/Scope**: adult v3.1 — 833 registered / 767 controlled participants, ~904 sessions among shared
participants, ~100k recordings, ~98 published files, 9 feature families, 2 published datasets × 2
cohorts.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Evaluated against `.specify/memory/constitution.md` v1.0.0. This feature exists to close the defects the
constitution was written from, so most principles are satisfied by construction rather than by
constraint.

| Principle | Status | Where |
|---|---|---|
| I. No Silent Empty | **Satisfied by design** | FR-004/FR-021 make an empty file list, placeholder, or unaccounted file a hard failure. R6 requires the C2M2 format lookups that currently default to `''` to be closed. |
| II. Derive, Never Transcribe | **Satisfied by design** | This is the feature's central invariant. Editorial content is the one exception and carries an approval record (FR-018, `release_metadata.yaml`). |
| III. Immutable References Only | **Satisfied by design** | FR-008/FR-009 require resolved model commits and citable software versions. Note the live blocker in Risks: the release env installs `b2aiprep 3.1.0+51.g34174e7.dirty`, which this principle forbids in a published artifact. |
| IV. Executed Scripts Over Prose | **Satisfied** | FR-024 makes `RELEASE.md` reconcile with the v3.1 scripts, with the scripts authoritative. |
| V. Fail Loudly When Publishing | **Satisfied by design** | FR-021 plus `validate-release-metadata`. Includes fixing `verify_sage_contents.py`, which today exits zero on a digest mismatch. |
| VI. Re-running Is Idempotent | **Satisfied** | FR-019. Run records are one-writer-per-file so a resumed array task replaces only its own (R4); the C2M2 writers are de-duplicated on relocation (R6). |
| VII. Nothing Identifying Leaves the Working Tree | **Satisfied** | FR-025/FR-026; run records are internal (FR-001), unit records carry deidentified identifiers only, and crosswalk absence is verified before upload (T050, T051, T056). |
| VIII. Synthetic Fixtures Only | **Satisfied** | Tests use the existing `data/` fixtures; the 3.0.0 comparison is a documented local procedure and no release data enters the repository. |

Two conventions also honoured, both load-bearing for existing users: CLI naming stays verb-first and
hyphenated, matching `generate-audio-features` and `validate-bundled-dataset`; and the documented C2M2
entry points keep working exactly as their README describes.

**Gate result: pass, no violations.** The dirty-version condition under Principle III is a property of
the current environment, not of this design — the design's correct behaviour is to refuse it.

*Post-Phase-1 re-check*: unchanged. Phase 1 added no mechanism that weakens a principle; the record
schemas in `data-model.md` were written to satisfy VI (one writer per file, status promoted on clean
exit) and VII (deidentified identifiers only, internal fields enumerated).

## Project Structure

### Documentation (this feature)

```text
docs/design/release-metadata/
├── spec.md              # feature specification
├── plan.md              # this file
├── research.md          # Phase 0 — R1..R9, all decisions with evidence
├── data-model.md        # Phase 1 — record and artifact shapes
├── quickstart.md        # Phase 1 — end-to-end run on fixtures
├── contracts/
│   └── cli.md           # Phase 1 — the two commands' contracts
└── tasks.md             # Phase 2 — created by /speckit.tasks, not here
```

### Source Code

```text
b2aiprep/
├── src/b2aiprep/
│   ├── cli.py                          # register the two new commands
│   ├── commands.py                     # their click entry points
│   ├── metadata/                       # NEW
│   │   ├── provenance.py               #   record writing + reading, run ids, version/model capture
│   │   ├── records.py                  #   the record schemas and their validation
│   │   ├── rocrate.py                  #   graph construction via fairscape_models; fairscape-cli build
│   │   ├── validate.py                 #   the completeness/consistency checks
│   │   ├── reconcile.py                #   cross-dataset participant and recording overlap
│   │   └── c2m2/                       #   relocated library from external_scripts/c2m2
│   │       ├── mappings.py             #     was c2m2_mappings.py (+ new file-type entries)
│   │       ├── constants.py
│   │       ├── bundle.py               #     was bundle_to_c2m2.py, de-duplicating writes
│   │       ├── controlled.py           #     was controlled_to_c2m2.py
│   │       └── subjects.py             #     was fill_subject_files.py
│   ├── prepare/
│   │   ├── prepare.py                  # record versions/models/config; pin the diarization model
│   │   ├── quality_control.py          # record the QC run
│   │   ├── dataset.py                  # record deidentification counts, in the deid id space
│   │   ├── redcap.py                   # record the source export identity and tier
│   │   └── resources/
│   │       ├── feature_schemas/*.json  # fix ppgs/torchaudio_spectrogram column names; drop frozen versions
│   │       └── release_metadata.yaml   # NEW — the single narrative/editorial source + approval record
│   └── template/dataset_description.json  # add GeneratedBy / SourceDatasets / DatasetDOI
├── external_scripts/c2m2/*.py          # become thin wrappers importing b2aiprep.metadata.c2m2
├── external_scripts/sage_upload_scripts/
│   ├── sage_generate_manifest.py       # decide+record whether provenance/ is uploaded
│   └── verify_sage_contents.py         # fail on digest mismatch; honour the same decision
├── RELEASE.md                          # add the two steps; reconcile with the v3.1 scripts
├── pyproject.toml                      # add the `metadata` optional-dependency group
└── tests/
    ├── test_provenance_records.py      # NEW
    ├── test_rocrate_generation.py      # NEW
    ├── test_metadata_validation.py     # NEW
    └── test_cross_dataset_reconcile.py # NEW

senselab/                               # upstream, only if the diarization model is not pinned locally
└── src/senselab/audio/tasks/speaker_diarization/  # accept and echo an explicit model
```

**Structure Decision**: a new `b2aiprep.metadata` subpackage inside the installed distribution, because
the C2M2 logic must be importable by the pipeline and `external_scripts/` is not part of the wheel
(R6). Everything else is an addition to existing modules at the point where the discarded information is
computed — no module is restructured. The two new commands sit alongside the existing ones in
`cli.py`/`commands.py`, following the established pattern.

## Delivery order

Follows the spec's story priorities. Each step is independently useful.

| Step | Delivers | Depends on |
|---|---|---|
| 1 | `generate-release-metadata` for a bundle, from a bundle-time record only. Retires the notebook. | — |
| 2 | Extraction, QC, deid, and redcap2bids records; diarization model pinned; feature-dictionary fixes. | 1 |
| 3 | `validate-release-metadata`, runnable against a fresh build or a published release. | 1 |
| 4 | Sage dataset crate + measured cross-dataset reconciliation. | 1, 3 |
| 5 | C2M2 from the same records; library relocation; de-duplicated writes. | 1, 2 |

Step 1 before step 2 is deliberate: a bundle-scoped record is enough to fix file coverage, schemas,
identifiers, sizes, and digests — the majority of the 3.0.0 defects — without touching the highest
fan-out part of the pipeline.

## Risks

| Risk | Mitigation |
|---|---|
| The release env installs `b2aiprep 3.1.0+51.g34174e7.dirty`; a dirty version cannot resolve to a citable record, so the generator's own gate would reject it (R2) | Surface this as an explicit, actionable failure early; the env must be installed from a tagged commit before a PhysioNet-bound crate is generated |
| Zenodo DOIs must exist before the release that cites them, and registration is an org-admin action (R9) | Sequence it first; the generator reads DOIs from config and fails loudly when absent |
| Session pseudonyms can differ in length between the two datasets if a truncation collision fires in one tree only (R1) | Length-tolerant matching on the 8-character prefix, plus a reported count of unmatched recordings |
| Provenance files are visible to two bare `rglob('*')` call sites and to the unfiltered Sage verifier (R4, R6) | Each is handled explicitly — described with a correct type, or excluded — and the choice recorded, per FR-028 |
| Two shipped feature dictionaries name columns that do not exist, so "use the shipped dictionary" would publish wrong schemas | Fix both, and make dictionary-vs-file column agreement a validation check (FR-007) |
| C2M2 writers append with no de-duplication, so a re-run doubles the submission | De-duplicate on relocation; covered by FR-019 |

## Complexity Tracking

No constitution gates exist to violate. One deviation from the request is recorded for visibility:

| Deviation | Why needed | Simpler alternative rejected because |
|-----------|------------|--------------------------------------|
| C2M2 library moves into the installed package rather than staying wholly in `external_scripts/` | The pipeline must import the mapping tables and writers; `external_scripts/` is excluded from the wheel, and the scripts only import each other as top-level modules, so they only run from their own directory (`external_scripts/c2m2/README.md:38-41`) | Shelling out with an explicit `cwd` works but leaves the generator unable to reuse the mappings, and preserves the "run from this directory" trap. The documented entry points are kept as wrappers, so no documented invocation changes. |
