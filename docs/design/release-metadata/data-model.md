# Phase 1 Data Model: Generated Release Metadata

Two things the pipeline writes (run records), and what the generator produces from them.

---

## Written by the pipeline

### Layout

```
<dataset_root>/provenance/
  runs/<step>-<run_id>.json      # one per invocation
  units/<step>-<run_id>.tsv      # one per invocation, only for per-recording steps
```

`<dataset_root>` is the BIDS tree for extraction/QC/deid, and the bundle for bundling. One writer per
file, so the 34 concurrent array tasks never contend and re-running one task replaces exactly its own
two files. ~70 files per release at v3.1 scale.

### RunRecord — `runs/<step>-<run_id>.json`

| Field | Type | Notes |
|---|---|---|
| `schema_version` | str | `"1"` |
| `step` | enum | `redcap2bids` \| `generate-audio-features` \| `quality-control` \| `deidentify` \| `bundle` |
| `run_id` | str | stable per invocation; includes `SLURM_JOB_ID`/`SLURM_ARRAY_TASK_ID` when present so a resumed task overwrites its own record |
| `started`, `ended` | str | ISO 8601, UTC |
| `status` | enum | `complete` \| `partial` \| `failed`. Written `partial` at start, promoted on clean exit, so an aborted task never reads as complete |
| `run_by` | str | operator; never published (FR-001) |
| `host`, `slurm` | obj | node, job id, array task id; never published |
| `command` | str | argv as invoked |
| `software` | list | `{name, version, source}` for `b2aiprep`, `senselab`, and each backend, read from `importlib.metadata` |
| `models` | list | `{role, provider, repo, requested_revision, resolved_sha}`; `role` ∈ `transcription`, `speaker_embedding`, `diarization` |
| `config` | obj | effective configuration as applied, per family where it differs |
| `source_export` | obj | `{path, sha256, record_count, access_tier}` — the RedCap export; determines participant coverage (FR-012) |
| `filters` | list | `{kind, policy_source, sha256, affected_count}`; `kind` ∈ `task_selection`, `sensitivity`, `participant_removal` |
| `inputs`, `outputs` | list | `{path, size, sha256?}`; digests required for files that will be published |
| `counts` | obj | processed / skipped / failed, and per family |

Only `software`, `models`, `config`, `filters`, `source_export.access_tier`, `counts`, and output
identity reach a published artifact. `run_by`, `host`, `slurm`, and absolute paths do not.

### UnitRecord — `units/<step>-<run_id>.tsv`

One row per `(recording, feature family)` the invocation touched. Exists to make absence accountable,
not to attribute versions — attribution comes from the RunRecord.

| Column | Notes |
|---|---|
| `participant_id`, `session_id`, `task_name` | deidentified identifiers only (FR-027) |
| `family` | `opensmile` \| `praat_parselmouth` \| `torchaudio` \| `torchaudio_squim` \| `sparc` \| `ppgs` \| `transcription` \| `diarization` \| `speaker_embedding` |
| `outcome` | `computed` \| `preserved` \| `skipped` \| `failed` |
| `reason` | required unless `outcome == computed`; from the closed set below |
| `run_id` | the invocation that produced this row |

**Closed reason set** (FR-011), each traceable to an existing code path:

| Reason | Source |
|---|---|
| `participant_not_in_tier` | participant absent from this dataset's RedCap export |
| `participant_removed` | `participants_to_remove.json` (`dataset.py:1731`) |
| `recording_removed_sensitive` | `audio_filestems_to_remove.json` (`dataset.py:1748`) |
| `family_withheld_sensitive` | per-recording family withholding (`dataset.py:71-75`, applied `:2263-2268`) |
| `task_not_selected` | `audio_tasks_to_include.json` (`dataset.py:1764`) |
| `audio_load_failed` | `prepare.py:157-160` |
| `computation_failed` | backend raised (`prepare.py:311-314`, `:330-335`, `:351-354`) |
| `value_all_missing` | all-NaN tensor (`bundle_data.py:49`, `:149-151`) |
| `feature_absent` | family key absent from the `.pt` (`bundle_data.py:144-146`) |
| `preserved_prior_invocation` | `--update` kept an existing family (`prepare.py:189-244`) |

**Accounting invariant** (FR-011): for each published family,
`published_rows + unit_rows_with_outcome != computed == total (recording × family) combinations in the source`.
Validation computes both sides and fails on inequality.

---

## Produced by the generator

### Distribution

| Field | Value for the two datasets |
|---|---|
| `name` | feature-only bundle \| deidentified audio |
| `host` | PhysioNet \| Sage/Synapse |
| `mutable_after_upload` | `false` \| `true` |
| `access_tier` | registered \| controlled |
| `cohort` | adult \| pediatric |
| `content_type` | derived features \| audio |
| `source_export` | from the RunRecord of the `redcap2bids` run that built it |
| `identifier` | PhysioNet DOI (must resolve before upload) \| Synapse synId |
| `sibling_identifiers` | resolving identifier for each other distribution (FR-013, FR-015) |

### PublishedFile

`path` (relative to the dataset root), `size`, `sha256`, `media_type`, `role`
(`feature_table` \| `phenotype_table` \| `data_dictionary` \| `documentation` \| `quality_metrics`),
and the activity that produced it. Digest and size come from the bundle RunRecord's outputs, not
recomputed (FR-002, and the C2M2 reuse in FR-016).

### Activity

One per distinct (software set, configuration) across a release's invocations. Carries `software`,
`models`, `config`, `started`/`ended`, and the files it generated. **If the UnitRecords cannot resolve
a published family to exactly one Activity, generation fails** (R5, FR-010) — the case that arises when
a failed run is resumed with `--update` after an environment change.

### SoftwareComponent

`name`, `version` as installed at run time, `concept_doi`, `version_doi`. A version containing a local
segment such as `+51.g34174e7.dirty` cannot resolve to a citable record and fails the gate — which the
release env currently would (R2).

### Coverage

Per feature family: participant, session, and recording counts, each differing between families
(FR-012). Per release: participant overlap between distributions in all three directions, computed as a
set operation on published identifiers, plus recording overlap matched on
`(participant, session_prefix_8, task)` with the count of unmatched recordings reported (R1). No
containment rule is asserted; the observed direction is stated per cohort (FR-014).

### EditorialContent

The single narrative source (`resources/release_metadata.yaml`): authorship with ORCIDs, rights, ethics,
collection method, limitations, intended uses, citation, funder with a structured award identifier — plus
`approved_by`, `approved_on`, and a digest of the approved text. Publication to PhysioNet is blocked
without it.

### Artifacts written

At each dataset root, so relative paths resolve (R3):

```
<dataset_root>/
  ro-crate-metadata.json          # generator, via fairscape_models
  ro-crate-datasheet.html         # fairscape-cli build datasheet
  ro-crate-preview.html           # fairscape-cli build subcrate
  ro-crate-croissant.json         #   "
  ro-crate-prov-graph.{json,html} #   "
  ro-crate-merkle-tree.json       #   "
  ai_ready_score.json             # fairscape-cli build datasheet
```

C2M2 goes to a separate output directory, never inside a published dataset, and never alongside the
identifier map or crosswalk files (FR-029).

---

## State transitions

**RunRecord.status**: `partial` at start → `complete` on clean exit, or `failed` on a handled error. A
record left `partial` means the process died; validation treats it as a gap (FR-021).

**Distribution readiness**: `built` → `metadata_generated` → `validated` → `published`. Only `validated`
may be uploaded to PhysioNet, since a correction there costs a new version. The Synapse distribution may
re-enter `metadata_generated` after publication to add the PhysioNet identifier once it resolves
(FR-015, and the amendment path in the spec's Story 4).
