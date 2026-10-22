# Triage as a release step, with a B2AI dataset profile

**Status**: proposal, no code · **Created**: 2026-10-22

Paired with senselab `specs/20261022-triage-dataset-profile/design.md` (branch
`design/triage-dataset-profile`). There the triage workflow stays in senselab, and what it knows about
Bridge2AI-Voice moves into a dataset profile that the caller supplies. This document is b2aiprep's half:
it holds the B2AI profile and runs triage as a step of the release.

This work lands on `release/v4.1.0`, the branch that collects the v4.1 changes before they merge into `main`.

## Where things stand (4.0.0)

- **Triage ran outside b2aiprep.** It was run with senselab's triage workflow on the adult WAV tree
  `bids_adult_2026_09_04`, on the `feat/task-events` branch at `93958bd`, config hash `b24b5a9d918a3110`.
  Its output is in `derivatives/senselab-triage/`.
- **The release used it through hand steps** (`RELEASE.md`, "Files made by other tools"):
  1. `external_scripts/triage/prep_audio_quality_annotations.py` turned `triage.tsv` into
     `audio_quality_annotations.tsv`, `consensus_transcripts.parquet` and `removal_candidates.tsv`.
     It took `recording_id` from the triage tree's sidecars and the task names from the FLAC tree's file
     names.
  2. The two tables were copied to the root of the BIDS tree before deidentify. Deidentify keeps the rows
     of released recordings and remaps the IDs (`_RECORDING_TABLES`, `prepare/dataset.py:266`;
     `_deidentify_recording_tables`, `:5430`).
  3. The adult removal lists were built by hand: every review and discard verdict, plus 51 earlier IDs.
     The IDs came in mixed case, and deidentify lowercases them (`load_audio_recording_ids_to_remove`,
     `:4417`). The lists were then promoted into the deidentify configs.
- **senselab keeps a copy of b2aiprep data.** It ships a verbatim copy of b2aiprep's task registry,
  frozen at commit `8c43256` (senselab `data/b2ai_task_registry/PROVENANCE.yaml`). The source is
  `src/b2aiprep/prepare/resources/task_registry/`.

## What moves to b2aiprep

1. **The B2AI triage profile:** the files the senselab design lists.
   - `tasks.yaml` holds the 48 adult task rows.
   - `families.yaml` holds the family sets and kinds.
   - `lexicons/` holds the Cinderella cast, the Stroop colours and the item categories.
   - `fitted/` holds the per-family values with the runs they came from.
   - `policy/` holds the verdict and redaction switches, the PII word lists and the reviewer's task
     guidance.
   - `profile.yaml` holds name, version and provenance.

   Location (decided 2026-10-22): in this package, at `src/b2aiprep/prepare/resources/triage_profile/adult/`.
   A pediatric profile is the planned next extension, as `pediatric/`.
2. **The hint builder.** senselab's `b2ai_hints.py` and `task_registry.py` move here. They read this
   package's own registry, so senselab's copy goes away.
3. **The prep script.** `external_scripts/triage/prep_audio_quality_annotations.py` moves into `src` as
   part of the step.

## The step

A command, for example `b2aiprep run-triage BIDS_DIR --profile adult --outdir ...`, between
"3. Quality Control" and "4. Deidentification" in `RELEASE.md`. It does the following:

1. **Builds the manifest from the tree that is released.** Participant, session, task and `recording_id`
   come from `phenotype/task/recording.tsv` and the sidecars, so senselab never parses file names. Triage
   runs on the released (FLAC) audio, which removes the 4.0.0 step of mapping from WAV to FLAC.
2. **Runs senselab triage with the profile.** It runs as SLURM GPU array jobs: the workflow needs the
   Qwen3 ASR and aligner, AST/HeAR/YAMNet, pyannote, MossFormer2, Gemma for review and Ollama for the
   second opinion.
3. **Checks the run.** `pipeline_commit`, `config_hash`, the profile version and `schema_version` must each
   have a single value. Every recording in `recording.tsv` must have exactly one row.
4. **Writes the release tables** that the prep script writes today.
5. **Writes candidate removal lists, not live ones.** Review and discard IDs are written lower-cased and
   de-duplicated, next to the deidentify configs as candidates. A person still promotes them, as in
   4.0.0: the verdict feeds a release decision but does not make it.

## Where the outputs go

- Triage's own output and the release tables go under `derivatives/senselab-triage/`, identified by
  `dataset_description.json` GeneratedBy rather than by folder name.
- Deidentify and the bundle then read them from there instead of from the root. That change belongs to the
  wider "derivative outputs in `derivatives/`" work, which also covers `audio_quality_metrics.*` and the
  `.pt` features.
- Until then the step can keep writing the two tables to the root, as 4.0.0 does.
- Satra's per-recording run folders are never copied into a release tree: they hold unredacted
  transcripts.

## Dependencies and order

- **senselab version.** `pyproject.toml` pins `senselab[articulatory]~=1.3.0` (line 24), but triage exists
  only on senselab `alpha`.
  - senselab publishes its alphas to PyPI. 1.4.0a7 (2026-10-21) is `origin/alpha`, triage included.
  - The plan is to point b2aiprep at it: `senselab[articulatory]>=1.4.0a7,<1.4.1`, then `~=1.4.0` once
    1.4.0 is released. 1.4.0 satisfies `>=1.4.0a7`.
  - The `<1.4.1` bound matters: a requirement that names a pre-release lets pip pick later
    pre-releases too.
  - PyPI does not accept git-URL dependencies, so a git pin is not an option.
  - The base pin has to move, because an optional extra cannot loosen it.
- **Heavy models.** Put them behind an optional extra, for example `b2aiprep[triage]`, so the base install
  and CI do not pull in the triage models.
- **Order of changes:**
  1. While senselab does its steps 1–5, b2aiprep needs nothing.
  2. At senselab step 6, the profile and hint builder land here in one PR together with the senselab
     removal.
  3. The step itself follows.
  4. The move to `derivatives/` is separate.

## Acceptance

Each check compares against 4.0.0. Every difference is explained, and the reviewer accepts it case by case:
some are expected, such as recordings now triaged on FLAC rather than WAV, or names that b2aiprep cleans up.
An unexplained difference blocks the change.

- **Verdicts:** run from this package on the 4.0.0 adult source, the profile gives verdicts for a fixed
  sample that are compared with 4.0.0's using senselab's `replay_diff`.
- **Tables:** the step's tables are compared row for row with the 4.0.0 `audio_quality_annotations.tsv` and
  `consensus_transcripts.parquet`. 4.0.0's own check found WAV and FLAC differing only where the audio
  clipped.
- **Removal candidates:** they are compared with the 4.0.0 review and discard lists after lower-casing.

## Decided (2026-10-22)

- **The profile lives in this package.**
- **Fitted values belong to whoever runs triage.** For B2AI that is the release's triage runner, and any
  change goes through a reviewed PR. Fitting values programmatically for a new dataset is a follow-up
  issue after the senselab refactor. It needs train/test split support that senselab does not have yet.
- **Who does the work.** The first attempt is by Jordan Wilke (with Claude Code). The senselab triage
  owner reviews and changes as needed.
- **Pediatric** is the planned next extension.

## Open questions

1. **Re-running and re-using.** Does a release re-run triage, or replay earlier verdicts? Replaying saves
   GPU time, but is valid only when the audio and the profile hash are unchanged.
2. **Who promotes the candidate removal lists, and with what review.** In 4.0.0 this was done by hand.
