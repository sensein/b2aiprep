# Triage for a Bridge2AI-Voice release

How to rerun everything that produced the adult release's triage results: per-recording quality decisions, the
`audio_quality_annotations.tsv` and `consensus_transcripts.parquet` tables, and the review/discard lists.

**Who does what.** senselab supplies the triage pieces as installed modules
(`python -m senselab.audio.workflows.triage.drivers.<name>`); it does not fix their order. This folder is B2AI's
side: the order the release ran them in (`chain/`), the settings it ran them with (`chain/configs/`), and the step
that turns triage's output into release tables (`prep_audio_quality_annotations.py`). The dataset profile and hint
builder are b2aiprep package resources (`src/b2aiprep/prepare/resources/triage_profile/adult/`).

## Running the chain

1. Copy `chain/settings.example.sh` somewhere outside the release trees and fill it in: the BIDS tree to read, a
   scratch `WORK` folder, the Python that has senselab installed, caches, the reviewer models, partitions.
2. Start from the top: `bash chain/submit.sh /abs/path/settings.sh inputs`.
   Each stage is a Slurm job or array; a small link job submits the next stage once every task of the current one
   succeeded, so only one stage is queued at a time (the per-user submit cap, not the array size, is the limit).
   `$WORK/jobs.txt` lists every submission; job logs are in `$WORK/slurm/`.
3. If a task fails, the chain stops there (its link is cancelled). Read the log, fix the cause, and submit that
   stage again: `bash chain/submit.sh settings.sh <stage>`. Every pass resumes: what finished is not redone.
   `--only` submits a stage without going on to the next.
4. The `check` stage ends the chain: one decision row per finished run, one commit, one configuration, and every
   pass's status counts (bad stems in `$WORK/logs/<pass>.bad_stems.txt`).
5. Publish by hand once the tables were looked at: `bash chain/publish.sh settings.sh DST` copies the run roots and
   tables into `DST` (a `derivatives/senselab-triage` folder; build it on scratch, never inside a built release tree)
   and verifies the copy.
6. Then run `prep_audio_quality_annotations.py` on the published folder (below).

A canary first: set `STEMS_FILE` to a few dozen stems (two participants with two sessions each covers every stage)
and small slice counts.

## The stages, in order

| Stage | What it does | senselab driver | Config | GPU |
|---|---|---|---|---|
| `inputs` | lists the tree's `sub-*/ses-*/audio/*.{wav,flac}` (or `STEMS_FILE`) | `make_input_manifest.py` (here) | | |
| `fresh` | the whole graph per recording; reviewer off | `triage_corpus run` | profile | |
| `runs` | the finished runs as the passes' manifest, with identity columns; stops if a recording has no finished run (`ALLOW_MISSING=1` to go on) | `triage_corpus manifest` | | |
| `session` | SESSION's floor, per BIDS session | `extend_session_floor` | profile | |
| `replay` | re-decide from TAXONOMY with the session floors (exit 3 = some reviewer readings to redo, a result) | `extend_replay_decisions` | profile | |
| `review_rows` | every row the replay did not read (`--all-unread`: the fresh runs were never reviewed), split into words / no words | `triage_review_rows rereads`, `split` | | |
| `nowords` | REVIEW and SECOND_OPINION recorded without a model for rows with no words | `extend_llm_review`, `extend_second_opinion` | `release_run_config.yaml` | |
| `review_opinion` | REVIEW through a pinned vLLM server, then SECOND_OPINION through the pinned Ollama server, over the same rows on the same GPU (the opinion reads the review's readings) | `extend_llm_review`, then `extend_second_opinion` | `review_vllm.yaml`, `second_opinion.yaml` | yes |
| `cohort_collect`, `cohort_quantiles` | COHORT's corpus facts, then the corpus quantiles | `extend_cohort collect`, `quantiles` | profile | |
| `cohort_apply`, `cohort_consistency` | speaker enrollment and COHORT readings per participant, then the consistency report | `extend_cohort apply`, `consistency` | `release_run_config.yaml` | |
| `refold` | VERDICT again over every reading | `extend_refold` | `release_run_config.yaml` | |
| `task_audio` | streams cut to the task extent | `extend_task_audio` | `release_run_config.yaml` | |
| `task_squim` | SQUIM over 10 s windows of the task extent (a fresh run does not compute it) | `extend_task_squim` | `release_run_config.yaml` | |
| `vectors`, `vectors_merge` | one row per recording (shards, then merged), decision and evidence tables | `triage_recording_vectors` | | |
| `table` | `triage.tsv` and `triage.json` | `triage_table` | | |
| `check` | coverage and status counts | `check_run.py` (here) | | |

Every pass gets `--profile` (b2aiprep's adult profile). The configs in `chain/configs/` are the released run's:
`release_run_config.yaml` turns on the reviewer and second opinion settings and `verdict.llm_redaction_withholds:
true` (the released decisions were folded with it on; the profile has it on too since 2026-11-10).

`review` and `opinion` are the same two steps as separate stages (`review` → `opinion` → `cohort_collect`), as the
released run had them; submit `review` by hand to use them.

Not in the chain (run by hand when needed): the review pages (`triage_review_page`, `free_speech_review_page`,
`triage_review_serve`), `fit_profile`, `check_identity`, `triage_replay_diff`, the repair passes
(`extend_clip_amplitudes`, `extend_withdraw_clips`, `extend_quality`, `extend_diarization`, `extend_ppg_praat`,
`extend_reprocessed_outputs`, `extend_background_floor`) and the evaluation drivers. The drivers' package docstring
(`senselab.audio.workflows.triage.drivers`) lists them.

## How the released v4.0.0 result was made

The released adult triage (62,550 recordings) was not one run of this chain. It is Satra's rounds in
`/orcd/scratch/bcs/002/satra/r18d_20261010`, each building on stores of earlier ones, at senselab 93958bd (before
dataset profiles existed), then the SQUIM recompute at 451af99c:
- stores from several fresh runs (r18 and earlier, 22,607 re-run in r18d, 13 patched in r34);
- the session floor from r31 (r34 for 15 patch sessions); the no-words review markings from r33/r34;
- r37: a full replay, review and second opinion of the 15,060 rows that needed it, COHORT applied with **r34's
  cohort quantiles** (not recomputed), refold, task audio, vectors, `triage.tsv`, published 2026-10-13;
- squim38: SQUIM over the task extent, vectors and `triage.tsv` rebuilt, published 2026-10-27.

This chain runs the same steps once, in the same order, from a fresh run, and computes the cohort quantiles itself.
Its results will not match v4.0.0 byte for byte: the code has changed since 93958bd.

## Before the next release's run

- **Use the new tree's names.** The v4 runs used `bids_adult_2026_09_04`'s file names; run fresh on the renamed tree,
  never replay the old stores. Rename the profile's `tasks.yaml` entries and per-task values to match, and keep an
  old→new map for comparisons.
- **`recording_id` and the registry fingerprint** go into the hints (`prepare/triage_hints.py`) just before that run.
- **SQUIM:** `task_squim` is in the chain; rewrite the three `squim_*` descriptions in
  `prepare/resources/audio_quality_annotations.json` (they still describe the per-span median).
- **senselab version:** this chain needs a senselab whose package carries the triage drivers
  (`senselab.audio.workflows.triage.drivers`); no published senselab does yet. Until one does, point
  `SENSELAB_SRC` at a checkout that has them.

## After triage: `audio_quality_annotations.tsv` and `consensus_transcripts.parquet`

`prep_audio_quality_annotations.py` turns the senselab triage output (`derivatives/senselab-triage/`) into two per-recording tables for a BIDS source tree. It also lists the recordings triage marked for review or discard.

### Inputs
- **`--triage-dir`:** the senselab-triage folder: `triage.tsv`, `triage.json`, `triage_decisions.parquet`, `recording_vectors.parquet`, and each recording's `summary/summary.json`.
- **`--triage-bids`:** the BIDS tree triage read. `source_path` is relative to it. Each row's `recording_id` comes from that file's `_recording-metadata.json` sidecar.
- **`--target-bids`:** the BIDS tree the release is built from. Every row must be in its `phenotype/task/recording.tsv`, with the same participant and session and an audio file with a sidecar. The task label comes from that file name. Any disagreement stops the run.

### Outputs (`--outdir`)
- **`audio_quality_annotations.tsv`:**
  - One row per recording, keyed by `recording_id`, with IDs as in the BIDS tree (no `sub-`/`ses-`).
  - Task timing and recording duration at full precision, from the parquets.
  - `task_family` recomputed from the current task label (`task_name`).
- **`consensus_transcripts.parquet`:** one row per consensus word, from `summary.json` (`evidence.consensus_transcript_tokens`): the word, its timing, whether it is a non-speech tag, how the two recognisers aligned (agreement / variant / insertion) and each recogniser's word and timing. The whole recording, not cut to the task.
- **`removal_candidates.tsv`:** recordings with a `--remove-verdicts` verdict (default `review,discard`), for adding to the deidentify removal list. Deidentify never reads triage, and this script removes nothing.
- **`prep_report.json`:** counts, the triage commit and config hash, and every task label that changed.

Both tables cover every recording. Copy them to the BIDS root before deidentifying. Deidentify keeps the rows of the recordings each tier releases, and transcripts only where the recording's audio is released (not free speech or open response); the bundle copies them into `features/`.

### Column descriptions
They are b2aiprep resources: `src/b2aiprep/prepare/resources/audio_quality_annotations.json` and `consensus_transcripts.json`, which deidentify writes next to each table. This script stops if a column is not described there, if a description has no column, or if `triage.json` now describes one of Satra's columns differently. Update the resource on purpose when triage changes.

### Columns left out
- **Annotations, file paths:** `source_path`, `task_audio_*`, `figure`.
- **Annotations, the pipeline's decision and rationale:** `verdict`, `release`, `release_reason`, `reason`, `reasons`, `issues_n`, `run_status`, `missing`, `decisive_evidence`, `annotations`.
- **Annotations, per-kind lists that are empty on released rows:** `interference_in_task`, `fault_in_task`.
- **Transcripts:** `variants` (the two readings), `sources` (which readings are filled), `onset_spread_s` / `offset_spread_s` / `temporal_uncertainty_s` (from the two timings), `entity_id`, `provenance`.

### Example
```bash
conda activate b2aiprep_test
python external_scripts/triage/prep_audio_quality_annotations.py \
  --triage-dir /orcd/data/satra/002/datasets/b2aivoice/4.0-release/adult/bids_adult_2026_09_04/derivatives/senselab-triage \
  --triage-bids /orcd/data/satra/002/datasets/b2aivoice/4.0-release/adult/bids_adult_2026_09_04 \
  --target-bids /orcd/scratch/bcs/002/wilke18/release_build_2026_10_14/adult/source \
  --outdir /orcd/data/satra/002/datasets/b2aivoice/4.0-release/qa/triage_prep_2026_10_14/out
```
