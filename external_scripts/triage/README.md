# Triage → `audio_quality_annotations.tsv` and `consensus_transcripts.parquet`

`prep_audio_quality_annotations.py` turns the senselab triage output (`derivatives/senselab-triage/`) into two per-recording tables for a BIDS source tree. It also lists the recordings triage marked for review or discard.

## Inputs
- **`--triage-dir`:** the senselab-triage folder: `triage.tsv`, `triage.json`, `triage_decisions.parquet`, `recording_vectors.parquet`, and each recording's `summary/summary.json`.
- **`--triage-bids`:** the BIDS tree triage read. `source_path` is relative to it. Each row's `recording_id` comes from that file's `_recording-metadata.json` sidecar.
- **`--target-bids`:** the BIDS tree the release is built from. Every row must be in its `phenotype/task/recording.tsv`, with the same participant and session and an audio file with a sidecar. The task label comes from that file name. Any disagreement stops the run.

## Outputs (`--outdir`)
- **`audio_quality_annotations.tsv`:**
  - One row per recording, keyed by `recording_id`, with IDs as in the BIDS tree (no `sub-`/`ses-`).
  - Task timing and recording duration at full precision, from the parquets.
  - `task_family` recomputed from the current task label (`task_name`).
- **`consensus_transcripts.parquet`:** one row per consensus word, from `summary.json` (`evidence.consensus_transcript_tokens`): the word, its timing, whether it is a non-speech tag, how the two recognisers aligned (agreement / variant / insertion) and each recogniser's word and timing. The whole recording, not cut to the task.
- **`removal_candidates.tsv`:** recordings with a `--remove-verdicts` verdict (default `review,discard`), for adding to the deidentify removal list. Deidentify never reads triage, and this script removes nothing.
- **`prep_report.json`:** counts, the triage commit and config hash, and every task label that changed.

Both tables cover every recording. Copy them to the BIDS root before deidentifying. Deidentify keeps the rows of the recordings each tier releases, and transcripts only where the recording's audio is released (not free speech or open response); the bundle copies them into `features/`.

## Column descriptions
They are b2aiprep resources: `src/b2aiprep/prepare/resources/audio_quality_annotations.json` and `consensus_transcripts.json`, which deidentify writes next to each table. This script stops if a column is not described there, if a description has no column, or if `triage.json` now describes one of Satra's columns differently. Update the resource on purpose when triage changes.

## Columns left out
- **Annotations, file paths:** `source_path`, `task_audio_*`, `figure`.
- **Annotations, the pipeline's decision and rationale:** `verdict`, `release`, `release_reason`, `reason`, `reasons`, `issues_n`, `run_status`, `missing`, `decisive_evidence`, `annotations`.
- **Annotations, per-kind lists that are empty on released rows:** `interference_in_task`, `fault_in_task`.
- **Transcripts:** `variants` (the two readings), `sources` (which readings are filled), `onset_spread_s` / `offset_spread_s` / `temporal_uncertainty_s` (from the two timings), `entity_id`, `provenance`.

## Example
```bash
conda activate b2aiprep_test
python external_scripts/triage/prep_audio_quality_annotations.py \
  --triage-dir /orcd/data/satra/002/datasets/b2aivoice/4.0-release/adult/bids_adult_2026_09_04/derivatives/senselab-triage \
  --triage-bids /orcd/data/satra/002/datasets/b2aivoice/4.0-release/adult/bids_adult_2026_09_04 \
  --target-bids /orcd/scratch/bcs/002/wilke18/release_build_2026_10_14/adult/source \
  --outdir /orcd/data/satra/002/datasets/b2aivoice/4.0-release/qa/triage_prep_2026_10_14/out
```
