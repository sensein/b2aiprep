# Triage → `audio_quality_annotations.tsv`

`prep_audio_quality_annotations.py` turns the senselab triage output (`derivatives/senselab-triage/`) into a per-recording table for a BIDS source tree. It also lists the recordings triage marked for review or discard.

## Inputs
- **`--triage-dir`:** the senselab-triage folder: `triage.tsv`, `triage.json`, `triage_decisions.parquet`, `recording_vectors.parquet`.
- **`--triage-bids`:** the BIDS tree triage read. `source_path` is relative to it. Each row's `recording_id` comes from that file's `_recording-metadata.json` sidecar.
- **`--target-bids`:** the BIDS tree the release is built from. Every row must be in its `phenotype/task/recording.tsv`, with the same participant and session and an audio file with a sidecar. The task label comes from that file name. Any disagreement stops the run.

## Outputs (`--outdir`)
- **`audio_quality_annotations.tsv` / `.json`:**
  - One row per recording, keyed by `recording_id`, with IDs as in the BIDS tree (no `sub-`/`ses-`).
  - Task timing and recording duration at full precision, from the parquets.
  - `task_family` recomputed from the current task label.
  - The JSON carries triage's own column descriptions.
- **`removal_candidates.tsv`:** recordings with a `--remove-verdicts` verdict (default `review,discard`), for adding to the deidentify removal list. Deidentify never reads triage, and this script removes nothing.
- **`prep_report.json`:** counts, the triage commit and config hash, and every task label that changed.

## Columns left out
- **File paths:** `source_path`, `task_audio_*`, `figure`.
- **The pipeline's decision and rationale:** `verdict`, `release`, `release_reason`, `reason`, `reasons`, `issues_n`, `run_status`, `missing`, `decisive_evidence`, `annotations`.
- **Per-kind lists that are empty on released rows:** `interference_in_task`, `fault_in_task`.

## Example
```bash
conda activate b2aiprep_test
python external_scripts/triage/prep_audio_quality_annotations.py \
  --triage-dir /orcd/data/satra/002/datasets/b2aivoice/4.0-release/adult/bids_adult_2026_09_04/derivatives/senselab-triage \
  --triage-bids /orcd/data/satra/002/datasets/b2aivoice/4.0-release/adult/bids_adult_2026_09_04 \
  --target-bids /orcd/scratch/bcs/002/wilke18/b2ai_flac_2026_10_10/adult/bids_adult_2026_09_16 \
  --outdir /orcd/data/satra/002/datasets/b2aivoice/4.0-release/qa/triage_prep_2026_10_13
```
