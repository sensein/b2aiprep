# CLI Contracts: Speaker Profile-Based Unconsented Speaker Detection

**Date**: 2026-04-24
**Plan**: [plan.md](../plan.md)

---

## Command 1: `build-speaker-profiles`

Build per-participant speaker profiles from pre-computed feature files.
Must be run before `qa-run` to enable profile-based unconsented-speaker detection.

```
b2aiprep-cli build-speaker-profiles BIDS_DIR PROFILES_DIR [OPTIONS]
```

**Arguments**:
- `BIDS_DIR` — Root of the BIDS dataset. Must contain `_features.pt` files produced
  by a prior `generate-audio-features` run.
- `PROFILES_DIR` — Directory where speaker profiles are written. Created if absent.

**Options**:
```
--pipeline-config PATH   Path to PipelineConfig JSON. If omitted, built-in defaults used.
--task-include TEXT      Comma-separated task name patterns to include in enrollment
                         (default: all speech tasks with sufficient active speech).
--task-exclude TEXT      Comma-separated task name patterns to always exclude
                         (default: ddk, breathing, silence, long-sounds).
--min-active-speech FLOAT  Minimum active speech seconds per recording for enrollment.
                           [default: 3.0]
--min-recordings INT     Minimum usable recordings to produce a ready profile.
                         [default: 3]
--age-col TEXT           Column name in participants.tsv containing participant age
                         in years. [default: age]
--child-age-threshold FLOAT  Age in years below which child embedding model is used.
                              [default: 14.0]
--part INT               1-based shard index for SLURM array jobs.
--num-parts INT          Total number of shards.
--log-level TEXT         DEBUG / INFO / WARNING / ERROR. [default: INFO]
```

**Outputs written to PROFILES_DIR**:
- `sub-{participant_id}/speaker_profile.json` — one per participant
- `build_speaker_profiles_config_{hash8}.json` — PipelineConfig snapshot

**Exit codes**:
- `0` — Completed (some profiles may have status `insufficient_data`)
- `1` — Fatal error (BIDS_DIR not found, no `.pt` files found)

**Pre-condition**: `generate-audio-features` must have been run so `_features.pt`
files exist. `build-speaker-profiles` reads `speaker_embedding` and `diarization`
from those files; it does not extract new embeddings.

---

## Command 2: `qa-run` (updated)

No new CLI options added. The unconsented-speaker check is updated automatically
when `--profiles-dir` is provided (or auto-discovered in the output directory).

**New option added to existing `qa-run`**:
```
--profiles-dir PATH   Directory containing pre-built speaker profiles from
                      build-speaker-profiles. If omitted, all recordings are
                      classified as needs_review for the unconsented-speaker check.
```

**Updated output fields in `qa_check_results.tsv`** for `check_type=unconsented_speakers`:
The `detail` JSON column now includes the fields documented in `data-model.md`
under EmbeddingVerificationResult.

---

## Command 3: `embedding-reliability-report` (US3)

Generate the embedding reliability research report.

```
b2aiprep-cli embedding-reliability-report BIDS_DIR PROFILES_DIR [OPTIONS]
```

**Arguments**:
- `BIDS_DIR` — BIDS dataset root with `_features.pt` files.
- `PROFILES_DIR` — Directory containing pre-built speaker profiles.

**Options**:
```
--output-dir PATH       Where to write the report. [default: PROFILES_DIR]
--speech-fraction-bins TEXT  Comma-separated bin edges, e.g. "0,0.15,0.30,0.50,1.0".
                              [default: 0,0.15,0.30,0.50,1.0]
--output-format TEXT    markdown, json, or both. [default: both]
--pipeline-config PATH  PipelineConfig JSON.
```

**Outputs written to output-dir**:
- `embedding_reliability_report.json`
- `embedding_reliability_report.md`

**Exit codes**:
- `0` — Report generated
- `1` — Required inputs not found
