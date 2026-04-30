# CLI Contracts: Speaker Profile-Based Unconsented Speaker Detection

**Date**: 2026-04-30 (updated from 2026-04-24)
**Plan**: [plan.md](../plan.md)

---

## Command 1: `build-speaker-profiles`

Build per-participant speaker profiles (dual ECAPA-TDNN + SPARC centroids) from
pre-computed feature files. Must run before `qa-run` to enable profile-based
unconsented-speaker detection.

```
b2aiprep-cli build-speaker-profiles BIDS_DIR PROFILES_DIR [OPTIONS]
```

**Arguments**:
- `BIDS_DIR` — Root of the BIDS dataset. Must contain `_features.pt` files produced
  by a prior `generate-audio-features` run with both `speaker_embedding` and `sparc`
  fields present.
- `PROFILES_DIR` — Directory where speaker profiles are written. Created if absent.

**Options**:
```
--pipeline-config PATH        Path to PipelineConfig JSON. If omitted, built-in defaults used.
--task-exclude TEXT           Comma-separated task name prefixes to always exclude from
                              enrollment (case-insensitive prefix match).
                              [default: Diadochokinesis,Prolonged-vowel,Maximum-phonation-time,
                               Respiration-and-cough,Glides,Loudness,long-sounds,silly-sounds,
                               repeat-words]
--min-active-speech FLOAT     Minimum active speech seconds per recording for enrollment.
                              [default: 3.0]
--min-recordings INT          Minimum usable recordings to produce a ready profile.
                              [default: 3]
--age-col TEXT                Column name in participants.tsv containing participant age
                              in years. [default: age]
--part INT                    1-based shard index for SLURM array jobs.
--num-parts INT               Total number of shards.
--log-level TEXT              DEBUG / INFO / WARNING / ERROR. [default: INFO]
```

**Outputs written to PROFILES_DIR**:
- `sub-{participant_id}/speaker_profile.json` — one per participant; contains both
  `ecapa_embedding_centroid` (192-dim) and `sparc_embedding_centroid` (64-dim)
- `build_speaker_profiles_config_{hash8}.json` — PipelineConfig snapshot

**Exit codes**:
- `0` — Completed (some profiles may have status `insufficient_data`)
- `1` — Fatal error (BIDS_DIR not found, no `.pt` files found, `.pt` files missing
  `speaker_embedding` or `sparc` keys)

**Pre-condition**: `generate-audio-features` must have been run so `_features.pt`
files exist with both `speaker_embedding` and `sparc["spk_emb"]` fields.

---

## Command 2: `qa-run` (updated)

No new CLI options added beyond `--profiles-dir`. The unconsented-speaker check now
computes two independent cosine scores (ECAPA-TDNN and SPARC) and applies OR logic.

**New option added to existing `qa-run`**:
```
--profiles-dir PATH   Directory containing pre-built speaker profiles from
                      build-speaker-profiles. If omitted, all recordings are
                      classified as needs_review for the unconsented-speaker check.
```

**Updated output fields in `qa_check_results.tsv`** for `check_type=unconsented_speakers`:
The `detail` JSON column now includes all fields from `EmbeddingVerificationResult`
in `data-model.md`, including `ecapa_cosine_similarity`, `sparc_cosine_similarity`,
and `or_flag`.

---

## Command 3: `embedding-reliability-report` (US3)

Generate the embedding reliability research report covering ECAPA-TDNN, SPARC, and
OR-combined operating characteristics, including synthetic mixture evaluation.

```
b2aiprep-cli embedding-reliability-report BIDS_DIR PROFILES_DIR [OPTIONS]
```

**Arguments**:
- `BIDS_DIR` — BIDS dataset root with `_features.pt` files.
- `PROFILES_DIR` — Directory containing pre-built speaker profiles.

**Options**:
```
--output-dir PATH              Where to write the report. [default: PROFILES_DIR]
--speech-fraction-bins TEXT    Comma-separated bin edges, e.g. "0,0.15,0.30,0.50,1.0".
                               [default: 0,0.15,0.30,0.50,1.0]
--output-format TEXT           markdown, json, or both. [default: both]
--intruder-ratios TEXT         Comma-separated intruder duration ratios for synthetic
                               mixtures, e.g. "0.10,0.20,0.40". [default: 0.10,0.20,0.40]
--intruder-snr-db TEXT         Comma-separated SNR values (dB) for synthetic mixtures.
                               [default: 0,10,20]
--pipeline-config PATH         PipelineConfig JSON.
```

**Outputs written to output-dir**:
- `embedding_reliability_report.json` — Full report including operating curves for
  ECAPA-TDNN, SPARC, and OR-combined
- `embedding_reliability_report.md` — Human-readable summary with threshold recommendations
- `synthetic_mixtures/` — Generated mixture audio (optional; removed after report if
  `--keep-mixtures` not set)

**Exit codes**:
- `0` — Report generated
- `1` — Required inputs not found (no profiles, no `.pt` files)
