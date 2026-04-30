# Quickstart: Speaker Profile-Based Unconsented Speaker Detection

**Date**: 2026-04-30 (updated from 2026-04-24)
**Plan**: [plan.md](plan.md)

---

## Prerequisites

1. A BIDS dataset with `_features.pt` files from `generate-audio-features`.
   The feature files must contain `speaker_embedding` (192-dim ECAPA-TDNN),
   `sparc["spk_emb"]` (64-dim), and `diarization` fields.
2. `b2aiprep` installed with this feature merged.

---

## Step 1 — Build speaker profiles

```bash
b2aiprep-cli build-speaker-profiles \
    /path/to/bids_dataset \
    /path/to/speaker_profiles
```

This reads every participant's `_features.pt` files, applies task-based gating
(case-insensitive prefix exclusion), builds quality-weighted outlier-rejected
centroids for both ECAPA-TDNN (192-dim) and SPARC (64-dim) embeddings, and
writes one `speaker_profile.json` per participant.

On SLURM (participants sharded across array tasks):

```bash
sbatch --array=1-20 build_profiles_array.sh
```

No merge step needed — one profile per participant, each written by exactly one shard.

---

## Step 2 — Run the QA pipeline with profiles

```bash
b2aiprep-cli qa-run \
    /path/to/bids_dataset \
    /path/to/qa_output \
    --profiles-dir /path/to/speaker_profiles
```

The unconsented-speaker check now computes two independent cosine similarity scores
(ECAPA-TDNN and SPARC) and applies OR logic: a recording is flagged `needs_review`
if either score falls below its configured threshold. Participants without a profile
(`insufficient_data`) are automatically routed to `needs_review`.

---

## Step 3 — Inspect per-participant profile quality

Each `speaker_profile.json` contains both centroids and per-embedding quality scores:

```json
{
  "participant_id": "007ab",
  "ecapa_model_id": "speechbrain/spkrec-ecapa-voxceleb",
  "sparc_model_id": "senselab/sparc-multi",
  "num_recordings_used": 12,
  "num_recordings_excluded": 4,
  "total_active_speech_s": 183.4,
  "ecapa_profile_quality_score": 0.72,
  "sparc_profile_quality_score": 0.68,
  "profile_status": "ready",
  "age_group": "adult",
  "ecapa_embedding_centroid": [0.021, -0.013, ...],
  "sparc_embedding_centroid": [0.041, 0.002, ...],
  "excluded_recordings": [
    {"task_name": "Diadochokinesis-PA", "session_id": "ses-01", "reason": "task_prefix_excluded"},
    {"task_name": "Respiration-and-cough-Breath-1", "session_id": "ses-01", "reason": "task_prefix_excluded"}
  ]
}
```

---

## Step 4 — Generate the embedding reliability report (optional, US3)

```bash
b2aiprep-cli embedding-reliability-report \
    /path/to/bids_dataset \
    /path/to/speaker_profiles \
    --output-format both \
    --intruder-ratios 0.10,0.20,0.40
```

Generates synthetic mixtures (two enrolled participants mixed at controlled ratios),
computes operating characteristic curves for ECAPA-TDNN, SPARC, and OR-combined
scoring, and recommends thresholds. Writes
`embedding_reliability_report.md` and `.json` to the profiles directory.

---

## Validation

```bash
pytest tests/test_speaker_profiles.py \
       tests/test_unconsented_speakers_profile.py \
       tests/test_embedding_reliability.py \
       -v
```
