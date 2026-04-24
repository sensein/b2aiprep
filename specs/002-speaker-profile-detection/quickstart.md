# Quickstart: Speaker Profile-Based Unconsented Speaker Detection

**Date**: 2026-04-24
**Plan**: [plan.md](plan.md)

---

## Prerequisites

1. A BIDS dataset with `_features.pt` files from `generate-audio-features`.
   The feature files must contain `speaker_embedding` (192-dim ECAPA-TDNN) and
   `diarization` fields.
2. `b2aiprep` installed with this feature merged.

---

## Step 1 — Build speaker profiles

```bash
b2aiprep-cli build-speaker-profiles \
    /path/to/bids_dataset \
    /path/to/speaker_profiles
```

This reads every participant's `_features.pt` files, selects recordings with
sufficient active speech, builds a quality-weighted outlier-rejected centroid
embedding per participant, and writes one `speaker_profile.json` per participant.

For a pediatric dataset (participants with age < 14):

```bash
b2aiprep-cli build-speaker-profiles \
    /path/to/bids_dataset \
    /path/to/speaker_profiles \
    --child-age-threshold 14.0 \
    --age-col age
```

On SLURM (participants are sharded across array tasks):

```bash
sbatch --array=1-20 build_profiles_array.sh
```

After the array completes, profiles from all shards are already in
`PROFILES_DIR/sub-*/` — no merge step needed (one profile per participant,
each written by exactly one shard).

---

## Step 2 — Run the QA pipeline with profiles

```bash
b2aiprep-cli qa-run \
    /path/to/bids_dataset \
    /path/to/qa_output \
    --profiles-dir /path/to/speaker_profiles
```

The unconsented-speaker check now uses the pre-built profiles instead of
diarization-only heuristics. Recordings for participants without a profile
(status `insufficient_data`) are automatically routed to `needs_review`.

---

## Step 3 — Inspect per-participant profile quality

Each `speaker_profile.json` contains:

```json
{
  "participant_id": "007ab",
  "model_id": "speechbrain/spkrec-ecapa-voxceleb",
  "num_recordings_used": 12,
  "num_recordings_excluded": 4,
  "total_active_speech_s": 183.4,
  "profile_quality_score": 0.72,
  "profile_status": "ready",
  "age_group": "adult",
  "excluded_recordings": [
    {"task_name": "ddk-1", "session_id": "ses-01", "reason": "task_type_excluded"},
    {"task_name": "breathing-1", "session_id": "ses-01", "reason": "task_type_excluded"}
  ]
}
```

---

## Step 4 — Generate the embedding reliability report (optional)

```bash
b2aiprep-cli embedding-reliability-report \
    /path/to/bids_dataset \
    /path/to/speaker_profiles \
    --output-format both
```

Writes `embedding_reliability_report.md` and `.json` to the profiles directory.
The report shows per-speech-fraction-bin accuracy and recommends threshold updates.

---

## Validation

```bash
pytest tests/test_speaker_profiles.py \
       tests/test_unconsented_speakers_profile.py \
       tests/test_embedding_reliability.py \
       -v
```
