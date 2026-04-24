# Data Model: Speaker Profile-Based Unconsented Speaker Detection

**Date**: 2026-04-24
**Plan**: [plan.md](plan.md)

---

## Entities

### SpeakerProfile

Persisted as `{PROFILES_DIR}/sub-{participant_id}/speaker_profile.json`.
One file per participant.

| Field | Type | Description |
|-------|------|-------------|
| `participant_id` | str | BIDS participant ID (without `sub-` prefix) |
| `model_id` | str | HuggingFace model ID used to produce the embeddings |
| `embedding_centroid` | list[float] | Weighted centroid of enrollment embeddings (192-dim for ECAPA-TDNN) |
| `num_recordings_used` | int | Count of recordings included in the centroid |
| `num_recordings_excluded` | int | Count of recordings rejected during enrollment |
| `total_active_speech_s` | float | Sum of active speech seconds from included recordings |
| `profile_quality_score` | float | Mean pairwise cosine similarity of included embeddings (0–1); proxy for profile coherence |
| `profile_status` | str | `ready` / `insufficient_data` / `contaminated` |
| `age_group` | str | `adult` / `child` / `unknown`; determines which embedding model was used |
| `included_recordings` | list[str] | BIDS task-name identifiers of recordings used |
| `excluded_recordings` | list[dict] | `{task_name, session_id, reason}` for each excluded recording |
| `created_at` | str | ISO-8601 UTC timestamp |
| `pipeline_config_hash` | str | Config snapshot hash for reproducibility |

**Validation rules**:
- `profile_status = ready` requires `num_recordings_used ≥ min_profile_recordings` (configurable, default 3)
- `embedding_centroid` length must match the model's embedding dimension
- `profile_quality_score < 0.30` triggers `profile_status = contaminated` warning

---

### EmbeddingVerificationResult

Returned by `check_unconsented_speakers()` as part of the existing `CheckResult`
dataclass. No separate persistence; written to `qa_check_results.tsv` via the
existing serialisation path.

New fields added to `CheckResult.detail` for this check:

| detail key | Type | Description |
|------------|------|-------------|
| `cosine_similarity` | float \| null | Cosine similarity between test embedding and profile centroid; null if no profile |
| `active_speech_fraction` | float | Voiced frames / total duration (0–1) |
| `active_speech_s` | float | Absolute voiced duration in seconds |
| `speech_fraction_confidence` | float | Confidence ceiling imposed by low speech fraction (0.30 if < threshold) |
| `profile_status` | str | Status of the participant profile at verification time |
| `diarization_primary_ratio` | float | Fraction of diarized speech attributed to primary speaker |
| `num_speakers_diarized` | int | Number of distinct speakers detected by diarization |
| `extra_speaker_count` | int | `num_speakers_diarized − 1` |
| `embedding_model` | str | Model ID used for the test-time embedding |
| `enrollment_n` | int | Number of recordings used in the profile centroid |
| `age_group` | str | `adult` / `child` / `unknown` |

---

### EmbeddingReliabilityReport

Written by the `embedding-reliability-report` CLI command (US3).
Persisted as both JSON and Markdown in the profiles output directory.

| Field | Type | Description |
|-------|------|-------------|
| `report_version` | str | Semantic version of the report format |
| `generated_at` | str | ISO-8601 UTC timestamp |
| `dataset_bids_dir` | str | Path to BIDS dataset used |
| `num_participants` | int | Total participants evaluated |
| `num_recordings` | int | Total recordings evaluated |
| `speech_fraction_bins` | list[tuple] | Bin boundaries, e.g., `[(0, 0.15), (0.15, 0.30), ...]` |
| `per_bin_stats` | list[dict] | Per-bin: `{bin, n, mean_cosine_same, mean_cosine_diff, accuracy, fpr}` |
| `adult_per_bin_stats` | list[dict] | Same structure filtered to adult participants |
| `child_per_bin_stats` | list[dict] | Same structure filtered to child participants |
| `recommended_low_confidence_threshold` | float | Speech fraction below which confidence is capped |
| `recommended_min_enrollment_duration_s` | float | Active speech duration floor for enrollment |
| `knee_point_fraction` | float | Speech fraction at which accuracy drops > 15 pp vs. top bin |

---

## State Transitions: SpeakerProfile.profile_status

```
[enrollment runs]
      │
      ├─ num_usable_recordings < min_profile_recordings
      │       → insufficient_data
      │
      ├─ num_usable_recordings ≥ min_profile_recordings
      │       → ready
      │       │
      │       └─ profile_quality_score < contamination_threshold (0.30)
      │               → contaminated  (advisory; verification still runs)
      │
      └─ qa-run detects unconsented speakers in all usable enrollment recordings
              → contaminated  (advisory)
```

---

## Files Written per Participant

```
PROFILES_DIR/
└── sub-{participant_id}/
    └── speaker_profile.json        # SpeakerProfile entity
```

```
PROFILES_DIR/
├── build_speaker_profiles_config_{hash8}.json  # PipelineConfig snapshot
└── embedding_reliability_report.json           # EmbeddingReliabilityReport (US3)
└── embedding_reliability_report.md             # Human-readable version (US3)
```

---

## Relationship to Existing QA Data Model

- `SpeakerProfile` is a new standalone artifact, not a `CheckResult` subtype.
- `EmbeddingVerificationResult` extends the existing `CheckResult.detail` dict;
  the `CheckResult` dataclass itself is unchanged.
- The `AudioRecord` dataclass gains one optional field: `participant_age_years: Optional[float]`
  (read from BIDS `participants.tsv` during `build-speaker-profiles` and `qa-run`).
- `PipelineConfig` gains new fields in a new `speaker_profile` namespace:
  `min_profile_recordings`, `min_active_speech_s`, `low_confidence_speech_fraction`,
  `outlier_rejection_std_multiplier`, `contamination_quality_threshold`,
  `child_age_threshold_years`, `child_embedding_model_id`.
