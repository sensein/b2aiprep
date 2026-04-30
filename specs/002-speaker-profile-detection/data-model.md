# Data Model: Speaker Profile-Based Unconsented Speaker Detection

**Date**: 2026-04-30 (updated from 2026-04-24)
**Plan**: [plan.md](plan.md)

---

## Entities

### SpeakerProfile

Persisted as `{PROFILES_DIR}/sub-{participant_id}/speaker_profile.json`.
One file per participant. Stores **two independent speaker centroids**: ECAPA-TDNN
(192-dim) and SPARC `spk_emb` (64-dim).

| Field | Type | Description |
|-------|------|-------------|
| `participant_id` | str | BIDS participant ID (without `sub-` prefix) |
| `ecapa_model_id` | str | HuggingFace model ID for ECAPA-TDNN embeddings |
| `sparc_model_id` | str | SPARC model identifier |
| `ecapa_embedding_centroid` | list[float] | Quality-weighted outlier-rejected centroid of ECAPA-TDNN embeddings (192-dim) |
| `sparc_embedding_centroid` | list[float] | Quality-weighted outlier-rejected centroid of SPARC `spk_emb` (64-dim) |
| `num_recordings_used` | int | Count of recordings included in both centroids |
| `num_recordings_excluded` | int | Count of recordings rejected during enrollment |
| `total_active_speech_s` | float | Sum of active speech seconds from included recordings |
| `ecapa_profile_quality_score` | float | Mean pairwise cosine similarity of included ECAPA-TDNN embeddings (0–1) |
| `sparc_profile_quality_score` | float | Mean pairwise cosine similarity of included SPARC embeddings (0–1) |
| `profile_status` | str | `ready` / `insufficient_data` / `contaminated` |
| `age_group` | str | `adult` / `child` / `unknown`; informational only (does not affect model selection) |
| `included_recordings` | list[str] | BIDS task-name identifiers of recordings used |
| `excluded_recordings` | list[dict] | `{task_name, session_id, reason}` for each excluded recording |
| `created_at` | str | ISO-8601 UTC timestamp |
| `pipeline_config_hash` | str | Config snapshot hash for reproducibility |

**Validation rules**:
- `profile_status = ready` requires `num_recordings_used ≥ min_profile_recordings` (default 3)
- `ecapa_embedding_centroid` length must be 192; `sparc_embedding_centroid` length must be 64
- `profile_status = contaminated` when either quality score < contamination threshold (0.30)
- Same set of recordings used for both centroids (enrollment gating applied once)

---

### EmbeddingVerificationResult

Returned by `check_unconsented_speakers()` as part of the existing `CheckResult`
dataclass. Both ECAPA-TDNN and SPARC scores are reported; OR logic determines
the final classification.

New fields added to `CheckResult.detail` for this check:

| detail key | Type | Description |
|------------|------|-------------|
| `ecapa_cosine_similarity` | float \| null | Cosine similarity between ECAPA-TDNN test embedding and profile centroid; null if no profile |
| `sparc_cosine_similarity` | float \| null | Cosine similarity between SPARC `spk_emb` and profile centroid; null if no profile |
| `or_flag` | bool | True if either cosine score is below its threshold (triggers needs_review) |
| `active_speech_fraction` | float | Voiced frames / total duration (0–1) |
| `active_speech_s` | float | Absolute voiced duration in seconds |
| `speech_fraction_confidence` | float | Confidence ceiling imposed by low speech fraction |
| `profile_status` | str | Status of the participant profile at verification time |
| `diarization_primary_ratio` | float | Fraction of diarized speech attributed to primary speaker |
| `num_speakers_diarized` | int | Number of distinct speakers detected by diarization |
| `extra_speaker_count` | int | `num_speakers_diarized − 1` |
| `ecapa_model_id` | str | ECAPA-TDNN model ID used |
| `sparc_model_id` | str | SPARC model ID used |
| `enrollment_n` | int | Number of recordings used in the profile centroids |
| `age_group` | str | `adult` / `child` / `unknown` (informational) |

---

### EmbeddingReliabilityReport

Written by the `embedding-reliability-report` CLI command (US3).
Covers ECAPA-TDNN, SPARC, and OR-combined performance including operating
characteristic curves derived from synthetic mixture evaluation.

| Field | Type | Description |
|-------|------|-------------|
| `report_version` | str | Semantic version of the report format |
| `generated_at` | str | ISO-8601 UTC timestamp |
| `dataset_bids_dir` | str | Path to BIDS dataset used |
| `num_participants` | int | Total participants evaluated |
| `num_recordings` | int | Total recordings evaluated |
| `num_synthetic_mixtures` | int | Synthetic mixture recordings used for FN/FP evaluation |
| `speech_fraction_bins` | list[tuple] | Bin boundaries, e.g., `[(0, 0.15), (0.15, 0.30), ...]` |
| `ecapa_per_bin_stats` | list[dict] | Per-bin: `{bin, n, mean_cosine_same, mean_cosine_diff, accuracy, fpr}` |
| `sparc_per_bin_stats` | list[dict] | Same structure for SPARC |
| `or_per_bin_stats` | list[dict] | Same structure for OR-combined |
| `ecapa_operating_curve` | list[dict] | `{threshold, fnr, fpr, review_fraction}` for ECAPA-TDNN |
| `sparc_operating_curve` | list[dict] | Same for SPARC |
| `or_operating_curve` | list[dict] | Same for OR combination |
| `recommended_ecapa_threshold` | float | ECAPA-TDNN threshold achieving ≤5% FNR |
| `recommended_sparc_threshold` | float | SPARC threshold achieving ≤5% FNR |
| `recommended_low_confidence_threshold` | float | Speech fraction below which confidence is capped |
| `recommended_min_enrollment_duration_s` | float | Active speech duration floor for enrollment |
| `knee_point_fraction` | float | Speech fraction where accuracy drops > 15 pp vs. top bin |
| `adult_subgroup_stats` | dict | Operating curves for adult participants |
| `child_subgroup_stats` | dict | Operating curves for child participants (adult-intruder detection rate) |

---

### SyntheticMixture (evaluation artifact, not persisted in production)

Created by the `embedding-reliability-report` command for operating characteristic
evaluation. Not written to `PROFILES_DIR` in normal pipeline runs.

| Field | Type | Description |
|-------|------|-------------|
| `target_participant_id` | str | Participant whose recording is the base (enrolled speaker) |
| `intruder_participant_id` | str | Participant whose audio is mixed in |
| `base_recording_path` | str | Source recording for the target |
| `intruder_segment_path` | str | Source recording for the intruder |
| `intruder_duration_ratio` | float | Fraction of the base recording replaced by intruder audio |
| `intruder_snr_db` | float | SNR of intruder relative to base |
| `label` | str | `positive` (intruder present) or `negative` (solo) |
| `mixed_audio_path` | str | Path to the generated mixture |

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
      │       └─ either ecapa_profile_quality_score OR sparc_profile_quality_score
      │               < contamination_threshold (0.30)
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
    └── speaker_profile.json        # SpeakerProfile entity (two centroids)
```

```
PROFILES_DIR/
├── build_speaker_profiles_config_{hash8}.json  # PipelineConfig snapshot
├── embedding_reliability_report.json           # EmbeddingReliabilityReport (US3)
├── embedding_reliability_report.md             # Human-readable version (US3)
└── synthetic_mixtures/                         # Created by US3 eval; optional
    └── {target_pid}_{intruder_pid}_{ratio}.wav
```

---

## Relationship to Existing QA Data Model

- `SpeakerProfile` is a new standalone artifact, not a `CheckResult` subtype.
- `EmbeddingVerificationResult` extends the existing `CheckResult.detail` dict;
  the `CheckResult` dataclass itself is unchanged.
- The `AudioRecord` dataclass gains one optional field: `participant_age_years: Optional[float]`
  (informational; does not affect embedding model selection).
- `PipelineConfig` gains new fields in a new `speaker_profile` namespace:
  `min_profile_recordings`, `min_active_speech_s`, `low_confidence_speech_fraction`,
  `outlier_rejection_std_multiplier`, `contamination_quality_threshold`,
  `ecapa_cosine_threshold`, `sparc_cosine_threshold`,
  `excluded_task_prefixes` (list of prefix strings, case-insensitive prefix match).
