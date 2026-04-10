# Data Model: Audio Quality Assurance Pipeline

**Date**: 2026-04-09
**Plan**: [plan.md](plan.md)

---

## Entities

### AudioRecord

Input unit for the pipeline. Derived from the existing BIDS structure — no new storage needed.

| Field | Type | Source | Description |
|-------|------|--------|-------------|
| `participant_id` | string | BIDS path | e.g., `sub-001` |
| `session_id` | string | BIDS path | e.g., `ses-01` |
| `task_name` | string | BIDS metadata | e.g., `harvard-sentences-list-1-1` |
| `audio_path` | Path | BIDS tree | Absolute path to `.wav` file |
| `features_path` | Path | BIDS tree | Path to precomputed `_features.pt` file |
| `task_instructions` | string | `audio_task_descriptions.json` | Participant instructions |
| `task_prompts` | list[string] | `audio_task_descriptions.json` | Expected spoken content |
| `task_category` | string | derived | One of: reading, phonation, diadochokinesis, pitch_glide, fluency, breathing, story, recitation, conversational, cognitive, loudness |
| `is_pediatric` | bool | participant metadata | Triggers Evan's model check |

**Validation**: `audio_path` must exist; `task_name` must be present in
`audio_task_descriptions.json`; `features_path` must exist if `deep_checks` mode enabled.

---

### CheckResult

Output of one quality check for one audio file.

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `participant_id` | string | — | Links to AudioRecord |
| `session_id` | string | — | Links to AudioRecord |
| `task_name` | string | — | Links to AudioRecord |
| `check_type` | enum | `audio_quality`, `unconsented_speakers`, `pii_disclosure`, `task_compliance` | Which check produced this |
| `score` | float | 0.0–1.0 | How well the audio passed this check (1.0 = perfect) |
| `confidence` | float | 0.0–1.0 | Model's confidence in `score` |
| `classification` | enum | `pass`, `fail`, `needs_review` | Per-check verdict |
| `detail` | dict | — | Check-specific fields (see below) |
| `model_versions` | dict | — | Versions/hashes of models used for this check |

**check_type-specific `detail` fields**:

*audio_quality*:
```json
{
  "proportion_clipped": 0.002,
  "proportion_silent": 0.12,
  "peak_snr_db": 24.3,
  "spectral_gating_snr_db": 19.8,
  "amplitude_modulation_depth": 0.73,
  "hard_gate_triggered": false
}
```

*unconsented_speakers*:
```json
{
  "num_speakers_diarized": 1,
  "primary_speaker_ratio": 0.97,
  "evans_model_flag": 0,
  "embedding_cosine_similarity_min": 0.91
}
```

*pii_disclosure*:
```json
{
  "entities_detected": [{"text": "...", "label": "name", "score": 0.82}],
  "transcript_confidence": 0.88,
  "model_used": "gliner-pii",
  "redacted_transcript": "My name is [NAME]."
}
```

*task_compliance*:
```json
{
  "compliance_tier": "A",
  "wer": 0.05,
  "phoneme_match_score": null,
  "active_speech_duration_s": 4.2,
  "llm_compliance": null
}
```

**State transitions**:
- score computed → classification assigned using thresholds from `PipelineConfig`
- If any `hard_gate_triggered=true` → classification forced to `fail`
- If `transcript_confidence < min_transcript_confidence` in PII check → classification
  forced to `needs_review`

---

### CompositeScore

Weighted combination of all four CheckResults for one audio.

| Field | Type | Description |
|-------|------|-------------|
| `participant_id` | string | |
| `session_id` | string | |
| `task_name` | string | |
| `composite_score` | float 0.0–1.0 | Weighted mean of per-check scores |
| `composite_confidence` | float 0.0–1.0 | 1 − std_dev of per-check confidences |
| `final_classification` | enum | `pass`, `fail`, `needs_review` |
| `check_results` | list[CheckResult] | One per check type |
| `config_hash` | string | SHA-256 of `PipelineConfig` used |
| `pipeline_version` | string | b2aiprep package version |

**Classification rules**:
```
fail:         any hard gate triggered  OR  composite_score < 0.40
pass:         composite_score ≥ 0.75  AND  all check scores ≥ 0.50
needs_review: all other cases
```

---

### ReviewDecision

Human override recorded during the review CLI session.

| Field | Type | Description |
|-------|------|-------------|
| `participant_id` | string | |
| `session_id` | string | |
| `task_name` | string | |
| `decision` | enum | `accept`, `reject` |
| `reviewer_id` | string | Username or initials of reviewer |
| `reviewed_at` | ISO-8601 datetime | When the decision was recorded |
| `notes` | string | Optional free-text note |

---

### QualityReport

Aggregate summary over a completed batch (automated + human review).

| Field | Type | Description |
|-------|------|-------------|
| `report_version` | string | Increments with each regeneration |
| `generated_at` | ISO-8601 datetime | |
| `pipeline_config_hash` | string | Config used for this batch |
| `total_audios` | int | |
| `auto_pass` | int | PASS without human review |
| `auto_fail` | int | FAIL without human review |
| `needs_review_total` | int | Routed to human review |
| `human_accepted` | int | Needs-review → accepted by human |
| `human_rejected` | int | Needs-review → rejected by human |
| `pending_review` | int | Needs-review, not yet reviewed |
| `released_count` | int | auto_pass + human_accepted |
| `excluded_count` | int | auto_fail + human_rejected |
| `per_check_pass_rates` | dict | `{check_type: pass_rate}` |
| `composite_score_percentiles` | dict | p10, p25, p50, p75, p90 |
| `claim_confidence` | float | Confidence in released-batch quality claim |
| `claim_statement` | string | e.g., "At 94% confidence, 97.3% of released audios pass all checks." |

---

### PipelineConfig

Versioned configuration artifact. Stored as JSON alongside every run's output so results can
be reproduced exactly.

| Field | Type | Description |
|-------|------|-------------|
| `config_version` | string | Semantic version of config schema |
| `created_at` | ISO-8601 datetime | |
| `model_versions` | dict | `{model_name: version_or_hash}` |
| `random_seed` | int | Applied to all models (default: 42) |
| `hard_gate_thresholds` | dict | clipping_max, silence_max, snr_min (dB) |
| `soft_score_thresholds` | dict | pass_min (0.75), fail_max (0.40), check_min (0.50) |
| `check_weights` | dict | Per-task-category weight maps for 4 checks |
| `min_transcript_confidence` | float | Below this → PII result triggers needs_review |
| `human_review_timeout_days` | int | Days before unreviewed items are excluded |

**Serialisation**: Stored as `qa_pipeline_config_{hash[:8]}.json` in the output directory.
Config SHA-256 hash is embedded in every `CompositeScore` and `QualityReport`.

---

## Output File Layout (BIDS root)

```text
{bids_root}/
├── audio_quality_metrics.tsv          (existing — technical QC metrics per audio)
├── qa_composite_scores.tsv            (new — CompositeScore per audio)
├── qa_check_results.tsv               (new — CheckResult per audio × check type)
├── needs_review_queue.tsv             (new — audios needing human review)
├── human_review_decisions.tsv         (new — ReviewDecision per reviewed audio)
├── qa_release_report.md               (new — QualityReport in Markdown)
├── qa_release_report.json             (new — QualityReport machine-readable)
└── qa_pipeline_config_{hash[:8]}.json (new — PipelineConfig snapshot)
```
