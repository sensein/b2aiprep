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
| `classification` | enum | `pass`, `fail`, `needs_review`, `error` | Per-check verdict (`error` = model failure; audio routed to human review) |
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
  "hard_gate_triggered": false,
  "environment_top_labels": [
    {"label": "Speech", "confidence": 0.71},
    {"label": "Inside, small room", "confidence": 0.18}
  ],
  "environment_noise_flag": false
}
```
`environment_top_labels` is the top-K output of an acoustic scene classifier (e.g., YAMNet or
equivalent). `environment_noise_flag` is `true` when the top-1 label belongs to a configured
noise superclass (Speech/Crowd/Music/Vehicle) with confidence ≥ `environment_noise_threshold`
from `PipelineConfig`.

*unconsented_speakers*:
```json
{
  "num_speakers_diarized": 2,
  "primary_speaker_ratio": 0.83,
  "extra_speaker_count": 1,
  "evans_model_flag": 0,
  "embedding_cosine_similarity_min": 0.71,
  "detected_languages": [
    {"speaker_index": 0, "language": "en", "confidence": 0.97},
    {"speaker_index": 1, "language": "es", "confidence": 0.81}
  ]
}
```
`extra_speaker_count` = `num_speakers_diarized` − 1 (number of non-primary speakers detected;
zero or one is the common case, but N ≥ 2 is possible and is fully reported).
`detected_languages` is populated when a language-ID model is run per diarized segment; an
entry per speaker is included even when all speakers share the same language, to assist human
reviewers in assessing consent for any non-primary voice.

*pii_disclosure*:
```json
{
  "entities_detected": [{"label": "name", "score": 0.82, "char_start": 11, "char_end": 18}],
  "transcript_confidence": 0.88,
  "model_used": "gliner-pii"
}
```

> **Privacy note**: `entities_detected` stores only entity labels, confidence scores, and
> character offsets — **no PII text**. This representation is safe for `qa_check_results.tsv`
> at the BIDS root, which may be included in data shares. The full transcript and PII spans
> with text are stored exclusively in the per-audio JSON sidecar, which is subject to existing
> dataset release controls.

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
  forced to `needs_review` (transcript untrustworthy; PII detection result unreliable)
- If `evans_model_flag == 1` in unconsented_speakers check (pediatric session) →
  classification forced to `needs_review` (independent of diarization score)

---

### CompositeScore

Weighted combination of all four CheckResults for one audio.

| Field | Type | Description |
|-------|------|-------------|
| `participant_id` | string | |
| `session_id` | string | |
| `task_name` | string | |
| `composite_score` | float 0.0–1.0 | Weighted mean of per-check scores |
| `composite_confidence` | float 0.0–1.0 | Weighted mean of per-check confidences, penalized by inter-check disagreement (see formula below) |
| `confidence_std_dev` | float 0.0–0.5 | Std dev of per-check confidences; stored separately for transparency |
| `final_classification` | enum | `pass`, `fail`, `needs_review` |
| `check_results` | list[CheckResult] | One per check type |
| `config_hash` | string | SHA-256 of `PipelineConfig` used |
| `pipeline_version` | string | b2aiprep package version |

**composite_confidence formula**:
```
composite_confidence = weighted_mean(per_check_confidences)
                       × (1 − λ × std_dev(per_check_confidences))
```
Where:
- `weighted_mean` uses the same per-task-category weights as `composite_score`
- `λ` (`confidence_disagreement_penalty`) is a configurable parameter in `PipelineConfig`
  (default: 0.5); it controls how much inter-check disagreement reduces composite confidence
- `std_dev(confidences)` ∈ [0, 0.5] for values in [0, 1], so the penalty factor is bounded
  to [0.75, 1.0] at the default λ=0.5

This ensures that uniformly low-confidence results (e.g., all checks at 0.1) correctly
yield low composite confidence (0.1), not artificially high confidence from low variance.
`confidence_std_dev` is stored alongside for auditability.

**Classification rules** (evaluated in stage order; first matching stage wins):
```
Stage 1 — Hard gates → FAIL (evaluated before scoring):
  any CheckResult has hard_gate_triggered = true
  OR composite_score < 0.40

Stage 2 — Forced review gates → NEEDS_REVIEW (evaluated before soft scoring):
  unconsented_speakers.evans_model_flag == 1  (pediatric session)
  OR pii_disclosure.transcript_confidence < min_transcript_confidence

Stage 3 — Soft classification (only reached if no Stage 1/2 gate triggered):
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
| `environment_noise_threshold` | float | Minimum classifier confidence for a noise-class label to set `environment_noise_flag=true` (default: 0.60) |
| `environment_noise_classes` | list[str] | AudioSet superclass labels treated as noise flags (default: `["Speech", "Crowd", "Music", "Vehicle"]`) |
| `soft_score_thresholds` | dict | pass_min (0.75), fail_max (0.40), check_min (0.50) |
| `check_weights` | dict | Per-task-category weight maps for 4 checks |
| `task_compliance_params` | dict | Per-task-category compliance thresholds (see below) |
| `min_transcript_confidence` | float | Below this → PII result triggers needs_review |
| `confidence_disagreement_penalty` | float | λ in composite_confidence formula (default: 0.5); higher = more penalty for inter-check disagreement |
| `human_review_timeout_days` | int | Days before unreviewed items are excluded (stored in config for future enforcement; automatic exclusion based on this timeout is deferred to v2) |

**`task_compliance_params` structure** (population-tunable defaults):
```json
{
  "diadochokinesis": {
    "ddk_rate_expected_hz": [5.0, 7.0],
    "ddk_rate_flag_outside_hz": [2.0, 9.0]
  },
  "phonation": {
    "min_duration_s": 3.0
  },
  "conversational": {
    "min_active_speech_duration_s": 3.0
  }
}
```
DDK rate bounds must be adjusted for pediatric populations (lower expected rate) and
neurodegenerative conditions (wider acceptable range) before running a batch that includes
those participants.

**Serialisation**: Stored as `qa_pipeline_config_{hash[:8]}.json` in the output directory.
Config SHA-256 hash is embedded in every `CompositeScore` and `QualityReport`.

---

## Output File Layout (BIDS root)

Files are separated by sensitivity. BIDS-root TSVs contain no PII text and are safe for
distribution. Per-audio JSON sidecars contain transcripts and PII spans and are subject to
existing dataset release controls.

```text
{bids_root}/
│  ── BIDS-root files (no PII text; distributable as dataset metadata) ──
├── audio_quality_metrics.tsv          (existing — technical QC metrics per audio)
├── qa_composite_scores.tsv            (new — CompositeScore per audio)
├── qa_check_results.tsv               (new — CheckResult per audio × check type;
│                                        PII detail contains labels + offsets only)
├── needs_review_queue.tsv             (new — audios needing human review)
├── human_review_decisions.tsv         (new — ReviewDecision per reviewed audio)
├── qa_release_report.md               (new — QualityReport in Markdown)
├── qa_release_report.json             (new — QualityReport machine-readable)
├── qa_pipeline_config_{hash[:8]}.json (new — PipelineConfig snapshot)
│
│  ── Per-audio JSON sidecars (sensitive; subject to release controls) ──
└── sub-{id}/ses-{id}/voice/
    └── sub-{id}_ses-{id}_task-{name}_qa.json
        (full transcript, PII spans with text,
         per-check scores, timing metrics)
```
