"""QA pipeline data model dataclasses.

This module contains only dataclass definitions — no I/O, no business logic.
All entities correspond to the data model described in
specs/001-audio-quality-pipeline/data-model.md.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class CheckType(str, Enum):
    """Which quality check produced a CheckResult."""

    AUDIO_QUALITY = "audio_quality"
    UNCONSENTED_SPEAKERS = "unconsented_speakers"
    PII_DISCLOSURE = "pii_disclosure"
    TASK_COMPLIANCE = "task_compliance"


class Classification(str, Enum):
    """Per-check verdict.

    ``error`` means the check model failed; the audio is routed to human review
    without halting the pipeline.
    """

    PASS = "pass"
    FAIL = "fail"
    NEEDS_REVIEW = "needs_review"
    ERROR = "error"


class FinalClassification(str, Enum):
    """Composite classification after all three evaluation stages."""

    PASS = "pass"
    FAIL = "fail"
    NEEDS_REVIEW = "needs_review"


class Decision(str, Enum):
    """Human reviewer's binary verdict on a needs-review audio."""

    ACCEPT = "accept"
    REJECT = "reject"


# ---------------------------------------------------------------------------
# Core pipeline entities
# ---------------------------------------------------------------------------


@dataclass
class AudioRecord:
    """Minimal per-file record passed to each QA check function.

    Constructed by the ``qa-run`` orchestrator from the BIDS path and features
    file; check functions read ``audio_path`` and ``features_path`` as needed.
    """

    participant_id: str
    session_id: str
    task_name: str
    audio_path: str
    features_path: str


@dataclass
class CheckResult:
    """Output of one quality check for one audio file.

    The ``detail`` dict is check-type-specific; see data-model.md for the
    expected keys per CheckType.  TSV-serialised detail contains no PII text;
    full transcript and PII spans are stored only in the per-audio sidecar.
    """

    participant_id: str
    session_id: str
    task_name: str
    check_type: CheckType
    score: float
    confidence: float
    classification: Classification
    detail: dict[str, Any] = field(default_factory=dict)
    model_versions: dict[str, str] = field(default_factory=dict)


@dataclass
class CompositeScore:
    """Weighted combination of all CheckResults for one audio.

    Classification stages (first matching wins):

    1. Hard gates → FAIL: any ``hard_gate_triggered=True`` or composite < fail_max
    2. Forced review → NEEDS_REVIEW: Evan's model flag, low transcript confidence,
       or any check classified as ERROR
    3. Soft: composite ≥ pass_min AND all checks ≥ check_min → PASS; else NEEDS_REVIEW

    ``composite_confidence`` formula:
        weighted_mean(confidences) × (1 − λ × std_dev(confidences))
    where λ = ``PipelineConfig.confidence_disagreement_penalty``.
    """

    participant_id: str
    session_id: str
    task_name: str
    composite_score: float
    composite_confidence: float
    confidence_std_dev: float
    final_classification: FinalClassification
    check_results: list[CheckResult] = field(default_factory=list)
    config_hash: str = ""
    pipeline_version: str = ""


@dataclass
class ReviewDecision:
    """Human override recorded during the ``qa-review`` CLI session."""

    participant_id: str
    session_id: str
    task_name: str
    decision: Decision
    reviewer_id: str
    reviewed_at: datetime
    notes: Optional[str] = None


@dataclass
class QualityReport:
    """Aggregate summary over a completed batch (automated + human review).

    Written to ``qa_release_report.json`` and ``qa_release_report.md`` in
    the output directory.
    """

    report_version: str
    generated_at: datetime
    pipeline_config_hash: str
    total_audios: int
    auto_pass: int
    auto_fail: int
    needs_review_total: int
    human_accepted: int
    human_rejected: int
    pending_review: int
    released_count: int
    excluded_count: int
    per_check_pass_rates: dict[str, float]
    composite_score_percentiles: dict[str, float]
    claim_confidence: float
    claim_statement: str


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Versioned pipeline configuration.

    Stored as ``qa_pipeline_config_{hash[:8]}.json`` alongside every run's
    output so that results can be reproduced and audited.

    All mutable defaults mirror ``resources/qa_pipeline_config.json``.
    ``created_at`` is set to the current UTC time by ``qa_utils.save_config_snapshot``
    when the snapshot is written; it is ``None`` for the in-memory default.

    ``human_review_timeout_days`` is stored for future enforcement; automatic
    exclusion based on this field is deferred to v2.
    """

    config_version: str = "1.0.0"
    created_at: Optional[datetime] = None
    random_seed: int = 42
    model_versions: dict[str, str] = field(
        default_factory=lambda: {
            "evans_model": (
                "TODO: HuggingFace model path to be added when model is published"
            ),
            "ast_model": "MIT/ast-finetuned-audioset-10-10-0.4593",
        }
    )
    hard_gate_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "snr_min_db": 12.0,
            "clipping_max": 0.05,
            "silence_max": 0.50,
        }
    )
    soft_score_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "pass_min": 0.75,
            "fail_max": 0.40,
            "check_min": 0.50,
        }
    )
    check_weights: dict[str, float] = field(
        default_factory=lambda: {
            "audio_quality": 0.30,
            "unconsented_speakers": 0.25,
            "pii_disclosure": 0.25,
            "task_compliance": 0.20,
        }
    )
    task_compliance_params: dict[str, Any] = field(
        default_factory=lambda: {
            "diadochokinesis": {
                "ddk_rate_expected_hz": [5.0, 7.0],
                "ddk_rate_flag_outside_hz": [2.0, 9.0],
            },
            "phonation": {"min_duration_s": 3.0},
            "conversational": {"min_active_speech_duration_s": 3.0},
        }
    )
    environment_noise_threshold: float = 0.60
    environment_noise_classes: list[str] = field(
        default_factory=lambda: ["Speech", "Crowd", "Music", "Vehicle"]
    )
    confidence_disagreement_penalty: float = 0.50
    min_transcript_confidence: float = 0.70
    human_review_timeout_days: int = 30
    sc_004_review_fraction_warn: float = 0.15
