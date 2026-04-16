"""QA pipeline composite scoring, review session, and release report (T021, T025–T033).

T021  — ``compute_composite_score``
T025  — ``format_review_card`` (US2 human review display)
T026  — ``record_decision`` / ``load_decided_keys`` (US2 persistence)
T030  — ``compute_quality_report`` (US3 release report)
T031  — ``write_quality_report_json`` / ``write_quality_report_markdown``

Only T021 is implemented in this commit; US2/US3 stubs are left for later
phases.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from b2aiprep.prepare.qa_models import (
    CheckResult,
    CheckType,
    Classification,
    CompositeScore,
    FinalClassification,
    PipelineConfig,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# T021 — Composite score
# ---------------------------------------------------------------------------


def compute_composite_score(
    check_results: list[CheckResult],
    config: PipelineConfig,
    participant_id: str = "",
    session_id: str = "",
    task_name: str = "",
    config_hash: str = "",
    pipeline_version: str = "",
) -> CompositeScore:
    """Compute a :class:`CompositeScore` from a list of :class:`CheckResult` s.

    Three-stage classification (first matching wins):

    1. **Hard gates → FAIL**: any ``hard_gate_triggered=True`` in detail
       OR ``composite_score < fail_max``.
    2. **Forced review → NEEDS_REVIEW**: ``evans_model_flag==1``,
       ``pii_disclosure.transcript_confidence < min_transcript_confidence``,
       or any check classified as ``ERROR``.
    3. **Soft**: ``composite_score ≥ pass_min`` AND all checks ``≥ check_min``
       → ``PASS``; otherwise ``NEEDS_REVIEW``.

    ``composite_confidence`` formula::

        weighted_mean(confidences) × (1 − λ × std_dev(confidences))

    where λ = ``config.confidence_disagreement_penalty`` (default 0.5).

    Args:
        check_results: One :class:`CheckResult` per check type.
        config: :class:`PipelineConfig` with weights and thresholds.
        participant_id: BIDS participant ID (without ``sub-`` prefix).
        session_id: BIDS session ID (without ``ses-`` prefix).
        task_name: BIDS task name.
        config_hash: SHA-256 digest of the config (from
                     :func:`~b2aiprep.prepare.qa_utils.hash_config`).
        pipeline_version: ``b2aiprep`` package version string.

    Returns:
        :class:`CompositeScore` for the audio.
    """
    if not check_results:
        return CompositeScore(
            participant_id=participant_id,
            session_id=session_id,
            task_name=task_name,
            composite_score=0.0,
            composite_confidence=0.0,
            confidence_std_dev=0.0,
            final_classification=FinalClassification.NEEDS_REVIEW,
            check_results=[],
            config_hash=config_hash,
            pipeline_version=pipeline_version,
        )

    weights = config.check_weights

    # Weighted mean of scores and confidences
    total_weight = 0.0
    weighted_score_sum = 0.0
    weighted_conf_sum = 0.0
    confidences: list[float] = []

    for result in check_results:
        ct_key = result.check_type.value  # e.g. "audio_quality"
        w = float(weights.get(ct_key, 0.0))
        total_weight += w
        weighted_score_sum += w * result.score
        weighted_conf_sum += w * result.confidence
        confidences.append(result.confidence)

    if total_weight > 0:
        composite_score = weighted_score_sum / total_weight
        weighted_mean_conf = weighted_conf_sum / total_weight
    else:
        composite_score = 0.0
        weighted_mean_conf = 0.0

    # composite_confidence = weighted_mean × (1 − λ × std_dev(confidences))
    conf_std_dev = float(np.std(confidences)) if len(confidences) > 1 else 0.0
    λ = float(config.confidence_disagreement_penalty)
    composite_confidence = weighted_mean_conf * (1.0 - λ * conf_std_dev)
    composite_confidence = float(max(0.0, min(1.0, composite_confidence)))

    soft = config.soft_score_thresholds
    fail_max = float(soft.get("fail_max", 0.40))
    pass_min = float(soft.get("pass_min", 0.75))
    check_min = float(soft.get("check_min", 0.50))

    # ---- Stage 1: Hard gates → FAIL ----
    hard_gate = any(
        r.detail.get("hard_gate_triggered") is True for r in check_results
    )
    if not hard_gate and composite_score < fail_max:
        hard_gate = True

    if hard_gate:
        final_classification = FinalClassification.FAIL
    else:
        # ---- Stage 2: Forced review ----
        forced_review = False
        for r in check_results:
            if r.classification == Classification.ERROR:
                forced_review = True
                break
            if r.check_type == CheckType.UNCONSENTED_SPEAKERS:
                if r.detail.get("evans_model_flag") == 1:
                    forced_review = True
                    break
            if r.check_type == CheckType.PII_DISCLOSURE:
                tc = float(r.detail.get("transcript_confidence", 1.0))
                if tc < float(config.min_transcript_confidence):
                    forced_review = True
                    break

        if forced_review:
            final_classification = FinalClassification.NEEDS_REVIEW
        else:
            # ---- Stage 3: Soft classification ----
            all_checks_pass_min = all(r.score >= check_min for r in check_results)
            if composite_score >= pass_min and all_checks_pass_min:
                final_classification = FinalClassification.PASS
            else:
                final_classification = FinalClassification.NEEDS_REVIEW

    return CompositeScore(
        participant_id=participant_id,
        session_id=session_id,
        task_name=task_name,
        composite_score=round(composite_score, 6),
        composite_confidence=round(composite_confidence, 6),
        confidence_std_dev=round(conf_std_dev, 6),
        final_classification=final_classification,
        check_results=check_results,
        config_hash=config_hash,
        pipeline_version=pipeline_version,
    )


# ---------------------------------------------------------------------------
# TSV serialisation helpers used by T022 (qa-run)
# ---------------------------------------------------------------------------


def _check_result_to_row(result: CheckResult) -> dict:
    """Serialise a :class:`CheckResult` to a flat dict for TSV output."""
    return {
        "participant_id": result.participant_id,
        "session_id": result.session_id,
        "task_name": result.task_name,
        "check_type": result.check_type.value,
        "score": result.score,
        "confidence": result.confidence,
        "classification": result.classification.value,
        "detail": json.dumps(result.detail),
        "model_versions": json.dumps(result.model_versions),
    }


def _composite_score_to_row(cs: CompositeScore) -> dict:
    """Serialise a :class:`CompositeScore` to a flat dict for TSV output."""
    return {
        "participant_id": cs.participant_id,
        "session_id": cs.session_id,
        "task_name": cs.task_name,
        "composite_score": cs.composite_score,
        "composite_confidence": cs.composite_confidence,
        "confidence_std_dev": cs.confidence_std_dev,
        "final_classification": cs.final_classification.value,
        "config_hash": cs.config_hash,
        "pipeline_version": cs.pipeline_version,
    }


def write_check_results_tsv(
    check_results: list[CheckResult],
    output_dir: Path,
) -> Path:
    """Write ``qa_check_results.tsv`` to *output_dir*."""
    out_path = output_dir / "qa_check_results.tsv"
    rows = [_check_result_to_row(r) for r in check_results]
    if rows:
        pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    else:
        out_path.write_text(
            "\t".join(
                ["participant_id", "session_id", "task_name", "check_type",
                 "score", "confidence", "classification", "detail", "model_versions"]
            ) + "\n"
        )
    return out_path


def write_composite_scores_tsv(
    composite_scores: list[CompositeScore],
    output_dir: Path,
) -> Path:
    """Write ``qa_composite_scores.tsv`` to *output_dir*."""
    out_path = output_dir / "qa_composite_scores.tsv"
    rows = [_composite_score_to_row(cs) for cs in composite_scores]
    if rows:
        pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    else:
        out_path.write_text(
            "\t".join(
                ["participant_id", "session_id", "task_name", "composite_score",
                 "composite_confidence", "confidence_std_dev", "final_classification",
                 "config_hash", "pipeline_version"]
            ) + "\n"
        )
    return out_path


def write_needs_review_queue_tsv(
    composite_scores: list[CompositeScore],
    output_dir: Path,
) -> Path:
    """Write ``needs_review_queue.tsv`` — NEEDS_REVIEW entries only."""
    out_path = output_dir / "needs_review_queue.tsv"
    review_scores = [
        cs for cs in composite_scores
        if cs.final_classification == FinalClassification.NEEDS_REVIEW
    ]
    rows = [_composite_score_to_row(cs) for cs in review_scores]
    if rows:
        pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
    else:
        out_path.write_text(
            "\t".join(
                ["participant_id", "session_id", "task_name", "composite_score",
                 "composite_confidence", "confidence_std_dev", "final_classification",
                 "config_hash", "pipeline_version"]
            ) + "\n"
        )
    return out_path
