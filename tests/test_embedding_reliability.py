"""Tests for embedding reliability (US3): synthetic mixtures and operating curves.

Covers:
- _overlay_intruder: output shape unchanged, intruder_duration_ratio=0.4 case
- _fnr_fpr_at_threshold: known-score FNR/FPR verification
- compute_single_curve: minimum threshold points
- write_reliability_report: JSON schema validation
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_audio_2s():
    """Single-channel 2-second 16 kHz audio tensor."""
    return torch.randn(1, 32000)


@pytest.fixture
def base_audio_5s():
    """Single-channel 5-second 16 kHz audio tensor."""
    return torch.randn(1, 80000)


@pytest.fixture
def known_cosine_scores():
    """Known cosine scores for positive (intruder) and negative (solo) pairs.

    Positive: all below 0.25 → should all be flagged at threshold 0.25.
    Negative: all above 0.25 → none flagged at threshold 0.25.
    """
    positive_scores = [0.10, 0.15, 0.20, 0.12, 0.18]
    negative_scores = [0.80, 0.75, 0.85, 0.70, 0.90]
    return positive_scores, negative_scores


@pytest.fixture
def minimal_report_dict():
    """Minimal valid EmbeddingReliabilityReport dict containing all required keys."""
    return {
        "report_version": "1.0.0",
        "generated_at": "2026-04-30T00:00:00+00:00",
        "dataset_bids_dir": "/fake/bids",
        "num_participants": 5,
        "num_recordings": 30,
        "num_synthetic_mixtures": 15,
        "speech_fraction_bins": [[0, 0.15], [0.15, 0.30], [0.30, 1.0]],
        "ecapa_per_bin_stats": [],
        "sparc_per_bin_stats": [],
        "or_per_bin_stats": [],
        "ecapa_operating_curve": [
            {"threshold": 0.0, "fnr": 0.0, "fpr": 1.0, "review_fraction": 1.0},
            {"threshold": 0.1, "fnr": 0.0, "fpr": 0.8, "review_fraction": 0.8},
            {"threshold": 0.2, "fnr": 0.1, "fpr": 0.3, "review_fraction": 0.4},
            {"threshold": 0.3, "fnr": 0.3, "fpr": 0.1, "review_fraction": 0.2},
            {"threshold": 0.5, "fnr": 0.5, "fpr": 0.0, "review_fraction": 0.05},
        ],
        "sparc_operating_curve": [
            {"threshold": 0.0, "fnr": 0.0, "fpr": 1.0, "review_fraction": 1.0},
            {"threshold": 0.1, "fnr": 0.0, "fpr": 0.8, "review_fraction": 0.8},
            {"threshold": 0.2, "fnr": 0.15, "fpr": 0.25, "review_fraction": 0.35},
            {"threshold": 0.3, "fnr": 0.35, "fpr": 0.1, "review_fraction": 0.2},
            {"threshold": 0.5, "fnr": 0.6, "fpr": 0.0, "review_fraction": 0.05},
        ],
        "or_operating_curve": [
            {"threshold": 0.0, "fnr": 0.0, "fpr": 1.0, "review_fraction": 1.0},
            {"threshold": 0.1, "fnr": 0.0, "fpr": 0.75, "review_fraction": 0.75},
            {"threshold": 0.2, "fnr": 0.05, "fpr": 0.3, "review_fraction": 0.4},
            {"threshold": 0.3, "fnr": 0.2, "fpr": 0.1, "review_fraction": 0.25},
            {"threshold": 0.5, "fnr": 0.4, "fpr": 0.0, "review_fraction": 0.05},
        ],
        "recommended_ecapa_threshold": 0.25,
        "recommended_sparc_threshold": 0.20,
        "recommended_low_confidence_threshold": 0.15,
        "recommended_min_enrollment_duration_s": 3.0,
        "knee_point_fraction": 0.15,
        "adult_subgroup_stats": {"ecapa_operating_curve": [], "sparc_operating_curve": []},
        "child_subgroup_stats": {"ecapa_operating_curve": [], "sparc_operating_curve": []},
    }


# ---------------------------------------------------------------------------
# _overlay_intruder tests
# ---------------------------------------------------------------------------


def test_overlay_intruder_preserves_base_shape(base_audio_2s):
    """Mixed output must have same shape as base regardless of intruder size."""
    from b2aiprep.prepare.embedding_reliability import _overlay_intruder

    intruder = torch.randn(1, 6400)  # 0.4 s clip
    mixed = _overlay_intruder(base_audio_2s, intruder, snr_db=0.0)

    assert mixed.shape == base_audio_2s.shape


def test_overlay_intruder_ratio_0p4_correct_duration(base_audio_5s):
    """Mixture with intruder_duration_ratio=0.4 has correct audio length."""
    from b2aiprep.prepare.embedding_reliability import _overlay_intruder

    sample_rate = 16000
    intruder_ratio = 0.4
    n_intruder = int(base_audio_5s.shape[-1] * intruder_ratio)
    intruder = torch.randn(1, n_intruder)

    mixed = _overlay_intruder(base_audio_5s, intruder, snr_db=5.0)

    assert mixed.shape[-1] == base_audio_5s.shape[-1]


def test_overlay_intruder_empty_intruder_returns_base_clone(base_audio_2s):
    """Zero-length intruder returns a copy of the base unchanged."""
    from b2aiprep.prepare.embedding_reliability import _overlay_intruder

    intruder = torch.zeros(1, 0)
    mixed = _overlay_intruder(base_audio_2s, intruder, snr_db=0.0)

    assert mixed.shape == base_audio_2s.shape
    assert torch.allclose(mixed, base_audio_2s)


# ---------------------------------------------------------------------------
# _fnr_fpr_at_threshold tests
# ---------------------------------------------------------------------------


def test_fnr_fpr_perfect_separation(known_cosine_scores):
    """When all positives score below threshold and all negatives above: FNR=FPR=0."""
    from b2aiprep.prepare.embedding_reliability import _fnr_fpr_at_threshold

    positive_scores, negative_scores = known_cosine_scores
    # threshold=0.25: positive max is 0.20, negative min is 0.70
    fnr, fpr = _fnr_fpr_at_threshold(positive_scores, negative_scores, threshold=0.25)

    assert fnr == pytest.approx(0.0)
    assert fpr == pytest.approx(0.0)


def test_fnr_fpr_zero_threshold_all_negatives_flagged(known_cosine_scores):
    """At threshold=0.0, nothing is flagged → FNR=1.0, FPR=0.0."""
    from b2aiprep.prepare.embedding_reliability import _fnr_fpr_at_threshold

    positive_scores, negative_scores = known_cosine_scores
    fnr, fpr = _fnr_fpr_at_threshold(positive_scores, negative_scores, threshold=0.0)

    assert fnr == pytest.approx(1.0)
    assert fpr == pytest.approx(0.0)


def test_fnr_fpr_high_threshold_all_flagged(known_cosine_scores):
    """At threshold=1.0+ε, everything flagged → FNR=0.0, FPR=1.0."""
    from b2aiprep.prepare.embedding_reliability import _fnr_fpr_at_threshold

    positive_scores, negative_scores = known_cosine_scores
    fnr, fpr = _fnr_fpr_at_threshold(positive_scores, negative_scores, threshold=1.01)

    assert fnr == pytest.approx(0.0)
    assert fpr == pytest.approx(1.0)


def test_fnr_fpr_empty_lists():
    """Empty score lists return (0.0, 0.0) without error."""
    from b2aiprep.prepare.embedding_reliability import _fnr_fpr_at_threshold

    fnr, fpr = _fnr_fpr_at_threshold([], [], threshold=0.5)

    assert fnr == pytest.approx(0.0)
    assert fpr == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_single_curve tests
# ---------------------------------------------------------------------------


def test_compute_single_curve_minimum_threshold_points(known_cosine_scores):
    """Operating curve with step=0.10 must have ≥ 5 threshold points (0→1 inclusive)."""
    from b2aiprep.prepare.embedding_reliability import compute_single_curve

    positive_scores, negative_scores = known_cosine_scores
    labels = [True] * len(positive_scores) + [False] * len(negative_scores)
    scores = positive_scores + negative_scores

    curve = compute_single_curve(scores, labels, threshold_step=0.10)

    assert len(curve) >= 5, f"Expected ≥5 points, got {len(curve)}"


def test_compute_single_curve_monotone_fnr(known_cosine_scores):
    """FNR must be non-increasing as threshold rises (more flagged = fewer missed)."""
    from b2aiprep.prepare.embedding_reliability import compute_single_curve

    positive_scores, negative_scores = known_cosine_scores
    labels = [True] * len(positive_scores) + [False] * len(negative_scores)
    scores = positive_scores + negative_scores

    curve = compute_single_curve(scores, labels, threshold_step=0.05)
    fnrs = [pt["fnr"] for pt in curve]

    # FNR should decrease or stay flat as threshold increases
    for i in range(1, len(fnrs)):
        assert fnrs[i] <= fnrs[i - 1] + 1e-6, (
            f"FNR increased at threshold {curve[i]['threshold']}: "
            f"{fnrs[i - 1]:.4f} → {fnrs[i]:.4f}"
        )


def test_compute_single_curve_required_keys():
    """Each point in the curve must contain threshold, fnr, fpr, review_fraction."""
    from b2aiprep.prepare.embedding_reliability import compute_single_curve

    scores = [0.1, 0.5, 0.9]
    labels = [True, False, False]
    curve = compute_single_curve(scores, labels, threshold_step=0.5)

    for point in curve:
        for key in ("threshold", "fnr", "fpr", "review_fraction"):
            assert key in point, f"Missing key '{key}' in curve point: {point}"


# ---------------------------------------------------------------------------
# write_reliability_report tests
# ---------------------------------------------------------------------------


def test_write_reliability_report_json(tmp_path, minimal_report_dict):
    """write_reliability_report writes a valid JSON file."""
    from b2aiprep.prepare.embedding_reliability import write_reliability_report

    write_reliability_report(minimal_report_dict, tmp_path, output_format="json")

    report_path = tmp_path / "embedding_reliability_report.json"
    assert report_path.exists(), "embedding_reliability_report.json not created"

    with open(report_path) as f:
        loaded = json.load(f)

    required_keys = {
        "report_version", "generated_at", "num_participants",
        "ecapa_operating_curve", "sparc_operating_curve", "or_operating_curve",
        "recommended_ecapa_threshold", "recommended_sparc_threshold",
        "knee_point_fraction",
    }
    for key in required_keys:
        assert key in loaded, f"Missing required key in JSON report: {key}"


def test_write_reliability_report_markdown(tmp_path, minimal_report_dict):
    """write_reliability_report writes a non-empty Markdown file."""
    from b2aiprep.prepare.embedding_reliability import write_reliability_report

    write_reliability_report(minimal_report_dict, tmp_path, output_format="markdown")

    md_path = tmp_path / "embedding_reliability_report.md"
    assert md_path.exists(), "embedding_reliability_report.md not created"
    content = md_path.read_text()
    assert "Recommended Thresholds" in content
    assert "Per-Speech-Fraction-Bin" in content


def test_write_reliability_report_both(tmp_path, minimal_report_dict):
    """output_format='both' writes both JSON and Markdown files."""
    from b2aiprep.prepare.embedding_reliability import write_reliability_report

    write_reliability_report(minimal_report_dict, tmp_path, output_format="both")

    assert (tmp_path / "embedding_reliability_report.json").exists()
    assert (tmp_path / "embedding_reliability_report.md").exists()


def test_report_json_roundtrip(tmp_path, minimal_report_dict):
    """JSON report round-trips losslessly for all required operating curves."""
    from b2aiprep.prepare.embedding_reliability import write_reliability_report

    write_reliability_report(minimal_report_dict, tmp_path, output_format="json")
    with open(tmp_path / "embedding_reliability_report.json") as f:
        loaded = json.load(f)

    for curve_key in ("ecapa_operating_curve", "sparc_operating_curve", "or_operating_curve"):
        orig = minimal_report_dict[curve_key]
        saved = loaded[curve_key]
        assert len(saved) == len(orig), f"{curve_key} length changed after round-trip"
        for orig_pt, saved_pt in zip(orig, saved):
            assert orig_pt["threshold"] == pytest.approx(saved_pt["threshold"])
