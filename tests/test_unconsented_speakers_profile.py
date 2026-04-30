"""Tests for profile-based unconsented-speaker verification (US2).

Covers: missing profile → needs_review, short recording (<1 s) with diarization
still running, low speech fraction confidence cap, ECAPA-only flag, SPARC-only
flag, both-above-threshold pass, and insufficient_data profile routing.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from b2aiprep.prepare.qa_models import AudioRecord, Classification, PipelineConfig
from b2aiprep.prepare.speaker_profiles import SpeakerProfile
from b2aiprep.prepare.unconsented_speakers import check_unconsented_speakers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ECAPA_DIM = 192
_SPARC_DIM = 64


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-10)


def _make_centroid(seed: int, dim: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _unit(rng.standard_normal(dim).astype(np.float64))


def _make_aligned_emb(centroid: np.ndarray, noise: float = 0.0) -> np.ndarray:
    """Return an embedding close to `centroid` (cosine ≈ 1.0 when noise=0)."""
    rng = np.random.default_rng(42)
    v = centroid + noise * rng.standard_normal(len(centroid))
    return _unit(v.astype(np.float64))


def _make_orthogonal_emb(centroid: np.ndarray) -> np.ndarray:
    """Return an embedding roughly orthogonal to centroid (cosine ≈ 0)."""
    rng = np.random.default_rng(99)
    v = rng.standard_normal(len(centroid)).astype(np.float64)
    # Gram-Schmidt: remove projection onto centroid
    v = v - np.dot(v, centroid) * centroid
    return _unit(v)


def _make_segment(start: float, end: float) -> dict:
    return {"start": start, "end": end, "speaker": "SPEAKER_00"}


def _write_pt(
    path: Path,
    ecapa_vec: np.ndarray,
    sparc_vec: np.ndarray,
    active_speech_s: float = 10.0,
    total_duration: float = 12.0,
) -> None:
    diarization = [_make_segment(0.0, active_speech_s)] if active_speech_s > 0 else []
    features = {
        "speaker_embedding": torch.tensor(ecapa_vec.astype(np.float32)),
        "sparc": {"spk_emb": sparc_vec.astype(np.float32)},
        "diarization": diarization,
        "duration": total_duration,
        "is_speech_task": True,
    }
    torch.save(features, str(path))


def _make_profile(
    ecapa_centroid: np.ndarray,
    sparc_centroid: np.ndarray,
    status: str = "ready",
    n_used: int = 5,
) -> SpeakerProfile:
    return SpeakerProfile(
        participant_id="testpid",
        ecapa_model_id="speechbrain/spkrec-ecapa-voxceleb",
        sparc_model_id="senselab/sparc-multi",
        num_recordings_used=n_used,
        num_recordings_excluded=0,
        total_active_speech_s=120.0,
        ecapa_profile_quality_score=0.80,
        sparc_profile_quality_score=0.75,
        profile_status=status,
        age_group="adult",
        included_recordings=["passage-1"],
        excluded_recordings=[],
        ecapa_embedding_centroid=ecapa_centroid.tolist(),
        sparc_embedding_centroid=sparc_centroid.tolist(),
        created_at="2026-04-30T00:00:00+00:00",
        pipeline_config_hash="deadbeef",
    )


def _make_record(features_path: str) -> AudioRecord:
    return AudioRecord(
        participant_id="testpid",
        session_id="ses-01",
        task_name="passage-reading-1",
        audio_path="/fake/audio.wav",
        features_path=features_path,
    )


def _default_config() -> PipelineConfig:
    return PipelineConfig()


# ---------------------------------------------------------------------------
# T009: no profile → needs_review
# ---------------------------------------------------------------------------


def test_no_profile_dir_falls_back_to_diarization_only(tmp_path):
    """When profiles_dir is None, existing diarization-only logic is used."""
    ecapa = _make_centroid(0, _ECAPA_DIM)
    sparc = _make_centroid(1, _SPARC_DIM)
    pt = tmp_path / "features.pt"
    _write_pt(pt, ecapa, sparc, active_speech_s=10.0, total_duration=12.0)

    record = _make_record(str(pt))
    result = check_unconsented_speakers(record, _default_config(), profiles_dir=None)

    assert result.classification != Classification.ERROR
    assert "ecapa_cosine_similarity" not in result.detail or result.detail["ecapa_cosine_similarity"] is None


def test_missing_profile_returns_needs_review(tmp_path):
    """Missing speaker profile file → needs_review with profile_status='missing'."""
    ecapa = _make_centroid(0, _ECAPA_DIM)
    sparc = _make_centroid(1, _SPARC_DIM)
    pt = tmp_path / "features.pt"
    _write_pt(pt, ecapa, sparc)

    record = _make_record(str(pt))
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()

    result = check_unconsented_speakers(record, _default_config(), profiles_dir=str(profiles_dir))

    assert result.classification == Classification.NEEDS_REVIEW
    assert result.detail.get("profile_status") == "missing"


# ---------------------------------------------------------------------------
# T009: active_speech_s < 1 s → confidence=0.10, diarization still runs
# ---------------------------------------------------------------------------


def test_very_short_recording_confidence_010_and_diarization_populated(tmp_path):
    """active_speech_s < 1 s → confidence=0.10, needs_review; diarization still runs."""
    ecapa_c = _make_centroid(0, _ECAPA_DIM)
    sparc_c = _make_centroid(1, _SPARC_DIM)
    # Profile with known centroids
    profile = _make_profile(ecapa_c, sparc_c)

    ecapa_emb = _make_aligned_emb(ecapa_c)
    sparc_emb = _make_aligned_emb(sparc_c)
    pt = tmp_path / "features.pt"
    # active_speech_s = 0.5, total_duration = 5.0
    _write_pt(pt, ecapa_emb, sparc_emb, active_speech_s=0.5, total_duration=5.0)

    profiles_dir = tmp_path / "profiles"
    (profiles_dir / "sub-testpid").mkdir(parents=True)
    import json
    (profiles_dir / "sub-testpid" / "speaker_profile.json").write_text(
        json.dumps(profile.to_json())
    )

    record = _make_record(str(pt))
    result = check_unconsented_speakers(record, _default_config(), profiles_dir=str(profiles_dir))

    assert result.confidence == pytest.approx(0.10)
    assert result.classification == Classification.NEEDS_REVIEW
    # Diarization signals must still be populated even for short recordings
    assert "num_speakers_diarized" in result.detail
    assert result.detail["num_speakers_diarized"] is not None


# ---------------------------------------------------------------------------
# T009: active_speech_fraction < 0.15 → confidence ≤ 0.30
# ---------------------------------------------------------------------------


def test_low_speech_fraction_caps_confidence(tmp_path):
    """active_speech_fraction < 0.15 → confidence ≤ 0.30."""
    ecapa_c = _make_centroid(0, _ECAPA_DIM)
    sparc_c = _make_centroid(1, _SPARC_DIM)
    profile = _make_profile(ecapa_c, sparc_c)

    ecapa_emb = _make_aligned_emb(ecapa_c)
    sparc_emb = _make_aligned_emb(sparc_c)
    pt = tmp_path / "features.pt"
    # active_s=2.0 / duration=30.0 → fraction=0.067 < 0.15
    _write_pt(pt, ecapa_emb, sparc_emb, active_speech_s=2.0, total_duration=30.0)

    profiles_dir = tmp_path / "profiles"
    (profiles_dir / "sub-testpid").mkdir(parents=True)
    import json
    (profiles_dir / "sub-testpid" / "speaker_profile.json").write_text(
        json.dumps(profile.to_json())
    )

    record = _make_record(str(pt))
    result = check_unconsented_speakers(record, _default_config(), profiles_dir=str(profiles_dir))

    assert result.confidence <= 0.30


# ---------------------------------------------------------------------------
# T009: ECAPA below threshold only → or_flag=True
# ---------------------------------------------------------------------------


def test_ecapa_below_threshold_only_sets_or_flag(tmp_path):
    """ECAPA cosine < threshold but SPARC passes → or_flag=True."""
    ecapa_c = _make_centroid(0, _ECAPA_DIM)
    sparc_c = _make_centroid(1, _SPARC_DIM)
    profile = _make_profile(ecapa_c, sparc_c)

    # ECAPA: orthogonal (cosine ≈ 0, below 0.25 threshold)
    ecapa_emb = _make_orthogonal_emb(ecapa_c)
    # SPARC: aligned (cosine ≈ 1, above 0.20 threshold)
    sparc_emb = _make_aligned_emb(sparc_c)
    pt = tmp_path / "features.pt"
    _write_pt(pt, ecapa_emb, sparc_emb)

    profiles_dir = tmp_path / "profiles"
    (profiles_dir / "sub-testpid").mkdir(parents=True)
    import json
    (profiles_dir / "sub-testpid" / "speaker_profile.json").write_text(
        json.dumps(profile.to_json())
    )

    record = _make_record(str(pt))
    result = check_unconsented_speakers(record, _default_config(), profiles_dir=str(profiles_dir))

    assert result.detail.get("or_flag") is True
    assert result.classification == Classification.NEEDS_REVIEW


# ---------------------------------------------------------------------------
# T009: SPARC below threshold only → or_flag=True
# ---------------------------------------------------------------------------


def test_sparc_below_threshold_only_sets_or_flag(tmp_path):
    """SPARC cosine < threshold but ECAPA passes → or_flag=True."""
    ecapa_c = _make_centroid(0, _ECAPA_DIM)
    sparc_c = _make_centroid(1, _SPARC_DIM)
    profile = _make_profile(ecapa_c, sparc_c)

    # ECAPA: aligned (above threshold)
    ecapa_emb = _make_aligned_emb(ecapa_c)
    # SPARC: orthogonal (below 0.20 threshold)
    sparc_emb = _make_orthogonal_emb(sparc_c)
    pt = tmp_path / "features.pt"
    _write_pt(pt, ecapa_emb, sparc_emb)

    profiles_dir = tmp_path / "profiles"
    (profiles_dir / "sub-testpid").mkdir(parents=True)
    import json
    (profiles_dir / "sub-testpid" / "speaker_profile.json").write_text(
        json.dumps(profile.to_json())
    )

    record = _make_record(str(pt))
    result = check_unconsented_speakers(record, _default_config(), profiles_dir=str(profiles_dir))

    assert result.detail.get("or_flag") is True
    assert result.classification == Classification.NEEDS_REVIEW


# ---------------------------------------------------------------------------
# T009: both above threshold → or_flag=False
# ---------------------------------------------------------------------------


def test_both_above_threshold_or_flag_false(tmp_path):
    """Both cosines above threshold → or_flag=False."""
    ecapa_c = _make_centroid(0, _ECAPA_DIM)
    sparc_c = _make_centroid(1, _SPARC_DIM)
    profile = _make_profile(ecapa_c, sparc_c)

    # Both aligned (cosine ≈ 1, well above thresholds 0.25 and 0.20)
    ecapa_emb = _make_aligned_emb(ecapa_c)
    sparc_emb = _make_aligned_emb(sparc_c)
    pt = tmp_path / "features.pt"
    _write_pt(pt, ecapa_emb, sparc_emb)

    profiles_dir = tmp_path / "profiles"
    (profiles_dir / "sub-testpid").mkdir(parents=True)
    import json
    (profiles_dir / "sub-testpid" / "speaker_profile.json").write_text(
        json.dumps(profile.to_json())
    )

    record = _make_record(str(pt))
    result = check_unconsented_speakers(record, _default_config(), profiles_dir=str(profiles_dir))

    assert result.detail.get("or_flag") is False


# ---------------------------------------------------------------------------
# T009: profile status=insufficient_data → needs_review
# ---------------------------------------------------------------------------


def test_insufficient_data_profile_returns_needs_review(tmp_path):
    """Profile with insufficient_data status → needs_review without cosine scoring."""
    ecapa_c = _make_centroid(0, _ECAPA_DIM)
    sparc_c = _make_centroid(1, _SPARC_DIM)
    profile = _make_profile(ecapa_c, sparc_c, status="insufficient_data", n_used=1)

    ecapa_emb = _make_aligned_emb(ecapa_c)
    sparc_emb = _make_aligned_emb(sparc_c)
    pt = tmp_path / "features.pt"
    _write_pt(pt, ecapa_emb, sparc_emb)

    profiles_dir = tmp_path / "profiles"
    (profiles_dir / "sub-testpid").mkdir(parents=True)
    import json
    (profiles_dir / "sub-testpid" / "speaker_profile.json").write_text(
        json.dumps(profile.to_json())
    )

    record = _make_record(str(pt))
    result = check_unconsented_speakers(record, _default_config(), profiles_dir=str(profiles_dir))

    assert result.classification == Classification.NEEDS_REVIEW
    assert result.detail.get("profile_status") == "insufficient_data"


# ---------------------------------------------------------------------------
# T009: detail dict contains all EmbeddingVerificationResult fields
# ---------------------------------------------------------------------------


def test_detail_dict_contains_all_required_fields(tmp_path):
    """CheckResult.detail must include all extended EmbeddingVerificationResult fields."""
    ecapa_c = _make_centroid(0, _ECAPA_DIM)
    sparc_c = _make_centroid(1, _SPARC_DIM)
    profile = _make_profile(ecapa_c, sparc_c)

    ecapa_emb = _make_aligned_emb(ecapa_c)
    sparc_emb = _make_aligned_emb(sparc_c)
    pt = tmp_path / "features.pt"
    _write_pt(pt, ecapa_emb, sparc_emb)

    profiles_dir = tmp_path / "profiles"
    (profiles_dir / "sub-testpid").mkdir(parents=True)
    import json
    (profiles_dir / "sub-testpid" / "speaker_profile.json").write_text(
        json.dumps(profile.to_json())
    )

    record = _make_record(str(pt))
    result = check_unconsented_speakers(record, _default_config(), profiles_dir=str(profiles_dir))

    required_keys = {
        "ecapa_cosine_similarity",
        "sparc_cosine_similarity",
        "or_flag",
        "active_speech_fraction",
        "active_speech_s",
        "speech_fraction_confidence",
        "profile_status",
        "num_speakers_diarized",
        "diarization_primary_ratio",
        "extra_speaker_count",
        "ecapa_model_id",
        "sparc_model_id",
        "enrollment_n",
        "age_group",
    }
    assert required_keys <= set(result.detail.keys()), (
        f"Missing keys: {required_keys - set(result.detail.keys())}"
    )
