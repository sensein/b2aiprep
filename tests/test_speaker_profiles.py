"""Tests for speaker profile construction and loading (US1).

Covers: task prefix exclusion, speech duration gating, dual-centroid
computation, outlier rejection, insufficient_data / contaminated status,
and JSON round-trip of SpeakerProfile.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from b2aiprep.prepare.qa_models import PipelineConfig
from b2aiprep.prepare.speaker_profiles import (
    SpeakerProfile,
    _compute_active_speech_s,
    _l2_normalize,
    _profile_quality,
    _reject_outliers,
    _weighted_centroid,
    build_speaker_profiles,
    load_speaker_profile,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_segment(start: float, end: float, speaker: str = "SPEAKER_00") -> dict:
    """Create a plain-dict diarization segment (avoids senselab dependency)."""
    return {"start": start, "end": end, "speaker": speaker}


def _make_features_pt(
    task_name: str,
    session_id: str = "ses1",
    active_speech_s: float = 12.0,
    duration: float = 15.0,
    ecapa_vec: np.ndarray | None = None,
    sparc_vec: np.ndarray | None = None,
    is_speech_task: bool = True,
) -> dict:
    """Build a minimal features dict matching the real .pt file structure."""
    if ecapa_vec is None:
        rng = np.random.default_rng(abs(hash(task_name)) % (2**32))
        ecapa_vec = rng.standard_normal(192).astype(np.float32)
    if sparc_vec is None:
        rng = np.random.default_rng(abs(hash(task_name + "sparc")) % (2**32))
        sparc_vec = rng.standard_normal(64).astype(np.float32)

    diarization = [_make_segment(0.0, active_speech_s)]

    return {
        "speaker_embedding": torch.tensor(ecapa_vec),
        "sparc": {"spk_emb": sparc_vec},
        "diarization": diarization,
        "duration": duration,
        "is_speech_task": is_speech_task,
        "audio_path": f"/fake/sub-test_{session_id}_task-{task_name}_audio.wav",
    }


def _write_features_pt(bids_dir: Path, pid: str, task_name: str, session_id: str, features: dict) -> Path:
    """Write a _features.pt file into a BIDS-layout directory."""
    audio_dir = bids_dir / f"sub-{pid}" / session_id / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    pt_path = audio_dir / f"sub-{pid}_{session_id}_task-{task_name}_features.pt"
    torch.save(features, str(pt_path))
    return pt_path


def _default_config(**overrides) -> PipelineConfig:
    cfg = PipelineConfig()
    for k, v in overrides.items():
        cfg.speaker_profile[k] = v
    return cfg


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------


def test_compute_active_speech_s_dict_segments():
    segs = [{"start": 0.0, "end": 5.0}, {"start": 6.0, "end": 10.0}]
    assert _compute_active_speech_s(segs) == pytest.approx(9.0)


def test_compute_active_speech_s_empty():
    assert _compute_active_speech_s([]) == 0.0
    assert _compute_active_speech_s(None) == 0.0


def test_l2_normalize():
    v = np.array([3.0, 4.0])
    normed = _l2_normalize(v)
    assert np.linalg.norm(normed) == pytest.approx(1.0, abs=1e-6)


def test_l2_normalize_zero_vector():
    v = np.zeros(5)
    normed = _l2_normalize(v)
    assert np.allclose(normed, np.zeros(5))


# ---------------------------------------------------------------------------
# Task prefix exclusion
# ---------------------------------------------------------------------------


def test_task_prefix_exclusion(tmp_path):
    """Recordings matching excluded prefixes should not contribute to profile."""
    pid = "testpid"
    cfg = _default_config(min_profile_recordings=1)

    # Write 3 speech recordings: one excluded, two usable
    _write_features_pt(
        tmp_path, pid, "Diadochokinesis-pa", "ses-01",
        _make_features_pt("Diadochokinesis-pa", active_speech_s=10.0, duration=12.0),
    )
    _write_features_pt(
        tmp_path, pid, "passage-reading-1", "ses-01",
        _make_features_pt("passage-reading-1", active_speech_s=15.0, duration=18.0),
    )
    _write_features_pt(
        tmp_path, pid, "passage-reading-2", "ses-01",
        _make_features_pt("passage-reading-2", active_speech_s=14.0, duration=17.0),
    )

    profiles = build_speaker_profiles(tmp_path, tmp_path / "profiles", config=cfg)

    assert pid in profiles
    profile = profiles[pid]
    # Diadochokinesis-pa must not appear in included recordings
    assert not any("Diadochokinesis" in r for r in profile.included_recordings)
    # It must appear in excluded recordings with the right reason
    reasons = [e["reason"] for e in profile.excluded_recordings]
    assert "task_prefix_excluded" in reasons


def test_task_prefix_is_prefix_not_substring(tmp_path):
    """'Glides-ascending' should be excluded; 'Glides-descending' too; 'Glides-x' yes."""
    pid = "prefixpid"
    cfg = _default_config(min_profile_recordings=1)

    _write_features_pt(
        tmp_path, pid, "Glides-ascending", "ses-01",
        _make_features_pt("Glides-ascending", active_speech_s=10.0),
    )
    _write_features_pt(
        tmp_path, pid, "passage-reading-1", "ses-01",
        _make_features_pt("passage-reading-1", active_speech_s=10.0),
    )

    profiles = build_speaker_profiles(tmp_path, tmp_path / "profiles", config=cfg)
    profile = profiles[pid]
    assert "Glides-ascending" not in profile.included_recordings
    assert "passage-reading-1" in profile.included_recordings


# ---------------------------------------------------------------------------
# Speech duration gating
# ---------------------------------------------------------------------------


def test_short_recording_excluded(tmp_path):
    """Recording with active_speech_s < 1 s must be excluded from enrollment."""
    pid = "shortpid"
    cfg = _default_config(min_profile_recordings=1)

    _write_features_pt(
        tmp_path, pid, "passage-short", "ses-01",
        _make_features_pt("passage-short", active_speech_s=0.5, duration=5.0),
    )
    _write_features_pt(
        tmp_path, pid, "passage-long", "ses-01",
        _make_features_pt("passage-long", active_speech_s=12.0, duration=15.0),
    )

    profiles = build_speaker_profiles(tmp_path, tmp_path / "profiles", config=cfg)
    profile = profiles[pid]
    assert "passage-short" not in profile.included_recordings
    excluded_reasons = {e["task_name"]: e["reason"] for e in profile.excluded_recordings}
    assert excluded_reasons.get("passage-short") == "active_speech_too_short"


def test_low_speech_fraction_excluded(tmp_path):
    """Recording with active_speech_fraction < 0.15 is excluded."""
    pid = "fracpid"
    cfg = _default_config(min_profile_recordings=1)

    # active_s=1.0, duration=30.0 → fraction=0.033 < 0.15
    _write_features_pt(
        tmp_path, pid, "passage-silence", "ses-01",
        _make_features_pt("passage-silence", active_speech_s=1.0, duration=30.0),
    )
    _write_features_pt(
        tmp_path, pid, "passage-speech", "ses-01",
        _make_features_pt("passage-speech", active_speech_s=12.0, duration=15.0),
    )

    profiles = build_speaker_profiles(tmp_path, tmp_path / "profiles", config=cfg)
    profile = profiles[pid]
    assert "passage-silence" not in profile.included_recordings
    excluded_reasons = {e["task_name"]: e["reason"] for e in profile.excluded_recordings}
    assert excluded_reasons.get("passage-silence") == "low_speech_fraction"


# ---------------------------------------------------------------------------
# Insufficient data status
# ---------------------------------------------------------------------------


def test_insufficient_data_status(tmp_path):
    """Participant with < min_profile_recordings usable recordings → insufficient_data."""
    pid = "insufpid"
    cfg = _default_config(min_profile_recordings=3)

    # Only 2 usable recordings
    for i in range(2):
        _write_features_pt(
            tmp_path, pid, f"passage-{i}", "ses-01",
            _make_features_pt(f"passage-{i}", active_speech_s=12.0),
        )

    profiles = build_speaker_profiles(tmp_path, tmp_path / "profiles", config=cfg)
    assert profiles[pid].profile_status == "insufficient_data"
    assert profiles[pid].ecapa_embedding_centroid is None
    assert profiles[pid].sparc_embedding_centroid is None


# ---------------------------------------------------------------------------
# Contaminated status
# ---------------------------------------------------------------------------


def test_contaminated_status_low_ecapa_quality(tmp_path):
    """Profile where ECAPA quality < contamination_threshold → contaminated."""
    pid = "contampid"
    # Use a very high threshold to force contaminated for random embeddings
    cfg = _default_config(min_profile_recordings=3, contamination_quality_threshold=0.99)

    for i in range(4):
        # Random unit vectors will have low pairwise cosine similarity
        rng = np.random.default_rng(i)
        ecapa = rng.standard_normal(192).astype(np.float32)
        sparc = rng.standard_normal(64).astype(np.float32)
        _write_features_pt(
            tmp_path, pid, f"passage-{i}", "ses-01",
            _make_features_pt(f"passage-{i}", active_speech_s=12.0, ecapa_vec=ecapa, sparc_vec=sparc),
        )

    profiles = build_speaker_profiles(tmp_path, tmp_path / "profiles", config=cfg)
    assert profiles[pid].profile_status == "contaminated"


# ---------------------------------------------------------------------------
# Dual centroid computation
# ---------------------------------------------------------------------------


def test_dual_centroid_shapes(tmp_path):
    """Ready profile must contain ecapa_embedding_centroid (192) and sparc (64)."""
    pid = "centpid"
    cfg = _default_config(min_profile_recordings=3)

    for i in range(4):
        _write_features_pt(
            tmp_path, pid, f"passage-{i}", "ses-01",
            _make_features_pt(f"passage-{i}", active_speech_s=12.0),
        )

    profiles = build_speaker_profiles(tmp_path, tmp_path / "profiles", config=cfg)
    profile = profiles[pid]
    assert profile.profile_status in ("ready", "contaminated")
    assert profile.ecapa_embedding_centroid is not None
    assert len(profile.ecapa_embedding_centroid) == 192
    assert profile.sparc_embedding_centroid is not None
    assert len(profile.sparc_embedding_centroid) == 64


# ---------------------------------------------------------------------------
# Outlier rejection
# ---------------------------------------------------------------------------


def test_outlier_rejection_removes_outlier():
    """An embedding that is orthogonal to all others should be rejected."""
    rng = np.random.default_rng(0)
    # 5 similar embeddings
    base = _l2_normalize(rng.standard_normal(32))
    similar = [_l2_normalize(base + 0.05 * rng.standard_normal(32)) for _ in range(5)]
    # 1 clear outlier (opposite direction)
    outlier = _l2_normalize(-base + 0.01 * rng.standard_normal(32))
    all_embs = similar + [outlier]
    all_w = [1.0] * 6

    kept, _ = _reject_outliers(all_embs, all_w, std_multiplier=1.5)
    assert len(kept) < 6


def test_outlier_rejection_preserves_all_similar():
    """All similar embeddings should survive rejection."""
    rng = np.random.default_rng(1)
    base = _l2_normalize(rng.standard_normal(32))
    similar = [_l2_normalize(base + 0.01 * rng.standard_normal(32)) for _ in range(5)]
    weights = [1.0] * 5

    kept, _ = _reject_outliers(similar, weights, std_multiplier=1.5)
    assert len(kept) == 5


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_speaker_profile_json_roundtrip(tmp_path):
    """SpeakerProfile serialises and deserialises without data loss."""
    profile = SpeakerProfile(
        participant_id="abc123",
        ecapa_model_id="speechbrain/spkrec-ecapa-voxceleb",
        sparc_model_id="senselab/sparc-multi",
        num_recordings_used=5,
        num_recordings_excluded=2,
        total_active_speech_s=90.5,
        ecapa_profile_quality_score=0.72,
        sparc_profile_quality_score=0.68,
        profile_status="ready",
        age_group="adult",
        included_recordings=["passage-1", "passage-2"],
        excluded_recordings=[{"task_name": "Diadochokinesis-pa", "session_id": "ses-01", "reason": "task_prefix_excluded"}],
        ecapa_embedding_centroid=list(np.zeros(192)),
        sparc_embedding_centroid=list(np.zeros(64)),
        created_at="2026-04-30T00:00:00+00:00",
        pipeline_config_hash="deadbeef",
    )

    data = profile.to_json()
    loaded = SpeakerProfile.from_json(data)

    assert loaded.participant_id == profile.participant_id
    assert loaded.profile_status == profile.profile_status
    assert len(loaded.ecapa_embedding_centroid) == 192
    assert len(loaded.sparc_embedding_centroid) == 64
    assert loaded.excluded_recordings[0]["reason"] == "task_prefix_excluded"


# ---------------------------------------------------------------------------
# load_speaker_profile
# ---------------------------------------------------------------------------


def test_load_speaker_profile_returns_none_when_missing(tmp_path):
    """Missing profile file returns None without raising."""
    result = load_speaker_profile(tmp_path, "nonexistent")
    assert result is None


def test_load_speaker_profile_reads_written_file(tmp_path):
    """load_speaker_profile reads a file written by build_speaker_profiles."""
    pid = "loadpid"
    cfg = _default_config(min_profile_recordings=3)

    for i in range(4):
        _write_features_pt(
            tmp_path, pid, f"passage-{i}", "ses-01",
            _make_features_pt(f"passage-{i}", active_speech_s=12.0),
        )

    profiles_dir = tmp_path / "profiles"
    build_speaker_profiles(tmp_path, profiles_dir, config=cfg)

    loaded = load_speaker_profile(profiles_dir, pid)
    assert loaded is not None
    assert loaded.participant_id == pid
    assert loaded.ecapa_embedding_centroid is not None
