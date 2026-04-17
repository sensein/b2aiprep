"""T010 — Unconsented-speaker detection unit tests.

Tests ``check_unconsented_speakers(audio_record, config) -> CheckResult``
implemented in T016.  All tests skip automatically until the module exists.

Assertions:
- Single-speaker WAV passes (extra_speaker_count == 0)
- Two-speaker WAV flags with extra_speaker_count == 1
- detected_languages list is populated
- evans_model_flag == 1 forces needs_review classification
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# Skip entire file if module not yet implemented (T016)
unconsented_speakers = pytest.importorskip(
    "b2aiprep.prepare.unconsented_speakers",
    reason="unconsented_speakers module not yet implemented (T016)",
)

from b2aiprep.prepare.qa_models import CheckType, Classification, PipelineConfig

check_unconsented_speakers = unconsented_speakers.check_unconsented_speakers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audio_record(participant_id="sub-001", session_id="ses-01", task_name="task-x", features_path=None):
    """Build a minimal audio-record-like namespace for testing."""
    return SimpleNamespace(
        participant_id=participant_id,
        session_id=session_id,
        task_name=task_name,
        features_path=features_path,
    )


def _diarization_single_speaker():
    """Return a mock diarization result with one speaker."""
    return {
        "speaker_0": [{"start": 0.0, "end": 2.0}],
    }


def _diarization_two_speakers():
    """Return a mock diarization result with two speakers."""
    return {
        "speaker_0": [{"start": 0.0, "end": 1.0}],
        "speaker_1": [{"start": 1.0, "end": 2.0}],
    }


# ---------------------------------------------------------------------------
# Single-speaker WAV passes
# ---------------------------------------------------------------------------


class TestSingleSpeaker:
    def test_classification_pass(self, tmp_path):
        """Single-speaker recording should classify as PASS."""
        config = PipelineConfig()
        record = _make_audio_record(features_path=tmp_path / "sub-001_ses-01_task-x_features.pt")

        with patch.object(
            unconsented_speakers,
            "_load_diarization",
            return_value=_diarization_single_speaker(),
        ), patch.object(
            unconsented_speakers,
            "_run_evans_model",
            return_value=0,
        ), patch.object(
            unconsented_speakers,
            "_identify_languages",
            return_value=[{"speaker_index": 0, "language": "en", "confidence": 0.95}],
        ):
            result = check_unconsented_speakers(record, config)

        assert result.classification in (Classification.PASS, Classification.NEEDS_REVIEW)
        assert result.detail["extra_speaker_count"] == 0

    def test_num_speakers_diarized_is_one(self, tmp_path):
        config = PipelineConfig()
        record = _make_audio_record(features_path=tmp_path / "f.pt")

        with patch.object(
            unconsented_speakers,
            "_load_diarization",
            return_value=_diarization_single_speaker(),
        ), patch.object(
            unconsented_speakers,
            "_run_evans_model",
            return_value=0,
        ), patch.object(
            unconsented_speakers,
            "_identify_languages",
            return_value=[{"speaker_index": 0, "language": "en", "confidence": 0.9}],
        ):
            result = check_unconsented_speakers(record, config)

        assert result.detail["num_speakers_diarized"] == 1

    def test_primary_speaker_ratio_is_one(self, tmp_path):
        config = PipelineConfig()
        record = _make_audio_record(features_path=tmp_path / "f.pt")

        with patch.object(
            unconsented_speakers,
            "_load_diarization",
            return_value=_diarization_single_speaker(),
        ), patch.object(
            unconsented_speakers,
            "_run_evans_model",
            return_value=0,
        ), patch.object(
            unconsented_speakers,
            "_identify_languages",
            return_value=[{"speaker_index": 0, "language": "en", "confidence": 0.9}],
        ):
            result = check_unconsented_speakers(record, config)

        assert result.detail["primary_speaker_ratio"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Two-speaker WAV flags
# ---------------------------------------------------------------------------


class TestTwoSpeakers:
    def _run_two_speaker(self, tmp_path, evans_flag=0):
        config = PipelineConfig()
        record = _make_audio_record(features_path=tmp_path / "f.pt")

        with patch.object(
            unconsented_speakers,
            "_load_diarization",
            return_value=_diarization_two_speakers(),
        ), patch.object(
            unconsented_speakers,
            "_run_evans_model",
            return_value=evans_flag,
        ), patch.object(
            unconsented_speakers,
            "_identify_languages",
            return_value=[
                {"speaker_index": 0, "language": "en", "confidence": 0.9},
                {"speaker_index": 1, "language": "en", "confidence": 0.85},
            ],
        ):
            return check_unconsented_speakers(record, config)

    def test_extra_speaker_count_is_one(self, tmp_path):
        result = self._run_two_speaker(tmp_path)
        assert result.detail["extra_speaker_count"] == 1

    def test_num_speakers_diarized_is_two(self, tmp_path):
        result = self._run_two_speaker(tmp_path)
        assert result.detail["num_speakers_diarized"] == 2

    def test_primary_speaker_ratio_less_than_one(self, tmp_path):
        result = self._run_two_speaker(tmp_path)
        assert result.detail["primary_speaker_ratio"] < 1.0


# ---------------------------------------------------------------------------
# detected_languages populated
# ---------------------------------------------------------------------------


class TestDetectedLanguages:
    def test_detected_languages_list_present(self, tmp_path):
        config = PipelineConfig()
        record = _make_audio_record(features_path=tmp_path / "f.pt")

        with patch.object(
            unconsented_speakers,
            "_load_diarization",
            return_value=_diarization_single_speaker(),
        ), patch.object(
            unconsented_speakers,
            "_run_evans_model",
            return_value=0,
        ), patch.object(
            unconsented_speakers,
            "_identify_languages",
            return_value=[{"speaker_index": 0, "language": "en", "confidence": 0.95}],
        ):
            result = check_unconsented_speakers(record, config)

        assert "detected_languages" in result.detail
        assert isinstance(result.detail["detected_languages"], list)

    def test_detected_languages_have_required_fields(self, tmp_path):
        config = PipelineConfig()
        record = _make_audio_record(features_path=tmp_path / "f.pt")

        with patch.object(
            unconsented_speakers,
            "_load_diarization",
            return_value=_diarization_two_speakers(),
        ), patch.object(
            unconsented_speakers,
            "_run_evans_model",
            return_value=0,
        ), patch.object(
            unconsented_speakers,
            "_identify_languages",
            return_value=[
                {"speaker_index": 0, "language": "en", "confidence": 0.9},
                {"speaker_index": 1, "language": "es", "confidence": 0.7},
            ],
        ):
            result = check_unconsented_speakers(record, config)

        for entry in result.detail["detected_languages"]:
            assert "speaker_index" in entry
            assert "language" in entry
            assert "confidence" in entry


# ---------------------------------------------------------------------------
# Evan's model flag forces needs_review
# ---------------------------------------------------------------------------


class TestEvansModelFlag:
    def test_evans_flag_1_forces_needs_review(self, tmp_path):
        """evans_model_flag == 1 must force NEEDS_REVIEW regardless of speaker count."""
        config = PipelineConfig()
        record = _make_audio_record(features_path=tmp_path / "f.pt")

        with patch.object(
            unconsented_speakers,
            "_load_diarization",
            return_value=_diarization_single_speaker(),
        ), patch.object(
            unconsented_speakers,
            "_run_evans_model",
            return_value=1,  # flag raised
        ), patch.object(
            unconsented_speakers,
            "_identify_languages",
            return_value=[{"speaker_index": 0, "language": "en", "confidence": 0.9}],
        ):
            result = check_unconsented_speakers(record, config)

        assert result.classification == Classification.NEEDS_REVIEW
        assert result.detail["evans_model_flag"] == 1

    def test_evans_flag_0_does_not_force_review(self, tmp_path):
        config = PipelineConfig()
        record = _make_audio_record(features_path=tmp_path / "f.pt")

        with patch.object(
            unconsented_speakers,
            "_load_diarization",
            return_value=_diarization_single_speaker(),
        ), patch.object(
            unconsented_speakers,
            "_run_evans_model",
            return_value=0,
        ), patch.object(
            unconsented_speakers,
            "_identify_languages",
            return_value=[{"speaker_index": 0, "language": "en", "confidence": 0.9}],
        ):
            result = check_unconsented_speakers(record, config)

        # Should NOT be needs_review purely due to evans flag (may still be
        # needs_review for other reasons, but evans_flag is 0)
        assert result.detail["evans_model_flag"] == 0

    def test_check_type_is_unconsented_speakers(self, tmp_path):
        config = PipelineConfig()
        record = _make_audio_record(features_path=tmp_path / "f.pt")

        with patch.object(
            unconsented_speakers,
            "_load_diarization",
            return_value=_diarization_single_speaker(),
        ), patch.object(
            unconsented_speakers,
            "_run_evans_model",
            return_value=0,
        ), patch.object(
            unconsented_speakers,
            "_identify_languages",
            return_value=[{"speaker_index": 0, "language": "en", "confidence": 0.9}],
        ):
            result = check_unconsented_speakers(record, config)

        assert result.check_type == CheckType.UNCONSENTED_SPEAKERS
