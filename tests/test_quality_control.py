"""T009 — Audio technical quality check unit tests.

Tests hard gate triggering, soft threshold assignment, and FR-011 invalid-input
handling.  Synthetic WAV files come from the ``wav_factory`` fixture defined in
conftest.py — no import needed; pytest injects it automatically.

These tests target ``check_audio_quality(audio_path, config) -> CheckResult``
which is added to ``quality_control.py`` in T014.  Tests that require this
function are skipped automatically until it is implemented.
"""

from pathlib import Path

import pytest

from b2aiprep.prepare.qa_models import Classification, PipelineConfig

# ---------------------------------------------------------------------------
# Conditional skip for the not-yet-implemented function
# ---------------------------------------------------------------------------

try:
    from b2aiprep.prepare.quality_control import check_audio_quality
    _IMPL_AVAILABLE = True
except ImportError:
    _IMPL_AVAILABLE = False

skip_until_t014 = pytest.mark.skipif(
    not _IMPL_AVAILABLE,
    reason="check_audio_quality not yet implemented (T014/T015)",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> PipelineConfig:
    return PipelineConfig()


@pytest.fixture
def clean_wav(wav_factory) -> Path:
    return wav_factory(wav_type="clean")


@pytest.fixture
def clipped_wav(wav_factory) -> Path:
    return wav_factory(wav_type="clipped")


@pytest.fixture
def silent_wav(wav_factory) -> Path:
    return wav_factory(wav_type="silent")


# ---------------------------------------------------------------------------
# Hard gate triggering tests
# ---------------------------------------------------------------------------


@skip_until_t014
class TestHardGates:
    def test_clean_audio_hard_gate_not_triggered(self, clean_wav, config):
        result = check_audio_quality(clean_wav, config)
        assert result.detail["hard_gate_triggered"] is False

    def test_clipped_audio_triggers_hard_gate(self, clipped_wav, config):
        """Clipped audio (>5 % of samples at full scale) must trigger the gate."""
        result = check_audio_quality(clipped_wav, config)
        assert result.detail["hard_gate_triggered"] is True

    def test_silent_audio_triggers_hard_gate(self, silent_wav, config):
        """Fully silent audio (>50 % silence) must trigger the gate."""
        result = check_audio_quality(silent_wav, config)
        assert result.detail["hard_gate_triggered"] is True

    def test_hard_gate_produces_fail_classification(self, clipped_wav, config):
        result = check_audio_quality(clipped_wav, config)
        assert result.classification == Classification.FAIL

    def test_clipped_field_present_in_detail(self, clipped_wav, config):
        result = check_audio_quality(clipped_wav, config)
        assert "proportion_clipped" in result.detail

    def test_silent_field_present_in_detail(self, silent_wav, config):
        result = check_audio_quality(silent_wav, config)
        assert "proportion_silent" in result.detail


# ---------------------------------------------------------------------------
# Soft threshold assignment tests
# ---------------------------------------------------------------------------


@skip_until_t014
class TestSoftThresholds:
    def test_clean_audio_returns_check_result(self, clean_wav, config):
        from b2aiprep.prepare.qa_models import CheckResult
        result = check_audio_quality(clean_wav, config)
        assert isinstance(result, CheckResult)

    def test_score_in_unit_range(self, clean_wav, config):
        result = check_audio_quality(clean_wav, config)
        assert 0.0 <= result.score <= 1.0

    def test_confidence_in_unit_range(self, clean_wav, config):
        result = check_audio_quality(clean_wav, config)
        assert 0.0 <= result.confidence <= 1.0

    def test_check_type_is_audio_quality(self, clean_wav, config):
        from b2aiprep.prepare.qa_models import CheckType
        result = check_audio_quality(clean_wav, config)
        assert result.check_type.value == "audio_quality"

    def test_snr_field_present(self, clean_wav, config):
        result = check_audio_quality(clean_wav, config)
        assert "peak_snr_db" in result.detail or "spectral_gating_snr_db" in result.detail

    def test_thresholds_sourced_from_config(self, clipped_wav):
        """A config with very high clipping_max should NOT trigger the gate."""
        lenient_config = PipelineConfig(
            hard_gate_thresholds={"snr_min_db": 0.0, "clipping_max": 1.0, "silence_max": 1.0}
        )
        result = check_audio_quality(clipped_wav, lenient_config)
        assert result.detail["hard_gate_triggered"] is False


# ---------------------------------------------------------------------------
# Environment noise (YAMNet) tests
# ---------------------------------------------------------------------------


@skip_until_t014
class TestEnvironmentNoise:
    def test_environment_top_labels_present(self, clean_wav, config):
        result = check_audio_quality(clean_wav, config)
        assert "environment_top_labels" in result.detail

    def test_top_labels_have_label_and_confidence(self, clean_wav, config):
        result = check_audio_quality(clean_wav, config)
        for entry in result.detail["environment_top_labels"]:
            assert "label" in entry
            assert "confidence" in entry
            assert 0.0 <= entry["confidence"] <= 1.0

    def test_environment_noise_flag_is_bool(self, clean_wav, config):
        result = check_audio_quality(clean_wav, config)
        assert isinstance(result.detail["environment_noise_flag"], bool)


# ---------------------------------------------------------------------------
# FR-011 — Invalid input handling
# ---------------------------------------------------------------------------


@skip_until_t014
class TestInvalidInputHandling:
    def test_zero_byte_file_does_not_raise(self, tmp_path, config):
        """A zero-byte file must not halt the pipeline."""
        bad = tmp_path / "empty.wav"
        bad.write_bytes(b"")
        try:
            result = check_audio_quality(bad, config)
            assert result is None or result.classification in (
                Classification.ERROR,
                Classification.FAIL,
            )
        except (ValueError, RuntimeError, OSError):
            pass  # caller wraps with make_error_check_result (FR-014)

    def test_truncated_wav_does_not_raise(self, tmp_path, config):
        truncated = tmp_path / "truncated.wav"
        truncated.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00")
        try:
            result = check_audio_quality(truncated, config)
            assert result is None or result.classification in (
                Classification.ERROR,
                Classification.FAIL,
            )
        except (ValueError, RuntimeError, OSError):
            pass

    def test_non_wav_binary_does_not_raise(self, tmp_path, config):
        garbage = tmp_path / "garbage.wav"
        garbage.write_bytes(b"\x00\x01\x02\x03" * 100)
        try:
            result = check_audio_quality(garbage, config)
            assert result is None or result.classification in (
                Classification.ERROR,
                Classification.FAIL,
            )
        except (ValueError, RuntimeError, OSError):
            pass


# ---------------------------------------------------------------------------
# WAV factory smoke tests (always run — validate the conftest fixture itself)
# ---------------------------------------------------------------------------


class TestWavFactory:
    """Validate that the wav_factory fixture (conftest.py) produces correct WAVs."""

    def test_clean_wav_created(self, wav_factory):
        path = wav_factory(wav_type="clean")
        assert path.exists() and path.stat().st_size > 0

    def test_clipped_wav_created(self, wav_factory):
        path = wav_factory(wav_type="clipped")
        assert path.exists() and path.stat().st_size > 0

    def test_silent_wav_created(self, wav_factory):
        path = wav_factory(wav_type="silent")
        assert path.exists() and path.stat().st_size > 0

    def test_noisy_wav_created(self, wav_factory):
        path = wav_factory(wav_type="noisy")
        assert path.exists() and path.stat().st_size > 0

    def test_wav_is_loadable_by_torchaudio(self, wav_factory):
        import torchaudio
        path = wav_factory(wav_type="clean")
        waveform, sr = torchaudio.load(str(path))
        assert waveform.shape[0] == 1   # mono
        assert sr == 16000
        assert waveform.shape[1] == 32000  # 2 s × 16 kHz

    def test_unknown_type_raises(self, wav_factory):
        with pytest.raises(ValueError):
            wav_factory(wav_type="unknown")
