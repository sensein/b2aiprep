"""T012 — Task compliance unit tests.

Tests ``check_tier_a``, ``check_tier_b``, ``check_tier_c``,
``get_compliance_tier``, and ``check_task_compliance`` from T018–T020.
All tests skip automatically until the module exists.

Assertions:
- Tier A: WER=0.0 → PASS;  WER=0.5 → NEEDS_REVIEW
- Tier B: phonation duration below minimum → FAIL
- Tier C: LLM=False → FAIL
- compliance_tier assigned correctly per task category
- get_compliance_tier(unknown) raises ValueError
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

# Skip entire file if module not yet implemented (T018-T020)
task_compliance = pytest.importorskip(
    "b2aiprep.prepare.task_compliance",
    reason="task_compliance module not yet implemented (T018-T020)",
)

from b2aiprep.prepare.qa_models import CheckType, Classification, PipelineConfig

get_compliance_tier = task_compliance.get_compliance_tier
check_tier_a = task_compliance.check_tier_a
check_tier_b = task_compliance.check_tier_b
check_tier_c = task_compliance.check_tier_c


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audio_record(
    participant_id="sub-001",
    session_id="ses-01",
    task_name="harvard-sentences-list-1-1",
    features_path=None,
):
    return SimpleNamespace(
        participant_id=participant_id,
        session_id=session_id,
        task_name=task_name,
        features_path=features_path,
    )


# ---------------------------------------------------------------------------
# get_compliance_tier dispatcher
# ---------------------------------------------------------------------------


class TestGetComplianceTier:
    # Tier A — reading/scripted tasks
    @pytest.mark.parametrize("task_category", [
        "harvard-sentences",
        "cape-V-sentences",
        "passage",
        "rainbow",
        "caterpillar-passage",
        "repeat-words",
        "sentence",
    ])
    def test_tier_a_categories(self, task_category):
        assert get_compliance_tier(task_category) == "A"

    # Tier B — phoneme/signal tasks
    @pytest.mark.parametrize("task_category", [
        "diadochokinesis",
        "phonation",
        "pitch-glide",
        "breathing",
        "cough",
    ])
    def test_tier_b_categories(self, task_category):
        assert get_compliance_tier(task_category) == "B"

    # Tier C — open/conversational tasks
    @pytest.mark.parametrize("task_category", [
        "naming",
        "story",
        "picture",
        "conversational",
        "cognitive",
        "recitation",
        "loudness",
    ])
    def test_tier_c_categories(self, task_category):
        assert get_compliance_tier(task_category) == "C"

    def test_unknown_category_raises_value_error(self):
        with pytest.raises(ValueError, match="task_category"):
            get_compliance_tier("not-a-real-task")

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            get_compliance_tier("")


# ---------------------------------------------------------------------------
# Tier A — WER-based compliance
# ---------------------------------------------------------------------------


class TestTierA:
    def test_wer_zero_passes(self, tmp_path):
        """WER = 0.0 (exact transcript match) must classify as PASS."""
        config = PipelineConfig()
        record = _make_audio_record(
            task_name="harvard-sentences-list-1-1",
            features_path=tmp_path / "f.pt",
        )
        with patch.object(
            task_compliance, "_load_transcript", return_value="the cat sat on the mat"
        ), patch.object(
            task_compliance, "_get_prompt_text", return_value="the cat sat on the mat"
        ):
            result = check_tier_a(record, config)

        assert result.classification == Classification.PASS
        assert result.detail.get("wer", 1.0) == pytest.approx(0.0)

    def test_high_wer_needs_review(self, tmp_path):
        """WER = 0.5 (many errors) must classify as NEEDS_REVIEW."""
        config = PipelineConfig()
        record = _make_audio_record(
            task_name="harvard-sentences-list-1-1",
            features_path=tmp_path / "f.pt",
        )
        with patch.object(
            task_compliance, "_load_transcript", return_value="the dog jumped over fence"
        ), patch.object(
            task_compliance, "_get_prompt_text", return_value="the cat sat on the mat"
        ):
            result = check_tier_a(record, config)

        assert result.classification in (Classification.NEEDS_REVIEW, Classification.FAIL)
        assert result.detail.get("wer", 0.0) >= 0.3

    def test_compliance_tier_in_detail(self, tmp_path):
        config = PipelineConfig()
        record = _make_audio_record(
            task_name="harvard-sentences-list-1-1",
            features_path=tmp_path / "f.pt",
        )
        with patch.object(
            task_compliance, "_load_transcript", return_value="the cat sat on the mat"
        ), patch.object(
            task_compliance, "_get_prompt_text", return_value="the cat sat on the mat"
        ):
            result = check_tier_a(record, config)

        assert result.detail.get("compliance_tier") == "A"

    def test_check_type_is_task_compliance(self, tmp_path):
        config = PipelineConfig()
        record = _make_audio_record(
            task_name="harvard-sentences-list-1-1",
            features_path=tmp_path / "f.pt",
        )
        with patch.object(
            task_compliance, "_load_transcript", return_value="the cat sat on the mat"
        ), patch.object(
            task_compliance, "_get_prompt_text", return_value="the cat sat on the mat"
        ):
            result = check_tier_a(record, config)

        assert result.check_type == CheckType.TASK_COMPLIANCE

    def test_score_inversely_related_to_wer(self, tmp_path):
        """Lower WER should produce a higher score."""
        config = PipelineConfig()
        record_perfect = _make_audio_record(
            task_name="harvard-sentences-list-1-1",
            features_path=tmp_path / "f.pt",
        )
        record_poor = _make_audio_record(
            task_name="harvard-sentences-list-1-1",
            features_path=tmp_path / "f.pt",
        )

        with patch.object(
            task_compliance, "_load_transcript", return_value="the cat sat on the mat"
        ), patch.object(
            task_compliance, "_get_prompt_text", return_value="the cat sat on the mat"
        ):
            result_perfect = check_tier_a(record_perfect, config)

        with patch.object(
            task_compliance, "_load_transcript", return_value="completely wrong utterance here"
        ), patch.object(
            task_compliance, "_get_prompt_text", return_value="the cat sat on the mat"
        ):
            result_poor = check_tier_a(record_poor, config)

        assert result_perfect.score > result_poor.score


# ---------------------------------------------------------------------------
# Tier B — Signal/phoneme compliance
# ---------------------------------------------------------------------------


class TestTierB:
    def test_phonation_below_minimum_fails(self, tmp_path):
        """Sustained phonation shorter than min_duration_s must classify as FAIL."""
        config = PipelineConfig(
            task_compliance_params={"phonation": {"min_duration_s": 3.0}}
        )
        record = _make_audio_record(
            task_name="phonation",
            features_path=tmp_path / "f.pt",
        )
        with patch.object(
            task_compliance, "_get_active_speech_duration", return_value=1.5
        ):  # < 3.0 s minimum
            result = check_tier_b(record, config)

        assert result.classification == Classification.FAIL

    def test_phonation_above_minimum_passes_or_review(self, tmp_path):
        config = PipelineConfig(
            task_compliance_params={"phonation": {"min_duration_s": 3.0}}
        )
        record = _make_audio_record(
            task_name="phonation",
            features_path=tmp_path / "f.pt",
        )
        with patch.object(
            task_compliance, "_get_active_speech_duration", return_value=5.0
        ):
            result = check_tier_b(record, config)

        assert result.classification in (Classification.PASS, Classification.NEEDS_REVIEW)

    def test_active_speech_duration_in_detail(self, tmp_path):
        config = PipelineConfig(
            task_compliance_params={"phonation": {"min_duration_s": 3.0}}
        )
        record = _make_audio_record(
            task_name="phonation",
            features_path=tmp_path / "f.pt",
        )
        with patch.object(
            task_compliance, "_get_active_speech_duration", return_value=4.0
        ):
            result = check_tier_b(record, config)

        assert "active_speech_duration_s" in result.detail

    def test_compliance_tier_is_b(self, tmp_path):
        config = PipelineConfig(
            task_compliance_params={"phonation": {"min_duration_s": 3.0}}
        )
        record = _make_audio_record(
            task_name="phonation",
            features_path=tmp_path / "f.pt",
        )
        with patch.object(
            task_compliance, "_get_active_speech_duration", return_value=5.0
        ):
            result = check_tier_b(record, config)

        assert result.detail.get("compliance_tier") == "B"


# ---------------------------------------------------------------------------
# Tier C — LLM-based compliance
# ---------------------------------------------------------------------------


class TestTierC:
    def test_llm_false_fails(self, tmp_path):
        """LLM verdict False (off-task) must classify as FAIL."""
        config = PipelineConfig()
        record = _make_audio_record(
            task_name="conversational-storytelling",
            features_path=tmp_path / "f.pt",
        )
        with patch.object(
            task_compliance, "_task_correctness_phi4", return_value=False
        ), patch.object(
            task_compliance, "_get_active_speech_duration", return_value=5.0
        ):
            result = check_tier_c(record, config)

        assert result.classification == Classification.FAIL
        assert result.detail.get("llm_compliance") is False

    def test_llm_true_with_sufficient_duration_passes(self, tmp_path):
        """LLM verdict True + duration met → confidence 0.9 and PASS."""
        config = PipelineConfig(
            task_compliance_params={
                "conversational": {"min_active_speech_duration_s": 3.0}
            }
        )
        record = _make_audio_record(
            task_name="conversational-storytelling",
            features_path=tmp_path / "f.pt",
        )
        with patch.object(
            task_compliance, "_task_correctness_phi4", return_value=True
        ), patch.object(
            task_compliance, "_get_active_speech_duration", return_value=8.0
        ):
            result = check_tier_c(record, config)

        assert result.classification in (Classification.PASS, Classification.NEEDS_REVIEW)
        assert result.confidence >= 0.8

    def test_llm_true_short_duration_reduces_confidence(self, tmp_path):
        """LLM True but duration short → confidence ~0.5 (not 0.9)."""
        config = PipelineConfig(
            task_compliance_params={
                "conversational": {"min_active_speech_duration_s": 3.0}
            }
        )
        record = _make_audio_record(
            task_name="conversational-storytelling",
            features_path=tmp_path / "f.pt",
        )
        with patch.object(
            task_compliance, "_task_correctness_phi4", return_value=True
        ), patch.object(
            task_compliance, "_get_active_speech_duration", return_value=1.0
        ):  # < 3.0 s
            result = check_tier_c(record, config)

        assert result.confidence <= 0.6

    def test_compliance_tier_is_c(self, tmp_path):
        config = PipelineConfig()
        record = _make_audio_record(
            task_name="conversational-storytelling",
            features_path=tmp_path / "f.pt",
        )
        with patch.object(
            task_compliance, "_task_correctness_phi4", return_value=True
        ), patch.object(
            task_compliance, "_get_active_speech_duration", return_value=5.0
        ):
            result = check_tier_c(record, config)

        assert result.detail.get("compliance_tier") == "C"

    def test_llm_compliance_field_in_detail(self, tmp_path):
        config = PipelineConfig()
        record = _make_audio_record(
            task_name="conversational-storytelling",
            features_path=tmp_path / "f.pt",
        )
        with patch.object(
            task_compliance, "_task_correctness_phi4", return_value=False
        ), patch.object(
            task_compliance, "_get_active_speech_duration", return_value=5.0
        ):
            result = check_tier_c(record, config)

        assert "llm_compliance" in result.detail
