"""T011 — PII detection unit tests.

Tests ``check_pii_disclosure(audio_record, config) -> CheckResult`` implemented
in T017.  All tests skip automatically until the module exists.

Assertions:
- Known PII transcript flagged with correct entity labels and char offsets
- Low-transcript-confidence recording forces needs_review
- Full transcript stored in sidecar but entity text NOT in TSV detail
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

# Skip entire file if module not yet implemented (T017)
pii_detection = pytest.importorskip(
    "b2aiprep.prepare.pii_detection",
    reason="pii_detection module not yet implemented (T017)",
)

from b2aiprep.prepare.qa_models import CheckType, Classification, PipelineConfig

check_pii_disclosure = pii_detection.check_pii_disclosure


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_audio_record(
    participant_id="sub-001",
    session_id="ses-01",
    task_name="conversational-storytelling",
    features_path=None,
):
    return SimpleNamespace(
        participant_id=participant_id,
        session_id=session_id,
        task_name=task_name,
        features_path=features_path,
    )


# Synthetic transcripts for testing
_TRANSCRIPT_WITH_PII = "My name is John Smith and my phone is 555-867-5309."
_TRANSCRIPT_CLEAN = "The quick brown fox jumped over the lazy dog."

# Synthetic GLiNER detection result
_PII_ENTITIES = [
    {"label": "name", "score": 0.92, "char_start": 11, "char_end": 21},
    {"label": "phone_number", "score": 0.88, "char_start": 37, "char_end": 49},
]


# ---------------------------------------------------------------------------
# PII entity detection tests
# ---------------------------------------------------------------------------


class TestPiiEntityDetection:
    def test_known_pii_transcript_flagged(self, tmp_path):
        """Transcript containing name + phone must produce >= 1 detected entity."""
        config = PipelineConfig()
        record = _make_audio_record(features_path=tmp_path / "f.pt")

        with patch.object(
            pii_detection, "_transcribe_audio", return_value=(_TRANSCRIPT_WITH_PII, 0.90)
        ), patch.object(
            pii_detection, "_detect_pii_entities", return_value=_PII_ENTITIES
        ):
            result = check_pii_disclosure(record, config)

        assert len(result.detail["entities_detected"]) >= 1

    def test_entity_label_preserved(self, tmp_path):
        config = PipelineConfig()
        record = _make_audio_record(features_path=tmp_path / "f.pt")

        with patch.object(
            pii_detection, "_transcribe_audio", return_value=(_TRANSCRIPT_WITH_PII, 0.90)
        ), patch.object(
            pii_detection, "_detect_pii_entities", return_value=_PII_ENTITIES
        ):
            result = check_pii_disclosure(record, config)

        labels = [e["label"] for e in result.detail["entities_detected"]]
        assert "name" in labels

    def test_char_offsets_present_in_tsv_detail(self, tmp_path):
        config = PipelineConfig()
        record = _make_audio_record(features_path=tmp_path / "f.pt")

        with patch.object(
            pii_detection, "_transcribe_audio", return_value=(_TRANSCRIPT_WITH_PII, 0.90)
        ), patch.object(
            pii_detection, "_detect_pii_entities", return_value=_PII_ENTITIES
        ):
            result = check_pii_disclosure(record, config)

        for entity in result.detail["entities_detected"]:
            assert "char_start" in entity
            assert "char_end" in entity

    def test_entity_text_not_in_tsv_detail(self, tmp_path):
        """TSV detail must NOT contain the raw PII text (only offsets)."""
        config = PipelineConfig()
        record = _make_audio_record(features_path=tmp_path / "f.pt")

        with patch.object(
            pii_detection, "_transcribe_audio", return_value=(_TRANSCRIPT_WITH_PII, 0.90)
        ), patch.object(
            pii_detection, "_detect_pii_entities", return_value=_PII_ENTITIES
        ):
            result = check_pii_disclosure(record, config)

        for entity in result.detail["entities_detected"]:
            assert "text" not in entity, (
                "PII text must NOT appear in TSV detail — only in per-audio sidecar"
            )

    def test_clean_transcript_produces_no_entities(self, tmp_path):
        config = PipelineConfig()
        record = _make_audio_record(features_path=tmp_path / "f.pt")

        with patch.object(
            pii_detection, "_transcribe_audio", return_value=(_TRANSCRIPT_CLEAN, 0.95)
        ), patch.object(
            pii_detection, "_detect_pii_entities", return_value=[]
        ):
            result = check_pii_disclosure(record, config)

        assert result.detail["entities_detected"] == []


# ---------------------------------------------------------------------------
# Transcript confidence gating
# ---------------------------------------------------------------------------


class TestTranscriptConfidence:
    def test_low_confidence_forces_needs_review(self, tmp_path):
        """Transcript confidence below min_transcript_confidence must force NEEDS_REVIEW."""
        config = PipelineConfig(min_transcript_confidence=0.70)
        record = _make_audio_record(features_path=tmp_path / "f.pt")

        # confidence = 0.50 < threshold 0.70 → NEEDS_REVIEW
        with patch.object(
            pii_detection, "_transcribe_audio", return_value=(_TRANSCRIPT_CLEAN, 0.50)
        ), patch.object(
            pii_detection, "_detect_pii_entities", return_value=[]
        ):
            result = check_pii_disclosure(record, config)

        assert result.classification == Classification.NEEDS_REVIEW

    def test_high_confidence_not_forced_to_review(self, tmp_path):
        config = PipelineConfig(min_transcript_confidence=0.70)
        record = _make_audio_record(features_path=tmp_path / "f.pt")

        with patch.object(
            pii_detection, "_transcribe_audio", return_value=(_TRANSCRIPT_CLEAN, 0.95)
        ), patch.object(
            pii_detection, "_detect_pii_entities", return_value=[]
        ):
            result = check_pii_disclosure(record, config)

        # High confidence → should NOT be forced to needs_review by confidence alone
        assert result.classification != Classification.NEEDS_REVIEW or (
            len(result.detail["entities_detected"]) > 0
        )

    def test_transcript_confidence_in_detail(self, tmp_path):
        config = PipelineConfig()
        record = _make_audio_record(features_path=tmp_path / "f.pt")

        with patch.object(
            pii_detection, "_transcribe_audio", return_value=(_TRANSCRIPT_CLEAN, 0.88)
        ), patch.object(
            pii_detection, "_detect_pii_entities", return_value=[]
        ):
            result = check_pii_disclosure(record, config)

        assert "transcript_confidence" in result.detail
        assert result.detail["transcript_confidence"] == pytest.approx(0.88)


# ---------------------------------------------------------------------------
# Transcript stored in sidecar only (not TSV detail)
# ---------------------------------------------------------------------------


class TestSidecarVsTsvBoundary:
    def test_check_type_is_pii_disclosure(self, tmp_path):
        config = PipelineConfig()
        record = _make_audio_record(features_path=tmp_path / "f.pt")

        with patch.object(
            pii_detection, "_transcribe_audio", return_value=(_TRANSCRIPT_CLEAN, 0.90)
        ), patch.object(
            pii_detection, "_detect_pii_entities", return_value=[]
        ):
            result = check_pii_disclosure(record, config)

        assert result.check_type == CheckType.PII_DISCLOSURE

    def test_transcript_not_in_check_result_detail(self, tmp_path):
        """The full transcript text must NOT appear in CheckResult.detail
        (it goes to the per-audio JSON sidecar only)."""
        config = PipelineConfig()
        record = _make_audio_record(features_path=tmp_path / "f.pt")

        with patch.object(
            pii_detection, "_transcribe_audio", return_value=(_TRANSCRIPT_WITH_PII, 0.90)
        ), patch.object(
            pii_detection, "_detect_pii_entities", return_value=_PII_ENTITIES
        ):
            result = check_pii_disclosure(record, config)

        # The full transcript string must not appear as a value in the detail dict
        for v in result.detail.values():
            assert v != _TRANSCRIPT_WITH_PII, (
                "Full transcript must NOT appear in CheckResult.detail (TSV-safe)"
            )

    def test_model_used_recorded(self, tmp_path):
        config = PipelineConfig()
        record = _make_audio_record(features_path=tmp_path / "f.pt")

        with patch.object(
            pii_detection, "_transcribe_audio", return_value=(_TRANSCRIPT_CLEAN, 0.90)
        ), patch.object(
            pii_detection, "_detect_pii_entities", return_value=[]
        ):
            result = check_pii_disclosure(record, config)

        assert "model_used" in result.detail
        assert result.detail["model_used"] in ("gliner-pii", "presidio")
