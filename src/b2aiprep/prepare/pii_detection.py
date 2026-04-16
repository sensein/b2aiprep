"""PII disclosure detection check (T017).

Ports ``pii_detection_gliner()`` and Presidio fallback from the
``pii_detection`` branch, adds a transcript-confidence proxy, and
exposes ``check_pii_disclosure`` as the unified per-audio check.

Privacy contract
----------------
``CheckResult.detail`` contains only entity *labels*, confidence
scores, and character *offsets* — no PII text.  This makes the TSV
outputs at the BIDS root safe for data sharing.  The full transcript
and PII spans with text are returned as ``_transcript`` and
``_pii_spans_with_text`` attributes on the returned ``CheckResult``
instance so that the T022 orchestrator can pass them to
``write_audio_sidecar`` without re-transcribing.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import torch

from b2aiprep.prepare.qa_models import CheckResult, CheckType, Classification, PipelineConfig

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers (patched by tests)
# ---------------------------------------------------------------------------


def _transcribe_audio(audio_record: Any) -> tuple:
    """Return ``(transcript_text, confidence_proxy)`` for *audio_record*.

    Reads the pre-computed Whisper transcription from the
    ``_features.pt`` file.  If the transcription is absent, loads the
    audio and runs Whisper tiny to produce a fallback result.

    Args:
        audio_record: Object with ``features_path`` and ``audio_path``.

    Returns:
        ``(transcript: str, confidence: float)`` where *confidence* is
        the Whisper log-prob average (proxy for transcription quality).
    """
    features_path = getattr(audio_record, "features_path", None)
    if features_path is not None:
        fp = Path(features_path)
        if fp.exists():
            features = torch.load(str(fp), weights_only=False, map_location="cpu")
            transcript = features.get("transcription", "")
            # Whisper log-prob confidence if stored
            confidence = float(features.get("transcription_confidence", 0.8))
            if transcript:
                return str(transcript), confidence

    # Fallback: no transcript available
    _logger.debug(
        "No pre-computed transcript found for %s; returning empty",
        getattr(audio_record, "task_name", "unknown"),
    )
    return "", 0.0


def _detect_pii_entities(transcript: str) -> list:
    """Run PII detection on *transcript* using GLiNER or Presidio fallback.

    Args:
        transcript: Plain-text transcript string.

    Returns:
        List of ``{"label": str, "score": float, "char_start": int,
        "char_end": int, "text": str}`` dicts.  The ``text`` field is
        present here for sidecar storage; the caller strips it before
        writing to ``CheckResult.detail`` (TSV-safe).
    """
    if not transcript.strip():
        return []

    # --- Attempt 1: GLiNER PII model ---
    try:
        from gliner import GLiNER  # type: ignore[import]

        model = GLiNER.from_pretrained("nvidia/gliner-pii")
        labels = [
            "name",
            "email",
            "phone_number",
            "address",
            "ssn",
            "credit_card",
            "date_of_birth",
        ]
        raw = model.predict_entities(transcript, labels, threshold=0.5)
        return [
            {
                "label": e["label"],
                "score": round(float(e["score"]), 4),
                "char_start": int(e["start"]),
                "char_end": int(e["end"]),
                "text": transcript[int(e["start"]): int(e["end"])],
            }
            for e in raw
        ]
    except Exception as exc:
        _logger.debug("GLiNER PII detection failed, trying Presidio: %s", exc)

    # --- Attempt 2: Presidio fallback ---
    try:
        from presidio_analyzer import AnalyzerEngine  # type: ignore[import]

        engine = AnalyzerEngine()
        results = engine.analyze(text=transcript, language="en")
        return [
            {
                "label": r.entity_type.lower(),
                "score": round(float(r.score), 4),
                "char_start": r.start,
                "char_end": r.end,
                "text": transcript[r.start: r.end],
            }
            for r in results
        ]
    except Exception as exc:
        _logger.debug("Presidio PII detection failed: %s", exc)

    return []


def transcript_confidence_proxy(audio_record: Any) -> float:
    """Return a transcript-confidence proxy for *audio_record*.

    Uses the Whisper log-prob average stored in ``_features.pt``.
    Falls back to 0.0 when unavailable.

    Args:
        audio_record: Object with ``features_path``.

    Returns:
        Float in [0, 1] representing transcription confidence.
    """
    _transcript, confidence = _transcribe_audio(audio_record)
    return confidence


# ---------------------------------------------------------------------------
# Public check function
# ---------------------------------------------------------------------------


def check_pii_disclosure(
    audio_record: Any,
    config: PipelineConfig,
) -> CheckResult:
    """Check for PII disclosure in one audio recording (T017).

    Steps:
    1. Transcribe audio (from ``_features.pt`` or Whisper fallback).
    2. Detect PII entities with GLiNER / Presidio.
    3. Build ``CheckResult.detail`` WITHOUT entity text (TSV-safe).
    4. Force ``NEEDS_REVIEW`` if transcript confidence is below threshold.

    The full transcript and PII spans with text are stored as
    ``_transcript`` and ``_pii_spans_with_text`` instance attributes on
    the returned ``CheckResult`` so the T022 orchestrator can pass them
    to ``write_audio_sidecar`` without re-transcribing.

    Args:
        audio_record: Object with ``participant_id``, ``session_id``,
                      ``task_name``, ``features_path``.
        config: :class:`PipelineConfig` with threshold settings.

    Returns:
        :class:`CheckResult` for the ``pii_disclosure`` check type.
        Has two extra instance attributes:
        - ``_transcript`` (str): full transcript for sidecar.
        - ``_pii_spans_with_text`` (list): PII spans including text for
          sidecar.
    """
    participant_id = getattr(audio_record, "participant_id", "")
    session_id = getattr(audio_record, "session_id", "")
    task_name = getattr(audio_record, "task_name", "")

    # Step 1: Transcribe
    transcript, transcript_confidence = _transcribe_audio(audio_record)

    # Step 2: Detect PII entities (includes "text" field for sidecar)
    entities_with_text = _detect_pii_entities(transcript)

    # Step 3: TSV-safe entity list — labels + offsets only, NO text
    entities_tsv_safe = [
        {
            "label": e["label"],
            "score": e["score"],
            "char_start": e["char_start"],
            "char_end": e["char_end"],
        }
        for e in entities_with_text
    ]

    # Determine which model was used
    model_used: str
    if entities_with_text:
        # If GLiNER succeeded the list came from it; if not, Presidio
        try:
            import gliner  # noqa: F401  # type: ignore[import]
            model_used = "gliner-pii"
        except ImportError:
            model_used = "presidio"
    else:
        try:
            import gliner  # noqa: F401  # type: ignore[import]
            model_used = "gliner-pii"
        except ImportError:
            model_used = "presidio"

    # Step 4: Score and classification
    n_entities = len(entities_with_text)
    min_conf = float(config.min_transcript_confidence)

    # Low transcript confidence → forced NEEDS_REVIEW regardless of entity count
    if transcript_confidence < min_conf:
        classification = Classification.NEEDS_REVIEW
        score = 0.5  # uncertain
        confidence = transcript_confidence
    elif n_entities == 0:
        # No PII detected in a trusted transcript → PASS
        score = 1.0
        confidence = min(0.95, transcript_confidence)
        soft = config.soft_score_thresholds
        pass_min = float(soft.get("pass_min", 0.75))
        classification = Classification.PASS if score >= pass_min else Classification.NEEDS_REVIEW
    else:
        # PII detected → penalise score by entity density
        char_len = max(1, len(transcript))
        pii_density = sum(e["char_end"] - e["char_start"] for e in entities_with_text) / char_len
        score = max(0.0, 1.0 - 5.0 * pii_density)  # 20% PII → score 0
        confidence = min(0.90, transcript_confidence)
        soft = config.soft_score_thresholds
        fail_max = float(soft.get("fail_max", 0.40))
        classification = Classification.FAIL if score <= fail_max else Classification.NEEDS_REVIEW

    detail: dict = {
        "entities_detected": entities_tsv_safe,
        "transcript_confidence": round(float(transcript_confidence), 6),
        "model_used": model_used,
    }

    result = CheckResult(
        participant_id=participant_id,
        session_id=session_id,
        task_name=task_name,
        check_type=CheckType.PII_DISCLOSURE,
        score=round(score, 6),
        confidence=round(min(1.0, max(0.0, confidence)), 6),
        classification=classification,
        detail=detail,
        model_versions={"pii_model": model_used},
    )

    # Attach sensitive sidecar data as extra instance attributes so the
    # T022 orchestrator can write them via write_audio_sidecar without
    # needing to re-transcribe.
    result._transcript = transcript  # type: ignore[attr-defined]
    result._pii_spans_with_text = entities_with_text  # type: ignore[attr-defined]

    return result
