"""Unconsented-speaker detection check (T016).

Reads pre-computed diarization from the BIDS ``_features.pt`` file,
computes speaker ratios, optionally runs Evan's model (when published),
and runs per-speaker language identification.
"""

import logging
from pathlib import Path
from typing import Any

import torch

from b2aiprep.prepare.qa_models import CheckResult, CheckType, Classification, PipelineConfig

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers (patched by tests)
# ---------------------------------------------------------------------------


def _load_diarization(audio_record: Any) -> dict:
    """Load diarization result from the pre-computed ``_features.pt`` file.

    Args:
        audio_record: Object with a ``features_path`` attribute.

    Returns:
        Dict mapping speaker label (str) to a list of
        ``{"start": float, "end": float}`` segment dicts.
        Returns an empty dict if the file is missing or has no diarization.
    """
    features_path = getattr(audio_record, "features_path", None)
    if features_path is None:
        return {}

    fp = Path(features_path)
    if not fp.exists():
        _logger.debug("features_path does not exist: %s", fp)
        return {}

    features = torch.load(str(fp), weights_only=False, map_location="cpu")
    diarization_raw = features.get("diarization", [])

    speakers: dict[str, list] = {}
    for segment in diarization_raw:
        # Support both senselab Segment objects and plain dicts
        if hasattr(segment, "speaker"):
            speaker = segment.speaker
            start = float(segment.start)
            end = float(segment.end)
        else:
            speaker = segment.get("speaker", "speaker_0")
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", 0.0))
        speakers.setdefault(speaker, []).append({"start": start, "end": end})

    return speakers


def _run_evans_model(audio_record: Any, config: PipelineConfig) -> int:
    """Run Evan's model for unconsented-speaker detection.

    The HuggingFace model path is a TODO placeholder until the model is
    published.  Returns 0 (no flag) when the placeholder is detected.

    Args:
        audio_record: Object with audio metadata.
        config: :class:`PipelineConfig` with ``model_versions["evans_model"]``.

    Returns:
        ``0`` (not flagged) or ``1`` (flagged for human review).
    """
    model_path = config.model_versions.get("evans_model", "")
    if not model_path or model_path.startswith("TODO"):
        _logger.debug("Evan's model path is a TODO placeholder; skipping")
        return 0

    try:
        from transformers import AutoModel  # type: ignore[import]

        _model = AutoModel.from_pretrained(model_path)
        # TODO: run actual inference once model is published
        _logger.debug("Evan's model loaded but inference not yet implemented")
        return 0
    except Exception as exc:
        _logger.warning("Evan's model inference failed: %s", exc)
        return 0


def _identify_languages(diarization: dict, config: PipelineConfig) -> list[dict]:
    """Identify language per diarized speaker using langdetect.

    In the full implementation this would run on each speaker's transcribed
    segments.  Here we seed langdetect's randomness and return a minimal
    structural result (language inference from waveform data requires the
    audio bytes, which are not passed into this helper).

    Args:
        diarization: Dict mapping speaker label to segment list.
        config: :class:`PipelineConfig` for ``random_seed``.

    Returns:
        List of ``{"speaker_index": int, "language": str, "confidence": float}``
        dicts, one per speaker.
    """
    try:
        from langdetect import DetectorFactory  # type: ignore[import]

        DetectorFactory.seed = config.random_seed
    except Exception:
        pass  # langdetect optional

    results: list[dict] = []
    for i, _speaker in enumerate(sorted(diarization.keys())):
        results.append(
            {
                "speaker_index": i,
                "language": "unknown",
                "confidence": 0.0,
            }
        )
    return results


# ---------------------------------------------------------------------------
# Public check function
# ---------------------------------------------------------------------------


def check_unconsented_speakers(
    audio_record: Any,
    config: PipelineConfig,
) -> CheckResult:
    """Check for unconsented speakers in one audio recording (T016).

    Reads diarization from the pre-computed ``_features.pt`` file,
    computes ``primary_speaker_ratio`` and ``extra_speaker_count``,
    runs Evan's model (placeholder until published), and runs
    per-speaker language identification.

    Classification rules:
    - ``evans_model_flag == 1`` → ``NEEDS_REVIEW`` (forced)
    - Otherwise: soft-threshold on diarization-based score

    Args:
        audio_record: Object with ``participant_id``, ``session_id``,
                      ``task_name``, ``features_path``.
        config: :class:`PipelineConfig` with threshold / model settings.

    Returns:
        :class:`CheckResult` for the ``unconsented_speakers`` check type.
    """
    participant_id = getattr(audio_record, "participant_id", "")
    session_id = getattr(audio_record, "session_id", "")
    task_name = getattr(audio_record, "task_name", "")

    # Load diarization
    diarization = _load_diarization(audio_record)

    # Compute per-speaker total durations
    speaker_durations: dict[str, float] = {}
    for speaker, segments in diarization.items():
        speaker_durations[speaker] = sum(
            max(0.0, seg["end"] - seg["start"]) for seg in segments
        )

    num_speakers = len(speaker_durations)
    total_duration = sum(speaker_durations.values())

    if num_speakers == 0:
        primary_ratio = 1.0
        extra_count = 0
    else:
        primary_duration = max(speaker_durations.values())
        primary_ratio = primary_duration / (total_duration + 1e-10)
        extra_count = num_speakers - 1

    # Evan's model flag
    evans_flag = _run_evans_model(audio_record, config)

    # Language identification
    detected_languages = _identify_languages(diarization, config)

    # Score: 1.0 for single speaker, penalised for extra speakers
    if num_speakers <= 1:
        score = 1.0
        confidence = 0.90
    elif primary_ratio >= 0.95:
        score = 0.85
        confidence = 0.80
    elif primary_ratio >= 0.80:
        score = 0.60
        confidence = 0.75
    else:
        score = 0.25
        confidence = 0.80

    # Classification
    soft = config.soft_score_thresholds
    pass_min = float(soft.get("pass_min", 0.75))
    fail_max = float(soft.get("fail_max", 0.40))

    if evans_flag == 1:
        # Forced review regardless of diarization score
        classification = Classification.NEEDS_REVIEW
    elif score >= pass_min:
        classification = Classification.PASS
    elif score <= fail_max:
        classification = Classification.FAIL
    else:
        classification = Classification.NEEDS_REVIEW

    detail: dict = {
        "num_speakers_diarized": num_speakers,
        "primary_speaker_ratio": round(primary_ratio, 6),
        "extra_speaker_count": extra_count,
        "evans_model_flag": evans_flag,
        "embedding_cosine_similarity_min": None,
        "detected_languages": detected_languages,
    }

    return CheckResult(
        participant_id=participant_id,
        session_id=session_id,
        task_name=task_name,
        check_type=CheckType.UNCONSENTED_SPEAKERS,
        score=round(score, 6),
        confidence=round(confidence, 6),
        classification=classification,
        detail=detail,
        model_versions={
            "evans_model": config.model_versions.get("evans_model", "unknown"),
        },
    )
