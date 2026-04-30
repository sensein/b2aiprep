"""Unconsented-speaker detection check.

Reads pre-computed diarization and speaker embeddings from the BIDS
``_features.pt`` file. When a ``profiles_dir`` is provided, compares
ECAPA-TDNN and SPARC speaker embeddings against a pre-built participant
profile using OR logic. Falls back to diarization-only scoring when
``profiles_dir`` is None.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from b2aiprep.prepare.qa_models import CheckResult, CheckType, Classification, PipelineConfig

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_features(audio_record: Any) -> dict:
    """Load all pre-computed features from ``_features.pt``."""
    features_path = getattr(audio_record, "features_path", None)
    if features_path is None:
        return {}
    fp = Path(features_path)
    if not fp.exists():
        _logger.debug("features_path does not exist: %s", fp)
        return {}
    return torch.load(str(fp), weights_only=False, map_location="cpu")


def _parse_diarization(raw: list) -> tuple[dict, float]:
    """Parse raw diarization segment list.

    Returns a (speaker_dict, total_active_speech_s) pair where
    speaker_dict maps speaker label → list of {"start", "end"} dicts.
    """
    speakers: dict[str, list] = {}
    total_active = 0.0
    for segment in raw:
        if hasattr(segment, "speaker"):
            speaker = str(segment.speaker)
            start = float(segment.start)
            end = float(segment.end)
        else:
            speaker = str(segment.get("speaker", "speaker_0"))
            start = float(segment.get("start", 0.0))
            end = float(segment.get("end", 0.0))
        dur = max(0.0, end - start)
        total_active += dur
        speakers.setdefault(speaker, []).append({"start": start, "end": end})
    return speakers, total_active


def _diarization_stats(speaker_dict: dict) -> tuple[int, float, int]:
    """Return (num_speakers, primary_ratio, extra_count) from speaker dict."""
    num_speakers = len(speaker_dict)
    if num_speakers == 0:
        return 0, 1.0, 0
    speaker_durations = {
        spk: sum(max(0.0, seg["end"] - seg["start"]) for seg in segs)
        for spk, segs in speaker_dict.items()
    }
    total = sum(speaker_durations.values())
    primary = max(speaker_durations.values())
    primary_ratio = primary / (total + 1e-10)
    return num_speakers, primary_ratio, num_speakers - 1


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / (norm + 1e-10)


def _run_evans_model(audio_record: Any, config: PipelineConfig) -> int:
    """Run Evan's model for unconsented-speaker detection.

    Returns 0 when the model path is a TODO placeholder (not yet published).
    """
    model_path = config.model_versions.get("evans_model", "")
    if not model_path or model_path.startswith("TODO"):
        _logger.debug("Evan's model path is a TODO placeholder; skipping")
        return 0
    try:
        from transformers import AutoModel  # type: ignore[import]

        _model = AutoModel.from_pretrained(model_path)
        _logger.debug("Evan's model loaded but inference not yet implemented")
        return 0
    except Exception as exc:
        _logger.warning("Evan's model inference failed: %s", exc)
        return 0


def _identify_languages(diarization: dict, config: PipelineConfig) -> list[dict]:
    """Identify language per diarized speaker (structural placeholder)."""
    try:
        from langdetect import DetectorFactory  # type: ignore[import]

        DetectorFactory.seed = config.random_seed
    except Exception:
        pass
    return [
        {"speaker_index": i, "language": "unknown", "confidence": 0.0}
        for i, _ in enumerate(sorted(diarization.keys()))
    ]


def _diarization_score(num_speakers: int, primary_ratio: float) -> tuple[float, float]:
    """Return (score, confidence) based on diarization signals."""
    if num_speakers <= 1:
        return 1.0, 0.90
    if primary_ratio >= 0.95:
        return 0.85, 0.80
    if primary_ratio >= 0.80:
        return 0.60, 0.75
    return 0.25, 0.80


def _classify(score: float, config: PipelineConfig) -> Classification:
    soft = config.soft_score_thresholds
    pass_min = float(soft.get("pass_min", 0.75))
    fail_max = float(soft.get("fail_max", 0.40))
    if score >= pass_min:
        return Classification.PASS
    if score <= fail_max:
        return Classification.FAIL
    return Classification.NEEDS_REVIEW


# ---------------------------------------------------------------------------
# Public check function
# ---------------------------------------------------------------------------


def check_unconsented_speakers(
    audio_record: Any,
    config: PipelineConfig,
    profiles_dir: Optional[str] = None,
) -> CheckResult:
    """Check for unconsented speakers in one audio recording.

    When ``profiles_dir`` is provided, compares ECAPA-TDNN and SPARC speaker
    embeddings against a pre-built profile using OR logic. Falls back to
    diarization-only scoring when ``profiles_dir`` is None.

    Diarization signals are always computed regardless of recording duration.

    Args:
        audio_record: Object with ``participant_id``, ``session_id``,
                      ``task_name``, ``features_path``.
        config: :class:`PipelineConfig` with threshold / model settings.
        profiles_dir: Root directory of pre-built speaker profiles, or None
                      to use diarization-only logic.

    Returns:
        :class:`CheckResult` for the ``unconsented_speakers`` check type.
    """
    participant_id = getattr(audio_record, "participant_id", "")
    session_id = getattr(audio_record, "session_id", "")
    task_name = getattr(audio_record, "task_name", "")

    # Load all features once
    features = _load_features(audio_record)
    diarization_raw = features.get("diarization", [])
    total_duration = float(features.get("duration", 0.0))

    # Always compute diarization signals
    speaker_dict, active_speech_s = _parse_diarization(diarization_raw)
    num_speakers, primary_ratio, extra_count = _diarization_stats(speaker_dict)

    # ---------- diarization-only fallback ----------
    if profiles_dir is None:
        evans_flag = _run_evans_model(audio_record, config)
        detected_languages = _identify_languages(speaker_dict, config)
        score, base_conf = _diarization_score(num_speakers, primary_ratio)

        if evans_flag == 1:
            classification = Classification.NEEDS_REVIEW
        else:
            classification = _classify(score, config)

        return CheckResult(
            participant_id=participant_id,
            session_id=session_id,
            task_name=task_name,
            check_type=CheckType.UNCONSENTED_SPEAKERS,
            score=round(score, 6),
            confidence=round(base_conf, 6),
            classification=classification,
            detail={
                "num_speakers_diarized": num_speakers,
                "primary_speaker_ratio": round(primary_ratio, 6),
                "extra_speaker_count": extra_count,
                "evans_model_flag": evans_flag,
                "embedding_cosine_similarity_min": None,
                "detected_languages": detected_languages,
            },
            model_versions={
                "evans_model": config.model_versions.get("evans_model", "unknown"),
            },
        )

    # ---------- profile-based path ----------
    from b2aiprep.prepare.speaker_profiles import load_speaker_profile

    sp_cfg = config.speaker_profile
    ecapa_threshold = float(sp_cfg.get("ecapa_cosine_threshold", 0.25))
    sparc_threshold = float(sp_cfg.get("sparc_cosine_threshold", 0.20))
    low_conf_fraction = float(sp_cfg.get("low_confidence_speech_fraction", 0.15))

    profile = load_speaker_profile(profiles_dir, participant_id)

    # --- missing profile ---
    if profile is None:
        return CheckResult(
            participant_id=participant_id,
            session_id=session_id,
            task_name=task_name,
            check_type=CheckType.UNCONSENTED_SPEAKERS,
            score=0.5,
            confidence=0.0,
            classification=Classification.NEEDS_REVIEW,
            detail={
                "profile_status": "missing",
                "num_speakers_diarized": num_speakers,
                "diarization_primary_ratio": round(primary_ratio, 6),
                "extra_speaker_count": extra_count,
                "ecapa_cosine_similarity": None,
                "sparc_cosine_similarity": None,
                "or_flag": True,
                "active_speech_fraction": round(
                    active_speech_s / (total_duration + 1e-10), 6
                ),
                "active_speech_s": round(active_speech_s, 6),
                "speech_fraction_confidence": 0.0,
                "ecapa_model_id": "unknown",
                "sparc_model_id": "unknown",
                "enrollment_n": 0,
                "age_group": "unknown",
            },
            model_versions={},
        )

    # --- profile exists but not ready ---
    if profile.profile_status != "ready":
        return CheckResult(
            participant_id=participant_id,
            session_id=session_id,
            task_name=task_name,
            check_type=CheckType.UNCONSENTED_SPEAKERS,
            score=0.5,
            confidence=0.0,
            classification=Classification.NEEDS_REVIEW,
            detail={
                "profile_status": profile.profile_status,
                "num_speakers_diarized": num_speakers,
                "diarization_primary_ratio": round(primary_ratio, 6),
                "extra_speaker_count": extra_count,
                "ecapa_cosine_similarity": None,
                "sparc_cosine_similarity": None,
                "or_flag": True,
                "active_speech_fraction": round(
                    active_speech_s / (total_duration + 1e-10), 6
                ),
                "active_speech_s": round(active_speech_s, 6),
                "speech_fraction_confidence": 0.0,
                "ecapa_model_id": profile.ecapa_model_id,
                "sparc_model_id": profile.sparc_model_id,
                "enrollment_n": profile.num_recordings_used,
                "age_group": profile.age_group,
            },
            model_versions={},
        )

    # --- compute speech-fraction confidence ceiling ---
    active_speech_fraction = active_speech_s / (total_duration + 1e-10)
    if active_speech_s < 1.0:
        speech_fraction_confidence = 0.10
    elif active_speech_fraction < low_conf_fraction:
        speech_fraction_confidence = 0.30
    else:
        speech_fraction_confidence = 1.0

    # --- very short recording: skip cosine comparison ---
    if active_speech_s < 1.0:
        return CheckResult(
            participant_id=participant_id,
            session_id=session_id,
            task_name=task_name,
            check_type=CheckType.UNCONSENTED_SPEAKERS,
            score=0.5,
            confidence=0.10,
            classification=Classification.NEEDS_REVIEW,
            detail={
                "profile_status": profile.profile_status,
                "num_speakers_diarized": num_speakers,
                "diarization_primary_ratio": round(primary_ratio, 6),
                "extra_speaker_count": extra_count,
                "ecapa_cosine_similarity": None,
                "sparc_cosine_similarity": None,
                "or_flag": True,
                "active_speech_fraction": round(active_speech_fraction, 6),
                "active_speech_s": round(active_speech_s, 6),
                "speech_fraction_confidence": speech_fraction_confidence,
                "ecapa_model_id": profile.ecapa_model_id,
                "sparc_model_id": profile.sparc_model_id,
                "enrollment_n": profile.num_recordings_used,
                "age_group": profile.age_group,
            },
            model_versions={},
        )

    # --- load and L2-normalise embeddings ---
    ecapa_raw = features.get("speaker_embedding")
    sparc_feat = features.get("sparc", {})
    sparc_raw = sparc_feat.get("spk_emb") if isinstance(sparc_feat, dict) else None

    ecapa_emb = (
        _l2_normalize(np.array(ecapa_raw, dtype=np.float64).ravel())
        if ecapa_raw is not None
        else None
    )
    sparc_emb = (
        _l2_normalize(np.array(sparc_raw, dtype=np.float64).ravel())
        if sparc_raw is not None
        else None
    )
    ecapa_centroid = _l2_normalize(
        np.array(profile.ecapa_embedding_centroid, dtype=np.float64)
    )
    sparc_centroid = _l2_normalize(
        np.array(profile.sparc_embedding_centroid, dtype=np.float64)
    )

    ecapa_cosine = (
        float(np.dot(ecapa_emb, ecapa_centroid)) if ecapa_emb is not None else None
    )
    sparc_cosine = (
        float(np.dot(sparc_emb, sparc_centroid)) if sparc_emb is not None else None
    )

    # OR logic: flag if either embedding is below its threshold
    ecapa_below = (ecapa_cosine is None) or (ecapa_cosine < ecapa_threshold)
    sparc_below = (sparc_cosine is None) or (sparc_cosine < sparc_threshold)
    or_flag = ecapa_below or sparc_below

    # Diarization-based score blended with speech-fraction confidence ceiling
    score, base_conf = _diarization_score(num_speakers, primary_ratio)
    confidence = min(base_conf, speech_fraction_confidence)

    if or_flag:
        classification = Classification.NEEDS_REVIEW
    else:
        classification = _classify(score, config)

    return CheckResult(
        participant_id=participant_id,
        session_id=session_id,
        task_name=task_name,
        check_type=CheckType.UNCONSENTED_SPEAKERS,
        score=round(score, 6),
        confidence=round(confidence, 6),
        classification=classification,
        detail={
            "profile_status": profile.profile_status,
            "num_speakers_diarized": num_speakers,
            "diarization_primary_ratio": round(primary_ratio, 6),
            "extra_speaker_count": extra_count,
            "ecapa_cosine_similarity": (
                round(ecapa_cosine, 6) if ecapa_cosine is not None else None
            ),
            "sparc_cosine_similarity": (
                round(sparc_cosine, 6) if sparc_cosine is not None else None
            ),
            "or_flag": or_flag,
            "active_speech_fraction": round(active_speech_fraction, 6),
            "active_speech_s": round(active_speech_s, 6),
            "speech_fraction_confidence": speech_fraction_confidence,
            "ecapa_model_id": profile.ecapa_model_id,
            "sparc_model_id": profile.sparc_model_id,
            "enrollment_n": profile.num_recordings_used,
            "age_group": profile.age_group,
        },
        model_versions={
            "ecapa_model": profile.ecapa_model_id,
            "sparc_model": profile.sparc_model_id,
        },
    )
