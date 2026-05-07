"""Embedding reliability evaluation for speaker profile verification (US3).

Generates synthetic audio mixtures, extracts embeddings, computes operating
characteristic curves (FNR vs. review-queue fraction), and writes a structured
reliability report for threshold calibration.

The ``embedding-reliability-report`` CLI command orchestrates all steps.
:func:`extract_embeddings_for_mixtures` requires model inference and may be
slow without GPU.
"""

import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torchaudio

_logger = logging.getLogger(__name__)

_REPORT_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Default evaluation parameters — see research.md Decision 9 for rationale.
# ---------------------------------------------------------------------------
_DEFAULT_MAX_FNR_TARGET = 0.05
_DEFAULT_KNEE_DROP_PP = 0.15
_DEFAULT_LOW_CONFIDENCE_SPEECH_FRACTION = 0.15
_DEFAULT_MIN_ACTIVE_SPEECH_S = 3.0


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _overlay_intruder(
    base: torch.Tensor,
    intruder: torch.Tensor,
    snr_db: float,
    position: str = "end",
) -> torch.Tensor:
    """Overlay intruder clip into base at the specified position and SNR.

    Both tensors are (channels, samples). The intruder is energy-scaled to
    achieve ``snr_db`` relative to the base segment it overlaps.
    Output has exactly the same shape as ``base``.

    Args:
        position: Where in the base recording to place the intruder.
            ``"end"`` (default) — tail; ``"start"`` — beginning;
            ``"middle"`` — centred.
    """
    base = base.float()
    intruder = intruder.float()

    n_base = base.shape[-1]
    n_intruder = intruder.shape[-1]

    if n_intruder == 0 or n_base == 0:
        return base.clone()

    actual_len = min(n_intruder, n_base)

    if position == "start":
        start_idx = 0
    elif position == "middle":
        start_idx = (n_base - actual_len) // 2
    else:  # "end"
        start_idx = max(0, n_base - actual_len)

    base_segment = base[:, start_idx:start_idx + actual_len]
    base_rms = float(base_segment.pow(2).mean().sqrt()) + 1e-10
    intruder_clip = intruder[:, :actual_len]
    intruder_rms = float(intruder_clip.pow(2).mean().sqrt()) + 1e-10

    target_rms = base_rms / (10.0 ** (snr_db / 20.0))
    scale = target_rms / intruder_rms
    intruder_scaled = intruder_clip * scale

    mixed = base.clone()
    mixed[:, start_idx:start_idx + actual_len] = (
        mixed[:, start_idx:start_idx + actual_len] + intruder_scaled
    )
    return mixed


def _fnr_fpr_at_threshold(
    positive_scores: list,
    negative_scores: list,
    threshold: float,
) -> tuple:
    """Compute False Negative Rate and False Positive Rate at a cosine threshold.

    Scores BELOW the threshold are flagged (intruder detected = predicted positive).
    ``positive_scores``: cosine similarities for true-positive mixture recordings.
    ``negative_scores``: cosine similarities for true-negative solo recordings.
    FNR = fraction of positives NOT flagged; FPR = fraction of negatives flagged.
    """
    fnr = 0.0
    if positive_scores:
        missed = sum(1 for s in positive_scores if s >= threshold)
        fnr = missed / len(positive_scores)

    fpr = 0.0
    if negative_scores:
        false_alarms = sum(1 for s in negative_scores if s < threshold)
        fpr = false_alarms / len(negative_scores)

    return fnr, fpr


def compute_single_curve(
    scores: list,
    labels: list,
    threshold_step: float = 0.01,
) -> list:
    """Compute a single operating characteristic curve by sweeping the threshold.

    Convenience function for testing and threshold exploration. For the full
    mixture-based analysis see :func:`compute_operating_curves`.

    Args:
        scores: Cosine similarity scores (higher = more similar = less suspicious).
        labels: True labels; ``True`` = intruder present, ``False`` = solo.
        threshold_step: Step size for threshold sweep.

    Returns:
        List of ``{threshold, fnr, fpr, review_fraction}`` dicts.
    """
    positive_scores = [s for s, l in zip(scores, labels) if l]
    negative_scores = [s for s, l in zip(scores, labels) if not l]

    curve = []
    threshold = 0.0
    while threshold <= 1.0 + 1e-9:
        fnr, fpr = _fnr_fpr_at_threshold(positive_scores, negative_scores, threshold)
        n_flagged = sum(1 for s in scores if s < threshold)
        review_fraction = n_flagged / max(len(scores), 1)
        curve.append({
            "threshold": round(threshold, 4),
            "fnr": round(fnr, 6),
            "fpr": round(fpr, 6),
            "review_fraction": round(review_fraction, 6),
        })
        threshold = round(threshold + threshold_step, 10)

    return curve


# ---------------------------------------------------------------------------
# BIDS helpers
# ---------------------------------------------------------------------------


def _find_features_files(bids_dir: Path) -> dict:
    """Scan BIDS directory for ``*_features.pt`` files grouped by participant ID."""
    result: dict = {}
    for pt_file in bids_dir.rglob("*_features.pt"):
        for part in pt_file.parts:
            if part.startswith("sub-"):
                pid = part[4:]
                result.setdefault(pid, []).append(pt_file)
                break
    return result


def _get_audio_path(features_path: Path, features: dict) -> Optional[Path]:
    """Return the audio file path corresponding to a ``_features.pt`` file."""
    stored = features.get("audio_path")
    if stored:
        p = Path(stored)
        if p.exists():
            return p

    name = features_path.name
    if name.endswith("_features.pt"):
        stem = name[: -len("_features.pt")]
        for ext in (".wav", ".mp3", ".flac", ".ogg"):
            candidate = features_path.parent / (stem + ext)
            if candidate.exists():
                return candidate

    return None


# ---------------------------------------------------------------------------
# T014: Synthetic mixture generation
# ---------------------------------------------------------------------------


def generate_synthetic_mixtures(
    bids_dir: str,
    profiles_dir: str,
    intruder_ratios: list,
    intruder_snr_db_values: list,
    output_dir: str,
    config: Optional[Any] = None,
    intruder_bids_dir: Optional[str] = None,
    positions: Optional[list] = None,
    seed: int = 42,
) -> list:
    """Generate synthetic intruder mixtures for operating characteristic evaluation.

    For each participant with a ``ready`` profile, pairs each clean recording
    with a random intruder participant and writes one mixture per
    (ratio, SNR, position) combination. A matching ``negative`` (unmixed) sample
    is also written.

    Args:
        bids_dir: Root of the BIDS dataset containing ``_features.pt`` files.
            Target (enrolled) participants are always drawn from here.
        profiles_dir: Directory of pre-built speaker profiles.
        intruder_ratios: Fractions of base recording duration to overlay,
            e.g. ``[0.10, 0.20, 0.40]``.
        intruder_snr_db_values: SNR values in dB, e.g. ``[0, 5, 10]``.
        output_dir: Where to write mixture ``.wav`` files.
        config: Optional :class:`PipelineConfig` (unused; reserved for future gating).
        intruder_bids_dir: Optional second BIDS directory to draw intruder audio
            from (e.g. adult recordings). When provided, ALL intruders are drawn
            from this pool, tagged ``intruder_type="adult"``.  When ``None``,
            intruders are drawn from other participants in ``bids_dir``, tagged
            ``intruder_type="peds"``.
        positions: List of intruder placement positions to generate. Each value
            must be one of ``"start"``, ``"middle"``, ``"end"``. Defaults to
            ``["end"]`` for backward compatibility. Passing all three produces
            three mixtures per (ratio, SNR) pair.
        seed: Random seed for reproducible intruder selection.

    Returns:
        List of mixture dicts with keys ``target_participant_id``,
        ``intruder_participant_id``, ``intruder_type``, ``base_recording_path``,
        ``intruder_segment_path``, ``intruder_duration_ratio``, ``intruder_snr_db``,
        ``intruder_position``, ``label``, ``mixed_audio_path``.
    """
    from b2aiprep.prepare.speaker_profiles import load_speaker_profile

    if positions is None:
        positions = ["end"]

    rng = random.Random(seed)
    bids_path = Path(bids_dir)
    profiles_path = Path(profiles_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_features = _find_features_files(bids_path)
    participants = list(all_features.keys())

    # Build intruder pool
    if intruder_bids_dir is not None:
        intruder_features = _find_features_files(Path(intruder_bids_dir))
        intruder_pool = list(intruder_features.keys())
        intruder_type_label = "adult"
        _logger.info(
            "Using external intruder pool from %s (%d participants)",
            intruder_bids_dir, len(intruder_pool),
        )
    else:
        intruder_features = all_features
        intruder_pool = None  # resolved per-target to exclude self
        intruder_type_label = "peds"

    if len(participants) < 2 and intruder_pool is None:
        _logger.warning("Need ≥ 2 participants for synthetic mixtures; found %d", len(participants))
        return []

    if intruder_pool is not None and len(intruder_pool) == 0:
        _logger.warning("Intruder BIDS dir has no feature files; no mixtures will be generated.")
        return []

    mixtures: list = []

    for target_pid in participants:
        profile = load_speaker_profile(profiles_path, target_pid)
        if profile is None or profile.profile_status != "ready":
            _logger.debug("Skipping %s: profile status=%s",
                          target_pid, profile.profile_status if profile else "missing")
            continue

        available_intruders = (
            intruder_pool
            if intruder_pool is not None
            else [p for p in participants if p != target_pid]
        )
        if not available_intruders:
            continue

        for pt_file in all_features[target_pid]:
            try:
                features = torch.load(str(pt_file), weights_only=False, map_location="cpu")
            except Exception as exc:
                _logger.debug("Cannot load %s: %s", pt_file, exc)
                continue

            audio_path = _get_audio_path(pt_file, features)
            if audio_path is None:
                _logger.debug("No audio file found for %s", pt_file)
                continue

            try:
                base_waveform, base_sr = torchaudio.load(str(audio_path))
            except Exception as exc:
                _logger.debug("Cannot load audio %s: %s", audio_path, exc)
                continue

            # Write negative (unmixed solo) — one per base recording, position-independent
            neg_name = f"{target_pid}_solo_{pt_file.stem}.wav"
            neg_out = out_path / neg_name
            if not neg_out.exists():
                torchaudio.save(str(neg_out), base_waveform, base_sr)
            mixtures.append({
                "target_participant_id": target_pid,
                "intruder_participant_id": target_pid,
                "intruder_type": intruder_type_label,
                "base_recording_path": str(audio_path),
                "intruder_segment_path": str(audio_path),
                "intruder_duration_ratio": 0.0,
                "intruder_snr_db": 0.0,
                "intruder_position": "none",
                "label": "negative",
                "mixed_audio_path": str(neg_out),
            })

            # Select random intruder
            intruder_pid = rng.choice(available_intruders)
            intruder_pt_files = intruder_features.get(intruder_pid, [])
            if not intruder_pt_files:
                continue
            intruder_pt = rng.choice(intruder_pt_files)

            try:
                int_features = torch.load(str(intruder_pt), weights_only=False, map_location="cpu")
                intruder_audio_path = _get_audio_path(intruder_pt, int_features)
                if intruder_audio_path is None:
                    continue
                intruder_waveform, int_sr = torchaudio.load(str(intruder_audio_path))
            except Exception as exc:
                _logger.debug("Cannot load intruder audio for %s: %s", intruder_pid, exc)
                continue

            if int_sr != base_sr:
                intruder_waveform = torchaudio.functional.resample(
                    intruder_waveform, int_sr, base_sr
                )

            # Match channel count
            if intruder_waveform.shape[0] < base_waveform.shape[0]:
                intruder_waveform = intruder_waveform.expand(base_waveform.shape[0], -1)
            elif intruder_waveform.shape[0] > base_waveform.shape[0]:
                intruder_waveform = intruder_waveform[: base_waveform.shape[0], :]

            for ratio in intruder_ratios:
                n_intruder = int(base_waveform.shape[-1] * ratio)
                if n_intruder == 0:
                    continue
                intruder_clip = intruder_waveform[:, :n_intruder]

                for snr_db in intruder_snr_db_values:
                    for pos in positions:
                        mixed = _overlay_intruder(
                            base_waveform, intruder_clip, float(snr_db), position=pos
                        )
                        ratio_str = f"{ratio}".replace(".", "p")
                        snr_str = f"{float(snr_db)}".replace(".", "p").replace("-", "n")
                        out_name = (
                            f"{target_pid}_{intruder_pid}"
                            f"_ratio{ratio_str}_snr{snr_str}_pos{pos}_{pt_file.stem}.wav"
                        )
                        out_file = out_path / out_name
                        torchaudio.save(str(out_file), mixed, base_sr)
                        mixtures.append({
                            "target_participant_id": target_pid,
                            "intruder_participant_id": intruder_pid,
                            "intruder_type": intruder_type_label,
                            "base_recording_path": str(audio_path),
                            "intruder_segment_path": str(intruder_audio_path),
                            "intruder_duration_ratio": float(ratio),
                            "intruder_snr_db": float(snr_db),
                            "intruder_position": pos,
                            "label": "positive",
                            "mixed_audio_path": str(out_file),
                        })

    return mixtures


# ---------------------------------------------------------------------------
# T015: Embedding extraction for mixture audio
# ---------------------------------------------------------------------------


def extract_embeddings_for_mixtures(mixture_list: list) -> dict:
    """Extract ECAPA-TDNN and SPARC embeddings from synthetic mixture audio files.

    Uses senselab for both extractors. Models are loaded once and reused across
    all mixtures. This function requires model inference and may be slow on CPU.

    Args:
        mixture_list: SyntheticMixture dicts from :func:`generate_synthetic_mixtures`.

    Returns:
        Dict mapping ``mixed_audio_path → {ecapa_emb: ndarray|None, sparc_emb: ndarray|None}``.
    """
    try:
        from senselab.audio.data_structures.audio import Audio
        from senselab.audio.tasks.speaker_embeddings.api import (
            extract_speaker_embeddings_from_audios,
        )
        _ecapa_ok = True
    except ImportError:
        _logger.warning(
            "senselab speaker-embeddings not available; ECAPA embeddings will be None."
        )
        _ecapa_ok = False
        Audio = None  # type: ignore[assignment]

    try:
        from senselab.audio.data_structures.audio import Audio  # noqa: F811
        from senselab.audio.tasks.features_extraction.sparc import SparcFeatureExtractor
        _sparc_extractor = SparcFeatureExtractor()
        _sparc_ok = True
    except Exception as exc:
        _logger.warning("SPARC extractor unavailable; SPARC embeddings will be None: %s", exc)
        _sparc_extractor = None
        _sparc_ok = False

    result: dict = {}

    for mixture in mixture_list:
        audio_path = mixture["mixed_audio_path"]
        ecapa_emb = None
        sparc_emb = None

        try:
            waveform, sr = torchaudio.load(audio_path)
            audio_obj = Audio(waveform=waveform, sampling_rate=sr)
        except Exception as exc:
            _logger.warning("Could not load audio %s: %s", audio_path, exc)
            result[audio_path] = {"ecapa_emb": None, "sparc_emb": None}
            continue

        if _ecapa_ok:
            try:
                embeddings = extract_speaker_embeddings_from_audios([audio_obj])
                if embeddings and embeddings[0] is not None:
                    raw = embeddings[0]
                    if hasattr(raw, "numpy"):
                        ecapa_emb = raw.numpy().ravel()
                    elif isinstance(raw, np.ndarray):
                        ecapa_emb = raw.ravel()
                    else:
                        ecapa_emb = np.array(raw, dtype=np.float64).ravel()
            except Exception as exc:
                _logger.warning("ECAPA extraction failed for %s: %s", audio_path, exc)

        if _sparc_ok:
            try:
                sparc_result = _sparc_extractor.extract_sparc_features([audio_obj])
                if sparc_result and sparc_result[0] is not None:
                    raw_spk = sparc_result[0].get("spk_emb")
                    if raw_spk is not None:
                        if hasattr(raw_spk, "numpy"):
                            sparc_emb = raw_spk.numpy().ravel()
                        elif isinstance(raw_spk, np.ndarray):
                            sparc_emb = raw_spk.ravel()
                        else:
                            sparc_emb = np.array(raw_spk, dtype=np.float64).ravel()
            except Exception as exc:
                _logger.warning("SPARC extraction failed for %s: %s", audio_path, exc)

        result[audio_path] = {"ecapa_emb": ecapa_emb, "sparc_emb": sparc_emb}

    return result


# ---------------------------------------------------------------------------
# T016: Operating characteristic computation
# ---------------------------------------------------------------------------


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / (norm + 1e-10)


def _recommended_threshold(curve: list, max_fnr: float = _DEFAULT_MAX_FNR_TARGET) -> float:
    """Return the lowest threshold where FNR ≤ max_fnr; fall back to minimum-FNR point."""
    for point in curve:
        if point["fnr"] <= max_fnr:
            return point["threshold"]
    best = min(curve, key=lambda p: p["fnr"])
    return best["threshold"]


def _per_bin_stats(score_label_fraction_age: list, bins: list) -> list:
    """Compute per-speech-fraction-bin mean cosine stats."""
    stats = []
    for lo, hi in bins:
        in_bin = [(s, l) for s, l, f, _ in score_label_fraction_age if lo <= f < hi]
        if not in_bin:
            stats.append({
                "bin": [lo, hi], "n": 0,
                "mean_cosine_same": None, "mean_cosine_diff": None,
                "accuracy": None, "fpr": None,
            })
            continue
        same = [s for s, l in in_bin if not l]
        diff = [s for s, l in in_bin if l]
        stats.append({
            "bin": [lo, hi],
            "n": len(in_bin),
            "mean_cosine_same": round(float(np.mean(same)), 6) if same else None,
            "mean_cosine_diff": round(float(np.mean(diff)), 6) if diff else None,
            "accuracy": None,
            "fpr": None,
        })
    return stats


def _compute_knee_point(
    score_label_fraction_age: list,
    bins: list,
    knee_drop_pp: float = _DEFAULT_KNEE_DROP_PP,
) -> float:
    """Speech fraction boundary where per-bin same-speaker cosine drops > knee_drop_pp."""
    bin_means = []
    for lo, hi in bins:
        same = [s for s, l, f, _ in score_label_fraction_age if lo <= f < hi and not l]
        bin_means.append((lo, hi, float(np.mean(same)) if same else None))

    top_mean = max((m for _, _, m in bin_means if m is not None), default=None)
    if top_mean is None:
        return 0.0
    for lo, _, m in bin_means:
        if m is not None and top_mean - m > knee_drop_pp:
            return float(lo)
    return 0.0


def _build_single_emb_curve(rows: list) -> list:
    """Build an operating curve from (cosine, is_positive, fraction, age) rows."""
    pos = [s for s, l, _, _ in rows if l]
    neg = [s for s, l, _, _ in rows if not l]
    all_scores = [s for s, _, _, _ in rows]
    curve = []
    t = 0.0
    while t <= 1.0 + 1e-9:
        fnr, fpr = _fnr_fpr_at_threshold(pos, neg, t)
        rf = sum(1 for s in all_scores if s < t) / max(len(all_scores), 1)
        curve.append({
            "threshold": round(t, 4),
            "fnr": round(fnr, 6),
            "fpr": round(fpr, 6),
            "review_fraction": round(rf, 6),
        })
        t = round(t + 0.01, 10)
    return curve


def _build_or_curve(ec_rows: list, sp_rows: list) -> list:
    """Build an OR-combined operating curve from paired ECAPA and SPARC rows."""
    if not sp_rows:
        return _build_single_emb_curve(ec_rows)

    if len(ec_rows) != len(sp_rows):
        _logger.debug("ECAPA/SPARC row counts differ; using ECAPA only for OR curve")
        return _build_single_emb_curve(ec_rows)

    curve = []
    t = 0.0
    while t <= 1.0 + 1e-9:
        tp = fn_count = fp = tn = 0
        for (ec_s, ec_l, _, _), (sp_s, _, _, _) in zip(ec_rows, sp_rows):
            or_flag = (ec_s < t) or (sp_s < t)
            is_pos = ec_l
            if is_pos and or_flag:
                tp += 1
            elif is_pos:
                fn_count += 1
            elif or_flag:
                fp += 1
            else:
                tn += 1
        total_pos = tp + fn_count
        total_neg = fp + tn
        fnr = fn_count / max(total_pos, 1)
        fpr = fp / max(total_neg, 1)
        rf = (tp + fp) / max(tp + fp + fn_count + tn, 1)
        curve.append({
            "threshold": round(t, 4),
            "fnr": round(fnr, 6),
            "fpr": round(fpr, 6),
            "review_fraction": round(rf, 6),
        })
        t = round(t + 0.01, 10)
    return curve


def compute_operating_curves(
    mixture_list: list,
    profiles_dir: str,
    emb_dict: dict,
    speech_fraction_bins: Optional[list] = None,
    max_fnr_target: float = _DEFAULT_MAX_FNR_TARGET,
    knee_drop_pp: float = _DEFAULT_KNEE_DROP_PP,
) -> dict:
    """Score mixtures against participant profiles and compute operating curves.

    Sweeps cosine threshold from 0.0 to 1.0 in steps of 0.01. Computes FNR,
    FPR, and review-queue fraction for ECAPA-TDNN, SPARC, and OR-combined.

    Args:
        mixture_list: SyntheticMixture dicts from :func:`generate_synthetic_mixtures`.
        profiles_dir: Directory containing per-participant ``speaker_profile.json`` files.
        emb_dict: Output of :func:`extract_embeddings_for_mixtures`.
        speech_fraction_bins: Bin boundaries as list of (lo, hi) pairs.
        max_fnr_target: FNR ceiling used to select recommended thresholds.
        knee_drop_pp: Accuracy drop (percentage points) that defines the knee point.

    Returns:
        :class:`EmbeddingReliabilityReport` dict.
    """
    from b2aiprep.prepare.speaker_profiles import load_speaker_profile

    if speech_fraction_bins is None:
        speech_fraction_bins = [(0, 0.15), (0.15, 0.30), (0.30, 0.60), (0.60, 1.01)]

    profiles_path = Path(profiles_dir)
    ecapa_rows: list = []
    sparc_rows: list = []
    # Per-intruder-type rows: type_label -> {"ecapa": [...], "sparc": [...]}
    by_intruder_type: dict = {}

    for mixture in mixture_list:
        audio_path = mixture["mixed_audio_path"]
        target_pid = mixture["target_participant_id"]
        is_positive = mixture["label"] == "positive"
        speech_fraction = max(0.0, 1.0 - mixture["intruder_duration_ratio"])
        intruder_type = mixture.get("intruder_type", "peds")

        embs = emb_dict.get(audio_path, {})
        ecapa_emb = embs.get("ecapa_emb")
        sparc_emb = embs.get("sparc_emb")

        profile = load_speaker_profile(profiles_path, target_pid)
        if profile is None or profile.profile_status != "ready":
            continue

        age_group = profile.age_group
        itype_rows = by_intruder_type.setdefault(intruder_type, {"ecapa": [], "sparc": []})

        if ecapa_emb is not None:
            ec = _l2_normalize(np.array(ecapa_emb, dtype=np.float64).ravel())
            centroid = _l2_normalize(
                np.array(profile.ecapa_embedding_centroid, dtype=np.float64)
            )
            row = (float(np.dot(ec, centroid)), is_positive, speech_fraction, age_group)
            ecapa_rows.append(row)
            itype_rows["ecapa"].append(row)

        if sparc_emb is not None:
            sp = _l2_normalize(np.array(sparc_emb, dtype=np.float64).ravel())
            centroid = _l2_normalize(
                np.array(profile.sparc_embedding_centroid, dtype=np.float64)
            )
            row = (float(np.dot(sp, centroid)), is_positive, speech_fraction, age_group)
            sparc_rows.append(row)
            itype_rows["sparc"].append(row)

    ecapa_curve = _build_single_emb_curve(ecapa_rows)
    sparc_curve = _build_single_emb_curve(sparc_rows)
    or_curve = _build_or_curve(ecapa_rows, sparc_rows)

    adult_ec = [(s, l, f, a) for s, l, f, a in ecapa_rows if a == "adult"]
    child_ec = [(s, l, f, a) for s, l, f, a in ecapa_rows if a == "child"]
    adult_sp = [(s, l, f, a) for s, l, f, a in sparc_rows if a == "adult"]
    child_sp = [(s, l, f, a) for s, l, f, a in sparc_rows if a == "child"]

    # Build per-intruder-type breakdown (always populated when multiple types or adult pool used)
    intruder_type_breakdown: dict = {}
    if len(by_intruder_type) >= 1:
        for itype, irows in by_intruder_type.items():
            ec = irows["ecapa"]
            sp = irows["sparc"]
            ec_curve = _build_single_emb_curve(ec)
            sp_curve = _build_single_emb_curve(sp)
            intruder_type_breakdown[itype] = {
                "num_positive": sum(1 for _, l, _, _ in ec if l),
                "num_negative": sum(1 for _, l, _, _ in ec if not l),
                "ecapa_operating_curve": ec_curve,
                "sparc_operating_curve": sp_curve,
                "or_operating_curve": _build_or_curve(ec, sp),
                "recommended_ecapa_threshold": _recommended_threshold(ec_curve, max_fnr_target),
                "recommended_sparc_threshold": _recommended_threshold(sp_curve, max_fnr_target),
            }

    bins_as_lists = [list(b) for b in speech_fraction_bins]

    return {
        "report_version": _REPORT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_bids_dir": str(profiles_path.parent),
        "num_participants": len({m["target_participant_id"] for m in mixture_list}),
        "num_recordings": len(mixture_list),
        "num_synthetic_mixtures": sum(1 for m in mixture_list if m["label"] == "positive"),
        "speech_fraction_bins": bins_as_lists,
        "ecapa_per_bin_stats": _per_bin_stats(ecapa_rows, speech_fraction_bins),
        "sparc_per_bin_stats": _per_bin_stats(sparc_rows, speech_fraction_bins),
        "or_per_bin_stats": _per_bin_stats(ecapa_rows, speech_fraction_bins),
        "ecapa_operating_curve": ecapa_curve,
        "sparc_operating_curve": sparc_curve,
        "or_operating_curve": or_curve,
        "recommended_ecapa_threshold": _recommended_threshold(ecapa_curve, max_fnr_target),
        "recommended_sparc_threshold": _recommended_threshold(sparc_curve, max_fnr_target),
        "recommended_low_confidence_threshold": _DEFAULT_LOW_CONFIDENCE_SPEECH_FRACTION,
        "recommended_min_enrollment_duration_s": _DEFAULT_MIN_ACTIVE_SPEECH_S,
        "knee_point_fraction": _compute_knee_point(ecapa_rows, speech_fraction_bins, knee_drop_pp),
        "adult_subgroup_stats": {
            "ecapa_operating_curve": _build_single_emb_curve(adult_ec),
            "sparc_operating_curve": _build_single_emb_curve(adult_sp),
        },
        "child_subgroup_stats": {
            "ecapa_operating_curve": _build_single_emb_curve(child_ec),
            "sparc_operating_curve": _build_single_emb_curve(child_sp),
        },
        "intruder_type_breakdown": intruder_type_breakdown,
    }


# ---------------------------------------------------------------------------
# Phase 5 Addendum: Real-data validation (peds exclusion list)
# ---------------------------------------------------------------------------

_REAL_DATA_CAVEAT = (
    "Exclusion-list membership does not confirm unconsented-speaker presence. "
    "Files were removed from the release for various reasons; not all removals "
    "are attributable to a second speaker. Recall on this set is an upper-bound "
    "estimate of signal sensitivity on uncertain positives only. "
    "No precision or false-positive-rate metrics are reported."
)


def _load_evans_predictions(
    predictions_csv: str,
    train_annotations_csv: Optional[str] = None,
) -> dict:
    """Load per-file Evans model predictions keyed by file stem.

    Returns a dict mapping stem → {evans_y_pred, evans_confidence,
    evans_uncertainty, evans_split}.  ``evans_split`` is ``"test"`` for
    stems absent from *train_annotations_csv*, ``"train"`` otherwise.
    """
    import csv as _csv

    train_stems: set = set()
    if train_annotations_csv:
        with open(train_annotations_csv) as fh:
            for row in _csv.DictReader(fh):
                train_stems.add(Path(row["file_path"]).stem)

    records: dict = {}
    with open(predictions_csv) as fh:
        for row in _csv.DictReader(fh):
            stem = Path(row["file_path"]).stem
            split = "train" if stem in train_stems else "test"
            records[stem] = {
                "evans_y_pred": int(row["y_pred"]),
                "evans_confidence": float(row["confidence"]),
                "evans_uncertainty": float(row["uncertainty"]),
                "evans_split": split,
            }
    return records


def _parse_stem(stem: str) -> tuple:
    """Return ``(participant_id, session_id, task_name)`` from a BIDS file stem."""
    pid = ses = task = ""
    for part in stem.split("_"):
        if part.startswith("sub-"):
            pid = part[4:]
        elif part.startswith("ses-"):
            ses = part[4:]
        elif part.startswith("task-"):
            task = part[5:]
    return pid, ses, task


def compute_real_data_validation(
    exclusion_list_path: str,
    bids_dir: str,
    profiles_dir: str,
    ecapa_threshold: float,
    sparc_threshold: float,
    evans_predictions_csv: Optional[str] = None,
    evans_train_annotations_csv: Optional[str] = None,
) -> dict:
    """Score uncertain positives from the peds release exclusion list.

    For each `_features.pt` file whose stem appears in the exclusion list,
    loads the participant's speaker profile, computes ECAPA-TDNN and SPARC
    cosine similarities, applies thresholds, and reports per-signal recall.

    Ground-truth interpretation: the exclusion list is treated as a pool of
    *uncertain positives* — files removed from the release for any reason.
    Recall on this set is an upper-bound sensitivity estimate only; precision
    and FPR cannot be computed because true negatives are not confirmed.

    Args:
        exclusion_list_path: Path to a JSON file containing a list of file
            stems (without extension) to treat as uncertain positives.
        bids_dir: Root BIDS directory to scan for ``*_features.pt`` files.
        profiles_dir: Directory of pre-built speaker profiles.
        ecapa_threshold: ECAPA-TDNN cosine threshold below which a recording
            is flagged (from the operating-curves recommended threshold).
        sparc_threshold: SPARC cosine threshold below which a recording is
            flagged.
        evans_predictions_csv: Optional path to Evans model predictions CSV
            (columns: file_path, y_pred, confidence, uncertainty).
        evans_train_annotations_csv: Optional train-split annotation CSV
            used to exclude train-contaminated Evans predictions.

    Returns:
        Dict with keys: ``num_uncertain_positives``,
        ``num_with_diarization_multispeaker``, ``diarization_fraction``,
        ``per_signal_recall`` (sub-keys: ecapa, sparc, or_combined,
        diarization, evans — ``null`` when no Evans CSV supplied), ``caveat``.
    """
    from b2aiprep.prepare.speaker_profiles import load_speaker_profile

    with open(exclusion_list_path) as fh:
        exclusion_stems: set = set(json.load(fh))

    profiles_path = Path(profiles_dir)
    bids_path = Path(bids_dir)

    evans_records: dict = {}
    if evans_predictions_csv:
        try:
            evans_records = _load_evans_predictions(
                evans_predictions_csv, evans_train_annotations_csv
            )
        except Exception as exc:
            _logger.warning("Could not load Evans predictions: %s", exc)

    feature_files = sorted(bids_path.rglob("*_features.pt"))

    flags: dict = {
        "ecapa": [], "sparc": [], "or_combined": [], "diarization": [], "evans": []
    }
    num_diarization_multi = 0
    processed = 0

    for pt_path in feature_files:
        stem = pt_path.name.replace("_features.pt", "")
        if stem not in exclusion_stems:
            continue

        pid, _ses, _task = _parse_stem(stem)
        profile = load_speaker_profile(profiles_path, pid)
        if profile is None or profile.profile_status != "ready":
            continue

        try:
            feat = torch.load(str(pt_path), weights_only=False, map_location="cpu")
        except Exception as exc:
            _logger.debug("Cannot load %s: %s", pt_path.name, exc)
            continue

        processed += 1

        # --- ECAPA cosine ---
        ecapa_raw = feat.get("speaker_embedding")
        ecapa_flag = False
        if ecapa_raw is not None:
            try:
                ec = _l2_normalize(np.array(ecapa_raw, dtype=np.float64).ravel())
                centroid = _l2_normalize(
                    np.array(profile.ecapa_embedding_centroid, dtype=np.float64)
                )
                ecapa_cos = float(np.dot(ec, centroid))
                ecapa_flag = ecapa_cos < ecapa_threshold
            except Exception as exc:
                _logger.debug("ECAPA scoring failed for %s: %s", stem, exc)
        flags["ecapa"].append(int(ecapa_flag))

        # --- SPARC cosine ---
        sparc_raw = feat.get("sparc", {})
        if hasattr(sparc_raw, "get"):
            sparc_raw = sparc_raw.get("spk_emb")
        else:
            sparc_raw = None
        sparc_flag = False
        if sparc_raw is not None:
            try:
                sp = _l2_normalize(np.array(sparc_raw, dtype=np.float64).ravel())
                centroid = _l2_normalize(
                    np.array(profile.sparc_embedding_centroid, dtype=np.float64)
                )
                sparc_cos = float(np.dot(sp, centroid))
                sparc_flag = sparc_cos < sparc_threshold
            except Exception as exc:
                _logger.debug("SPARC scoring failed for %s: %s", stem, exc)
        flags["sparc"].append(int(sparc_flag))

        flags["or_combined"].append(int(ecapa_flag or sparc_flag))

        # --- Diarization ---
        diarization_raw = feat.get("diarization", [])
        speaker_durations: dict = {}
        for seg in diarization_raw:
            if hasattr(seg, "speaker"):
                spk = str(seg.speaker)
                start, end = float(seg.start), float(seg.end)
            else:
                spk = str(seg.get("speaker", "speaker_0"))
                start, end = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
            speaker_durations[spk] = speaker_durations.get(spk, 0.0) + max(0.0, end - start)
        num_spk = len(speaker_durations)
        diar_flag = num_spk > 1
        if diar_flag:
            num_diarization_multi += 1
        flags["diarization"].append(int(diar_flag))

        # --- Evans ---
        ev = evans_records.get(stem)
        if ev is not None:
            flags["evans"].append(int(ev["evans_y_pred"]))
        else:
            flags["evans"].append(None)

    n = processed
    if n == 0:
        _logger.warning(
            "compute_real_data_validation: no exclusion-list stems matched feature files"
            " with ready profiles in %s", bids_dir
        )
        return {
            "num_uncertain_positives": 0,
            "num_with_diarization_multispeaker": 0,
            "diarization_fraction": None,
            "per_signal_recall": {
                "ecapa": None, "sparc": None, "or_combined": None,
                "diarization": None, "evans": None,
            },
            "caveat": _REAL_DATA_CAVEAT,
        }

    def _recall(flag_list: list) -> Optional[float]:
        non_null = [v for v in flag_list if v is not None]
        if not non_null:
            return None
        return round(sum(non_null) / len(non_null), 6)

    evans_recall = _recall(flags["evans"]) if any(v is not None for v in flags["evans"]) else None

    return {
        "num_uncertain_positives": n,
        "num_with_diarization_multispeaker": num_diarization_multi,
        "diarization_fraction": round(num_diarization_multi / n, 6),
        "per_signal_recall": {
            "ecapa": _recall(flags["ecapa"]),
            "sparc": _recall(flags["sparc"]),
            "or_combined": _recall(flags["or_combined"]),
            "diarization": _recall(flags["diarization"]),
            "evans": evans_recall,
        },
        "caveat": _REAL_DATA_CAVEAT,
    }


# ---------------------------------------------------------------------------
# T017: Report writer
# ---------------------------------------------------------------------------


def write_reliability_report(
    report_dict: dict,
    output_dir: str,
    output_format: str = "both",
) -> None:
    """Write the reliability report to JSON and/or Markdown.

    Args:
        report_dict: :class:`EmbeddingReliabilityReport` dict.
        output_dir: Directory where report files are written.
        output_format: ``"json"``, ``"markdown"``, or ``"both"``.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if output_format in ("json", "both"):
        json_file = out_path / "embedding_reliability_report.json"
        with open(json_file, "w") as f:
            json.dump(report_dict, f, indent=2, default=str)
        _logger.info("Wrote JSON report: %s", json_file)

    if output_format in ("markdown", "both"):
        md_file = out_path / "embedding_reliability_report.md"
        with open(md_file, "w") as f:
            f.write(_render_markdown_report(report_dict))
        _logger.info("Wrote Markdown report: %s", md_file)


def _render_markdown_report(report: dict) -> str:
    """Render an EmbeddingReliabilityReport dict as a Markdown document."""
    ecapa_thresh = report.get("recommended_ecapa_threshold")
    sparc_thresh = report.get("recommended_sparc_threshold")
    ecapa_curve = report.get("ecapa_operating_curve", [])
    sparc_curve = report.get("sparc_operating_curve", [])

    def _queue_at(curve, thresh):
        if thresh is None:
            return "N/A"
        for pt in curve:
            if abs(pt["threshold"] - thresh) < 1e-6:
                return f"{pt['review_fraction']:.1%}"
        return "N/A"

    def _fnr_at(curve, thresh):
        if not curve or thresh is None:
            return "N/A"
        for pt in curve:
            if abs(pt["threshold"] - thresh) < 1e-6:
                return f"{pt['fnr']:.1%}"
        return "N/A"

    adult = report.get("adult_subgroup_stats", {})
    child = report.get("child_subgroup_stats", {})
    adult_ec_curve = adult.get("ecapa_operating_curve", [])
    child_ec_curve = child.get("ecapa_operating_curve", [])
    knee = report.get("knee_point_fraction", 0.0)

    lines = [
        "# Embedding Reliability Report",
        "",
        f"**Generated**: {report.get('generated_at', 'unknown')}  ",
        f"**Version**: {report.get('report_version', 'unknown')}  ",
        f"**Dataset**: {report.get('dataset_bids_dir', 'unknown')}  ",
        f"**Participants**: {report.get('num_participants', 0)}  ",
        f"**Recordings**: {report.get('num_recordings', 0)} "
        f"({report.get('num_synthetic_mixtures', 0)} synthetic mixtures)",
        "",
        "> **Adult cohort**: No real-data validation section. "
        "Threshold calibration for adults is synthetic-only.",
        "",
        "---",
        "",
        "## Recommended Thresholds (≤ 5% FNR)",
        "",
        "| Embedding | Threshold | Review-queue fraction |",
        "|-----------|-----------|----------------------|",
        f"| ECAPA-TDNN | `{ecapa_thresh}` | {_queue_at(ecapa_curve, ecapa_thresh)} |",
        f"| SPARC      | `{sparc_thresh}` | {_queue_at(sparc_curve, sparc_thresh)} |",
        "",
        f"**Knee-point speech fraction**: `{knee:.2f}` "
        "(accuracy drops > 15 pp below this threshold)",
        "",
        "---",
        "",
        "## Per-Speech-Fraction-Bin Accuracy (ECAPA-TDNN)",
        "",
        "| Bin | N | Mean cosine (solo) | Mean cosine (intruder) |",
        "|-----|---|-------------------|----------------------|",
    ]

    for stat in report.get("ecapa_per_bin_stats", []):
        lo, hi = stat["bin"]
        n = stat["n"]
        same = f"{stat['mean_cosine_same']:.4f}" if stat["mean_cosine_same"] is not None else "—"
        diff = f"{stat['mean_cosine_diff']:.4f}" if stat["mean_cosine_diff"] is not None else "—"
        lines.append(f"| [{lo:.2f}, {hi:.2f}) | {n} | {same} | {diff} |")

    lines += [
        "",
        "---",
        "",
        "## Adult vs. Child Subgroup Breakdown",
        "",
        f"| Group | ECAPA FNR @ {ecapa_thresh} |",
        "|-------|------------|",
        f"| Adult | {_fnr_at(adult_ec_curve, ecapa_thresh)} |",
        f"| Child | {_fnr_at(child_ec_curve, ecapa_thresh)} |",
        "",
    ]

    intruder_breakdown = report.get("intruder_type_breakdown", {})
    if intruder_breakdown:
        lines += [
            "---",
            "",
            "## Intruder-Type Breakdown",
            "",
            "Performance by the type of intruder used in synthetic mixtures.",
            "",
            "| Intruder type | N positive | Rec. ECAPA threshold | ECAPA FNR | Rec. SPARC threshold | SPARC FNR |",
            "|---------------|-----------|---------------------|-----------|---------------------|-----------|",
        ]
        for itype, stats in sorted(intruder_breakdown.items()):
            ec_c = stats.get("ecapa_operating_curve", [])
            sp_c = stats.get("sparc_operating_curve", [])
            rec_ec = stats.get("recommended_ecapa_threshold")
            rec_sp = stats.get("recommended_sparc_threshold")
            n_pos = stats.get("num_positive", 0)
            label = "adult→peds *(primary)*" if itype == "adult" else itype
            lines.append(
                f"| {label} | {n_pos} "
                f"| `{rec_ec}` | {_fnr_at(ec_c, rec_ec)} "
                f"| `{rec_sp}` | {_fnr_at(sp_c, rec_sp)} |"
            )
        lines.append("")

    rdv = report.get("real_data_validation")
    if rdv is not None:
        n_up = rdv.get("num_uncertain_positives", 0)
        n_dm = rdv.get("num_with_diarization_multispeaker", 0)
        diar_frac = rdv.get("diarization_fraction")
        psr = rdv.get("per_signal_recall", {})
        caveat = rdv.get("caveat", "")

        def _pct(v):
            return f"{v:.1%}" if v is not None else "—"

        lines += [
            "---",
            "",
            "## Real-Data Validation (Peds Only)",
            "",
            f"**Uncertain positives evaluated**: {n_up}  ",
            f"**With diarization multi-speaker signal**: {n_dm} "
            f"({_pct(diar_frac)} of uncertain positives)  ",
            "",
            "| Signal | Recall on uncertain positives |",
            "|--------|------------------------------|",
            f"| ECAPA-TDNN cosine | {_pct(psr.get('ecapa'))} |",
            f"| SPARC cosine | {_pct(psr.get('sparc'))} |",
            f"| OR-combined | {_pct(psr.get('or_combined'))} |",
            f"| Diarization (>1 speaker) | {_pct(psr.get('diarization'))} |",
            f"| Evans model | {_pct(psr.get('evans'))} |",
            "",
            f"> **Caveat**: {caveat}",
            "",
        ]

    return "\n".join(lines)
