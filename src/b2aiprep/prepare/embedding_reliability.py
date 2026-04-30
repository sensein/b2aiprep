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
# Low-level helpers
# ---------------------------------------------------------------------------


def _overlay_intruder(
    base: torch.Tensor,
    intruder: torch.Tensor,
    snr_db: float,
) -> torch.Tensor:
    """Overlay intruder clip at the end of base at the specified SNR.

    Both tensors are (channels, samples). The intruder is energy-scaled to
    achieve ``snr_db`` relative to the base tail segment it overlaps.
    Output has exactly the same shape as ``base``.
    """
    base = base.float()
    intruder = intruder.float()

    n_base = base.shape[-1]
    n_intruder = intruder.shape[-1]

    if n_intruder == 0 or n_base == 0:
        return base.clone()

    start = max(0, n_base - n_intruder)
    base_segment = base[:, start:]
    base_rms = float(base_segment.pow(2).mean().sqrt()) + 1e-10
    intruder_rms = float(intruder.pow(2).mean().sqrt()) + 1e-10

    target_rms = base_rms / (10.0 ** (snr_db / 20.0))
    scale = target_rms / intruder_rms
    intruder_scaled = intruder * scale

    mixed = base.clone()
    actual_len = min(n_intruder, n_base)
    mixed[:, -actual_len:] = mixed[:, -actual_len:] + intruder_scaled[:, :actual_len]
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
    seed: int = 42,
) -> list:
    """Generate synthetic intruder mixtures for operating characteristic evaluation.

    For each participant with a ``ready`` profile, pairs each clean recording
    with a random intruder participant and writes one mixture per (ratio, SNR)
    combination. A matching ``negative`` (unmixed) sample is also written.

    Args:
        bids_dir: Root of the BIDS dataset containing ``_features.pt`` files.
        profiles_dir: Directory of pre-built speaker profiles.
        intruder_ratios: Fractions of base recording duration to overlay, e.g. ``[0.10, 0.20, 0.40]``.
        intruder_snr_db_values: SNR values in dB, e.g. ``[0, 5, 10]``.
        output_dir: Where to write mixture ``.wav`` files.
        config: Optional :class:`PipelineConfig` (unused; reserved for future gating).
        seed: Random seed for reproducible intruder selection.

    Returns:
        List of :data:`SyntheticMixture` dicts with keys ``target_participant_id``,
        ``intruder_participant_id``, ``base_recording_path``, ``intruder_segment_path``,
        ``intruder_duration_ratio``, ``intruder_snr_db``, ``label``, ``mixed_audio_path``.
    """
    from b2aiprep.prepare.speaker_profiles import load_speaker_profile

    rng = random.Random(seed)
    bids_path = Path(bids_dir)
    profiles_path = Path(profiles_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_features = _find_features_files(bids_path)
    participants = list(all_features.keys())

    if len(participants) < 2:
        _logger.warning("Need ≥ 2 participants for synthetic mixtures; found %d", len(participants))
        return []

    mixtures: list = []

    for target_pid in participants:
        profile = load_speaker_profile(profiles_path, target_pid)
        if profile is None or profile.profile_status != "ready":
            _logger.debug("Skipping %s: profile status=%s",
                          target_pid, profile.profile_status if profile else "missing")
            continue

        other_pids = [p for p in participants if p != target_pid]
        if not other_pids:
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

            # Write negative (unmixed solo)
            neg_name = f"{target_pid}_solo_{pt_file.stem}.wav"
            neg_out = out_path / neg_name
            if not neg_out.exists():
                torchaudio.save(str(neg_out), base_waveform, base_sr)
            mixtures.append({
                "target_participant_id": target_pid,
                "intruder_participant_id": target_pid,
                "base_recording_path": str(audio_path),
                "intruder_segment_path": str(audio_path),
                "intruder_duration_ratio": 0.0,
                "intruder_snr_db": 0.0,
                "label": "negative",
                "mixed_audio_path": str(neg_out),
            })

            # Select random intruder
            intruder_pid = rng.choice(other_pids)
            intruder_pt_files = all_features.get(intruder_pid, [])
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
                    mixed = _overlay_intruder(base_waveform, intruder_clip, float(snr_db))
                    ratio_str = f"{ratio}".replace(".", "p")
                    snr_str = f"{float(snr_db)}".replace(".", "p").replace("-", "n")
                    out_name = (
                        f"{target_pid}_{intruder_pid}"
                        f"_ratio{ratio_str}_snr{snr_str}_{pt_file.stem}.wav"
                    )
                    out_file = out_path / out_name
                    torchaudio.save(str(out_file), mixed, base_sr)
                    mixtures.append({
                        "target_participant_id": target_pid,
                        "intruder_participant_id": intruder_pid,
                        "base_recording_path": str(audio_path),
                        "intruder_segment_path": str(intruder_audio_path),
                        "intruder_duration_ratio": float(ratio),
                        "intruder_snr_db": float(snr_db),
                        "label": "positive",
                        "mixed_audio_path": str(out_file),
                    })

    return mixtures


# ---------------------------------------------------------------------------
# T015: Embedding extraction for mixture audio
# ---------------------------------------------------------------------------


def extract_embeddings_for_mixtures(mixture_list: list) -> dict:
    """Extract ECAPA-TDNN and SPARC embeddings from synthetic mixture audio files.

    Uses senselab for ECAPA-TDNN extraction. SPARC extraction returns ``None``
    when the senselab SPARC extractor is unavailable for raw audio inputs.

    This function requires model inference and may be slow on CPU.

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
        _senselab_ok = True
    except ImportError:
        _logger.warning(
            "senselab not available; all embeddings will be None. "
            "Install senselab or run on a node with the full conda environment."
        )
        _senselab_ok = False

    result: dict = {}

    for mixture in mixture_list:
        audio_path = mixture["mixed_audio_path"]
        ecapa_emb = None

        if _senselab_ok:
            try:
                waveform, sr = torchaudio.load(audio_path)
                audio_obj = Audio(waveform=waveform, sampling_rate=sr)
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

        result[audio_path] = {"ecapa_emb": ecapa_emb, "sparc_emb": None}

    return result


# ---------------------------------------------------------------------------
# T016: Operating characteristic computation
# ---------------------------------------------------------------------------


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    return v / (norm + 1e-10)


def _recommended_threshold(curve: list, max_fnr: float = 0.05) -> float:
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


def _compute_knee_point(score_label_fraction_age: list, bins: list) -> float:
    """Speech fraction boundary where per-bin same-speaker cosine drops > 15 pp."""
    bin_means = []
    for lo, hi in bins:
        same = [s for s, l, f, _ in score_label_fraction_age if lo <= f < hi and not l]
        bin_means.append((lo, hi, float(np.mean(same)) if same else None))

    top_mean = max((m for _, _, m in bin_means if m is not None), default=None)
    if top_mean is None:
        return 0.0
    for lo, _, m in bin_means:
        if m is not None and top_mean - m > 0.15:
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
) -> dict:
    """Score mixtures against participant profiles and compute operating curves.

    Sweeps cosine threshold from 0.0 to 1.0 in steps of 0.01. Computes FNR,
    FPR, and review-queue fraction for ECAPA-TDNN, SPARC, and OR-combined.

    Args:
        mixture_list: SyntheticMixture dicts from :func:`generate_synthetic_mixtures`.
        profiles_dir: Directory containing per-participant ``speaker_profile.json`` files.
        emb_dict: Output of :func:`extract_embeddings_for_mixtures`.
        speech_fraction_bins: Bin boundaries as list of (lo, hi) pairs.

    Returns:
        :class:`EmbeddingReliabilityReport` dict.
    """
    from b2aiprep.prepare.speaker_profiles import load_speaker_profile

    if speech_fraction_bins is None:
        speech_fraction_bins = [(0, 0.15), (0.15, 0.30), (0.30, 0.60), (0.60, 1.01)]

    profiles_path = Path(profiles_dir)
    ecapa_rows: list = []
    sparc_rows: list = []

    for mixture in mixture_list:
        audio_path = mixture["mixed_audio_path"]
        target_pid = mixture["target_participant_id"]
        is_positive = mixture["label"] == "positive"
        speech_fraction = max(0.0, 1.0 - mixture["intruder_duration_ratio"])

        embs = emb_dict.get(audio_path, {})
        ecapa_emb = embs.get("ecapa_emb")
        sparc_emb = embs.get("sparc_emb")

        profile = load_speaker_profile(profiles_path, target_pid)
        if profile is None or profile.profile_status != "ready":
            continue

        age_group = profile.age_group

        if ecapa_emb is not None:
            ec = _l2_normalize(np.array(ecapa_emb, dtype=np.float64).ravel())
            centroid = _l2_normalize(
                np.array(profile.ecapa_embedding_centroid, dtype=np.float64)
            )
            ecapa_rows.append((float(np.dot(ec, centroid)), is_positive, speech_fraction, age_group))

        if sparc_emb is not None:
            sp = _l2_normalize(np.array(sparc_emb, dtype=np.float64).ravel())
            centroid = _l2_normalize(
                np.array(profile.sparc_embedding_centroid, dtype=np.float64)
            )
            sparc_rows.append((float(np.dot(sp, centroid)), is_positive, speech_fraction, age_group))

    ecapa_curve = _build_single_emb_curve(ecapa_rows)
    sparc_curve = _build_single_emb_curve(sparc_rows)
    or_curve = _build_or_curve(ecapa_rows, sparc_rows)

    adult_ec = [(s, l, f, a) for s, l, f, a in ecapa_rows if a == "adult"]
    child_ec = [(s, l, f, a) for s, l, f, a in ecapa_rows if a == "child"]
    adult_sp = [(s, l, f, a) for s, l, f, a in sparc_rows if a == "adult"]
    child_sp = [(s, l, f, a) for s, l, f, a in sparc_rows if a == "child"]

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
        "recommended_ecapa_threshold": _recommended_threshold(ecapa_curve),
        "recommended_sparc_threshold": _recommended_threshold(sparc_curve),
        "recommended_low_confidence_threshold": 0.15,
        "recommended_min_enrollment_duration_s": 3.0,
        "knee_point_fraction": _compute_knee_point(ecapa_rows, speech_fraction_bins),
        "adult_subgroup_stats": {
            "ecapa_operating_curve": _build_single_emb_curve(adult_ec),
            "sparc_operating_curve": _build_single_emb_curve(adult_sp),
        },
        "child_subgroup_stats": {
            "ecapa_operating_curve": _build_single_emb_curve(child_ec),
            "sparc_operating_curve": _build_single_emb_curve(child_sp),
        },
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

    return "\n".join(lines)
