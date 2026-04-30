"""Speaker profile construction and loading for unconsented-speaker detection.

Builds per-participant dual-centroid speaker profiles (ECAPA-TDNN 192-dim +
SPARC spk_emb 64-dim) from pre-computed ``_features.pt`` files and writes
them as reusable JSON artifacts consumed by the qa-run pipeline.
"""

import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from b2aiprep.prepare.qa_models import PipelineConfig

_logger = logging.getLogger(__name__)

# Model identifiers embedded in every profile for reproducibility
_ECAPA_MODEL_ID = "speechbrain/spkrec-ecapa-voxceleb"
_SPARC_MODEL_ID = "senselab/sparc-multi"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ExcludedRecording:
    """One recording excluded during enrollment, with the reason."""

    task_name: str
    session_id: str
    reason: str


@dataclass
class SpeakerProfile:
    """Per-participant dual speaker profile (ECAPA-TDNN + SPARC centroids).

    Written to ``{profiles_dir}/sub-{participant_id}/speaker_profile.json``.
    Fields follow data-model.md exactly.
    """

    participant_id: str
    ecapa_model_id: str
    sparc_model_id: str
    num_recordings_used: int
    num_recordings_excluded: int
    total_active_speech_s: float
    ecapa_profile_quality_score: float
    sparc_profile_quality_score: float
    profile_status: str  # "ready" | "insufficient_data" | "contaminated"
    age_group: str  # "adult" | "child" | "unknown"
    included_recordings: list[str] = field(default_factory=list)
    excluded_recordings: list[dict] = field(default_factory=list)
    ecapa_embedding_centroid: Optional[list[float]] = None
    sparc_embedding_centroid: Optional[list[float]] = None
    created_at: str = ""
    pipeline_config_hash: str = ""

    def to_json(self) -> dict:
        """Serialise to a JSON-safe dict."""
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "SpeakerProfile":
        """Deserialise from a JSON dict (e.g. loaded from speaker_profile.json)."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_speaker_profile(
    profiles_dir: str | Path,
    participant_id: str,
) -> Optional[SpeakerProfile]:
    """Load a pre-built speaker profile for one participant.

    Returns ``None`` when the profile file is absent (triggers needs_review
    in downstream verification).

    Args:
        profiles_dir: Root directory written by ``build-speaker-profiles``.
        participant_id: BIDS participant ID without the ``sub-`` prefix.

    Returns:
        :class:`SpeakerProfile` if the file exists, else ``None``.
    """
    profile_path = Path(profiles_dir) / f"sub-{participant_id}" / "speaker_profile.json"
    if not profile_path.exists():
        _logger.debug("No speaker profile for participant %s at %s", participant_id, profile_path)
        return None
    try:
        with open(profile_path, encoding="utf-8") as fh:
            data = json.load(fh)
        return SpeakerProfile.from_json(data)
    except Exception as exc:
        _logger.warning("Failed to load speaker profile %s: %s", profile_path, exc)
        return None


# ---------------------------------------------------------------------------
# Active speech helpers
# ---------------------------------------------------------------------------


def _compute_active_speech_s(diarization: Any) -> float:
    """Return total active speech seconds from a diarization result.

    Handles both senselab ``ScriptLine`` objects (with ``.start`` / ``.end``)
    and plain ``{"start": float, "end": float}`` dicts.
    """
    if not diarization:
        return 0.0
    total = 0.0
    for seg in diarization:
        if hasattr(seg, "start") and hasattr(seg, "end"):
            total += max(0.0, float(seg.end) - float(seg.start))
        elif isinstance(seg, dict):
            total += max(0.0, float(seg.get("end", 0)) - float(seg.get("start", 0)))
    return total


def _l2_normalize(v: np.ndarray) -> np.ndarray:
    """Return L2-normalised copy; returns zero vector if norm ≈ 0."""
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return v.copy()
    return v / norm


# ---------------------------------------------------------------------------
# Outlier rejection
# ---------------------------------------------------------------------------


def _reject_outliers(
    embeddings: list[np.ndarray],
    weights: list[float],
    std_multiplier: float,
) -> tuple[list[np.ndarray], list[float]]:
    """Remove embeddings whose mean pairwise cosine similarity is low.

    An embedding is rejected when its mean pairwise cosine similarity with all
    others falls below ``overall_mean - std_multiplier * overall_std``.

    Args:
        embeddings: L2-normalised embedding vectors.
        weights: Per-embedding quality weights (same length as embeddings).
        std_multiplier: Multiplier on the standard deviation for the rejection
            threshold.

    Returns:
        Pair of ``(kept_embeddings, kept_weights)``.
    """
    n = len(embeddings)
    if n <= 2:
        return embeddings, weights

    mat = np.stack(embeddings)  # (n, d)
    sim_matrix = mat @ mat.T  # (n, n) pairwise cosines (already L2-normed)

    # Mean pairwise similarity per embedding (excluding self-similarity)
    mean_sims = (sim_matrix.sum(axis=1) - 1.0) / max(n - 1, 1)
    overall_mean = float(mean_sims.mean())
    overall_std = float(mean_sims.std())
    threshold = overall_mean - std_multiplier * overall_std

    kept_embs = []
    kept_weights = []
    for i, (emb, w, ms) in enumerate(zip(embeddings, weights, mean_sims)):
        if float(ms) >= threshold:
            kept_embs.append(emb)
            kept_weights.append(w)
        else:
            _logger.debug(
                "Outlier rejected: embedding %d mean_sim=%.4f threshold=%.4f", i, ms, threshold
            )
    if not kept_embs:
        # Fallback: keep the one closest to the overall mean
        best = int(np.argmax(mean_sims))
        kept_embs = [embeddings[best]]
        kept_weights = [weights[best]]
    return kept_embs, kept_weights


# ---------------------------------------------------------------------------
# Centroid + quality
# ---------------------------------------------------------------------------


def _weighted_centroid(embeddings: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Compute weighted L2-normalised centroid."""
    w = np.array(weights, dtype=np.float64)
    w /= w.sum() + 1e-10
    centroid = sum(wi * e for wi, e in zip(w, embeddings))
    return _l2_normalize(centroid)


def _profile_quality(embeddings: list[np.ndarray]) -> float:
    """Mean pairwise cosine similarity of a set of L2-normalised embeddings."""
    n = len(embeddings)
    if n <= 1:
        return 0.0
    mat = np.stack(embeddings)
    sim_matrix = mat @ mat.T
    # Upper triangle (excluding diagonal)
    upper = sim_matrix[np.triu_indices(n, k=1)]
    return float(upper.mean()) if len(upper) > 0 else 0.0


# ---------------------------------------------------------------------------
# Profile builder
# ---------------------------------------------------------------------------


def build_speaker_profiles(
    bids_dir: str | Path,
    profiles_dir: str | Path,
    config: Optional[PipelineConfig] = None,
    part: int = 1,
    num_parts: int = 1,
    age_col: str = "age",
) -> dict[str, SpeakerProfile]:
    """Build per-participant dual-centroid speaker profiles.

    Scans ``bids_dir`` for ``*_features.pt`` files, groups them by participant,
    applies enrollment gating, builds quality-weighted outlier-rejected centroids
    for ECAPA-TDNN and SPARC embeddings, and writes one
    ``sub-{pid}/speaker_profile.json`` per participant.

    Args:
        bids_dir: Root of the BIDS dataset (contains ``sub-*/`` directories).
        profiles_dir: Output directory. Created if absent.
        config: :class:`PipelineConfig` with ``speaker_profile`` sub-dict.
            Uses built-in defaults when ``None``.
        part: 1-based shard index for SLURM array jobs.
        num_parts: Total number of shards.
        age_col: Column name in ``participants.tsv`` for participant age.

    Returns:
        Dict mapping participant_id → :class:`SpeakerProfile`.
    """
    import torch

    if config is None:
        config = PipelineConfig()

    sp_cfg = config.speaker_profile
    min_recordings = int(sp_cfg.get("min_profile_recordings", 3))
    min_active_s = float(sp_cfg.get("min_active_speech_s", 3.0))
    low_conf_frac = float(sp_cfg.get("low_confidence_speech_fraction", 0.15))
    std_mult = float(sp_cfg.get("outlier_rejection_std_multiplier", 1.5))
    contam_thresh = float(sp_cfg.get("contamination_quality_threshold", 0.30))
    excluded_prefixes: list[str] = sp_cfg.get(
        "excluded_task_prefixes",
        [
            "Diadochokinesis", "Prolonged-vowel", "Maximum-phonation-time",
            "Respiration-and-cough", "Glides", "Loudness",
            "long-sounds", "silly-sounds", "repeat-words",
        ],
    )

    config_hash = _hash_config(sp_cfg)
    bids_path = Path(bids_dir)
    out_path = Path(profiles_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # --- Load participant age map (optional) ---
    age_map: dict[str, float | None] = {}
    participants_tsv = bids_path / "participants.tsv"
    if participants_tsv.exists():
        try:
            import csv
            with open(participants_tsv, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                for row in reader:
                    pid = row.get("participant_id", "").lstrip("sub-")
                    age_raw = row.get(age_col, "")
                    try:
                        age_map[pid] = float(age_raw)
                    except (ValueError, TypeError):
                        age_map[pid] = None
        except Exception as exc:
            _logger.warning("Could not read participants.tsv: %s", exc)

    # --- Gather all _features.pt files grouped by participant ---
    all_pt_files = sorted(bids_path.rglob("*_features.pt"))
    if not all_pt_files:
        _logger.error("No _features.pt files found under %s", bids_path)
        return {}

    participant_files: dict[str, list[Path]] = {}
    for pt_file in all_pt_files:
        # Extract participant ID from BIDS stem
        stem = pt_file.stem
        pid = ""
        for part_token in stem.split("_"):
            if part_token.startswith("sub-"):
                pid = part_token[4:]
                break
        if not pid:
            continue
        participant_files.setdefault(pid, []).append(pt_file)

    # --- Shard participants ---
    all_pids = sorted(participant_files.keys())
    if num_parts > 1:
        shard_pids = [p for i, p in enumerate(all_pids) if (i % num_parts) == (part - 1)]
    else:
        shard_pids = all_pids

    _logger.info(
        "build-speaker-profiles: %d participants (shard %d/%d)",
        len(shard_pids), part, num_parts,
    )

    profiles: dict[str, SpeakerProfile] = {}

    for pid in shard_pids:
        pt_files = participant_files[pid]
        _logger.debug("Processing participant %s (%d files)", pid, len(pt_files))

        ecapa_embs: list[np.ndarray] = []
        sparc_embs: list[np.ndarray] = []
        weights: list[float] = []
        included: list[str] = []
        excluded: list[dict] = []
        total_speech_s = 0.0

        for pt_file in pt_files:
            try:
                features = torch.load(str(pt_file), weights_only=False, map_location="cpu")
            except Exception as exc:
                _logger.warning("Failed to load %s: %s", pt_file, exc)
                continue

            # Extract BIDS task name from filename
            stem = pt_file.stem
            task_name = ""
            for token in stem.split("_"):
                if token.startswith("task-"):
                    task_name = token[5:]
                    break
            session_id = ""
            for token in stem.split("_"):
                if token.startswith("ses-"):
                    session_id = token[4:]
                    break

            # --- Gate 1: task prefix exclusion ---
            task_lower = task_name.lower()
            excluded_by_prefix = any(
                task_lower.startswith(prefix.lower()) for prefix in excluded_prefixes
            )
            if excluded_by_prefix:
                excluded.append({
                    "task_name": task_name,
                    "session_id": session_id,
                    "reason": "task_prefix_excluded",
                })
                _logger.debug("Excluded %s (task_prefix_excluded)", task_name)
                continue

            # --- Load required fields ---
            speaker_embedding = features.get("speaker_embedding")
            sparc_data = features.get("sparc")
            diarization = features.get("diarization")
            duration = float(features.get("duration", 0.0))

            if speaker_embedding is None or sparc_data is None:
                excluded.append({
                    "task_name": task_name,
                    "session_id": session_id,
                    "reason": "missing_embeddings",
                })
                continue

            sparc_spk = sparc_data.get("spk_emb") if isinstance(sparc_data, dict) else None
            if sparc_spk is None:
                excluded.append({
                    "task_name": task_name,
                    "session_id": session_id,
                    "reason": "missing_sparc_spk_emb",
                })
                continue

            # --- Gate 2: absolute duration gate (checked before fraction) ---
            active_s = _compute_active_speech_s(diarization)

            if active_s < 1.0:
                excluded.append({
                    "task_name": task_name,
                    "session_id": session_id,
                    "reason": "active_speech_too_short",
                })
                _logger.debug("Excluded %s (active_speech_too_short=%.2fs)", task_name, active_s)
                continue

            # --- Gate 3: speech fraction gate ---
            active_frac = active_s / duration if duration > 0 else 0.0

            if active_frac < low_conf_frac and duration > 0:
                excluded.append({
                    "task_name": task_name,
                    "session_id": session_id,
                    "reason": "low_speech_fraction",
                })
                _logger.debug(
                    "Excluded %s (low_speech_fraction=%.2f)", task_name, active_frac
                )
                continue

            # --- Down-weight 1–3 s recordings ---
            down_weight = 0.3 if active_s < min_active_s else 1.0

            # --- Quality weight ---
            w = min(active_s / 10.0, 1.0) * down_weight

            # --- Accumulate ---
            ecapa_np = speaker_embedding.numpy() if hasattr(speaker_embedding, "numpy") else np.array(speaker_embedding)
            sparc_np = np.array(sparc_spk, dtype=np.float64)

            ecapa_embs.append(_l2_normalize(ecapa_np.astype(np.float64)))
            sparc_embs.append(_l2_normalize(sparc_np))
            weights.append(w)
            included.append(task_name)
            total_speech_s += active_s

        # --- Determine age_group ---
        raw_age = age_map.get(pid)
        if raw_age is None:
            age_group = "unknown"
        elif raw_age < 18:
            age_group = "child"
        else:
            age_group = "adult"

        n_used = len(ecapa_embs)
        n_excluded = len(excluded)

        # --- Insufficient data check ---
        if n_used < min_recordings:
            profile = SpeakerProfile(
                participant_id=pid,
                ecapa_model_id=_ECAPA_MODEL_ID,
                sparc_model_id=_SPARC_MODEL_ID,
                num_recordings_used=n_used,
                num_recordings_excluded=n_excluded,
                total_active_speech_s=total_speech_s,
                ecapa_profile_quality_score=0.0,
                sparc_profile_quality_score=0.0,
                profile_status="insufficient_data",
                age_group=age_group,
                included_recordings=included,
                excluded_recordings=excluded,
                ecapa_embedding_centroid=None,
                sparc_embedding_centroid=None,
                created_at=datetime.now(timezone.utc).isoformat(),
                pipeline_config_hash=config_hash,
            )
            _write_profile(out_path, profile)
            profiles[pid] = profile
            _logger.info(
                "Participant %s: insufficient_data (%d/%d usable)", pid, n_used, min_recordings
            )
            continue

        # --- Outlier rejection (independent per embedding type) ---
        ecapa_kept, w_ecapa = _reject_outliers(ecapa_embs, weights, std_mult)
        sparc_kept, w_sparc = _reject_outliers(sparc_embs, weights, std_mult)

        # --- Centroids ---
        ecapa_centroid = _weighted_centroid(ecapa_kept, w_ecapa)
        sparc_centroid = _weighted_centroid(sparc_kept, w_sparc)

        # --- Quality scores ---
        ecapa_quality = _profile_quality(ecapa_kept)
        sparc_quality = _profile_quality(sparc_kept)

        # --- Profile status ---
        if ecapa_quality < contam_thresh or sparc_quality < contam_thresh:
            status = "contaminated"
        else:
            status = "ready"

        profile = SpeakerProfile(
            participant_id=pid,
            ecapa_model_id=_ECAPA_MODEL_ID,
            sparc_model_id=_SPARC_MODEL_ID,
            num_recordings_used=n_used,
            num_recordings_excluded=n_excluded,
            total_active_speech_s=total_speech_s,
            ecapa_profile_quality_score=round(ecapa_quality, 6),
            sparc_profile_quality_score=round(sparc_quality, 6),
            profile_status=status,
            age_group=age_group,
            included_recordings=included,
            excluded_recordings=excluded,
            ecapa_embedding_centroid=ecapa_centroid.tolist(),
            sparc_embedding_centroid=sparc_centroid.tolist(),
            created_at=datetime.now(timezone.utc).isoformat(),
            pipeline_config_hash=config_hash,
        )
        _write_profile(out_path, profile)
        profiles[pid] = profile
        _logger.info(
            "Participant %s: %s (used=%d, ecapa_q=%.3f, sparc_q=%.3f)",
            pid, status, n_used, ecapa_quality, sparc_quality,
        )

    return profiles


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _write_profile(profiles_dir: Path, profile: SpeakerProfile) -> None:
    """Write ``speaker_profile.json`` for one participant."""
    sub_dir = profiles_dir / f"sub-{profile.participant_id}"
    sub_dir.mkdir(parents=True, exist_ok=True)
    out_file = sub_dir / "speaker_profile.json"
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(profile.to_json(), fh, indent=2)
    _logger.debug("Wrote %s", out_file)


def _hash_config(cfg: dict) -> str:
    """Return an 8-character hex hash of the speaker_profile config dict."""
    payload = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:8]
