"""QA pipeline utility functions.

Cross-cutting helpers for the audio quality assurance pipeline:

- Config I/O and hashing      (T004)
- Per-audio JSON sidecar       (T005)
- Per-stage wall-clock timing  (T006)
- SLURM sharding               (T007)
- Model-failure error results  (T008)

No business logic lives here — this module is for I/O and cross-cutting
concerns only.
"""

import hashlib
import json
import logging
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.resources import files
from pathlib import Path
from typing import Optional

from b2aiprep.prepare.qa_models import (
    CheckResult,
    CheckType,
    Classification,
    CompositeScore,
    PipelineConfig,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal serialisation helpers
# ---------------------------------------------------------------------------


def _make_jsonable(obj: object) -> object:
    """Recursively convert non-JSON-serialisable types to plain Python types.

    Handles ``datetime`` → ISO-8601 string.  ``str``-based Enums produced by
    :func:`dataclasses.asdict` are already strings and require no special case.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _make_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_jsonable(i) for i in obj]
    return obj


def _config_as_jsonable(config: PipelineConfig) -> dict:
    """Return a JSON-serialisable dict representation of *config*."""
    return _make_jsonable(asdict(config))


# ---------------------------------------------------------------------------
# T004 — PipelineConfig loading and hashing
# ---------------------------------------------------------------------------


def load_config(path: Optional[str] = None) -> PipelineConfig:
    """Load a :class:`~b2aiprep.prepare.qa_models.PipelineConfig` from a JSON
    file or the bundled default.

    Args:
        path: Path to a JSON config file.  If ``None``, the bundled
              ``resources/qa_pipeline_config.json`` is used.

    Returns:
        Populated :class:`PipelineConfig` instance.
    """
    if path is None:
        resource = files("b2aiprep.prepare.resources").joinpath(
            "qa_pipeline_config.json"
        )
        raw: dict = json.loads(resource.read_text(encoding="utf-8"))
    else:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)

    # Strip JSON-only comment keys (e.g. "_comment") before mapping to
    # dataclass fields — they have no corresponding constructor argument.
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}

    created_at: Optional[datetime] = raw.get("created_at")
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)

    return PipelineConfig(
        config_version=raw.get("config_version", "1.0.0"),
        created_at=created_at,
        random_seed=raw.get("random_seed", 42),
        model_versions=raw.get("model_versions", {}),
        hard_gate_thresholds=raw.get("hard_gate_thresholds", {}),
        soft_score_thresholds=raw.get("soft_score_thresholds", {}),
        check_weights=raw.get("check_weights", {}),
        task_compliance_params=raw.get("task_compliance_params", {}),
        environment_noise_threshold=raw.get("environment_noise_threshold", 0.60),
        environment_noise_classes=raw.get("environment_noise_classes", []),
        confidence_disagreement_penalty=raw.get(
            "confidence_disagreement_penalty", 0.50
        ),
        min_transcript_confidence=raw.get("min_transcript_confidence", 0.70),
        human_review_timeout_days=raw.get("human_review_timeout_days", 30),
        sc_004_review_fraction_warn=raw.get("sc_004_review_fraction_warn", 0.15),
    )


def hash_config(config: PipelineConfig) -> str:
    """Return the SHA-256 hex digest of the serialised :class:`PipelineConfig`.

    ``created_at`` is excluded from the hash so that the digest is stable
    across snapshot writes with different timestamps.

    Args:
        config: :class:`PipelineConfig` instance to hash.

    Returns:
        Lowercase 64-character SHA-256 hex digest.
    """
    d = _config_as_jsonable(config)
    d.pop("created_at", None)
    canonical = json.dumps(d, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def save_config_snapshot(config: PipelineConfig, output_dir: str) -> Path:
    """Write a timestamped PipelineConfig snapshot to *output_dir*.

    Sets ``config.created_at`` to the current UTC time in-place and writes
    ``qa_pipeline_config_{hash[:8]}.json``.

    Args:
        config: :class:`PipelineConfig` to snapshot (mutated in-place).
        output_dir: Directory where the snapshot will be written.

    Returns:
        Path to the written JSON file.
    """
    config.created_at = datetime.now(timezone.utc)
    digest = hash_config(config)
    out_path = Path(output_dir) / f"qa_pipeline_config_{digest[:8]}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(_config_as_jsonable(config), fh, indent=2, sort_keys=True)
    _logger.info("Config snapshot written to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# T005 — Per-audio JSON sidecar writer
# ---------------------------------------------------------------------------


def write_audio_sidecar(
    bids_root: str,
    participant_id: str,
    session_id: str,
    task_name: str,
    check_results: list,
    composite_score: CompositeScore,
    transcript: Optional[str] = None,
    pii_spans: Optional[list] = None,
    timing_s: Optional[dict] = None,
) -> Path:
    """Write a per-audio JSON sidecar to the BIDS subject directory.

    The sidecar is written to::

        bids_root/<participant_id>/<session_id>/voice/
        <participant_id>_<session_id>_task-<task_name>_qa.json

    This file is sensitive — it contains the full transcript and PII spans
    with text.  It is release-gated separately from the distributable TSV
    outputs in OUTPUT_DIR.

    Args:
        bids_root: Root of the BIDS dataset directory.
        participant_id: BIDS participant identifier (e.g. ``"sub-001"``).
        session_id: BIDS session identifier (e.g. ``"ses-01"``).
        task_name: Task identifier string (e.g. ``"harvard-sentences-list-1-1"``).
        check_results: List of :class:`CheckResult` dataclass instances.
        composite_score: :class:`CompositeScore` dataclass instance.
        transcript: Full Whisper transcript text (key absent from output if
                    ``None``).
        pii_spans: PII span list including the ``text`` field (key absent if
                   ``None``).
        timing_s: Per-stage wall-clock timing dict from
                  :class:`TimingContext` (key absent if ``None``).

    Returns:
        Path to the written sidecar JSON file.
    """
    out_dir = Path(bids_root) / participant_id / session_id / "voice"
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{participant_id}_{session_id}_task-{task_name}_qa.json"
    out_path = out_dir / filename

    payload: dict = {
        "participant_id": participant_id,
        "session_id": session_id,
        "task_name": task_name,
        "check_results": [_make_jsonable(asdict(r)) for r in check_results],
        "composite_score": _make_jsonable(asdict(composite_score)),
    }
    if transcript is not None:
        payload["transcript"] = transcript
    if pii_spans is not None:
        payload["pii_spans"] = pii_spans
    if timing_s is not None:
        payload["timing_s"] = timing_s

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    _logger.debug("Audio sidecar written to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# T006 — Per-stage wall-clock timing context manager
# ---------------------------------------------------------------------------


class TimingContext:
    """Accumulates per-stage wall-clock timings for one audio file.

    Usage::

        timer = TimingContext()
        with timer.time("audio_quality"):
            ...
        with timer.time("pii_detection"):
            ...
        sidecar_timing = timer.get_timing_summary()
        # → {"audio_quality": 0.42, "pii_detection": 1.07}

    Each call to :meth:`time` creates an independent stage entry.  If the
    same *stage* label is used more than once the last measurement wins.
    """

    def __init__(self) -> None:
        self._timings: dict[str, float] = {}

    def time(self, stage: str) -> "_StageTimer":
        """Return a context manager that records *stage* duration.

        Args:
            stage: Human-readable stage label (e.g. ``"audio_quality"``).

        Returns:
            :class:`_StageTimer` context manager.
        """
        return _StageTimer(self, stage)

    def _record(self, stage: str, duration_s: float) -> None:
        self._timings[stage] = duration_s

    def get_timing_summary(self) -> dict:
        """Return a shallow copy of all recorded stage timings.

        Returns:
            ``{stage_name: duration_seconds}`` mapping.
        """
        return dict(self._timings)


class _StageTimer:
    """Context manager returned by :meth:`TimingContext.time`.

    Not intended for direct instantiation outside this module.
    """

    def __init__(self, ctx: TimingContext, stage: str) -> None:
        self._ctx = ctx
        self._stage = stage
        self._start: Optional[float] = None

    def __enter__(self) -> "_StageTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        elapsed = time.perf_counter() - self._start  # type: ignore[operator]
        self._ctx._record(self._stage, elapsed)


# ---------------------------------------------------------------------------
# T007 — SLURM sharding utility
# ---------------------------------------------------------------------------


def shard_audio_list(audio_paths: list, part: int, num_parts: int) -> list:
    """Return the non-overlapping subset for this SLURM array task.

    Shards are distributed evenly using Python's slice step so that all
    elements are covered and no element appears in more than one shard.

    Examples::

        shard_audio_list([0,1,2,3,4], part=1, num_parts=2) → [0,2,4]
        shard_audio_list([0,1,2,3,4], part=2, num_parts=2) → [1,3]
        shard_audio_list([0,1,2,3,4], part=1, num_parts=1) → [0,1,2,3,4]

    Args:
        audio_paths: Complete list of audio paths to shard.
        part: 1-based index of this shard (``1 ≤ part ≤ num_parts``).
        num_parts: Total number of shards.

    Returns:
        The subset of *audio_paths* assigned to this shard.

    Raises:
        ValueError: If *part* or *num_parts* are outside the valid range.
    """
    if num_parts < 1:
        raise ValueError(f"num_parts must be >= 1, got {num_parts}")
    if part < 1 or part > num_parts:
        raise ValueError(f"part must be in [1, {num_parts}], got {part}")
    return audio_paths[part - 1 :: num_parts]


# ---------------------------------------------------------------------------
# T008 — Model-failure error result constructor
# ---------------------------------------------------------------------------


def make_error_check_result(
    participant_id: str,
    session_id: str,
    task_name: str,
    check_type: CheckType,
    exception: Exception,
    model_versions: dict,
) -> CheckResult:
    """Build a :class:`CheckResult` with ``classification=error`` after a model failure.

    Logs the full exception traceback at ERROR level then returns a
    :class:`CheckResult` so the pipeline can route this audio to the human
    review queue and continue to the next audio without halting (FR-014).

    Args:
        participant_id: Participant identifier.
        session_id: Session identifier.
        task_name: Task name.
        check_type: Which quality check failed.
        exception: The caught exception instance.
        model_versions: Model version dict at the time of failure.

    Returns:
        :class:`CheckResult` with ``classification=Classification.ERROR``,
        ``score=0.0``, and ``confidence=0.0``.
    """
    _logger.error(
        "Check %s failed for %s/%s/%s: %s",
        check_type.value,
        participant_id,
        session_id,
        task_name,
        exception,
        exc_info=True,
    )
    return CheckResult(
        participant_id=participant_id,
        session_id=session_id,
        task_name=task_name,
        check_type=check_type,
        score=0.0,
        confidence=0.0,
        classification=Classification.ERROR,
        detail={
            "error_type": type(exception).__name__,
            "error_message": str(exception),
            "traceback": traceback.format_exc(),
        },
        model_versions=model_versions,
    )
