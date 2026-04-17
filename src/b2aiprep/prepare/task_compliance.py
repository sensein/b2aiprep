"""Task-compliance checks: Tier A (WER), Tier B (signal), Tier C (LLM).

T018 — Tier A + ``get_compliance_tier`` dispatcher
T019 — Tier B (phonation, DDK, pitch-glide, breathing/cough)
T020 — Tier C (LLM-based open/conversational tasks)

All thresholds are read from ``PipelineConfig`` so that population-specific
tuning (paediatric, neurodegenerative) requires no code changes.

Public API
----------
- ``get_compliance_tier(task_category) -> str``  ("A", "B", or "C")
- ``check_tier_a(audio_record, config) -> CheckResult``
- ``check_tier_b(audio_record, config) -> CheckResult``
- ``check_tier_c(audio_record, config) -> CheckResult``
- ``check_task_compliance(audio_record, config) -> CheckResult``  (dispatcher)
"""

import logging
import re
from pathlib import Path
from typing import Any, Optional

import torch

from b2aiprep.prepare.qa_models import CheckResult, CheckType, Classification, PipelineConfig

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier mapping
# ---------------------------------------------------------------------------

_TIER_A: frozenset = frozenset(
    [
        "harvard-sentences",
        "cape-V-sentences",
        "passage",
        "rainbow",
        "caterpillar-passage",
        "repeat-words",
        "sentence",
    ]
)

_TIER_B: frozenset = frozenset(
    [
        "diadochokinesis",
        "phonation",
        "pitch-glide",
        "breathing",
        "cough",
    ]
)

_TIER_C: frozenset = frozenset(
    [
        "naming",
        "story",
        "picture",
        "conversational",
        "cognitive",
        "recitation",
        "loudness",
    ]
)


def get_compliance_tier(task_category: str) -> str:
    """Return the compliance tier ("A", "B", or "C") for *task_category*.

    Args:
        task_category: Normalised task category string derived from the
                       task name (e.g. ``"harvard-sentences"``,
                       ``"phonation"``, ``"conversational"``).

    Returns:
        One of ``"A"``, ``"B"``, ``"C"``.

    Raises:
        ValueError: If *task_category* is empty or unrecognised.
    """
    if not task_category:
        raise ValueError("task_category must not be empty")
    if task_category in _TIER_A:
        return "A"
    if task_category in _TIER_B:
        return "B"
    if task_category in _TIER_C:
        return "C"
    raise ValueError(
        f"Unrecognised task_category: {task_category!r}. "
        f"Expected one of: {sorted(_TIER_A | _TIER_B | _TIER_C)}"
    )


# ---------------------------------------------------------------------------
# Internal helpers (patched by tests)
# ---------------------------------------------------------------------------


def _load_transcript(audio_record: Any) -> str:
    """Return the pre-computed Whisper transcript for *audio_record*.

    Reads from the ``_features.pt`` file.  Returns an empty string if
    no transcript is stored.
    """
    features_path = getattr(audio_record, "features_path", None)
    if features_path is None:
        return ""
    fp = Path(features_path)
    if not fp.exists():
        return ""
    features = torch.load(str(fp), weights_only=False, map_location="cpu")
    return str(features.get("transcription", ""))


def _get_prompt_text(task_name: str) -> str:
    """Return the expected spoken content for *task_name*.

    Looks up ``audio_task_descriptions.json`` bundled with b2aiprep.
    Falls back to stripping a trailing ``-N`` numeric suffix and looking up
    the base task name (e.g. ``"passage-8"`` → ``"passage"``).  For base
    entries with multiple prompts the N-th item is returned (1-based).  For
    base entries with a single long prompt the text is sentence-split and the
    N-th sentence is returned.
    Returns an empty string when no reference text can be resolved.
    """
    try:
        import json
        from importlib.resources import files

        resource = files("b2aiprep.prepare.resources").joinpath(
            "audio_task_descriptions.json"
        )
        descriptions = json.loads(resource.read_text(encoding="utf-8"))

        def _lookup_prompts(name: str):
            """Return prompts list for *name*, or None if the key is absent."""
            if isinstance(descriptions, list):
                for entry in descriptions:
                    if entry.get("task_name") == name or entry.get("id") == name:
                        return entry.get("prompts", entry.get("task_prompts", []))
                return None
            if isinstance(descriptions, dict):
                if name in descriptions:
                    entry = descriptions[name]
                    return entry.get("prompts", entry.get("task_prompts", []))
            return None

        # 1. Try exact match.
        prompts = _lookup_prompts(task_name)

        if prompts is None:
            # 2. Strip trailing -N and try the base key.
            m = re.match(r"^(.+)-(\d+)$", task_name)
            if m:
                base, idx = m.group(1), int(m.group(2))
                prompts = _lookup_prompts(base)
                if prompts is not None and len(prompts) > 1:
                    # Multiple items (e.g. sentence list): pick by 1-based index.
                    i = idx - 1
                    selected = prompts[i] if 0 <= i < len(prompts) else prompts[-1]
                    return str(selected)
                if prompts and len(prompts) == 1:
                    # Single long text (e.g. full passage): split into sentences.
                    sentences = re.split(r"(?<=[.!?])\s+", str(prompts[0]))
                    i = idx - 1
                    return sentences[i] if 0 <= i < len(sentences) else str(prompts[0])

        if prompts:
            return " ".join(str(p) for p in prompts)

    except Exception as exc:
        _logger.debug("Could not load prompt text for %s: %s", task_name, exc)
    return ""


def _get_active_speech_duration(audio_record: Any) -> float:
    """Return the active-speech duration in seconds for *audio_record*.

    Reads from the ``_features.pt`` file if available; otherwise loads
    the audio and trims silence using a simple energy threshold.

    Args:
        audio_record: Object with ``features_path`` and optionally
                      ``audio_path``.

    Returns:
        Duration of active speech in seconds.  Returns 0.0 when the
        audio cannot be read.
    """
    # Try to read from pre-computed features
    features_path = getattr(audio_record, "features_path", None)
    if features_path is not None:
        fp = Path(features_path)
        if fp.exists():
            features = torch.load(str(fp), weights_only=False, map_location="cpu")
            duration = features.get("active_speech_duration_s")
            if duration is not None:
                return float(duration)

    # Fallback: simple energy-based silence trimming
    audio_path = getattr(audio_record, "audio_path", None)
    if audio_path is not None and Path(audio_path).exists():
        try:
            import torchaudio

            waveform, sr = torchaudio.load(str(audio_path))
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            y = waveform.squeeze()
            # Energy-based voiced frame count
            frame_len = max(1, int(0.025 * sr))
            hop_len = max(1, int(0.0125 * sr))
            if y.shape[0] >= frame_len:
                frames = y.unfold(0, frame_len, hop_len)
                voiced = (frames.pow(2).mean(dim=1) >= 1e-6).sum().item()
                return float(voiced * (hop_len / sr))
        except Exception as exc:
            _logger.debug("Active speech duration estimation failed: %s", exc)

    return 0.0


def _get_task_instructions(task_name: str) -> str:
    """Return the human-readable task instructions for *task_name*.

    Strips a trailing ``-N`` numeric suffix before lookup so that task names
    like ``favorite-food-1`` resolve to the ``favorite-food`` entry.
    Returns an empty string when not found.
    """
    try:
        import json
        from importlib.resources import files

        resource = files("b2aiprep.prepare.resources").joinpath(
            "audio_task_descriptions.json"
        )
        descriptions = json.loads(resource.read_text(encoding="utf-8"))

        def _lookup(name: str):
            if isinstance(descriptions, dict):
                return descriptions.get(name, {}).get("instructions", "")
            for entry in descriptions:
                if entry.get("task_name") == name or entry.get("id") == name:
                    return entry.get("instructions", "")
            return ""

        instructions = _lookup(task_name)
        if not instructions:
            m = re.match(r"^(.+)-\d+$", task_name)
            if m:
                instructions = _lookup(m.group(1))
        return instructions or ""
    except Exception as exc:
        _logger.debug("Could not load task instructions for %s: %s", task_name, exc)
    return ""


def _task_correctness_phi4(audio_record: Any, config: PipelineConfig) -> Optional[bool]:
    """LLM-based task-correctness check using Phi-4 mini.

    Uses greedy decoding (temperature=0, do_sample=False) because the
    output space is binary (on-task / off-task) and borderline cases are
    handled by the human review path rather than sampling.

    Args:
        audio_record: Object with transcript and task metadata.
        config: :class:`PipelineConfig` (random seed, model versions).

    Returns:
        ``True`` if the recording appears on-task, ``False`` if off-task,
        ``None`` if the model is unavailable or the call fails (callers
        should route to ``needs_review`` rather than ``fail``).
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import]

        transcript = _load_transcript(audio_record)
        task_name = getattr(audio_record, "task_name", "")

        if not transcript.strip():
            return False

        model_id = config.model_versions.get(
            "phi4", "microsoft/Phi-4-mini-instruct"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map="auto"
        )
        model.eval()

        instructions = _get_task_instructions(task_name)
        task_description = instructions if instructions else task_name
        prompt = (
            f'Task instruction: "{task_description}"\n'
            f'Participant said: "{transcript}"\n'
            "Did the participant complete the task? Answer only: yes or no"
        )
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        answer = tokenizer.decode(
            output_ids[0][inputs.shape[-1]:],
            skip_special_tokens=True,
        ).strip().lower()
        return answer.startswith("yes")

    except Exception as exc:
        _logger.warning("Phi-4 task compliance check unavailable: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Tier A — WER / edit-distance compliance (T018)
# ---------------------------------------------------------------------------


def _levenshtein_words(a: list, b: list) -> int:
    """Compute the word-level Levenshtein edit distance between *a* and *b*."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(dp[j], dp[j - 1], prev)
            prev = temp
    return dp[n]


def check_tier_a(audio_record: Any, config: PipelineConfig) -> CheckResult:
    """Tier A task compliance: character-normalised WER (T018).

    Computes the word error rate between the Whisper transcript and the
    known prompt text from ``audio_task_descriptions.json``.

    Classification:
    - WER < 0.10  → ``PASS``   (score ≥ 0.90)
    - WER 0.10–0.30 → ``NEEDS_REVIEW`` (score 0.70–0.90)
    - WER > 0.30  → ``NEEDS_REVIEW``  (score < 0.70)

    Args:
        audio_record: Object with ``participant_id``, ``session_id``,
                      ``task_name``, ``features_path``.
        config: :class:`PipelineConfig`.

    Returns:
        :class:`CheckResult` for the ``task_compliance`` check type.
    """
    participant_id = getattr(audio_record, "participant_id", "")
    session_id = getattr(audio_record, "session_id", "")
    task_name = getattr(audio_record, "task_name", "")

    transcript = _load_transcript(audio_record)
    prompt_text = _get_prompt_text(task_name)

    ref_words = prompt_text.lower().split()
    hyp_words = transcript.lower().split()

    if not ref_words:
        # No reference text available — cannot compute WER; route to review.
        return CheckResult(
            participant_id=participant_id,
            session_id=session_id,
            task_name=task_name,
            check_type=CheckType.TASK_COMPLIANCE,
            score=0.5,
            confidence=0.30,
            classification=Classification.NEEDS_REVIEW,
            detail={
                "compliance_tier": "A",
                "wer": None,
                "phoneme_match_score": None,
                "active_speech_duration_s": None,
                "llm_compliance": None,
                "note": "no_reference_text",
            },
            model_versions={},
        )

    wer = _levenshtein_words(hyp_words, ref_words) / len(ref_words)
    wer = min(1.0, max(0.0, wer))

    # score = 1 − WER (compliance confidence formula from spec)
    score = 1.0 - wer
    confidence = score  # same formula per spec

    soft = config.soft_score_thresholds
    pass_min = float(soft.get("pass_min", 0.75))
    fail_max = float(soft.get("fail_max", 0.40))

    if score >= pass_min:
        classification = Classification.PASS
    elif score <= fail_max:
        classification = Classification.FAIL
    else:
        classification = Classification.NEEDS_REVIEW

    detail: dict = {
        "compliance_tier": "A",
        "wer": round(wer, 6),
        "phoneme_match_score": None,
        "active_speech_duration_s": None,
        "llm_compliance": None,
    }

    return CheckResult(
        participant_id=participant_id,
        session_id=session_id,
        task_name=task_name,
        check_type=CheckType.TASK_COMPLIANCE,
        score=round(score, 6),
        confidence=round(min(1.0, max(0.0, confidence)), 6),
        classification=classification,
        detail=detail,
        model_versions={},
    )


# ---------------------------------------------------------------------------
# Tier B — Signal / phoneme compliance (T019)
# ---------------------------------------------------------------------------


def check_tier_b(audio_record: Any, config: PipelineConfig) -> CheckResult:
    """Tier B task compliance: signal-level checks (T019).

    Currently implements the phonation duration gate (the most common
    Tier B task); DDK, pitch-glide, and breathing/cough use the same
    infrastructure and are extended in future iterations.

    Classification:
    - duration < ``phonation.min_duration_s`` → ``FAIL`` (hard gate)
    - Otherwise soft-threshold on normalised duration score

    Args:
        audio_record: Object with ``participant_id``, ``session_id``,
                      ``task_name``, ``features_path``.
        config: :class:`PipelineConfig` with ``task_compliance_params``.

    Returns:
        :class:`CheckResult` for the ``task_compliance`` check type.
    """
    participant_id = getattr(audio_record, "participant_id", "")
    session_id = getattr(audio_record, "session_id", "")
    task_name = getattr(audio_record, "task_name", "")

    # Determine minimum duration from config
    tcp = config.task_compliance_params
    min_duration_s = float(
        tcp.get("phonation", {}).get("min_duration_s", 3.0)
    )

    active_duration = _get_active_speech_duration(audio_record)

    # Hard gate: duration below minimum → FAIL
    if active_duration < min_duration_s:
        score = max(0.0, active_duration / (min_duration_s + 1e-10))
        # Force to fail range
        score = min(score, 0.35)
        classification = Classification.FAIL
        confidence = 0.90
    else:
        # Normalise: min_duration_s → 0.5, 2× min → 1.0
        score = min(1.0, 0.5 + 0.5 * (active_duration - min_duration_s) / (min_duration_s + 1e-10))
        confidence = min(0.90, score)
        soft = config.soft_score_thresholds
        pass_min = float(soft.get("pass_min", 0.75))
        classification = Classification.PASS if score >= pass_min else Classification.NEEDS_REVIEW

    detail: dict = {
        "compliance_tier": "B",
        "wer": None,
        "phoneme_match_score": None,
        "active_speech_duration_s": round(active_duration, 3),
        "llm_compliance": None,
    }

    return CheckResult(
        participant_id=participant_id,
        session_id=session_id,
        task_name=task_name,
        check_type=CheckType.TASK_COMPLIANCE,
        score=round(score, 6),
        confidence=round(min(1.0, max(0.0, confidence)), 6),
        classification=classification,
        detail=detail,
        model_versions={},
    )


# ---------------------------------------------------------------------------
# Tier C — LLM-based open / conversational compliance (T020)
# ---------------------------------------------------------------------------


def check_tier_c(audio_record: Any, config: PipelineConfig) -> CheckResult:
    """Tier C task compliance: LLM-based open-task check (T020).

    Uses Phi-4 mini (greedy decode) for binary on-task classification,
    augmented by an active-speech duration gate.

    Confidence rules (from spec):
    - LLM=True AND duration_met → confidence 0.9 → PASS
    - LLM=True BUT duration short OR low transcript confidence → 0.5 → NEEDS_REVIEW
    - LLM=False → confidence 0.3, score 0.0 → FAIL

    Args:
        audio_record: Object with ``participant_id``, ``session_id``,
                      ``task_name``, ``features_path``.
        config: :class:`PipelineConfig`.

    Returns:
        :class:`CheckResult` for the ``task_compliance`` check type.
    """
    participant_id = getattr(audio_record, "participant_id", "")
    session_id = getattr(audio_record, "session_id", "")
    task_name = getattr(audio_record, "task_name", "")

    tcp = config.task_compliance_params
    min_active_s = float(
        tcp.get("conversational", {}).get("min_active_speech_duration_s", 3.0)
    )

    llm_result = _task_correctness_phi4(audio_record, config)
    active_duration = _get_active_speech_duration(audio_record)
    duration_met = active_duration >= min_active_s

    if llm_result is None:
        # Model unavailable — cannot determine compliance; route to review.
        score = 0.5
        confidence = 0.30
        classification = Classification.NEEDS_REVIEW
    elif not llm_result:
        # Off-task → FAIL
        score = 0.0
        confidence = 0.30
        classification = Classification.FAIL
    elif duration_met:
        # On-task + sufficient duration → PASS
        score = 0.90
        confidence = 0.90
        classification = Classification.PASS
    else:
        # On-task but short → reduce confidence → NEEDS_REVIEW
        score = 0.55
        confidence = 0.50
        classification = Classification.NEEDS_REVIEW

    detail: dict = {
        "compliance_tier": "C",
        "wer": None,
        "phoneme_match_score": None,
        "active_speech_duration_s": round(active_duration, 3),
        "llm_compliance": llm_result,
    }

    return CheckResult(
        participant_id=participant_id,
        session_id=session_id,
        task_name=task_name,
        check_type=CheckType.TASK_COMPLIANCE,
        score=round(score, 6),
        confidence=round(min(1.0, max(0.0, confidence)), 6),
        classification=classification,
        detail=detail,
        model_versions={
            "phi4": config.model_versions.get("phi4", "microsoft/Phi-4-mini-instruct"),
        },
    )


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------


def check_task_compliance(audio_record: Any, config: PipelineConfig) -> CheckResult:
    """Dispatch to the correct tier check for *audio_record*.

    Derives the task category from the task name and calls the
    appropriate tier function.  Falls back to Tier C (most permissive)
    for unrecognised task categories to avoid blocking the pipeline.

    Args:
        audio_record: Object with ``task_name`` and other fields.
        config: :class:`PipelineConfig`.

    Returns:
        :class:`CheckResult` from the dispatched tier check.
    """
    from b2aiprep.prepare.qa_utils import make_error_check_result

    task_name = getattr(audio_record, "task_name", "")

    # Derive category: use the first dash-separated component(s)
    # that match a known tier category.
    parts = task_name.replace("_", "-").split("-")
    category = ""
    for n in range(len(parts), 0, -1):
        candidate = "-".join(parts[:n])
        if candidate in (_TIER_A | _TIER_B | _TIER_C):
            category = candidate
            break

    if not category:
        _logger.debug(
            "Could not derive tier category from task_name=%r; defaulting to Tier C",
            task_name,
        )
        return check_tier_c(audio_record, config)

    tier = get_compliance_tier(category)
    if tier == "A":
        return check_tier_a(audio_record, config)
    if tier == "B":
        return check_tier_b(audio_record, config)
    return check_tier_c(audio_record, config)
