"""This file implements logic for doing quality control on the b2ai dataset."""

from senselab.audio.tasks.quality_control.quality_control import check_quality
from senselab.audio.tasks.quality_control.review import review_files
from senselab.audio.tasks.quality_control.metrics import primary_speaker_ratio_metric
from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios

import logging
import numpy as np
import torch
import pandas as pd
import tempfile
from pathlib import Path
from typing import Optional

import parselmouth

from b2aiprep.prepare.dataset import BIDSDataset
from b2aiprep.prepare.qa_models import CheckResult, CheckType, Classification, PipelineConfig
from b2aiprep.prepare.utils import copy_package_resource

RESAMPLE_RATE=16000

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# T014/T015 — Per-audio technical quality check
# ---------------------------------------------------------------------------


def _estimate_snr_db(y: torch.Tensor, sr: int) -> float:  # noqa: ARG001
    """Estimate SNR using the spectral flatness (Wiener entropy) method.

    A pure tone or clean speech has very low spectral flatness → high SNR.
    White noise has spectral flatness ≈ 1.0 → SNR ≈ 0 dB.  This method
    correctly returns high SNR for both pure tones and clean speech without
    needing a reference signal.

    Args:
        y:  Mono 1-D float32 waveform tensor.
        sr: Sample rate (reserved for future adaptive n_fft selection).

    Returns:
        Estimated SNR in dB, clamped to ≥ 0.0.
    """
    n_fft = 512
    if y.shape[0] < n_fft:
        return 0.0

    window = torch.hann_window(n_fft, device=y.device)
    spec = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=n_fft // 4,
        window=window,
        return_complex=True,
    )  # [freq, time]

    power = spec.abs().pow(2) + 1e-12  # avoid log(0)

    # Mean power spectrum across time frames
    mean_power = power.mean(dim=1)  # [freq]

    # Spectral flatness = geometric_mean / arithmetic_mean
    geom_mean = mean_power.log().mean().exp()
    arith_mean = mean_power.mean()

    sfm = float((geom_mean / (arith_mean + 1e-12)).clamp(1e-10, 1.0))
    snr_db = -10.0 * float(np.log10(sfm + 1e-10))
    return max(0.0, snr_db)


def _classify_environment_yamnet(
    y: torch.Tensor,
    sr: int,
    config: PipelineConfig,
    top_k: int = 3,
) -> tuple:
    """Run YAMNet (T015) to classify the acoustic environment.

    Uses ``torchaudio.pipelines.YAMNET`` for AudioSet-class predictions.
    Gracefully returns ``([], False)`` on any import error or inference
    failure so that a missing model never halts the pipeline.

    Args:
        y:      Mono 1-D float32 waveform at *sr* Hz.
        sr:     Sample rate of *y*.
        config: :class:`PipelineConfig` with noise-class settings.
        top_k:  Number of top predictions to return (default 3).

    Returns:
        ``(top_labels, noise_flag)`` where *top_labels* is a list of
        ``{"label": str, "confidence": float}`` dicts (may be empty on
        failure) and *noise_flag* is ``True`` when the top-1 label
        matches a configured noise superclass with confidence ≥ threshold.
    """
    try:
        import torchaudio  # already a project dependency

        bundle = torchaudio.pipelines.YAMNET
        model = bundle.get_model()
        model.eval()

        # YAMNet requires 16 kHz mono
        if sr != 16000:
            wav = torchaudio.functional.resample(y, sr, 16000)
        else:
            wav = y

        with torch.no_grad():
            scores, _embeddings, _spec = model(wav.unsqueeze(0))

        # Average logit scores over time frames then softmax → probabilities
        probs = torch.softmax(scores.mean(dim=0), dim=0)  # [n_classes]
        k = min(top_k, probs.shape[0])
        top_scores, top_indices = probs.topk(k)
        labels = bundle.get_labels()

        top_labels = [
            {
                "label": labels[int(idx)],
                "confidence": round(float(s), 4),
            }
            for idx, s in zip(top_indices.tolist(), top_scores.tolist())
        ]

        # Noise flag: top-1 label in a configured noise superclass
        noise_flag = False
        if top_labels:
            top1_label_lower = top_labels[0]["label"].lower()
            top1_conf = top_labels[0]["confidence"]
            if top1_conf >= config.environment_noise_threshold:
                for nc in config.environment_noise_classes:
                    nc_lower = nc.lower()
                    if nc_lower in top1_label_lower or top1_label_lower in nc_lower:
                        noise_flag = True
                        break

        return top_labels, noise_flag

    except Exception as exc:
        _logger.warning("YAMNet classification skipped: %s", exc)
        return [], False


def check_audio_quality(
    audio_path: Path,
    config: PipelineConfig,
    participant_id: str = "",
    session_id: str = "",
    task_name: str = "",
) -> CheckResult:
    """Check technical audio quality for one audio file (T014/T015).

    Computes proportion-clipped, proportion-silent, spectral SNR, and
    amplitude-modulation depth; evaluates hard-gate and soft-score
    thresholds from *config*; and runs YAMNet for acoustic scene
    classification.

    Args:
        audio_path:     Path to the audio file (WAV or any torchaudio-
                        readable format).
        config:         :class:`PipelineConfig` with threshold/weight
                        settings.
        participant_id: BIDS participant ID (without ``sub-`` prefix).
        session_id:     BIDS session ID (without ``ses-`` prefix).
        task_name:      BIDS task name string.

    Returns:
        :class:`CheckResult` for the ``audio_quality`` check type.

    Raises:
        Any ``torchaudio.load`` exception propagates to the caller, which
        should call
        :func:`~b2aiprep.prepare.qa_utils.make_error_check_result`
        (FR-014 model-failure path).
    """
    import torchaudio

    # --- Load audio (exceptions propagate → make_error_check_result) ---
    waveform, sr = torchaudio.load(str(audio_path))

    # Downmix to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16 kHz
    if sr != RESAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, RESAMPLE_RATE)
        sr = RESAMPLE_RATE

    y = waveform.squeeze()  # 1-D float32 tensor
    n_samples = y.shape[0]

    # --- Raw metric computation ---

    # Clipping: proportion of samples at or near full scale (≥ 0.99)
    proportion_clipped = float((y.abs() >= 0.99).float().mean())

    # Short-time energy frames: 25 ms window / 12.5 ms hop
    frame_len = max(1, min(int(0.025 * sr), n_samples))
    hop_len = max(1, min(int(0.0125 * sr), n_samples))
    if n_samples >= frame_len:
        frames = y.unfold(0, frame_len, hop_len)       # [n_frames, frame_len]
        frame_energy = frames.pow(2).mean(dim=1)        # [n_frames]
    else:
        frame_energy = y.pow(2).mean().unsqueeze(0)

    # Silence: proportion of frames below energy threshold (≈ −60 dBFS)
    _silence_threshold = 1e-6
    proportion_silent = float((frame_energy < _silence_threshold).float().mean())

    # SNR via spectral flatness (handles pure tones and speech correctly)
    peak_snr_db = _estimate_snr_db(y, sr)

    # Amplitude-modulation depth: coefficient of variation of short-time RMS
    ste_rms = frame_energy.sqrt()
    mean_rms = float(ste_rms.mean())
    amplitude_modulation_depth = float(
        ste_rms.std() / (mean_rms + 1e-10) if mean_rms > 1e-10 else 0.0
    )

    # --- Hard gate evaluation ---
    gates = config.hard_gate_thresholds
    snr_min_db = float(gates.get("snr_min_db", 12.0))
    clipping_max = float(gates.get("clipping_max", 0.05))
    silence_max = float(gates.get("silence_max", 0.50))

    hard_gate_triggered = bool(
        proportion_clipped > clipping_max
        or proportion_silent > silence_max
        or peak_snr_db < snr_min_db
    )

    # --- YAMNet environment classification (T015) ---
    environment_top_labels, environment_noise_flag = _classify_environment_yamnet(
        y, sr, config
    )

    # --- Score and classification ---
    if hard_gate_triggered:
        score = 0.0
        confidence = 0.90
        classification = Classification.FAIL
    else:
        # Normalise each sub-metric to [0, 1]
        # SNR: snr_min_db → 0.0, 40 dB → 1.0
        snr_ceil = 40.0
        snr_score = min(
            1.0,
            max(0.0, (peak_snr_db - snr_min_db) / max(snr_ceil - snr_min_db, 1e-10)),
        )
        # Clipping: 0 → 1.0, clipping_max → 0.0
        clipping_score = 1.0 - min(1.0, proportion_clipped / max(clipping_max, 1e-10))
        # Silence: 0 → 1.0, silence_max → 0.0
        silence_score = 1.0 - min(1.0, proportion_silent / max(silence_max, 1e-10))

        score = float(0.5 * snr_score + 0.3 * clipping_score + 0.2 * silence_score)
        score = min(1.0, max(0.0, score))

        soft = config.soft_score_thresholds
        pass_min = float(soft.get("pass_min", 0.75))
        fail_max_t = float(soft.get("fail_max", 0.40))

        if score >= pass_min:
            classification = Classification.PASS
            margin = (score - pass_min) / max(1.0 - pass_min, 1e-10)
            confidence = 0.75 + 0.20 * min(1.0, margin)
        elif score <= fail_max_t:
            classification = Classification.FAIL
            margin = (fail_max_t - score) / max(fail_max_t, 1e-10)
            confidence = 0.75 + 0.20 * min(1.0, margin)
        else:
            classification = Classification.NEEDS_REVIEW
            confidence = 0.50

    try:
        import torchaudio as _ta
        _torchaudio_version = _ta.__version__
    except Exception:
        _torchaudio_version = "unknown"

    detail = {
        "proportion_clipped": round(proportion_clipped, 6),
        "proportion_silent": round(proportion_silent, 6),
        "peak_snr_db": round(peak_snr_db, 2),
        "amplitude_modulation_depth": round(amplitude_modulation_depth, 6),
        "hard_gate_triggered": hard_gate_triggered,
        "environment_top_labels": environment_top_labels,
        "environment_noise_flag": environment_noise_flag,
    }

    return CheckResult(
        participant_id=participant_id,
        session_id=session_id,
        task_name=task_name,
        check_type=CheckType.AUDIO_QUALITY,
        score=round(score, 6),
        confidence=round(min(1.0, max(0.0, confidence)), 6),
        classification=classification,
        detail=detail,
        model_versions={
            "yamnet": f"torchaudio.pipelines.YAMNET (torchaudio {_torchaudio_version})",
        },
    )

def quality_control_wrapper(
    audio_paths,
    bids_path,
    outdir=None,
    batch_size=8,
    num_cores=4,
    skip_windowing=False,
    deep_checks=False,
):
    """
    Wrapper for running quality control metrics.

    Args:
        audio_paths: List of audios to run quality metrics on
        bids_path: Path to the root of the BIDS directory; audio_quality_metrics.tsv
            and audio_quality_metrics.json will be written here
        outdir: Optional directory for storing intermediate senselab output files.
            If not provided, a temporary directory is used.
        batch_size: Batch size for parallelizing results
        num_cores: Number of cores for parallelizing results across
        skip_windowing: If true, don't compute windowed metrics
        deep_checks: Whether to run deep quality checks like snorkel recommendations and silence trimming
    """

    bids_path = Path(bids_path)

    temp_dir_obj = None
    try:
        if not outdir:
            temp_dir_obj = tempfile.TemporaryDirectory()
            quality_check_dir = temp_dir_obj.name
        else:
            quality_check_dir = outdir

        evaluation_df = check_quality(
            audio_paths=audio_paths,
            output_dir=quality_check_dir,
            batch_size=batch_size,
            n_cores=num_cores,
            window_size_sec=0.025,
            step_size_sec=0.0125,
            skip_windowing=skip_windowing,
        )

        _logger.info(f"Result of check_quality: {evaluation_df}")

        # Remap to standard dataset identifiers and drop senselab-internal columns.
        evaluation_df["participant_id"] = evaluation_df["path"].apply(
            BIDSDataset._extract_participant_id_from_path
        )
        evaluation_df["session_id"] = evaluation_df["path"].apply(
            BIDSDataset._extract_session_id_from_path
        )
        evaluation_df["task_name"] = evaluation_df["path"].apply(
            BIDSDataset._extract_task_name_from_path
        )
        drop_cols = [c for c in ("id", "path", "activity") if c in evaluation_df.columns]
        id_cols = ["participant_id", "session_id", "task_name"]
        metric_cols = [c for c in evaluation_df.columns if c not in drop_cols + id_cols]
        evaluation_df = evaluation_df[id_cols + metric_cols]
        evaluation_df.to_csv(bids_path / "audio_quality_metrics.tsv", sep="\t", index=False)
        copy_package_resource(
            "b2aiprep.prepare.resources",
            "audio_quality_metrics.json",
            str(bids_path),
        )

        if not deep_checks: return # goes to finally block first

        df_path = quality_check_dir / "quality_control_results_non_windowed.csv"
        evaluation_review_df = review_files(
            df_path=df_path,
            correlation_threshold= 0.99,
            output_dir=outdir,
        )

        _logger.info(f"Result of review files {evaluation_review_df}")

        _logger.info("Running custom checks using features that were precomputed")


        # For each audio:
        #   - Grab the features of the audio that might be useful (diarization)
        #   - run checks based on those features (number of speakers check)
        #   - trimming silence check

        trimming = {
            'path': [],
            'subject': [],
            'task': [],
            'flag': [],
            'start': [],
            'end': [],
            'percentage_trimmed': [],
            'proposed_trimming_method': []
        }

        diarization_qc = {
            'path': [],
            'subject': [],
            'task': [],
            'flag': [],
            'num_speakers': [],
            'proportion_primary_speaker': [],
        }

        # Process each record
        for audio_path in audio_paths:

            aduio_path_split = audio_path.stem.split('_')
            subject, task = aduio_path_split[0], '-'.join(aduio_path_split[2:])

            audio_feature_path = audio_path.parent / f"{audio_path.stem}_features.pt"
            if not audio_feature_path.exists():
                _logger.warning(f"Features file {audio_feature_path} does not exist for existing audio {audio_path}")
                continue
            features = torch.load(audio_feature_path, weights_only=False, map_location=torch.device('cpu'))

            audio_obj = Audio(filepath=audio_path)
            audio_orig = downmix_audios_to_mono([audio_obj])[0]

            if audio_orig.sampling_rate != RESAMPLE_RATE:
                # Resample both audios to 16kHz
                audio_16k = resample_audios([audio_orig], RESAMPLE_RATE)[0]
            else:
                audio_16k = audio_orig

            # Silence trimming
            y, sr = audio_16k.waveform.squeeze(), audio_16k.sampling_rate
            _logger.info(f"{audio_path} has shape {y.shape} and sr of {sr}")
            duration_before = y.shape[-1] / sr

            # First trim: pitch-based (Praat), fallback to energy-based
            start_praat, end_praat = trim_audio_with_praat(y, sr) #y_trimmed = trim_audio_with_praat(y, sr)
            duration_after_praat = (end_praat-start_praat+1) / sr

            trimming['path'].append(audio_path)
            trimming['subject'].append(subject)
            trimming['task'].append(task)
            if duration_after_praat < 0.2 * duration_before:
                _logger.info(f"Praat trim too short ({duration_after_praat:.2f}s < 20% of {duration_before:.2f}s) for {audio_path} → retrying with energy-based trim")
                #y_trimmed = trim_until_silent(y, sr, threshold_ratio=0.10)
                _logger.info(f"Checking again: shape {y.shape} and sr of {sr}")
                start_energy, end_energy = trim_until_silent(y, sr, threshold_ratio=0.10)
                duration_after_energy = (end_energy-start_energy+1) / sr

                if duration_after_energy < 0.2 * duration_before:
                    _logger.info(f"Energy trim too short ({duration_after_energy:.2f}s < 20% of {duration_before:.2f}s) for {audio_path} → Flagging audio for possible removal")
                    trimming['flag'].append('large_proportion_silence')
                    trimming['start'].append(start_energy)
                    trimming['end'].append(end_energy)
                    trimming['percentage_trimmed'].append((duration_before-duration_after_energy)/duration_before)
                    trimming['proposed_trimming_method'].append(None)
                elif end_energy/sr + 0.5 >= duration_before and start_energy/sr - 0.5 <= 0:
                    _logger.info(f"Trimming {audio_path} with .5s padding will have no effect on audio length")
                    trimming['flag'].append(None)
                    trimming['start'].append(0)
                    trimming['end'].append(duration_before)
                    trimming['percentage_trimmed'].append(0)
                    trimming['proposed_trimming_method'].append(None)
                else:
                    trimming['flag'].append(None)
                    trimming['start'].append(start_energy)
                    trimming['end'].append(end_energy)
                    trimming['percentage_trimmed'].append((duration_before-duration_after_energy)/duration_before)
                    trimming['proposed_trimming_method'].append('energy')
            elif end_praat/sr + 0.5 >= duration_before and start_praat/sr - 0.5 <= 0:
                _logger.info(f"Trimming {audio_path} with .5s padding will have no effect on audio length")
                trimming['flag'].append(None)
                trimming['start'].append(0)
                trimming['end'].append(duration_before)
                trimming['percentage_trimmed'].append(0)
                trimming['proposed_trimming_method'].append(None)
            else:
                trimming['flag'].append(None)
                trimming['start'].append(start_praat)
                trimming['end'].append(end_praat)
                trimming['percentage_trimmed'].append((duration_before-duration_after_praat)/duration_before)
                trimming['proposed_trimming_method'].append('praat')
            
            # Diarization check
            diarization_result = features['diarization']
            audio_obj.metadata['diarization'] = diarization_result
            speakers = []
            for line in diarization_result:
                speakers.append(line.speaker)
            num_speakers = len(set(speakers))
            speaker_ratio = primary_speaker_ratio_metric(audio_obj)
            
            diarization_qc['path'].append(audio_path)
            diarization_qc['subject'].append(subject)
            diarization_qc['task'].append(task)
            diarization_qc['num_speakers'].append(num_speakers)
            diarization_qc['proportion_primary_speaker'].append(speaker_ratio)
            if num_speakers == 0 and features['is_speech_task']:
                diarization_qc['flag'].append('no_speakers_found')
            elif num_speakers > 1 and speaker_ratio < .8:
                diarization_qc['flag'].append('no_primary_speaker_found')
            elif num_speakers > 2:
                diarization_qc['flag'].append('many_speakers_found')
            else:
                diarization_qc['flag'].append(None)

        trimming_df = pd.DataFrame(trimming)
        diarization_df = pd.DataFrame(diarization_qc)

        trimming_df.to_csv(quality_check_dir / "silence_removal.csv")
        diarization_df.to_csv(quality_check_dir / "diarization_check.csv")
    finally:
        if temp_dir_obj:
            temp_dir_obj.cleanup()


# Function to trim audio using Praat-parselmouth
def trim_audio_with_praat(y, sr):
    sound = parselmouth.Sound(y, sampling_frequency=sr)
    pitch = sound.to_pitch(time_step=0.01)
    voiced_flags = pitch.selected_array['frequency'] > 0
    time_stamps = pitch.xs()
    voiced_indices = np.where(voiced_flags)[0]
    if len(voiced_indices) == 0:
        return 0,0#0,y.shape[-1]#y
    start_sample = int(time_stamps[voiced_indices[0]] * sr)
    end_sample = int(time_stamps[voiced_indices[-1]] * sr)
    return start_sample, end_sample #y[start_sample:end_sample]

# Energy-based trimming function (fallback)
def trim_until_silent(y, sr, threshold_ratio=0.10):
    abs_y = np.abs(np.asarray(y))
    peak = np.max(abs_y)
    threshold = threshold_ratio * peak
    start = next((i for i, amp in enumerate(abs_y) if amp > threshold), 0)

    end = len(abs_y)
    for i in range(len(abs_y) - 1, start, -1):
        if abs_y[i] > threshold:
            end = i + 1
            break

    while True:
        remaining = abs_y[end:]
        if np.any(remaining > threshold):
            next_peak = np.argmax(remaining > threshold)
            end += next_peak + 1
        else:
            break
    return start, end#y[start:end]
