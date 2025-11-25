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

import parselmouth

RESAMPLE_RATE=16000

_logger = logging.getLogger(__name__)

def quality_control_wrapper(
    audio_paths,
    outdir,
    batch_size=8,
    num_cores=4,
    skip_windowing=False,
):

    evaluation_df = check_quality(
        audio_paths=audio_paths,
        output_dir=outdir,
        batch_size=batch_size,
        n_cores=num_cores,
        window_size_sec=0.025,
        step_size_sec=0.0125,
        skip_windowing=skip_windowing,
    )

    _logger.info(f"Result of check_quality: {evaluation_df}")

    df_path = outdir / "quality_control_results_non_windowed.csv"#"quality_control_results_all.json"

    evaluation_review_df = review_files(
        df_path=df_path,
        correlation_threshold= 0.99,
        output_dir=outdir,
    )

    _logger.info(f"Resutl of review files {evaluation_review_df}")

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
        'percentage_trimmed': []
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
        features = torch.load(audio_feature_path, weights_only=False, map_location=torch.device('cpu'))
        audio_obj = Audio(filepath=audio_path)

        audio_orig = downmix_audios_to_mono([audio_obj])[0]

        # Resample both audios to 16kHz
        audio_16k = resample_audios([audio_orig], RESAMPLE_RATE)[0]

        # Silence trimming
        y, sr = audio_16k.waveform.squeeze(), audio_16k.sampling_rate#librosa.load(audio_path, sr=None)
        _logger.info(f"{audio_path} has shape {y.shape} and sr of {sr}")
        duration_before = y.shape[-1] / sr

        # First trim: pitch-based (Praat), fallback to energy-based
        start_praat, end_praat = trim_audio_with_praat(y, sr) #y_trimmed = trim_audio_with_praat(y, sr)
        duration_after_praat = (end_praat-start_praat+1) / sr
        if duration_after_praat < 0.2 * duration_before:
            _logger.info(f"Praat trim too short ({duration_after_praat:.2f}s < 20% of {duration_before:.2f}s) for {audio_path} → retrying with energy-based trim")
            #y_trimmed = trim_until_silent(y, sr, threshold_ratio=0.10)
            _logger.info(f"Checking again: shape {y.shape} and sr of {sr}")
            start_energy, end_energy = trim_until_silent(y, sr, threshold_ratio=0.10)
            duration_after_energy = (end_energy-start_energy+1) / sr

            if duration_after_energy < 0.2 * duration_before:
                _logger.info(f"Energy trim too short ({duration_after_energy:.2f}s < 20% of {duration_before:.2f}s) for {audio_path} → Flagging audio for possible removal")
                trimming['path'].append(audio_path)
                trimming['subject'].append(subject)
                trimming['task'].append(task)
                trimming['flag'].append('large_proportion_silence')
                trimming['start'].append(start_energy)
                trimming['end'].append(end_energy)
                trimming['percentage_trimmed'].append((duration_before-duration_after_energy)/duration_before)
            elif end_energy/sr + 0.5 >= duration_before and start_energy/sr - 0.5 <= 0:
                _logger.info(f"Trimming {audio_path} with .5s padding will have no effect on audio length")
            else:
                trimming['path'].append(audio_path)
                trimming['subject'].append(subject)
                trimming['task'].append(task)
                trimming['flag'].append(None)
                trimming['start'].append(start_energy)
                trimming['end'].append(end_energy)
                trimming['percentage_trimmed'].append((duration_before-duration_after_energy)/duration_before)
        elif end_praat/sr + 0.5 >= duration_before and start_praat/sr - 0.5 <= 0:
            _logger.info(f"Trimming {audio_path} with .5s padding will have no effect on audio length")
        else:
            trimming['path'].append(audio_path)
            trimming['subject'].append(subject)
            trimming['task'].append(task)
            trimming['flag'].append(None)
            trimming['start'].append(start_praat)
            trimming['end'].append(end_praat)
            trimming['percentage_trimmed'].append((duration_before-duration_after_praat)/duration_before)
        
        # Diarization check
        diarization_result = features['diarization']
        audio_obj.metadata['diarization'] = diarization_result
        speakers = []
        for line in diarization_result:
            speakers.append(line.speaker)
        num_speakers = len(set(speakers))
        speaker_ratio = primary_speaker_ratio_metric(audio_obj)

        if num_speakers != 1:
            if num_speakers == 0 and features['is_speech_task']:
                diarization_qc['path'].append(audio_path)
                diarization_qc['subject'].append(subject)
                diarization_qc['task'].append(task)       
                diarization_qc['num_speakers'].append(num_speakers)
                diarization_qc['proportion_primary_speaker'].append(speaker_ratio)
                diarization_qc['flag'].append('no_speakers_found')
            elif num_speakers > 1 and speaker_ratio < .8:
                diarization_qc['path'].append(audio_path)
                diarization_qc['subject'].append(subject)
                diarization_qc['task'].append(task)       
                diarization_qc['num_speakers'].append(num_speakers)
                diarization_qc['proportion_primary_speaker'].append(speaker_ratio)
                diarization_qc['flag'].append('no_primary_speaker_found')
            elif num_speakers > 2:
                diarization_qc['path'].append(audio_path)
                diarization_qc['subject'].append(subject)
                diarization_qc['task'].append(task)       
                diarization_qc['num_speakers'].append(num_speakers)
                diarization_qc['proportion_primary_speaker'].append(speaker_ratio)
                diarization_qc['flag'].append('many_speakers_found')

    trimming_df = pd.DataFrame(trimming)
    diarization_df = pd.DataFrame(diarization_qc)

    trimming_df.to_csv(outdir / "silence_removal.csv")
    diarization_df.to_csv(outdir / "diarization_check.csv")


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
