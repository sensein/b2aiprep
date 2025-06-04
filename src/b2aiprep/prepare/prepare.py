"""
Organizes data, extracts features, and bundles everything together in an
easily distributable format for the Bridge2AI Summer School.

This script performs the following steps:
1. Organizes RedCap data and audio files into a BIDS-like directory structure.
2. Extracts audio features from the organized data using a Pydra workflow.
3. Bundles the processed data into a .tar file with gzip compression.

Feature extraction is parallelized using Pydra.

Usage:
    b2aiprep-cli prepsummerdata \
       [path to RedCap CSV] \
       [path to Wasabi export directory] \
       [desired path to BIDS output] \
       [desired output path for .tar file]

    python3 b2aiprep/src/b2aiprep/bids_like_data.py \
        --redcap_csv_path [path to RedCap CSV] \
        --audio_dir_path  [path to Wasabi export directory] \
        --bids_dir_path [desired path to BIDS output] \
        --tar_file_path [desired output path for .tar file]

Functions:
    - wav_to_features: Extracts features from a single audio file.
    - get_audio_paths: Retrieves all .wav audio file paths from a BIDS-like
        directory structure.
    - extract_features_workflow: Runs a Pydra workflow to extract audio
        features from BIDS-like directory.
    - bundle_data: Saves data bundle as a tar file with gzip compression.
    - prepare_bids_like_data: Organizes and processes Bridge2AI data for
        distribution.
    - parse_arguments: Parses command line arguments for processing audio data.

"""

import logging
import os
import traceback
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import pydra
import torch
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.features_extraction.api import extract_features_from_audios
from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios
from senselab.audio.tasks.speaker_embeddings import (
    extract_speaker_embeddings_from_audios,
)
from senselab.audio.tasks.speech_to_text import transcribe_audios
from senselab.utils.data_structures import (
    DeviceType,
    HFModel,
    Language,
    SpeechBrainModel,
)
from tqdm import tqdm

from b2aiprep.prepare.constants import (
    AUDIO_FILESTEMS_TO_REMOVE,
    FEATURE_EXTRACTION_SPEECH_RATE,
    FEATURE_EXTRACTION_DURATION,
    FEATURE_EXTRACTION_PITCH_AND_INTENSITY,
    FEATURE_EXTRACTION_HARMONIC_DESCRIPTORS,
    FEATURE_EXTRACTION_FORMANTS,
    FEATURE_EXTRACTION_SPECTRAL_MOMENTS,
    FEATURE_EXTRACTION_JITTER, FEATURE_EXTRACTION_SHIMMER,
)

from b2aiprep.prepare.bids import get_audio_paths
from b2aiprep.prepare.constants import SPEECH_TASKS
from b2aiprep.prepare.utils import retry

SUBJECT_ID = "sub"
SESSION_ID = "ses"
AUDIO_ID = "audio"
RESAMPLE_RATE = 16000
SPECTROGRAM_SHAPE = 201
# Parcelmouth feature groupings

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def extract_single(
    wav_path: str | os.PathLike,
    transcription_model_size: str,
    with_sensitive: bool,
    overwrite: bool = False,
    device: DeviceType = DeviceType.CPU,
    update: bool = False,
):
    wav_path = Path(wav_path)
    # Define the save directory for features
    audio_dir = wav_path.parent
    features_dir = audio_dir.parent / "audio"
    features_dir.mkdir(exist_ok=True)
    save_to = features_dir / f"{wav_path.stem}_features.pt"

    preloaded_pt = {}
    if save_to.exists() and not overwrite:
        if not update:
            _logger.info(f"{save_to} already exists. Skipping.")
            return save_to
        else:
            preloaded_pt = torch.load(save_to, map_location='cpu', weights_only=False)

    logging.disable(logging.ERROR)
    # Load audio
    audio_orig = Audio(filepath=wav_path)
    # Downmix to mono
    audio_orig = downmix_audios_to_mono([audio_orig])[0]

    # Resample both audios to 16kHz
    audio_16k = resample_audios([audio_orig], RESAMPLE_RATE)[0]

    win_length = 25
    hop_length = 10

    is_speech_task = any([v.replace(" ", "-") in wav_path.name for v in SPEECH_TASKS])

    opensmile = True
    torchaudio_squim = True
    parsel_mouth_config = {
        "time_step": hop_length / 1000,
        "window_length": win_length / 1000,
        "plugin": "serial",
    }
    torch_config = {
        "freq_low": 80,
        "freq_high": 500,
        "n_fft": (win_length * audio_16k.sampling_rate) // 1000,
        "n_mels": 60,
        "n_mfcc": 60,
        "win_length": (win_length * audio_16k.sampling_rate) // 1000,
        "hop_length": (hop_length * audio_16k.sampling_rate) // 1000,
        "plugin": "serial",
    }
    # Extract features
    if update and not overwrite:
        if "opensmile" in preloaded_pt:
            opensmile = False
        if "praat_parselmouth" in preloaded_pt:
            praat_features = list(preloaded_pt["praat_parselmouth"].keys())
            parsel_mouth_config = {
                "time_step": hop_length / 1000,
                "window_length": win_length / 1000,
                "plugin": "serial",
                "speech_rate": False,
                "intensity_descriptors": False,
                "harmonicity_descriptors": False,
                "formants": False,
                "spectral_moments": False,
                "pitch": False,
                "slope_tilt": False,
                "cpp_descriptors": False,
                "duration": False,
                "jitter": False,
                "shimmer": False,
            }
            if not all(features in praat_features for features in FEATURE_EXTRACTION_SPEECH_RATE):
                parsel_mouth_config["speech_rate"] = True

            if not all(features in praat_features for features in FEATURE_EXTRACTION_DURATION):
                parsel_mouth_config["duration"] = True

            if not all(features in praat_features for features in FEATURE_EXTRACTION_PITCH_AND_INTENSITY):
                parsel_mouth_config["pitch"] = True
                parsel_mouth_config["intensity_descriptors"] = True

            if not all(features in praat_features for features in FEATURE_EXTRACTION_HARMONIC_DESCRIPTORS):
                parsel_mouth_config["cpp_descriptors"] = True
                parsel_mouth_config["slope_tilt"] = True
                parsel_mouth_config["harmonicity_descriptors"] = True

            if not all(features in praat_features for features in FEATURE_EXTRACTION_FORMANTS):
                parsel_mouth_config["formants"] = True

            if not all(features in praat_features for features in FEATURE_EXTRACTION_SPECTRAL_MOMENTS):
                parsel_mouth_config["spectral_moments"] = True

            if not all(features in praat_features for features in FEATURE_EXTRACTION_JITTER):
                parsel_mouth_config["jitter"] = True

            if not all(features in praat_features for features in FEATURE_EXTRACTION_SHIMMER):
                parsel_mouth_config["shimmer"] = True
        if "torchaudio" in preloaded_pt:
            spectrogram = preloaded_pt["torchaudio"]["spectrogram"]
            if spectrogram.shape[0] == SPECTROGRAM_SHAPE:
                torch_config = False
        if "torchaudio_squim" in preloaded_pt:
            torchaudio_squim = False

    features = extract_features_from_audios(
        audios=[audio_16k],
        opensmile=opensmile,
        parselmouth=parsel_mouth_config,
        torchaudio=torch_config,
        torchaudio_squim=torchaudio_squim,
    ).pop()

    if update and not overwrite:
        # if squim was not in previous release
        if "torchaudio_squim" not in preloaded_pt and "torchaudio_squim" in features :
            preloaded_pt["torchaudio_squim"] = features["torchaudio_squim"]
        if "opensmile" not in preloaded_pt and "opensmile" in features:
            preloaded_pt["opensmile"] = features["opensmile"]
        if "torchaudio" not in preloaded_pt and "torchaudio" in features:
            preloaded_pt["torchaudio"] = features["torchaudio"]
        if "praat_parselmouth" not in preloaded_pt and "praat_parselmouth" in features:
            preloaded_pt["praat_parselmouth"] = features["praat_parselmouth"]

        # case where the first generation was using n_fft = 1024 and we need to replace
        if preloaded_pt["torchaudio"]["spectrogram"].shape[0] != SPECTROGRAM_SHAPE:
            preloaded_pt["torchaudio"] = features["torchaudio"]

        # combine to have all features
        if "praat_parselmouth" in features and "praat_parselmouth" in preloaded_pt:
            preloaded_pt["praat_parselmouth"] = {
                **preloaded_pt["praat_parselmouth"], **features["praat_parselmouth"]}
        features = preloaded_pt

    if not update:
        features["parselmouth_config"] = parsel_mouth_config
        features["torch_config"] = torch_config
        features["is_speech_task"] = is_speech_task
        features["sample_rate"] = audio_16k.sampling_rate
        features["duration"] = len(audio_16k.waveform) / audio_16k.sampling_rate
        features["sensitive_features"] = None

    if with_sensitive:
        features["audio_path"] = wav_path
        try:
            model = SpeechBrainModel(
                path_or_uri="speechbrain/spkrec-ecapa-voxceleb", revision="main"
            )
            features["speaker_embedding"] = extract_speaker_embeddings_from_audios(
                [audio_16k], model, device
            )[0]
        except Exception as e:
            features["speaker_embedding"] = None
            _logger.error(
                f"Speaker embeddings: An error occurred with extracting speaker embeddings. {e}"
            )
        features["transcription"] = None
        try:
            speech_to_text_model = HFModel(
                path_or_uri=f"openai/whisper-{transcription_model_size}", revision="main"
            )
            language = Language.model_validate({"language_code": "en"})
            transcription = retry(transcribe_audios)(
                [audio_16k], model=speech_to_text_model, device=device, language=language
            )
            features["transcription"] = transcription[0]
        except Exception as e:
            logging.disable(logging.NOTSET)
            _logger.error(f"Transcription: An error occurred with transcription: {e}")
            _logger.error(traceback.print_exc())
        features["sensitive_features"] = ["audio_path", "speaker_embedding", "transcription"]
    logging.disable(logging.NOTSET)
    _logger.setLevel(logging.INFO)

    torch.save(features, save_to)
    return save_to


def wav_to_features(
    wav_paths: List[str | os.PathLike],
    transcription_model_size: str,
    with_sensitive: bool,
    overwrite: bool = False,
    device: DeviceType = DeviceType.CPU,
) -> List[str | os.PathLike]:
    """Extract features from a list of audio files.

    Extracts various audio features from .wav files
    using the Audio class and feature extraction functions.

    Args:
      wav_path:
        The file path to the .wav audio file.

    Returns:
      A dictionary mapping feature names to their extracted values.
    """
    all_features = []
    if len(wav_paths) > 1:
        _logger.info(f"Number of audio files: {len(wav_paths)}")
        for wav_path in tqdm(wav_paths, total=len(wav_paths), desc="Extracting features"):
            save_path = extract_single(
                wav_path,
                transcription_model_size=transcription_model_size,
                with_sensitive=with_sensitive,
                device=device,
                overwrite=overwrite,
            )
            all_features.append(save_path)
    else:
        save_path = extract_single(
            wav_paths[0],
            transcription_model_size=transcription_model_size,
            with_sensitive=with_sensitive,
            device=device,
            overwrite=overwrite,
        )
        all_features.append(save_path)
    return all_features


def generate_features_wrapper(
    bids_path,
    transcription_model_size,
    n_cores,
    with_sensitive,
    overwrite,
    cache,
    address,
    percentile,
    subject_id,
    update=False,
    is_sequential=False

):
    if is_sequential:
        extract_features_sequentially(
            bids_path,
            transcription_model_size=transcription_model_size,
            with_sensitive=with_sensitive,
            update=update,
        )
    else:
        extract_features_workflow(
            bids_path,
            transcription_model_size=transcription_model_size,
            n_cores=n_cores,
            with_sensitive=with_sensitive,
            overwrite=overwrite,
            cache_dir=cache,
            plugin="dask" if address is not None else "cf",
            address=address,
            percentile=percentile,
            subject_id=subject_id,
            update=update,
        )


def extract_features_sequentially(
    bids_dir_path: Path,
    transcription_model_size: str = "tiny",
    with_sensitive: bool = True,
    update: bool = False
):
    audio_paths = get_audio_paths(bids_dir_path=bids_dir_path)
    audio_paths = sorted(audio_paths, key=lambda wave_file: wave_file["size"])

    for audio_file in audio_paths:
        extract_single(
            wav_path=audio_file["path"],
            transcription_model_size=transcription_model_size,
            with_sensitive=with_sensitive,
            update=update)


def extract_features_workflow(
    bids_dir_path: Path,
    transcription_model_size: str = "tiny",
    with_sensitive: bool = True,
    overwrite: bool = False,
    n_cores: int = 8,
    plugin: str = "cf",
    address: str = None,
    cache_dir: str | os.PathLike = None,
    percentile: int = 100,
    subject_id: str = None,
    update: bool = False
):
    """Run a Pydra workflow to extract audio features from BIDS-like directory.

    This function initializes a Pydra workflow that processes a BIDS-like
    directory structure to extract features from .wav audio files. It retrieves
    the paths to the audio files and applies the wav_to_features to each.

    Args:
      bids_dir_path:
        The root directory of the BIDS dataset.
      transcription_model_size:
        The size of the Whisper model to use for transcription.
      n_cores:
        The number of cores to use for parallel processing.
      with_sensitive:
        Whether to extract sensitive features such as speaker embeddings and transcriptions.
      plugin:
        The Pydra plugin to use for parallel processing.
      cache_dir:
        The directory to use for caching intermediate results.

    Returns:
      pydra.Task:
        The Pydra Task object with the extracted feature paths.
    """
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set
        pass

    if n_cores > 1:
        plugin_args: dict = {"n_procs": n_cores} if plugin == "cf" else {}
    else:
        plugin = "serial"
        plugin_args = {}
    # Get paths to every audio file.
    audio_paths = get_audio_paths(bids_dir_path=bids_dir_path)
    df = pd.DataFrame(audio_paths)
    if "path" not in df.columns:
        _logger.warning("No audio files found in the BIDS directory.")
        return
    if "size" in df.columns:
        df = df[df["size"] <= np.percentile(df["size"].values, percentile)]
    if subject_id is not None:
        df = df[df["subject"] == subject_id]
    # randomize to distribute sizes
    df = df.sample(frac=1).reset_index(drop=True)
    audio_paths = df.path.values.tolist()
    _logger.info(f"Number of audio files: {len(audio_paths)}")
    # Run the task
    extract_task = pydra.mark.task(extract_single)(
        transcription_model_size=transcription_model_size,
        with_sensitive=with_sensitive,
        overwrite=overwrite,
        cache_dir=cache_dir,
        update=update
    )
    extract_task.split("wav_path", wav_path=audio_paths)
    if plugin == "dask":
        plugin_args = {"address": address}
    with pydra.Submitter(plugin=plugin, **plugin_args) as run:
        run(extract_task)
    return extract_task


def validate_bids_data(
    bids_dir_path: Path,
    fix: bool = True,
    transcription_model_size: str = "tiny",
    with_sensitive: bool = True,
    n_cores: int = 8,
    plugin: str = "cf",
    address: str = None,
    cache_dir: str | os.PathLike = None,
) -> None:
    """Scans BIDS audio data and verifies that all expected features are present."""
    _logger.info("Scanning for features in BIDS directory.")
    # TODO: add a check to see if the audio feature extraction is complete
    # before proceeding to the next step
    # can verify features are generated for each audio_dir by looking for .pt files
    # in audio_dir.parent / "audio"
    audio_paths = get_audio_paths(bids_dir_path)
    df = pd.DataFrame(audio_paths)
    audio_to_reprocess = []
    for audio_path in df.path.values.tolist():
        audio_path = Path(audio_path)
        audio_dir = audio_path.parent
        features_dir = audio_dir.parent / "audio"
        if features_dir.joinpath(f"{audio_path.stem}_features.pt").exists() is False:
            audio_to_reprocess.append(audio_path)

    if len(audio_to_reprocess) > 0:
        _logger.info(
            f"Missing features for {len(audio_to_reprocess)} / {len(audio_paths)} audio files"
        )
    else:
        _logger.info("All audio files have been processed and all feature files are present.")
        return

    if not fix:
        return

    if n_cores > 1:
        plugin_args: dict = {"n_procs": n_cores} if plugin == "cf" else {}
    else:
        plugin = "serial"
        plugin_args = {}
    extract_task = wav_to_features(
        transcription_model_size=transcription_model_size,
        with_sensitive=with_sensitive,
    )
    if n_cores > 1:
        extract_task.split("wav_paths", wav_paths=[[x] for x in audio_to_reprocess])
    else:
        extract_task.inputs.wav_paths = audio_to_reprocess
    if plugin == "dask":
        plugin_args = {"address": address}
    with pydra.Submitter(plugin=plugin, **plugin_args) as run:
        run(extract_task)
    with pydra.Submitter(plugin="cf") as run:
        run(extract_task)
    _logger.info("Process completed.")

def is_audio_sensitive(filepath: Path) -> bool:
    return filepath.stem in AUDIO_FILESTEMS_TO_REMOVE

def filter_audio_paths(audio_paths: List[Path]) -> List[Path]:
    """Filter audio paths to remove audio check and sensitive audio files."""
    
    # remove audio-check
    n_audio = len(audio_paths)
    audio_paths = [
        x for x in audio_paths if 'audio-check' not in x.stem.split("_")[2].lower()
    ]
    if len(audio_paths) < n_audio:
        _logger.info(
            f"Removed {n_audio - len(audio_paths)} audio check recordings."
        )

    n_audio = len(audio_paths)
    audio_paths = [
        x for x in audio_paths if not is_audio_sensitive(x)
    ]
    if len(audio_paths) < n_audio:
        _logger.info(
            f"Removed {n_audio - len(audio_paths)} records due to sensitive audio."
        )
    
    return audio_paths

def reduce_id_length(x):
    """Reduce length of ID."""
    if pd.isnull(x):
        return x
    return x.split('-')[0]

def reduce_length_of_id(df: pd.DataFrame, id_name: str) -> pd.DataFrame:
    """Reduce length of ID in the dataframe."""
    for c in df.columns:
        if c == id_name or c.endswith(id_name):
            df[c] = df[c].apply(reduce_id_length)
    
    return df

def get_value_from_metadata(metadata: dict, linkid: str, endswith: bool = False) -> str:
    for item in metadata['item']:
        if 'linkId' not in item:
            continue
        if item['linkId'] == linkid:
            return item['answer'][0]['valueString']
        if endswith and item['linkId'].endswith(linkid):
            return item['answer'][0]['valueString']
    return None

def update_metadata_record_and_session_id(metadata: dict):
    for item in metadata['item']:
        if 'linkId' not in item:
            continue

        if item['linkId'] == 'record_id':
            record_id = item['answer'][0]['valueString']
            item['answer'][0]['valueString'] = reduce_id_length(record_id)
            # rename to participant_id
            item['linkId'] = 'participant_id'
        elif (item['linkId'] == 'session_id') or (item['linkId'].endswith('_session_id')):
            item['answer'][0]['valueString'] = reduce_id_length(
                item['answer'][0]['valueString']
            )