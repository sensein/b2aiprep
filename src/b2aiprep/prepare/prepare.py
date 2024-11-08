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
import tarfile
import traceback
from collections import defaultdict
from pathlib import Path
from time import sleep
from typing import List

import numpy as np
import pydra
import torch
from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.features_extraction.opensmile import (
    extract_opensmile_features_from_audios,
)
from senselab.audio.tasks.features_extraction.torchaudio import (
    extract_mel_filter_bank_from_audios,
    extract_mfcc_from_audios,
    extract_spectrogram_from_audios,
)
from senselab.audio.tasks.preprocessing.preprocessing import resample_audios
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.language import Language
from senselab.utils.data_structures.model import HFModel
from tqdm import tqdm

from b2aiprep.prepare.bids import get_audio_paths, batch_elements

SUBJECT_ID = "sub"
SESSION_ID = "ses"
AUDIO_ID = "audio"
RESAMPLE_RATE = 16000

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    from senselab.audio.tasks.speaker_embeddings.api import (
        extract_speaker_embeddings_from_audios,
    )
except Exception as e:
    _logger.error(
        f"Speaker embeddings import error occurred. \
                  This is likely because this device is not connected to wifi. {e}"
    )
try:
    from senselab.audio.tasks.speech_to_text.api import transcribe_audios
except Exception as e:
    _logger.error(
        f"Speech to text import error occurred. \
                  This is likely because this device is not connected to wifi. {e}"
    )

def transcribe(audio, features, transcription_model_size):
    """
    Transcribes an audio input using a specified transcription model size.

    Args:
        audio (str): Path to the audio file to be transcribed.
        features (list): List of features required for the transcription process.
        transcription_model_size (str): Size of the transcription model to use
        (e.g., 'small', 'medium', 'large').

    Returns:
        str: The transcription of the audio file.
    """
    speech_to_text_model = HFModel(path_or_uri=f"openai/whisper-{transcription_model_size}")
    device = DeviceType.CPU
    language = Language.model_validate({"language_code": "en"})
    return transcribe_audios(
        audios=[audio], model=speech_to_text_model, device=device, language=language
    )[0]


@pydra.mark.task
def wav_to_features(wav_paths: List[Path], transcription_model_size: str, with_sensitive: bool):
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
    for wav_path in tqdm(wav_paths, total=len(wav_paths), desc="Extracting features"):
        wav_path = Path(wav_path)

        logging.disable(logging.ERROR)
        audio = Audio.from_filepath(str(wav_path))
        audio = resample_audios([audio], resample_rate=RESAMPLE_RATE)[0]

        features = {}
        try:
            features["speaker_embedding"] = extract_speaker_embeddings_from_audios([audio])[0]
        except Exception as e:
            _logger.error(
                f"Speaker embeddings: An error occurred with extracting speaker embeddings. {e}"
            )
        features["specgram"] = extract_spectrogram_from_audios([audio])[0]
        features["melfilterbank"] = extract_mel_filter_bank_from_audios([audio])[0]
        features["mfcc"] = extract_mfcc_from_audios([audio])[0]
        features["sample_rate"] = audio.sampling_rate
        features["opensmile"] = extract_opensmile_features_from_audios([audio])[0]
        if with_sensitive:
            try:
                features["transcription"] = transcribe(audio, features, transcription_model_size)
            except Exception:
                sleep(0.1)
                try:
                    features["transcription"] = transcribe(
                        audio, features, transcription_model_size
                    )
                except Exception as e:
                    logging.disable(logging.NOTSET)
                    _logger.error(f"Transcription: An error occurred with transcription: {e}")
                    _logger.error(traceback.print_exc())
        logging.disable(logging.NOTSET)
        _logger.setLevel(logging.INFO)

        # Define the save directory for features
        audio_dir = wav_path.parent
        features_dir = audio_dir.parent / "audio"
        features_dir.mkdir(exist_ok=True)

        # Save each feature in a separate file
        for feature_name, feature_value in features.items():
            file_extension = "pt"
            if feature_name == "transcription":
                file_extension = "txt"
            save_path = features_dir / f"{wav_path.stem}_{feature_name}.{file_extension}"
            if file_extension == "pt":
                torch.save(feature_value, save_path)
            else:
                with open(save_path, "w", encoding="utf-8") as text_file:
                    text_file.write(feature_value.text)
        all_features.append(features)

        # debug
        break

    return all_features

def extract_features_workflow(
    bids_dir_path: Path,
    transcription_model_size: str = "tiny",
    n_cores: int = 8,
    with_sensitive: bool = True,
):
    """Run a Pydra workflow to extract audio features from BIDS-like directory.

    This function initializes a Pydra workflow that processes a BIDS-like
    directory structure to extract features from .wav audio files. It retrieves
    the paths to the audio files and applies the wav_to_features to each.

    Args:
      bids_dir_path:
        The root directory of the BIDS dataset.
      remove:
        Whether to remove temporary files created during processing. Default is True.

    Returns:
      pydra.Workflow:
        The Pydra workflow object with the extracted features and audio paths as outputs.
    """
    # Get paths to every audio file.
    audio_paths = get_audio_paths(bids_dir_path=bids_dir_path)
    if n_cores > 1:
        audio_paths = batch_elements(audio_paths, n_cores)

    # Initialize the Pydra workflow.
    ef_wf = pydra.Workflow(
        name="ef_wf", input_spec=["audio_paths"], audio_paths=audio_paths, cache_dir=None
    )

    # Run wav_to_features for each audio file
    ef_wf.add(
        wav_to_features(
            name="features",
            wav_paths=ef_wf.lzin.audio_paths,
            transcription_model_size=transcription_model_size,
            with_sensitive=with_sensitive,
        ).split("wav_paths", wav_paths=ef_wf.lzin.audio_paths)
    )

    ef_wf.set_output({"features": ef_wf.features.lzout.out})

    with pydra.Submitter(plugin="cf") as run:
        run(ef_wf)

    return ef_wf

def extract_features_serially(
    bids_dir_path: Path,
    transcription_model_size: str,
    n_cores: int,
    with_sensitive: bool,
):
    """Provides a non-Pydra implementation of extract_features_workflow"""
    audio_paths = get_audio_paths(
        name="audio_paths", bids_dir_path=bids_dir_path
    )().output.out
    all_features = []
    all_features.append(
        wav_to_features(
            wav_paths=audio_paths,
            transcription_model_size=transcription_model_size,
            with_sensitive=with_sensitive,
        )().output.out
    )

def validate_bids_data(
    bids_dir_path: Path,
    fix: bool = True,
    transcription_model_size: str = "medium",
) -> None:
    """Scans BIDS audio data and verifies that all expected features are present."""
    _logger.info("Scanning for features in BIDS directory.")
    # TODO: add a check to see if the audio feature extraction is complete
    # before proceeding to the next step
    # can verify features are generated for each audio_dir by looking for .pt files
    # in audio_dir.parent / "audio"
    audio_paths = get_audio_paths(bids_dir_path)
    audio_to_reprocess = defaultdict(list)
    features = (
        "speaker_embedding",
        "specgram",
        "melfilterbank",
        "mfcc",
        "sample_rate",
        "opensmile",
    )
    for audio_path in audio_paths:
        audio_path = Path(audio_path)
        audio_dir = audio_path.parent
        features_dir = audio_dir.parent / "audio"
        for feat_name in features:
            if features_dir.joinpath(f"{audio_path.stem}_{feat_name}.pt").exists() is False:
                audio_to_reprocess[audio_path].append(feat_name)

        # also check for transcription
        if features_dir.joinpath(f"{audio_path.stem}_transcription.txt").exists() is False:
            audio_to_reprocess[audio_path].append("transcription")

    if len(audio_to_reprocess) > 0:
        _logger.info(
            f"Missing features for {len(audio_to_reprocess)} / {len(audio_paths)} audio files"
        )
    else:
        _logger.info("All audio files have been processed and all feature files are present.")
        return

    if not fix:
        return

    feature_extraction_fcns = {
        "speaker_embedding": extract_speaker_embeddings_from_audios,
        "specgram": extract_spectrogram_from_audios,
        "melfilterbank": extract_mel_filter_bank_from_audios,
        "mfcc": extract_mfcc_from_audios,
        "opensmile": extract_opensmile_features_from_audios,
    }
    for audio_path, missing_feats in tqdm(
        audio_to_reprocess.items(), total=len(audio_to_reprocess), desc="Reprocessing audio files"
    ):
        audio_dir = audio_path.parent
        features_dir = audio_dir.parent / "audio"
        features_dir.mkdir(exist_ok=True)
        for feat_name in missing_feats:
            audio = Audio.from_filepath(str(audio_path))
            audio = resample_audios([audio], resample_rate=RESAMPLE_RATE)[0]

            file_extension = "pt"
            if feat_name == "sample_rate":
                feature_value = audio.sampling_rate
            elif feat_name in feature_extraction_fcns:
                feature_value = feature_extraction_fcns[feat_name]([audio])[0]
            elif feat_name == "transcription":
                language = Language.model_validate({"language_code": "en"})
                speech_to_text_model = HFModel(
                    path_or_uri=f"openai/whisper-{transcription_model_size}"
                )
                feature_value = transcribe_audios(
                    audios=[audio], model=speech_to_text_model, language=language
                )[0]
                file_extension = "txt"
            else:
                _logger.warning(f"Unsupported feature: {feat_name}")
                continue

            save_path = features_dir / f"{audio_path.stem}_{feat_name}.{file_extension}"
            if file_extension == "pt":
                torch.save(feature_value, save_path)
            else:
                with open(save_path, "w", encoding="utf-8") as text_file:
                    text_file.write(feature_value.text)

    _logger.info("Process completed.")
