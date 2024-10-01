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

from collections import defaultdict
import logging
import os
import tarfile
from pathlib import Path
from time import sleep
from typing import List

import numpy as np
from pydantic import ValidationError
import pydra
import torch
from senselab.audio.data_structures.audio import Audio
from senselab.utils.data_structures.language import Language
from senselab.audio.tasks.features_extraction.opensmile import (
    extract_opensmile_features_from_audios,
)
from senselab.audio.tasks.features_extraction.torchaudio import (
    extract_mel_filter_bank_from_audios,
    extract_mfcc_from_audios,
    extract_spectrogram_from_audios,
)
from senselab.audio.tasks.preprocessing.preprocessing import resample_audios
from senselab.audio.tasks.speaker_embeddings.api import (
    extract_speaker_embeddings_from_audios,
)
from senselab.audio.tasks.speech_to_text.api import transcribe_audios
from senselab.utils.data_structures.model import HFModel
from tqdm import tqdm

from b2aiprep.prepare.bids_like_data import redcap_to_bids
from b2aiprep.prepare.utils import copy_package_resource, remove_files_by_pattern

SUBJECT_ID = "sub"
SESSION_ID = "ses"
AUDIO_ID = "audio"
RESAMPLE_RATE = 16000

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def copy_phenotype_files(template_package, bids_dir_path):
    copy_package_resource(template_package, "phenotype", bids_dir_path)
    phenotype_path = Path(bids_dir_path) / Path("phenotype")
    remove_files_by_pattern(phenotype_path, "<measurement_tool_name>*")


def initialize_data_directory(bids_dir_path: str) -> None:
    """Initializes the data directory using the template.

    Args:
        bids_dir_path (str): The path to the BIDS directory where the data should be initialized.

    Returns:
        None
    """
    if not os.path.exists(bids_dir_path):
        os.makedirs(bids_dir_path)
        _logger.info(f"Created directory: {bids_dir_path}")

    template_package = "b2aiprep.prepare.resources.b2ai-data-bids-like-template"
    copy_package_resource(template_package, "CHANGES.md", bids_dir_path)
    copy_package_resource(template_package, "README.md", bids_dir_path)
    copy_package_resource(template_package, "dataset_description.json", bids_dir_path)
    copy_package_resource(template_package, "participants.json", bids_dir_path)
    copy_phenotype_files(template_package, bids_dir_path)


@pydra.mark.task
def wav_to_features(wav_path: Path, transcription_model_size: str, with_sensitive: bool) -> Path:
    """Extract features from a single audio file.

    Extracts various audio features from a .wav file at the specified
    path using the Audio class and feature extraction functions.

    Args:
      wav_path:
        The file path to the .wav audio file.

    Returns:
      A dictionary mapping feature names to their extracted values.
    """
    _logger.info(wav_path)
    logging.disable(logging.ERROR)
    audio = Audio.from_filepath(str(wav_path))
    audio = resample_audios([audio], resample_rate=RESAMPLE_RATE)[0]

    features = {}
    features["speaker_embedding"] = extract_speaker_embeddings_from_audios([audio])[0]
    features["specgram"] = extract_spectrogram_from_audios([audio])[0]
    features["melfilterbank"] = extract_mel_filter_bank_from_audios([audio])[0]
    features["mfcc"] = extract_mfcc_from_audios([audio])[0]
    features["sample_rate"] = audio.sampling_rate
    features["opensmile"] = extract_opensmile_features_from_audios([audio])[0]
    language = Language.model_validate({"language_code": "en"}) # used for transcription
    if with_sensitive:
        try:
            speech_to_text_model = HFModel(path_or_uri=f"openai/whisper-{transcription_model_size}")
        except ValidationError as e:
            # just try it again, unsure why this error is raised
            sleep(1)
            speech_to_text_model = HFModel(path_or_uri=f"openai/whisper-{transcription_model_size}")
        features["transcription"] = transcribe_audios(
            audios=[audio], model=speech_to_text_model, language=language
        )[0]
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

    return features_dir

def get_audio_paths(bids_dir_path):
    """Retrieve all .wav audio file paths from a BIDS-like directory structure.

    This function traverses the specified BIDS directory, collecting paths to
    .wav audio files from all subject and session directories that match the
    expected naming conventions.

    Args:
      bids_dir_path:
        The root directory of the BIDS dataset.

    Returns:
      list of str:
        A list of file paths to .wav audio files within the BIDS directory.
    """
    audio_paths = []

    # Iterate over each subject directory.
    for sub_file in os.listdir(bids_dir_path):
        subject_dir_path = os.path.join(bids_dir_path, sub_file)
        if sub_file.startswith(SUBJECT_ID) and os.path.isdir(subject_dir_path):

            # Iterate over each session directory within a subject.
            for session_dir_path in os.listdir(subject_dir_path):
                audio_path = os.path.join(subject_dir_path, session_dir_path, AUDIO_ID)

                if session_dir_path.startswith(SESSION_ID) and os.path.isdir(audio_path):
                    # _logger.info(audio_path)
                    # Iterate over each audio file in the voice directory.
                    for audio_file in os.listdir(audio_path):
                        if audio_file.endswith(".wav"):
                            audio_file_path = os.path.join(audio_path, audio_file)
                            audio_paths.append(audio_file_path)

    return audio_paths


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

    # Initialize the Pydra workflow.
    ef_wf = pydra.Workflow(
        name="ef_wf", input_spec=["audio_paths"], audio_paths=audio_paths, cache_dir=None
    )

    # Run wav_to_features for each audio file.
    ef_wf.add(
        wav_to_features(
            name="features",
            wav_path=ef_wf.lzin.audio_paths,
            transcription_model_size=transcription_model_size,
            with_sensitive=with_sensitive,
        ).split("wav_path", wav_path=ef_wf.lzin.audio_paths)
    )

    ef_wf.set_output(
        {"audio_paths": ef_wf.lzin.audio_paths}
    )

    with pydra.Submitter(plugin="cf") as run:
        run(ef_wf)

    return ef_wf


def _extract_features_workflow(
    bids_dir_path: Path, transcription_model_size: str = "tiny", n_cores: int = 1, with_sensitive: bool = True
):
    """Provides a non-Pydra implementation of _extract_features_workflow"""
    audio_paths = get_audio_paths(bids_dir_path=bids_dir_path)
    audio_paths = audio_paths[4000+5570+3000:]
    out_paths = []
    for path in tqdm(audio_paths, total=len(audio_paths), desc='Extracting features'):
        wav_to_features(
            wav_path=path, transcription_model_size=transcription_model_size, with_sensitive=True
        )()
        out_paths.extend(path)

    return out_paths


def bundle_data(source_directory: str, save_path: str) -> None:
    """Saves data bundle as a tar file with gzip compression.

    Args:
      source_directory:
        The directory containing the data to be bundled.
      save_path:
        The file path where the tar.gz file will be saved.
    """
    with tarfile.open(save_path, "w:gz") as tar:
        tar.add(source_directory, arcname=os.path.basename(source_directory))


def prepare_bids_like_data(
    redcap_csv_path: Path,
    audio_dir_path: Path,
    bids_dir_path: Path,
    tar_file_path: Path,
    transcription_model_size: str,
    n_cores: int,
    with_sensitive: bool,
) -> None:
    """Organizes and processes Bridge2AI data for distribution.

    This function organizes the data from RedCap and the audio files from
    Wasabi into a BIDS-like directory structure, extracts audio features from
    the organized data, and bundles the processed data into a .tar file with
    gzip compression.

    Args:
      redcap_csv_path:
        The file path to the REDCap CSV file.
      audio_dir_path:
        The directory path containing audio files.
      bids_dir_path:
        The directory path for the BIDS dataset.
      tar_file_path:
        The file path where the .tar.gz file will be saved.
    """
    # initialize_data_directory(bids_dir_path)

    _logger.info("Organizing data into BIDS-like directory structure...")
    # redcap_to_bids(redcap_csv_path, bids_dir_path, audio_dir_path)
    _logger.info("Data organization complete.")

    _logger.info("Beginning audio feature extraction...")
    _extract_features_workflow(
        bids_dir_path,
        transcription_model_size=transcription_model_size,
        n_cores=n_cores,
        with_sensitive=with_sensitive,
    )

    _logger.info("Audio feature extraction complete.")

    # Below code checks to see if we have all the expected feature/transcript files.
    audio_paths = get_audio_paths(bids_dir_path)
    missing_features = []
    missing_transcriptions = []
    for audio_path in audio_paths:
        audio_path = Path(audio_path)
        audio_dir = audio_path.parent
        features_dir = audio_dir.parent / "audio"
        feature_files = [file for file in features_dir.glob(f'{audio_path.stem}*.pt')]
        if len(feature_files) == 0:
            missing_features.append(audio_path)
        feature_files = [file for file in features_dir.glob(f'{audio_path.stem}*.txt')]
        if len(feature_files) == 0:
            missing_transcriptions.append(audio_path)
    if len(missing_transcriptions) > 0:
        _logger.warning(f"Missing transcriptions for {len(missing_transcriptions)} / {len(audio_paths)} audio files")
    if len(missing_features) > 0:
        _logger.warning(f"Missing features for {len(missing_features)} / {len(audio_paths)} audio files")
    else:
        _logger.info("All audio files have been processed.")

    _logger.info("Saving .tar file with processed data...")
    bundle_data(bids_dir_path, tar_file_path)
    _logger.info(f"Saved processed data .tar file at: {tar_file_path}")

    _logger.info("Process completed.")


def main():
    pass


if __name__ == "__main__":
    main()
