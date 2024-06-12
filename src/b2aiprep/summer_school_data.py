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

    python3 b2aiprep/src/b2aiprep/summer_school_data.py \
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
    - prepare_summer_school_data: Organizes and processes Bridge2AI data for 
        distribution.
    - parse_arguments: Parses command line arguments for processing audio data.
"""

import argparse
import os
import logging
import tarfile

import torch
import pydra
from pathlib import Path

from b2aiprep.prepare import redcap_to_bids
from b2aiprep.process import (
    embed_speaker,
    specgram,
    melfilterbank,
    MFCC,
    extract_opensmile,
    Audio,
)

SUBJECT_ID = "sub"
SESSION_ID = "ses"
AUDIO_ID = "audio"

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@pydra.mark.task
def wav_to_features(wav_path):
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
    audio = Audio.from_file(wav_path)
    audio = audio.to_16khz()

    features = {}
    features["speaker_embedding"] = embed_speaker(
        audio,
        model="speechbrain/spkrec-ecapa-voxceleb"
    )
    features["specgram"] = specgram(audio)
    features["melfilterbank"] = melfilterbank(features["specgram"])
    features["mfcc"] = MFCC(features["melfilterbank"])
    features["sample_rate"] = audio.sample_rate
    features["opensmile"] = extract_opensmile(audio)
    logging.disable(logging.NOTSET)
    _logger.setLevel(logging.INFO)

    save_path = wav_path[:-len(".wav")] + ".pt"
    torch.save(features, save_path)
    return features


@pydra.mark.task
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
                audio_path = os.path.join(subject_dir_path, 
                                          session_dir_path, 
                                          AUDIO_ID)

                if session_dir_path.startswith(SESSION_ID) \
                        and os.path.isdir(audio_path):
                    _logger.info(audio_path)
                    # Iterate over each audio file in the voice directory.
                    for audio_file in os.listdir(audio_path):
                        if audio_file.endswith(".wav"):
                            audio_file_path = os.path.join(audio_path, 
                                                           audio_file)
                            audio_paths.append(audio_file_path)
    return audio_paths


def extract_features_workflow(bids_dir_path: Path, remove: bool = True):
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
    # Initialize the Pydra workflow.
    ef_wf = pydra.Workflow(
        name="ef_wf",
        input_spec=["bids_dir_path"],
        bids_dir_path=bids_dir_path,
        cache_dir=None
    )

    # Get paths to every audio file.
    ef_wf.add(get_audio_paths(
            name="audio_paths",
            bids_dir_path=bids_dir_path
        )
    )

    # Run wav_to_features for each audio file.
    ef_wf.add(wav_to_features(
            name="features",
            wav_path=ef_wf.audio_paths.lzout.out
        ).split(
            "wav_path",
            wav_path=ef_wf.audio_paths.lzout.out
        )
    )

    ef_wf.set_output({"audio_paths": ef_wf.audio_paths.lzout.out,
                      "features": ef_wf.features.lzout.out})
    with pydra.Submitter(plugin='cf') as run:
        run(ef_wf)
    return ef_wf


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


def prepare_summer_school_data(
            redcap_csv_path: Path, 
            audio_dir_path: Path, 
            bids_dir_path: Path, 
            tar_file_path: Path
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
    if not os.path.exists(bids_dir_path):
        os.makedirs(bids_dir_path)
        _logger.info(f"Created directory: {bids_dir_path}")

    _logger.info("Organizing data into BIDS-like directory structure...")
    redcap_to_bids(redcap_csv_path, bids_dir_path, audio_dir_path)
    _logger.info("Data organization complete.")

    _logger.info("Beginning audio feature extraction...")
    extract_features_workflow(bids_dir_path, remove=False)
    _logger.info("Audio feature extraction complete.")

    _logger.info("Saving .tar file with processed data...")
    bundle_data(bids_dir_path, tar_file_path)
    _logger.info(f"Saved processed data .tar file at: {tar_file_path}")

    _logger.info("Process completed.")


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments for processing audio data.

    This function sets up and parses command line arguments required for 
    processing audio data for multiple subjects. It includes paths for the 
    REDCap CSV file, audio files directory, BIDS-like data directory, and 
    the output tar file path.

    Returns:
      argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process audio data for multiple subjects."
    )
    parser.add_argument('--redcap_csv_path',
                        type=Path,
                        required=True,
                        help='Path to the REDCap CSV file.')
    parser.add_argument('--audio_dir_path',
                        type=Path,
                        required=True,
                        help='Path to the audio files directory.')
    parser.add_argument('--bids_dir_path',
                        type=Path,
                        required=False,
                        default="summer-school-processed-data",
                        help='Where to save the BIDS-like data.')
    parser.add_argument('--tar_file_path',
                        type=Path,
                        required=True,
                        help='Where to save the data bundle .tar file.')
    return parser.parse_args()


def main():
    args = parse_arguments()
    prepare_summer_school_data(
        redcap_csv_path=args.redcap_csv_path,
        audio_dir_path=args.audio_dir_path,
        bids_dir_path=args.bids_dir_path,
        tar_file_path=args.tar_file_path
    )


if __name__ == "__main__":
    main()
