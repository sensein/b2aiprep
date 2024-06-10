"""
Organizes data, extracts features, and bundles everything together in an
easily distributable format for the Bridge2AI Summer School.

Feature extraction is parallelized using Pydra.

Example usage:
python3 b2aiprep/scripts/summer_school_data.py \
    --audio_dir_path bridge2ai-voice-corpus-1-ftf/ftf-features \
    --tar_file_path ./summer_school_data_bundle.tar \
    --bids_files_path ./b2ai-data-bids-like
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
    """
    Extracts features from a single audio file at specified path.
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
    audio_paths = []

    # Iterate over each subject directory.
    for sub_file in os.listdir(bids_dir_path):
        subject_dir_path = os.path.join(bids_dir_path, sub_file)
        if sub_file.startswith(SUBJECT_ID) and os.path.isdir(subject_dir_path):
       
            # Iterate over each session directory within a subject.
            for session_dir_path in os.listdir(subject_dir_path):
                audio_path = os.path.join(subject_dir_path, session_dir_path, AUDIO_ID)
                _logger.info(audio_path)
                if session_dir_path.startswith(SESSION_ID) and os.path.isdir(audio_path):
                    
                    # Iterate over each audio file in the voice directory.
                    for audio_file in os.listdir(audio_path):
                        if audio_file.endswith(".wav"):
                            audio_file_path = os.path.join(audio_path, audio_file)
                            audio_paths.append(audio_file_path)
    return audio_paths


def extract_features_workflow(bids_dir_path, remove=True):

    # Initialize the Pydra workflow.
    ef_wf = pydra.Workflow(
        name="ef_wf",
        input_spec=["bids_dir_path"],
        bids_dir_path=bids_dir_path
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


def bundle_data(source_directory, save_path):
    """Saves data bundle as a tar file with gzip compression."""
    with tarfile.open(save_path, "w:gz") as tar:
        tar.add(source_directory, arcname=os.path.basename(source_directory))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process audio data for multiple subjects."
    )
    parser.add_argument('--audio_dir_path',
                        type=Path,
                        required=True,
                        help='Path to the audio files directory.')
    parser.add_argument('--tar_file_path',
                        type=Path,
                        required=True,
                        help='Where to save the data bundle .tar file.')
    parser.add_argument('--bids_files_path',
                        type=Path,
                        required=False,
                        default="summer-school-processed-data",
                        help='Where to save the BIDS-like data.')
    return parser.parse_args()


def main():
    args = parse_arguments()

    _logger.info("Organizing data into BIDS-like directory structure...")
    redcap_to_bids(args.updated_redcap_file_path,
                   args.bids_files_path,
                   args.audio_dir_path)
    _logger.info("Data organization complete.")

    _logger.info("Beginning audio feature extraction...")
    extract_features_workflow(args.bids_files_path, remove=False)
    _logger.info("Audio feature extraction complete.")

    _logger.info("Saving .tar file with processed data...")
    bundle_data(args.bids_files_path, args.tar_file_path)
    _logger.info(f"Saved processed data .tar file at: {args.tar_file_path}")

    _logger.info("Process completed.")


if __name__ == "__main__":
    main()