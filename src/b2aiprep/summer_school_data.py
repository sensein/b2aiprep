"""
Organizes data, extracts features, and bundles everything together in an
easily distributable format for the Bridge2AI Summer School.

Only works with the face-to-face RedCap CSV and a folder containing the
face-to-face features files presently.

Example usage:
python3 b2aiprep/scripts/summer_school_data.py \
    --ftf_redcap_file_path bridge2ai-voice-corpus-1-ftf/bridge2ai_voice_data.csv \
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
# from senselab.audio.tasks.speech_to_text import transcribe_dataset
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

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)

def wav_to_features(wav_path):
    """
    Extracts features from a single audio file at specified path.
    """
    logging.disable(logging.WARNING)
    audio = Audio.from_file(str(wav_path))
    audio = audio.to_16khz()
    features = {}
    # features["transcription"] = transcribe_dataset({"audio": audio}, model_id="openai/whisper-tiny")
    # _logger.info(features["transcription"])
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
    return features


def extract_features(processed_files_path, remove=False):
    """
    Extracts features for every activity audio file.

    Args:
        processed_files_path (str): The path to the directory saved as a .tar.

    Returns:
        None
    """
    # Iterate over each subject directory.
    for sub_file in os.listdir(processed_files_path):
        sub_file_path = os.path.join(processed_files_path, sub_file)
        if sub_file.startswith("sub") and os.path.isdir(sub_file_path):
       
            # Iterate over each session directory within a subject.
            for ses_file in os.listdir(sub_file_path):
                voice_path = os.path.join(sub_file_path, ses_file, "audio")
                _logger.info(voice_path)
                if ses_file.startswith("ses") and os.path.isdir(voice_path):
                    
                    # Iterate over each audio file in the voice directory.
                    for audio_file in os.listdir(voice_path):
                        if audio_file.endswith('.wav'):
                            audio_path = os.path.join(voice_path, audio_file)
                            
                            # Extract features from the audio file.
                            _logger.info(f"Extracting features: {audio_file}")
                            features = wav_to_features(audio_path)
                            
                            # Remove the original audio file.
                            if remove:
                                os.remove(audio_path)
                            
                            # Save the extracted voice features.
                            save_path = audio_path[:-len(".wav")] + ".pt"
                            torch.save(features, save_path)


def bundle_data(source_directory, save_path):
    """Saves data bundle as a tar file with gzip compression."""
    with tarfile.open(save_path, "w:gz") as tar:
        tar.add(source_directory, arcname=os.path.basename(source_directory))


# def parse_arguments():
#     parser = argparse.ArgumentParser(
#         description="Process audio data for multiple subjects."
#     )
#     parser.add_argument('--ftf_redcap_file_path',
#                         type=Path,
#                         required=True,
#                         help='Path to the RedCap data file.')
#     parser.add_argument('--audio_dir_path',
#                         type=Path,
#                         required=True,
#                         help='Path to the audio files directory.')
#     parser.add_argument('--tar_file_path',
#                         type=Path,
#                         required=True,
#                         help='Where to save the data bundle .tar file.')
#     parser.add_argument('--bids_files_path',
#                         type=Path,
#                         required=False,
#                         default="summer-school-processed-data",
#                         help='Where to save the BIDS-like data.')
#     return parser.parse_args()

class parse_arguments():
    def __init__(self) -> None:
        self.ftf_redcap_file_path=Path("/Users/isaacbevers/sensein/b2ai-wrapper/b2ai-data/bridge2ai-voice-corpus-1/bridge2ai_voice_data.csv")
        self.updated_redcap_file_path=Path("/Users/isaacbevers/sensein/b2ai-wrapper/b2ai-data/Bridge2AIDEVELOPMENT_DATA_2024-06-03_1705.csv")
        self.audio_dir_path=Path("/Users/isaacbevers/sensein/b2ai-wrapper/b2ai-data/audio-data")
        self.tar_file_path=Path("/Users/isaacbevers/sensein/b2ai-wrapper/b2ai-data/summer_school_data_test_bundle.tar")
        self.bids_files_path=Path("/Users/isaacbevers/sensein/b2ai-wrapper/b2ai-data-bids-like")


def main():
    args = parse_arguments()


    # _logger.info("Organizing data into BIDS-like directory structure...")
    # redcap_to_bids(args.ftf_redcap_file_path,
    #                args.bids_files_path,
    #                args.audio_dir_path)

    _logger.info("Extracting acoustic features from audio...")
    extract_features(args.bids_files_path, remove=False)
    # _logger.info("Extracting features in ")

    # _logger.info("Saving .tar file with processed data...")
    # bundle_data(args.bids_files_path, args.tar_file_path)
    # _logger.info(f"Saved .tar file with processed data at {args.tar_file_path}")

    _logger.info("Process completed.")


if __name__ == "__main__":
    main()
