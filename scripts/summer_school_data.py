"""
Organizes data, extracts features, and bundles everything together in an easily distributable
format for the Bridge2AI Summer School.
"""

import argparse
import os
import shutil
import tarfile

import torch
import pydra
from senselab.audio.tasks.speech_to_text import transcribe_dataset

from b2aiprep.prepare import redcap_to_bids
from b2aiprep.process import (
    embed_speaker,
    specgram,
    melfilterbank,
    MFCC,
    extract_opensmile,
    Audio,
)


def wav_to_features(wav_path):
    """
    Extracts features from a single audio file at specified path.
    """
    audio = Audio.from_file(str(wav_path))
    audio = audio.to_16khz()
    features = {}
    # features["transcription"] = transcribe_dataset({"audio": audio}, model_id="openai/whisper-tiny")
    print(features["transcription"])
    features["speaker_embedding"] = embed_speaker(
        audio,
        model="speechbrain/spkrec-ecapa-voxceleb"
    )
    features["specgram"] = specgram(audio)
    features["melfilterbank"] = melfilterbank(features["specgram"])
    features["mfcc"] = MFCC(features["melfilterbank"])
    features["sample_rate"] = audio.sample_rate
    features["opensmile"] = extract_opensmile(audio)
    return features


def extract_features(processed_files_path):
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
                voice_path = os.path.join(sub_file_path, ses_file, "voice")
                if ses_file.startswith("ses") and os.path.isdir(voice_path):
                    
                    # Iterate over each audio file in the voice directory.
                    for audio_file in os.listdir(voice_path):
                        if audio_file.endswith('.wav'):
                            audio_path = os.path.join(voice_path, audio_file)
                            
                            # Extract features from the audio file.
                            features = wav_to_features(audio_path)
                            
                            # Remove the original audio file.
                            os.remove(audio_path)
                            
                            # Save the extracted voice features.
                            save_path = audio_path[:-len(".wav")] + ".pt"
                            torch.save(features, save_path)


def bundle_data(source_directory, save_path):
    """Saves data bundle as a tar file with gzip compression."""
    with tarfile.open(save_path, "w:gz") as tar:
        tar.add(source_directory, arcname=os.path.basename(source_directory))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process audio data for multiple subjects."
    )
    parser.add_argument('--redcap_file_path',
                        type=str,
                        required=True,
                        help='Path to the RedCap data file.')
    parser.add_argument('--audio_dir_path',
                        type=str,
                        required=True,
                        help='Path to the audio files directory.')
    parser.add_argument('--tar_file_path',
                        type=str,
                        required=True,
                        help='Where to save the data bundle .tar file.')
    parser.add_argument('--bids_files_path',
                        type=str,
                        required=False,
                        default="summer-school-processed-data",
                        help='Where to save the BIDS-like data.')
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("Organizing data into BIDS-like directory structure...")
    redcap_to_bids(args.redcap_file_path,
                   args.bids_files_path,
                   args.audio_dir_path)

    print("Removing old processed files...")
    if os.path.isdir(args.bids_files_path):
        shutil.rmtree(args.bids_files_path)

    # print("Copying BIDS-like data directory...")
    # shutil.copytree(args.data_path, args.processed_files_path)

    print("Extracting acoustic features from audio...")
    extract_features(args.processed_files_path)

    print("Saving .tar file with processed data...")
    bundle_data(args.processed_files_path, args.tar_save_path)

    print("Process completed.")


if __name__ == "__main__":
    main()
