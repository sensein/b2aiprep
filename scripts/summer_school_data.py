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

_LOGGER = logging.getLogger(__name__)


def wav_to_features(wav_path):
    """
    Extracts features from a single audio file at specified path.
    """
    logging.disable(logging.CRITICAL)
    audio = Audio.from_file(str(wav_path))
    audio = audio.to_16khz()
    features = {}
    # features["transcription"] = transcribe_dataset({"audio": audio}, model_id="openai/whisper-tiny")
    # _LOGGER.info(features["transcription"])
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
    _LOGGER.setLevel(logging.INFO)
    return features


@pydra.mark.task
def extract_features_from_audio(audio_path, remove=True):
    _LOGGER.info(f"Extracting features: {audio_path}")
    features = wav_to_features(audio_path)
    if remove:
        os.remove(audio_path)
    save_path = audio_path[:-len(".wav")] + ".pt"
    torch.save(features, save_path)
    return save_path

@pydra.mark.task
def list_audio_files_indices(session_dirs, index):
    session_dir = session_dirs[index]
    audio_files = [os.path.join(session_dir, f) for f in os.listdir(session_dir) if f.endswith('.wav')]
    return list(range(len(audio_files))), audio_files

@pydra.mark.task
def get_audio_file(audio_files, index):
    return audio_files[index]

@pydra.mark.task
def list_session_dirs_indices(subject_dirs, index):
    subject_dir = subject_dirs[index]
    session_dirs = [os.path.join(subject_dir, ses_file, "audio") for ses_file in os.listdir(subject_dir) if ses_file.startswith("ses") and os.path.isdir(os.path.join(subject_dir, ses_file, "audio"))]
    return list(range(len(session_dirs))), session_dirs

@pydra.mark.task
def get_session_dir(session_dirs, index):
    return session_dirs[index]

@pydra.mark.task
def list_subject_dirs_indices(processed_files_path):
    subject_dirs = [os.path.join(processed_files_path, sub_file) for sub_file in os.listdir(processed_files_path) if sub_file.startswith("sub") and os.path.isdir(os.path.join(processed_files_path, sub_file))]
    return list(range(len(subject_dirs))), subject_dirs

@pydra.mark.task
def get_subject_dir(subject_dirs, index):
    return subject_dirs[index]

def extract_features_pipeline(processed_files_path, remove=True):
    with pydra.Submitter(plugin="cf") as sub:
        wf = pydra.Workflow(name="extract_features_wf", input_spec=["processed_files_path", "remove"], processed_files_path=processed_files_path, remove=remove)

        # Task to list subject directories and indices
        wf.add(list_subject_dirs_indices(name="list_subjects", processed_files_path=wf.lzin.processed_files_path))
        
        # Split the workflow for each subject
        wf.add(get_subject_dir(name="get_subject", subject_dirs=wf.list_subjects.lzout[1], index=wf.list_subjects.lzout[0]))
        wf.get_subject.split("index")
        
        # Task to list session directories and indices for each subject
        wf.add(list_session_dirs_indices(name="list_sessions", subject_dirs=wf.get_subject.lzout))
        
        # Split the workflow for each session
        wf.add(get_session_dir(name="get_session", session_dirs=wf.list_sessions.lzout[1], index=wf.list_sessions.lzout[0]))
        wf.get_session.split("index")
        
        # Task to list audio files and indices for each session
        wf.add(list_audio_files_indices(name="list_audio", session_dirs=wf.get_session.lzout))
        
        # Split the workflow for each audio file
        wf.add(get_audio_file(name="get_audio", audio_files=wf.list_audio.lzout[1], index=wf.list_audio.lzout[0]))
        wf.get_audio.split("index")
        
        # Task to extract features for each audio file
        wf.add(extract_features_from_audio(name="extract", audio_path=wf.get_audio.lzout, remove=wf.lzin.remove))

        wf.set_output(("features_paths", wf.extract.lzout))

        result = sub(wf)
        return result.outputs.features_paths


# def extract_features(processed_files_path, remove=True):
#     """
#     Extracts features for every activity audio file.

#     Args:
#         processed_files_path (str): The path to the directory saved as a .tar.

#     Returns:
#         None
#     """
#     # Iterate over each subject directory.
#     for sub_file in os.listdir(processed_files_path):
#         sub_file_path = os.path.join(processed_files_path, sub_file)
#         if sub_file.startswith("sub") and os.path.isdir(sub_file_path):
       
#             # Iterate over each session directory within a subject.
#             for ses_file in os.listdir(sub_file_path):
#                 voice_path = os.path.join(sub_file_path, ses_file, "audio")
#                 _LOGGER.info(voice_path)
#                 if ses_file.startswith("ses") and os.path.isdir(voice_path):
                    
#                     # Iterate over each audio file in the voice directory.
#                     for audio_file in os.listdir(voice_path):
#                         if audio_file.endswith('.wav'):
#                             audio_path = os.path.join(voice_path, audio_file)
                            
#                             # Extract features from the audio file.
#                             _LOGGER.info(f"Extracting features: {audio_file}")
#                             features = wav_to_features(audio_path)
                            
#                             # Remove the original audio file.
#                             if remove:
#                                 os.remove(audio_path)
                            
#                             # Save the extracted voice features.
#                             save_path = audio_path[:-len(".wav")] + ".pt"
#                             torch.save(features, save_path)


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
    logging.basicConfig(level=logging.INFO)

    # _LOGGER.info("Organizing data into BIDS-like directory structure...")
    # redcap_to_bids(args.ftf_redcap_file_path,
    #                args.bids_files_path,
    #                args.audio_dir_path)

    _LOGGER.info("Extracting acoustic features from audio...")
    extract_features_pipeline(args.bids_files_path, remove=False)
    # _LOGGER.info("Extracting features in ")

    # _LOGGER.info("Saving .tar file with processed data...")
    # bundle_data(args.bids_files_path, args.tar_file_path)
    # _LOGGER.info(f"Saved .tar file with processed data at {args.tar_file_path}")

    _LOGGER.info("Process completed.")


if __name__ == "__main__":
    main()
