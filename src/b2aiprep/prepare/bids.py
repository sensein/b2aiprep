"""Reorganize RedCap CSV and voice waveforms into a BIDS-like structure.

BIDS is a format designed to organize research data, primarily neuroinformatics
research data. It prescribes an organization of folders and files:
https://bids-standard.github.io/bids-starter-kit/folders_and_files/folders.html

BIDS has a general structure of:
project/
└── subject
    └── session
        └── datatype

Files must follow a filename convention:
https://bids-standard.github.io/bids-starter-kit/folders_and_files/files.html

BIDS covers many data types, but audio data is not one of them. We therefore
refer to the reorganized format as "BIDS-like". We use the existing behavioural
data type for the information captured in RedCap, and a new audio data type for
the voice recordings.
"""

import logging
import os
from pathlib import Path

SUBJECT_PREFIX = "sub"
SESSION_PREFIX = "ses"
AUDIO_FOLDER = "audio"
_LOGGER = logging.getLogger(__name__)


def get_paths(
    dir_path: str | os.PathLike,
    file_extension: str,
):
    """Retrieve all file paths from a BIDS-like directory structure.

    This function traverses the specified BIDS directory, collecting paths to
    files with the provided extension from all subject and session directories that match the
    expected naming conventions.

    Args:
        dir_path: The root directory of the BIDS dataset.
        file_extension: The file extension to search for (e.g., ".wav").

    Returns:
        A list of dictionaries, each containing the path to an audio file and
        its size in bytes.
    """
    paths: list[dict[str, Path | int]] = []

    # Iterate over each subject directory.
    for sub_file in os.listdir(dir_path):
        subject_dir_path = os.path.join(dir_path, sub_file)
        if sub_file.startswith(SUBJECT_PREFIX) and os.path.isdir(subject_dir_path):
            # Iterate over each session directory within a subject.
            for session_dir_path in os.listdir(subject_dir_path):
                audio_path = os.path.join(subject_dir_path, session_dir_path, AUDIO_FOLDER)
                if session_dir_path.startswith(SESSION_PREFIX) and os.path.isdir(audio_path):
                    # _logger.info(audio_path)
                    # Iterate over each audio file in the voice directory.
                    for audio_file in os.listdir(audio_path):
                        if audio_file.endswith(file_extension):
                            file_path = Path(os.path.join(audio_path, audio_file))

                            # Extract subject ID more robustly
                            filename_first_part = file_path.name.split("_")[0]
                            if "sub-" in filename_first_part:
                                subject_id = filename_first_part.split("sub-")[1]
                            else:
                                # Skip files that don't follow BIDS naming convention
                                continue

                            paths.append(
                                {
                                    "path": file_path.absolute(),
                                    "subject": subject_id,
                                    "size": file_path.stat().st_size,
                                }
                            )
    return paths


def get_audio_paths(
    bids_dir_path: str | os.PathLike,
) -> list[dict[str, Path | int]]:
    """Retrieve all .wav audio file paths from a BIDS-like directory structure.

    This function traverses the specified BIDS directory, collecting paths to
    .wav audio files from all subject and session directories that match the
    expected naming conventions.

    Args:
        bids_dir_path: The root directory of the BIDS dataset.

    Returns:
        A list of dictionaries, each containing the path to an audio file and
        its size in bytes.
    """
    return get_paths(
        dir_path=bids_dir_path,
        file_extension=".wav",
    )


def validate_bids_folder_audios(bids_dir_path: Path):
    """Validate the audio aspect of the BIDS-like folder structure.

    This function checks that all audio files have corresponding feature
    files and transcriptions.

    Args:
        bids_dir_path: The root directory of the BIDS dataset.
    """
    audio_paths = [x["path"] for x in get_audio_paths(bids_dir_path)]
    missing_features = []
    missing_transcriptions = []
    for audio_path in audio_paths:
        audio_path = Path(audio_path)
        audio_dir = audio_path.parent
        features_dir = audio_dir.parent / "audio"
        feature_files = [file for file in features_dir.glob(f"{audio_path.stem}*.pt")]
        if len(feature_files) == 0:
            missing_features.append(audio_path)
        feature_files = [file for file in features_dir.glob(f"{audio_path.stem}*.txt")]
        if len(feature_files) == 0:
            missing_transcriptions.append(audio_path)

    if len(missing_transcriptions) > 0:
        _LOGGER.warning(
            f"Missing transcriptions for {len(missing_transcriptions)} / "
            f"{len(audio_paths)} audio files"
        )
    if len(missing_features) > 0:
        _LOGGER.warning(
            f"Missing features for {len(missing_features)} / {len(audio_paths)} audio files"
        )
    else:
        _LOGGER.info(f"All {len(audio_paths)} audio files have been processed.")
