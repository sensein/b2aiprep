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

import json
import logging
import os
import typing as t
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel
from tqdm import tqdm

from b2aiprep.prepare.constants import AUDIO_TASKS, Instrument, RepeatInstrument
from b2aiprep.prepare.fhir_utils import convert_response_to_fhir
from b2aiprep.prepare.redcap import RedCapDataset

SUBJECT_PREFIX = "sub"
SESSION_PREFIX = "ses"
AUDIO_FOLDER = "audio"
RESAMPLE_RATE = 16000
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


# NOTE: The following functions have been moved to the RedCapDataset class in redcap.py:
# - load_redcap_csv
# - insert_missing_columns
# - validate_redcap_df_column_names
# - get_df_of_repeat_instrument
# - get_recordings_for_acoustic_task
#
# They are kept here as deprecated wrappers for backward compatibility.


# TODO Modify the save path and naming of this to correspond to the voice file
def write_pydantic_model_to_bids_file(
    output_path: Path,
    data: BaseModel,
    schema_name: str,
    subject_id: str,
    session_id: t.Optional[str] = None,
    task_name: t.Optional[str] = None,
    recording_name: t.Optional[str] = None,
):
    """Write a Pydantic model (presumably a FHIR resource) to a JSON file.

    Follows the BIDS file name conventions.

    Args:
        output_path: The path to write the file to.
        data: The data to write.
        schema_name: The name of the schema.
        subject_id: The subject ID.
        session_id: The session ID.
        task_name: The task name.
        recording_name: The recording name.
    """
    # sub-<participant_id>_ses-<session_id>_task-<task_name>_run-_metadata.json
    filename = f"sub-{subject_id}"
    if session_id is not None:
        session_id = session_id.replace(" ", "-")
        filename += f"_ses-{session_id}"
    if task_name is not None:
        task_name = task_name.replace(" ", "-")
        if recording_name is not None:
            task_name = recording_name.replace(" ", "-")
        filename += f"_task-{task_name}"

    schema_name = schema_name.replace(" ", "-").replace("schema", "")
    schema_name = schema_name + "-metadata"
    filename += f"_{schema_name}.json"

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=False)
    with open(output_path / filename, "w") as f:
        f.write(data.json(indent=2))


def create_file_dir(participant_id, session_id):
    """Create a file directory structure which follows BIDS.

    BIDS folders are formatted as follows:
    base_project_folder/
    ├── sub-<participant_id>/
    │   ├── ses-<session_id>/
    │   │   ├── beh/
    │   │   │   ├── sub-<participant_id>_ses-<session_id>_task-<task_name>_rec
                    -<recording_name>_schema.json
    │   │   │   ├── sub-<participant_id>_ses-<session_id>_task-<task_name>_rec
                    -<recording_name>_schema.json

    Args:
        participant_id: The participant ID.
        session_id: The session ID.
    """
    output_path = Path("./output/")
    paths = [
        output_path / f"sub-{participant_id}/",
        output_path / f"sub-{participant_id}/ses-{session_id}/",
        output_path / f"sub-{participant_id}/ses-{session_id}/beh/",
        output_path / f"sub-{participant_id}/ses-{session_id}/audio/",
    ]

    if session_id == "":
        paths = paths[0:1]

    for folder in paths:
        folder.mkdir(parents=True, exist_ok=True)


def questionnaire_mapping(questionnaire_name: str) -> dict:
    """Get questionnaire mapping for a given questionnaire name.

    Args:
        questionnaire_name: The name of the questionnaire.

    Returns:
        The questionnaire mapping data.
    """
    with open("questionnaire_mapping/mappings.json") as mapping_file:
        mapping_str = mapping_file.read()
        mapping_data = json.loads(mapping_str)
        return mapping_data[questionnaire_name]


def get_instrument_for_name(name: str) -> Instrument:
    """Get instrument for a given name.

    Args:
        name: The instrument name.

    Returns:
        The instrument object.

    Raises:
        ValueError: If no instrument found for the given name.
    """
    for repeat_instrument in RepeatInstrument:
        instrument = repeat_instrument.value
        if instrument.name == name:
            return instrument
    raise ValueError(f"No instrument found for value {name}")


def output_participant_data_to_fhir(
    participant: dict, outdir: Path, audiodir: t.Optional[Path] = None
):
    """Output participant data to FHIR format.

    Args:
        participant: The participant data dictionary.
        outdir: The output directory path.
        audiodir: The audio directory path (optional).
    """
    participant_id = participant["record_id"]
    subject_path = outdir / f"sub-{participant_id}"

    if audiodir is not None and not Path(audiodir).exists():
        audiodir = None

    # TODO: prepare a Patient resource to use as the reference for each questionnaire
    # patient = create_fhir_patient(participant)

    session_instrument = get_instrument_for_name("sessions")
    task_instrument = get_instrument_for_name("acoustic_tasks")
    recording_instrument = get_instrument_for_name("recordings")

    sessions_df = pd.DataFrame(columns=session_instrument.columns)

    # validated questionnaires are asked per session
    for session in participant["sessions"]:
        sessions_row = {key: session[key] for key in session_instrument.columns}
        sessions_df = pd.concat([sessions_df, pd.DataFrame([sessions_row])], ignore_index=True)
        session_id = session["session_id"]
        # TODO: prepare a session resource to use as the encounter reference for
        # each session questionnaire
        session_path = subject_path / f"ses-{session_id}"
        audio_output_path = session_path / "audio"
        if not audio_output_path.exists():
            audio_output_path.mkdir(parents=True, exist_ok=True)

        # multiple acoustic tasks are asked per session
        for task in session["acoustic_tasks"]:
            if task is None:
                continue

            acoustic_task_name = task["acoustic_task_name"].replace(" ", "-")
            fhir_data = convert_response_to_fhir(
                task,
                questionnaire_name=task_instrument.name,
                mapping_name=task_instrument.schema_name,
                columns=task_instrument.columns,
            )
            write_pydantic_model_to_bids_file(
                audio_output_path,
                fhir_data,
                schema_name=task_instrument.schema_name,
                subject_id=participant_id,
                session_id=session_id,
                task_name=acoustic_task_name,
            )

            # prefix is used to name audio files, if they are copied over
            prefix = f"sub-{participant_id}_ses-{session_id}"

            # there may be more than one recording per acoustic task
            for recording in task["recordings"]:
                fhir_data = convert_response_to_fhir(
                    recording,
                    questionnaire_name=recording_instrument.name,
                    mapping_name=recording_instrument.schema_name,
                    columns=recording_instrument.columns,
                )
                write_pydantic_model_to_bids_file(
                    audio_output_path,
                    fhir_data,
                    schema_name=recording_instrument.schema_name,
                    subject_id=participant_id,
                    session_id=session_id,
                    task_name=acoustic_task_name,
                    recording_name=recording["recording_name"],
                )

                # we also need to organize the audio file
                if audiodir is not None:
                    # audio files are under a folder with the site name,
                    # so we need to recursively glob
                    audio_files = list(audiodir.rglob(f"{recording['recording_id']}*"))
                    if len(audio_files) == 0:
                        logging.warning(
                            f"No audio file found for recording \
                                {recording['recording_id']}."
                        )
                    else:
                        if len(audio_files) > 1:
                            logging.warning(
                                (
                                    f"Multiple audio files found for recording \
                                        {recording['recording_id']}."
                                    f" Using only {audio_files[0]}"
                                )
                            )
                        audio_file = audio_files[0]

                        # copy file
                        ext = audio_file.suffix

                        recording_name = recording["recording_name"].replace(" ", "-")
                        audio_file_destination = (
                            audio_output_path / f"{prefix}_task-{recording_name}{ext}"
                        )
                        if audio_file_destination.exists():
                            logging.warning(
                                f"Audio file {audio_file_destination} already exists. Skipping."
                            )
                        audio_file_destination.write_bytes(audio_file.read_bytes())
    # Save sessions.tsv
    if not os.path.exists(subject_path):
        os.mkdir(subject_path)
    sessions_tsv_path = subject_path / "sessions.tsv"
    sessions_df.to_csv(sessions_tsv_path, sep="\t", index=False)


def construct_tsv_from_json(
    df: pd.DataFrame, json_file_path: str, output_dir: str, output_file_name: t.Optional[str] = None
) -> None:
    """Construct a TSV file from a DataFrame and a JSON file specifying column labels.

    Combines entries so that there is one row per record_id.

    Args:
        df: DataFrame containing the data.
        json_file_path: Path to the JSON file with the column labels.
        output_dir: Output directory where the TSV file will be saved.
        output_file_name: The name of the output TSV file.
            If not provided, the JSON file name is used with a .tsv extension.

    Raises:
        ValueError: If no valid columns found in DataFrame that match JSON file.
    """
    # Load the column labels from the JSON file
    with open(json_file_path, "r") as f:
        json_data = json.load(f)

    first_key = next(iter(json_data))

    column_labels = []
    if "data_elements" in json_data[first_key]:
        column_labels = list(json_data[first_key]["data_elements"])
    else:
        column_labels = list(json_data.keys())

    # Filter column labels to only include those that exist in the DataFrame
    valid_columns = [col for col in column_labels if col in df.columns]
    if not valid_columns:
        raise ValueError("No valid columns found in DataFrame that match JSON file")

    if "record_id" not in valid_columns:
        valid_columns = ["record_id"] + valid_columns

    # Select the relevant columns from the DataFrame
    selected_df = df[valid_columns]

    # Combine entries so there is one row per record_id
    combined_df = selected_df.groupby("record_id").first().reset_index()

    # Define the output file name and path
    if output_file_name is None:
        output_file_name = os.path.splitext(os.path.basename(json_file_path))[0] + ".tsv"

    tsv_path = os.path.join(output_dir, output_file_name)

    # Save the combined DataFrame to a TSV file
    combined_df.to_csv(tsv_path, sep="\t", index=False)

    print(f"TSV file created and saved to: {tsv_path}")


def construct_all_tsvs_from_jsons(
    df: pd.DataFrame,
    input_dir: str,
    output_dir: str,
    excluded_files: t.Optional[t.List[str]] = None,
) -> None:
    """Construct TSV files from all JSON files in a specified directory.

    Excludes specific files if provided.

    Args:
        df: DataFrame containing the data.
        input_dir: Directory containing JSON files with column labels.
        output_dir: Directory where the TSV files will be saved.
        excluded_files: List of JSON filenames to exclude from processing.
            Defaults to None.
    """
    # Ensure the excluded_files list is initialized if None is provided
    if excluded_files is None:
        excluded_files = []

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        # Process only JSON files that are not excluded
        if filename.endswith(".json") and filename not in excluded_files:
            json_file_path = os.path.join(input_dir, filename)

            # Construct the TSV file from the JSON file
            construct_tsv_from_json(df=df, json_file_path=json_file_path, output_dir=output_dir)



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
