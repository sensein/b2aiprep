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
from b2aiprep.prepare.utils import (
    _transform_str_for_bids_filename,
    construct_all_tsvs_from_jsons,
    construct_tsv_from_json,
)

_LOGGER = logging.getLogger(__name__)


def load_redcap_csv(file_path):
    """Load the RedCap CSV file into a DataFrame. Makes modifications to the data
    to ensure compatibility with downstream tooling.

    Parameters
    ----------
    file_path : str
        The path to the CSV file.

    Returns
    -------
    DataFrame
        The loaded data.
    """
    try:
        data = pd.read_csv(file_path, low_memory=False, na_values="")
        if "redcap_repeat_instrument" in data.columns:
            col = "redcap_repeat_instrument"
        elif "Repeat Instrument" in data.columns:
            col = "Repeat Instrument"
        else:
            raise ValueError("No repeat instrument column found in CSV file.")

        # The RedCap CSV exports all rows with a repeat instrument *except*
        # a single row per subject, which we refer to as the "Participant" instrument.
        # We need to fill in the missing values for the Participant instrument
        # so that downstream tools can filter rows based on this column.
        participant_instrument = RepeatInstrument.PARTICIPANT.value
        data[col] = data[col].fillna(participant_instrument.text).astype(str)
        return data
    except FileNotFoundError:
        print("File not found.")
        return None
    except pd.errors.ParserError:
        print("Error parsing CSV file.")
        return None


def validate_redcap_df_column_names(df: DataFrame) -> DataFrame:
    """RedCap allows two distinct export formats: raw data or with labels.
    The raw data format exports column names as coded entries, e.g. "record_id".
    This would be ideal, but it modifies the values of the data export. To avoid this,
    the dataset is exported with labels, e.g. "Record ID". Afterward, the dataset
    has the header manually modified to match the coded entries.

    This cannot be done with the data dictionary, as the data dictionary dimensions
    do not match the dataset dimensions. The raw data must also be exported and the header
    should be copied over to the label data.

    This function verifies the headers are exported correctly.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to update.

    Raises
    ------
    ValueError
        If the columns in the DataFrame do not match the expected columns
        for the Bridge2AI data voice data.
    """

    # this column mapping is derived from the data dictionary, and is a subset
    # of the columns exported from redcap.
    b2ai_resources = files("b2aiprep").joinpath("prepare").joinpath("resources")
    column_mapping: dict = json.loads(b2ai_resources.joinpath("column_mapping.json").read_text())
    
    # only map columns if we have a full overlap with the mapping dict
    overlap_with_label = set(df.columns.tolist()).intersection(set(column_mapping.values()))
    overlap_with_coded = set(df.columns.tolist()).intersection(set(column_mapping.keys()))

    if len(overlap_with_coded) == df.shape[1]:
        return
    
    if len(overlap_with_coded) == 0:
        raise ValueError(
            (
                "Dataframe has no coded headers. Please modify the source data to have "
                "coded labels instead."
            )
        )

    # if we have more than half of the columns as label headers, we assume the data is
    # exported with labels
    if len(overlap_with_label) > (df.shape[1] * 0.5):
        raise ValueError(
            (
                "Dataframe has label headers rather than coded headers. Please modify the source data to have "
                "coded labels instead."
            )
        )

    # raise a warning about the labels - unclear why there would be a mix
    _LOGGER.warning(
        (
            f"Dataframe has a mix of label and coded headers: {len(overlap_with_coded)} coded and "
            f"{len(overlap_with_label)} label. Downstream processing expects only coded labels."
        )
    )


def get_df_of_repeat_instrument(df: DataFrame, instrument: Instrument) -> pd.DataFrame:
    """Filters rows and columns of a RedCap dataframe to correspond to a specific repeat instrument.

    Parameters
    ----------
    df : DataFrame
        The DataFrame to filter.
    instrument : Instrument
        The repeat instrument to filter for.

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame.
    """

    columns = instrument.get_columns()
    idx = df["redcap_repeat_instrument"] == instrument.text

    columns_present = [c for c in columns if c in df.columns]
    dff = df.loc[idx, columns_present].copy()
    if len(columns_present) < len(columns):
        _LOGGER.warning(
            (
                f"Inserting None for missing columns of {RepeatInstrument.PARTICIPANT} instrument: "
                f"{set(columns) - set(columns_present)}"
            )
        )
        impute_value = np.nan
        for c in set(columns) - set(columns_present):
            dff[c] = impute_value
    return dff


def get_recordings_for_acoustic_task(df: pd.DataFrame, acoustic_task: str) -> pd.DataFrame:
    """Get recordings for a specific acoustic task.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to filter.
    acoustic_task : str
        The acoustic task to filter for.

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame.
    """
    if acoustic_task not in AUDIO_TASKS:
        raise ValueError(f"Unrecognized {acoustic_task}. Options: {AUDIO_TASKS}")

    acoustic_tasks_df = get_df_of_repeat_instrument(df, RepeatInstrument.ACOUSTIC_TASK.value)
    recordings_df = get_df_of_repeat_instrument(df, RepeatInstrument.RECORDING.value)

    idx = acoustic_tasks_df["acoustic_task_name"] == acoustic_task

    dff = recordings_df.merge(
        acoustic_tasks_df.loc[idx, ["acoustic_task_id"]],
        left_on="recording_acoustic_task_id",
        right_on="acoustic_task_id",
        how="inner",
    )
    return dff


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
    """Write a Pydantic model (presumably a FHIR resource) to a JSON file following
    the BIDS file name conventions.

    Parameters
    ----------
    output_path : Path
        The path to write the file to.
    data : BaseModel
        The data to write.
    schema_name : str
        The name of the schema.
    subject_id : str
        The subject ID.
    session_id : str, optional
        The session ID.
    task_name : str, optional
        The task name.
    recording_name : str, optional
        The recording name.
    """
    # sub-<participant_id>_ses-<session_id>_task-<task_name>_run-_metadata.json
    filename = f"sub-{subject_id}"
    if session_id is not None:
        session_id = _transform_str_for_bids_filename(session_id)
        filename += f"_ses-{session_id}"
    if task_name is not None:
        task_name = _transform_str_for_bids_filename(task_name)
        if recording_name is not None:
            task_name = _transform_str_for_bids_filename(recording_name)
        filename += f"_task-{task_name}"

    schema_name = _transform_str_for_bids_filename(schema_name).replace("schema", "")
    schema_name = schema_name + "-metadata"
    filename += f"_{schema_name}.json"

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=False)
    with open(output_path / filename, "w") as f:
        f.write(data.json(indent=2))


def _df_to_dict(df: pd.DataFrame, index_col: str) -> t.Dict[str, t.Any]:
    """Convert a DataFrame to a dictionary of dictionaries, with the given column as the index.

    Retains the index column within the dictionary.

    Args:
        df: DataFrame to convert.
        index_col: Column to use as the index.

    Returns:
        Dictionary of dictionaries.
    """
    if index_col not in df.columns:
        raise ValueError(f"Index column {index_col} not found in DataFrame.")

    if df[index_col].isnull().any():
        _LOGGER.warn(
            f"Found {df[index_col].isnull().sum()} null value(s) for {index_col}. Removing."
        )
        df = df.dropna(subset=[index_col])

    if df[index_col].nunique() < df.shape[0]:
        raise ValueError(f"Non-unique {index_col} values found.")

    # *copy* the given column into the index, preserving the original column
    # so that it is output in the later call to to_dict()
    df.index = df[index_col]

    return df.to_dict("index")


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
    with open("questionnaire_mapping/mappings.json") as mapping_file:
        mapping_str = mapping_file.read()
        mapping_data = json.loads(mapping_str)
        return mapping_data[questionnaire_name]


def get_instrument_for_name(name: str) -> Instrument:
    for repeat_instrument in RepeatInstrument:
        instrument = repeat_instrument.value
        if instrument.name == name:
            return instrument
    raise ValueError(f"No instrument found for value {name}")


def output_participant_data_to_fhir(
    participant: dict, outdir: Path, audiodir: t.Optional[Path] = None
):
    participant_id = participant["record_id"]
    subject_path = outdir / f"sub-{participant_id}"

    if audiodir is not None and not audiodir.exists():
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

            acoustic_task_name = _transform_str_for_bids_filename(task["acoustic_task_name"])
            fhir_data = convert_response_to_fhir(
                task,
                questionnaire_name=task_instrument.name,
                mapping_name=task_instrument.schema,
                columns=task_instrument.columns,
            )
            write_pydantic_model_to_bids_file(
                audio_output_path,
                fhir_data,
                schema_name=task_instrument.schema,
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
                    mapping_name=recording_instrument.schema,
                    columns=recording_instrument.columns,
                )
                write_pydantic_model_to_bids_file(
                    audio_output_path,
                    fhir_data,
                    schema_name=recording_instrument.schema,
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

                        recording_name = _transform_str_for_bids_filename(
                            recording["recording_name"]
                        )
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


def redcap_to_bids(
    filename: Path,
    outdir: Path,
    update_column_names: bool,
    audiodir: t.Optional[Path] = None,
):
    """Converts the Bridge2AI RedCap CSV output and a folder of audio files
    into the BIDS folder organization.

    BIDS has a general structure of:
    project/
    └── subject
        └── session
            └── datatype

    And filenames follow a convention of:
    sub-<participant_id>_ses-<session_id>[_task-<task_name>_rec-<recording_name>]_<schema>.json

    Parameters
    ----------
    filename : Path
        The path to the RedCap CSV file.
    outdir : Path
        The path to write the BIDS-like data to.
    audiodir : Path, optional
        The path to the folder containing audio files. If omitted, no audio data will be included
        in the reorganized output.

    Raises
    ------
    ValueError
        If the RedCap CSV file is not found, or if the columns in the CSV file do not match
        the expected columns for the Bridge2AI data.
    """
    df = load_redcap_csv(filename)

    # It is possible for each column in the dataframe to have one of two names:
    #   1. coded column names ("record_id")
    #   2. text column names ("Record ID")

    # we require coded column names for downstream processing. it is not trivial to map
    # them one to one, as the text column names are not unique (e.g. there are ~6 columns
    # with the name "Strain").
    validate_redcap_df_column_names(df)
    
    construct_tsv_from_json(  # construct participants.tsv
        df=df,
        json_file_path=os.path.join(outdir, "participants.json"),
        output_dir=outdir,
        output_file_name="participants.tsv",
    )

    construct_all_tsvs_from_jsons(  # construct phenotype tsvs
        df=df,
        input_dir=os.path.join(outdir, "phenotype"),
        output_dir=os.path.join(outdir, "phenotype"),
    )

    # The repeat instrument columns also defines all the possible
    # repeat instruments we would like to extract from the RedCap CSV.
    # In this case, repeat instruments are not necessarily repeatedly collected.
    # Rather they are entries that correspond with specific activities.
    repeat_instruments: t.List[RepeatInstrument] = list(RepeatInstrument.__members__.values())

    # we use the RepeatInstrument values to create a dict that maps the instrument
    # record_id: dictionary where keys=column names and values=values for that record_id
    # i.e. we end up with dataframe_dicts = {
    #       'demographics': {"session_id_1": {data}, ...}},
    #       'confounders': {"session_id_1": {data}, ...}},
    #    ...
    #   }
    dataframe_dicts: t.Dict[RepeatInstrument, pd.DataFrame] = {}
    dataframe_dicts: t.Dict[RepeatInstrument, pd.DataFrame] = {}
    for repeat_instrument in repeat_instruments:
        instrument = repeat_instrument.value

        # load in the df based on the instrument
        questionnaire_df = get_df_of_repeat_instrument(df, instrument)
        _LOGGER.info(f"Number of {instrument.name} entries: {len(questionnaire_df)}")
        dataframe_dicts[repeat_instrument] = questionnaire_df

    # # create separate data frames for sets of columns
    # # number of participants
    # participants_df = get_df_of_repeat_instrument(
    #     df, RepeatInstrument.PARTICIPANT.value
    # )
    # _LOGGER.info(f"Number of participants: {len(participants_df)}")

    # # session info
    # sessions_df = get_df_of_repeat_instrument(df, RepeatInstrument.SESSION.value)
    # _LOGGER.info(f"Number of sessions: {len(sessions_df)}")

    # # subject id (record_id) to accoustic_task_id and accoustic_task_name
    # acoustic_tasks_df = get_df_of_repeat_instrument(
    #     df, RepeatInstrument.ACOUSTIC_TASK.value
    # )
    # _LOGGER.info(f"Number of Acoustic Tasks: {len(acoustic_tasks_df)}")

    # # recording info
    # recordings_df = get_df_of_repeat_instrument(
    #     df, RepeatInstrument.RECORDING.value
    # )
    # _LOGGER.info(f"Number of Recordings: {len(recordings_df)}")

    participants_df = dataframe_dicts.pop(RepeatInstrument.PARTICIPANT)
    sessions_df = dataframe_dicts.pop(RepeatInstrument.SESSION)
    acoustic_tasks_df = dataframe_dicts.pop(RepeatInstrument.ACOUSTIC_TASK)
    recordings_df = dataframe_dicts.pop(RepeatInstrument.RECORDING)

    # convert the remaining dataframes to dictionaries indexed by session_id
    for repeat_instrument, questionnaire_df in dataframe_dicts.items():
        session_id_col = repeat_instrument.value.session_id
        dataframe_dicts[repeat_instrument] = _df_to_dict(questionnaire_df, session_id_col)

    # create a list of dict containing FHIR formatted version of everyone's data
    participants = []
    for participant in participants_df.to_dict("records"):
        participants.append(participant)
        participant["sessions"] = sessions_df[
            sessions_df["record_id"] == participant["record_id"]
        ].to_dict("records")

        # sessions_df = pd.DataFrame()
        for session in participant["sessions"]:
            # there can be multiple acoustic tasks per session
            session_id = session["session_id"]
            session["acoustic_tasks"] = acoustic_tasks_df[
                acoustic_tasks_df["acoustic_task_session_id"] == session_id
            ].to_dict("records")
            for task in session["acoustic_tasks"]:
                # there can be multiple recordings per acoustic task
                task["recordings"] = recordings_df[
                    recordings_df["recording_acoustic_task_id"] == task["acoustic_task_id"]
                ].to_dict("records")

            # there can be only one demographics per session
            for key, df_by_session_id in dataframe_dicts.items():
                if session_id not in df_by_session_id:
                    _LOGGER.info(f"No {key} found for session {session_id}.")
                    session[key] = None
                else:
                    session[key] = df_by_session_id[session_id]

    # participants is a list of dictionaries; each dictionary has the same RedCap fields
    # but it respects the nesting / hierarchy present in the original data collection
    # TODO: maybe this warning should go in the main function
    if (audiodir is not None) and (not audiodir.exists()):
        logging.warning(f"{audiodir} path does not exist. No audio files will be reorganized.")
    for participant in tqdm(participants, desc="Writing participant data to file"):
        output_participant_data_to_fhir(participant, outdir, audiodir=audiodir)
