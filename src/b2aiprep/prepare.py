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
from pathlib import Path
import json
import typing as t

import numpy as np
import pandas as pd
from pandas import DataFrame
from importlib.resources import files
from pydantic import BaseModel
from tqdm import tqdm

from b2aiprep.fhir_utils import (
    convert_response_to_fhir,
    is_empty_questionnaire_response,
)
from b2aiprep.constants import (
    Instrument,
    RepeatInstrument,
    AUDIO_TASKS,   
    GENERAL_QUESTIONNAIRES,
    VALIDATED_QUESTIONNAIRES,
)
from b2aiprep.utils import _transform_str_for_bids_filename

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
        data = pd.read_csv(file_path, low_memory=False, na_values='')
        if 'redcap_repeat_instrument' in data.columns:
            col = 'redcap_repeat_instrument'
        elif 'Repeat Instrument' in data.columns:
            col = 'Repeat Instrument'
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
        print('File not found.')
        return None
    except pd.errors.ParserError:
        print('Error parsing CSV file.')
        return None

def update_redcap_df_column_names(df: DataFrame) -> DataFrame:
    """Update column names for a RedCap derived dataframe to match the coded
    column names used in the B2AI data processing pipeline.

    RedCap can export coded column names and text column names ("labels"), e.g.
    "redcap_repeat_instrument" vs. "Repeat Instrument". For downstream consistency,
    we map all column names to the coded form, i.e. "redcap_repeat_instrument".

    Parameters
    ----------
    df : DataFrame
        The DataFrame to update.
    
    Returns
    -------
    DataFrame
        The updated DataFrame.
    """

    b2ai_resources = files("b2aiprep").joinpath("resources")
    column_mapping: dict = json.loads(b2ai_resources.joinpath("column_mapping.json").read_text())
    # the mapping by default is {"coded_entry": "Coded Entry"}
    # we want our columns to be named "coded_entry", so we reverse the dict
    column_mapping = {v: k for k, v in column_mapping.items()}

    # only map columns if we have a full overlap with the mapping dict
    overlap_keys = set(df.columns.tolist()).intersection(set(column_mapping.keys()))
    overlap_values = set(df.columns.tolist()).intersection(set(column_mapping.values()))
    
    if len(overlap_keys) == df.shape[1]:
        _LOGGER.info("Mapping columns to coded format.")
        return df.rename(columns=column_mapping)
    elif len(overlap_values) == df.shape[1]:
        # no need to map columns
        return df
    else:
        non_overlapping_columns = set(df.columns.tolist()) - set(column_mapping.keys())
        raise ValueError(f"Found {len(non_overlapping_columns)} columns not in mapping: {non_overlapping_columns}")

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
    idx = df['redcap_repeat_instrument'] == instrument.text

    columns_present = [c for c in columns if c in df.columns]
    dff = df.loc[idx, columns_present].copy()
    if len(columns_present) < len(columns):
        _LOGGER.warning((
            f"Inserting None for missing columns of {RepeatInstrument.PARTICIPANT} instrument: "
            f"{set(columns) - set(columns_present)}"
        ))
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
        raise ValueError(f'Unrecognized {acoustic_task}. Options: {AUDIO_TASKS}')

    acoustic_tasks_df = get_df_of_repeat_instrument(df, RepeatInstrument.ACOUSTIC_TASK)
    recordings_df = get_df_of_repeat_instrument(df, RepeatInstrument.RECORDING)

    idx = acoustic_tasks_df['acoustic_task_name'] == acoustic_task

    dff = recordings_df.merge(
        acoustic_tasks_df.loc[idx, ['acoustic_task_id']],
        left_on='recording_acoustic_task_id',
        right_on='acoustic_task_id',
        how='inner'
    )
    return dff


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

    filename = f"sub-{subject_id}"
    if session_id is not None:
        session_id = _transform_str_for_bids_filename(session_id)
        filename += f"_ses-{session_id}"
    if task_name is not None:
        task_name = _transform_str_for_bids_filename(task_name)
        filename += f"_task-{task_name}"
    if recording_name is not None:
        recording_name = _transform_str_for_bids_filename(recording_name)
        filename += f"_rec-{recording_name}"


    schema_name = _transform_str_for_bids_filename(schema_name)
    filename += f"_{schema_name}.json"

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
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
        _LOGGER.warn(f"Found {df[index_col].isnull().sum()} null value(s) for {index_col}. Removing.")
        df = df.dropna(subset=[index_col])
    
    if df[index_col].nunique() < df.shape[0]:
        raise ValueError(f"Non-unique {index_col} values found.")

    # *copy* the given column into the index, preserving the original column
    # so that it is output in the later call to to_dict()
    df.index = df[index_col]

    return df.to_dict('index')

def create_file_dir(participant_id, session_id):
    """Create a file directory structure which follows the BIDS folder structure convention.
    
    BIDS folders are formatted as follows:
    base_project_folder/
    ├── sub-<participant_id>/
    │   ├── ses-<session_id>/
    │   │   ├── beh/
    │   │   │   ├── sub-<participant_id>_ses-<session_id>_task-<task_name>_rec-<recording_name>_schema.json
    │   │   │   ├── sub-<participant_id>_ses-<session_id>_task-<task_name>_rec-<recording_name>_schema.json

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

def output_participant_data_to_fhir(participant: dict, outdir: Path, audiodir: t.Optional[Path] = None):
    participant_id = participant['record_id']
    subject_path = outdir / f"sub-{participant_id}"

    if audiodir is not None and not audiodir.exists():
        audiodir = None

    # TODO: prepare a Patient resource to use as the reference for each questionnaire
    # patient = create_fhir_patient(participant)

    for questionnaire_name in GENERAL_QUESTIONNAIRES:
        instrument = get_instrument_for_name(questionnaire_name)
        fhir_data = convert_response_to_fhir(
            participant,
            questionnaire_name=questionnaire_name,
            mapping_name=instrument.schema,
            columns=instrument.columns,
        )
        if is_empty_questionnaire_response(fhir_data):
            continue

        write_pydantic_model_to_bids_file(
            subject_path,
            fhir_data,
            schema_name=questionnaire_name,
            subject_id=participant_id,
        )

    session_instrument = get_instrument_for_name("sessions")
    task_instrument = get_instrument_for_name("acoustic_tasks")
    recording_instrument = get_instrument_for_name("recordings")

    # validated questionnaires are asked per session
    for session in participant["sessions"]:
        session_id = session["session_id"]
        # TODO: prepare a session resource to use as the encounter reference for
        # each session questionnaire
        session_path = subject_path / f"ses-{session_id}"
        beh_path = session_path / 'beh'
        audio_output_path = session_path / 'audio'
        if not audio_output_path.exists():
            audio_output_path.mkdir(parents=True, exist_ok=True)
        fhir_data = convert_response_to_fhir(
            session,
            questionnaire_name=session_instrument.name,
            mapping_name=session_instrument.schema,
            columns=session_instrument.columns,
        )
        write_pydantic_model_to_bids_file(
            session_path,
            fhir_data,
            schema_name=session_instrument.schema,
            subject_id=participant_id,
            session_id=session_id
        )

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
                beh_path,
                fhir_data,
                schema_name=task_instrument.schema,
                subject_id=participant_id,
                session_id=session_id,
                task_name=acoustic_task_name,
            )
            
            # prefix is used to name audio files, if they are copied over
            prefix = f"sub-{participant_id}_ses-{session_id}_{acoustic_task_name}"

            # there may be more than one recording per acoustic task
            for recording in task["recordings"]:
                fhir_data = convert_response_to_fhir(
                    recording,
                    questionnaire_name=recording_instrument.name,
                    mapping_name=recording_instrument.schema,
                    columns=recording_instrument.columns,
                )
                write_pydantic_model_to_bids_file(
                    beh_path,
                    fhir_data,
                    schema_name=recording_instrument.schema,
                    subject_id=participant_id,
                    session_id=session_id,
                    task_name=acoustic_task_name,
                    recording_name=recording["recording_name"]
                )

                # we also need to organize the audio file
                if audiodir is not None:
                    # audio files are under a folder with the site name, so we need to recursively glob
                    audio_files = list(audiodir.rglob(f"{recording['recording_id']}*"))
                    if len(audio_files) == 0:
                        logging.warning(f"No audio file found for recording {recording['recording_id']}.")
                    else:
                        if len(audio_files) > 1:
                            logging.warning((
                                f"Multiple audio files found for recording {recording['recording_id']}."
                                f" Using only {audio_files[0]}"
                            ))
                        audio_file = audio_files[0]

                        # copy file
                        ext = audio_file.suffix
                        
                        recording_name = _transform_str_for_bids_filename(recording['recording_name'])
                        audio_file_destination = audio_output_path / f"{prefix}_rec-{recording_name}{ext}"
                        if audio_file_destination.exists():
                            logging.warning(f"Audio file {audio_file_destination} already exists. Overwriting.")
                        audio_file_destination.write_bytes(audio_file.read_bytes())
        
        # validated questionnaires
        for repeat_instrument in VALIDATED_QUESTIONNAIRES:
            if (repeat_instrument not in session) or (session[repeat_instrument] is None):
                continue
            instrument = repeat_instrument.value

            fhir_data = convert_response_to_fhir(
                session[repeat_instrument],
                questionnaire_name=instrument.name,
                mapping_name=instrument.schema,
                columns=instrument.columns,
            )

            write_pydantic_model_to_bids_file(
                beh_path,
                fhir_data,
                schema_name=instrument.schema,
                subject_id=participant_id,
                session_id=session_id
            )

def redcap_to_bids(
    filename: Path,
    outdir: Path,
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
    # for simplicity, we always map columns to coded columns before processing,
    # that way we only ever need to manually subselect using one version of the column name
    df = update_redcap_df_column_names(df)

    # the repeat instrument columns also defines all the possible
    # repeat instruments we would like to extract from the RedCap CSV
    repeat_instruments: t.List[RepeatInstrument] = list(RepeatInstrument.__members__.values())

    # we use the RepeatInstrument values to create a dict that maps the instrument
    # i.e. we end up with dataframe_dicts = {
    #       'demographics': {"session_id_1": {data}, ...}},
    #       'confounders': {"session_id_1": {data}, ...}},
    #    ...
    #   }
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

        for session in participant["sessions"]:
            # there can be multiple acoustic tasks per session
            session_id = session["session_id"]
            session["acoustic_tasks"] = acoustic_tasks_df[
                acoustic_tasks_df["acoustic_task_session_id"] == session_id
            ].to_dict("records")
            for task in session["acoustic_tasks"]:
                # there can be multiple recordings per acoustic task
                task["recordings"] = recordings_df[
                    recordings_df["recording_acoustic_task_id"]
                    == task["acoustic_task_id"]
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
