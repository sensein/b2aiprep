"""Reorganize RedCap CSV and voice waveforms into a BIDS-like structure."""
import logging
from pathlib import Path
import typing as t

import pandas as pd
from pandas import DataFrame
from enum import Enum
import json
from importlib.resources import files
from pydantic import BaseModel
from tqdm import tqdm

from b2aiprep.fhir_utils import (
    convert_response_to_fhir,
)

_LOGGER = logging.getLogger(__name__)

_GENERAL_QUESTIONNAIRES = [
    "participant",
    "eligibility",
    "enrollment",
    "vocalFoldParalysis",
    "laryngealdystonia",
    "precancerouseLesions",
    "laryngealcancer",
    "benignLesion",
    "bipolar",
    "depression",
    "airwaystenosis",
    "alzheimers",
    "parkinsons",
    "als"
]

_VALIDATED_QUESTIONNAIRES = [
    "demographics",
    "confounders",
    "gad7",
    "phq9",
    "voicePerception",
    "vhi10",
    "panas",
    "customAffectScale",
    "dsm5",
    "ptsd",
    "adhd",
    "dyspnea",
    "leicester",
    "voiceSeverity",
    "winograd",
    "stroop",
    "vocab",
    "random"

]
_AUDIO_TASKS = (
    'Animal fluency', 'Audio Check', 'Breath Sounds',
    'Cape V sentences', 'Caterpillar Passage', 'Cinderella Story',
    'Diadochokinesis', 'Free Speech', 'Glides', 'Loudness',
    'Maximum phonation time', 'Picture description',
    'Productive Vocabulary', 'Prolonged vowel', 'Rainbow Passage',
    'Random Item Generation', 'Respiration and cough', 'Story recall',
    'Voluntary Cough', 'Word-color Stroop',
)

class RepeatInstrument(Enum):
    SESSION = 'Session'
    ACOUSTIC_TASK = 'Acoustic Task'
    RECORDING = 'Recording'
    PARTICIPANT = 'Participant'
    GENERIC_DEMOGRAPHICS = 'Q Generic Demographics'
    GENERIC_CONFOUNDERS = 'Q Generic Confounders'
    GENERIC_VOICE_PERCEPTION = 'Q Generic Voice Perception'
    GENERIC_VOICE_HANDICAP = 'Q Generic Voice Handicap Index Vhi10'
    GENERIC_PHQ9_DEPRESSION = 'Q Generic Patient Health Questionnaire9'
    GENERIC_GAD7_ANXIETY = 'Q - Generic - Gad7 Anxiety'
    RESP_DYSPNEA_INDEX = 'Q - Resp - Dyspnea Index Di'
    RESP_LEICESTER_COUGH = 'Q - Resp - Leicester Cough Questionnaire Lcq'
    VOICE_VOICE_PROBLEM_SEVERITY = 'Q - Voice - Voice Problem Severity'
    MOOD_PANAS = 'Q - Mood - Panas'
    MOOD_CUSTOM_AFFECT = 'Q - Mood - Custom Affect Scale'
    MOOD_DSM5_ADULT = 'Q - Mood - Dsm5 Adult'
    MOOD_PTSD_ADULT = 'Q - Mood - Ptsd Adult'
    MOOD_ADHD_ADULT = 'Q - Mood - Adhd Adult'
    NEURO_WINOGRAD_SCHEMAS = 'Q - Neuro Winograd Schemas'
    NEURO_WORDCOLOR_STROOP = 'Q - Neuro - Wordcolor Stroop'
    NEURO_PRODUCTIVE_VOCABULARY = 'Q - Neuro - Productive Vocabulary'
    NEURO_RANDOM_ITEM_GENERATION = 'Q - Neuro - Random Item Generation'

def load_csv_file(file_path):
    try:
        data = pd.read_csv(file_path, low_memory=False, na_values='')
        return data
    except FileNotFoundError:
        print('File not found.')
        return None
    except pd.errors.ParserError:
        print('Error parsing CSV file.')
        return None

def load_data_columns() -> t.Dict[str, t.List[str]]:  
    b2ai_resources = files("b2aiprep").joinpath("resources")
    columns = json.loads(b2ai_resources.joinpath("all_columns.json").read_text())

    b2ai_resources = b2ai_resources.joinpath("instrument_columns")
    confounders_columns = json.loads(b2ai_resources.joinpath("confounders.json").read_text())
    demographics_columns = json.loads(b2ai_resources.joinpath("demographics.json").read_text())
    gad7_columns = json.loads(b2ai_resources.joinpath("gad7.json").read_text())
    participant_columns = json.loads(b2ai_resources.joinpath("participant.json").read_text())
    phq9_columns = json.loads(b2ai_resources.joinpath("phq9.json").read_text())
    voice_perception_columns = json.loads(b2ai_resources.joinpath("voice_perception.json").read_text())
    vhi10_columns = json.loads(b2ai_resources.joinpath("vhi10.json").read_text())
    panas_columns = json.loads(b2ai_resources.joinpath("panas.json").read_text())
    custom_affect_scale_columns = json.loads(b2ai_resources.joinpath("custom_affect_scale.json").read_text())
    dsm5_columns = json.loads(b2ai_resources.joinpath("dsm5.json").read_text())
    ptsd_columns = json.loads(b2ai_resources.joinpath("ptsd.json").read_text())
    adhd_columns = json.loads(b2ai_resources.joinpath("adhd.json").read_text())
    dyspnea_columns = json.loads(b2ai_resources.joinpath("dyspnea.json").read_text())
    leicester_cough_columns = json.loads(b2ai_resources.joinpath("leicester_cough.json").read_text())
    voice_problem_severity_columns = json.loads(b2ai_resources.joinpath("voice_problem_severity.json").read_text())
    winograd_columns =  json.loads(b2ai_resources.joinpath("winograd.json").read_text())
    stroop_columns =  json.loads(b2ai_resources.joinpath("stroop.json").read_text())
    vocab_columns = json.loads(b2ai_resources.joinpath("vocab.json").read_text())
    random_columns = json.loads(b2ai_resources.joinpath("random.json").read_text())

    data_columns = {
        'columns': columns,
        'participant_columns': participant_columns,
        'demographics_columns': demographics_columns,
        'confounders_columns': confounders_columns,
        'phq9_columns': phq9_columns,
        'gad7_columns': gad7_columns,
        'voice_perception_columns': voice_perception_columns,
        'vhi10_columns': vhi10_columns,
        'panas_columns': panas_columns,
        "custom_affect_scale_columns": custom_affect_scale_columns,
        "dsm5_columns" : dsm5_columns,
        "ptsd_columns" : ptsd_columns,
        "adhd_columns" : adhd_columns,
        "dyspnea_columns": dyspnea_columns,
        "leicester_cough_columns": leicester_cough_columns,
        "voice_problem_severity_columns": voice_problem_severity_columns,
        'winograd_columns': winograd_columns,
        "stroop_columns" : stroop_columns,
        "vocab_columns" : vocab_columns,
        "random_columns" : random_columns
    }

    return data_columns
    
def get_columns_of_repeat_instrument(repeat_instrument: RepeatInstrument) -> t.List[str]:
    data_columns = load_data_columns()
    repeat_instrument_columns = {
        RepeatInstrument.PARTICIPANT: 'participant_columns',
        RepeatInstrument.GENERIC_DEMOGRAPHICS: 'demographics_columns',
        RepeatInstrument.GENERIC_CONFOUNDERS: 'confounders_columns',
        RepeatInstrument.GENERIC_PHQ9_DEPRESSION: 'phq9_columns',
        RepeatInstrument.GENERIC_GAD7_ANXIETY: 'gad7_columns',
        RepeatInstrument.GENERIC_VOICE_PERCEPTION: 'voice_perception_columns',
        RepeatInstrument.GENERIC_VOICE_HANDICAP: 'vhi10_columns',
        RepeatInstrument.MOOD_PANAS: 'panas_columns',
        RepeatInstrument.MOOD_CUSTOM_AFFECT: 'custom_affect_scale_columns',
        RepeatInstrument.MOOD_DSM5_ADULT: 'dsm5_columns',
        RepeatInstrument.MOOD_PTSD_ADULT: 'ptsd_columns',
        RepeatInstrument.MOOD_ADHD_ADULT: 'adhd_columns',
        RepeatInstrument.RESP_DYSPNEA_INDEX: 'dyspnea_columns',
        RepeatInstrument.RESP_LEICESTER_COUGH: 'leicester_cough_columns',
        RepeatInstrument.VOICE_VOICE_PROBLEM_SEVERITY: 'voice_problem_severity_columns',
        RepeatInstrument.NEURO_WINOGRAD_SCHEMAS: 'winograd_columns',
        RepeatInstrument.NEURO_WORDCOLOR_STROOP: 'stroop_columns',
        RepeatInstrument.NEURO_PRODUCTIVE_VOCABULARY : 'vocab_columns',
        RepeatInstrument.NEURO_RANDOM_ITEM_GENERATION : 'random_columns'
        
    }

    repeat_instrument_prefix_mapping = {
        RepeatInstrument.SESSION: 'session_',
        RepeatInstrument.ACOUSTIC_TASK: 'acoustic_task_',
        RepeatInstrument.RECORDING: 'recording_',
    }
    if repeat_instrument in repeat_instrument_columns:
        columns = data_columns[repeat_instrument_columns[repeat_instrument]]
    elif repeat_instrument in repeat_instrument_prefix_mapping:
        columns = [c for c in data_columns['columns'] if c.startswith(repeat_instrument_prefix_mapping[repeat_instrument])]
    else:
        # add rest of columns for other repeat instruments
        raise NotImplementedError('Repeat Instrument not implemented.')

    if 'record_id' not in columns:
        columns.insert(0, 'record_id')
    return columns

def update_column_names(df: DataFrame) -> DataFrame:
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

def get_df_of_repeat_instrument(df: DataFrame, repeat_instrument: RepeatInstrument) -> pd.DataFrame:
    columns = get_columns_of_repeat_instrument(repeat_instrument)
    if repeat_instrument in (RepeatInstrument.PARTICIPANT,):
        idx = df['redcap_repeat_instrument'].isnull()
    else:
        idx = df['redcap_repeat_instrument'] == repeat_instrument.value
        
    columns_present = [c for c in columns if c in df.columns]
    dff = df.loc[idx, columns_present].copy()
    if len(columns_present) < len(columns):
        _LOGGER.warning((
            f"Inserting None for missing columns of {RepeatInstrument.PARTICIPANT} instrument: "
            f"{set(columns) - set(columns_present)}"
        ))
        for c in set(columns) - set(columns_present):
            dff[c] = None
    return dff

def get_recordings_for_acoustic_task(df: pd.DataFrame, acoustic_task: str) -> pd.DataFrame:
    if acoustic_task not in _AUDIO_TASKS:
        raise ValueError(f'Unrecognized {acoustic_task}. Options: {_AUDIO_TASKS}')

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

def _transform_str_for_filename(filename: str):
    return filename.replace(' ', '-')


def write_pydantic_model_to_bids_file(
        output_path: Path,
        data: BaseModel,
        schema_name: str,
        subject_id: str,
        session_id: t.Optional[str] = None,
        task_name: t.Optional[str] = None,
        recording_name: t.Optional[str] = None,
        ):
    filename = f"sub-{subject_id}"
    if session_id is not None:
        session_id = _transform_str_for_filename(session_id)
        filename += f"_ses-{session_id}"
    if task_name is not None:
        task_name = _transform_str_for_filename(task_name)
        filename += f"_task-{task_name}"
    if recording_name is not None:
        recording_name = _transform_str_for_filename(recording_name)
        filename += f"_rec-{recording_name}"


    schema_name = _transform_str_for_filename(schema_name)
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
    output_path = Path("./output/")
    paths = [
        output_path / f"sub-{participant_id}/",
        output_path / f"sub-{participant_id}/ses-{session_id}/",
        output_path / f"sub-{participant_id}/ses-{session_id}/beh/",
    ]

    if session_id == "":
        paths = paths[0:1]

    for folder in paths:
        folder.mkdir(parents=True, exist_ok=True)


def questionnaire_mapping(questionnaire_name):
    with open("questionnaire_mapping/mappings.json") as mapping_file:
        mapping_str = mapping_file.read()
        mapping_data = json.loads(mapping_str)
        return mapping_data[questionnaire_name]

def output_participant_data_to_fhir(participant: dict, outdir: Path, audiodir: t.Optional[Path] = None):
    participant_id = participant['record_id']
    subject_path = outdir / f"sub-{participant_id}"

    if audiodir is not None and not audiodir.exists():
        audiodir = None

    # TODO: prepare a Patient resource to use as the reference for each questionnaire
    # patient = create_fhir_patient(participant)

    for questionnaire_name in _GENERAL_QUESTIONNAIRES:
        # given a dictionary of individual data, this function:
        # (1) identifies specific data elements using the questionnaire name
        # (2) extracts the responses associated with these elements
        # (3) reformats the data into a QuestionnaireResponse object and returns it
        fhir_data = convert_response_to_fhir(participant, questionnaire_name)
        if len(fhir_data.item) == 0:
            continue
        write_pydantic_model_to_bids_file(
            subject_path,
            fhir_data,
            schema_name=questionnaire_name,
            subject_id=participant_id,
        )

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
            questionnaire_name="sessions",
        )
        write_pydantic_model_to_bids_file(
            session_path,
            fhir_data,
            schema_name="sessionschema",
            subject_id=participant_id,
            session_id=session_id
        )

        # multiple acoustic tasks are asked per session
        for task in session["acoustic_tasks"]:
            if task is None:
                continue
            
            # replace spaces for the file name
            acoustic_task_name = _transform_str_for_filename(task["acoustic_task_name"])
            fhir_data = convert_response_to_fhir(
                task,
                "acoustic_tasks",
            )
            write_pydantic_model_to_bids_file(
                beh_path,
                fhir_data,
                schema_name="acoustictaskschema",
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
                    "recordings",
                )
                write_pydantic_model_to_bids_file(
                    beh_path,
                    fhir_data,
                    schema_name="recordingschema",
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
                        
                        recording_name = _transform_str_for_filename(recording['recording_name'])
                        audio_file_destination = audio_output_path / f"{prefix}_rec-{recording_name}{ext}"
                        if audio_file_destination.exists():
                            logging.warning(f"Audio file {audio_file_destination} already exists. Overwriting.")
                        audio_file_destination.write_bytes(audio_file.read_bytes())
        
        # validated questionnaires
        for questionnaire_name in _VALIDATED_QUESTIONNAIRES:
            if (questionnaire_name not in session) or (session[questionnaire_name] is None):
                continue

            fhir_data = convert_response_to_fhir(
                session[questionnaire_name],
                questionnaire_name,
            )

            write_pydantic_model_to_bids_file(
                beh_path,
                fhir_data,
                schema_name=f"{questionnaire_name}",
                subject_id=participant_id,
                session_id=session_id
            )

def redcap_to_bids(
    filename: Path,
    outdir: Path,
    audiodir: t.Optional[Path] = None,
):
    """Converts from RedCap to a BIDS-like structure."""
    df = load_csv_file(filename)

    # It is possible for each column in the dataframe to have one of two names:
    #   1. coded column names ("record_id")
    #   2. text column names ("Record ID")
    # for simplicity, we always map columns to coded columns before processing,
    # that way we only ever need to manually subselect using one version of the column name
    df = update_column_names(df)

    # create separate data frames for sets of columns
    # number of participants
    participants_df = get_df_of_repeat_instrument(
        df, RepeatInstrument.PARTICIPANT
    )
    _LOGGER.info(f"Number of participants: {len(participants_df)}")

    # session info
    sessions_df = get_df_of_repeat_instrument(df, RepeatInstrument.SESSION)
    _LOGGER.info(f"Number of sessions: {len(sessions_df)}")

    # subject id (record_id) to accoustic_task_id and accoustic_task_name
    acoustic_tasks_df = get_df_of_repeat_instrument(
        df, RepeatInstrument.ACOUSTIC_TASK
    )
    _LOGGER.info(f"Number of Acoustic Tasks: {len(acoustic_tasks_df)}")

    # recording info
    recordings_df = get_df_of_repeat_instrument(
        df, RepeatInstrument.RECORDING
    )
    _LOGGER.info(f"Number of Recordings: {len(recordings_df)}")

    # load in questionnaires based on the repeat instruments
    dataframes_with_columns = {
        'demographics': ('demographics_session_id', RepeatInstrument.GENERIC_DEMOGRAPHICS),
        'confounders': ('confounders_session_id', RepeatInstrument.GENERIC_CONFOUNDERS),
        'gad7': ('gad_7_session_id', RepeatInstrument.GENERIC_GAD7_ANXIETY),
        'phq9': ('phq_9_session_id', RepeatInstrument.GENERIC_PHQ9_DEPRESSION),
        'voicePerception': ('voice_perception_session_id', RepeatInstrument.GENERIC_VOICE_PERCEPTION),
        'vhi10': ('vhi_session_id', RepeatInstrument.GENERIC_VOICE_HANDICAP),
        'panas': ('panas_session_id', RepeatInstrument.MOOD_PANAS),
        'customAffectScale': ('custom_affect_scale_session_id', RepeatInstrument.MOOD_CUSTOM_AFFECT),
        'dsm5': ('dsm_5_session_id', RepeatInstrument.MOOD_DSM5_ADULT),
        'ptsd': ('ptsd_session_id', RepeatInstrument.MOOD_PTSD_ADULT),
        'adhd': ('adhd_session_id', RepeatInstrument.MOOD_ADHD_ADULT),
        'dyspnea': ('dyspnea_index_session_id', RepeatInstrument.RESP_DYSPNEA_INDEX),
        'leicester': ('leicester_cough_session_id', RepeatInstrument.RESP_LEICESTER_COUGH),
        'voiceSeverity': ('voice_severity_session_id', RepeatInstrument.VOICE_VOICE_PROBLEM_SEVERITY),
        'winograd': ('winograd_session_id', RepeatInstrument.NEURO_WINOGRAD_SCHEMAS),
        'stroop': ('stroop_session_id', RepeatInstrument.NEURO_WORDCOLOR_STROOP),
        'vocab': ('vocabulary_session_id', RepeatInstrument.NEURO_PRODUCTIVE_VOCABULARY),
        'random': ('random_session_id', RepeatInstrument.NEURO_RANDOM_ITEM_GENERATION),
    }

    # the above dictionary maps the session_id column to an enum
    # we use these values to create a dict that is {session_id: {data ...}}
    # i.e. we end up with dataframe_dicts = {
    #       'demographics': {"session_id_1": {data}, ...}},
    #       'confounders': {"session_id_1": {data}, ...}},
    #    ...
    #   }
    dataframe_dicts = {}
    for questionnaire_name, (session_id_col, repeat_instrument) in dataframes_with_columns.items():
        # load in the df based on the instrument
        questionnaire_df = get_df_of_repeat_instrument(df, repeat_instrument)
        _LOGGER.info(f"Number of {questionnaire_name} entries: {len(questionnaire_df)}")
        # convert the df to a dictionary
        dataframe_dicts[questionnaire_name] = _df_to_dict(questionnaire_df, session_id_col)

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
