from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from pandas import DataFrame
from enum import Enum
import json
import pickle
from importlib.resources import files

import torch

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

def load_data_columns() -> Dict[str, List[str]]:  
    b2ai_resources = files("b2aiprep")
    columns = json.loads(b2ai_resources.joinpath("resources", "columns.json").read_text())
    confounders_columns = json.loads(b2ai_resources.joinpath("resources", "confounders.json").read_text())
    demographics_columns = json.loads(b2ai_resources.joinpath("resources", "demographics.json").read_text())
    gad7_columns = json.loads(b2ai_resources.joinpath("resources", "gad7.json").read_text())
    participant_columns = json.loads(b2ai_resources.joinpath("resources", "participant.json").read_text())
    phq9_columns = json.loads(b2ai_resources.joinpath("resources", "phq9.json").read_text())
    voice_perception_columns = json.loads(b2ai_resources.joinpath("resources", "voice_perception.json").read_text())
    vhi10_columns = json.loads(b2ai_resources.joinpath("resources", "vhi10.json").read_text())
    panas_columns = json.loads(b2ai_resources.joinpath("resources", "panas.json").read_text())
    custom_affect_scale_columns = json.loads(b2ai_resources.joinpath("resources", "custom_affect_scale.json").read_text())
    dsm5_columns = json.loads(b2ai_resources.joinpath("resources", "dsm5.json").read_text())
    ptsd_columns = json.loads(b2ai_resources.joinpath("resources", "ptsd.json").read_text())
    adhd_columns = json.loads(b2ai_resources.joinpath("resources", "adhd.json").read_text())
    dyspnea_columns = json.loads(b2ai_resources.joinpath("resources", "dyspnea.json").read_text())
    dyspnea_columns = json.loads(b2ai_resources.joinpath("resources", "dyspnea.json").read_text())
    leicester_cough_columns = json.loads(b2ai_resources.joinpath("resources", "leicester_cough.json").read_text())
    voice_problem_severity_columns = json.loads(b2ai_resources.joinpath("resources", "voice_problem_severity.json").read_text())
    winograd_columns =  json.loads(b2ai_resources.joinpath("resources", "winograd.json").read_text())
    stroop_columns =  json.loads(b2ai_resources.joinpath("resources", "stroop.json").read_text())
    vocab_columns = json.loads(b2ai_resources.joinpath("resources", "vocab.json").read_text())
    random_columns = json.loads(b2ai_resources.joinpath("resources", "random.json").read_text())

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
    
def get_columns_of_repeat_instrument(repeat_instrument: RepeatInstrument) -> List[str]:
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

def get_df_of_repeat_instrument(df: DataFrame, repeat_instrument: RepeatInstrument) -> pd.DataFrame:
    columns = get_columns_of_repeat_instrument(repeat_instrument)
    if repeat_instrument in (RepeatInstrument.PARTICIPANT,):
        idx = df['redcap_repeat_instrument'].isnull()
    else:
        idx = df['redcap_repeat_instrument'] == repeat_instrument.value
    return df.loc[idx, columns].copy()

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

def load_features_for_recordings(df: pd.DataFrame, data_path: Path, feature: Optional[str] = None) -> Dict[str, torch.Tensor]:
    output = {}
    feature_options = (
        'specgram', 'melfilterbank', 'mfcc', 'opensmile',
        'sample_rate', 'checksum', 'transcription'
    )
    if feature is not None:
        if feature not in feature_options:
            raise ValueError(f'Unrecognized feature {feature}. Options: {feature_options}')

    for recording_id in df['recording_id'].unique():
        output[recording_id] = torch.load(data_path / f"{recording_id}_features.pt")

        # if requested, we subselect to the given feature
        if feature is not None:
            output[recording_id] = output[recording_id][feature]

    return output
