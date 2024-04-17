import pandas as pd
from pandas import DataFrame
from enum import Enum
import json
import pickle


class RepeatInstrument(Enum):
    SESSION = 'Session'
    ACOUSTIC_TASK = 'Acoustic Task'
    RECORDING = 'Recording'
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

def pickle_new_data_columns(category: str, column_names: list):
    
    out_file = f'{category}.pkl'
    with open(out_file, "wb") as f:
        pickle.dump(column_names, f)
    print(f'New column names saved to: {out_file}')
    return

def load_data_columns():       
    data_columns = {}
    with open('participant_columns.pkl', "rb") as f:
        data_columns['participant_columns'] = pickle.load(f)
    
    with open('demographics_columns.pkl', "rb") as f:
        data_columns['demographics_columns'] = pickle.load(f)
    
    with open('confounders_columns.pkl', "rb") as f:
        data_columns['confounders_columns'] = pickle.load(f)

    with open('phq9_columns.pkl', "rb") as f:
        data_columns['phq9_columns'] = pickle.load(f)
    
    with open('gad7_columns.pkl', "rb") as f:
        data_columns['gad7_columns'] = pickle.load(f)

    return data_columns
    
def get_columns_of_repeat_instrument(df: DataFrame, repeat_instrument: RepeatInstrument):
    columns = pd.Index(['record_id'])
    
    data_columns = load_data_columns()
    participant_columns=data_columns['participant_columns']
    demographics_columns=data_columns['demographics_columns']
    confounders_columns=data_columns['confounders_columns']
    phq9_columns=data_columns['phq9_columns']
    gad7_columns=data_columns['gad7_columns']
    
    if repeat_instrument == RepeatInstrument.SESSION:
        columns = columns.append(df.columns[pd.Series(df.columns).str.startswith('session_')])
    elif repeat_instrument == RepeatInstrument.ACOUSTIC_TASK:
        columns = columns.append(df.columns[pd.Series(df.columns).str.startswith('acoustic_task_')])
    elif repeat_instrument == RepeatInstrument.RECORDING:
        columns = columns.append(df.columns[pd.Series(df.columns).str.startswith('recording_')])
    elif repeat_instrument == RepeatInstrument.GENERIC_DEMOGRAPHICS:
        columns = columns.append(pd.Index(demographics_columns))
    elif repeat_instrument == RepeatInstrument.GENERIC_CONFOUNDERS:
        columns = columns.append(pd.Index(confounders_columns))
    elif repeat_instrument == RepeatInstrument.GENERIC_PHQ9_DEPRESSION:
        columns = columns.append(pd.Index(phq9_columns))
    elif repeat_instrument == RepeatInstrument.GENERIC_GAD7_ANXIETY:
        columns = columns.append(pd.Index(gad7_columns))
    else:
        # add rest of columns for other repeat instruments
        raise NotImplementedError('Repeat Instrument not implemented.')
    return columns

def get_df_of_repeat_instrument(df: DataFrame, repeat_instrument: RepeatInstrument):
    return df[df['redcap_repeat_instrument'] == repeat_instrument.value][get_columns_of_repeat_instrument(df, repeat_instrument)]

