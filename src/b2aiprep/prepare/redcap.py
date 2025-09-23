"""
RedCapDataset: A centralized module for parsing data from RedCap and ReproSchema sources.

This module provides a unified interface for ingesting data from different sources
and converting them to a standardized format for BIDS export. It supports:
- RedCap CSV files (direct ingestion)
- ReproSchema data (conversion to RedCap format then processing)

The RedCapDataset class handles validation, imputation, and export functionality
that is common to both data sources.
"""

import json
import logging
import os
import re
import typing as t
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from importlib.resources import files
from tempfile import NamedTemporaryFile

import requests

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from b2aiprep.prepare.constants import AUDIO_TASKS, Instrument, RepeatInstrument
from b2aiprep.prepare.utils import fetch_json_options_number, get_wav_duration

# ReproSchema parsing functions (moved from reproschema_to_redcap.py)

def get_choice_name(url, value):
    """Get the choice name from a ReproSchema URL and value."""
    response = requests.get(url)

    if response.status_code != 200:
        return value

    data = response.json()
    choices = data.get("responseOptions", {}).get("choices", [])

    for choice in choices:
        if choice.get("value") == value:
            name = choice.get("name", None)
            if isinstance(name, dict) and "en" in name:
                return name.get("en", value)
            else:
                return value

    return value


def parse_survey(survey_data, record_id, session_path):
    """
    Function that generates a list of data frames in order to generate a redcap csv
    Args:
        survey_data is the raw json generated from reproschema ui
        record_id is the id that identifies the participant
        session_path is the path containing the session id
    """
    session_id = session_path.split("/")[0]
    questionnaire_name = survey_data[0]["used"][1].split("/")[-1]
    questions_answers = dict()
    questions_answers["record_id"] = [record_id]
    questions_answers["redcap_repeat_instrument"] = [questionnaire_name]
    questions_answers["redcap_repeat_instance"] = [1]
    start_time = survey_data[0]["startedAtTime"]
    end_time = survey_data[0]["endedAtTime"]
    for i in range(len(survey_data)):
        if survey_data[i]["@type"] == "reproschema:Response":
            question = survey_data[i]["isAbout"].split("/")[-1]
            # answer = survey_data[i]["value"]
            answer = get_choice_name(survey_data[i]["isAbout"], survey_data[i]["value"])
            if not isinstance(answer, list):
                questions_answers[question] = [str(answer).capitalize()]

            else:
                num = fetch_json_options_number(survey_data[i]["isAbout"])

                for options in range(num):
                    if options in answer:
                        questions_answers[f"""{question}___{options}"""] = ["Checked"]

                    else:
                        questions_answers[f"""{question}___{options}"""] = ["Unchecked"]

        else:
            end_time = survey_data[i]["endedAtTime"]
    # Adding metadata values for redcap
    questions_answers[f"{questionnaire_name}_start_time"] = [start_time]
    questions_answers[f"{questionnaire_name}_end_time"] = [end_time]

    # Convert the time strings to datetime objects with UTC format
    time_format = "%Y-%m-%dT%H:%M:%S.%fZ"
    start = datetime.strptime(start_time, time_format)
    end = datetime.strptime(end_time, time_format)
    duration = end - start
    # convert to milliseconds
    duration = duration.microseconds // 1000

    questions_answers[f"{questionnaire_name}_duration"] = [duration]

    questions_answers[f"{questionnaire_name}_session_id"] = [session_id]

    df = pd.DataFrame(questions_answers)
    return [df]


def parse_audio(audio_list, dummy_audio_files=False):
    """
    Function that generates a list of Json's to be converted into a redcap csv based on audio files.
    Args:
        audio_list is a list of paths to each audio files
        dummy_audio_files is an optional variable for testing
    """
    protocol_order = {
        "ready_for_school": [],
        "favorite_show_movie": [],
        "favorite_food": [],
        "outside_of_school": [],
        "abcs": [],
        "123s": [],
        "naming_animals": [],
        "role_naming": [],
        "naming_food": [],
        "noisy_sounds": [],
        "long_sounds": [],
        "silly_sounds": [],
        "picture": [],
        "sentence": [],
        "picture_description": [],
        "passage": [],
        "other": [],
    }

    for name in audio_list:
        found = False
        for order in protocol_order:
            if order in name:
                protocol_order[order].append(name)
                found = True
                break
        if not found:
            protocol_order["other"].append(name)

    for questions in protocol_order:
        protocol_order[questions] = sorted(protocol_order[questions])

    flattened_list = [value for key in protocol_order for value in protocol_order[key]]

    audio_output_list = []
    count = 1
    acoustic_task_count = 1
    acoustic_count = 1
    acoustic_prev = None
    acoustic_tasks = set()
    # acoustic_task_dict = dict()
    for file_path in flattened_list:
        record_id = Path(file_path).parent.parent.name
        session = Path(file_path).parent.name
        if dummy_audio_files:
            duration = 0
            file_size = 0
        else:
            duration = get_wav_duration(file_path)
            file_size = Path(file_path).stat().st_size
        file_name = file_path.split("/")[-1]
        recording_id = re.search(r"([a-f0-9\-]{36})\.", file_name).group(1)
        acoustic_task = re.search(r"^(.*?)(_\d+)", file_name).group(1)
        if acoustic_task not in acoustic_tasks:

            acoustic_tasks.add(acoustic_task)
            acoustic_task_dict = {
                "record_id": record_id,
                "redcap_repeat_instrument": "Acoustic Task",
                "redcap_repeat_instance": acoustic_task_count,
                "acoustic_task_id": f"{acoustic_task}-{session}",
                "acoustic_task_session_id": session,
                "acoustic_task_name": acoustic_task,
                "acoustic_task_cohort": "Pediatrics",
                "acoustic_task_status": "Completed",
                "acoustic_task_duration": duration,
            }
            audio_output_list.append(acoustic_task_dict)
            acoustic_task_count += 1
        else:
            for index in audio_output_list:
                if "acoustic_task_id" in index:
                    if acoustic_task == index["acoustic_task_name"]:
                        index["acoustic_task_duration"] += duration

        if acoustic_prev != acoustic_task:
            acoustic_count = 1
        file_dict = {
            "record_id": record_id,
            "redcap_repeat_instrument": "Recording",
            "redcap_repeat_instance": count,
            "recording_id": recording_id,
            "recording_acoustic_task_id": f"{acoustic_task}-{session}",
            "recording_session_id": session,
            "recording_name": f"{acoustic_task}-{acoustic_count}",
            "recording_duration": duration,
            "recording_size": file_size,
            "recording_profile_name": "Speech",
            "recording_profile_version": "v1.0.0",
            "recording_input_gain": 0.0,
            "recording_microphone": "USB-C to 3.5mm Headphone Jack Adapter",
        }

        acoustic_prev = acoustic_task
        acoustic_count += 1
        audio_output_list.append(file_dict)
        count += 1

    return audio_output_list


_LOGGER = logging.getLogger(__name__)


class RedCapDataset:
    """
    A centralized class for parsing and managing data from RedCap and ReproSchema sources.
    
    This class provides a unified interface for:
    - Loading data from different sources (RedCap CSV, ReproSchema)
    - Validating and imputing missing data
    - Converting data to BIDS format
    - Exporting processed data
    
    Attributes:
        df: The main DataFrame containing all parsed data
        source_type: The type of data source ('redcap' or 'reproschema')
        metadata: Additional metadata about the dataset
    """
    
    def __init__(self, df: DataFrame = None, source_type: str = None, metadata: dict = None):
        """
        Initialize a RedCapDataset instance.
        
        Args:
            df: Optional DataFrame containing the data
            source_type: The source type ('redcap' or 'reproschema')
            metadata: Optional metadata dictionary
        """
        self.df = df if df is not None else pd.DataFrame()
        self.source_type = source_type
        self.metadata = metadata if metadata is not None else {}
        
    @classmethod
    def from_redcap(cls, csv_path: t.Union[str, Path]) -> 'RedCapDataset':
        """
        Create a RedCapDataset from a RedCap CSV file.
        
        Args:
            csv_path: Path to the RedCap CSV file
            
        Returns:
            RedCapDataset instance loaded with RedCap data
            
        Raises:
            FileNotFoundError: If the CSV file doesn't exist
            ValueError: If the CSV format is invalid
        """
        _LOGGER.info(f"Loading RedCap data from {csv_path}")

        df = cls._load_redcap_csv(csv_path)
        dataset = cls(df=df, source_type='redcap', metadata={'source_file': str(csv_path)})
        
        # Validate and process the data
        dataset._validate_redcap_columns()
        dataset._insert_missing_columns()
        
        _LOGGER.info(f"Successfully loaded RedCap data with {len(df)} rows and {len(df.columns)} columns")
        return dataset
        
    @classmethod
    def from_reproschema(
        cls, 
        audio_dir: t.Union[str, Path], 
        survey_dir: t.Union[str, Path],
        participant_group: str = None
    ) -> 'RedCapDataset':
        """
        Create a RedCapDataset from ReproSchema data.
        
        Args:
            audio_dir: Path to directory containing audio files
            survey_dir: Path to directory containing survey data
            participant_group: Optional participant group identifier
            
        Returns:
            RedCapDataset instance loaded with converted ReproSchema data
            
        Raises:
            FileNotFoundError: If directories don't exist
            ValueError: If ReproSchema format is invalid
        """
        _LOGGER.info(f"Loading ReproSchema data from audio_dir: {audio_dir}, survey_dir: {survey_dir}")
    
        # Convert ReproSchema to RedCap format
        df = cls._convert_reproschema_to_redcap(audio_dir, survey_dir, participant_group)

        #remap reproschema columns to their redcap equivalents
        current_file = Path(__file__)
        resource_path = current_file.parent / "resources" / "field_remap.json"

        with open(resource_path, "r") as f:
            column_mapping = json.load(f)

        df.rename(columns=column_mapping, inplace=True)

        dataset = cls(
            df=df, 
            source_type='reproschema', 
            metadata={
                'audio_dir': str(audio_dir),
                'survey_dir': str(survey_dir),
                'participant_group': participant_group
            }
        )
        # Apply standard processing
        dataset._insert_missing_columns()
        
        _LOGGER.info(f"Successfully loaded ReproSchema data with {len(df)} rows and {len(df.columns)} columns")
        return dataset
    
    @staticmethod
    def _load_redcap_csv(file_path: t.Union[str, Path]) -> DataFrame:
        """
        Load the RedCap CSV file into a DataFrame.
        
        Makes modifications to the data to ensure compatibility with downstream tooling.
        
        Args:
            file_path: The path to the CSV file.
            
        Returns:
            The loaded data.
            
        Raises:
            FileNotFoundError: If file doesn't exist
            pd.errors.ParserError: If CSV parsing fails
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")

        data = pd.read_csv(file_path, low_memory=False, na_values="")
        
        # Determine the repeat instrument column name
        if "redcap_repeat_instrument" in data.columns:
            col = "redcap_repeat_instrument"
        elif "Repeat Instrument" in data.columns:
            col = "Repeat Instrument"
        else:
            raise ValueError("No repeat instrument column found in CSV file.")

        # Fill missing values for the Participant instrument
        participant_instrument = RepeatInstrument.PARTICIPANT.value
        data[col] = data[col].fillna(participant_instrument.text).astype(str)
        
        return data
    
    @staticmethod
    def _convert_reproschema_to_redcap(
        audio_dir: t.Union[str, Path], 
        survey_dir: t.Union[str, Path], 
        participant_group: str = None
    ) -> DataFrame:
        """
        Convert ReproSchema data to RedCap format.
        
        Args:
            audio_dir: Path to audio files directory
            survey_dir: Path to survey data directory  
            participant_group: Optional participant group identifier
            
        Returns:
            DataFrame in RedCap format
        """
        folder = Path(survey_dir)
        sub_folders = sorted([str(f) for f in folder.iterdir() if f.is_dir()])

        if not folder.is_dir():
            raise FileNotFoundError(f"Survey directory {folder} does not exist.")

        merged_questionnaire_data = []
        subject_count = 1
        
        # Process survey data
        for subject in sub_folders:
            content = OrderedDict()
            content["record_id"] = subject_count
            subject_count += 1
            
            # Parse sessions within each subject
            sessions = Path(subject).iterdir()
            for session in sessions:
                if session.is_dir():
                    session_files = list(session.glob("*.jsonld"))
                    for session_file in session_files:
                        try:
                            with open(session_file, 'r') as f:
                                session_data = json.load(f)
                        except json.JSONDecodeError as e:
                            _LOGGER.warning(f"Failed to decode session file {session_file}: {str(e)}")
                            continue
                        record_id = (subject.split("/")[-1]).split()[0]
                        session_path = str(session.relative_to(subject))
                        questionnaire_df_list = parse_survey(session_data, record_id, session_path)

                        for df in questionnaire_df_list:
                            merged_questionnaire_data.append(df)

                    session_id = [f.name for f in os.scandir(subject) if f.is_dir()]
                    session_df = pd.DataFrame(
                        {
                            "record_id": (subject.split("/")[-1]).split()[0],
                            "redcap_repeat_instrument": ["Session"],
                            "redcap_repeat_instance": [1],
                            "session_id": session_id,
                            "session_status": ["Completed"],
                            "session_is_control_participant": ["No"],
                            "session_duration": ["0.0"],
                            "session_site": ["SickKids"],
                        }
                    )
                    merged_questionnaire_data += [session_df]

        audio_folders = Path(audio_dir)
        audio_sub_folders = sorted([str(f) for f in audio_folders.iterdir() if f.is_dir()])
        merged_csv = []
        for subject in audio_sub_folders:
            audio_session_id = [f.name for f in os.scandir(subject) if f.is_dir()]
            audio_session_dict = {
                "record_id": (subject.split("/")[-1]).split()[0],
                "redcap_repeat_instrument": "Session",
                "redcap_repeat_instance": 1,
                "session_id": audio_session_id[0],
                "session_status": "Completed",
                "session_is_control_participant": "No",
                "session_duration": 0.0,
                "session_site": "SickKids",
            }
            merged_csv.append(audio_session_dict)
            # Process audio data
            audio_files = []
            for audio_file in Path(subject).rglob("*.wav"):
                audio_files.append(str(audio_file))

            merged_csv += parse_audio(audio_files)
        audio_df = pd.DataFrame(merged_csv)
        # Convert audio data to DataFrames
        audio_dfs = [audio_df]

        # Combine all data
        all_dfs = merged_questionnaire_data + audio_dfs

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True, sort=False)
            
            # Handle participant group replacement
            if participant_group:
                combined_df["redcap_repeat_instrument"] = combined_df["redcap_repeat_instrument"].replace(
                    participant_group, "Participant"
                )
            
            return combined_df
        else:
            return pd.DataFrame()
    
    def _validate_redcap_columns(self) -> None:
        """
        Validate RedCap DataFrame column names.
        
        RedCap allows two distinct export formats: raw data or with labels.
        This function verifies the headers are exported correctly as coded entries.
        
        Raises:
            ValueError: If columns don't match expected format
        """
        # Load column mapping
        b2ai_resources = files("b2aiprep").joinpath("prepare").joinpath("resources")
        column_mapping: dict = json.loads(b2ai_resources.joinpath("column_mapping.json").read_text())

        # Check overlap with coded and label headers
        overlap_with_label = set(self.df.columns.tolist()).intersection(set(column_mapping.values()))
        overlap_with_coded = set(self.df.columns.tolist()).intersection(set(column_mapping.keys()))

        if len(overlap_with_coded) == self.df.shape[1]:
            return

        if len(overlap_with_coded) == 0:
            raise ValueError(
                "DataFrame has no coded headers. Please modify the source data to have "
                "coded labels instead."
            )

        # Check if data is exported with labels
        if len(overlap_with_label) > (self.df.shape[1] * 0.5):
            raise ValueError(
                "DataFrame has label headers rather than coded headers. "
                "Please modify the source data to have coded labels instead."
            )

        # Warn about mixed headers
        if len(overlap_with_label) > 0:
            _LOGGER.warning(
                f"DataFrame has a mix of label and coded headers: {len(overlap_with_coded)} coded, "
                f"{len(overlap_with_label)} label, and {self.df.shape[1]} total columns. "
                f"Downstream processing expects only coded labels."
            )
        elif len(overlap_with_coded) < self.df.shape[1]:
            _LOGGER.warning(
                f"DataFrame has only {len(overlap_with_coded)} / {self.df.shape[1]} coded headers."
            )
    
    def _insert_missing_columns(self) -> None:
        """
        Add any missing columns to the dataframe.
        
        This ensures all expected columns are present for downstream processing.
        """
        from importlib.resources import files
        
        all_columns_path = files("b2aiprep").joinpath("prepare", "resources", "all_columns.json")
        all_columns = json.load(all_columns_path.open())
        columns_to_add = [col for col in all_columns if col not in self.df.columns]
        
        if columns_to_add:
            nan_cols = pd.DataFrame(np.nan, index=self.df.index, columns=columns_to_add)
            self.df = pd.concat([self.df, nan_cols], axis=1)
        
        _LOGGER.info(f"Added {len(columns_to_add)} missing columns to DataFrame")
    
    def get_df_of_repeat_instrument(self, instrument: Instrument) -> DataFrame:
        """
        Filter rows and columns of the DataFrame to correspond to a specific repeat instrument.
        
        Args:
            instrument: The repeat instrument to filter for
            
        Returns:
            The filtered DataFrame
        """
        columns = instrument.get_columns()
        idx = self.df["redcap_repeat_instrument"] == instrument.text
        columns_present = [c for c in columns if c in self.df.columns]
        dff = self.df.loc[idx, columns_present].copy()
        
        if len(columns_present) < len(columns):
            _LOGGER.warning(
                f"Inserting None for missing columns of {instrument.name} instrument: "
                f"{set(columns) - set(columns_present)}"
            )
            impute_value = np.nan
            for c in set(columns) - set(columns_present):
                dff[c] = impute_value
                
        return dff
    
    def get_recordings_for_acoustic_task(self, acoustic_task: str) -> DataFrame:
        """
        Get recordings for a specific acoustic task.
        
        Args:
            acoustic_task: The acoustic task to filter for
            
        Returns:
            The filtered DataFrame
            
        Raises:
            ValueError: If acoustic_task is not recognized
        """
        if acoustic_task not in AUDIO_TASKS:
            raise ValueError(f"Unrecognized {acoustic_task}. Options: {AUDIO_TASKS}")

        acoustic_tasks_df = self.get_df_of_repeat_instrument(RepeatInstrument.ACOUSTIC_TASK.value)
        recordings_df = self.get_df_of_repeat_instrument(RepeatInstrument.RECORDING.value)

        idx = acoustic_tasks_df["acoustic_task_name"] == acoustic_task

        dff = recordings_df.merge(
            acoustic_tasks_df.loc[idx, ["acoustic_task_id"]],
            left_on="recording_acoustic_task_id",
            right_on="acoustic_task_id",
            how="inner",
        )
        return dff
    
    def to_csv(self, file_path: t.Union[str, Path], **kwargs) -> None:
        """
        Export the dataset to CSV format.
        
        Args:
            file_path: Path where the CSV file will be saved
            **kwargs: Additional arguments passed to pandas.DataFrame.to_csv()
        """
        self.df.to_csv(file_path, index=False, **kwargs)

    def validate(self) -> dict:
        """
        Validate the dataset and return validation results.
        
        Returns:
            Dictionary containing validation results and any issues found
        """
        validation_results = {
            'valid': True,
            'issues': [],
            'warnings': [],
            'stats': {
                'total_rows': len(self.df),
                'total_columns': len(self.df.columns),
                'missing_values': self.df.isnull().sum().sum()
            }
        }
        
        # Check for required columns
        required_columns = ['record_id', 'redcap_repeat_instrument']
        missing_required = [col for col in required_columns if col not in self.df.columns]
        
        if missing_required:
            validation_results['valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing_required}")
        
        # Check for duplicate record IDs within instruments
        if 'record_id' in self.df.columns and 'redcap_repeat_instrument' in self.df.columns:
            duplicates = self.df.groupby(['record_id', 'redcap_repeat_instrument']).size()
            duplicates = duplicates[duplicates > 1]
            
            if len(duplicates) > 0:
                validation_results['warnings'].append(
                    f"Found {len(duplicates)} duplicate record_id/instrument combinations"
                )
        
        return validation_results
    
    def get_summary(self) -> dict:
        """
        Get a summary of the dataset.
        
        Returns:
            Dictionary containing dataset summary statistics
        """
        summary = {
            'source_type': self.source_type,
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'metadata': self.metadata
        }
        
        if 'record_id' in self.df.columns:
            summary['unique_participants'] = self.df['record_id'].nunique()
        
        if 'redcap_repeat_instrument' in self.df.columns:
            summary['instruments'] = self.df['redcap_repeat_instrument'].value_counts().to_dict()
        
        return summary
    
    def __repr__(self) -> str:
        """String representation of the dataset."""
        return f"RedCapDataset(source_type='{self.source_type}', rows={len(self.df)}, columns={len(self.df.columns)})"
