"""
Utilities for extracting subsets of data from a BIDS-like formatted dataset.

The BIDS-like format assumed is in the following structure. Let's assume we have
a participant (p1) and they have multiple sessios (s1, s2, s3). Then the BIDS-like
structure is:

sub-p1/
    ses-s1/
        beh/
            sub-p1_ses-s1_session-questionnaire-a.json
            sub-p1_ses-s1_session-questionnaire-b.json
        sub-p1_ses-s1_sessionschema.json
    sub-p1_subject-questionnaire-a.json
    sub-p1_subject-questionnaire-b.json
    ...
"""

import logging
import os
import re
import typing as t
from collections import OrderedDict
from pathlib import Path
from importlib.resources import files
import json

import numpy as np
import pandas as pd
import torch
from fhir.resources.questionnaireresponse import QuestionnaireResponse
from senselab.audio.data_structures.audio import Audio
from soundfile import LibsndfileError
from tqdm import tqdm



class BIDSDataset:
    def __init__(self, data_path: t.Union[Path, str, os.PathLike]):
        self.data_path = Path(data_path).resolve()

    def find_questionnaires(self, questionnaire_name: str) -> t.List[Path]:
        """
        Find all the questionnaires with a given suffix.

        Parameters
        ----------
        questionnaire_name : str
            The name of the questionnaire.

        Returns
        -------
        List[Path]
            A list of questionnaires which have the given questionnaire suffix.
        """
        questionnaires = []
        
        # Handle special cases for the new data structure
        if questionnaire_name == "recordingschema":
            # Find all audio metadata JSON files which contain recordingschema data
            for audio_json in self.data_path.rglob("*/audio/*.json"):
                questionnaires.append(audio_json)
        else:
            # Try the original approach first for backward compatibility
            for questionnaire in self.data_path.rglob(f"sub-*_{questionnaire_name}.json"):
                questionnaires.append(questionnaire)
        
        return questionnaires

    def find_subject_questionnaires(self, subject_id: str) -> t.List[Path]:
        """
        Find all the questionnaires for a given subject.

        Parameters
        ----------
        subject_id : str
            The subject identifier.

        Returns
        -------
        List[Path]
            A list of questionnaires for the specific subject.
        """
        subject_path = self.data_path / f"sub-{subject_id}"
        questionnaires = []
        for questionnaire in subject_path.glob("sub-*.json"):
            questionnaires.append(questionnaire)
        return questionnaires

    def find_session_questionnaires(self, subject_id: str, session_id: str) -> t.List[Path]:
        """
        Find all the questionnaires for a given subject and session.

        Parameters
        ----------
        subject_id : str
            The subject identifier.
        session_id : str
            The session identifier.

        Returns
        -------
        List[Path]
            A list of questionnaires for the specific subject and session.
        """
        session_path = self.data_path / f"sub-{subject_id}" / f"ses-{session_id}"
        questionnaires = []
        for questionnaire in session_path.glob("sub-*.json"):
            questionnaires.append(questionnaire)
        return questionnaires

    def find_subjects(self) -> t.List[Path]:
        """
        Find all the subjects in the dataset.

        Returns
        -------
        List[Path]
            A list of subject paths.
        """
        subjects = []
        for subject in self.data_path.glob("sub-*"):
            subjects.append(subject)
        return subjects

    def find_sessions(self, subject_id: str) -> t.List[Path]:
        """
        Find all the sessions for a given subject.

        Parameters
        ----------
        subject_id : str
            The subject identifier.

        Returns
        -------
        List[Path]
            A list of session paths.
        """
        subject_path = self.data_path / f"sub-{subject_id}"
        sessions = []
        for session in subject_path.glob("ses-*"):
            sessions.append(session)
        return sessions

    def find_tasks(self, subject_id: str, session_id: str) -> t.Dict[str, Path]:
        """
        Find all the tasks for a given subject and session.

        Parameters
        ----------
        subject_id : str
            The subject identifier.
        session_id : str
            The session identifier.

        Returns
        -------
        Dict[str, Path]
            A dictionary of tasks for the subject and session.
        """
        session_path = self.data_path / f"sub-{subject_id}" / f"ses-{session_id}"
        tasks = {}
        prefix_length = len(f"sub-{subject_id}_ses-{session_id}_task-")
        for task in session_path.glob(f"sub-{subject_id}_ses-{session_id}_task-*"):
            task_id = task.stem[prefix_length:-4]
            tasks[task_id] = task.stem
        return tasks

    def list_questionnaire_types(self, subject_only: bool = False) -> t.List[str]:
        """
        List all the questionnaire types in the dataset.

        Returns
        -------
        List[str]
            A list of questionnaire types.
        """
        questionnaire_types = set()
        for subject_path in self.data_path.glob("sub-*"):
            # subject-wide resources
            for questionnaire in subject_path.glob("sub-*.json"):
                questionnaire_types.add(questionnaire.stem.split("_")[-1])
            if subject_only:
                continue
            # session-wide resources
            for session_path in subject_path.glob("ses-*"):
                beh_path = session_path.joinpath("beh")
                if beh_path.exists():
                    for questionnaire in beh_path.glob("sub-*.json"):
                        questionnaire_types.add(questionnaire.stem.split("_")[-1])
        return sorted(list(questionnaire_types))

    def load_questionnaire(self, questionnaire_path: Path) -> QuestionnaireResponse:
        """
        Load a questionnaire from a given path.

        Parameters
        ----------
        questionnaire_path : Path
            The path to the questionnaire.

        Returns
        -------
        pd.DataFrame
            The questionnaire data.
        """
        return QuestionnaireResponse.parse_raw(questionnaire_path.read_text())

    def load_subject_questionnaires(self, subject_id: str) -> t.List[QuestionnaireResponse]:
        """
        Load all the questionnaires for a given subject.

        Parameters
        ----------
        subject_id : str
            The subject identifier.

        Returns
        -------
        List[QuestionnaireResponse]
            A list of questionnaires for the specific subject. Each element is a FHIR
            QuestionnaireResponse object, which inherits from Pydantic.
        """
        questionnaires = self.find_subject_questionnaires(subject_id)
        return [self.load_questionnaire(path) for path in questionnaires]

    def load_questionnaires(self, questionnaire_name: str) -> t.List[QuestionnaireResponse]:
        """
        Load all the questionnaires with a given name.

        Parameters
        ----------
        questionnaire_name : str
            The name of the questionnaire.

        Returns
        -------
        List[QuestionnaireResponse]
            A list of questionnaires for the specific subject. Each element is a FHIR
            QuestionnaireResponse object, which inherits from Pydantic.
        """
        questionnaires = self.find_questionnaires(questionnaire_name)
        return [self.load_questionnaire(path) for path in questionnaires]

    def questionnaire_to_dataframe(self, questionnaire: QuestionnaireResponse) -> pd.DataFrame:
        """
        Convert a questionnaire to a pandas DataFrame.

        Parameters
        ----------
        questionnaire : pd.DataFrame
            The questionnaire data.

        Returns
        -------
        pd.DataFrame
            The questionnaire data as a DataFrame. The dataframe is in a "long" format
            with a column for "linkId" and multiple value columns (e.g. "valueString")
        """
        questionnaire_dict = questionnaire.dict()
        items = questionnaire_dict["item"]
        has_multiple_answers = False
        for item in items:
            if ("answer" in item) and (len(item["answer"]) > 1):
                has_multiple_answers = True
                break
        if has_multiple_answers:
            raise NotImplementedError("Questionnaire has multiple answers per question.")

        items = []
        for item in questionnaire_dict["item"]:
            # Rename record_id to participant_id
            link_id = item["linkId"]
            if link_id == "record_id":
                link_id = "participant_id"
            if "answer" in item:
                items.append(
                    OrderedDict(
                        linkId=link_id,
                        **item["answer"][0],
                    )
                )
            else:
                items.append(
                    OrderedDict(
                        linkId=link_id,
                        valueString=None,
                    )
                )
        # unroll based on the possible value options
        return pd.DataFrame(items)

    def find_audio(self, subject_id: str, session_id: str) -> t.List[Path]:
        """
        Find all the audio recordings for a given subject and session.

        Parameters
        ----------
        subject_id : str
            The subject identifier.
        session_id : str
            The session identifier.

        Returns
        -------
        List[Path]
            A list of audio recordings.
        """
        session_path = self.data_path / f"sub-{subject_id}" / f"ses-{session_id}" / "audio"
        audio = []
        for audio_file in session_path.glob("*.wav"):
            audio.append(audio_file)
        return audio

    def find_audio_features(self, subject_id: str, session_id: str) -> t.List[Path]:
        """
        Find all the audio features for a given subject and session.

        Parameters
        ----------
        subject_id : str
            The subject identifier.
        session_id : str
            The session identifier.

        Returns
        -------
        List[Path]
            A list of audio features.
        """
        session_path = self.data_path / f"sub-{subject_id}" / f"ses-{session_id}" / "audio"
        features = []
        for feature_file in session_path.glob("*.pt"):
            features.append(feature_file)
        return features

    def find_audio_transcripts(self, subject_id: str, session_id: str) -> t.List[Path]:
        """
        Find all the audio transcripts for a given subject and session.

        Parameters
        ----------
        subject_id : str
            The subject identifier.
        session_id : str
            The session identifier.

        Returns
        -------
        List[Path]
            A list of audio transcripts.
        """
        session_path = (
            self.data_path / f"sub-{subject_id}" / f"ses-{session_id}" / "audio_transcripts"
        )
        transcripts = []
        for transcript_file in session_path.glob("*.json"):
            transcripts.append(transcript_file)
        return transcripts


class VBAIDataset(BIDSDataset):
    """Extension of BIDS format dataset implementing helper functions for data specific
    to the Bridge2AI Voice as a Biomarker of Health project.
    """

    def __init__(self, data_path: t.Union[Path, str, os.PathLike]):
        super().__init__(data_path)

    def _merge_columns_with_underscores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Merges columns which are exported by RedCap in a one-hot encoding manner, i.e.
        they correspond to a single category but are split into multiple yes/no columns.

        Modifies the dataframe in place.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to modify.

        Returns
        -------
        pd.DataFrame
            The modified dataframe.
        """

        # identify all the columns which end with __1, __2, etc.
        # extract the prefix for these columns only
        columns_with_underscores = sorted(
            list(
                set(
                    [
                        col[: col.rindex("__") - 1]
                        for col in df.columns
                        if re.search("__[0-9]+$", col) is not None
                    ]
                )
            )
        )

        # iterate through each prefix and merge together data into this prefix
        for col in columns_with_underscores:
            columns_to_merge = df.filter(like=f"{col}__").columns
            df[col] = df[columns_to_merge].apply(
                lambda x: next((i for i in x if i is not None), None), axis=1
            )
            df.drop(columns=columns_to_merge, inplace=True)
        return df

    def load_and_pivot_questionnaire(self, questionnaire_name: str) -> pd.DataFrame:
        """
        Loads all data for a questionnaire and pivots on the appropriate identifier column.

        Parameters
        ----------
        questionnaire_name : str
            The name of the questionnaire to load.

        Returns
        -------
        pd.DataFrame
            A "wide" format dataframe with questionnaire data. Returns empty DataFrame 
            if columns are not found in participants.tsv.
        """
        # Load participants.tsv file
        participants_file = self.data_path / "participants.tsv"
        if not participants_file.exists():
            logging.warning(f"participants.tsv file not found at {participants_file}")
            return pd.DataFrame()

        participants_df = pd.read_csv(participants_file, sep="\t")
        # Load questionnaire columns from instrument_columns resources
        instrument_columns_path = files("b2aiprep").joinpath("prepare").joinpath("resources").joinpath("instrument_columns")
        questionnaire_file = instrument_columns_path.joinpath(f"{questionnaire_name}.json")
        
        if not questionnaire_file.exists():
            logging.warning(f"Questionnaire JSON file not found: {questionnaire_name}.json")
            return pd.DataFrame()
            
        questionnaire_columns = json.loads(questionnaire_file.read_text())

        # Filter to only include columns that exist in participants.tsv
        available_columns = [col for col in questionnaire_columns if col in participants_df.columns]
        
        if not available_columns:
            logging.warning(f"No columns from '{questionnaire_name}' questionnaire found in participants.tsv")
            return pd.DataFrame()

        # Select the available columns
        questionnaire_df = participants_df[available_columns].copy()
        
        return questionnaire_df

    def load_participants(self) -> pd.DataFrame:
        """
        Loads the participants.tsv file and returns a dataframe with participant data.

        Returns
        -------
        pd.DataFrame
            A dataframe with participant data.
        """
        participants_file = self.data_path / "participants.tsv"
        if not participants_file.exists():
            logging.warning("participants.tsv file not found")
            return pd.DataFrame()

        try:
            df = pd.read_csv(participants_file, sep='\t')
            # Rename record_id to participant_id for consistency
            if 'record_id' in df.columns:
                df.rename(columns={'record_id': 'participant_id'}, inplace=True)
            return df
        except Exception as e:
            logging.error(f"Error loading participants data: {e}")
            return pd.DataFrame()

    def _load_session_schema_from_participants_tsv(self) -> pd.DataFrame:
        """
        Load session schema data from the participants.tsv file.
        
        Returns
        -------
        pd.DataFrame
            Session schema data with participant_id and session information.
        """
        participants_file = self.data_path / "participants.tsv"
        if not participants_file.exists():
            logging.warning("participants.tsv file not found")
            return pd.DataFrame()
        
        try:
            # Read the participants file
            df = pd.read_csv(participants_file, sep='\t')
            
            # Select session-related columns
            session_columns = [
                'participant_id', 'session_id', 'session_status', 
                'session_is_control_participant', 'session_duration'
            ]
            
            # Only include columns that exist in the file
            available_columns = [col for col in session_columns if col in df.columns]
            
            if not available_columns:
                logging.warning("No session-related columns found in participants.tsv")
                return pd.DataFrame()
            
            # Filter to rows that have session data (session_id is not null)
            session_df = df[available_columns].copy()
            if 'session_id' in session_df.columns:
                session_df = session_df.dropna(subset=['session_id'])
            
            return session_df
            
        except Exception as e:
            logging.error(f"Error loading session data from participants.tsv: {e}")
            return pd.DataFrame()

    def _load_recording_and_acoustic_task_df(self) -> pd.DataFrame:
        """Loads recording schema dataframe with the acoustic task name.

        Returns
        -------
        pd.DataFrame
            The recordings dataframe with the additional "acoustic_task_name" column.
        """
        recording_df = self.load_and_pivot_questionnaire("recordingschema")
        task_df = self.load_and_pivot_questionnaire("acoustictaskschema")

        recording_df = recording_df.merge(
            task_df[["acoustic_task_id", "acoustic_task_name"]],
            how="inner",
            left_on="recording_acoustic_task_id",
            right_on="acoustic_task_id",
        )
        return recording_df

    def load_recording(self, recording_id: str) -> Audio:
        """Checks for and loads in the given recording_id.

        Parameters
        ----------
        recording_id : str
            The recording identifier.

        Returns
        -------
        Audio
            The loaded audio.
        """
        # verify the recording_id is in the recording_df
        recording_df = self._load_recording_and_acoustic_task_df()
        idx = recording_df["recording_id"] == recording_id
        if not idx.any():
            raise ValueError(
                f"Recording ID '{recording_id}' not found in \
                             recordings dataframe."
            )

        row = recording_df.loc[idx].iloc[0]

        # Use participant_id or fall back to record_id for backward compatibility
        subject_id = row.get("participant_id", row.get("record_id"))
        session_id = row["recording_session_id"]
        task = row["acoustic_task_name"].replace(" ", "-")
        name = row["recording_name"].replace(" ", "-")

        audio_file = self.data_path.joinpath(
            f"sub-{subject_id}",
            f"ses-{session_id}",
            "audio",
            f"sub-{subject_id}_ses-{session_id}_{task}_rec-{name}.wav",
        )
        return Audio(filepath=str(audio_file))

    def load_recordings(self) -> t.List[Audio]:
        """Loads all audio recordings in the dataset.

        Returns
        -------
        List[Audio]
            The loaded audio recordings.
        """
        recording_df = self._load_recording_and_acoustic_task_df()
        audio_data = []
        missed_files = []
        for _, row in tqdm(
            recording_df.iterrows(), total=recording_df.shape[0], desc="Loading audio"
        ):
            # Use participant_id or fall back to record_id for backward compatibility
            subject_id = row.get("participant_id", row.get("record_id"))
            session_id = row["recording_session_id"]
            task = row["acoustic_task_name"].replace(" ", "-")
            name = row["recording_name"].replace(" ", "-")
            audio_file = self.data_path.joinpath(
                f"sub-{subject_id}",
                f"ses-{session_id}",
                "audio",
                f"sub-{subject_id}_ses-{session_id}_{task}_rec-{name}.wav",
            )
            try:
                audio_data.append(Audio(filepath=str(audio_file)))
            except (LibsndfileError, FileNotFoundError):
                # assuming lbsnd file error is a file not found, usually it is
                missed_files.append(audio_file)
                continue

        if len(missed_files) > 0:
            logging.warning(
                f"Could not find {len(missed_files)} / {recording_df.shape[0]} audio files."
            )

        return audio_data

    def load_spectrograms(self) -> t.List[np.array]:
        """Loads all audio recordings in the dataset."""
        recording_df = self._load_recording_and_acoustic_task_df()
        audio_data = []
        missed_files = []
        for _, row in tqdm(
            recording_df.iterrows(), total=recording_df.shape[0], desc="Loading audio"
        ):
            # Use participant_id or fall back to record_id for backward compatibility
            subject_id = row.get("participant_id", row.get("record_id"))
            session_id = row["recording_session_id"]
            task = row["acoustic_task_name"].replace(" ", "-")
            name = row["recording_name"].replace(" ", "-")
            audio_file = self.data_path.joinpath(
                f"sub-{subject_id}",
                f"ses-{session_id}",
                "audio",
                f"sub-{subject_id}_ses-{session_id}_{task}_rec-{name}.pt",
            )
            try:
                features = torch.load(str(audio_file), weights_only=False)
                audio_data.append(features["specgram"])
            except FileNotFoundError:
                # assuming lbsnd file error is a file not found, usually it is
                missed_files.append(audio_file)
                continue

        if len(missed_files) > 0:
            logging.warning(
                f"Could not find {len(missed_files)} / {recording_df.shape[0]} feature files."
            )

        return audio_data

    def validate_audio_files_exist(self) -> bool:
        """
        Validates that the audio recordings for all sessions are present.

        Parameters
        ----------
        subject_id : str
            The subject identifier.
        session_id : str
            The session identifier.

        Returns
        -------
        bool
            Whether the audio files are present.
        """
        missing_audio_files = []
        # iterate over all of the audio tasks in beh subfolder
        subjects = self.find_subjects()
        for subject in subjects:
            sessions = self.find_sessions(subject)
            for session in sessions:
                tasks = self.find_tasks(subject, session)
                for task_name, task_filename in tasks.items():
                    # check if the audio file is present
                    if "_rec-" not in task_name:
                        continue
                    if not task_name.endswith("_recordingschema.json"):
                        continue

                    suffix_len = len("_recordingschema")
                    audio_filename = (
                        Path(task_filename.parent).joinpath("..", "audio"),
                        f"{task_filename.stem[:suffix_len]}",
                    )
                    if not audio_filename.exists():
                        missing_audio_files.append(audio_filename)

        logging.debug(f"Missing audio files: {missing_audio_files}")
        return len(missing_audio_files) == 0
