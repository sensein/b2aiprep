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
import shutil
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

from b2aiprep.prepare.constants import RepeatInstrument, Instrument
from b2aiprep.prepare.utils import copy_package_resource, remove_files_by_pattern
from b2aiprep.prepare.fhir_utils import convert_response_to_fhir
from b2aiprep.prepare.prepare import (
    filter_audio_paths, 
    get_value_from_metadata, 
    update_metadata_record_and_session_id,
    reduce_id_length
)
from b2aiprep.prepare.bids import get_paths
from pydantic import BaseModel
from b2aiprep.prepare.redcap import RedCapDataset



class BIDSDataset:
    def __init__(self, data_path: t.Union[Path, str, os.PathLike]):
        self.data_path = Path(data_path).resolve()
    
    @classmethod
    def from_redcap(
        cls,
        redcap_dataset: RedCapDataset,
        outdir: t.Union[str, Path],
        audiodir: t.Optional[t.Union[str, Path]] = None
    ) -> 'BIDSDataset':
        """
        Create a BIDSDataset by converting a RedCapDataset to BIDS format.
        
        Args:
            redcap_dataset: The RedCapDataset to convert
            outdir: Output directory for BIDS structure
            audiodir: Optional directory containing audio files
            
        Returns:
            BIDSDataset instance pointing to the created BIDS directory
        """
        # Use instance methods instead of importing from bids module
        
        outdir = Path(outdir).as_posix()
        BIDSDataset._initialize_data_directory(outdir)

        # for participants.tsv we skip cleaning the phenotype data
        # also note that participants files are in the root outdir folder,
        # not the phenotype subfolder
        BIDSDataset._process_phenotype_tsv_and_json(
            df=redcap_dataset.df,
            input_dir=outdir,
            output_dir=outdir,
            filename="participants.json",
            clean_phenotype_data=False,
        )

        # Subselect the RedCap dataframe and output components to individual files in the phenotype directory
        BIDSDataset._construct_all_tsvs_from_jsons(
            df=redcap_dataset.df,
            input_dir=os.path.join(outdir, "phenotype"),
            output_dir=os.path.join(outdir, "phenotype"),
        )

        # Process repeat instruments
        repeat_instruments: t.List[RepeatInstrument] = list(RepeatInstrument.__members__.values())
        dataframe_dicts: t.Dict[RepeatInstrument, pd.DataFrame] = {}
        
        for repeat_instrument in repeat_instruments:
            instrument = repeat_instrument.value
            questionnaire_df = redcap_dataset.get_df_of_repeat_instrument(instrument)
            logging.info(f"Number of {instrument.name} entries: {len(questionnaire_df)}")
            dataframe_dicts[repeat_instrument] = questionnaire_df

        # Extract main dataframes
        participants_df = dataframe_dicts.pop(RepeatInstrument.PARTICIPANT)
        sessions_df = dataframe_dicts.pop(RepeatInstrument.SESSION)
        acoustic_tasks_df = dataframe_dicts.pop(RepeatInstrument.ACOUSTIC_TASK)
        recordings_df = dataframe_dicts.pop(RepeatInstrument.RECORDING)

        # Convert remaining dataframes to dictionaries indexed by session_id
        for repeat_instrument, questionnaire_df in dataframe_dicts.items():
            session_id_col = repeat_instrument.value.session_id
            dataframe_dicts[repeat_instrument] = cls._df_to_dict(questionnaire_df, session_id_col)

        # Create participant hierarchy
        participants = []
        for participant in participants_df.to_dict("records"):
            participants.append(participant)
            participant["sessions"] = sessions_df[
                sessions_df["record_id"] == participant["record_id"]
            ].to_dict("records")

            for session in participant["sessions"]:
                session_id = session["session_id"]
                session["acoustic_tasks"] = acoustic_tasks_df[
                    acoustic_tasks_df["acoustic_task_session_id"] == session_id
                ].to_dict("records")
                
                for task in session["acoustic_tasks"]:
                    task["recordings"] = recordings_df[
                        recordings_df["recording_acoustic_task_id"] == task["acoustic_task_id"]
                    ].to_dict("records")

                # Add questionnaire data per session
                for key, df_by_session_id in dataframe_dicts.items():
                    if session_id not in df_by_session_id:
                        session[key] = None
                    else:
                        session[key] = df_by_session_id[session_id]

        # Output participant data to FHIR format
        if audiodir is not None and not Path(audiodir).exists():
            logging.warning(f"{audiodir} path does not exist. No audio files will be reorganized.")
            audiodir = None
            
        for participant in tqdm(participants, desc="Writing participant data to file"):
            cls._output_participant_data_to_fhir(participant, Path(outdir), audiodir=audiodir)
        
        # Return a new BIDSDataset instance pointing to the created directory
        return cls(outdir)

    @staticmethod
    def _initialize_data_directory(bids_dir_path: str) -> None:
        """Initializes the data directory using the template.

        Args:
            bids_dir_path (str): The path to the BIDS directory where the data should be initialized.

        Returns:
            None
        """
        if not os.path.exists(bids_dir_path):
            os.makedirs(bids_dir_path)
            logging.info(f"Created directory: {bids_dir_path}")

        template_package = "b2aiprep.prepare.resources.b2ai-data-bids-like-template"
        copy_package_resource(template_package, "CHANGELOG.md", bids_dir_path)
        copy_package_resource(template_package, "README.md", bids_dir_path)
        copy_package_resource(template_package, "dataset_description.json", bids_dir_path)
        copy_package_resource(template_package, "participants.json", bids_dir_path)
        copy_package_resource(template_package, "participants.tsv", bids_dir_path)
        copy_package_resource(template_package, "phenotype", bids_dir_path)
        phenotype_path = Path(bids_dir_path).joinpath("phenotype")
        remove_files_by_pattern(phenotype_path, "<measurement_tool_name>*")

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

    @staticmethod
    def _df_to_dict(df: pd.DataFrame, index_col: str) -> t.Dict[str, t.Any]:
        """Convert a DataFrame to a dictionary of dictionaries, with the given column as the index.

        Retains the index column within the dictionary.

        Args:
            df: DataFrame to convert.
            index_col: Column to use as the index.

        Returns:
            Dictionary of dictionaries.

        Raises:
            ValueError: If index column not found in DataFrame or non-unique values found.
        """
        if index_col not in df.columns:
            raise ValueError(f"Index column {index_col} not found in DataFrame.")
        if df[index_col].isnull().any():
            logging.warning(
                f"Found {df[index_col].isnull().sum()} null value(s) for {index_col}. Removing."
            )
            df = df.dropna(subset=[index_col])

        non_unique = df[df[index_col].duplicated(keep=False)]
        if df[index_col].nunique() < df.shape[0]:
            raise ValueError(f"Non-unique {index_col} values found. {non_unique}")
        

        # *copy* the given column into the index, preserving the original column
        # so that it is output in the later call to to_dict()
        df.index = df[index_col]

        return df.to_dict("index")

    @staticmethod
    def _dataframe_to_tsv(df: pd.DataFrame, tsv_path: str) -> None:
        """Construct a TSV file from a DataFrame.

        Args:
            df: DataFrame containing the data.
            tsv_path: Path to the output TSV file.
        """
        # Save the combined DataFrame to a TSV file
        df.to_csv(tsv_path, sep="\t", index=False)
        logging.info(f"TSV file created and saved to: {tsv_path}")

    @staticmethod
    def _subselect_dataframe_using_json(
        df: pd.DataFrame, json_data: dict
    ) -> pd.DataFrame:
        """Extracts a subset of columns from a DataFrame using the given JSON file,
        removes duplicates, and returns a new DataFrame.

        Combines entries so that there is one row per record_id.

        Args:
            df: DataFrame containing the data.
            json_data: Dictionary containing the column labels.

        Raises:
            ValueError: If no valid columns found in DataFrame that match JSON file.
        """

        # The phenotype JSON files are nested below a key that corresponds to the schema name
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
        return combined_df

    @staticmethod
    def _process_phenotype_tsv_and_json(
        df: pd.DataFrame, input_dir: str, output_dir: str, filename: str,
        clean_phenotype_data: bool = True
    ) -> None:
        """Process phenotype data."""
        with open(os.path.join(input_dir, filename), "r") as f:
            json_data = json.load(f)
        df_subselected = BIDSDataset._subselect_dataframe_using_json(df=df, json_data=json_data)
        if clean_phenotype_data:
            df_subselected, json_data = BIDSDataset._clean_phenotype_data(df_subselected, json_data)
        BIDSDataset._dataframe_to_tsv(df_subselected, os.path.join(output_dir, filename.replace(".json", ".tsv")))
        with open(os.path.join(output_dir, filename), "w") as f:
            json.dump(json_data, f, indent=2)

    @staticmethod
    def _construct_all_tsvs_from_jsons(
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

        for filename in os.listdir(input_dir):
            if filename.endswith(".json") and filename not in excluded_files:
                BIDSDataset._process_phenotype_tsv_and_json(
                    df=df, input_dir=input_dir, output_dir=output_dir, filename=filename,
                    clean_phenotype_data=True,
                )

    @staticmethod
    def _get_instrument_for_name(name: str) -> Instrument:
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

    @staticmethod
    def _write_pydantic_model_to_bids_file(
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

    @staticmethod
    def _output_participant_data_to_fhir(
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

        session_instrument = BIDSDataset._get_instrument_for_name("sessions")
        task_instrument = BIDSDataset._get_instrument_for_name("acoustic_tasks")
        recording_instrument = BIDSDataset._get_instrument_for_name("recordings")

        sessions_df = pd.DataFrame(columns=session_instrument.columns)

        if audiodir is not None:
            # audio files are under a folder with the site name,
            # so we need to recursively glob
            audio_files = list(audiodir.rglob(f"*.wav"))
        else:
            audio_files = []

        # validated questionnaires are asked per session
        sessions_rows = []

        for session in participant["sessions"]:
            sessions_row = {key: session[key] for key in session_instrument.columns}
            sessions_rows.append(sessions_row)
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
                BIDSDataset._write_pydantic_model_to_bids_file(
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
                    BIDSDataset._write_pydantic_model_to_bids_file(
                        audio_output_path,
                        fhir_data,
                        schema_name=recording_instrument.schema_name,
                        subject_id=participant_id,
                        session_id=session_id,
                        task_name=acoustic_task_name,
                        recording_name=recording["recording_name"],
                    )

                    # we also need to organize the audio file
                    audio_files_for_recording = [
                        audio_file for audio_file in audio_files if recording['recording_id'] in audio_file.name
                    ]
                    if len(audio_files_for_recording) == 0:
                        logging.warning(
                            f"No audio file found for recording "
                            f"{recording['recording_id']}."
                        )
                    else:
                        if len(audio_files_for_recording) > 1:
                            logging.warning(
                                f"Multiple audio files found for recording "
                                f"{recording['recording_id']}. "
                                f"Using only {audio_files_for_recording[0]}"
                            )
                        audio_file = audio_files_for_recording[0]

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
                        else:
                            audio_file_destination.write_bytes(audio_file.read_bytes())

        # Save sessions.tsv
        sessions_df = pd.DataFrame(sessions_rows)
        if not os.path.exists(subject_path):
            os.mkdir(subject_path)
        sessions_tsv_path = subject_path / "sessions.tsv"
        sessions_df.to_csv(sessions_tsv_path, sep="\t", index=False)

    @staticmethod
    def load_phenotype_data(data_path: Path, phenotype_name: str) -> t.Tuple[pd.DataFrame, t.Dict[str, t.Any]]:
        """
        Load phenotype data from TSV and JSON files.
        
        Args:
            phenotype_name: Name of the phenotype file (without extension)
            
        Returns:
            Tuple of (DataFrame, phenotype_metadata_dict)
        """
        # Clean up phenotype name
        if phenotype_name.endswith('.tsv'):
            phenotype_name = phenotype_name[:-4]
        elif phenotype_name.endswith('.json'):
            phenotype_name = phenotype_name[:-5]

        # Load TSV and JSON files
        df = pd.read_csv(data_path.joinpath(f"{phenotype_name}.tsv"), sep="\t")
        with open(data_path.joinpath(f"{phenotype_name}.json"), "r") as f:
            phenotype = json.load(f)
            if phenotype_name != "participants":
                data_elements = {}
                for schema in phenotype:
                    data_elements.update(phenotype[schema].get("data_elements", {}))
                phenotype = data_elements

        # Add record_id to phenotype if missing
        if df.shape[1] > 0 and df.columns[0] == 'record_id' and 'record_id' not in list(phenotype.keys()):
            phenotype = BIDSDataset._add_record_id_to_phenotype(phenotype)

        # Validate column count
        if len(phenotype) != df.shape[1]:
            logging.warning(
                f"Phenotype {phenotype_name} has {len(phenotype)} columns, but the data has {df.shape[1]} columns."
            )

        # Handle nested phenotype structure
        if len(phenotype) == 1:
            only_key = next(iter(phenotype))
            if 'data_elements' in phenotype[only_key]:
                phenotype = phenotype[only_key]['data_elements']

        return df, phenotype

    @staticmethod
    def _deidentify_phenotype(df: pd.DataFrame, phenotype: dict, participant_ids_to_remove: t.List[str] = [], participant_ids_to_remap: dict = {}, participant_session_id_to_remap: dict = {}) -> t.Tuple[pd.DataFrame, dict]:
        """
        Apply deidentification operations to phenotype data.
        
        Args:
            df: DataFrame containing the phenotype data
            phenotype: Dictionary containing the phenotype metadata
            
        Returns:
            Tuple of (deidentified_df, deidentified_phenotype_dict)
        """
        # Remove sensitive columns
        df, phenotype = BIDSDataset._remove_sensitive_columns(df, phenotype)

        # Rename record_id to participant_id
        if "record_id" in df.columns:
            df, phenotype = BIDSDataset._rename_record_id_to_participant_id(df, phenotype)

        # Remove participants
        idx = df["participant_id"].isin(participant_ids_to_remove)
        if idx.any():
            logging.info(f"Removing {idx.sum()} participants from phenotype data.")
            df = df.loc[~idx]

        # Remap IDs
        if participant_ids_to_remap and "participant_id" in df.columns:
            ids_before = df["participant_id"]
            df["participant_id"] = df["participant_id"].map(participant_ids_to_remap).fillna(df["participant_id"])
            n_different = (ids_before != df["participant_id"]).sum()
            logging.info(f"Remapped {n_different} / {len(ids_before)} IDs for 'participant_id'")

        # Reduce ID length (remap IDs)
        # if "session_id" in df.columns:
        #     df = BIDSDataset._reduce_id_length(df, "session_id")
        
        if participant_session_id_to_remap and "session_id" in df.columns:
            ids_before = df["session_id"]
            df["session_id"] = df["session_id"].map(participant_ids_to_remap).fillna(df["session_id"])
            n_different = (ids_before != df["session_id"]).sum()
            logging.info(f"Remapped {n_different} / {len(ids_before)} IDs for 'session_id'")

        return df, phenotype

    @staticmethod
    def _clean_phenotype_data(df: pd.DataFrame, phenotype: dict) -> t.Tuple[pd.DataFrame, dict]:
        """
        Apply data cleaning operations to phenotype data.
        
        Args:
            df: DataFrame containing the phenotype data
            phenotype: Dictionary containing the phenotype metadata
            
        Returns:
            Tuple of (cleaned_df, cleaned_phenotype_dict)
        """
        # Fix alcohol column date values
        df = BIDSDataset._fix_alcohol_column(df)
        
        # Remove unwanted columns
        df, phenotype = BIDSDataset._remove_empty_columns(df, phenotype)
        df, phenotype = BIDSDataset._remove_system_columns(df, phenotype)
        
        # Add derived columns
        if ("gender_identity" in df.columns) and ("specify_gender_identity" in df.columns):
            df, phenotype = BIDSDataset._add_sex_at_birth_column(df, phenotype)

        # Rename columns for usability
        df, phenotype = BIDSDataset._rename_columns(df, phenotype)

        return df, phenotype

    @staticmethod
    def _fix_alcohol_column(df: pd.DataFrame) -> pd.DataFrame:
        """Fix known date values in the alcohol_amt column."""
        if "alcohol_amt" in df:
            date_fix_map = {
                "4-Mar": "3 - 4",
                "6-May": "5 - 6",
                "9-Jul": "7 - 9",
            }
            df["alcohol_amt"] = df["alcohol_amt"].apply(
                lambda x: date_fix_map[x] if x in date_fix_map else x
            )
        return df

    @staticmethod
    def _remove_empty_columns(df: pd.DataFrame, phenotype: dict) -> t.Tuple[pd.DataFrame, dict]:
        """Remove columns that are empty."""
        columns_to_drop = [
            "consent_status",
            "enrollment_institution",
            "subjectparticipant_eligible_studies_complete",
            "ef_primary_language_other",
            "ef_fluent_language_other",
            "session_site",
        ]
        return BIDSDataset._drop_columns_from_df_and_data_dict(
            df, phenotype, columns_to_drop, "Removing empty columns"
        )

    @staticmethod
    def _remove_system_columns(df: pd.DataFrame, phenotype: dict) -> t.Tuple[pd.DataFrame, dict]:
        """Remove system/technical columns that shouldn't be in the final dataset."""
        columns_to_drop = [
            "acoustic_task_id",
            "acoustic_task_session_id",
            "acoustic_task_name",
            "acoustic_task_cohort",
            "acoustic_task_status",
            "acoustic_task_duration",
            "recording_id",
            "recording_session_id",
            "recording_acoustic_task_id",
            "recording_name",
            "recording_duration",
            "recording_size",
            "recording_profile_name",
            "recording_profile_version",
            "recording_input_gain",
            "recording_microphone",
        ]
        return BIDSDataset._drop_columns_from_df_and_data_dict(
            df, phenotype, columns_to_drop, "Removing system columns"
        )

    @staticmethod
    def _remove_sensitive_columns(df: pd.DataFrame, phenotype: dict) -> t.Tuple[pd.DataFrame, dict]:
        """Remove columns with sensitive data (free-text, geo-location, etc)."""
        columns_to_drop = [
            "state_province",
            "other_edu_level",
            "others_household_specify",
            "diagnosis_alz_dementia_mci_ds_cdr",
            "diagnosis_alz_dementia_mci_ca_rudas_score",
            "diagnosis_alz_dementia_mci_ca_mmse_score",
            "diagnosis_alz_dementia_mci_ca_moca_score",
            "diagnosis_alz_dementia_mci_ca_adas_cog_score",
            "diagnosis_alz_dementia_mci_ca_other",
            "diagnosis_alz_dementia_mci_ca_other_score",
            "diagnosis_parkinsons_ma_uprds",
            "diagnosis_parkinsons_ma_updrs_part_i_score",
            "diagnosis_parkinsons_ma_updrs_part_ii_score",
            "diagnosis_parkinsons_ma_updrs_part_iii_score",
            "diagnosis_parkinsons_ma_updrs_part_iv_score",
            "diagnosis_parkinsons_non_motor_symptoms_yes",
            "traumatic_event",
            "is_regular_smoker",  # all null values
            # pediatric columns
            "city",
            "state_province",
            "zipcode",
            "other_race_specify",
            "other_primary_language",
            "conditions_other",
            "chronic_medical_conditions_specify",
            "genetic_disorders_specify",
            "hospitalize_medical_conditions_specify",
            "allergies_specify",
            "difficulty_swallowing_specify",
            "ear_infection_specify",
            "ear_tube_specify",
            "ent_evaluation_specify",
            "feeding_tube_specify",
            "gerd_specify",
            "hearing_aid_specify",
            "hoarse_voice_specify",
            "neurological_surgery_specify",
            "noisy_breathing_specify",
            "oxygen_specify",
            "pediatric_medication_specify",
            "speech_disorder_and_speech_delay_specify",
            "speech_therapy_specify",
            "surgery_specify",
            "tonsillitis_specify",
            "vocal_strain_specify",
            # below could be considered for inclusion in the future
            "tonsillectomy_date",
            "adenoidectomy_date",
            "branchial_cleft_cyst_date",
            "ear_tube_date",
            "dermoid_cyst_date",
            "enlarged_lymph_node_date",
            "lingual_tonsillectomy_date",
            "neurological_surgery_date",
            "thyroglossal_duct_cyst_date",
            "thyroid_nodule_or_cancer_date",
        ]
        return BIDSDataset._drop_columns_from_df_and_data_dict(
            df, phenotype, columns_to_drop, "Removing low utility / PHI containing columns"
        )
    
    @staticmethod
    def _rename_columns(df: pd.DataFrame, phenotype: dict) -> t.Tuple[pd.DataFrame, dict]:
        """Rename columns for improved usability."""
        column_rename_map = {
            "diagnois_gi_gsd": "diagnois_gi_gold_standard_diagnosis",
        }
        df = df.rename(columns=column_rename_map)
        phenotype = {column_rename_map.get(k, k): v for k, v in phenotype.items()}
        return df, phenotype

    @staticmethod
    def _drop_columns_from_df_and_data_dict(
        df: pd.DataFrame, phenotype: dict, columns_to_drop: t.List[str], message: str
    ) -> t.Tuple[pd.DataFrame, dict]:
        """Drop columns from the DataFrame and phenotype dictionary."""
        columns_to_drop_in_df = [col for col in columns_to_drop if col in df.columns]
        
        if columns_to_drop_in_df:
            logging.info(f"{message}: {columns_to_drop_in_df}")
            df = df.drop(columns=columns_to_drop_in_df)
            phenotype = {k: v for k, v in phenotype.items() if k not in columns_to_drop_in_df}
        
        return df, phenotype

    @staticmethod
    def _add_record_id_to_phenotype(phenotype: dict) -> dict:
        """Add record_id to phenotype metadata if missing."""
        if 'record_id' in phenotype:
            return phenotype

        phenotype_updated = {
            'record_id': {
                "description": "Unique identifier for each participant."
            }
        }
        phenotype_updated.update(phenotype)
        return phenotype_updated

    @staticmethod
    def _rename_record_id_to_participant_id(df: pd.DataFrame, phenotype: dict) -> t.Tuple[pd.DataFrame, dict]:
        """Rename record_id column to participant_id."""
        phenotype_updated = {}
        for name, value in phenotype.items():
            if name == 'record_id':
                phenotype_updated['participant_id'] = value
                continue
            phenotype_updated[name] = value

        # Only rename if record_id exists and participant_id doesn't exist
        if 'record_id' in df.columns and 'participant_id' not in df.columns:
            df = df.rename(columns={"record_id": "participant_id"})
        elif 'record_id' in df.columns and 'participant_id' in df.columns:
            # If both exist, drop record_id since participant_id takes precedence
            df = df.drop(columns=['record_id'])
            # Remove record_id from phenotype_updated if it exists
            phenotype_updated = {k: v for k, v in phenotype_updated.items() if k != 'record_id'}
        
        return df, phenotype_updated

    @staticmethod
    def _add_sex_at_birth_column(df: pd.DataFrame, phenotype: dict) -> t.Tuple[pd.DataFrame, dict]:
        """Add sex_at_birth column derived from gender_identity and specify_gender_identity."""
        df["sex_at_birth"] = None
        for sex_at_birth in ["Male", "Female"]:
            idx = (
                df["gender_identity"].str.contains(sex_at_birth)
                & df["specify_gender_identity"].notnull()
            )
            df.loc[idx, "sex_at_birth"] = sex_at_birth

        # Re-order columns to place sex_at_birth after gender_identity
        phenotype_reordered = {}
        columns = []
        for c in df.columns:
            if c == "specify_gender_identity":
                continue
            elif c == "gender_identity":
                first_key = next(iter(phenotype))
                columns.append(c)
                columns.append("sex_at_birth")
                phenotype_reordered[c] = phenotype[first_key]["data_elements"][c]
                phenotype_reordered["sex_at_birth"] = {
                    "description": "The sex at birth for the individual."
                }
            elif c == "sex_at_birth":
                continue
            else:
                first_key = next(iter(phenotype))
                columns.append(c)
                if c in phenotype[first_key]["data_elements"]:
                    phenotype_reordered[c] = phenotype[first_key]["data_elements"][c]

        df = df[columns]
        return df, phenotype_reordered

    @staticmethod
    def _reduce_id_length(df: pd.DataFrame, id_name: str) -> pd.DataFrame:
        """Reduce the length of ID columns to 8 characters."""
        if id_name in df.columns:
            df = df.copy()  # Avoid SettingWithCopyWarning
            df[id_name] = df[id_name].apply(reduce_id_length)
        return df

    @staticmethod
    def load_remap_id_list(publish_config_dir: Path) -> t.Dict[str, str]:
        audio_to_remap_path = publish_config_dir / "id_remapping.json"
        if not audio_to_remap_path.exists():
            raise FileNotFoundError(f"ID remapping file {audio_to_remap_path} does not exist.")

        with open(audio_to_remap_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"ID remapping file {audio_to_remap_path} should contain a dict of participant_id:new_id.")

        return data
    
    @staticmethod
    def load_remap_session_id_list(publish_config_dir: Path) -> t.Dict[str, str]:
        session_id_to_remap_path = publish_config_dir / "session_id_remapping.json"
        if not session_id_to_remap_path.exists():
            raise FileNotFoundError(f"ID remapping file {session_id_to_remap_path} does not exist.")

        with open(session_id_to_remap_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"ID remapping file {session_id_to_remap_path} should contain a dict of session_id:new_id.")

        return data
    
    @staticmethod
    def load_participant_ids_to_remove(publish_config_dir: Path) -> t.List[str]:
        """Load list of participant IDs to remove from JSON file."""
        participant_to_remove_path = publish_config_dir / "participant_ids_to_remove.json"
        if not participant_to_remove_path.exists():
            # If file doesn't exist, return empty list
            return []

        with open(participant_to_remove_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Participant IDs to remove file {participant_to_remove_path} should contain a list of participant IDs.")
        
        return data
    
    @staticmethod
    def load_audio_filestems_to_remove(publish_config_dir: Path) -> t.List[str]:
        """Load list of audio file stems to remove from JSON file."""
        audio_to_remove_path = publish_config_dir / "audio_filestems_to_remove.json"
        if not audio_to_remove_path.exists():
            # If file doesn't exist, return empty list
            return []

        with open(audio_to_remove_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Audio filestems to remove file {audio_to_remove_path} should contain a list of audio file stems.")
        
        return data

    def deidentify(self, outdir: t.Union[str, Path], publish_config_dir: Path, skip_audio: bool = False) -> 'BIDSDataset':
        """
        Create a deidentified version of the BIDS dataset.
        
        This method performs the following deidentification steps:
        1. Load phenotype data
        2. Apply deidentification (remove participants, remap IDs, rename columns)
        3. Apply data cleaning (fix values, remove unwanted columns)
        4. Process audio files (if not skipped)
        5. Copy template files
        
        Args:
            outdir: Output directory for the deidentified dataset
            skip_audio: If True, skip copying/processing audio files
            
        Returns:
            New BIDSDataset instance pointing to the deidentified directory
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=False)  # Don't overwrite existing directories
        
        participant_ids_to_remap = BIDSDataset.load_remap_id_list(publish_config_dir)
        participant_ids_to_remove = BIDSDataset.load_participant_ids_to_remove(publish_config_dir)
        participant_session_id_to_remap = BIDSDataset.load_remap_session_id_list(publish_config_dir)

        # Check if source directory has required files
        participant_filepath = self.data_path.joinpath("participants.tsv")
        if not participant_filepath.exists():
            raise FileNotFoundError(f"Participant file {participant_filepath} does not exist.")
        
        logging.info("Processing participants.tsv for deidentification.")
        df, phenotype = BIDSDataset.load_phenotype_data(self.data_path, "participants")
        df, phenotype = BIDSDataset._deidentify_phenotype(df, phenotype, participant_ids_to_remove, participant_ids_to_remap, participant_session_id_to_remap)
        
        # Write out deidentified phenotype data and data dictionary
        df.to_csv(outdir.joinpath("participants.tsv"), sep="\t", index=False)
        with open(outdir.joinpath("participants.json"), "w") as f:
            json.dump(phenotype, f, indent=2)
        logging.info("Finished processing participants data.")
        
        # Process phenotype directory if it exists
        phenotype_base_path = self.data_path.joinpath("phenotype")
        if phenotype_base_path.exists():
            logging.info("Processing phenotype data for deidentification.")
            phenotype_output_path = outdir.joinpath("phenotype")
            phenotype_output_path.mkdir(parents=True, exist_ok=True)
            
            for phenotype_filepath in phenotype_base_path.glob("*.tsv"):
                logging.info(f"Processing {phenotype_filepath.stem}.")
                df_pheno, phenotype_dict = BIDSDataset.load_phenotype_data(phenotype_base_path, phenotype_filepath.stem)
                df_pheno, phenotype_dict = BIDSDataset._deidentify_phenotype(df_pheno, phenotype_dict, participant_ids_to_remove, participant_ids_to_remap, participant_session_id_to_remap)
                
                # Write out phenotype data and data dictionary
                df_pheno.to_csv(
                    phenotype_output_path.joinpath(f"{phenotype_filepath.stem}.tsv"), 
                    sep="\t", index=False
                )
                with open(phenotype_output_path.joinpath(f"{phenotype_filepath.stem}.json"), "w") as f:
                    json.dump(phenotype_dict, f, indent=2)
            logging.info("Finished processing phenotype data.")
        
        if not skip_audio:
            # Process audio files
            logging.info("Processing audio files for deidentification.")
            audio_filestems_to_remove = BIDSDataset.load_audio_filestems_to_remove(publish_config_dir)
            BIDSDataset._deidentify_audio_files(self.data_path, outdir, participant_ids_to_remove, audio_filestems_to_remove, participant_ids_to_remap)
            logging.info("Finished processing audio files.")
        
        # Copy over the standard BIDS template files if they exist
        for template_file in ["README.md", "CHANGES.md", "dataset_description.json"]:
            template_path = self.data_path.joinpath(template_file)
            if template_path.exists():
                shutil.copy(template_path, outdir)
        
        logging.info("Deidentification completed.")
        return BIDSDataset(outdir)

    @staticmethod
    def _deidentify_audio_files(
        data_path: Path,
        outdir: Path,
        exclude_participant_ids: t.List[str] = [],
        exclude_audio_filestems: t.List[str] = [],
        participant_ids_to_remap: t.Dict[str, str] = {}
    ):
        """
        Copy and deidentify audio files to the output directory.
        
        This method:
        1. Gets all audio paths from the dataset
        2. Removes participants from hard-coded exclusion list
        3. Filters out sensitive audio files
        4. Updates metadata with deidentified IDs
        5. Copies audio files and metadata to new location
        
        Args:
            outdir: Output directory for deidentified audio files
        """
        _LOGGER = logging.getLogger(__name__)
        
        # Get all audio paths
        audio_paths = get_paths(data_path, file_extension=".wav")
        audio_paths = [x["path"] for x in audio_paths]
        
        if len(audio_paths) == 0:
            _LOGGER.warning(f"No audio files (.wav) found in {data_path}.")
            return
        
        # Sort audio paths for consistent processing
        audio_paths = sorted(
            audio_paths,
            # sort first by subject, then by task
            key=lambda x: (x.stem.split("_")[0], x.stem.split("_")[2]),
        )
        
        # Remove known individuals (hard-coded participant removal)
        n = len(audio_paths)
        for participant_id in exclude_participant_ids:
            audio_paths = [x for x in audio_paths if f"sub-{participant_id}" not in str(x)]
        
        if len(audio_paths) < n:
            _LOGGER.info(
                f"Removed {n - len(audio_paths)} records due to participant_id removal."
            )
        
        n = len(audio_paths)
        if len(exclude_audio_filestems) > 0:
            audio_paths = [
                a for a in audio_paths
                if a.stem not in exclude_audio_filestems
            ]
        if len(audio_paths) < n:
            _LOGGER.info(
                f"Removed {n - len(audio_paths)} records due to audio_filestem removal."
            )
        
        # Remove audio check and sensitive audio files
        audio_paths = filter_audio_paths(audio_paths)

        # TODO: Add audio processing for further deidentification here
        # This could include:
        # - Voice conversion/anonymization
        # - Pitch shifting
        # - Other audio deidentification techniques
        
        _LOGGER.info(f"Copying {len(audio_paths)} recordings.")
        for audio_path in tqdm(
            audio_paths, desc="Copying audio and metadata files", total=len(audio_paths)
        ):
            json_path = audio_path.parent.joinpath(f'{audio_path.stem}_recording-metadata.json')
            
            if not json_path.exists():
                _LOGGER.warning(f"Metadata file {json_path} not found. Skipping {audio_path}.")
                continue
            
            metadata = json.loads(json_path.read_text())
            
            # Update metadata with deidentified IDs
            update_metadata_record_and_session_id(metadata, participant_ids_to_remap)
            participant_id = get_value_from_metadata(metadata, linkid="participant_id", endswith=False)
            session_id = get_value_from_metadata(metadata, linkid="session_id", endswith=True)
            
            # Create output path with deidentified structure
            audio_path_stem_ending = "_".join(audio_path.stem.split("_")[2:])
            output_path = outdir.joinpath(
                f"sub-{participant_id}/ses-{session_id}/audio/{audio_path_stem_ending}.wav"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy over the associated .json file and audio data
            with open(output_path.with_suffix(".json"), "w") as fp:
                json.dump(metadata, fp, indent=2)
            shutil.copy(audio_path, output_path)


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
