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

from copy import copy, deepcopy
import logging
import os
import re
import shutil
import typing as t
from collections import OrderedDict, defaultdict
from pathlib import Path
from importlib.resources import files
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from pyparsing import col
import torch
from fhir.resources.questionnaireresponse import QuestionnaireResponse
from senselab.audio.data_structures.audio import Audio
from soundfile import LibsndfileError
from tqdm import tqdm

from b2aiprep.prepare.constants import RepeatInstrument, Instrument
from b2aiprep.prepare.update import build_activity_payload
from b2aiprep.prepare.utils import copy_package_resource, get_commit_sha
from b2aiprep.prepare.fhir_utils import convert_response_to_fhir
from b2aiprep.prepare.prepare import (
    get_value_from_metadata, 
    update_metadata_record_and_session_id,
    reduce_id_length
)
from b2aiprep.prepare.bids import get_paths
from pydantic import BaseModel
from b2aiprep.prepare.redcap import RedCapDataset

_LOGGER = logging.getLogger(__name__)

def _copy_audio_files_parallel(copy_tasks: t.List[t.Tuple[Path, Path]], max_workers: int = 16):
    """Copy audio files in parallel using ThreadPoolExecutor.
    
    Args:
        copy_tasks: List of (source_path, dest_path) tuples
        max_workers: Number of parallel worker threads
    """
    def copy_one_file(src: Path, dst: Path) -> t.Optional[str]:
        """Copy a single file, return error message if failed."""
        try:
            shutil.copyfile(src, dst)
            return None
        except Exception as e:
            return f"Failed to copy {src} -> {dst}: {e}"
    
    if not copy_tasks:
        return
    
    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(copy_one_file, src, dst): (src, dst) for src, dst in copy_tasks}
        
        for future in as_completed(futures):
            error = future.result()
            if error:
                errors.append(error)
                logging.error(error)
    
    if errors:        
        failed_files = [err.split("->")[0].replace("Failed to copy", "").strip() for err in errors]  
        total_files = len(copy_tasks)  
        unique_error_types = set()  
        for err in errors:  
            # Try to extract the exception type from the error message  
            if ":" in err:  
                unique_error_types.add(err.split(":")[-1].strip())  
        summary = (  
            f"Encountered {len(errors)} errors out of {total_files} files during parallel audio copying.\n"  
            f"Failed files (up to 5 shown): {failed_files[:5]}\n"  
            f"Unique error types (up to 3 shown): {list(unique_error_types)[:3]}"  
        )  
        logging.warning(summary)  


class BIDSDataset:
    def __init__(self, data_path: t.Union[Path, str, os.PathLike]):
        self.data_path = Path(data_path).resolve()
    
    @classmethod
    def from_redcap(
        cls,
        redcap_dataset: RedCapDataset,
        reproschema_source_dir: str,
        outdir: t.Union[str, Path],
        audiodir: t.Optional[t.Union[str, Path]] = None,
        max_audio_workers: int = 16
    ) -> 'BIDSDataset':
        """
        Create a BIDSDataset by converting a RedCapDataset to BIDS format.
        
        Args:
            redcap_dataset: The RedCapDataset to convert
            outdir: Output directory for BIDS structure
            audiodir: Optional directory containing audio files
            max_audio_workers: Number of parallel threads for audio copying (default: 16)
            
        Returns:
            BIDSDataset instance pointing to the created BIDS directory
        """
        outdir = Path(outdir).as_posix()
        BIDSDataset._initialize_data_directory(outdir)

        logging.info("Converting RedCap dataset to BIDS phenotype files.")
        # Subselect the RedCap dataframe and output components to individual files in the phenotype directory
        BIDSDataset._construct_phenotype_from_reproschema(
            df=redcap_dataset.df,
            source_dir=reproschema_source_dir,
            output_dir=os.path.join(outdir, "phenotype"),
        )

        if audiodir is None:
            # Return a new BIDSDataset instance pointing to the created directory
            return cls(outdir)

        logging.info("Processing audio files into BIDS format.")
        # We have two remaining tasks: (1) copy the audio files, and (2) create sidecar .json files.
        # First we prepare the metadata necessary for the .json files.
        participants_df = redcap_dataset.get_df_of_repeat_instrument(RepeatInstrument.PARTICIPANT.value)
        logging.info(f"Number of {RepeatInstrument.PARTICIPANT.name} entries: {len(participants_df)}")
        sessions_df = redcap_dataset.get_df_of_repeat_instrument(RepeatInstrument.SESSION.value)
        logging.info(f"Number of {RepeatInstrument.SESSION.name} entries: {len(sessions_df)}")
        acoustic_tasks_df = redcap_dataset.get_df_of_repeat_instrument(RepeatInstrument.ACOUSTIC_TASK.value)
        logging.info(f"Number of {RepeatInstrument.ACOUSTIC_TASK.name} entries: {len(acoustic_tasks_df)}")
        recordings_df = redcap_dataset.get_df_of_repeat_instrument(RepeatInstrument.RECORDING.value)
        logging.info(f"Number of {RepeatInstrument.RECORDING.name} entries: {len(recordings_df)}")

        logging.info("Creating dictionary lookups for BIDS hierarchy.")
        sessions_by_participant = defaultdict(list)
        for session in sessions_df.to_dict("records"):
            sessions_by_participant[session["record_id"]].append(session)

        tasks_by_session = defaultdict(list)
        for task in acoustic_tasks_df.to_dict("records"):
            tasks_by_session[task["acoustic_task_session_id"]].append(task)

        recordings_by_task = defaultdict(list)
        for recording in recordings_df.to_dict("records"):
            recordings_by_task[recording["recording_acoustic_task_id"]].append(recording)

        participants = []
        for participant in participants_df.to_dict("records"):
            participants.append(participant)
            participant["sessions"] = sessions_by_participant.get(participant["record_id"], [])

            for session in participant["sessions"]:
                session_id = session["session_id"]
                session["acoustic_tasks"] = tasks_by_session.get(session_id, [])
                
                for task in session["acoustic_tasks"]:
                    task["recordings"] = recordings_by_task.get(task["acoustic_task_id"], [])

        # Output participant data to FHIR format
        audio_files: t.List[Path] = []
        if audiodir is not None and Path(audiodir).exists():
            audio_files = list(Path(audiodir).rglob("*.wav"))

        # create an index of recording_id: audio_file for later use
        # ASSUMES that audio files are named with the recording_id in the filename
        # we use a defensive regex to grab uuid-like IDs from the stem just in case
        p_uuid = re.compile(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}')
        audio_files_by_recording: t.Dict[str, Path] = {}
        for audio_file in audio_files:
            match = p_uuid.search(audio_file.stem)
            if not match:
                continue
            uuid = match.group(0)
            if uuid in audio_files_by_recording:
                logging.warning(
                    f"Multiple audio files found for recording UUID {uuid}: "
                    f"{audio_files_by_recording[uuid]} and {audio_file}. "
                    "Only the last one will be retained."
                )
            audio_files_by_recording[uuid] = audio_file

        for participant in tqdm(participants, desc="Writing participant data to file"):
            cls._output_participant_data_to_fhir(
                participant,
                Path(outdir),
                audio_files_by_recording=audio_files_by_recording,
                max_audio_workers=max_audio_workers
            )
        
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

        template_package = "b2aiprep.template"
        copy_package_resource(template_package, "CHANGELOG.md", bids_dir_path)
        copy_package_resource(template_package, "README.md", bids_dir_path)
        copy_package_resource(template_package, "dataset_description.json", bids_dir_path)
        copy_package_resource(template_package, "phenotype", bids_dir_path)

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
    def _process_phenotype_tsv_and_json(
        df: pd.DataFrame, input_dir: str, output_dir: str, filename: str,
        clean_phenotype_data: bool = True
    ) -> None:
        """
        Create a stable, shareable phenotype artifact (TSV + JSON) driven by a schema.

        Args:
            df: DataFrame containing the data.
            input_dir: Directory containing JSON file with column labels.
            output_dir: Directory where the TSV and JSON files will be saved.
            filename: The JSON filename to process.
            clean_phenotype_data: Whether to clean the phenotype data (default: True).

        Outputs:
        - <output_dir>/<filename>.tsv: phenotype table constrained by the schema.
        - <output_dir>/<filename>.json: schema (data dictionary) aligned to the TSV.
        """
        # Load in the JSON which defines the columns to output to this subset
        with open(os.path.join(input_dir, filename), "r") as f:
            json_data = json.load(f)

        # The phenotype JSON files are nested below a key that corresponds to the schema name
        first_key = next(iter(json_data))
        column_labels = list(json_data[first_key]["data_elements"])

        # Our output columns are defined by the ReproSchema protocol file,
        # but we need to subselect to available data columns, as many columns
        # have been removed already before export from RedCap.
        data_elements = {}
        valid_columns = []
        for column in column_labels:
            if column not in df.columns:
                continue
            valid_columns.append(column)
            data_elements[column] = json_data[first_key]["data_elements"][column]

        if not valid_columns:
            _LOGGER.warning(f"No valid columns in dataframe for {first_key}")
            return df.iloc[:, 0:0].copy() # return empty dataframe with same index

        if "record_id" not in valid_columns:
            valid_columns = ["record_id"] + valid_columns

        # Select the relevant columns from the DataFrame and JSON
        selected_df = df[valid_columns]
        json_data[first_key]["data_elements"] = data_elements

        # Combine entries so there is one row per record_id
        df_subselected = selected_df.groupby("record_id").first().reset_index()

        if clean_phenotype_data:
            df_subselected, json_data = BIDSDataset._clean_phenotype_data(df_subselected, json_data)
        
        # Output to a TSV/JSON file.
        BIDSDataset._dataframe_to_tsv(df_subselected, os.path.join(output_dir, filename.replace(".json", ".tsv")))
        with open(os.path.join(output_dir, filename), "w") as f:
            json.dump(json_data, f, indent=2)

    @staticmethod
    def _load_reproschema(
        reproschema_file: Path,
    ) -> t.Dict[str, t.Dict[str, t.Any]]:
        with reproschema_file.open("r", encoding="utf-8") as fp:
            schema_json = json.load(fp)
        
        protocol_order = schema_json.get("ui", {}).get("order")

        reproschema_folder: Path = reproschema_file.parents[1]
        commit_sha = get_commit_sha(reproschema_folder)

        activities = {}
        for rel_path in protocol_order:
            # activities are relative to the reproschema schema file itself
            activity_path = reproschema_file.parent.joinpath(rel_path).resolve()
            if not activity_path.exists():
                _LOGGER.warning("Skipping missing activity %s", rel_path)
                continue

            activity_json = json.loads(activity_path.read_text())
            activity_id = activity_json.get("id")

            payload = build_activity_payload(
                activity_json=activity_json,
                activity_path=activity_path,
                reproschema_folder=reproschema_folder,
                commit_sha=commit_sha,
            )

            activities[activity_id] = payload

        return activities

    @staticmethod
    def _load_reorganization_file() -> pd.DataFrame:
        """Load the reproschema reorganization CSV file.

        Returns:
            DataFrame containing the reorganization data.
        """
        reorganization_file = files("b2aiprep.prepare.resources").joinpath("bids_field_organization.csv")
        return pd.read_csv(reorganization_file, sep=',', header=0)

    @staticmethod
    def _construct_phenotype_from_reproschema(
        df: pd.DataFrame,
        output_dir: str,
        source_dir: str,
        clean_phenotype_data: bool = True
    ) -> None:
        """Construct TSV/JSON files from a source ReproSchema folder.

        Args:
            df: DataFrame containing the data.
            output_dir: Directory where the TSV files will be saved.
            source_dir: Directory containing the ReproSchema JSON files.
            clean_phenotype_data: Whether to clean the phenotype data (default: True).
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Load reproschema file
        source_path = Path(source_dir).resolve()
        if source_path.joinpath('b2ai-redcap2rs_schema').exists():
            resolved_schema_file = source_path.joinpath('b2ai-redcap2rs_schema')
        elif source_path.joinpath('b2ai-redcap2rs', 'b2ai-redcap2rs_schema').exists():
            resolved_schema_file = source_path.joinpath('b2ai-redcap2rs', 'b2ai-redcap2rs_schema')
        else:
            raise FileNotFoundError(
                f"Could not find 'b2ai-redcap2rs_schema' in source directory: {source_dir}"
            )

        schemas = BIDSDataset._load_reproschema(resolved_schema_file)

        # create an index for data_element -> schema
        element_to_schema = {}
        for schema_name, data in schemas.items():
            for element_name in data['data_elements'].keys():
                element_to_schema[element_name] = schema_name

        # with all the reproschema activities defined, we now parse our manual reorganization
        # this is a CSV with:
        #   schema_name_source, column_name_source
        # ... used to identify the source reproschema element, and:
        #   schema_name, column_name, group, description
        # ... used to arrange & describe the output in the phenotype/ folder.
        df_reorg = BIDSDataset._load_reorganization_file()

        element_used = {} # keep track of whether we have used an element, for logging later.
        payload = {
            "description": "",
            "data_elements": {},
            # this variable will track the name of the original columns in RedCap
            "columns_for_indexing": [],
            # this variable will track the name of the *new* columns in the output df
            "columns_for_output": [],
            # group is popped before saving
            "group": "",
        }
        updated_schemas = defaultdict(lambda: deepcopy(payload))
        for activity_id, group in df_reorg.groupby('schema_name_source'):
            # get the *new* schema name, as this will be the base key of our updated dict
            column_mapping = group.set_index('column_name_source').to_dict(orient='index')
            for column, updated_data in column_mapping.items():
                if column not in df.columns:
                    _LOGGER.warning(f'Requested output for "{column}", but this column was not found in the source df.')
                    continue
                # the source schema is defined based on the element itself;
                # we do not need the schema_name_source column, but it is kept for ease of reading the CSV.
                schema_to_use = element_to_schema[column]

                # populate the detailed metadata for this column
                data_element = copy(schemas[schema_to_use]["data_elements"][column])
                if "description" in updated_data and (updated_data["description"] != ""):
                    description = updated_data["description"]
                elif "description" in data_element and (data_element["description"] != ""):
                    description = data_element["description"]
                else:
                    description = data_element.get("question", "").get("en", "")
                data_element["description"] = description
                new_element_name = updated_data["column_name"]

                updated_schema_name = updated_data["schema_name"]
                updated_schemas[updated_schema_name]['columns_for_indexing'].append(column)
                updated_schemas[updated_schema_name]['columns_for_output'].append(new_element_name)

                # update the payload so we have a reproschema json for this df
                updated_schemas[updated_schema_name]["data_elements"][new_element_name] = data_element
                updated_schemas[updated_schema_name]["group"] = updated_data["group"]
                element_used[column] = True

        # with our full updated_schemas dict prepared, we can iterate through the *new* schema names
        # and output a single folder for each
        for schema_name, payload in updated_schemas.items():
            columns_for_indexing = payload.pop("columns_for_indexing")
            columns_for_output = payload.pop("columns_for_output")
            group = payload.pop("group")
            if "record_id" not in columns_for_indexing:
                columns_for_indexing = ["record_id"] + columns_for_indexing
                columns_for_output = ["participant_id"] + columns_for_output
            # we now extract the sub-dataframe from our source redcap data and output it to tsv
            selected_df = df[columns_for_indexing]
            # Combine entries so there is one row per record_id
            selected_df = selected_df.groupby("record_id").first().reset_index()

            # TODO: verify we added the record_id description to the payload before writing
            updated_schema = {schema_name: payload}
            if clean_phenotype_data:
                selected_df, updated_schema = BIDSDataset._clean_phenotype_data(selected_df, updated_schema)
            
            # TODO: why is record_id sometimes here w/o this rename?
            selected_df = selected_df.rename(
                columns={old: new for old, new in zip(columns_for_indexing, columns_for_output)}
            )

            # check if *everything* is missing except for participant_id, if so we omit
            if "participant_id" in selected_df:
                null_cols = selected_df.drop(columns=["participant_id"]).isnull().all()
            else:
                null_cols = selected_df.isnull().all()
            if null_cols.all():
                _LOGGER.warning(f"All data missing for schema {schema_name}, skipping output.")
                continue

            # Output to a TSV/JSON file.
            filename = f'{schema_name}.json'
            if group != "":
                output_dir_grouped = os.path.join(output_dir, group)
                os.makedirs(output_dir_grouped, exist_ok=True)
            else:
                output_dir_grouped = output_dir
            BIDSDataset._dataframe_to_tsv(
                selected_df,
                os.path.join(output_dir_grouped, filename.replace(".json", ".tsv"))
            )
            with open(os.path.join(output_dir_grouped, filename), "w") as f:
                json.dump(updated_schema, f, indent=2)

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
        if pd.notna(session_id):
            session_id = str(session_id).replace(" ", "-").replace("_", "-")
            filename += f"_ses-{session_id}"
        if pd.notna(task_name):
            task_name = str(task_name).replace(" ", "-").replace("_", "-")
            if pd.notna(recording_name):
                task_name = str(recording_name).replace(" ", "-").replace("_", "-")
            filename += f"_task-{task_name}"

        schema_name = schema_name.replace(" ", "-").replace("schema", "").replace("_", "-")
        schema_name = schema_name + "-metadata"
        filename += f"_{schema_name}.json"

        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=False)
        with open(output_path / filename, "w") as f:
            f.write(data.json(indent=2))

    @staticmethod
    def _output_participant_data_to_fhir(
        participant: dict, outdir: Path, audio_files_by_recording: t.Optional[t.Dict[str, Path]] = None,
        max_audio_workers: int = 16
    ):
        """Output participant data to FHIR format.

        Args:
            participant: The participant data dictionary.
            outdir: The output directory path.
            audio_files_by_recording: Dictionary mapping recording IDs to audio file paths (optional).
            max_audio_workers: Number of parallel threads for audio copying (default: 16).
        """
        participant_id = participant["record_id"]
        subject_path = outdir / f"sub-{participant_id}"

        # TODO: prepare a Patient resource to use as the reference for each questionnaire
        # patient = create_fhir_patient(participant)

        session_instrument = BIDSDataset._get_instrument_for_name("sessions")
        task_instrument = BIDSDataset._get_instrument_for_name("acoustic_tasks")
        recording_instrument = BIDSDataset._get_instrument_for_name("recordings")

        # Collect all audio copy tasks for parallel execution
        audio_copy_tasks = []

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
                
                # Handle NaN/None values in acoustic_task_name
                acoustic_task_name = task.get("acoustic_task_name")
                if pd.isna(acoustic_task_name):
                    logging.warning(f"Skipping task with missing acoustic_task_name for participant {participant_id}, session {session_id}")
                    continue
                
                acoustic_task_name = acoustic_task_name.replace(" ", "-").replace("_", "-")
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
                    if audio_files_by_recording is None:
                        continue
                    audio_file = audio_files_by_recording.get(recording["recording_id"], None)

                    if not audio_file:
                        continue

                    # Schedule audio copy (to be executed in parallel later)
                    ext = audio_file.suffix
                    recording_name = recording["recording_name"].replace(" ", "-").replace("_", "-")
                    audio_file_destination = (
                        audio_output_path / f"{prefix}_task-{recording_name}{ext}"
                    )
                    
                    if not audio_file_destination.exists():
                        audio_copy_tasks.append((audio_file, audio_file_destination))

        # Execute all audio copies in parallel
        if audio_copy_tasks:
            _copy_audio_files_parallel(audio_copy_tasks, max_workers=max_audio_workers)

        # Save sessions.tsv
        sessions_df = pd.DataFrame(sessions_rows)
        if not os.path.exists(subject_path):
            os.mkdir(subject_path)
        sessions_tsv_path = subject_path / "sessions.tsv"
        sessions_df.to_csv(sessions_tsv_path, sep="\t", index=False)

    @staticmethod
    def load_phenotype_data(phenotype_filepath: Path) -> t.Tuple[pd.DataFrame, t.Dict[str, t.Any]]:
        """
        Load phenotype data from TSV and JSON files.
        
        Args:
            phenotype_filepath: Path to the phenotype file (without extension)
            
        Returns:
            Tuple of (DataFrame, phenotype_metadata_dict)
        """
        phenotype_name = phenotype_filepath.stem
        # Load TSV and JSON files
        df = pd.read_csv(phenotype_filepath.with_suffix('.tsv'), sep="\t")
        with open(phenotype_filepath.with_suffix('.json'), "r") as f:
            phenotype = json.load(f)
            data_elements = {}
            for schema in phenotype:
                data_elements.update(phenotype[schema].get("data_elements", {}))
            phenotype = data_elements

        # Handle nested phenotype structure which occurs in ReproSchema activities
        if len(phenotype) == 1:
            only_key = next(iter(phenotype))
            if 'data_elements' in phenotype[only_key]:
                phenotype = phenotype[only_key]['data_elements']

        # Add record_id to phenotype if missing
        if df.shape[1] > 0:
            phenotype_has_id = 'record_id' in list(phenotype.keys()) or 'participant_id' in list(phenotype.keys())
            df_has_id = 'record_id' in df.columns or 'participant_id' in df.columns
            if not phenotype_has_id and not df_has_id:
                phenotype = BIDSDataset._add_record_id_to_phenotype(phenotype)

        # Validate column count
        if len(phenotype) != df.shape[1]:
            logging.warning(
                f"Phenotype {phenotype_name} has {len(phenotype)} columns, but the data has {df.shape[1]} columns."
            )

        return df, phenotype

    @staticmethod
    def _map_series(series: pd.Series, mapping: dict) -> pd.Series:
        """
        Map values in a pandas Series using a provided mapping dictionary. Log any issues arising.
        
        Args:
            series: The pandas Series to map.
            mapping: The mapping dictionary.
        Returns:
            The mapped pandas Series.
        """
        ids_before = series.copy()
        series = series.map(mapping).fillna(series)
        idxUnchanged = ids_before == series
        n_unchanged = idxUnchanged.sum()
        logging.info(f"Remapped {len(ids_before) - n_unchanged} / {len(ids_before)} IDs for '{series.name}'")
        if n_unchanged > 0:
            logging.warning(f"A subset of IDs are missing remapping for '{series.name}': {set(ids_before[idxUnchanged])}")
        return series

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
            df["participant_id"] = BIDSDataset._map_series(df["participant_id"], participant_ids_to_remap)

        if participant_session_id_to_remap:
            for col in df.columns:
                if "session_id" in col:
                    df[col] = BIDSDataset._map_series(df[col], participant_session_id_to_remap)

        # Remove sensitive columns
        df, phenotype = BIDSDataset._remove_sensitive_columns(df, phenotype)

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
        
        # Warn about empty columns - but keep them as some are expected
        BIDSDataset._warn_about_empty_columns(df, phenotype)
        
        # Add derived columns
        if ("gender_identity" in df.columns) and ("specify_gender_identity" in df.columns):
            df, phenotype = BIDSDataset._add_sex_at_birth_column(df, phenotype)

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
    def _warn_about_empty_columns(df: pd.DataFrame, phenotype: dict) -> None:
        """Warn about columns that are empty."""
        empty_columns = []
        for column in df.columns:
            if df[column].isnull().all():
                empty_columns.append(column)
        
        if empty_columns:
            logging.warning(f"Found {len(empty_columns)} empty columns: {empty_columns}")

    @staticmethod
    def _remove_sensitive_columns(df: pd.DataFrame, phenotype: dict) -> t.Tuple[pd.DataFrame, dict]:
        """Remove columns with sensitive data (free-text, geo-location, etc)."""
        columns_to_drop = [
            "state_province",
            "zipcode"
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
            "peds_zipcode",
            "peds_other_race_specify",
            "peds_other_primary_language",
            "peds_mc_conditions_other_specified",
            "peds_mc_chronic_medical_condition_specified",
            "peds_mc_genetic_syndromes_specified",
            "peds_mc_hospitalized_specified",
            "peds_mc_allergies_specified",
            "peds_mc_dif_swallowing_specified",
            "peds_mc_ear_inf_ant_py_specified",
            "peds_mc_etp_procedure",
            "peds_mc_eval_voice_swal_c_specified",
            "peds_mc_ft_specified",
            "peds_mc_reflux_specified",
            "peds_mc_hl_specified",
            "peds_mc_hw_voice_2_w_specified",
            "peds_mc_no_surgeries_procedure",
            "peds_mc_stridor_specified",
            "peds_mc_ox_sup_specified",
            "peds_mc_meds_specified",
            "peds_mc_v_dis_specified",
            "peds_mc_a_therapy_specified",
            "peds_mc_surgery_t_vc_a_specified",
            "peds_mc_fr_inf_tons_specified",
            "peds_mc_fr_v_f_specified",
            # below could be considered for inclusion in the future
            "peds_mc_tonsillectomy_date",
            "peds_mc_adenoidectomy_date",
            "peds_mc_neck_mass_branchial_cleft_cyst_surgery_date",
            "peds_mc_etp_procedure_date",
            "peds_mc_neck_mass_dermoid_cyst_surgery_date",
            "peds_mc_neck_mass_enlarged_lymph_node_surgery_date",
            "peds_mc_lingual_tonsillectomy_date",
            "peds_mc_no_surgeries_procedure_date",
            "peds_mc_neck_mass_thyroglossal_duct_cyst_surgery_date",
            "peds_mc_neck_mass_hyroid_nodule_or_cancer_surgery_date",
            "peds_mc_v_dis_specified"
        ]
        df, phenotype = BIDSDataset._drop_columns_from_df_and_data_dict(
            df, phenotype, columns_to_drop, "Removing low utility / PHI containing columns"
        )

        if 'gender_identity' in df.columns:
            _LOGGER.info(f"sex_at_birth value_counts: {df['sex_at_birth'].value_counts(dropna=False).to_dict()}")
            _LOGGER.info(f"gender_identity value_counts: {df['gender_identity'].value_counts(dropna=False).to_dict()}")
            df, phenotype = BIDSDataset._drop_columns_from_df_and_data_dict(
                df, phenotype, ["gender_identity"], "Remove sensitive demographic columns"
            )
        
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
                # this continue implicitly removes this column from the output
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
    def map_sequential_session_ids(folder_path: Path) ->  t.Dict[str, str]:
        """
            Function retrieves all session ids for all participants from the sessions.tsv files and maps them to an integer.
            If there are mutliple session exists for a participant, the seqiential will increase sequentially.
        """
        _LOGGER = logging.getLogger(__name__)
        folder = Path(folder_path)
        session_files = list(folder.rglob("sessions.tsv"))

        if not session_files:
            _LOGGER.warning((f"No 'sessions.tsv' files found under {folder_path}"))
            return {}

        dfs = []
        for f in session_files:
            try:
                df = pd.read_csv(f, sep='\t', dtype={'session_id': str})
                dfs.append(df)
            except Exception as e:
                _LOGGER.error(f" Could not read {f}: {e}")

        combined = pd.concat(dfs, ignore_index=True)

        # Sort so all rows of a record_id appear together
        if 'record_id' in combined.columns and 'session_id' in combined.columns:
            combined.sort_values(by=['record_id', 'session_id'], inplace=True)
        
        combined['session_number'] = combined.groupby('record_id').cumcount() + 1
        session_id_dict = combined.set_index('session_id')['session_number'].astype(str).to_dict()
                
        return session_id_dict

    @staticmethod
    def load_participant_ids_to_remove(publish_config_dir: Path) -> t.List[str]:
        """Load list of participant IDs to remove from JSON file."""
        participant_to_remove_path = publish_config_dir / "participants_to_remove.json"
        if not participant_to_remove_path.exists():
            # If file doesn't exist, raise an error
            raise FileNotFoundError(f"Participant IDs to remove file {participant_to_remove_path} does not exist.")

        with open(participant_to_remove_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Participant IDs to remove file {participant_to_remove_path} should contain a list of participant IDs.")
        
        logging.info(f"Loaded {len(data)} participant IDs to remove: {data}.")
        return data
    
    @staticmethod
    def load_audio_filestems_to_remove(publish_config_dir: Path) -> t.List[str]:
        """Load list of audio file stems to remove from JSON file."""
        audio_to_remove_path = publish_config_dir / "audio_filestems_to_remove.json"
        if not audio_to_remove_path.exists():
            # If file doesn't exist, raise an error
            raise FileNotFoundError(f"Audio filestems to remove file {audio_to_remove_path} does not exist.")

        with open(audio_to_remove_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Audio filestems to remove file {audio_to_remove_path} should contain a list of audio file stems.")
        
        return data

    @staticmethod
    def load_sensitive_audio_tasks(deidentify_config_dir: Path) -> t.List[str]:
        """Load list of audio tasks that are sensitive from JSON file."""
        sensitive_audio_tasks_path = deidentify_config_dir / "sensitive_audio_tasks.json"
        if not sensitive_audio_tasks_path.exists():
            # If file doesn't exist, raise an error
            raise FileNotFoundError(f"Sensitive audio tasks file {sensitive_audio_tasks_path} does not exist.")

        with open(sensitive_audio_tasks_path, 'r') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Sensitive audio tasks file {sensitive_audio_tasks_path} should contain a list of audio task names.")
        
        return data

    def deidentify(self, outdir: t.Union[str, Path], deidentify_config_dir: Path, skip_audio: bool = False, skip_audio_features: bool = False) -> 'BIDSDataset':
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
            deidentify_config_dir: Config directory for doing deidentification
            skip_audio: If True, skip copying/processing audio files
            
        Returns:
            New BIDSDataset instance pointing to the deidentified directory
        """
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=False)  # Don't overwrite existing directories
        
        participant_ids_to_remap = BIDSDataset.load_remap_id_list(deidentify_config_dir)
        participant_ids_to_remove = BIDSDataset.load_participant_ids_to_remove(deidentify_config_dir)
        audio_filestems_to_remove = BIDSDataset.load_audio_filestems_to_remove(deidentify_config_dir)
        sensitive_audio_tasks = BIDSDataset.load_sensitive_audio_tasks(deidentify_config_dir)
        participant_session_id_to_remap = BIDSDataset.map_sequential_session_ids(self.data_path)
        
        # Process phenotype directory if it exists
        phenotype_base_path = self.data_path.joinpath("phenotype")
        if phenotype_base_path.exists():
            logging.info("Processing phenotype data for deidentification.")
            phenotype_output_path = outdir.joinpath("phenotype")
            phenotype_output_path.mkdir(parents=True, exist_ok=True)
            
            for phenotype_filepath in phenotype_base_path.rglob("*.tsv"):
                logging.info(f"Processing {phenotype_filepath.stem}.")
                df_pheno, phenotype_dict = BIDSDataset.load_phenotype_data(phenotype_filepath)
                df_pheno, phenotype_dict = BIDSDataset._deidentify_phenotype(df_pheno, phenotype_dict, participant_ids_to_remove, participant_ids_to_remap, participant_session_id_to_remap)
                
                # Write out phenotype data and data dictionary
                phenotype_subdir = phenotype_output_path.joinpath(phenotype_filepath.parent.relative_to(phenotype_base_path))
                phenotype_subdir.mkdir(parents=True, exist_ok=True)
                df_pheno.to_csv(
                    phenotype_subdir.joinpath(f"{phenotype_filepath.stem}.tsv"), 
                    sep="\t", index=False
                )
                with open(phenotype_subdir.joinpath(f"{phenotype_filepath.stem}.json"), "w") as f:
                    json.dump(phenotype_dict, f, indent=2)
            logging.info("Finished processing phenotype data.")
        
        if not skip_audio:
            # Process audio files
            logging.info("Processing audio files for deidentification.")
            BIDSDataset._deidentify_audio_files(
                self.data_path, 
                outdir, 
                participant_ids_to_remove, 
                audio_filestems_to_remove,
                sensitive_audio_tasks, 
                participant_ids_to_remap, 
                participant_session_id_to_remap,
            )
            logging.info("Finished processing audio files.")

        if not skip_audio_features:
            # Process audio files
            logging.info("Processing audio features for deidentification.")
            BIDSDataset._deidentify_feature_files(
                self.data_path, 
                outdir, 
                participant_ids_to_remove, 
                audio_filestems_to_remove,
                sensitive_audio_tasks, 
                participant_ids_to_remap, 
                participant_session_id_to_remap,
            )
            logging.info("Finished processing features.")
        
        # Copy over the standard BIDS template files if they exist
        for template_file in ["README.md", "CHANGES.md", "dataset_description.json"]:
            template_path = self.data_path.joinpath(template_file)
            if template_path.exists():
                shutil.copy(template_path, outdir)
        
        logging.info("Deidentification completed.")
        return BIDSDataset(outdir)

    @staticmethod
    def _collect_paths(
        data_path: Path,
        file_extension: str
    ) -> t.List[Path]:
        """Collect all file paths with the given extension from the dataset."""
        paths = get_paths(data_path, file_extension=file_extension)
        paths = [x["path"] for x in paths]
        
        # Sort audio paths for consistent processing
        paths = sorted(
            paths,
            # sort first by subject, then by task
            key=lambda x: (x.stem.split("_")[0], x.stem.split("_")[2]),
        )
        return paths

    @staticmethod
    def _apply_exclusion_list_to_filepaths(
        paths: t.List[Path],
        exclusion_list: t.List[str],
        exclusion_type: str = 'participant'
    ) -> t.List[Path]:
        """Remove filepaths based on overlap with a specified exclusion list."""
        n = len(paths)
        exclusion = set(exclusion_list)
        if len(exclusion) == 0:
            return paths

        if exclusion_type == 'participant':
            paths = [
                x for x in paths
                if all(f"sub-{pid}" not in str(x) for pid in exclusion)
            ]
        elif exclusion_type == 'filename':
            paths = [
                a for a in paths
                if a.stem not in exclusion
            ]
        elif exclusion_type == 'filestem_contains':
            paths = [
                a for a in paths
                if all(excl not in a.stem for excl in exclusion)
            ]
            # for better logging, add the list of exclusions to the exclusion type
            exclusion_type += f" ({', '.join(exclusion)})"
        else:
            raise ValueError(f"Unknown exclusion_type: {exclusion_type}")
        if len(paths) < n:
            _LOGGER.info(
                f"Removed {n - len(paths)} records due to exclusion: {exclusion_type}."
            )
        return paths

    @staticmethod
    def _extract_participant_id_from_path(path: Path) -> str:
        """Extract participant ID from the path, preferring directory parts.
        Falls back to regex on filestem if needed."""
        # Prefer directory parts
        for part in path.parts:
            if part.startswith("sub-"):
                return part[4:]

        # Fallback to filestem regex
        m = re.search(r"sub-([A-Za-z0-9\-]+)", path.stem)
        if m:
            return m.group(1)

        raise ValueError(f"Could not extract participant ID from path: {path}")

    @staticmethod
    def _extract_session_id_from_path(path: Path) -> str:
        """Extract session ID from the path, preferring directory parts.
        Falls back to regex on filestem using ses-(.+?)_ if needed."""
        # Prefer directory parts
        for part in path.parts:
            if part.startswith("ses-"):
                return part[4:]

        # Fallback to filestem regex: ses-(.+?)_
        m = re.search(r"ses-(.+?)_", path.stem)
        if m:
            return m.group(1)

        raise ValueError(f"Could not extract session ID from path: {path}")

    @staticmethod
    def _extract_task_name_from_path(path: Path) -> str:
        """Extract the task name from the stem of the path.
        
        Tasks are optional components of filenames. They must follow the `task-<label>` pattern."""
        m = re.search(r"task-(.+?)(_|$)", path.stem)
        if m:
            return m.group(1)

        raise ValueError(f"Could not extract task name from path: {path}")

    @staticmethod
    def _deidentify_audio_files(
        data_path: Path,
        outdir: Path,
        exclude_participant_ids: t.List[str] = [],
        exclude_audio_filestems: t.List[str] = [],
        sensitive_audio_task_list: t.List[str] = [],
        participant_ids_to_remap: t.Dict[str, str] = {},
        participant_session_id_to_remap: t.Dict[str, str] = {},
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
            data_path: Directory of the input BIDS dataset
            outdir: Output directory for deidentified audio files
            exclude_participant_ids: list of participant IDs to exclude
            exclude_audio_filestems: list of audio filenames to exclude
            sensitive_audio_task_list: list of sensitive audio tasks
            participant_ids_to_remap: map between old and new participant IDs
            participant_session_id_to_remap: map between old and new session IDs
        """
        # Get all audio paths
        audio_paths = BIDSDataset._collect_paths(data_path, file_extension=".wav")
        if len(audio_paths) == 0:
            _LOGGER.warning(f"No audio files (.wav) found in {data_path}.")
            return
        
        # Remove known individuals (hard-coded participant removal)
        audio_paths = BIDSDataset._apply_exclusion_list_to_filepaths(
            audio_paths, exclusion_list=exclude_participant_ids, exclusion_type='participant'
        )

        # Remove known files (hard-coded file-based removal)
        audio_paths = BIDSDataset._apply_exclusion_list_to_filepaths(
            audio_paths, exclusion_list=exclude_audio_filestems, exclusion_type='filename'
        )

        # Remove specific tasks
        audio_paths = BIDSDataset._apply_exclusion_list_to_filepaths(
            audio_paths, exclusion_list=['audio-check'], exclusion_type='filestem_contains'
        )

        sensitive_audio_task_list = [f'task-{task}' for task in sensitive_audio_task_list]
        audio_paths = BIDSDataset._apply_exclusion_list_to_filepaths(
            audio_paths, exclusion_list=sensitive_audio_task_list, exclusion_type='filestem_contains'
        )

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
            update_metadata_record_and_session_id(metadata, participant_ids_to_remap, participant_session_id_to_remap)
            participant_id = get_value_from_metadata(metadata, linkid="participant_id", endswith=False)
            session_id = get_value_from_metadata(metadata, linkid="session_id", endswith=True)
            
            # Create output path with deidentified structure
            audio_path_stem_ending = '-'.join(audio_path.stem.split("_")[2:])
            output_path = outdir.joinpath(
                f"sub-{participant_id}/ses-{session_id}/audio/sub-{participant_id}_ses-{session_id}_{audio_path_stem_ending}.wav"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
            # Copy over the associated .json file and audio data
            with open(output_path.with_suffix(".json"), "w") as fp:
                json.dump(metadata, fp, indent=2)
            shutil.copy(audio_path, output_path)


    @staticmethod
    def _deidentify_feature_files(
        data_path: Path,
        outdir: Path,
        exclude_participant_ids: t.List[str] = [],
        exclude_audio_filestems: t.List[str] = [],
        sensitive_audio_task_list: t.List[str] = [],
        participant_ids_to_remap: t.Dict[str, str] = {},
        participant_session_id_to_remap: t.Dict[str, str] = {},
    ):
        """
        Copy and deidentify audio feature files to the output directory.
        
        This method:
        1. Gets all audio feature paths from the dataset
        2. Removes participants from hard-coded exclusion list
        3. Filters out sensitive files
        4. Removes subfields from sensitive files
        5. Copies feature files to the new location
        
        Args:
            data_path: Directory of the input BIDS dataset
            outdir: Output directory for deidentified audio files
            exclude_participant_ids: list of participant IDs to exclude
            exclude_audio_filestems: list of audio filenames to exclude
            sensitive_audio_task_list: list of sensitive audio tasks
            participant_ids_to_remap: map between old and new participant IDs
            participant_session_id_to_remap: map between old and new session IDs
        """
        # Get all audio paths
        paths = BIDSDataset._collect_paths(data_path, file_extension=".pt")
        if len(paths) == 0:
            _LOGGER.warning(f"No feature files (.pt) found in {data_path}.")
            return
        
        # Remove known individuals (hard-coded participant removal)
        paths = BIDSDataset._apply_exclusion_list_to_filepaths(
            paths, exclusion_list=exclude_participant_ids, exclusion_type='participant'
        )

        # Remove known files (hard-coded file-based removal)
        paths = BIDSDataset._apply_exclusion_list_to_filepaths(
            paths, exclusion_list=exclude_audio_filestems, exclusion_type='filename'
        )

        # Remove specific tasks
        paths = BIDSDataset._apply_exclusion_list_to_filepaths(
            paths, exclusion_list=['audio-check'], exclusion_type='filestem_contains'
        )
        
        _LOGGER.info(f"Copying {len(paths)} feature files.")
        for features_path in tqdm(
            paths, desc="Copying and de-identifying feature files", total=len(paths)
        ):
            participant_id = BIDSDataset._extract_participant_id_from_path(features_path)
            participant_id = participant_ids_to_remap.get(participant_id, participant_id)
            session_id = BIDSDataset._extract_session_id_from_path(features_path)
            session_id = participant_session_id_to_remap.get(session_id, session_id)

            # Create output path with deidentified structure
            features_ending = '_features'
            audio_path_stem = features_path.stem.replace(features_ending,'')
            path_stem_ending = '-'.join(audio_path_stem.split("_")[2:]) + features_ending
            output_path = outdir.joinpath(
                f"sub-{participant_id}/ses-{session_id}/audio/sub-{participant_id}_ses-{session_id}_{path_stem_ending}{features_path.suffix}"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # if it is not sensitive and we want to keep features, move all features over
            task_name = BIDSDataset._extract_task_name_from_path(features_path)
            if task_name not in sensitive_audio_task_list:
                shutil.copy(features_path, output_path)
            else:
                features = torch.load(features_path, weights_only=False, map_location=torch.device('cpu'))
                # Sensitive features to remove
                for torchaudio_field in ['mel_filter_bank', 'mfcc', 'mel_spectrogram', 'spectrogram']:
                    features['torchaudio'].pop(torchaudio_field, None)
                for field in ['ppgs', 'transcription']:
                    features.pop(field, None)
                torch.save(features, output_path)


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
                device = 'cpu' # not checking for cuda because optimization would be minimal if any
                features = torch.load(str(audio_file), weights_only=False, map_location=torch.device(device))
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
