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

import os
from collections import OrderedDict
from pathlib import Path
import re
import typing as t


from fhir.resources.questionnaireresponse import QuestionnaireResponse
import pandas as pd

class BIDSDataset:
    def __init__(self, data_path: t.Union[Path, str, os.PathLike]):
        self.data_path = Path(data_path).resolve()

    def find_subject_questionnaires(self, subject_id: str) -> t.Dict[str, Path]:
        """
        Find all the questionnaires for a given subject.

        Parameters
        ----------
        subject_id : str
            The subject identifier.

        Returns
        -------
        Dict[str, Path]
            A dictionary of questionnaires for the subject.
        """
        subject_path = self.data_path / f"sub-{subject_id}"
        questionnaires = {}
        for questionnaire in subject_path.glob("sub-*.json"):
            questionnaire_name = questionnaire.stem
            questionnaires[questionnaire_name] = questionnaire
        return questionnaires

    def find_session_questionnaires(self, subject_id: str, session_id: str) -> t.Dict[str, Path]:
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
        Dict[str, Path]
            A dictionary of questionnaires for the subject and session.
        """
        session_path = self.data_path / f"sub-{subject_id}" / f"ses-{session_id}"
        questionnaires = {}
        for questionnaire in session_path.glob("sub-*.json"):
            questionnaire_name = questionnaire.stem
            questionnaires[questionnaire_name] = questionnaire
        return questionnaires

    def find_subjects(self) -> t.Dict[str, Path]:
        """
        Find all the subjects in the dataset.

        Returns
        -------
        Dict[str, Path]
            A dictionary of subjects.
        """
        subjects = {}
        for subject in self.data_path.glob("sub-*"):
            subject_id = subject.stem[4:]
            subjects[subject_id] = subject
        return subjects

    def find_sessions(self, subject_id: str) -> t.Dict[str, Path]:
        """
        Find all the sessions for a given subject.

        Parameters
        ----------
        subject_id : str
            The subject identifier.

        Returns
        -------
        Dict[str, Path]
            A dictionary of sessions for the subject.
        """
        subject_path = self.data_path / f"sub-{subject_id}"
        sessions = {}
        for session in subject_path.glob("ses-*"):
            session_id = session.stem[4:]
            sessions[session_id] = session
        return sessions

    def list_subjects(self) -> list:
        """
        List all the subjects in the dataset.

        Returns
        -------
        list
            A list of subject identifiers.
        """
        return list(self.find_subjects().keys())
    
    def list_sessions(self, subject_id: str) -> list:
        """
        List all the sessions for a given subject.

        Parameters
        ----------
        subject_id : str
            The subject identifier.

        Returns
        -------
        list
            A list of session identifiers.
        """
        return list(self.find_sessions(subject_id).keys())
    
    def list_questionnaires(self, subject_id: t.Optional[str] = None) -> list:
        """
        List all the questionnaires. If subject_id is provided, lists
        only the questionnaires for a given subject.
        
        Parameters
        ----------
        subject_id : str, optional
            The subject identifier, by default None.
        
        Returns
        -------
        list
            A list of questionnaire names.
        """
        if subject_id is not None:
            questionnaires = self.find_subject_questionnaires(subject_id)
        else:
            questionnaires = self.find_questionnaires()
        return list(questionnaires.keys())
        
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
    
    def load_subject_questionnaires(self, subject_id: str) -> t.Dict[str, QuestionnaireResponse]:
        """
        Load all the questionnaires for a given subject.

        Parameters
        ----------
        subject_id : str
            The subject identifier.

        Returns
        -------
        Dict[str, pd.DataFrame]
            A dictionary of questionnaire data.
        """
        questionnaires = self.find_subject_questionnaires(subject_id)
        return {name: self.load_questionnaire(path) for name, path in questionnaires.items()}
    
    def find_questionnaires(self, questionnaire_name: str) -> t.Dict[str, Path]:
        """
        Find all the questionnaires with a given name.

        Parameters
        ----------
        questionnaire_name : str
            The name of the questionnaire.

        Returns
        -------
        Dict[str, Path]
            A dictionary of questionnaires.
        """
        questionnaires = {}
        for questionnaire in self.data_path.rglob(f"sub-*_{questionnaire_name}.json"):
            subject_id = questionnaire.stem.split("_")[0]
            questionnaires[subject_id] = questionnaire
        return questionnaires
    
    def load_questionnaires(self, questionnaire_name: str) -> t.Dict[str, QuestionnaireResponse]:
        """
        Load all the questionnaires with a given name.

        Parameters
        ----------
        questionnaire_name : str
            The name of the questionnaire.

        Returns
        -------
        Dict[str, pd.DataFrame]
            A dictionary of questionnaire data.
        """
        questionnaires = self.find_questionnaires(questionnaire_name)
        return {subject_id: self.load_questionnaire(path) for subject_id, path in questionnaires.items()}
    
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
            The questionnaire data as a DataFrame.
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
            if "answer" in item:
                items.append(OrderedDict(
                    linkId=item["linkId"],
                    **item["answer"][0],
                ))
            else:
                items.append(OrderedDict(
                    linkId=item["linkId"],
                    valueString=None,
                ))
            # HACK: parse the questionnaire ID to get the record_id
            # this should use the subjectOf reference, when implemented
            items[-1]['record_id'] = questionnaire_dict['id'][:36]
        # unroll based on the possible value options
        return pd.DataFrame(items)

    def list_audio(self, subject_id: str, session_id: str) -> list:
        """
        List all the audio recordings for a given subject and session.

        Parameters
        ----------
        subject_id : str
            The subject identifier.
        session_id : str
            The session identifier.

        Returns
        -------
        list
            A list of audio recordings.
        """
        session_path = self.data_path / f"sub-{subject_id}" / f"ses-{session_id}" / "audio"
        audio = []
        for audio_file in session_path.glob("*.wav"):
            audio.append(audio_file)
        return audio


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
        columns_with_underscores = sorted(list(set([
            col[:col.rindex('__')-1]
            for col in df.columns
            if re.search('__[0-9]+$', col) is not None
        ])))

        # iterate through each prefix and merge together data into this prefix
        for col in columns_with_underscores:
            columns_to_merge = df.filter(like=f'{col}__').columns
            df[col] = df[columns_to_merge].apply(lambda x: next((i for i in x if i is not None), None), axis=1)
            df.drop(columns=columns_to_merge, inplace=True)
        return df

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
        session_path = self.data_path / f"sub-{subject_id}" / f"ses-{session_id}" / "audio_features"
        features = []
        for feature_file in session_path.glob("*.json"):
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
        session_path = self.data_path / f"sub-{subject_id}" / f"ses-{session_id}" / "audio_transcripts"
        transcripts = []
        for transcript_file in session_path.glob("*.json"):
            transcripts.append(transcript_file)
        return transcripts
    
    def load_and_pivot_questionnaire(self, questionnaire_name: str) -> pd.DataFrame:
        """
        Loads all data for a questionnaire and pivots on the appropriate identifier column.

        Returns
        -------
        pd.DataFrame
            A "wide" format dataframe with either record_id or session_id as the index.
        """
        # search across all subjects and get any file ending in demographics
        qs = self.load_questionnaires(questionnaire_name)
        q_dfs = []
        for _, questionnaire in qs.items():
            # get the dataframe for this questionnaire
            df = self.questionnaire_to_dataframe(questionnaire)
            q_dfs.append(df)

        # concatenate all the dataframes
        pivoted_df = pd.concat(q_dfs)
        pivoted_df = pd.pivot(pivoted_df, index='record_id', columns='linkId', values='valueString')

        # restore the original order of the columns, as pd.pivot removes it and sorts alphabetically
        if len(q_dfs) > 0:
            columns = q_dfs[-1]['linkId'].tolist()
            columns = [col for col in columns if col in pivoted_df.columns]
            # add in any columns which may be in pivoted
            columns.extend([col for col in pivoted_df.columns if col not in columns])
            pivoted_df = pivoted_df[columns]

        return pivoted_df
