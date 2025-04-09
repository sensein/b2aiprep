import logging
import typing as t
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from b2aiprep.prepare.constants import PARTICIPANT_ID_TO_REMOVE


_LOGGER = logging.getLogger(__name__)

def spectrogram_generator(
    audio_paths,
) -> t.Generator[t.Dict[str, t.Any], None, None]:
    """Load audio features from individual files and yield dictionaries amenable to HuggingFace's Dataset from_generator."""
    audio_paths = sorted(
        audio_paths,
        # sort first by subject, then by task
        key=lambda x: (x.stem.split('_')[0], x.stem.split('_')[2])
    )

    for wav_path in tqdm(audio_paths, total=len(audio_paths), desc="Extracting features"):
        output = {}
        pt_file = wav_path.parent / f"{wav_path.stem}_features.pt"
        features = torch.load(pt_file, weights_only=False)

        output['participant_id'] = wav_path.stem.split('_')[0][4:] # skip "sub-" prefix
        output['session_id'] = wav_path.stem.split('_')[1][4:] # skip "ses-" prefix
        output['task_name'] = wav_path.stem.split('_')[2][5:] # skip "task-" prefix

        spectrogram = features['torchaudio']['spectrogram']
        spectrogram = 10.0 * torch.log10(torch.maximum(spectrogram, torch.tensor(1e-10)))
        spectrogram = torch.maximum(spectrogram, spectrogram.max() - 80)
        spectrogram = spectrogram.numpy().astype(np.float32)
        # skip every other column
        spectrogram = spectrogram[:, ::2]
        output['spectrogram'] = spectrogram

        yield output


def load_audio_features(
    audio_paths,
) -> t.Generator[t.Dict[str, t.Any], None, None]:
    """Load audio features from individual files and yield dictionaries amenable to HuggingFace's Dataset from_generator."""
    for wav_path in tqdm(audio_paths, total=len(audio_paths), desc="Extracting features"):
        wav_path = Path(wav_path).resolve()
        audio_dir = wav_path.parent
        features_dir = audio_dir.parent / "audio"

        output = {}
        # get the subject_id and session_id for this data
        # TODO: we should appropriately load these as FHIR resources to validate the data
        metadata_filepath = wav_path.parent.joinpath(wav_path.stem + "_recording-metadata.json")
        metadata = json.loads(metadata_filepath.read_text())

        for item in metadata["item"]:
            if item["linkId"] == "record_id":
                output["subject_id"] = item["answer"][0]["valueString"]
            elif item["linkId"] == "recording_session_id":
                output["session_id"] = item["answer"][0]["valueString"]
            elif item["linkId"] == "recording_name":
                output["recording_name"] = item["answer"][0]["valueString"]
            elif item["linkId"] == "recording_duration":
                output["recording_duration"] = item["answer"][0]["valueString"]

        # feature_path = features_dir / f"{wav_path.stem}_transcription.txt"
        # with open(feature_path, "r", encoding="utf-8") as text_file:
        #     transcription = text_file.read()
        # output["transcription"] = transcription

        pt_file = features_dir / f"{wav_path.stem}_features.pt"
        features = torch.load(pt_file, weights_only=False)
        output['spectrogram'] = features['torchaudio']['spectrogram'].numpy().astype(np.float32)
        # for feature_name in ["speaker_embedding", "specgram", "melfilterbank", "mfcc", "opensmile"]:
        #     feature_path = features_dir / f"{wav_path.stem}_{feature_name}.{file_extension}"
        #     if not feature_path.exists():
        #         continue
        #     if feature_name == "speaker_embedding":
        #         output["speaker_embedding"] = torch.load(feature_path)
        #     else:
        #         data = torch.load(feature_path)
        #         if len(data) == 1:
        #             key = next(iter(data.keys()))
        #             data[key] = data[key].numpy().astype(np.float32)
        #         output.update(torch.load(feature_path))

        # load transcription
        # feature_path = features_dir / f"{wav_path.stem}_transcription.txt"
        # if feature_path.exists():
        #     output["transcription"] = feature_path.read_text()

        yield output

def feature_extraction_generator(
    audio_paths,
    feature_name: str,
) -> t.Generator[t.Dict[str, t.Any], None, None]:
    """Load audio features from individual files and yield dictionaries amenable to HuggingFace's Dataset from_generator."""
    audio_paths = sorted(
        audio_paths,
        # sort first by subject, then by task
        key=lambda x: (x.stem.split('_')[0], x.stem.split('_')[2])
    )
    if feature_name not in ('spectrogram', 'mfcc'):
        raise ValueError(f"Feature name {feature_name} not supported.")

    for wav_path in tqdm(audio_paths, total=len(audio_paths), desc="Extracting features"):
        output = {}
        pt_file = wav_path.parent / f"{wav_path.stem}_features.pt"
        features = torch.load(pt_file, weights_only=False)

        output['participant_id'] = wav_path.stem.split('_')[0][4:] # skip "sub-" prefix
        output['session_id'] = wav_path.stem.split('_')[1][4:] # skip "ses-" prefix
        output['task_name'] = wav_path.stem.split('_')[2][5:] # skip "task-" prefix

        data = features['torchaudio'][feature_name]
        if feature_name == 'spectrogram':
            data = 10.0 * torch.log10(torch.maximum(data, torch.tensor(1e-10)))
            data = torch.maximum(data, data.max() - 80)
            data = data.numpy().astype(np.float32)
        else:
            data = data.numpy()

        data = data[:, ::2]
        output[feature_name] = data

        yield output

def _drop_columns_from_df_and_data_dict(
        df: pd.DataFrame,
        phenotype: dict,
        columns_to_drop: t.List[str],
        message: str
):
    """Drop columns from the DataFrame and phenotype dictionary."""
    columns_to_drop_in_df = []
    for col in columns_to_drop:
        if col in df:
            columns_to_drop_in_df.append(col)

    if len(columns_to_drop_in_df) > 0:
        _LOGGER.info(message + f": {columns_to_drop_in_df}")
        df = df.drop(columns=columns_to_drop_in_df)
        phenotype = {
            k: v for k, v in phenotype.items() if k not in columns_to_drop_in_df
        }
    return df, phenotype

def clean_phenotype_data(df: pd.DataFrame, phenotype: dict) -> t.Tuple[pd.DataFrame, dict]:
    """Remove known errors occurring in the phenotype dataframe."""
    # the alcohol_amt column has dates instead of values
    date_fix_map = {
        "4-Mar": "3 - 4",
        "6-May": "5 - 6",
        "9-Jul": "7 - 9",
    }
    df['alcohol_amt'] = df['alcohol_amt'].apply(
        lambda x: date_fix_map[x] if x in date_fix_map else x)

    # remove columns which are empty
    # columns_to_drop = df.columns[df.isnull().all()].tolist()
    # df, phenotype = _drop_columns_from_df_and_data_dict(
    #     df,
    #     phenotype,
    #     columns_to_drop,
    #     message="Removing empty columns"
    # )

    # remove columns with minimal data science utility (free-text, all null values, etc)
    df, phenotype = _drop_columns_from_df_and_data_dict(
        df,
        phenotype,
        # the following columns contain free-text
        columns_to_drop=[
            'state_province',
            'other_edu_level', 'others_household_specify',
            'diagnosis_alz_dementia_mci_ds_cdr', 'diagnosis_alz_dementia_mci_ca_rudas_score',
            'diagnosis_alz_dementia_mci_ca_mmse_score', 'diagnosis_alz_dementia_mci_ca_moca_score',
            'diagnosis_alz_dementia_mci_ca_adas_cog_score', 'diagnosis_alz_dementia_mci_ca_other',
            'diagnosis_alz_dementia_mci_ca_other_score',
            'diagnosis_parkinsons_ma_uprds', 'diagnosis_parkinsons_ma_updrs_part_i_score',
            'diagnosis_parkinsons_ma_updrs_part_ii_score', 'diagnosis_parkinsons_ma_updrs_part_iii_score',
            'diagnosis_parkinsons_ma_updrs_part_iv_score', 'diagnosis_parkinsons_non_motor_symptoms_yes',
            'traumatic_event',
            # following columns have all null values
            'is_regular_smoker',
        ],
        message="Removing columns with free-text"
    )

    return df, phenotype

def load_phenotype_data(bids_path: Path) -> t.Tuple[pd.DataFrame, t.Dict[str, t.Any]]:
    # load in the participants.tsv which has all phenotype data merged
    df = pd.read_csv(bids_path.joinpath("participants.tsv"), sep="\t")

    # remove subject
    idx = df["record_id"].isin(PARTICIPANT_ID_TO_REMOVE)
    if idx.sum() > 0:
        _LOGGER.info(
            f"Removing {idx.sum()} records from phenotype due to hard-coded subject removal."
        )
        df = df.loc[~idx]

    with open(bids_path.joinpath("participants.json"), "r") as f:
        participants = json.load(f)
    phenotype = {"participant_id": participants["record_id"]}

    # fix some data values and remove columns we do not want to publish at this time
    df, phenotype = clean_phenotype_data(df, phenotype)

    return df, phenotype

def is_audio_sensitive(task_name: str) -> bool:
   return task_name.lower().startswith("free-speech") \
        or task_name.lower().startswith("audio-check") \
        or task_name.lower().startswith("open-response-questions")