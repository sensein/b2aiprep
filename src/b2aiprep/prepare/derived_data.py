import json
import logging
import typing as t
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from b2aiprep.prepare.constants import AUDIO_FILESTEMS_TO_REMOVE, PARTICIPANT_ID_TO_REMOVE
from b2aiprep.prepare.prepare import reduce_id_length, reduce_length_of_id

_LOGGER = logging.getLogger(__name__)


def spectrogram_generator(
    audio_paths,
) -> t.Generator[t.Dict[str, t.Any], None, None]:
    """Load audio features from individual files and yield dictionaries amenable to HuggingFace's Dataset from_generator."""
    audio_paths = sorted(
        audio_paths,
        # sort first by subject, then by task
        key=lambda x: (x.stem.split("_")[0], x.stem.split("_")[2]),
    )

    for wav_path in tqdm(audio_paths, total=len(audio_paths), desc="Extracting features"):
        output = {}
        pt_file = wav_path.parent / f"{wav_path.stem}_features.pt"
        features = torch.load(pt_file, weights_only=False)

        output["participant_id"] = wav_path.stem.split("_")[0][4:]  # skip "sub-" prefix
        output["session_id"] = wav_path.stem.split("_")[1][4:]  # skip "ses-" prefix
        output["task_name"] = wav_path.stem.split("_")[2][5:]  # skip "task-" prefix

        spectrogram = features["torchaudio"]["spectrogram"]
        spectrogram = 10.0 * torch.log10(torch.maximum(spectrogram, torch.tensor(1e-10)))
        spectrogram = torch.maximum(spectrogram, spectrogram.max() - 80)
        spectrogram = spectrogram.numpy().astype(np.float32)
        # skip every other column
        spectrogram = spectrogram[:, ::2]
        output["spectrogram"] = spectrogram

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
        metadata_filepath = wav_path.parent.joinpath(wav_path.stem + ".json")
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
        output["spectrogram"] = features["torchaudio"]["spectrogram"].numpy().astype(np.float32)
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
        key=lambda x: (x.stem.split("_")[0], x.stem.split("_")[2]),
    )
    if feature_name not in ("spectrogram", "mfcc"):
        raise ValueError(f"Feature name {feature_name} not supported.")

    for wav_path in tqdm(audio_paths, total=len(audio_paths), desc="Extracting features"):
        output = {}
        pt_file = wav_path.parent / f"{wav_path.stem}_features.pt"
        features = torch.load(pt_file, weights_only=False)

        output["participant_id"] = wav_path.stem.split("_")[0][4:]  # skip "sub-" prefix
        output["session_id"] = wav_path.stem.split("_")[1][4:]  # skip "ses-" prefix
        output["task_name"] = wav_path.stem.split("_")[2][5:]  # skip "task-" prefix

        data = features["torchaudio"][feature_name]
        if feature_name == "spectrogram":
            data = 10.0 * torch.log10(torch.maximum(data, torch.tensor(1e-10)))
            data = torch.maximum(data, data.max() - 80)
            data = data.numpy().astype(np.float32)
        else:
            data = data.numpy()

        data = data[:, ::2]
        output[feature_name] = data

        yield output


def add_record_id_to_phenotype(phenotype: dict) -> dict:
    if 'record_id' in list(phenotype.keys()):
        return phenotype

    phenotype_updated = {
        'record_id': {
            "description": "Unique identifier for each participant."
        }
    }
    phenotype_updated.update(phenotype)
    return phenotype_updated
    
def _rename_record_id_to_participant_id(df: pd.DataFrame, phenotype: dict) -> dict:
    phenotype_updated = {}
    for c in df.columns:
        if c == 'record_id':
            phenotype['participant_id'] = phenotype[c]
        else:
            phenotype_updated[c] = phenotype[c]

    df = df.rename(columns={"record_id": "participant_id"})

    return df, phenotype_updated

def _add_sex_at_birth_column(df: pd.DataFrame, phenotype: dict) -> t.Tuple[pd.DataFrame, dict]:
    df["sex_at_birth"] = None
    for sex_at_birth in ["Male", "Female"]:
        # case-sensitive match
        idx = (
            df["gender_identity"].str.contains(sex_at_birth)
            & df["specify_gender_identity"].notnull()
        )
        df.loc[idx, "sex_at_birth"] = sex_at_birth

    # re-order columns
    phenotype_reordered = {}
    columns = []
    for c in df.columns:
        if c == "specify_gender_identity":
            continue

        if c == "gender_identity":
            columns.append("sex_at_birth")
            phenotype_reordered["sex_at_birth"] = {
                "description": "The sex at birth for the individual."
            }
        elif c == "sex_at_birth":
            continue
        else:
            columns.append(c)
            phenotype_reordered[c] = phenotype[c]

    df = df[columns]
    return df, phenotype_reordered

def load_phenotype_data(base_path: Path, phenotype_name: str) -> t.Tuple[pd.DataFrame, t.Dict[str, t.Any]]:
    # load in the participants.tsv which has all phenotype data merged
    if phenotype_name.endswith('.tsv'):
        phenotype_name = phenotype_name[:-4]
    elif phenotype_name.endswith('.json'):
        phenotype_name = phenotype_name[:-5]

    df = pd.read_csv(base_path.joinpath(f"{phenotype_name}.tsv"), sep="\t")
    with open(base_path.joinpath(f"{phenotype_name}.json"), "r") as f:
        phenotype = json.load(f)
    
    if df.shape[1] > 0 and df.columns[0] == 'record_id' and 'record_id' not in list(phenotype.keys()):
        phenotype = add_record_id_to_phenotype(phenotype)

    if len(phenotype) != df.shape[1]:
        _LOGGER.warning(
            f"Phenotype {phenotype_name} has {len(phenotype)} columns, but the data has {df.shape[1]} columns."
        )

    # skip the first-level of the hierarchy if present, which is the name of the phenotype
    if len(phenotype) == 1:
        only_key = next(iter(phenotype))
        if 'data_elements' in phenotype[only_key]:
            phenotype = phenotype[only_key]['data_elements']

    # remove hard-coded individuals
    idx = df["record_id"].isin(PARTICIPANT_ID_TO_REMOVE)
    if idx.sum() > 0:
        _LOGGER.info(
            f"Removing {idx.sum()} records from {phenotype_name}."
        )
        df = df.loc[~idx]

    # fix some data values and remove columns we do not want to publish at this time
    df, phenotype = clean_phenotype_data(df, phenotype)

    # reduce record_id to 8 characters
    df = reduce_length_of_id(df, id_name='record_id')
    df = reduce_length_of_id(df, id_name='session_id')

    # create columns missing in the original data
    if ("gender_identity" in df.columns) and ("specify_gender_identity" in df.columns):
        df, phenotype = _add_sex_at_birth_column(df, phenotype)

    df, phenotype = _rename_record_id_to_participant_id(df, phenotype)

    return df, phenotype
