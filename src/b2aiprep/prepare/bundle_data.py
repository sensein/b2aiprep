import json
import logging
import typing as t
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

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
        device = 'cpu' # not checking for cuda because optimization would be minimal if any
        features = torch.load(pt_file, weights_only=False, map_location=torch.device(device))

        output["participant_id"] = wav_path.stem.split("_")[0][4:]  # skip "sub-" prefix
        output["session_id"] = wav_path.stem.split("_")[1][4:]  # skip "ses-" prefix
        output["task_name"] = wav_path.stem.split("_")[2][5:]  # skip "task-" prefix

        spectrogram = features["torchaudio"].get("spectrogram", None)
        
        if spectrogram is not None:
            spectrogram = torch.tensor(spectrogram)
            if not torch.isnan(spectrogram).all().item():
                spectrogram = 10.0 * torch.log10(torch.maximum(spectrogram, torch.tensor(1e-10)))
                spectrogram = torch.maximum(spectrogram, spectrogram.max() - 80)
                spectrogram = spectrogram.numpy().astype(np.float32)
                # skip every other column
                spectrogram = spectrogram[:, ::2]
                output["spectrogram"] = spectrogram
                output["n_frames"] = spectrogram.shape[-1]
            else:
                _LOGGER.warning(f"Spectrogram for {wav_path} found to be all NaNs. Skipping.")
                continue
        else:
            _LOGGER.warning(f"Spectrogram for {wav_path} not found. Likely sensitive file. Skipping.")
            continue
            

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
        device = 'cpu' # not checking for cuda because optimization would be minimal if any
        features = torch.load(pt_file, weights_only=False, map_location=torch.device(device))
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
    audio_paths: t.List[Path],
    feature_name: str,
    feature_class: t.Optional[str] = None,
) -> t.Generator[t.Dict[str, t.Any], None, None]:
    """Load audio features from individual files and yield dictionaries amenable to HuggingFace's Dataset from_generator."""
    audio_paths = sorted(
        audio_paths,
        # sort first by subject, then by task
        key=lambda x: (x.stem.split("_")[0], x.stem.split("_")[2]),
    )
    if feature_name not in ("spectrogram", "mfcc"):
        _LOGGER.warning(f"Feature name {feature_name} has not been tested. Proceeding anyway.")

    for wav_path in tqdm(audio_paths, total=len(audio_paths), desc="Extracting features"):
        output = {}
        pt_file = wav_path.parent / f"{wav_path.stem}_features.pt"
        device = 'cpu' # not checking for cuda because optimization would be minimal if any
        features = torch.load(pt_file, weights_only=False, map_location=torch.device(device))

        output["participant_id"] = wav_path.stem.split("_")[0][4:]  # skip "sub-" prefix
        output["session_id"] = wav_path.stem.split("_")[1][4:]  # skip "ses-" prefix
        output["task_name"] = wav_path.stem.split("_")[2][5:]  # skip "task-" prefix

        if feature_class:
            data = features.get(feature_class, {}).get(feature_name, None)
        else:
            data = features.get(feature_name, None)
        
        if data is None:
            _LOGGER.warning(f"Feature {feature_name} for {wav_path} not found in feature file likely due to sensitive. Skipping.")
            continue

        data = torch.tensor(data)
        if torch.isnan(data).all().item():
            _LOGGER.warning(f"Feature {feature_name} for {wav_path} is all NaNs in feature file. Skipping.")
            continue
    
        if feature_name == "spectrogram":
            data = 10.0 * torch.log10(torch.maximum(data, torch.tensor(1e-10)))
            data = torch.maximum(data, data.max() - 80)
            data = data.numpy().astype(np.float32)
        else:
            data = data.numpy()

        if feature_name in ("spectrogram", "mfcc", "mel_spectrogram"):
            data = data[:, ::2]
        output[feature_name] = data
        output["n_frames"] = data.shape[-1]
        yield output
