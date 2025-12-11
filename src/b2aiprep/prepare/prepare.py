"""
Organizes data, extracts features, and bundles everything together in an
easily distributable format for the Bridge2AI Summer School.

This script performs the following steps:
1. Organizes RedCap data and audio files into a BIDS-like directory structure.
2. Extracts audio features from the organized data using a Pydra workflow.
3. Bundles the processed data into a .tar file with gzip compression.

Feature extraction is parallelized using Pydra.

Usage:
    b2aiprep-cli prepsummerdata \
       [path to RedCap CSV] \
       [path to Wasabi export directory] \
       [desired path to BIDS output] \
       [desired output path for .tar file]

    python3 b2aiprep/src/b2aiprep/bids_like_data.py \
        --redcap_csv_path [path to RedCap CSV] \
        --audio_dir_path  [path to Wasabi export directory] \
        --bids_dir_path [desired path to BIDS output] \
        --tar_file_path [desired output path for .tar file]

Functions:
    - wav_to_features: Extracts features from a single audio file.
    - get_audio_paths: Retrieves all .wav audio file paths from a BIDS-like
        directory structure.
    - extract_features_workflow: Runs a Pydra workflow to extract audio
        features from BIDS-like directory.
    - bundle_data: Saves data bundle as a tar file with gzip compression.
    - prepare_bids_like_data: Organizes and processes Bridge2AI data for
        distribution.
    - parse_arguments: Parses command line arguments for processing audio data.

"""

from collections import Counter
import json
import logging
import os
import traceback
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import time

import numpy as np
import pandas as pd
import pydra
#from pydra.compose import python, workflow
import torch
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.features_extraction.api import extract_features_from_audios
from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios
from senselab.audio.tasks.speaker_diarization import diarize_audios
from senselab.audio.tasks.speaker_embeddings import (
    extract_speaker_embeddings_from_audios,
)
from senselab.audio.tasks.speech_to_text import transcribe_audios
from senselab.utils.data_structures import (
    DeviceType,
    HFModel,
    Language,
    SpeechBrainModel,
)
from tqdm import tqdm

from b2aiprep.prepare.constants import (
    FEATURE_EXTRACTION_SPEECH_RATE,
    FEATURE_EXTRACTION_DURATION,
    FEATURE_EXTRACTION_PITCH_AND_INTENSITY,
    FEATURE_EXTRACTION_HARMONIC_DESCRIPTORS,
    FEATURE_EXTRACTION_FORMANTS,
    FEATURE_EXTRACTION_SPECTRAL_MOMENTS,
    FEATURE_EXTRACTION_JITTER, FEATURE_EXTRACTION_SHIMMER,
)

from b2aiprep.prepare.bids import get_audio_paths
from b2aiprep.prepare.constants import SPEECH_TASKS
from b2aiprep.prepare.utils import retry

SUBJECT_ID = "sub"
SESSION_ID = "ses"
AUDIO_ID = "audio"
RESAMPLE_RATE = 16000
SPECTROGRAM_SHAPE = 201
# Parcelmouth feature groupings

_logger = logging.getLogger(__name__)


#@python.define
def extract_single(
    wav_path: str | os.PathLike,
    transcription_model_size: str,
    with_sensitive: bool,
    overwrite: bool = False,
    device = None,
    update: bool = False,
):


    _logger.info(f"PyTorch version: {torch.__version__}")
    _logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        _logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
        _logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        _logger.info("No GPU detected.")
    wav_path = Path(wav_path)
    # Define the save directory for features
    audio_dir = wav_path.parent
    features_dir = audio_dir.parent / "audio"
    features_dir.mkdir(exist_ok=True)
    save_to = features_dir / f"{wav_path.stem}_features.pt"

    _logger.info(f"Saving to {save_to} and does it exist {save_to.exists()}")
    _logger.info(f"Working on audio {wav_path}")

    preloaded_pt = {}
    if save_to.exists() and not overwrite:
        if not update:
            _logger.info(f"{save_to} already exists. Skipping.")
            return save_to
        else:
            preloaded_pt = torch.load(save_to, map_location='cpu', weights_only=False)


    start_time=time.perf_counter()
    # Load audio
    try:
        audio_orig = Audio(filepath=wav_path)
        _logger.info(f"Audio duration: {audio_orig.waveform.shape[-1]/audio_orig.sampling_rate}")
        # Downmix to mono
        audio_orig = downmix_audios_to_mono([audio_orig])[0]

        # Resample both audios to 16kHz
        audio_16k = resample_audios([audio_orig], RESAMPLE_RATE)[0]
    except Exception as e:
        _logger.error(f"Transcription: An error occurred with loading and downsampling {wav_path}: {e}")
        _logger.error(f"{traceback.print_exc()}")
        return None


    end_time = time.perf_counter()
    execution_time = end_time - start_time
    _logger.info(f"Execution time audio_loading: {execution_time} seconds")
    win_length = 25
    hop_length = 10

    is_speech_task = any([v.replace(" ", "-") in wav_path.name for v in SPEECH_TASKS])

    opensmile = True
    torchaudio_squim = True
    sparc = True
    ppgs = True
    parsel_mouth_config = {
        "time_step": hop_length / 1000,
        "window_length": win_length / 1000,
    }
    torch_config = {
        "freq_low": 80,
        "freq_high": 500,
        "n_fft": (win_length * audio_16k.sampling_rate) // 1000,
        "n_mels": 60,
        "n_mfcc": 60,
        "win_length": (win_length * audio_16k.sampling_rate) // 1000,
        "hop_length": (hop_length * audio_16k.sampling_rate) // 1000,
    }
    # Extract features
    if update and not overwrite:
        if "opensmile" in preloaded_pt:
            opensmile = False
        if "praat_parselmouth" in preloaded_pt:
            praat_features = list(preloaded_pt["praat_parselmouth"].keys())
            parsel_mouth_config = {
                "time_step": hop_length / 1000,
                "window_length": win_length / 1000,
                "speech_rate": False,
                "intensity_descriptors": False,
                "harmonicity_descriptors": False,
                "formants": False,
                "spectral_moments": False,
                "pitch": False,
                "slope_tilt": False,
                "cpp_descriptors": False,
                "duration": False,
                "jitter": False,
                "shimmer": False,
            }
            if not all(features in praat_features for features in FEATURE_EXTRACTION_SPEECH_RATE):
                parsel_mouth_config["speech_rate"] = True

            if not all(features in praat_features for features in FEATURE_EXTRACTION_DURATION):
                parsel_mouth_config["duration"] = True

            if not all(features in praat_features for features in FEATURE_EXTRACTION_PITCH_AND_INTENSITY):
                parsel_mouth_config["pitch"] = True
                parsel_mouth_config["intensity_descriptors"] = True

            if not all(features in praat_features for features in FEATURE_EXTRACTION_HARMONIC_DESCRIPTORS):
                parsel_mouth_config["cpp_descriptors"] = True
                parsel_mouth_config["slope_tilt"] = True
                parsel_mouth_config["harmonicity_descriptors"] = True

            if not all(features in praat_features for features in FEATURE_EXTRACTION_FORMANTS):
                parsel_mouth_config["formants"] = True

            if not all(features in praat_features for features in FEATURE_EXTRACTION_SPECTRAL_MOMENTS):
                parsel_mouth_config["spectral_moments"] = True

            if not all(features in praat_features for features in FEATURE_EXTRACTION_JITTER):
                parsel_mouth_config["jitter"] = True

            if not all(features in praat_features for features in FEATURE_EXTRACTION_SHIMMER):
                parsel_mouth_config["shimmer"] = True
        if "torchaudio" in preloaded_pt:
            spectrogram = preloaded_pt["torchaudio"]["spectrogram"]
            if spectrogram.shape[0] == SPECTROGRAM_SHAPE:
                torch_config = False
        if "torchaudio_squim" in preloaded_pt:
            torchaudio_squim = False
        if "sparc" in preloaded_pt:
            sparc = False
        if "ppgs" in preloaded_pt:
            ppgs = False

    #logging.disable(logging.ERROR)
    start_time = time.perf_counter()
    features = extract_features_from_audios(
        audios=[audio_16k],
        opensmile=opensmile,
        parselmouth=parsel_mouth_config,
        torchaudio=torch_config,
        torchaudio_squim=torchaudio_squim,
        sparc=sparc,
        ppgs=ppgs,
        device=device,
    ).pop()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    _logger.info(f"Execution time extract_features: {execution_time} seconds")


    if update and not overwrite:
        # if squim was not in previous release
        if "torchaudio_squim" not in preloaded_pt and "torchaudio_squim" in features :
            preloaded_pt["torchaudio_squim"] = features["torchaudio_squim"]
        if "opensmile" not in preloaded_pt and "opensmile" in features:
            preloaded_pt["opensmile"] = features["opensmile"]
        if "torchaudio" not in preloaded_pt and "torchaudio" in features:
            preloaded_pt["torchaudio"] = features["torchaudio"]
        if "praat_parselmouth" not in preloaded_pt and "praat_parselmouth" in features:
            preloaded_pt["praat_parselmouth"] = features["praat_parselmouth"]
        if "sparc" not in preloaded_pt and "sparc" in features:
            preloaded_pt["sparc"] = features["sparc"]
        if "ppgs" not in preloaded_pt and "ppgs" in features:
            preloaded_pt["ppgs"] = features["ppgs"]

        # case where the first generation was using n_fft = 1024 and we need to replace
        if preloaded_pt["torchaudio"]["spectrogram"].shape[0] != SPECTROGRAM_SHAPE:
            preloaded_pt["torchaudio"] = features["torchaudio"]

        # combine to have all features
        if "praat_parselmouth" in features and "praat_parselmouth" in preloaded_pt:
            preloaded_pt["praat_parselmouth"] = {
                **preloaded_pt["praat_parselmouth"], **features["praat_parselmouth"]}
        features = preloaded_pt

    if not update:
        features["parselmouth_config"] = parsel_mouth_config
        features["torch_config"] = torch_config
        features["is_speech_task"] = is_speech_task
        features["sample_rate"] = audio_16k.sampling_rate
        features["duration"] = len(audio_16k.waveform) / audio_16k.sampling_rate
        features["sensitive_features"] = None

    try:
        start_time = time.perf_counter()
        #diarization_model = HFModel(
        #    path_or_uri=f"nvidia/diar_streaming_sortformer_4spk-v2", revision="main"
        #)

        diarization = diarize_audios(
            [audio_16k], device=device
        )
        _logger.info(f"diarization: {diarization}")
        features["diarization"] = diarization[0]

        end_time = time.perf_counter()
        execution_time = end_time - start_time
        _logger.info(f"Execution time diarization: {execution_time} seconds")
    except Exception as e:
        _logger.error(f"Diarization: An error occurred with diarization of {wav_path}: {e}")
        _logger.error(traceback.print_exc())
        features["diarization"] = None

    #logging.disable(logging.ERROR)
    if with_sensitive:
        features["audio_path"] = wav_path
        try:
            start_time = time.perf_counter()
            model = SpeechBrainModel(
                path_or_uri="speechbrain/spkrec-ecapa-voxceleb", revision="main"
            )
            features["speaker_embedding"] = extract_speaker_embeddings_from_audios(
                [audio_16k], model, device
            )[0]
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            _logger.info(f"Execution time speaker_embedding: {execution_time} seconds")
        except Exception as e:
            features["speaker_embedding"] = None
            _logger.error(
                f"Speaker embeddings: An error occurred with extracting speaker embeddings of {wav_path}. {e}"
            )
            _logger.error(traceback.print_exc())
        features["transcription"] = None
        try:
            start_time = time.perf_counter()
            speech_to_text_model = HFModel(
                path_or_uri=f"openai/whisper-{transcription_model_size}", revision="main"
            )
            language = Language.model_validate({"language_code": "en"})
            transcription = retry(transcribe_audios)(
                [audio_16k], model=speech_to_text_model, device=device, language=language,return_timestamps=None
            )
            features["transcription"] = transcription[0]

            end_time = time.perf_counter()
            execution_time = end_time - start_time
            _logger.info(f"Execution time transcription: {execution_time} seconds")
        except Exception as e:
            logging.disable(logging.NOTSET)
            _logger.error(f"Transcription: An error occurred with transcription of {wav_path}: {e}")
            _logger.error(traceback.print_exc())

        features["sensitive_features"] = ["audio_path", "speaker_embedding", "transcription"]
    _logger.setLevel(logging.INFO)

    torch.save(features, save_to)
    return save_to


def wav_to_features(
    wav_paths: List[str | os.PathLike],
    transcription_model_size: str,
    with_sensitive: bool,
    overwrite: bool = False,
    device: DeviceType = DeviceType.CPU,
) -> List[str | os.PathLike]:
    """Extract features from a list of audio files.

    Extracts various audio features from .wav files
    using the Audio class and feature extraction functions.

    Args:
      wav_path:
        The file path to the .wav audio file.

    Returns:
      A dictionary mapping feature names to their extracted values.
    """
    all_features = []
    if len(wav_paths) > 1:
        _logger.info(f"Number of audio files: {len(wav_paths)}")
        for wav_path in tqdm(wav_paths, total=len(wav_paths), desc="Extracting features"):
            save_path = extract_single(
                wav_path,
                transcription_model_size=transcription_model_size,
                with_sensitive=with_sensitive,
                device=device,
                overwrite=overwrite,
            )
            all_features.append(save_path)
    else:
        save_path = extract_single(
            wav_paths[0],
            transcription_model_size=transcription_model_size,
            with_sensitive=with_sensitive,
            device=device,
            overwrite=overwrite,
        )
        all_features.append(save_path)
    return all_features


def generate_features_wrapper(
    bids_path,
    transcription_model_size,
    n_cores,
    with_sensitive,
    overwrite,
    cache,
    address,
    percentile,
    subject_id,
    subject_file,
    update=False,
    is_sequential=False

):
    if is_sequential:
        extract_features_sequentially(
            bids_path,
            transcription_model_size=transcription_model_size,
            with_sensitive=with_sensitive,
            update=update,
        )
    else:
        extract_features_workflow(
            bids_path,
            transcription_model_size=transcription_model_size,
            n_cores=n_cores,
            with_sensitive=with_sensitive,
            overwrite=overwrite,
            cache_dir=cache,
            plugin="dask" if address is not None else "cf",
            address=address,
            percentile=percentile,
            subject_id=subject_id,
            subject_file=subject_file,
            update=update,
        )


def extract_features_sequentially(
    bids_dir_path: Path,
    transcription_model_size: str = "tiny",
    with_sensitive: bool = True,
    update: bool = False
):
    audio_paths = get_audio_paths(bids_dir_path=bids_dir_path)
    audio_paths = sorted(audio_paths, key=lambda wave_file: wave_file["size"])

    for audio_file in audio_paths:
        extract_single(
            wav_path=audio_file["path"],
            transcription_model_size=transcription_model_size,
            with_sensitive=with_sensitive,
            update=update)


def _select_worker(plugin: str) -> str:
    return "debug" if plugin in ("serial", "debug") else plugin

def extract_features_workflow(
    bids_dir_path: Path,
    transcription_model_size: str = "tiny",
    with_sensitive: bool = True,
    overwrite: bool = False,
    n_cores: int = 8,
    plugin: str = "cf",
    address: str = None,
    cache_dir: str | os.PathLike = None,
    percentile: int = 100,
    subject_id: str = None,
    subject_file: str = None,
    update: bool = False
):
    """Run a Pydra workflow to extract audio features from BIDS-like directory.

    This function initializes a Pydra workflow that processes a BIDS-like
    directory structure to extract features from .wav audio files. It retrieves
    the paths to the audio files and applies the wav_to_features to each.

    Args:
      bids_dir_path:
        The root directory of the BIDS dataset.
      transcription_model_size:
        The size of the Whisper model to use for transcription.
      n_cores:
        The number of cores to use for parallel processing.
      with_sensitive:
        Whether to extract sensitive features such as speaker embeddings and transcriptions.
      plugin:
        The Pydra plugin to use for parallel processing.
      cache_dir:
        The directory to use for caching intermediate results.

    Returns:
      pydra.Task:
        The Pydra Task object with the extracted feature paths.
    """
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set
        pass

    cache_dir = Path(cache_dir) if cache_dir is not None else None
    if n_cores > 1:
        plugin_args: dict = {"n_procs": n_cores} if plugin == "cf" else {}
    else:
        plugin = "serial"
        plugin_args = {}
    # Get paths to every audio file.
    audio_paths = get_audio_paths(bids_dir_path=bids_dir_path)
    df = pd.DataFrame(audio_paths)
    if "path" not in df.columns:
        _logger.warning("No audio files found in the BIDS directory.")
        return
    if "size" in df.columns:
        df = df[df["size"] <= np.percentile(df["size"].values, percentile)]
    if subject_id is not None:
        _logger.info(f"Processing subject_id {subject_id}")
        
        df = df[df["subject"] == subject_id]
        if len(df)==0:
            raise ValueError(f"Subject ID {subject_id} not found")

    elif subject_file is not None:
        subject_file = Path(subject_file)
        if not subject_file.exists():
            raise ValueError(f"Subject file provided {subject_file} does not exist")
        if subject_file.suffix not in ['.txt']:
            raise ValueError(f"Subject file provided {subject_file} must be txt file")
        with subject_file.open() as f:
            subject_ids = [subj_id.rstrip() for subj_id in f.readlines()]

        results = []
        for subject_id in subject_ids:
            subject_cache_dir = cache_dir / subject_id if cache_dir is not None else None
            result = extract_features_workflow(
                bids_dir_path=bids_dir_path,
                transcription_model_size=transcription_model_size,
                with_sensitive=with_sensitive,
                overwrite=overwrite,
                n_cores=n_cores,
                plugin=plugin,
                address=address,
                cache_dir=subject_cache_dir,
                percentile=percentile,
                subject_id=subject_id,
                subject_file=None,
                update=update
            )
            results.append(result)
        return results
            
    # randomize to distribute sizes
    df = df.sample(frac=1).reset_index(drop=True)
    audio_paths = df.path.values.tolist()
    _logger.info(f"Running audio paths in order: {audio_paths}")

    #OLD PYDRA
    _logger.info(f"Number of audio files: {len(audio_paths)}")
    # Run the task
    extract_task = pydra.mark.task(extract_single)(
        transcription_model_size=transcription_model_size,
        with_sensitive=with_sensitive,
        overwrite=overwrite,
        cache_dir=cache_dir,
        update=update
    )
    extract_task.split("wav_path", wav_path=audio_paths)
    if plugin == "dask":
        plugin_args = {"address": address}
    with pydra.Submitter(plugin=plugin, **plugin_args) as run:
        run(extract_task)
    return extract_task
    

def validate_bids_audio_features(bids_dir_path: Path, report_path: Path = None):
    """Validate the audio features of a BIDS dataset.

    This function checks that all audio files have corresponding feature
    files and transcriptions.

    Args:
        bids_dir_path: The root directory of the BIDS dataset.
        report_path: where to save a file reporting the validation results
    """
    audio_paths = [x["path"] for x in get_audio_paths(bids_dir_path)]
    missing_features = {}
    missing_features_files = []
    missing_transcriptions = []
    missing_speaker_embeddings = []
    missing_diarizations = []
    unrecognized_features = Counter()
    multiple_feature_files = []
    for audio_path in audio_paths:
        audio_path = Path(audio_path)
        audio_dir = audio_path.parent
        features_dir = audio_dir.parent / "audio"
        feature_files = [file for file in features_dir.glob(f"{audio_path.stem}_*.pt")]
        audio_path = str(audio_path)
        if len(feature_files) == 0:
            missing_features_files.append(audio_path)
        elif len(feature_files) == 1:
            features = torch.load(feature_files[0],weights_only=False,map_location=torch.device('cpu'))
            if features["transcription"] is None:
                missing_transcriptions.append(audio_path)
            if features["diarization"] is None:
                missing_diarizations.append(audio_path)
            if features["speaker_embedding"] is None:
                missing_speaker_embeddings.append(audio_path)

            for feature_name in ["torchaudio_squim","opensmile","torchaudio","praat_parselmouth","sparc","ppgs"]:
                feature = features[feature_name]
                
                feature_list = missing_features.setdefault(feature_name, [])
                if feature is None:
                    feature_list.append(audio_path)
                elif isinstance(feature, dict):
                    for key, val in feature.items():
                        if torch.isnan(torch.tensor(val)).any().item():
                            feature_list.append(audio_path)
                            break
                elif isinstance(feature, torch.Tensor):
                    if torch.isnan(feature).any().item():
                        feature_list.append(audio_path)
                elif isinstance(feature, np.ndarray):
                    if np.isnan(feature).any():
                        feature_list.append(audio_path)
                elif isinstance(feature, float):
                    if np.isnan(feature):
                        feature_list.append(audio_path)
                else:
                    unrecognized_features[feature_name] += 1
        else:
            multiple_feature_files.append(audio_path)
    
    # report out the findings
    if len(missing_features_files) > 0:
        _logger.info(
            f"Missing all features for {len(missing_features_files)} / {len(audio_paths)} audio files"
        )
    if len(missing_diarizations) > 0:
        _logger.info(
            f"Missing diarizations for {len(missing_diarizations)} / {len(audio_paths)} audio files"
        )
    if len(missing_transcriptions) > 0:
        _logger.info(
            f"Missing transcriptions for {len(missing_transcriptions)} / {len(audio_paths)} audio files"
        )
    if len(missing_speaker_embeddings) > 0:
        _logger.info(
            f"Missing speaker_embeddings for {len(missing_speaker_embeddings)} / {len(audio_paths)} audio files"
        )
        
    for feature in missing_features:
        if len(missing_features[feature]) > 0:
            _logger.info(
                f"Missing {feature} for {len(missing_features[feature])} / {len(audio_paths)} audio files"
            )
    
    # unrecognized features
    for feature_name, count in unrecognized_features.items():
        _logger.info(
            f"Unrecognized feature type for {feature_name} in {count} audio files"
        )
    
    # multiple feature files
    if len(multiple_feature_files) > 0:
        _logger.info(
            f"Multiple feature files found for {len(multiple_feature_files)} / {len(audio_paths)} audio files"
        )

    if report_path:
        report_file = report_path #/ "missing_features.json"
        _logger.info(f"Placing missing features details info into {report_file}")
        with open(report_file, "w") as f:
            json.dump(
                {
                    "missing_feature_extractions": missing_features,
                    "missing_diarizations": missing_diarizations,
                    "missing_transcriptions": missing_transcriptions,
                    "missing_speaker_embeddings": missing_speaker_embeddings,
                    "missing_feature_files": missing_features_files,
                    "multiple_feature_files": multiple_feature_files,
                }, f, indent=4
            )

def load_audio_to_remove(publish_config_dir: Path) -> List[str]:
    """Load list of audio file stems to remove from JSON file."""
    audio_to_remove_path = publish_config_dir / "audio_to_remove.json"
    if not audio_to_remove_path.exists():
        raise FileNotFoundError(f"Audio to remove file {audio_to_remove_path} does not exist.")

    with open(audio_to_remove_path, 'r') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Audio to remove file {audio_to_remove_path} should contain a list of audio file stems.")
    
    return data

def is_audio_sensitive(filepath: Path, publish_config_dir: Path) -> bool:
    audio_to_remove = load_audio_to_remove(publish_config_dir)
    return filepath.stem in audio_to_remove

def reduce_id_length(x, length=8):
    """Reduce length of ID, removes hashes."""
    if pd.isnull(x):
        return x
    x = str(x).replace("-", "")
    return x[:length]
    

def reduce_length_of_id(df: pd.DataFrame, id_name: str) -> pd.DataFrame:
    """Reduce length of ID in the dataframe."""
    for c in df.columns:
        if c == id_name or c.endswith(id_name):
            df[c] = df[c].apply(reduce_id_length)
    
    return df

def load_remap_id_list(publish_config_dir: Path) -> Dict:
    audio_to_remap_path = publish_config_dir / "id_remapping.json"
    if not audio_to_remap_path.exists():
        raise FileNotFoundError(f"Audio to remove file {audio_to_remap_path} does not exist.")

    with open(audio_to_remap_path, 'r') as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Audio to remove file {audio_to_remap_path} should contain a dict of participant_ids to new ids.")

    return data

def get_value_from_metadata(metadata: dict, linkid: str, endswith: bool = False) -> str:
    for item in metadata['item']:
        if 'linkId' not in item:
            continue
        if item['linkId'] == linkid:
            return item['answer'][0]['valueString']
        if endswith and item['linkId'].endswith(linkid):
            return item['answer'][0]['valueString']
    return None

def update_metadata_record_and_session_id(metadata: dict, ids_to_remap: dict, participant_session_id_to_remap: dict):
    for item in metadata['item']:
        if 'linkId' not in item:
            continue

        if item['linkId'] == 'record_id':
            for old_id, new_id in ids_to_remap.items():
                if old_id == item['answer'][0]['valueString']:
                    item['answer'][0]['valueString'] = new_id
                    break
            record_id = item['answer'][0]['valueString']
            item['answer'][0]['valueString'] = reduce_id_length(record_id)
            # rename to participant_id
            item['linkId'] = 'participant_id'
        elif (item['linkId'] == 'session_id') or (item['linkId'].endswith('_session_id')):
            for old_id, new_id in participant_session_id_to_remap.items():
                if old_id == item['answer'][0]['valueString']:
                    item['answer'][0]['valueString'] = new_id
                    break
            item['answer'][0]['valueString'] = reduce_id_length(
                item['answer'][0]['valueString']
            )
