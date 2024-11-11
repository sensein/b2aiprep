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

import logging
import traceback
from pathlib import Path
from typing import List

import pydra
import torch
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.features_extraction.api import extract_features_from_audios
from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios
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

from b2aiprep.prepare.bids import batch_elements, get_audio_paths
from b2aiprep.prepare.utils import retry

SUBJECT_ID = "sub"
SESSION_ID = "ses"
AUDIO_ID = "audio"
RESAMPLE_RATE = 16000

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@pydra.mark.task
def wav_to_features(
    wav_paths: List[Path],
    transcription_model_size: str,
    with_sensitive: bool,
    device: DeviceType = DeviceType.CPU,
):
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
    for wav_path in tqdm(wav_paths, total=len(wav_paths), desc="Extracting features"):
        wav_path = Path(wav_path)

        logging.disable(logging.ERROR)
        # Load audio
        audio_orig = Audio.from_filepath(wav_path)

        # Downmix to mono
        audio_orig = downmix_audios_to_mono([audio_orig])[0]

        # Resample both audios to 16kHz
        audio_16k = resample_audios([audio_orig], 16000)[0]

        features = extract_features_from_audios(
            audios=[audio_16k],
            opensmile=True,
            parselmouth=True,
            torchaudio=True,
            torchaudio_squim=True,
        )
        features = features.pop()
        features["sample_rate"] = audio_16k.sampling_rate
        if with_sensitive:
            features["audio_path"] = wav_path
            try:
                model = SpeechBrainModel(
                    path_or_uri="speechbrain/spkrec-ecapa-voxceleb", revision="main"
                )
                features["speaker_embedding"] = extract_speaker_embeddings_from_audios(
                    [audio_16k], model, device
                )[0]
            except Exception as e:
                features["speaker_embedding"] = None
                _logger.error(
                    f"Speaker embeddings: An error occurred with extracting speaker embeddings. {e}"
                )
            try:
                speech_to_text_model = HFModel(
                    path_or_uri=f"openai/whisper-{transcription_model_size}", revision="main"
                )
                language = Language.model_validate({"language_code": "en"})
                transcription = retry(transcribe_audios)(
                    [audio_16k], model=speech_to_text_model, device=device, language=language
                )
                features["transcription"] = transcription[0]
            except Exception as e:
                features["transcription"] = None
                logging.disable(logging.NOTSET)
                _logger.error(f"Transcription: An error occurred with transcription: {e}")
                _logger.error(traceback.print_exc())
        logging.disable(logging.NOTSET)
        _logger.setLevel(logging.INFO)

        # Define the save directory for features
        audio_dir = wav_path.parent
        features_dir = audio_dir.parent / "audio"
        features_dir.mkdir(exist_ok=True)
        save_path = features_dir / f"{wav_path.stem}_features.pt"
        torch.save(features, save_path)
        all_features.append(features)

    return all_features


def extract_features_workflow(
    bids_dir_path: Path,
    transcription_model_size: str = "tiny",
    n_cores: int = 8,
    with_sensitive: bool = True,
    plugin="cf",
):
    """Run a Pydra workflow to extract audio features from BIDS-like directory.

    This function initializes a Pydra workflow that processes a BIDS-like
    directory structure to extract features from .wav audio files. It retrieves
    the paths to the audio files and applies the wav_to_features to each.

    Args:
      bids_dir_path:
        The root directory of the BIDS dataset.
      remove:
        Whether to remove temporary files created during processing. Default is True.

    Returns:
      pydra.Workflow:
        The Pydra workflow object with the extracted features and audio paths as outputs.
    """
    # Get paths to every audio file.
    audio_paths = get_audio_paths(bids_dir_path=bids_dir_path)
    if n_cores > 1:
        audio_paths = batch_elements(audio_paths, n_cores)

    # Initialize the Pydra workflow.
    ef_wf = pydra.Workflow(
        name="ef_wf", input_spec=["audio_paths"], audio_paths=audio_paths, cache_dir=None
    )

    # Run wav_to_features for each audio file
    ef_wf.add(
        wav_to_features(
            name="features",
            wav_paths=ef_wf.lzin.audio_paths,
            transcription_model_size=transcription_model_size,
            with_sensitive=with_sensitive,
        ).split("wav_paths", wav_paths=ef_wf.lzin.audio_paths)
    )

    ef_wf.set_output({"features": ef_wf.features.lzout.out})

    plugin_args = {"n_procs": n_cores} if n_cores > 1 else {}
    plugin = plugin if n_cores > 1 else "serial"
    with pydra.Submitter(plugin=plugin, **plugin_args) as run:
        run(ef_wf)

    return ef_wf


def validate_bids_data(
    bids_dir_path: Path,
    fix: bool = True,
    transcription_model_size: str = "medium",
) -> None:
    """Scans BIDS audio data and verifies that all expected features are present."""
    _logger.info("Scanning for features in BIDS directory.")
    # TODO: add a check to see if the audio feature extraction is complete
    # before proceeding to the next step
    # can verify features are generated for each audio_dir by looking for .pt files
    # in audio_dir.parent / "audio"
    audio_paths = get_audio_paths(bids_dir_path)
    audio_to_reprocess = []
    for audio_path in audio_paths:
        audio_path = Path(audio_path)
        audio_dir = audio_path.parent
        features_dir = audio_dir.parent / "audio"
        if features_dir.joinpath(f"{audio_path.stem}_features.pt").exists() is False:
            audio_to_reprocess.append(audio_path)

    if len(audio_to_reprocess) > 0:
        _logger.info(
            f"Missing features for {len(audio_to_reprocess)} / {len(audio_paths)} audio files"
        )
    else:
        _logger.info("All audio files have been processed and all feature files are present.")
        return

    if not fix:
        return

    for audio_path in tqdm(
        audio_to_reprocess, total=len(audio_to_reprocess), desc="Reprocessing audio files"
    ):
        audio_dir = audio_path.parent
        features_dir = audio_dir.parent / "audio"
        features_dir.mkdir(exist_ok=True)
        wav_to_features([audio_path], transcription_model_size=transcription_model_size)

    _logger.info("Process completed.")
