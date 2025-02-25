"""Commands available through the CLI."""

import csv
import json
import logging
import os
import shutil
import tarfile
from collections import OrderedDict
from functools import partial
from glob import glob
from importlib import resources
from pathlib import Path

import click
import pandas as pd
import pkg_resources
import pydra
import torch
from datasets import Dataset
from pyarrow.parquet import SortingColumn
from pydra.mark import annotate
from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.features_extraction.opensmile import (
    extract_opensmile_features_from_audios,
)
from senselab.audio.tasks.features_extraction.torchaudio import (
    extract_mel_filter_bank_from_audios,
    extract_mfcc_from_audios,
    extract_spectrogram_from_audios,
)
from senselab.audio.tasks.preprocessing.preprocessing import resample_audios
from senselab.audio.tasks.speaker_embeddings.api import (
    extract_speaker_embeddings_from_audios,
)
from senselab.audio.tasks.speaker_verification.speaker_verification import (
    verify_speaker,
)
from senselab.audio.tasks.speech_to_text.api import transcribe_audios
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.language import Language
from senselab.utils.data_structures.model import HFModel
from streamlit import config as _config
from streamlit.web.bootstrap import run
from tqdm import tqdm

from b2aiprep.prepare.bids import get_audio_paths, redcap_to_bids, validate_bids_folder
from b2aiprep.prepare.derived_data import (
    feature_extraction_generator,
    spectrogram_generator,
)
from b2aiprep.prepare.prepare import (
    clean_phenotype_data,
    extract_features_sequentially,
    extract_features_workflow,
    validate_bids_data,
)
from b2aiprep.prepare.reproschema_to_redcap import parse_audio, parse_survey

# from b2aiprep.synthetic_data import generate_synthetic_tabular_data

_LOGGER = logging.getLogger(__name__)


@click.command()
@click.argument("bids_dir", type=click.Path(exists=True))
def dashboard(bids_dir: str):
    """Launches a dashboard for exploring BIDS data.

    This function starts a Streamlit-based dashboard for visualizing and exploring
    BIDS-formatted data.

    Args:
        bids_dir (str): Path to the BIDS directory to be explored.

    Raises:
        ValueError: If the specified path does not exist or is not a directory.

    Returns:
        None: The function launches the dashboard and does not return a value.
    """
    bids_path = Path(bids_dir).resolve()
    if not bids_path.exists():
        raise ValueError(f"Input path {bids_path} does not exist.")

    if not bids_path.is_dir():
        raise ValueError(f"Input path {bids_path} is not a directory.")
    _config.set_option("server.headless", True)

    dashboard_path = pkg_resources.resource_filename("b2aiprep", "app/Dashboard.py")
    run(dashboard_path, args=[bids_path.as_posix()], flag_options=[], is_hello=False)


@click.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option(
    "--outdir",
    type=click.Path(),
    default=Path.cwd().joinpath("output").as_posix(),
    show_default=True,
)
@click.option("--audiodir", type=click.Path(), default=None, show_default=True)
def redcap2bids(
    filename,
    outdir,
    audiodir,
):
    """Organizes into BIDS-like without features.

    This function processes a REDCap data file and optionally links audio files
    to structure them in a format similar to the Brain Imaging Data Structure (BIDS).

    Args:
        filename (str): Path to the REDCap data file.
        outdir (str, optional): Directory where the converted BIDS-like data
                                will be saved. Defaults to "output" in the current working directory.
        audiodir (str, optional): Directory containing associated audio files.
                                  If not provided, only the REDCap data is processed.

    Raises:
        ValueError: If the specified output directory path exists but is not a directory.

    Returns:
        None: Saves the structured data in the specified output directory.
    """

    outdir = Path(outdir)
    if outdir.exists() and not outdir.is_dir():
        raise ValueError(f"Output path {outdir} is not a directory.")
    if audiodir is not None:
        audiodir = Path(audiodir)
    redcap_to_bids(
        filename,
        outdir=Path(outdir),
        audiodir=audiodir,
    )


@click.command()
@click.argument("bids_dir_path", type=click.Path())
@click.option("--redcap_csv_path", type=click.Path(exists=True), default=None, show_default=True)
@click.option("--audio_dir_path", type=click.Path(exists=True), default=None, show_default=True)
@click.option("-z", "--tar_file_path", type=click.Path(), default=None, show_default=True)
@click.option("-t", "--transcription_model_size", type=str, default="tiny", show_default=True)
@click.option("-n", "--n_cores", type=int, default=8, show_default=True)
@click.option("--with_sensitive/--no-with_sensitive", type=bool, default=True, show_default=True)
@click.option("--overwrite/--no-overwrite", type=bool, default=False, show_default=True)
@click.option("--validate/--no-validate", type=bool, default=False, show_default=True)
@click.option("-c", "--cache", type=click.Path(), default=None, show_default=True)
@click.option("-a", "--address", type=str, default=None, show_default=True)
@click.option("-p", "--percentile", type=int, default=100, show_default=True)
@click.option("-s", "--subject_id", type=str, default=None, show_default=True)
@click.option("--is_sequential", type=bool, default=False, show_default=True)
def prepare_bids(
    bids_dir_path,
    redcap_csv_path,
    audio_dir_path,
    tar_file_path,
    transcription_model_size,
    n_cores,
    with_sensitive,
    overwrite,
    validate,
    cache,
    address,
    percentile,
    subject_id,
    is_sequential,
):
    """Organizes into BIDS-like with features.

    This command follows a specific workflow:
    1. If both redcap_csv_path and audio_dir_path are provided:
       - Converts REDCap data to BIDS format
       - Organizes audio files into the BIDS structure
    2. Runs audio feature extraction on the BIDS directory:
       - Uses specified transcription model
       - Processes in parallel using n_cores
       - Can use Dask for distributed processing if address is provided
       - Can filter by percentile or specific subject
    3. Optionally validates the BIDS structure if --validate is set
    4. Creates a compressed tar file if tar_file_path is specified

    Args:
        bids_dir_path: Path to store BIDS-like data
        redcap_csv_path: Path to the REDCap CSV file. Required with audio_dir_path for initial BIDS creation
        audio_dir_path: Path to directory with audio files. Required with redcap_csv_path for initial BIDS creation
        tar_file_path: Path to store tar file. If provided, creates a compressed archive of the BIDS directory
        transcription_model_size: Size of transcription model ('tiny', 'small', 'medium', or 'large')
        n_cores: Number of cores to run feature extraction on. Ignored if using Dask
        with_sensitive: Whether to include sensitive data in the output
        overwrite: Whether to overwrite existing files during processing
        validate: Whether to validate the BIDS folder structure after processing
        cache: Path to cache directory. Defaults to 'b2aiprep_cache' in parent of bids_dir_path
        address: Dask scheduler address for distributed processing. If provided, uses Dask instead of concurrent.futures
        percentile: Percentile threshold for processing. Use to process subset of data
        subject_id: Specific subject ID to process. If provided, only processes this subject
        is_sequential: Specifies whether to extract audio features sequentially
    """
    if cache is None:
        cache = Path(bids_dir_path).parent / "b2aiprep_cache"
    os.makedirs(cache, exist_ok=True)
    if redcap_csv_path is not None and audio_dir_path is not None:
        _LOGGER.info("Organizing data into BIDS-like directory structure...")
        redcap_to_bids(Path(redcap_csv_path), Path(bids_dir_path), Path(audio_dir_path))
        _LOGGER.info("Data organization complete.")

    _LOGGER.info("Beginning audio feature extraction...")

    if is_sequential:
        extract_features_sequentially(
            Path(bids_dir_path),
            transcription_model_size=transcription_model_size,
            with_sensitive=with_sensitive,
        )
    else:
        extract_features_workflow(
            Path(bids_dir_path),
            transcription_model_size=transcription_model_size,
            n_cores=n_cores,
            with_sensitive=with_sensitive,
            overwrite=overwrite,
            cache_dir=cache,
            plugin="dask" if address is not None else "cf",
            address=address,
            percentile=percentile,
            subject_id=subject_id,
        )
    _LOGGER.info("Audio feature extraction complete.")

    if validate:
        # Below code checks to see if we have all the expected feature/transcript files.
        validate_bids_folder(Path(bids_dir_path))

    if tar_file_path is not None:
        _LOGGER.info("Saving .tar file with processed data...")
        with tarfile.open(tar_file_path, "w:gz") as tar:
            tar.add(bids_dir_path, arcname=os.path.basename(bids_dir_path))
        _LOGGER.info(f"Saved processed data .tar file at: {tar_file_path}")

    _LOGGER.info("Process completed.")


@click.command()
@click.argument("bids_path", type=click.Path(exists=True))
@click.argument("outdir", type=click.Path())
def create_derived_dataset(bids_path, outdir):
    """Creates derived dataset from BIDS data.

    The derived dataset output loads data from generated .pt files, which have
    the following keys:
     - pitch
     - mfcc
     - mel_spectrogram
     - mel_filter_bank (== mel_spectrogram)
     - spectrogram
     - transcription

    For output, we grab only the spectrograms. For a subset of tasks, we include
    the transcriptions in the metadata.

    derivatives/b2aiprep/
    ├── spectrogram.parquet
    ├── phenotype.tsv
    ├── phenotype.json
    ├── static_features.tsv
    ├── static_features.json
    """
    bids_path = Path(bids_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    audio_paths = get_audio_paths(bids_dir_path=bids_path)
    audio_paths = [x["path"] for x in audio_paths]

    audio_paths = sorted(
        audio_paths,
        # sort first by subject, then by task
        key=lambda x: (x.stem.split("_")[0], x.stem.split("_")[2]),
    )

    # remove known subjects without any audio
    SUBJECTS_TO_REMOVE = {"6e6216b7-1f2a-407f-9dcd-22052b704b82"}
    n = len(audio_paths)
    for participant_id in SUBJECTS_TO_REMOVE:
        audio_paths = [x for x in audio_paths if f"sub-{participant_id}" not in str(x)]

    if len(audio_paths) < n:
        _LOGGER.info(f"Removed {n - len(audio_paths)} records due to hard-coded subject removal.")

    _LOGGER.info("Loading spectrograms into a single HF dataset.")

    for feature_name in ["spectrogram", "mfcc"]:
        if feature_name == "mfcc":
            use_byte_stream_split = True
            audio_feature_generator = partial(
                feature_extraction_generator,
                audio_paths=audio_paths,
                feature_name=feature_name,
            )
        else:
            use_byte_stream_split = False
            audio_feature_generator = partial(
                spectrogram_generator,
                audio_paths=audio_paths,
            )

        # sort the dataset by identifier and task_name
        ds = Dataset.from_generator(audio_feature_generator, num_proc=1)
        ds.to_parquet(
            str(outdir.joinpath(f"{feature_name}.parquet")),
            version="2.6",
            compression="zstd",  # Better compression ratio than snappy, still good speed
            compression_level=3,
            # Enable dictionary encoding for strings
            use_dictionary=["participant_id", "session_id", "task_name"],
            write_statistics=True,
            # enable page index for better filtering
            data_page_size=1_048_576,  # 1MB pages
            write_page_index=True,
            use_byte_stream_split=use_byte_stream_split,
            # sort should help readers more efficiently read data
            # requires us to have manually sorted the data (as we have done above)
            sorting_columns=(
                SortingColumn(column_index=0, descending=False),
                SortingColumn(column_index=2, descending=False),
            ),
        )

    _LOGGER.info("Parquet dataset created.")
    _LOGGER.info("Loading derived data")
    static_features = []
    for filename in tqdm(audio_paths, desc="Loading static features", total=len(audio_paths)):
        filename = str(filename)
        pt_file = Path(filename.replace(".wav", "_features.pt"))
        if not pt_file.exists():
            continue

        for participant_id in SUBJECTS_TO_REMOVE:
            if f"sub-{participant_id}" in str(pt_file):
                _LOGGER.info(f"Skipping subject {participant_id}")
                continue

        features = torch.load(pt_file)
        subj_info = {
            "participant_id": str(pt_file).split("sub-")[1].split("/ses-")[0],
            "session_id": str(pt_file).split("ses-")[1].split("/audio")[0],
            "task_name": str(pt_file).split("task-")[1].split("_features")[0],
        }

        transcription = features.get("transcription", None)
        if transcription is not None:
            transcription = transcription.text
            if (
                subj_info["task_name"].lower().startswith("free-speech")
                or subj_info["task_name"].lower().startswith("audio-check")
                or subj_info["task_name"].lower().startswith("open-response-questions")
            ):
                # we omit tasks where free speech occurs
                transcription = None
        subj_info["transcription"] = transcription

        for key in ["opensmile", "praat_parselmouth", "torchaudio_squim"]:
            subj_info.update(features.get(key, {}))

        static_features.append(subj_info)

    df_static = pd.DataFrame(static_features)
    df_static.to_csv(outdir / "static_features.tsv", sep="\t", index=False)
    # load in the JSON with descriptions of each feature and copy it over
    # write it out again so formatting is consistent between JSONs
    static_features_json_file = resources.files("b2aiprep").joinpath(
        "prepare", "resources", "static_features.json"
    )
    static_features_json = json.load(static_features_json_file.open())
    with open(outdir / "static_features.json", "w") as f:
        json.dump(static_features_json, f, indent=2)
    _LOGGER.info("Finished creating static and dynamic features")

    _LOGGER.info("Creating merged phenotype data.")

    # first initialize a dataframe with all the record IDs which are equivalent to participant IDs.
    df = pd.read_csv(bids_path.joinpath("participants.tsv"), sep="\t")

    # remove subject
    idx = df["record_id"].isin(SUBJECTS_TO_REMOVE)
    if idx.sum() > 0:
        _LOGGER.info(
            f"Removing {idx.sum()} records from phenotype due to hard-coded subject removal."
        )
        df = df.loc[~idx]

    # temporarily keep record_id as the column name to enable joining the dataframes together
    # later we will rename this to participant_id
    df = df[["record_id"]]
    with open(bids_path.joinpath("participants.json"), "r") as f:
        participants = json.load(f)
    phenotype = {"participant_id": participants["record_id"]}

    phenotype_files = list(bids_path.joinpath("phenotype").glob("*.tsv"))
    phenotype_order = [
        "eligibility",
        "enrollment",
        "participant",
        "demographics",
        "confounders",
    ]
    # order phenotype files by (1) the order of the list above then (2) alphabetically
    # since sorts are stable, we can guarantee this by doing it in reverse
    phenotype_files = sorted(phenotype_files)
    phenotype_files = sorted(
        phenotype_files,
        key=lambda x: (
            phenotype_order.index(x.stem) if x.stem in phenotype_order else 100,
            x.stem,
        ),
    )

    if len(phenotype_files) == 0:
        _LOGGER.warning("No phenotype files found.")

    for phenotype_filepath in phenotype_files:
        df_add = pd.read_csv(phenotype_filepath, sep="\t")
        phenotype_name = phenotype_filepath.stem
        if phenotype_name in ["participant", "eligibility", "enrollment"]:
            continue
        # check for overlapping columns
        for col in df_add.columns:
            if col == "record_id":
                continue
            if col in df.columns:
                raise ValueError(f"Column {col} already exists in the dataframe")

        df = df.merge(df_add, on="record_id", how="left")
        with open(bids_path.joinpath("phenotype", f"{phenotype_name}.json"), "r") as f:
            phenotype_add = json.load(f)
        # add the data elements to the overall phenotype dict
        if len(phenotype_add) != 1:
            # we expect there to only be one key
            _LOGGER.warning(
                f"Unexpected keys in phenotype file {phenotype_filepath.stem}: {phenotype_add.keys()}"
            )
        else:
            phenotype_add = next(iter(phenotype_add.values()))["data_elements"]
        phenotype.update(phenotype_add)
    df = df.rename(columns={"record_id": "participant_id"})

    # fix some data values and remove columns we do not want to publish at this time
    df, phenotype = clean_phenotype_data(df, phenotype)

    # write out phenotype data and data dictionary
    df.to_csv(outdir.joinpath("phenotype.tsv"), sep="\t", index=False)
    with open(outdir.joinpath("phenotype.json"), "w") as f:
        json.dump(phenotype, f, indent=2)
    _LOGGER.info("Finished creating merged phenotype data.")


@click.command()
@click.argument("bids_dir_path", type=click.Path())
@click.argument("fix", type=bool)
def validate(
    bids_dir_path,
    fix,
):
    """Validates data in BIDS structure.

    This function checks the integrity and structure of a BIDS directory and
    optionally fixes detected issues.

    Args:
        bids_dir_path (str): Path to the BIDS directory to validate.
        fix (bool): If True, attempts to fix detected issues in the BIDS structure.

    Returns:
        None: Performs validation and optionally fixes errors in-place.
    """

    validate_bids_data(
        bids_dir_path=Path(bids_dir_path),
        fix=fix,
    )


# @click.command()
# @click.argument("source_data_csv_path", type=click.Path(exists=True))
# @click.argument("synthetic_data_path", type=click.Path())
# @click.option(
#     "--n_synthetic_rows", default=100, type=int, help="Number of synthetic rows to generate."
# )
# @click.option("--synthesizer_path", type=click.Path(), help="Path to save/load the synthesizer.")
# def gensynthtabdata(source_data_csv_path, synthetic_data_path, n_synthetic_rows, synthesizer_path):
#     generate_synthetic_tabular_data(
#         source_data_csv_path=Path(source_data_csv_path),
#         synthetic_data_path=Path(synthetic_data_path),
#         n_synthetic_rows=n_synthetic_rows,
#         synthesizer_path=Path(synthesizer_path) if synthesizer_path else None,
#     )


@click.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("-s", "--subject", type=str, default=None)
@click.option("-t", "--task", type=str, default=None)
@click.option("--outdir", type=click.Path(), default=os.getcwd(), show_default=True)
@click.option("--n_mels", type=int, default=20, show_default=True)
@click.option("--win_length", type=int, default=20, show_default=True)
@click.option("--hop_length", type=int, default=10, show_default=True)
@click.option("--transcribe/--no-transcribe", type=bool, default=False, show_default=True)
@click.option("--opensmile", default="eGeMAPSv02", show_default=True)
def convert(
    filename,
    subject,
    task,
    outdir,
    n_mels,
    win_length,
    hop_length,
    transcribe,
    opensmile,
):
    """Extracts features from one audio file.

    This function loads an audio file, extracts relevant features such as
    spectrogram, mel filter bank, MFCCs, speaker embeddings, and OpenSMILE
    features, and optionally transcribes the audio. The extracted features
    are then saved as a PyTorch tensor file.

    Args:
        filename (str): Path to the input audio file.
        subject (str, optional): Identifier for the subject associated with the audio.
        task (str, optional): Identifier for the task related to the audio.
        outdir (str, optional): Directory where the extracted features will be saved. Defaults to the current working directory.
        n_mels (int, optional): Number of mel filter banks for mel spectrogram extraction. Defaults to 20.
        win_length (int, optional): Window length for spectrogram computation. Defaults to 20.
        hop_length (int, optional): Hop length for spectrogram computation. Defaults to 10.
        transcribe (bool, optional): Whether to transcribe the audio. Defaults to False.
        opensmile (str, optional): OpenSMILE feature set to use. Defaults to "eGeMAPSv02".

    Returns:
        dict: Extracted features including spectrogram, mel filter bank, MFCCs, speaker embeddings,
              sample rate, OpenSMILE features, and optional transcription.
    """
    os.makedirs(outdir, exist_ok=True)
    audio = Audio.from_filepath(filename)
    features = {}
    features["subject"] = subject
    features["task"] = task
    resampled_audio = resample_audios([audio], resample_rate=16000)[0]
    features["speaker_embedding"] = extract_speaker_embeddings_from_audios([resampled_audio])
    features["specgram"] = extract_spectrogram_from_audios(
        [audio], win_length=win_length, hop_length=hop_length
    )
    features["melfilterbank"] = extract_mel_filter_bank_from_audios([audio], n_mels=n_mels)
    features["mfcc"] = extract_mfcc_from_audios([audio])
    features["sample_rate"] = audio.sampling_rate
    features["opensmile"] = extract_opensmile_features_from_audios([audio], feature_set=opensmile)
    if transcribe:
        features["transcription"] = transcribe_audios(audios=[audio])[0]
    save_path = Path(outdir) / (Path(filename).stem + ".pt")
    torch.save(features, save_path)
    return features


@click.command()
@click.argument("csvfile", type=click.Path(exists=True))
@click.option("--outdir", type=click.Path(), default=os.getcwd(), show_default=True)
@click.option("--n_mels", type=int, default=20, show_default=True)
@click.option("--n_coeff", type=int, default=20, show_default=True)
@click.option("--win_length", type=int, default=20, show_default=True)
@click.option("--hop_length", type=int, default=10, show_default=True)
@click.option(
    "-p",
    "--plugin",
    nargs=2,
    default=["cf", "n_procs=1"],
    help="Pydra plugin to use",
    show_default=True,
)
@click.option(
    "-c",
    "--cache",
    default=os.path.join(os.getcwd(), "cache-wf"),
    help="Cache dir",
    show_default=True,
)
@click.option("--dataset/--no-dataset", type=bool, default=False, show_default=True)
@click.option("--speech2text/--no-speech2text", type=bool, default=False, show_default=True)
@click.option("--opensmile", nargs=2, default=["eGeMAPSv02", "Functionals"], show_default=True)
def batchconvert(
    csvfile,
    outdir,
    n_mels,
    n_coeff,
    win_length,
    hop_length,
    plugin,
    cache,
    dataset,
    speech2text,
    opensmile,
):
    """Extracts features from audio file list (CSV).

    This function reads a CSV file containing a list of audio file paths (or paths with metadata)
    and extracts various audio features such as spectrogram, mel filter bank, MFCCs,
    speaker embeddings, and OpenSMILE features. It uses Pydra for parallel processing
    and supports optional transcription and dataset generation.

    Args:
        csvfile (str): Path to the CSV file containing audio file paths (with optional metadata).
        outdir (str, optional): Directory where extracted features will be saved. Defaults to the current working directory.
        n_mels (int, optional): Number of mel filter banks for mel spectrogram extraction. Defaults to 20.
        n_coeff (int, optional): Number of coefficients for MFCC extraction. Defaults to 20.
        win_length (int, optional): Window length for spectrogram computation. Defaults to 20.
        hop_length (int, optional): Hop length for spectrogram computation. Defaults to 10.
        plugin (tuple, optional): Pydra plugin and arguments for parallel execution. Defaults to ["cf", "n_procs=1"].
        cache (str, optional): Directory for caching intermediate results. Defaults to "cache-wf" in the current directory.
        dataset (bool, optional): Whether to compile extracted features into a dataset. Defaults to False.
        speech2text (bool, optional): Whether to transcribe audio files. Defaults to False.
        opensmile (tuple, optional): OpenSMILE feature set and function type. Defaults to ["eGeMAPSv02", "Functionals"].

    Returns:
        None: Saves extracted features as PyTorch tensor files and optionally compiles them into a dataset.
    """
    plugin_args = dict()
    for item in plugin[1].split():
        key, value = item.split("=")
        if plugin[0] == "cf" and key == "n_procs":
            value = int(value)
        plugin_args[key] = value

    @pydra.task
    @annotate({"return": {"features": dict}})
    def convert(
        filename,
        subject,
        task,
        outdir,
        n_mels,
        n_coeff,
        win_length,
        hop_length,
        transcribe,
        opensmile,
    ):
        os.makedirs(outdir, exist_ok=True)
        audio = Audio.from_filepath(filename)
        features = {}
        features["subject"] = subject
        features["task"] = task
        resampled_audio = resample_audios([audio], resample_rate=16000)[0]
        features["speaker_embedding"] = extract_speaker_embeddings_from_audios([resampled_audio])
        features["specgram"] = extract_spectrogram_from_audios(
            [audio], win_length=win_length, hop_length=hop_length
        )
        features["melfilterbank"] = extract_mel_filter_bank_from_audios([audio], n_mels=n_mels)
        features["mfcc"] = extract_mfcc_from_audios([audio])
        features["sample_rate"] = audio.sampling_rate
        features["opensmile"] = extract_opensmile_features_from_audios(
            [audio], feature_set=opensmile[0]
        )
        if transcribe:
            features["transcription"] = transcribe_audios(audios=[audio])[0]
        save_path = Path(outdir) / (Path(filename).stem + ".pt")
        torch.save(features, save_path)
        return features

    with open(csvfile, "r") as f:
        reader = csv.DictReader(f)
        num_cols = len(reader.fieldnames)
        lines = [line.strip() for line in f.readlines()]

    if num_cols == 1:
        filenames = [Path(line).absolute().as_posix() for line in lines]
        featurize_task = convert.map(
            filename=filenames,
            subject=[None] * len(filenames),
            task=[None] * len(filenames),
            outdir=[outdir] * len(filenames),
            n_mels=[n_mels] * len(filenames),
            n_coeff=[n_coeff] * len(filenames),
            win_length=[win_length] * len(filenames),
            hop_length=[hop_length] * len(filenames),
            transcribe=[speech2text] * len(filenames),
            opensmile=[opensmile] * len(filenames),
        )
    elif num_cols == 3:
        filenames, subjects, tasks = zip(*[line.split(",") for line in lines])
        featurize_task = convert.map(
            filename=[Path(f).absolute().as_posix() for f in filenames],
            subject=subjects,
            task=tasks,
            outdir=[outdir] * len(filenames),
            n_mels=[n_mels] * len(filenames),
            n_coeff=[n_coeff] * len(filenames),
            win_length=[win_length] * len(filenames),
            hop_length=[hop_length] * len(filenames),
            transcribe=[speech2text] * len(filenames),
            opensmile=[opensmile] * len(filenames),
        )

    cwd = os.getcwd()
    try:
        with pydra.Submitter(plugin=plugin[0], **plugin_args) as sub:
            sub(featurize_task)
    except Exception:
        print("Run finished with errors")
    else:
        print("Run finished successfully")
    os.chdir(cwd)
    results = featurize_task.result(return_inputs=True)
    Path(outdir).mkdir(exist_ok=True, parents=True)
    stored_results = []
    for input_params, result in results:
        if result.errored:
            print(f"File: {input_params['filename']} errored")
            continue
        shutil.copy(result.output["features"], Path(outdir))
        stored_results.append(Path(outdir) / Path(result.output["features"]).name)
    if dataset:

        def gen():
            for val in stored_results:
                yield torch.load(val)

        print(f"Input: {len(results)} files. Processed: {len(stored_results)}")

        def to_hf_dataset(generator, outdir: Path) -> None:
            ds = Dataset.from_generator(generator)
            ds.to_parquet(outdir / "b2aivoice.parquet")

        to_hf_dataset(gen, Path(outdir))


@click.command()
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2", type=click.Path(exists=True))
@click.option("--device", type=str, default=None, show_default=True)
def verify(file1, file2, device):
    """Performs speaker verification.

    This function loads two audio files, resamples them to 16 kHz, and
    computes a speaker verification score to determine whether they
    belong to the same speaker.

    Args:
        file1 (str): Path to the first audio file.
        file2 (str): Path to the second audio file.
        device (str, optional): Device to use for processing. Defaults to None.

    Returns:
        None: Prints the verification score and prediction result.
    """
    audio1 = Audio.from_filepath(file1)
    audio2 = Audio.from_filepath(file2)
    resampled_audios = resample_audios([audio1, audio2], resample_rate=16000)
    audio_pair = (resampled_audios[0], resampled_audios[1])
    score, prediction = verify_speaker(audios=[audio_pair])[0]
    print(f"Score: {float(score):.2f} Prediction: {bool(prediction)}")


@click.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--model", type=str, default="openai/whisper-tiny", show_default=True)
@click.option("--device", type=str, default="cpu", help="Device to use for inference.")
@click.option(
    "--language",
    type=str,
    default="en",
    help="Language of the audio for transcription (default is multi-language).",
)
def transcribe(
    audio_file,
    model,
    device,
    language,
):
    """Transcribes an audio file.

    This function loads an audio file, applies a specified Whisper model for
    transcription, and outputs the transcribed text. It supports multi-language
    transcription and allows specifying the device for inference.

    Args:
        audio_file (str): Path to the input audio file.
        model (str, optional): Path or identifier for the speech-to-text model.
                              Defaults to "openai/whisper-tiny".
        device (str, optional): Device to use for inference (e.g., "cpu" or "cuda").
                                Defaults to "cpu".
        language (str, optional): Language code for the audio file. Defaults to "en"
                                  (English), but supports multi-language transcription.

    Returns:
        None: Prints the transcribed text to the console.
    """
    audio_data = Audio.from_filepath(audio_file)
    hf_model = HFModel(path_or_uri=model)
    device = DeviceType(device.lower())
    language = Language.model_validate({"language_code": language})
    _LOGGER.info("Transcribing audio...")
    transcription = transcribe_audios(
        [audio_data], model=hf_model, device=device, language=language
    )[0]
    print(transcription.text)


@click.command()
@click.argument("input_dir", type=str)
@click.argument("out_file", type=str)
def createbatchcsv(input_dir, out_file):
    """Generates CSV list of .wav paths.

    This function scans a specified top-level directory for .wav files in all
    subdirectories, then saves a CSV file containing the absolute paths of
    the detected audio files.

    Args:
        input_dir (str): Path to the top-level directory containing institution subdirectories (e.g., MIT, UCF).
        out_file (str): Path to the output CSV file where the list of audio files will be saved.

    Returns:
        None: The function writes the output directly to a CSV file and prints the output file path.
    """

    # input_dir is the top level directory of the b2ai Production directory from Wasabi
    # it is expected to contain subfolders with each institution e.g. MIT, UCF, etc.

    # out_file is where a csv file will be saved and should be in the format 'path/name/csv'
    input_dir = Path(input_dir)
    audiofiles = sorted(glob(f"{input_dir}/**/*.wav", recursive=True))

    with open(out_file, "w") as f:

        # using csv.writer method from CSV package
        write = csv.writer(f)

        # write header row
        write.writerow(["filename"])

        for item in audiofiles:
            write.writerow([Path(item).absolute().as_posix()])

    print(f"csv of audiofiles generated at: {out_file}")


@click.command()
@click.argument("src_dir", type=str)
@click.argument("dest_dir", type=str)
def reproschema_audio_to_folder(src_dir, dest_dir):
    """Flattens .wav files using UUIDs as filenames.

    This function scans a source directory for .wav audio files, extracts their UUIDs
    from the filenames, and copies them to a specified destination directory with
    the UUID as the new filename.

    Args:
        src_dir (str): Path to the top-level directory containing the audio files
                      in the ReproSchema export structure.
        dest_dir (str): Path to the directory where the reorganized audio files
                        will be saved.

    Returns:
        None: Copies audio files to the destination folder and logs the output.
    """
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # Walk through the source directory recursively
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            # Check if the file is a .wav file
            if file.lower().endswith(".wav"):
                try:
                    base_filename = os.path.splitext(file)[0]
                    uuid_list = base_filename.split("-")[1:]
                    file_uuid = "-".join(uuid_list)
                except IndexError:
                    print(f"Skipping {file} because UUID could not be extracted.")
                    continue

                # Generate the full destination path with the UUID as the filename
                dest_file_path = os.path.join(dest_dir, f"{file_uuid}.wav")

                # Move and rename the file
                src_file_path = os.path.join(root, file)
                shutil.copy(src_file_path, dest_file_path)

    _LOGGER.info(f"audio files generated at: {dest_dir}")


@click.command()
@click.argument("audio_dir", type=str)
@click.argument("survey_file", type=str)
@click.argument("redcap_csv", type=str)
@click.option("--participant_group", type=str, default=None, show_default=True)
def reproschema_to_redcap(audio_dir, survey_file, redcap_csv, participant_group):
    """Converts reproschema ui data to redcap CSV.

    This function processes survey data and audio metadata from ReproSchema UI exports
    and formats them into a REDCap-compatible CSV file.

    Args:
        audio_dir (str): Path to the directory containing the audio files.
        survey_file (str): Path to the directory containing survey data exported from ReproSchema UI.
        redcap_csv (str): Path to save the generated REDCap-compatible CSV file.
        participant_group (str, optional): Optional argument to replace a REDCap repeat
                                          instance with "Participant".

    Raises:
        FileNotFoundError: If the survey directory or audio directory does not exist.

    Returns:
        None: Saves the converted data as a CSV file at the specified location.
    """

    folder = Path(survey_file)
    sub_folders = sorted([str(f) for f in folder.iterdir() if f.is_dir()])

    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} does not exist.")

    merged_questionnaire_data = []
    # load each file recursively within the folder into its own key
    subject_count = 1
    for subject in sub_folders:
        content = OrderedDict()
        for file in Path(subject).rglob("*"):
            if file.is_file():
                filename = str(file.relative_to(subject))
                with open(f"{subject}/{filename}", "r") as f:
                    content[filename] = json.load(f)

        for questionnaire in content.keys():  # activity files
            try:
                record_id = (subject.split("/")[-1]).split()[0]
                survey_data = content[questionnaire]
                merged_questionnaire_data += parse_survey(survey_data, record_id, questionnaire)
            except Exception:
                continue
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
        subject_count += 1

    survey_df = pd.concat(merged_questionnaire_data, ignore_index=True)
    Path(redcap_csv).mkdir(parents=True, exist_ok=True)

    audio_folders = Path(audio_dir)
    audio_sub_folders = sorted([str(f) for f in audio_folders.iterdir() if f.is_dir()])
    if not audio_folders.is_dir():
        raise FileNotFoundError(f"{audio_folders} does not exist.")

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

        audio_list = []
        for file in Path(subject).glob("**/*"):
            if file.is_file() and str(file).endswith(".wav"):
                audio_list.append(str(file))

        merged_csv += parse_audio(audio_list)

    audio_df = pd.DataFrame(merged_csv)

    merged_df = [survey_df, audio_df]
    output_df = pd.concat(merged_df, ignore_index=True)

    all_columns_path = resources.files("b2aiprep").joinpath(
        "prepare", "resources", "all_columns.json"
    )
    all_columns = json.load(all_columns_path.open())

    columns_to_add = [col for col in all_columns if col not in output_df.columns]
    new_df = pd.DataFrame(columns=columns_to_add)

    df_final = pd.concat([output_df, new_df], axis=1)
    # Option to save on post processing the redcap csv
    if participant_group is not None:
        df_final["redcap_repeat_instrument"] = df_final["redcap_repeat_instrument"].replace(
            participant_group, "Participant"
        )

    merged_csv_path = os.path.join(redcap_csv, "merged-redcap.csv")
    df_final.to_csv(merged_csv_path, index=False)
