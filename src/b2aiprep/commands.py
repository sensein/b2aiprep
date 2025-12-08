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
import typing as t

import click
import pandas as pd
import pydra
import torch
from datasets import Dataset
from pyarrow.parquet import SortingColumn
#from pydra.mark import annotate
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

from b2aiprep.prepare.bids import get_paths, validate_bids_folder_audios
from b2aiprep.prepare.redcap import RedCapDataset
from b2aiprep.prepare.dataset import BIDSDataset
from b2aiprep.prepare.bundle_data import (
    feature_extraction_generator,
    spectrogram_generator,
)

from b2aiprep.prepare.prepare import (
    generate_features_wrapper,
    is_audio_sensitive,
    validate_bids_audio_features,

)
from b2aiprep.prepare.quality_control import quality_control_wrapper

from b2aiprep.prepare.data_validation import validate_phenotype
from b2aiprep.prepare.update import TemplateUpdateError, reorganize_bids_activities, update_bids_template_files

_LOGGER = logging.getLogger(__name__)


def _prime_generator(
    generator_callable: t.Callable[[], t.Iterator[t.Dict[str, t.Any]]],
) -> t.Tuple[bool, t.Optional[t.Callable[[], t.Iterator[t.Dict[str, t.Any]]]]]:
    """Peek at the first item while returning a generator that still yields it."""
    gen = generator_callable()
    try:
        first_item = next(gen)
    except StopIteration:
        close_fn = getattr(gen, "close", None)
        if callable(close_fn):
            close_fn()
        return False, None

    def seeded_generator():
        # yield the first item
        yield first_item

        # then yield the rest by recreating the generator, skipping the first one
        gen2 = generator_callable()
        next(gen2)  # skip first
        for item in gen2:
            yield item

    return True, seeded_generator

import numpy as np

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

    dashboard_path = str(resources.files("b2aiprep").joinpath("dashboard/app.py"))
    run(dashboard_path, args=[bids_path.as_posix()], flag_options=[], is_hello=False)

@click.command()
@click.argument("filename", type=click.Path(exists=True))
@click.argument("reproschema", type=click.Path(exists=True))
@click.option(
    "--outdir",
    type=click.Path(),
    default=Path.cwd().joinpath("output").as_posix(),
    show_default=True,
)
@click.option("--audiodir", type=click.Path(), default=None, show_default=True)
@click.option("--max-audio-workers", type=int, default=16, show_default=True, help="Number of parallel threads for audio file copying")
def redcap2bids(
    filename,
    reproschema,
    outdir,
    audiodir,
    max_audio_workers,
):
    """Organizes into BIDS-like without features.

    This function processes a REDCap data file and optionally links audio files
    to structure them in a format similar to the Brain Imaging Data Structure (BIDS).

    Args:
        filename (str): Path to the REDCap data file.
        reproschema (str): Path to the ReproSchema JSON files directory. Should contain b2ai-redcap2bids_schema file.
        outdir (str, optional): Directory where the converted BIDS-like data
                                will be saved. Defaults to "output" in the current working directory.
        audiodir (str, optional): Directory containing associated audio files.
                                  If not provided, only the REDCap data is processed.
        max_audio_workers (int, optional): Number of parallel threads for audio copying. Defaults to 16.

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
    redcap_dataset = RedCapDataset.from_redcap(filename)
    
    BIDSDataset.from_redcap(
        redcap_dataset,
        outdir=Path(outdir),
        reproschema_source_dir=reproschema,
        audiodir=audiodir,
        max_audio_workers=max_audio_workers
    )

@click.command()
@click.argument("source_bids_dir_path", type=click.Path())
@click.argument("target_bids_dir_path", type=click.Path())
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
@click.option("-S", "--subject_file", type=str, default=None, show_default=True)
@click.option("--is_sequential", type=bool, default=False, show_default=True)
@click.option("--update", type=bool, default=False, show_default=True)
def generate_audio_features(
    source_bids_dir_path,
    target_bids_dir_path,
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
    subject_file,
    is_sequential,
    update,
):
    """
    Generates features given bids folder

    Args:
        source_bids_dir_path: Path to store BIDS-like data
        target_bids_dir_path: Path to target folder to store feature extractions
        tar_file_path: Path to store tar file. If provided, creates a compressed archive of the target directory
        transcription_model_size: Size of transcription model ('tiny', 'small', 'medium', or 'large')
        n_cores: Number of cores to run feature extraction on. Ignored if using Dask
        with_sensitive: Whether to include sensitive data in the output
        overwrite: Whether to overwrite existing files during processing
        validate: Whether to validate the BIDS folder structure after processing
        cache: Path to cache directory. Defaults to 'b2aiprep_cache' in parent of bids_dir_path
        address: Dask scheduler address for distributed processing. If provided, uses Dask instead of concurrent.futures
        percentile: Percentile threshold for processing. Use to process subset of data
        subject_id: Specific subject ID to process. If provided, only processes this subject
        subject_file: File that lists specific subject IDs to process. If provided, only processes these subjects
        is_sequential: Specifies whether to extract audio features sequentially
        update: Specifies if we wish to update by creating target folder and storing extracts there
    """
    os.makedirs(source_bids_dir_path, exist_ok=True)   
    if source_bids_dir_path != target_bids_dir_path:
        shutil.copytree(source_bids_dir_path, target_bids_dir_path, dirs_exist_ok=True)
    else:
        _LOGGER.warning("Target folder is the same a source folder, will extract features in source folder...")

    if update:
        shadow = target_bids_dir_path
        os.makedirs(target_bids_dir_path, exist_ok=True)
        shutil.copytree(source_bids_dir_path, shadow, dirs_exist_ok=True)
        bids_path = Path(target_bids_dir_path)
    elif source_bids_dir_path == target_bids_dir_path:
        bids_path = Path(source_bids_dir_path)
    else:
        bids_path = Path(target_bids_dir_path)

    _LOGGER.info("Beginning audio feature extraction...")

    generate_features_wrapper(
        bids_path=bids_path,
        transcription_model_size=transcription_model_size,
        n_cores=n_cores,
        with_sensitive=with_sensitive,
        overwrite=overwrite,
        cache=cache,
        address=address,
        percentile=percentile,
        subject_id=subject_id,
        subject_file=subject_file,
        update=update,
        is_sequential=is_sequential,
    )

    _LOGGER.info("Audio feature extraction complete.")
    if validate:
        # Below code checks to see if we have all the expected feature/transcript files.
        validate_bids_folder_audios(bids_path)

    if tar_file_path is not None:
        _LOGGER.info("Saving .tar file with processed data...")
        with tarfile.open(tar_file_path, "w:gz") as tar:
            tar.add(target_bids_dir_path, arcname=os.path.basename(target_bids_dir_path))
        _LOGGER.info(f"Saved processed data .tar file at: {tar_file_path}")

    _LOGGER.info("Process completed.")


@click.command()
@click.argument("bids_path", type=click.Path())
@click.argument("output_metrics_path", type=click.Path())
@click.option("-b", "--batch_size", type=int, default=8, show_default=True)
@click.option("-n", "--num_cores", type=int, default=4, show_default=True)
@click.option("--skip_windowing/--no-skip_windowing", type=bool, default=False, show_default=True)
def run_quality_control_on_audios(
    bids_path,
    output_metrics_path,
    batch_size,
    num_cores,
    skip_windowing,
):

    bids_path = Path(bids_path)

    audio_paths = get_paths(bids_path, file_extension=".wav")
    audio_paths = [x["path"] for x in audio_paths]

    outdir = Path(output_metrics_path)
    outdir.mkdir(parents=True, exist_ok=True)


    quality_control_wrapper(
        audio_paths=audio_paths,
        outdir=outdir,
        batch_size=batch_size,
        num_cores=num_cores,
        skip_windowing=skip_windowing
    )


def _build_dataset_from_generator(generator_factory, feature_label: str):
    """Materialize a datasets.Dataset from a generator without double-loading files."""
    generator = generator_factory()
    try:
        first_record = next(generator)
    except StopIteration:
        _LOGGER.info("No values found for %s; skipping parquet export.", feature_label)
        return None

    def _stream():
        yield first_record
        for record in generator:
            yield record

    return Dataset.from_generator(_stream, num_proc=1)

@click.command()
@click.argument("bids_path", type=click.Path(exists=True))
@click.argument("outdir", type=click.Path())
@click.option("--skip_audio/--no-skip_audio", type=bool, default=True, show_default=True, help="Skip processing audio files")
@click.option("--skip_audio_features/--no-skip_audio_features", type=bool, default=False, show_default=True, help="Skip processing feature files")
def create_bundled_dataset(bids_path, outdir, skip_audio, skip_audio_features):
    """Bundles dataset from BIDS data.

    The bundled output loads data from generated .pt files, which have
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
    if not skip_audio:
        _LOGGER.warning("Audio data is currently not supported for bundling. No audio files will be included in the bundle.")

    bids_path = Path(bids_path)
    feature_paths = get_paths(bids_path, file_extension=".pt")
    feature_paths = [x["path"] for x in feature_paths]

    if len(feature_paths) == 0:
        raise FileNotFoundError(
            f"No feature files (.pt) found in {bids_path}. Please check the directory structure."
        )

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # remove _features at the end of the file stem
    audio_paths = [
        x.parent.joinpath(x.stem[:-9]).with_suffix(".wav") if "_features" in x.name else x
        for x in feature_paths
    ]

    audio_paths = sorted(
        audio_paths,
        # sort first by subject, then by task
        key=lambda x: (x.stem.split("_")[0], x.stem.split("_")[2]),
    )

    _LOGGER.info("Creating phenotype data")
    phenotype_path = outdir / "phenotype"
    bids_phenotype_path = bids_path / "phenotype"

    if not bids_phenotype_path.exists():
        _LOGGER.warning("Could not locate Phenotype Folder, skipping... ")
    else:
        shutil.copytree(bids_phenotype_path, phenotype_path, dirs_exist_ok=True)
        _LOGGER.info("Finished creating phenotype data")

    _LOGGER.info("Loading audio static features.")
    static_features = []
    for filepath in tqdm(audio_paths, desc="Loading static features", total=len(audio_paths)):
        pt_file = filepath.parent.joinpath(f"{filepath.stem}_features.pt")
        if not filepath.exists(): #pt will exist based on how audio paths were generated, but audio file might not exist
            continue

        device = 'cpu' # not checking for cuda because optimization would be minimal if any
        features = torch.load(pt_file, weights_only=False, map_location=torch.device(device))
        subj_info = {
            "participant_id": str(pt_file).split("sub-")[1].split("/ses-")[0],
            "session_id": str(pt_file).split("ses-")[1].split("/audio")[0],
            "task_name": str(pt_file).split("task-")[1].split("_features")[0],
        }

        transcription = features.get("transcription", None)
        if transcription is not None:
            transcription = transcription.text
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
    _LOGGER.info("Finished creating static features.")

    _LOGGER.info("Loading other features into HF datasets.")

    features_to_extract = [
        {'feature_class': None, 'feature_name': 'ppgs'},
        {'feature_class': 'torchaudio', 'feature_name': 'pitch'},
        {'feature_class': 'torchaudio', 'feature_name': 'mel_filter_bank'},
        {'feature_class': 'torchaudio', 'feature_name': 'mel_spectrogram'},
        {'feature_class': 'torchaudio', 'feature_name': 'spectrogram'},
        {'feature_class': 'torchaudio', 'feature_name': 'mfcc'},
        {'feature_class': 'sparc', 'feature_name': 'ema'},
        {'feature_class': 'sparc', 'feature_name': 'loudness'},
        {'feature_class': 'sparc', 'feature_name': 'pitch'},
        {'feature_class': 'sparc', 'feature_name': 'periodicity'},
        {'feature_class': 'sparc', 'feature_name': 'pitch_stats'},

    ]
    
    features_dir = outdir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    for feature in features_to_extract:
        feature_class = feature['feature_class']
        feature_name = feature['feature_name']

        feature_output = f"{feature_class}_{feature_name}" if feature_class else feature_name

        if feature_name != "spectrogram":
            use_byte_stream_split = True
            maybe_empty_generator = partial(
                feature_extraction_generator,
                audio_paths=audio_paths,
                feature_name=feature_name,
                feature_class=feature_class
            )
        else:
            use_byte_stream_split = False
            maybe_empty_generator = partial(
                spectrogram_generator,
                audio_paths=audio_paths,
            )
        has_data, feature_generator = _prime_generator(maybe_empty_generator)
        if not has_data or feature_generator is None:
            _LOGGER.warning(
                "No non-NaN entries found for %s feature. Skipping parquet export.",
                feature_output,
            )
            continue

        # # sort the dataset by identifier and task_name
        # ds = Dataset.from_generator(maybe_empty_generator, num_proc=1)
        ds = Dataset.from_generator(feature_generator, num_proc=1)
        if len(ds) == 0:  
            _LOGGER.warning(
                "No non-NaN entries found for %s feature. Skipping parquet export.",
                feature_output,
            )
            continue
        ds.to_parquet(
            str((f"{features_dir}/{feature_output}.parquet")),
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


@click.command()
@click.argument("dataset_path", type=click.Path(exists=True))
def validate_derived_dataset(dataset_path):
    """Validates derived dataset.

    This function checks the integrity and structure of a derived dataset
    directory and optionally fixes detected issues.

    Args:
        dataset_path (str): Path to the derived dataset directory to validate.

    Returns:
        None: Performs validation and optionally fixes errors in-place.
    """
    dataset_path = Path(dataset_path)

    assert dataset_path.exists(), f"Dataset path {dataset_path} does not exist."
    assert dataset_path.is_dir(), f"Dataset path {dataset_path} is not a directory."

    # we expect spectrograms.parquet and mfcc.parquet to exist
    assert dataset_path.joinpath(
        "spectrogram.parquet"
    ).exists(), f"Dataset path {dataset_path} does not contain spectrogram.parquet."
    assert dataset_path.joinpath(
        "mfcc.parquet"
    ).exists(), f"Dataset path {dataset_path} does not contain mfcc.parquet."

    for base_dataframe_name in ["static_features", "phenotype"]:
        assert dataset_path.joinpath(
            f"{base_dataframe_name}.tsv"
        ).exists(), f"Dataset path {dataset_path} does not contain {base_dataframe_name}.tsv."
        assert dataset_path.joinpath(
            f"{base_dataframe_name}.json"
        ).exists(), f"Dataset path {dataset_path} does not contain {base_dataframe_name}.json."
        df = pd.read_csv(dataset_path.joinpath(f"{base_dataframe_name}.tsv"), sep="\t")
        with open(dataset_path.joinpath(f"{base_dataframe_name}.json"), "r") as f:
            info = json.load(f)
        missing_columns = []
        for column in info.keys():
            if column not in df.columns:
                missing_columns.append(column)
        assert (
            len(missing_columns) == 0
        ), f"Columns not found in {base_dataframe_name}.tsv: {missing_columns}"

@click.command()
@click.argument("bids_path", type=click.Path(exists=True))
@click.argument("outdir", type=click.Path())
@click.argument("deidentify_config_dir", type=click.Path(exists=True))
@click.option("--skip_audio/--no-skip_audio", type=bool, default=False, show_default=True, help="Skip processing audio files")
@click.option("--skip_audio_features/--no-skip_audio_features", type=bool, default=False, show_default=True, help="Skip processing audio feature files")
def deidentify_bids_dataset(bids_path, outdir, deidentify_config_dir, skip_audio, skip_audio_features):
    """Creates a deidentified version of a given BIDS dataset.

    The deidentified version of the dataset: 

    - only includes audio (omits features)
    - cleans/processes the various phenotype files and participants.tsv
    - removes sensitive audio records
    - applies deidentification procedures

    Requires deidentify_config_dir to contain
    - participants_to_remove.json
    - audio_to_remove.json
    - id_remapping.json
    - sensitive_audio_tasks.json
    """
    bids_path = Path(bids_path)
    deidentify_config_dir = Path(deidentify_config_dir)
    
    # Create BIDSDataset instance and use the deidentify method
    bids_dataset = BIDSDataset(bids_path)
    bids_dataset.deidentify(
        outdir=outdir, deidentify_config_dir=deidentify_config_dir, skip_audio=skip_audio, skip_audio_features=skip_audio_features
    )
    
    _LOGGER.info("Deidentified dataset created successfully.")

@click.command()
@click.argument("bids_dir_path", type=click.Path())
@click.option("--report_path", type=click.Path(), default=None, show_default=True)
def validate_feature_extraction(
    bids_dir_path,
    report_path
):
    bids_dir_path = Path(bids_dir_path)
    report_path = Path(report_path) if report_path else None

    validate_bids_audio_features(
        bids_dir_path=bids_dir_path,
        report_path=report_path
    )


@click.command('validate-phenotype')
@click.argument("phenotype_path", type=click.Path())
def validate_phenotype_command(phenotype_path):
    """
    This function takes the phenotype data and validates each record.

    Args:
        phenotype_path (str): Path to the phenotype folder with CSV files.
       
    Returns:
        None: Performs validation

    Args:
        phenotype_path: Path to the derivatives tsv
    """
    _LOGGER.info("Validating Phenotype Data...")
    phenotype_path = Path(phenotype_path)
    phenotype_files: t.List[Path] = []
    for file in sorted(phenotype_path.glob('*.tsv')):
        if file.with_suffix('.json').exists():
            phenotype_files.append(file)
        else:
            _LOGGER.warning(f"Skipping {file} as no associated JSON file found.")
    
    # fine associated json files
    for phenotype_file in tqdm(phenotype_files, desc="Validating phenotype files"):
        df = pd.read_csv(phenotype_file, sep='\t', header=0)
        phenotype_dictionary = json.loads(phenotype_file.with_suffix('.json').read_text())
        validate_phenotype(df, phenotype_dictionary)

    _LOGGER.info("Phenotype Data Validation Complete.")

@click.command(name="update-bids-template")
@click.option(
    "--submodule-path",
    type=click.Path(path_type=Path, exists=True, resolve_path=True),
    default=None,
    help="Override path to the b2ai-redcap2rs submodule.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, resolve_path=True),
    default=None,
    help="Directory to write regenerated phenotype template JSON files.",
)
@click.option(
    "--schema-file",
    type=click.Path(path_type=Path, exists=True, resolve_path=True),
    default=None,
    help="Optional path to the protocol schema file to parse.",
)
@click.option(
    "--reorganize",
    is_flag=True,
    help="Reorganize phenotype JSONs into manually curated file organization."
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Report which files would be generated without writing them.",
)
def update_bids_template_command(submodule_path, output_dir, schema_file, reorganize, dry_run):
    """Regenerate phenotype template JSONs from the reproschema repository."""

    target_dir = (
        Path(output_dir)
        if output_dir
        else Path(__file__).resolve().parents[2] / "src/b2aiprep/template/phenotype"
    ).resolve()

    try:
        generated_files = update_bids_template_files(
            submodule_path=submodule_path,
            output_dir=target_dir,
            schema_file=schema_file,
            dry_run=dry_run,
        )
    except TemplateUpdateError as exc:
        raise click.ClickException(str(exc)) from exc

    if reorganize:
        generated_files = reorganize_bids_activities(target_dir, dry_run)

    verb = "Planned" if dry_run else "Wrote"
    click.echo(f"{verb} {len(generated_files)} template file(s) in {target_dir}")



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
    audio = Audio(filepath=filename)
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
    pass
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
        audio = Audio(filepath=filename)
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
                yield torch.load(val, weights_only=False)

        print(f"Input: {len(results)} files. Processed: {len(stored_results)}")

        def to_hf_dataset(generator, outdir: Path) -> None:
            ds = Dataset.from_generator(generator)
            ds.to_parquet(outdir / "b2aivoice.parquet")

        to_hf_dataset(gen, Path(outdir))
"""

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
    audio1 = Audio(filepath=file1)
    audio2 = Audio(filepath=file2)
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
    audio_data = Audio(filepath=audio_file)
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
@click.argument("audio_dir", type=str)
@click.argument("survey_file", type=str)
@click.argument("redcap_csv", type=str)
@click.option("--is_import", type=bool, default=False, show_default=True)
@click.option("--disable-manual-fixes", is_flag=True, default=False, show_default=True, help="Disable manual fixes for known issues in ReproSchema data.")
def reproschema_to_redcap(audio_dir, survey_file, redcap_csv, is_import, disable_manual_fixes):
    """Converts reproschema ui data to redcap CSV.

    This function processes survey data and audio metadata from ReproSchema UI exports
    and formats them into a REDCap-compatible CSV file using the new RedCapDataset class.

    Args:
        audio_dir (str): Path to the directory containing the audio files.
        survey_file (str): Path to the directory containing survey data exported from ReproSchema UI.
        redcap_csv (str): Path to save the generated REDCap-compatible CSV file.
        is_import (bool): If True, modifies how we convert to redcap csv due to difference in import csv and exports
        disable_manual_fixes (bool): If True, disables manual fixes for known issues in ReproSchema data.

    Raises:
        FileNotFoundError: If the survey directory or audio directory does not exist.

    Returns:
        None: Saves the converted data as a CSV file at the specified location.
    """
    redcap_csv_path = Path(redcap_csv)
    if redcap_csv_path.is_dir():
        raise ValueError("redcap_csv argument must be a file path, not a directory.")

    dataset = RedCapDataset.from_reproschema(
        audio_dir=audio_dir,
        survey_dir=survey_file,
        is_import=is_import,
        disable_manual_fixes=disable_manual_fixes,
    )
    
    # Ensure the output directory exists
    redcap_csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not is_import:
        dataset.df = RedCapDataset.collapse_checked_columns(dataset.df)
    dataset.df.to_csv(redcap_csv_path, index=False)    
    _LOGGER.info(f"Successfully converted ReproSchema data to RedCap CSV: {redcap_csv_path}")


@click.command()
@click.argument("bids_src_dir", type=str)
@click.argument("dest_dir", type=str)
def bids2shadow(bids_src_dir, dest_dir):
    """
    This function scans the bids folder for all the .pt files, and copies all of the
    .pt files over to the target directory, while maintaining bids structure.

    Args:
        bids_src_dir: Path to the  directory containing the pytorch files
        dest_dir: Path to the directory where we wish to have the shadow tree
    """
    # Convert to Path objects
    src_dir = Path(bids_src_dir)
    target_dir = Path(dest_dir)

    # Ensure the target directory exists
    target_dir.mkdir(parents=True, exist_ok=True)

    # Walk through the source directory
    for root, dirs, files in os.walk(src_dir):
        # Filter for .pt files
        for file in files:
            if file.endswith(".pt"):
                # Create the relative path to the source directory
                relative_path = Path(root).relative_to(src_dir)
                target_path = target_dir / relative_path

                # Ensure the target directory exists
                target_path.mkdir(parents=True, exist_ok=True)

                # Copy the .pt file
                shutil.copy(Path(root) / file, target_path / file)

    _LOGGER.info(f".pt files have been added to {dest_dir}")
