"""Commands available through the CLI."""

import csv
import json
import logging
import os
import shutil
import tarfile
import typing as t
from glob import glob
from pathlib import Path

import click
import numpy as np
import pandas as pd
import pkg_resources
import pydra
import torch
from datasets import Dataset
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
from b2aiprep.prepare.prepare import extract_features_workflow, validate_bids_data

# from b2aiprep.synthetic_data import generate_synthetic_tabular_data

_LOGGER = logging.getLogger(__name__)


@click.command()
@click.argument("bids_dir", type=click.Path(exists=True))
def dashboard(bids_dir: str):
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
):
    """Organizes the data into a BIDS-like directory structure and extracts audio features.

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
    """
    if cache is None:
        cache = Path(bids_dir_path).parent / "b2aiprep_cache"
    os.makedirs(cache, exist_ok=True)

    if redcap_csv_path is not None and audio_dir_path is not None:
        _LOGGER.info("Organizing data into BIDS-like directory structure...")
        redcap_to_bids(Path(redcap_csv_path), Path(bids_dir_path), Path(audio_dir_path))
        _LOGGER.info("Data organization complete.")

    _LOGGER.info("Beginning audio feature extraction...")
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


def load_audio_features(
    audio_paths,
    file_extension,
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

        feature_path = features_dir / f"{wav_path.stem}_transcription.txt"
        with open(feature_path, "r", encoding="utf-8") as text_file:
            transcription = text_file.read()
        output["transcription"] = transcription

        for feature_name in ["speaker_embedding", "specgram", "melfilterbank", "mfcc", "opensmile"]:
            feature_path = features_dir / f"{wav_path.stem}_{feature_name}.{file_extension}"
            if not feature_path.exists():
                continue
            if feature_name == "speaker_embedding":
                output["speaker_embedding"] = torch.load(feature_path)
            else:
                data = torch.load(feature_path)
                if len(data) == 1:
                    key = next(iter(data.keys()))
                    data[key] = data[key].numpy().astype(np.float32)
                output.update(torch.load(feature_path))

        # load transcription
        feature_path = features_dir / f"{wav_path.stem}_transcription.txt"
        if feature_path.exists():
            output["transcription"] = feature_path.read_text()

        yield output


@click.command()
@click.argument("bids_path", type=click.Path(exists=True))
@click.argument("outdir", type=click.Path())
def create_derived_dataset(bids_path, outdir):
    bids_path = Path(bids_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    audio_paths = get_audio_paths(bids_dir_path=bids_path)
    audio_paths = [x["path"] for x in audio_paths]
    _LOGGER.info("Loading derived data")
    static_features = []
    for filename in audio_paths:
        filename = str(filename)
        pt_file = Path(filename.replace(".wav", "_features.pt"))
        if not pt_file.exists():
            continue
        features = torch.load(pt_file)
        subj_info = {
            "participant": filename.split("sub-")[1].split("/ses-")[0],
            "task": filename.split("task-")[1].split("_features")[0],
        }
        for key in ["opensmile", "praat_parselmouth", "torchaudio_squim"]:
            subj_info.update(features.get(key, {}))
        subj_info["duration"] = features.get("duration", None)
        dynamic = features["torchaudio"]
        dynamic["transcription"] = features.get("transcription", None)
        if dynamic["transcription"] is not None:
            subj_info["transcription"] = dynamic["transcription"].text
        static_features.append(subj_info)
        outfile = (
            outdir
            / f"sub-{subj_info['participant']}"
            / f"sub-{subj_info['participant']}_task-{subj_info['task']}_temporalfeatures.pt"
        )
        if not outfile.parent.exists():
            outfile.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dynamic, outfile)
    df_static = pd.DataFrame(static_features)
    df_static.to_csv(outdir / "static_features.csv", index=False)
    _LOGGER.info("Finished creating static and dynamic features")

    _LOGGER.info("Creating merged phenotype data.")

    # first initialize a dataframe with all the record IDs which are equivalent to participant IDs.
    df = pd.read_csv(bids_path.joinpath("participants.tsv"), sep="\t")
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
        phenotype.update(phenotype_add)

    df = df.rename(columns={"record_id": "participant_id"})
    df.to_csv(outdir.joinpath("phenotype.tsv"), sep="\t", index=False)

    # write out phenotype
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
    """Organizes the data into a BIDS-like directory structure.

    redcap_csv_path: path to the redcap csv\n
    audio_dir_path: path to directory with audio files\n
    bids_dir_path: path to store bids-like data\n
    tar_file_path: path to store tar file\n
    transcription_model_size: tiny, small, medium, or large\n
    n_cores: number of cores to run feature extraction on\n
    with_sensitive: whether to include sensitive data
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
    """
    Transcribes the audio_file.
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
