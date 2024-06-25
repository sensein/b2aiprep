import csv
import os
import shutil
import typing as ty
from glob import glob
from pathlib import Path

import click
import pkg_resources
import pydra
import torch
from pydra.mark import annotate
from pydra.mark import task as pydratask
from streamlit import config as _config
from streamlit.web.bootstrap import run

from b2aiprep.prepare import (
    redcap_to_bids,
)
from b2aiprep.summer_school_data import prepare_summer_school_data
from b2aiprep.process import (
    Audio,
    SpeechToText,
    to_features,
    to_hf_dataset,
    verify_speaker_from_files,
)

@click.group()
def main():
    pass


@main.command()
@click.argument("bids_dir", type=click.Path(exists=True))
def dashboard(bids_dir: str):
    bids_path = Path(bids_dir).resolve()
    if not bids_path.exists():
        raise ValueError(f"Input path {bids_path} does not exist.")
        
    if not bids_path.is_dir():
        raise ValueError(f"Input path {bids_path} is not a directory.")
    _config.set_option("server.headless", True)

    dashboard_path = pkg_resources.resource_filename('b2aiprep', 'app/Dashboard.py')
    run(dashboard_path, args=[bids_path.as_posix()], flag_options=[], is_hello=False)

@main.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("--outdir", type=click.Path(), default=Path.cwd().joinpath('output').as_posix(), show_default=True)
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


@main.command()
@click.argument("redcap_csv_path", type=click.Path(exists=True))
@click.argument("audio_dir_path", type=click.Path(exists=True))
@click.argument("bids_dir_path", type=click.Path())
@click.argument("tar_file_path", type=click.Path())
def prepsummerdata(
    redcap_csv_path, 
    audio_dir_path, 
    bids_dir_path, 
    tar_file_path
):
    prepare_summer_school_data(
        redcap_csv_path=Path(redcap_csv_path),
        audio_dir_path=Path(audio_dir_path),
        bids_dir_path=Path(bids_dir_path),
        tar_file_path=Path(tar_file_path)
    )


@main.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("-s", "--subject", type=str, default=None)
@click.option("-t", "--task", type=str, default=None)
@click.option("--outdir", type=click.Path(), default=os.getcwd(), show_default=True)
@click.option("--save_figures/--no-save_figures", default=False, show_default=True)
@click.option("--n_mels", type=int, default=20, show_default=True)
@click.option("--n_coeff", type=int, default=20, show_default=True)
@click.option("--win_length", type=int, default=20, show_default=True)
@click.option("--hop_length", type=int, default=10, show_default=True)
@click.option("--compute_deltas/--no-compute_deltas", default=True, show_default=True)
@click.option("--speech2text/--no-speech2text", type=bool, default=False, show_default=True)
@click.option("--opensmile", nargs=2, default=["eGeMAPSv02", "Functionals"], show_default=True)
def convert(
    filename,
    subject,
    task,
    outdir,
    save_figures,
    n_mels,
    n_coeff,
    win_length,
    hop_length,
    compute_deltas,
    speech2text,
    opensmile,
):
    os.makedirs(outdir, exist_ok=True)
    to_features(
        filename,
        subject,
        task,
        outdir=Path(outdir),
        save_figures=save_figures,
        extract_text=speech2text,
        n_mels=n_mels,
        n_coeff=n_coeff,
        win_length=win_length,
        hop_length=hop_length,
        compute_deltas=compute_deltas,
        opensmile_feature_set=opensmile[0],
        opensmile_feature_level=opensmile[1],
        device="cpu",
    )


@main.command()
@click.argument("csvfile", type=click.Path(exists=True))
@click.option("--outdir", type=click.Path(), default=os.getcwd(), show_default=True)
@click.option("--save_figures/--no-save_figures", default=False, show_default=True)
@click.option("--n_mels", type=int, default=20, show_default=True)
@click.option("--n_coeff", type=int, default=20, show_default=True)
@click.option("--win_length", type=int, default=20, show_default=True)
@click.option("--hop_length", type=int, default=10, show_default=True)
@click.option("--compute_deltas/--no-compute_deltas", default=True, show_default=True)
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
    save_figures,
    n_mels,
    n_coeff,
    win_length,
    hop_length,
    compute_deltas,
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

    featurize_pdt = pydratask(annotate({"return": {"features": ty.Any}})(to_features))
    featurize_task = featurize_pdt(
        n_mels=n_mels,
        n_coeff=n_coeff,
        win_length=win_length,
        hop_length=hop_length,
        compute_deltas=compute_deltas,
        cache_dir=Path(cache).absolute(),
        save_figures=save_figures,
        extract_text=speech2text,
        opensmile_feature_set=opensmile[0],
        opensmile_feature_level=opensmile[1],
        device="cpu",
    )

    with open(csvfile, "r") as f:
        reader = csv.DictReader(f)
        num_cols = len(reader.fieldnames)
        lines = [line.strip() for line in f.readlines()]

    # parse csv file differently if it is one column 'filename'
    # or three column 'filename','subject','task'
    if num_cols == 1:
        filenames = []
        for line in lines:
            filename = line
            filenames.append(Path(filename).absolute().as_posix())
        featurize_task.split(
            splitter=("filename",),
            filename=filenames,
        )
    elif num_cols == 3:
        filenames = []
        subjects = []
        tasks = []
        for line in lines:
            filename, subject, task = line.split(",")
            filenames.append(Path(filename).absolute().as_posix())
            subjects.append(subject)
            tasks.append(task)
        featurize_task.split(
            splitter=("filename", "subject", "task"),
            filename=filenames,
            subject=subjects,
            task=tasks,
        )

    cwd = os.getcwd()
    try:
        with pydra.Submitter(plugin=plugin[0], **plugin_args) as sub:
            sub(runnable=featurize_task)
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
            print(f"File: {input_params['to_features.filename']} errored")
            continue
        shutil.copy(result.output.features[1], Path(outdir))
        if save_figures:
            shutil.copy(result.output.features[2], Path(outdir))
        stored_results.append(Path(outdir) / Path(result.output.features[1]).name)
    if dataset:

        def gen():
            for val in stored_results:
                yield torch.load(val)

        print(f"Input: {len(results)} files. Processed: {len(stored_results)}")
        to_hf_dataset(gen, Path(outdir))


@main.command()
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2", type=click.Path(exists=True))
@click.argument("model", type=str)
@click.option("--device", type=str, default=None, show_default=True)
def verify(file1, file2, model, device):
    score, prediction = verify_speaker_from_files(file1, file2, model=model, device=device)
    print(f"Score: {float(score):.2f} Prediction: {bool(prediction)}")

@main.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--model_id", type=str, default="openai/whisper-tiny", show_default=True)
@click.option("--max_new_tokens", type=int, default=128, show_default=True)
@click.option("--chunk_length_s", type=int, default=30, show_default=True)
@click.option("--batch_size", type=int, default=16, show_default=True)
@click.option("--device", type=str, default=None, help="Device to use for inference.")
@click.option(
    "--return_timestamps",
    type=str,
    default="false",
    help="Return timestamps with the transcription. Can be 'true', 'false', or 'word'.",
)
@click.option(
    "--language",
    type=str,
    default=None,
    help="Language of the audio for transcription (default is multi-language).",
)
def transcribe(
    audio_file,
    model_id,
    max_new_tokens,
    chunk_length_s,
    batch_size,
    device,
    return_timestamps,
    language,
):
    """
    Transcribes the audio_file.
    """
    # Convert return_timestamps to the correct type
    if return_timestamps.lower() == "true":
        return_timestamps = True
    elif return_timestamps.lower() == "false":
        return_timestamps = False
    else:
        return_timestamps = "word"

    stt = SpeechToText(
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        chunk_length_s=chunk_length_s,
        batch_size=batch_size,
        return_timestamps=return_timestamps,
        device=device,
    )

    audio_data = Audio.from_file(audio_file)
    transcription = stt.transcribe(audio_data, language=language)
    print("Transcription:", transcription)


@main.command()
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


if __name__ == '__main__':
    # include main to enable python debugging
    main()  # pylint: disable=no-value-for-parameter