import csv
import os
import shutil
import typing as ty
from glob import glob
from pathlib import Path

import click
import pydra
import torch
from pydra.mark import annotate
from pydra.mark import task as pydratask

from .process import (
    Audio,
    SpeechToText,
    to_features,
    to_hf_dataset,
    verify_speaker_from_files,
)

try:
    import TTS
except ImportError:
    TTS = None
else:
    from .process import VoiceConversion


@click.group()
def main():
    pass


@main.command()
@click.argument("filename", type=click.Path(exists=True))
@click.option("-s", "--subject", type=ty.Optional[str], default=None)
@click.option("-t", "--task", type=ty.Optional[str], default=None)
@click.option("--outdir", type=click.Path(), default=os.getcwd(), show_default=True)
@click.option("--save_figures/--no-save_figures", default=False, show_default=True)
@click.option("--n_mels", type=int, default=20, show_default=True)
@click.option("--n_coeff", type=int, default=20, show_default=True)
@click.option("--compute_deltas/--no-compute_deltas", default=True, show_default=True)
@click.option("--opensmile", nargs=2, default=["eGeMAPSv02", "Functionals"], show_default=True)
def convert(
    filename, subject, task, outdir, save_figures, n_mels, n_coeff, compute_deltas, opensmile
):
    to_features(
        filename,
        subject,
        task,
        outdir=Path(outdir),
        save_figures=save_figures,
        n_mels=n_mels,
        n_coeff=n_coeff,
        compute_deltas=compute_deltas,
        opensmile_feature_set=opensmile[0],
        opensmile_feature_level=opensmile[1],
    )


@main.command()
@click.argument("csvfile", type=click.Path(exists=True))
@click.option("--outdir", type=click.Path(), default=os.getcwd(), show_default=True)
@click.option("--save_figures/--no-save_figures", default=False, show_default=True)
@click.option("--n_mels", type=int, default=20, show_default=True)
@click.option("--n_coeff", type=int, default=20, show_default=True)
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
@click.option("--opensmile", nargs=2, default=["eGeMAPSv02", "Functionals"], show_default=True)
def batchconvert(
    csvfile,
    outdir,
    save_figures,
    n_mels,
    n_coeff,
    compute_deltas,
    plugin,
    cache,
    dataset,
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
        compute_deltas=compute_deltas,
        cache_dir=Path(cache).absolute(),
        save_figures=save_figures,
        opensmile_feature_set=opensmile[0],
        opensmile_feature_level=opensmile[1],
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
    with pydra.Submitter(plugin=plugin[0], **plugin_args) as sub:
        sub(runnable=featurize_task)
    os.chdir(cwd)
    results = featurize_task.result()
    Path(outdir).mkdir(exist_ok=True, parents=True)
    for val in results:
        shutil.copy(val.output.features[1], Path(outdir))
        if save_figures:
            shutil.copy(val.output.features[2], Path(outdir))
    if dataset:

        def gen():
            for val in results:
                yield torch.load(val.output.features[1])

        to_hf_dataset(gen, Path(outdir) / "hf_dataset")


@main.command()
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2", type=click.Path(exists=True))
@click.argument("model", type=str)
@click.option("--device", type=str, default=None, show_default=True)
def verify(file1, file2, model, device):
    score, prediction = verify_speaker_from_files(file1, file2, model=model, device=device)
    print(f"Score: {float(score):.2f} Prediction: {bool(prediction)}")


if TTS is not None:

    @main.command()
    @click.argument("source_file", type=click.Path(exists=True))
    @click.argument("target_voice_file", type=click.Path(exists=True))
    @click.argument("output_file", type=click.Path())
    @click.option(
        "--model_name",
        type=str,
        default="voice_conversion_models/multilingual/vctk/freevc24",
        show_default=True,
    )
    @click.option(
        "--device", type=str, default=None, show_default=True, help="Device to use for inference."
    )
    @click.option("--progress_bar", type=bool, default=True, show_default=True)
    def convert_voice(
        source_file, target_voice_file, output_file, model_name, device, progress_bar
    ):
        """
        Converts the voice in the source_file to match the voice in the target_voice_file,
        and saves the output to output_file.
        """
        vc = VoiceConversion(model_name=model_name, progress_bar=progress_bar, device=device)
        vc.convert_voice(source_file, target_voice_file, output_file)
        print(f"Conversion complete. Output saved to: {output_file}")


@main.command()
@click.argument("audio_file", type=click.Path(exists=True))
@click.option("--model_id", type=str, default="openai/whisper-tiny", show_default=True)
@click.option("--max_new_tokens", type=int, default=128, show_default=True)
@click.option("--chunk_length_s", type=int, default=30, show_default=True)
@click.option("--batch_size", type=int, default=16, show_default=True)
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
    audiofiles = glob(f"{input_dir}/**/*.wav", recursive=True)

    with open(out_file, "w") as f:

        # using csv.writer method from CSV package
        write = csv.writer(f)

        for item in audiofiles:
            write.writerow([Path(item).absolute().as_posix()])

    print(f"csv of audiofiles generated at: {out_file}")
