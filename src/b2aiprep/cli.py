import os
import typing as ty
from pathlib import Path

import click
import pydra
from pydra.mark import annotate
from pydra.mark import task as pydratask

from .process import to_features, verify_speaker_from_files


@click.group()
def main():
    pass


@main.command()
@click.argument("filename", type=click.Path(exists=True))
@click.argument("subject", type=str, default="unknown")
@click.argument("task", type=str, default="unknown")
@click.option("--outdir", type=click.Path(), default=os.getcwd(), show_default=True)
@click.option("--n_mels", type=int, default=20, show_default=True)
@click.option("--n_coeff", type=int, default=20, show_default=True)
@click.option("--compute_deltas/--no-compute_deltas", default=True, show_default=True)
def convert(filename, subject, task, outdir, n_mels, n_coeff, compute_deltas):
    to_features(
        filename,
        subject,
        task,
        outdir=Path(outdir),
        n_mels=n_mels,
        n_coeff=n_coeff,
        compute_deltas=compute_deltas,
    )


@main.command()
@click.argument("csvfile", type=click.Path(exists=True))
@click.option("--outdir", type=click.Path(), default=os.getcwd(), show_default=True)
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
def batchconvert(csvfile, outdir, n_mels, n_coeff, compute_deltas, plugin, cache):
    plugin_args = dict()
    for item in plugin[1].split():
        key, value = item.split("=")
        if plugin[0] == "cf" and key == "n_procs":
            value = int(value)
        plugin_args[key] = value
    with open(csvfile, "r") as f:
        lines = [line.strip() for line in f.readlines()]
    filenames = []
    subjects = []
    tasks = []
    for line in lines:
        filename, subject, task = line.split(",")
        filenames.append(Path(filename).absolute().as_posix())
        subjects.append(subject)
        tasks.append(task)
    featurize_pdt = pydratask(annotate({"return": {"features": ty.Any}})(to_features))
    Path(outdir).mkdir(exist_ok=True, parents=True)
    featurize_task = featurize_pdt(
        outdir=Path(outdir).absolute(),
        n_mels=n_mels,
        n_coeff=n_coeff,
        compute_deltas=compute_deltas,
        cache_dir=Path(cache).absolute(),
    )
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


@main.command()
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2", type=click.Path(exists=True))
@click.argument("model", type=str)
@click.option("--device", type=str, default=None, show_default=True)
def verify(file1, file2, model, device=None):
    score, prediction = verify_speaker_from_files(file1, file2, model=model)
    print(f"Score: {float(score):.2f} Prediction: {bool(prediction)}")
