import os
import typing as ty
from pathlib import Path
import csv
from glob import glob

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
def convert(filename, subject, task, outdir, save_figures, 
            n_mels, n_coeff, compute_deltas, opensmile_feature_set, opensmile_feature_level):
    to_features(
        filename,
        subject,
        task,
        outdir=Path(outdir),
        save_figures=save_figures,
        n_mels=n_mels,
        n_coeff=n_coeff,
        compute_deltas=compute_deltas,
        opensmile_feature_set=opensmile_feature_set,
        opensmile_feature_level=opensmile_feature_level
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
    
    
    featurize_pdt = pydratask(annotate({"return": {"features": ty.Any}})(to_features))
    Path(outdir).mkdir(exist_ok=True, parents=True)
    featurize_task = featurize_pdt(
        outdir=Path(outdir).absolute(),
        n_mels=n_mels,
        n_coeff=n_coeff,
        compute_deltas=compute_deltas,
        cache_dir=Path(cache).absolute(),
    )
    
    with open(csvfile, "r") as f:
        reader = csv.DictReader(f)
        num_cols = len(reader.fieldnames)
        lines = [line.strip() for line in f.readlines()]

    #parse csv file differently if it is one column 'filename'
    #or three column 'filename','subject','task'
    if num_cols == 1:
        filenames = []
        for line in lines:
            filename = line
            filenames.append(Path(filename).absolute().as_posix())
        featurize_task.split(
        splitter=("filename"),
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


@main.command()
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2", type=click.Path(exists=True))
@click.argument("model", type=str)
@click.option("--device", type=str, default=None, show_default=True)
def verify(file1, file2, model, device=None):
    score, prediction = verify_speaker_from_files(file1, file2, model=model)
    print(f"Score: {float(score):.2f} Prediction: {bool(prediction)}")

    
@main.command()
@click.argument("input_dir", type=str)
@click.argument("out_file", type=str)
def createbatchcsv(input_dir, out_file):
    
    #input_dir is the top level directory of the b2ai Production directory from Wasabi
    #it is expected to contain subfolders with each institution e.g. MIT, UCF, etc.
    
    #out_file is where a csv file will be saved and should be in the format 'path/name/csv'
    input_dir = Path(input_dir)
    audiofiles = glob(f'{input_dir}/*/*.wav')

    with open(out_file, 'w') as f:

        # using csv.writer method from CSV package
        write = csv.writer(f)

        for item in audiofiles:
            write.writerow([item])
            
    print(f"csv of audiofiles generated at: {out_file}")
