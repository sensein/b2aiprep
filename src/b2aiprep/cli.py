import os
from pathlib import Path

import click

from .process import to_features, verify_speaker_from_files


@click.group()
def main():
    pass


@main.command()
@click.argument("filename", type=click.Path(exists=True))
@click.argument("subject", type=str)
@click.argument("task", type=str)
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
@click.argument("file1", type=click.Path(exists=True))
@click.argument("file2", type=click.Path(exists=True))
@click.argument("model", type=str)
@click.option("--device", type=str, default=None, show_default=True)
def verify(file1, file2, model, device=None):
    score, prediction = verify_speaker_from_files(file1, file2, model=model)
    print(f"Score: {float(score):.2f} Prediction: {bool(prediction)}")
