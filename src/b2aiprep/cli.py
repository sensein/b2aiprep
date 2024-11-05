import csv
import logging
import os
import shutil
from glob import glob
from pathlib import Path

import click
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
from senselab.utils.data_structures.model import HFModel
from streamlit import config as _config
from streamlit.web.bootstrap import run

from b2aiprep.commands import (
    dashboard,
    redcap2bids,
    # extract_praat,
    prepbidslikedata,
    validate,
    gensynthtabdata,
    convert,
    batchconvert,
    verify,
    transcribe,
    createbatchcsv,
)

@click.group()
@click.option(
    '--log-level',
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    case_sensitive=False,
    help="Set the log level for the CLI."
)
@click.pass_context
def cli(ctx, log_level):
    ctx.ensure_object(dict)
    ctx.obj["LOG_LEVEL"] = log_level

    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    _LOGGER = logging.getLogger(__name__)

cli.add_command(dashboard)
cli.add_command(redcap2bids)
cli.add_command(prepbidslikedata)
cli.add_command(validate)
cli.add_command(gensynthtabdata)
cli.add_command(convert)
cli.add_command(batchconvert)
cli.add_command(verify)
cli.add_command(transcribe)
cli.add_command(createbatchcsv)

if __name__ == "__main__":
    # include main to enable python debugging
    cli()  # pylint: disable=no-value-for-parameter
