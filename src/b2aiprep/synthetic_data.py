

import json
import os
import logging
import pickle

import pandas as pd
import sdv
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

from pathlib import Path

#TODO CLI command
#TODO tests

_logger = logging.getLogger(__name__)
logging.basicConfig(filename=None, level=logging.INFO)
logging.getLogger('sdv').setLevel(logging.WARNING)

def generate_synthetic_redcap_data(
        source_data_csv_path: Path,
        synthetic_data_path: Path = None,
        n_synthetic_rows: int = 100,
        synthesizer_path: Path = None,
):
    data = pd.read_csv(source_data_csv_path)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=data)
    _logger.info("Metadata detected from source data CSV.")

    # Get the synthesizer.
    synthesizer = None
    if synthesizer_path and os.path.exists(synthesizer_path):
        with open(synthesizer_path, 'rb') as f:
            synthesizer = pickle.load(f)
    else:
        _logger.info("Fitting synthesizer...")
        synthesizer = GaussianCopulaSynthesizer(metadata)
        synthesizer.fit(data=data)
        _logger.info("Synthesizer fitted.")
        synthesizer.save(synthesizer_path)

    _logger.info(f"Sampling {n_synthetic_rows} rows with the synthesizer...")
    synthetic_data = synthesizer.sample(num_rows=n_synthetic_rows)
    synthetic_data.to_csv(synthetic_data_path, index=False)
    _logger.info(f"{n_synthetic_rows} rows sampled.")
