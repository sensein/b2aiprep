

import json
import os
import logging
import pickle
import uuid

import pandas as pd
import sdv
from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
from sdv.evaluation.single_table import evaluate_quality, get_column_plot
from sdv.metadata import SingleTableMetadata

from pathlib import Path

#TODO CLI command
#TODO audio with correct naming
#TODO tests

_logger = logging.getLogger(__name__)
logging.basicConfig(filename=None, level=logging.INFO)
logging.getLogger('sdv').setLevel(logging.WARNING)





def evaluate_synthetic_data(
    synthetic_data_csv_path,
    real_data_csv_path
):
    real_data = pd.read_csv(real_data_csv_path)
    synthetic_data = pd.read_csv(synthetic_data_csv_path)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_data)
    quality_report = evaluate_quality(
        real_data,
        synthetic_data,
        metadata
    )
    print(quality_report)


def generate_column_plot(
    synthetic_data_csv_path,
    real_data_csv_path,
    save_path = None 
):
    real_data = pd.read_csv(real_data_csv_path)
    synthetic_data = pd.read_csv(synthetic_data_csv_path)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_data)

    print(real_data.columns[:1])
    for column_name in real_data.columns:

        fig = get_column_plot(
            real_data=real_data,
            synthetic_data=synthetic_data,
            column_name=column_name,
            metadata=metadata
        )

        fig.show()
    
    if save_path:
        pass #TODO implement saving



def generate_synthetic_tabular_data(
        source_data_csv_path: Path,
        synthetic_data_path: Path = None,
        n_synthetic_rows: int = 100,
        synthesizer_path: Path = None,
):
    """Once the synthesizer is fit, seems to take roughly 5 seconds per 100 rows."""
    data = pd.read_csv(source_data_csv_path)
    data["Record ID"] = [str(uuid.uuid4()) for _ in range(len(data))]
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
        synthesizer = CTGANSynthesizer(metadata)
        synthesizer.fit(data=data)
        _logger.info("Synthesizer fitted.")
        synthesizer.save(synthesizer_path)

    _logger.info(f"Sampling {n_synthetic_rows} rows with the synthesizer...")
    synthetic_data = synthesizer.sample(num_rows=n_synthetic_rows)
    _logger.info(f"{n_synthetic_rows} rows sampled.")

    #TODO Quality report


    synthetic_data.to_csv(synthetic_data_path, index=False)

