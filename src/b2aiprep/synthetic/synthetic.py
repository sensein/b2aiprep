import logging
import os
import pickle
import uuid
from pathlib import Path

import pandas as pd
from sdv.evaluation.single_table import evaluate_quality, get_column_plot
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

_logger = logging.getLogger(__name__)
logging.basicConfig(filename=None, level=logging.INFO)
logging.getLogger("sdv").setLevel(logging.WARNING)


def evaluate_data(synthetic_data_csv_path, real_data_csv_path):
    real_data = pd.read_csv(real_data_csv_path)
    synthetic_data = pd.read_csv(synthetic_data_csv_path)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_data)
    quality_report = evaluate_quality(real_data, synthetic_data, metadata)
    print(quality_report)


def generate_column_plot(synthetic_data_csv_path, real_data_csv_path, save_path=None):
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
            metadata=metadata,
        )

        fig.show()

    if save_path:
        pass  # TODO implement saving


def fit_synthesizer(source_data_csv_path: Path, synthesizer_path: Path = None):
    data = pd.read_csv(source_data_csv_path)

    # Add unique Record IDs
    data["Record ID"] = [str(uuid.uuid4()) for _ in range(len(data))]

    # Detect metadata from the data
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=data)
    _logger.info("Metadata detected from source data CSV.")

    # Initialize synthesizer
    synthesizer = None
    # Load the synthesizer if it exists.
    if synthesizer_path and os.path.exists(synthesizer_path):
        with open(synthesizer_path, "rb") as f:
            synthesizer = pickle.load(f)
        _logger.info("Loaded synthesizer from file.")
    else:
        # Fit a new synthesizer if no valid path or file exists
        _logger.info("Fitting new synthesizer...")
        synthesizer = CTGANSynthesizer(metadata)
        _logger.info(f"synthesizer path: {synthesizer_path}")
        synthesizer.save(synthesizer_path)  # ensures valid path
        synthesizer.fit(data=data)
        synthesizer.save(synthesizer_path)
        _logger.info("Synthesizer fitted.")
    return synthesizer


source_data_csv_path = "/Users/isaacbevers/sensein/b2ai-wrapper/b2ai-data/bridge2ai-voice-corpus-3/bridge2ai-voice-corpus-3-replaced-cols-combined.csv"
synthetic_data_path = (
    "b2aiprep/data/corpus-3/sdv_redcap_synthetic_data_100_rows_corpus_3_combine_first.csv"
)
n_synthetic_rows = 100
synthesizer_path = "b2aiprep/models/synthesizers/corpus_3_csv_CTGAN_synthesizer_combined.pkl"

fit_synthesizer(source_data_csv_path, synthesizer_path)


def generate_tabular_data(
    source_data_csv_path: Path,
    synthetic_data_path: Path = None,
    n_synthetic_rows: int = 100,
    synthesizer_path: Path = None,
):
    """Once the synthesizer is fit, seems to take roughly 5 seconds per 100 rows."""
    synthesizer_dir = os.path.dirname(synthesizer_path)
    if not os.path.exists(synthesizer_dir):
        _logger.info(
            f"Path does not exist: {synthesizer_dir}. \
                        Path created."
        )
        os.makedirs(synthesizer_dir)

    synthetic_data_dir = os.path.dirname(synthetic_data_path)
    if not os.path.exists(synthetic_data_dir):
        _logger.info(
            f"Path does not exist: {synthetic_data_dir}. \
                        Path created."
        )
        os.makedirs(synthetic_data_dir)

    # _logger.info(f"Sampling {n_synthetic_rows} rows with the synthesizer...")
    # synthetic_data = synthesizer.sample(num_rows=n_synthetic_rows)
    # _logger.info(f"{n_synthetic_rows} rows sampled.")

    # # TODO Quality report
    # synthetic_data.to_csv(synthetic_data_path, index=False)


# source_data_csv_path = "/data/test_tiny.csv"
# synthetic_data_path = os.getcwd() + "/../data/corpus-3/test_tiny_synthetic.csv"
# n_synthetic_rows = 3
# synthesizer_path = os.getcwd() + "/../models/synthesizers/corpus_3_csv_CTGAN_synthesizer_combined.pkl"


# generate_tabular_data(
#     source_data_csv_path=source_data_csv_path,
#     synthetic_data_path=synthetic_data_path,
#     n_synthetic_rows=n_synthetic_rows,
#     synthesizer_path=synthesizer_path)
