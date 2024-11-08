import json
import logging
import os
import pickle
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from sdv.evaluation.single_table import (
    evaluate_quality,
    get_column_plot,
    run_diagnostic,
)
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

_logger = logging.getLogger(__name__)
logging.basicConfig(filename=None, level=logging.INFO)
logging.getLogger("sdv").setLevel(logging.WARNING)


def fit_synthesizer(
    source_data_csv_path: Path, refit: bool = False, synthesizer_path: Optional[Path] = None
) -> CTGANSynthesizer:
    """
    Fit a synthesizer to the given source data or load an existing synthesizer if available.

    Args:
        source_data_csv_path (Path): Path to the source data CSV file.
        refit (bool): Whether to refit the synthesizer even if a saved synthesizer exists. Defaults to False.
        synthesizer_path (Optional[Path]): Path to save/load the synthesizer model. Defaults to None.

    Returns:
        synthesizer (CTGANSynthesizer): The fitted synthesizer model.
    """
    data = pd.read_csv(source_data_csv_path)

    # Detect metadata from the data
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=data)
    _logger.info("Metadata detected from source data CSV.")

    # Load the synthesizer if it exists and refit == False.
    synthesizer = None
    if os.path.exists(synthesizer_path) and not refit:
        with open(synthesizer_path, "rb") as f:
            synthesizer = pickle.load(f)
        _logger.info("Loaded synthesizer from file.")
    else:  # Fit a new synthesizer if no valid path or file exists
        _logger.info("Fitting new synthesizer...")
        synthesizer = CTGANSynthesizer(metadata)
        synthesizer.fit(data=data)
        synthesizer.save(synthesizer_path)
        _logger.info("Synthesizer fit and saved.")
    return synthesizer


def generate_tabular_data(
    n_synthetic_rows: int = 100,
    synthetic_data_path: Optional[Path] = None,
    synthesizer_path: Path = None,
) -> pd.DataFrame:
    """
    Generate synthetic tabular data using the given synthesizer.

    Args:
        n_synthetic_rows (int): Number of synthetic rows to generate. Defaults to 100.
        synthetic_data_path (Optional[Path]): Path to save the generated synthetic data as a CSV. Defaults to None.
        synthesizer_path (Path): Path to load the synthesizer model.

    Returns:
        pd.DataFrame: The generated synthetic data.
    """
    with open(synthesizer_path, "rb") as f:
        synthesizer = pickle.load(f)

    _logger.info(f"Sampling {n_synthetic_rows} rows with the synthesizer...")
    synthetic_data = synthesizer.sample(num_rows=n_synthetic_rows)
    _logger.info(f"{n_synthetic_rows} rows sampled.")

    if synthetic_data_path:
        if os.path.exists(synthetic_data_path):
            synthetic_data.to_csv(synthetic_data_path, index=False)
            _logger.info(f"Data saved to {synthetic_data_path}.")
        else:
            raise FileNotFoundError(f"Provided path {synthetic_data_path} does not exist.")

    return synthetic_data


def run_diagnostics(
    synthetic_data_path: Union[str, Path],
    real_data_path: Union[str, Path],
    diagnostic_report_path: Optional[Path] = None,
) -> dict:
    """
    Run diagnostics comparing synthetic and real data, and save the results in a report.

    Args:
        synthetic_data_path (Union[str, Path]): Path to the synthetic data CSV file.
        real_data_path (Union[str, Path]): Path to the real data CSV file.
        diagnostic_report_path (Optional[Path]): Path to save the diagnostic report as a JSON file.

    Returns:
        dict: The diagnostic results.
    """
    real_data = pd.read_csv(real_data_path)
    synthetic_data = pd.read_csv(synthetic_data_path)

    # Generate metadata from real data
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_data)

    # Perform basic validity checks
    diagnostic = run_diagnostic(real_data, synthetic_data, metadata)
    _logger.info("Basic validity checks completed.")

    # Save diagnostic info to JSON
    diagnostic_info_dict = diagnostic.get_info()
    diagnostic_info_dict["score"] = diagnostic.get_score()
    diagnostic_info_json = json.dumps(diagnostic_info_dict, ensure_ascii=False, indent=4)
    with open(diagnostic_report_path, "w", encoding="utf-8") as report_file:
        report_file.write(diagnostic_info_json)

    _logger.info(f"Diagnostic report saved to {diagnostic_report_path}.")
    return diagnostic_info_dict


def evaluate_data(
    synthetic_data_path: Union[str, Path],
    real_data_path: Union[str, Path],
    evaluation_report_path: Optional[Path] = None,
) -> dict:
    """
    Evaluate the statistical similarity between synthetic and real data, and save the evaluation report.

    Args:
        synthetic_data_path (Union[str, Path]): Path to the synthetic data CSV file.
        real_data_path (Union[str, Path]): Path to the real data CSV file.
        evaluation_report_path (Optional[Path]): Path to save the evaluation report as a JSON file.

    Returns:
        dict: The evaluation results.
    """
    real_data = pd.read_csv(real_data_path)
    synthetic_data = pd.read_csv(synthetic_data_path)

    # Generate metadata from real data
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_data)

    # Evaluate the quality of the synthetic data
    evaluation = evaluate_quality(real_data, synthetic_data, metadata)

    # Save evaluation info to JSON
    evaluation_info_dict = evaluation.get_info()
    evaluation_info_dict["score"] = evaluation.get_score()
    evaluation_info_json = json.dumps(evaluation_info_dict, ensure_ascii=False, indent=4)
    with open(evaluation_report_path, "w", encoding="utf-8") as report_file:
        report_file.write(evaluation_info_json)

    _logger.info(f"Evaluation report saved to {evaluation_report_path}.")
    return evaluation_info_dict


def get_column_plots(
    synthetic_data_csv_path: Union[str, Path],
    real_data_csv_path: Union[str, Path],
    save_directory: Optional[Path] = None,
) -> List:
    """
    Generate and optionally save column plots comparing the distributions of real and synthetic data.

    Args:
        synthetic_data_csv_path (Union[str, Path]): Path to the synthetic data CSV file.
        real_data_csv_path (Union[str, Path]): Path to the real data CSV file.
        save_directory (Optional[Path]): Directory to save the generated plots as image files.

    Returns:
        List: A list of plotly figures for each column.
    """
    real_data = pd.read_csv(real_data_csv_path)
    synthetic_data = pd.read_csv(synthetic_data_csv_path)

    # Generate metadata from real data
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_data)

    column_plots = []

    # Iterate over each column in the real data
    for column_name in real_data.columns:
        fig = get_column_plot(
            real_data=real_data,
            synthetic_data=synthetic_data,
            column_name=column_name,
            metadata=metadata,
        )

        column_plots.append(fig)

        if save_directory:  # Save the plot to the specified directory
            file_path = os.path.join(
                save_directory, f"{column_name}_column_real_and_synthetic_distribution_plot.png"
            )
            fig.write_image(file_path)
            _logger.info(f"Plot saved for column '{column_name}' at: {file_path}")
    return column_plots
