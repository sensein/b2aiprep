import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import pandas as pd
import pytest

from b2aiprep.prepare.data_validation import (
    validate_derivatives,
    validate_participant_data,
    clean_number,
    get_choices
)


@pytest.fixture
def setup_temp_files():
    """Fixture to set up temporary directories and files for testing."""
    # Paths to required data files
    project_root = Path(__file__).parent.parent

    # Synthetic record
    synthetic_record = {"record_id": "test_0001",
                        "height": 170,
                        "weight": 80,
                        "unit": "Metric",
                        "nervous_anxious": "Not at all",
                        "cant_control_worry": "Not at all",
                        "worry_too_much": "Several days",
                        "trouble_relaxing": "Several days",
                        "hard_to_sit_still": "Several days",
                        "easily_agitated": "Several days",
                        "afraid_of_things": "Several days",
                        "tough_to_work": "Not difficult at all"}
    df = pd.DataFrame([synthetic_record])

    # Save sythetic df into temp directory
    redcap_csv_path = os.path.join(tempfile.gettempdir(), "sdv_redcap_synthetic_data.csv")
    df.to_csv(redcap_csv_path, index=False)

    yield redcap_csv_path


def test_validate_data_cli(setup_temp_files):
    """Test the 'b2aiprep-cli prepbidslikedata' command using subprocess."""
    redcap_csv_path = setup_temp_files

    # Define the CLI command
    command = [
        "b2aiprep-cli",
        "validate-data",
        redcap_csv_path,

    ]
    # Run the CLI command
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if the command was successful
    assert result.returncode == 0, f"CLI command failed: {result.stderr}"


def test_validate_derivatives(setup_temp_files):
    redcap_csv_path = setup_temp_files
    assert validate_derivatives(redcap_csv_path) is None


def test_validate_participant_data():
    synthetic_record = {"record_id": "test_0001",
                        "height": 170,
                        "weight": 80,
                        "unit": "Metric",
                        "nervous_anxious": "Not at all",
                        "cant_control_worry": "Not at all",
                        "worry_too_much": "Several days",
                        "trouble_relaxing": "Several days",
                        "hard_to_sit_still": "Several days",
                        "easily_agitated": "Several days",
                        "afraid_of_things": "Several days",
                        "tough_to_work": "Not difficult at all"}
    assert validate_participant_data(synthetic_record) is None


def test_clean_number():
    assert clean_number("5") == 5
    assert clean_number("test") == "test"
    assert clean_number(5.0) == 5
    assert clean_number(3.14) == 3.14


def test_get_choices():
    choices = [
        {
            "name": {
                "en": "Not at all"
            },
            "value": 0
        },
        {
            "name": {
                "en": "Several days"
            },
            "value": 1
        },
        {
            "name": {
                "en": "More than half the days"
            },
            "value": 2
        },
        {
            "name": {
                "en": "Nearly every day"
            },
            "value": 3
        }
    ]

    actual = get_choices(choices=choices)
    expected = {"Not at all","Several days", "More than half the days", "Nearly every day"}

    assert actual == expected
