import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from importlib import resources
import json
import pandas as pd

import pytest

from b2aiprep.prepare.data_validation import (
    validate_derivatives,
    validate_participant_data,
    Validator,
)


@pytest.fixture
def setup_temp_files(tmp_path):
    """Fixture to set up temporary directories and files for testing."""
    # Paths to required data files
    field_requirements = json.load(
        (Path(__file__).parent.parent / "data" / "field_requirements.json").open()
    )

    # Synthetic record
    synthetic_record = {
        "record_id": "test_0001",
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
        "tough_to_work": "Not difficult at all",
    }
    df = pd.DataFrame([synthetic_record])

    data_dictionary = json.load(
        (
            resources.files("b2aiprep")
            .joinpath("prepare", "resources", "b2ai-data-bids-like-template", "participants.json")
            .open()
        )
    )

    # Save sythetic df into temp directory
    redcap_csv_path = os.path.join(tmp_path, "sdv_redcap_synthetic_data.csv")
    df.to_csv(redcap_csv_path, index=False)

    validator = Validator(data_dictionary, "test_id_1", field_requirements)

    yield redcap_csv_path, data_dictionary, field_requirements, validator


def test_validate_data_cli(setup_temp_files):
    """Test the 'b2aiprep-cli prepbidslikedata' command using subprocess."""
    redcap_csv_path, data_dictionary, field_requirements, validator = setup_temp_files

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
    redcap_csv_path, data_dictionary, field_requirements, validator = setup_temp_files
    assert validate_derivatives(redcap_csv_path) is None


def test_validate_participant_data():
    synthetic_record = {
        "record_id": "test_0001",
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
        "tough_to_work": "Not difficult at all",
    }
    assert validate_participant_data(synthetic_record) is None


def test_validator_validate_choices(setup_temp_files):
    redcap_csv_path, data_dictionary, field_requirements, validator = setup_temp_files
    field = "nervous_anxious"
    value = "Not at all"

    assert validator.validate_choices(field=field, value=value) == True


def test_validator_validate_choices_wrong_choice(setup_temp_files):
    redcap_csv_path, data_dictionary, field_requirements, validator = setup_temp_files
    field = "nervous_anxious"
    value = "Not at all, but sometimes"

    assert validator.validate_choices(field=field, value=value) == False


def test_clean_number(setup_temp_files):
    redcap_csv_path, data_dictionary, field_requirements, validator = setup_temp_files

    assert validator.clean("test") == "test"
    assert validator.clean("'field'") == "field"
    assert validator.clean(5.0) == 5
    assert validator.clean(3.14) == 3.14


def test_get_choices(setup_temp_files):
    redcap_csv_path, data_dictionary, field_requirements, validator = setup_temp_files

    choices = [
        {"name": {"en": "Not at all"}, "value": 0},
        {"name": {"en": "Several days"}, "value": 1},
        {"name": {"en": "More than half the days"}, "value": 2},
        {"name": {"en": "Nearly every day"}, "value": 3},
    ]

    actual = validator.get_choices(choices=choices)
    expected = {"Not at all", "Several days", "More than half the days", "Nearly every day"}

    assert actual == expected


def test_validator_validate_range(setup_temp_files):
    redcap_csv_path, data_dictionary, field_requirements, validator = setup_temp_files

    assert validator.validate_range("height", 171) == True  # assumes metric


def test_validator_validate_range_incorrect_range(setup_temp_files):
    redcap_csv_path, data_dictionary, field_requirements, validator = setup_temp_files
    
    assert validator.validate_range("height", 99999) == False  # assumes metric
