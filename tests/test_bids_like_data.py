from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
from pydantic import BaseModel

from b2aiprep.prepare.bids_like_data import (
    _df_to_dict,
    create_file_dir,
    get_df_of_repeat_instrument,
    get_instrument_for_name,
    get_recordings_for_acoustic_task,
    load_redcap_csv,
    questionnaire_mapping,
    redcap_to_bids,
    update_redcap_df_column_names,
    write_pydantic_model_to_bids_file,
)
from b2aiprep.prepare.constants import AUDIO_TASKS, RepeatInstrument
from b2aiprep.prepare.prepare import initialize_data_directory


@patch("pandas.read_csv")
def test_load_redcap_csv(mock_read_csv):
    # Mocking the DataFrame returned by read_csv
    mock_data = pd.DataFrame(
        {"record_id": [1, 2], "redcap_repeat_instrument": ["instrument_1", "instrument_2"]}
    )
    mock_read_csv.return_value = mock_data

    df = load_redcap_csv("dummy_path.csv")
    assert df is not None
    assert "record_id" in df.columns
    assert "redcap_repeat_instrument" in df.columns


def test_update_redcap_df_column_names():
    # Path to the actual 'column_mapping.json' file
    real_file_path = Path("src/b2aiprep/prepare/resources/column_mapping.json")

    # Check if the file exists
    assert real_file_path.exists(), f"File not found: {real_file_path}"

    # Create a DataFrame using the "verbose" column names (the right side of your JSON mapping)
    mock_data = pd.DataFrame(
        {
            "Record ID": [1, 2],
            "Repeat Instrument": ["instrument_1", "instrument_2"],
            "Repeat Instance": [1, 1],
            "Language": ["English", "Spanish"],
            "Consent Status": ["Yes", "No"],
            "Is Feasibility Participant?": ["Yes", "No"],
            "Enrollment Institution": ["MIT", "Harvard"],
            "Age": [34, 28],
        }
    )

    # Call the function, which will now use the real file system
    updated_df = update_redcap_df_column_names(mock_data)

    # Ensure the column names have been updated according to 'column_mapping.json'
    assert "record_id" in updated_df.columns
    assert "redcap_repeat_instrument" in updated_df.columns
    assert "redcap_repeat_instance" in updated_df.columns
    assert "selected_language" in updated_df.columns
    assert "consent_status" in updated_df.columns
    assert "is_feasibility_participant" in updated_df.columns
    assert "enrollment_institution" in updated_df.columns
    assert "age" in updated_df.columns

    # Ensure that the verbose column names are no longer present
    assert "Record ID" not in updated_df.columns
    assert "Repeat Instrument" not in updated_df.columns


def test_get_df_of_repeat_instrument():
    mock_data = pd.DataFrame(
        {
            "redcap_repeat_instrument": ["instrument_1", "instrument_2"],
            "column_1": [1, 2],
            "column_2": [3, 4],
        }
    )
    mock_instrument = MagicMock()
    mock_instrument.get_columns.return_value = ["column_1", "column_2"]
    mock_instrument.text = "instrument_1"

    filtered_df = get_df_of_repeat_instrument(mock_data, mock_instrument)
    assert len(filtered_df) == 1
    assert "column_1" in filtered_df.columns


def test_get_recordings_for_acoustic_task():
    # Use actual tasks from the AUDIO_TASKS list
    task_name = AUDIO_TASKS[0]  # Pick the first valid task from the list for the test

    # Create mock data for acoustic tasks and recordings based on the real task names
    mock_acoustic_tasks = pd.DataFrame(
        {
            "acoustic_task_name": [task_name, AUDIO_TASKS[1]],  # Use real AUDIO_TASKS
            "acoustic_task_id": [1, 2],
            "redcap_repeat_instrument": ["Acoustic Task", "Acoustic Task"],
        }
    )

    # Access the Instrument instance stored in the RepeatInstrument enum member
    acoustic_task_instrument = (
        RepeatInstrument.ACOUSTIC_TASK.value
    )  # This is an Instrument instance

    # Manually call get_df_of_repeat_instrument() to mimic what get_recordings_for_acoustic_task() does internally
    acoustic_tasks_df = get_df_of_repeat_instrument(mock_acoustic_tasks, acoustic_task_instrument)

    acoustic_tasks_df["redcap_repeat_instrument"] = "Acoustic Task"

    # Now, pass the processed DataFrame and task name to the function
    get_recordings_for_acoustic_task(acoustic_tasks_df, task_name)


@patch("builtins.open", new_callable=mock_open)
def test_write_pydantic_model_to_bids_file(mock_open):
    mock_output_path = MagicMock()
    mock_data = MagicMock(BaseModel)
    mock_data.json.return_value = '{"test": "data"}'

    write_pydantic_model_to_bids_file(
        output_path=mock_output_path,
        data=mock_data,
        schema_name="schema_name",
        subject_id="sub_01",
        session_id="ses_01",
        task_name="task_name",
    )
    mock_open.assert_called_once()


def test_df_to_dict():
    mock_data = pd.DataFrame({"index_col": ["A", "B", "C"], "data_col": [1, 2, 3]})

    result = _df_to_dict(mock_data, "index_col")
    assert result["A"]["data_col"] == 1


@patch("pathlib.Path.mkdir")
def test_create_file_dir(mock_mkdir):
    create_file_dir("sub_01", "ses_01")
    mock_mkdir.assert_called()


@patch("builtins.open", new_callable=mock_open)
@patch("json.loads")
def test_questionnaire_mapping(mock_json_loads, mock_open):
    mock_json_loads.return_value = {"questionnaire_1": {"key": "value"}}
    result = questionnaire_mapping("questionnaire_1")
    assert result["key"] == "value"


def test_get_instrument_for_name():
    # Use an actual instrument name from RepeatInstrument
    instrument_name = "participant"

    # Call the function with the actual instrument name
    instrument = get_instrument_for_name(instrument_name)

    # Assert that the returned instrument matches the expected one
    assert instrument == RepeatInstrument.PARTICIPANT.value


def test_redcap_to_bids():
    project_root = Path(__file__).parent.parent
    csv_file_path = project_root / "data/sdv_redcap_synthetic_data_1000_rows.csv"

    # Check if the file exists before proceeding
    if not csv_file_path.exists():
        raise FileNotFoundError(f"CSV file not found at: {csv_file_path}")

    # Use TemporaryDirectory for the output directory
    with TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "bids_output"
        # output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        initialize_data_directory(output_dir)
        # Call the actual redcap_to_bids function with the real CSV and the temporary output directory
        redcap_to_bids(csv_file_path, output_dir)

        # Check if the expected output files exist in the temporary directory
        if not any(output_dir.iterdir()):
            raise AssertionError("No output was created in the output directory")
