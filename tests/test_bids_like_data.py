import json
import os
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
from pydantic import BaseModel

from b2aiprep.prepare.bids import (
    _df_to_dict,
    construct_all_tsvs_from_jsons,
    construct_tsv_from_json,
    create_file_dir,
    get_df_of_repeat_instrument,
    get_instrument_for_name,
    get_recordings_for_acoustic_task,
    load_redcap_csv,
    questionnaire_mapping,
    redcap_to_bids,
    write_pydantic_model_to_bids_file,
)
from b2aiprep.prepare.constants import AUDIO_TASKS, RepeatInstrument
from b2aiprep.prepare.utils import initialize_data_directory


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
        initialize_data_directory(output_dir)
        # Call the actual redcap_to_bids function with the real CSV and the temporary output directory
        redcap_to_bids(csv_file_path, output_dir)

        # Check if the expected output files exist in the temporary directory
        if not any(output_dir.iterdir()):
            raise AssertionError("No output was created in the output directory")


def test_construct_tsv_from_json():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a sample DataFrame
        data = {
            "record_id": [1, 2, 3, 4],
            "element_1": ["A", "B", "C", "D"],
            "element_2": [10, 20, 30, 40],
        }
        df = pd.DataFrame(data)

        # Create a sample JSON file
        json_data = {
            "schema_name": {
                "description": "Description of the schema.",
                "data_elements": {
                    "record_id": {},
                    "element_1": {
                        "description": "Description of the first element.",
                        "question": {"en": "Question text for the first element."},
                        "datatype": ["xsd:string"],
                    },
                    "element_2": {
                        "description": "Description of the second element.",
                        "question": {"en": "Question text for the second element."},
                        "datatype": ["xsd:decimal"],
                    },
                },
            }
        }
        json_path = os.path.join(temp_dir, "sample.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f)

        # Run the function
        construct_tsv_from_json(df, json_path, temp_dir)

        # Check if the TSV file is created
        tsv_path = os.path.join(temp_dir, "sample.tsv")
        assert os.path.exists(tsv_path)

        # Load TSV file and check content
        result_df = pd.read_csv(tsv_path, sep="\t")
        assert list(result_df.columns) == ["record_id", "element_1", "element_2"]
        assert result_df.shape == (4, 3)  # Check number of rows and columns


def test_construct_all_tsvs_from_jsons():
    with tempfile.TemporaryDirectory() as temp_dir, tempfile.TemporaryDirectory() as output_dir:
        # Create a sample DataFrame
        data = {
            "record_id": [1, 2, 3, 4],
            "element_1": ["A", "B", "C", "D"],
            "element_2": [10, 20, 30, 40],
            "element_3": ["X", "Y", "Z", "W"],
        }
        df = pd.DataFrame(data)

        # Create multiple sample JSON files with the updated structure
        json_data_1 = {
            "schema_name": {
                "description": "Schema 1 description.",
                "data_elements": {
                    "record_id": {},
                    "element_1": {
                        "description": "Description for element_1.",
                        "question": {"en": "Question text for element_1."},
                        "datatype": ["xsd:string"],
                    },
                },
            }
        }
        json_path_1 = os.path.join(temp_dir, "sample1.json")
        with open(json_path_1, "w") as f:
            json.dump(json_data_1, f)

        json_data_2 = {
            "schema_name": {
                "description": "Schema 2 description.",
                "data_elements": {
                    "record_id": {},
                    "element_2": {
                        "description": "Description for element_2.",
                        "question": {"en": "Question text for element_2."},
                        "datatype": ["xsd:decimal"],
                    },
                },
            }
        }
        json_path_2 = os.path.join(temp_dir, "sample2.json")
        with open(json_path_2, "w") as f:
            json.dump(json_data_2, f)

        # Create an excluded file
        json_data_excluded = {
            "schema_name": {
                "description": "Excluded schema description.",
                "data_elements": {
                    "record_id": {},
                    "element_3": {
                        "description": "Description for element_3.",
                        "question": {"en": "Question text for element_3."},
                        "datatype": ["xsd:string"],
                    },
                },
            }
        }
        excluded_json_path = os.path.join(temp_dir, "excluded.json")
        with open(excluded_json_path, "w") as f:
            json.dump(json_data_excluded, f)

        # Run the function
        construct_all_tsvs_from_jsons(
            df, input_dir=temp_dir, output_dir=output_dir, excluded_files=["excluded.json"]
        )

        # Check if the TSV files are created
        tsv_path_1 = os.path.join(output_dir, "sample1.tsv")
        tsv_path_2 = os.path.join(output_dir, "sample2.tsv")
        excluded_tsv_path = os.path.join(output_dir, "excluded.tsv")

        assert os.path.exists(tsv_path_1)
        assert os.path.exists(tsv_path_2)
        assert not os.path.exists(excluded_tsv_path)  # Ensure the excluded file was not processed

        # Load TSV files and check content
        result_df_1 = pd.read_csv(tsv_path_1, sep="\t")
        assert list(result_df_1.columns) == ["record_id", "element_1"]
        assert result_df_1.shape == (4, 2)  # Check number of rows and columns

        result_df_2 = pd.read_csv(tsv_path_2, sep="\t")
        assert list(result_df_2.columns) == ["record_id", "element_2"]
        assert result_df_2.shape == (4, 2)  # Check number of rows and columns
