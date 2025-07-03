import json
import os
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest
from pydantic import BaseModel

from b2aiprep.prepare.bids import (
    _df_to_dict,
    construct_all_tsvs_from_jsons,
    construct_tsv_from_json,
    create_file_dir,
    get_audio_paths,
    get_df_of_repeat_instrument,
    get_instrument_for_name,
    get_paths,
    get_recordings_for_acoustic_task,
    insert_missing_columns,
    load_redcap_csv,
    output_participant_data_to_fhir,
    questionnaire_mapping,
    redcap_to_bids,
    validate_bids_folder,
    validate_redcap_df_column_names,
    write_pydantic_model_to_bids_file,
)
from b2aiprep.prepare.constants import AUDIO_TASKS, RepeatInstrument
from b2aiprep.prepare.utils import initialize_data_directory


def test_get_paths():
    """Test get_paths function with a proper BIDS-like directory structure."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create BIDS-like directory structure
        # sub-001/ses-001/audio/
        subject1_session1_audio = temp_path / "sub-001" / "ses-001" / "audio"
        subject1_session1_audio.mkdir(parents=True)

        # sub-001/ses-002/audio/
        subject1_session2_audio = temp_path / "sub-001" / "ses-002" / "audio"
        subject1_session2_audio.mkdir(parents=True)

        # sub-002/ses-001/audio/
        subject2_session1_audio = temp_path / "sub-002" / "ses-001" / "audio"
        subject2_session1_audio.mkdir(parents=True)

        # Create some test files with .wav extension
        wav_file1 = subject1_session1_audio / "sub-001_task-reading.wav"
        wav_file1.write_text("fake audio content 1")

        wav_file2 = subject1_session2_audio / "sub-001_task-speaking.wav"
        wav_file2.write_text("fake audio content 2")

        wav_file3 = subject2_session1_audio / "sub-002_task-reading.wav"
        wav_file3.write_text("fake audio content 3")

        # Create some files with different extensions that should be ignored
        txt_file = subject1_session1_audio / "sub-001_metadata.txt"
        txt_file.write_text("metadata content")

        json_file = subject1_session2_audio / "sub-001_config.json"
        json_file.write_text('{"config": "value"}')

        # Call get_paths with .wav extension
        result = get_paths(temp_path, ".wav")

        # Verify results
        assert len(result) == 3

        # Sort results by path for consistent testing
        result.sort(key=lambda x: str(x["path"]))

        # Check first file
        assert result[0]["path"] == wav_file1.absolute()
        assert result[0]["subject"] == "001"
        assert result[0]["size"] == len("fake audio content 1")

        # Check second file
        assert result[1]["path"] == wav_file2.absolute()
        assert result[1]["subject"] == "001"
        assert result[1]["size"] == len("fake audio content 2")

        # Check third file
        assert result[2]["path"] == wav_file3.absolute()
        assert result[2]["subject"] == "002"
        assert result[2]["size"] == len("fake audio content 3")


def test_get_paths_with_different_extension():
    """Test get_paths function with a different file extension."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create BIDS-like directory structure
        subject_audio = temp_path / "sub-001" / "ses-001" / "audio"
        subject_audio.mkdir(parents=True)

        # Create files with different extensions
        wav_file = subject_audio / "sub-001_task-reading.wav"
        wav_file.write_text("audio content")

        txt_file = subject_audio / "sub-001_transcript.txt"
        txt_file.write_text("transcript content")

        json_file = subject_audio / "sub-001_metadata.json"
        json_file.write_text('{"key": "value"}')

        # Test with .txt extension
        result = get_paths(temp_path, ".txt")
        assert len(result) == 1
        assert result[0]["path"] == txt_file.absolute()
        assert result[0]["subject"] == "001"

        # Test with .json extension
        result = get_paths(temp_path, ".json")
        assert len(result) == 1
        assert result[0]["path"] == json_file.absolute()
        assert result[0]["subject"] == "001"


def test_get_paths_empty_directory():
    """Test get_paths function with an empty directory."""
    with TemporaryDirectory() as temp_dir:
        result = get_paths(temp_dir, ".wav")
        assert result == []


def test_get_paths_no_matching_files():
    """Test get_paths function when no files match the extension."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create BIDS-like directory structure
        subject_audio = temp_path / "sub-001" / "ses-001" / "audio"
        subject_audio.mkdir(parents=True)

        # Create files with different extensions
        txt_file = subject_audio / "sub-001_transcript.txt"
        txt_file.write_text("transcript content")

        # Look for .wav files when only .txt files exist
        result = get_paths(temp_path, ".wav")
        assert result == []


def test_get_paths_ignores_non_bids_directories():
    """Test that get_paths ignores directories that don't follow BIDS naming conventions."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create proper BIDS directory
        proper_audio = temp_path / "sub-001" / "ses-001" / "audio"
        proper_audio.mkdir(parents=True)
        proper_file = proper_audio / "sub-001_task-reading.wav"
        proper_file.write_text("proper audio content")

        # Create directories that don't follow BIDS conventions
        # Wrong subject prefix (doesn't start with exactly "sub-")
        wrong_subject = temp_path / "participant-001" / "ses-001" / "audio"
        wrong_subject.mkdir(parents=True)
        wrong_subject_file = wrong_subject / "participant-001_task-reading.wav"
        wrong_subject_file.write_text("wrong subject audio")

        # Wrong session prefix (doesn't start with exactly "ses-")
        wrong_session = temp_path / "sub-002" / "visit-001" / "audio"
        wrong_session.mkdir(parents=True)
        wrong_session_file = wrong_session / "sub-002_task-reading.wav"
        wrong_session_file.write_text("wrong session audio")

        # No audio directory
        no_audio = temp_path / "sub-003" / "ses-001"
        no_audio.mkdir(parents=True)
        no_audio_file = no_audio / "sub-003_task-reading.wav"
        no_audio_file.write_text("no audio dir")

        # Regular files in root (should be ignored)
        root_file = temp_path / "README.txt"
        root_file.write_text("readme content")

        # Call get_paths
        result = get_paths(temp_path, ".wav")

        # Should only find the properly structured file
        assert len(result) == 1
        assert result[0]["path"] == proper_file.absolute()
        assert result[0]["subject"] == "001"


def test_get_paths_complex_subject_extraction():
    """Test get_paths with complex subject ID extraction from filenames."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create BIDS-like directory structure
        subject_audio = temp_path / "sub-001" / "ses-001" / "audio"
        subject_audio.mkdir(parents=True)

        # Create file with complex naming
        complex_file = subject_audio / "sub-001_ses-001_task-reading_run-01.wav"
        complex_file.write_text("complex audio content")

        result = get_paths(temp_path, ".wav")

        assert len(result) == 1
        assert result[0]["path"] == complex_file.absolute()
        # Should extract "001" from "sub-001_ses-001_task-reading_run-01.wav"
        assert result[0]["subject"] == "001"
        assert result[0]["size"] == len("complex audio content")


@patch("os.listdir")
def test_get_paths_handles_os_errors(mock_listdir):
    """Test that get_paths handles OS errors gracefully."""
    # Mock os.listdir to raise an exception
    mock_listdir.side_effect = OSError("Permission denied")

    with pytest.raises(OSError):
        get_paths("/fake/path", ".wav")


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


@patch("b2aiprep.prepare.bids.files")
@patch("json.load")
def test_insert_missing_columns_with_missing_columns(mock_json_load, mock_files):
    """Test insert_missing_columns when DataFrame is missing some expected columns."""
    # Mock the expected columns from JSON file
    expected_columns = ["record_id", "column_a", "column_b", "column_c", "column_d"]
    mock_json_load.return_value = expected_columns

    # Mock the file path chain
    mock_path = MagicMock()
    mock_path.open.return_value = MagicMock()
    mock_files.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value = (
        mock_path
    )

    # Create test DataFrame with only some of the expected columns
    test_df = pd.DataFrame(
        {
            "record_id": [1, 2, 3],
            "column_a": ["A", "B", "C"],
            "column_b": [10, 20, 30],
            # Missing column_c and column_d
        }
    )

    # Call the function
    result_df = insert_missing_columns(test_df)

    # Verify all expected columns are present
    assert set(result_df.columns) == set(expected_columns)

    # Verify original data is preserved
    assert list(result_df["record_id"]) == [1, 2, 3]
    assert list(result_df["column_a"]) == ["A", "B", "C"]
    assert list(result_df["column_b"]) == [10, 20, 30]

    # Verify missing columns were added with NaN values
    assert result_df["column_c"].isna().all()
    assert result_df["column_d"].isna().all()

    # Verify correct file path was accessed
    mock_files.assert_called_with("b2aiprep")


@patch("b2aiprep.prepare.bids.files")
@patch("json.load")
def test_insert_missing_columns_all_present(mock_json_load, mock_files):
    """Test insert_missing_columns when DataFrame already has all expected columns."""
    # Mock the expected columns from JSON file
    expected_columns = ["record_id", "column_a", "column_b"]
    mock_json_load.return_value = expected_columns

    # Mock the file path chain
    mock_path = MagicMock()
    mock_path.open.return_value = MagicMock()
    mock_files.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value = (
        mock_path
    )

    # Create test DataFrame with all expected columns
    test_df = pd.DataFrame(
        {"record_id": [1, 2, 3], "column_a": ["A", "B", "C"], "column_b": [10, 20, 30]}
    )

    # Store original for comparison
    original_columns = set(test_df.columns)
    original_shape = test_df.shape

    # Call the function
    result_df = insert_missing_columns(test_df)

    # Verify no new columns were added
    assert set(result_df.columns) == original_columns
    assert result_df.shape == original_shape

    # Verify original data is unchanged
    assert list(result_df["record_id"]) == [1, 2, 3]
    assert list(result_df["column_a"]) == ["A", "B", "C"]
    assert list(result_df["column_b"]) == [10, 20, 30]


@patch("b2aiprep.prepare.bids.files")
@patch("json.load")
def test_insert_missing_columns_no_existing_columns(mock_json_load, mock_files):
    """Test insert_missing_columns when DataFrame has none of the expected columns."""
    # Mock the expected columns from JSON file
    expected_columns = ["expected_col_1", "expected_col_2", "expected_col_3"]
    mock_json_load.return_value = expected_columns

    # Mock the file path chain
    mock_path = MagicMock()
    mock_path.open.return_value = MagicMock()
    mock_files.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value = (
        mock_path
    )

    # Create test DataFrame with completely different columns
    test_df = pd.DataFrame({"existing_col_1": [1, 2, 3], "existing_col_2": ["A", "B", "C"]})

    # Call the function
    result_df = insert_missing_columns(test_df)

    # Verify all expected columns were added
    for col in expected_columns:
        assert col in result_df.columns
        assert result_df[col].isna().all()

    # Verify original columns are preserved
    assert "existing_col_1" in result_df.columns
    assert "existing_col_2" in result_df.columns
    assert list(result_df["existing_col_1"]) == [1, 2, 3]
    assert list(result_df["existing_col_2"]) == ["A", "B", "C"]


@patch("b2aiprep.prepare.bids.files")
@patch("json.load")
def test_insert_missing_columns_file_path_construction(mock_json_load, mock_files):
    """Test that insert_missing_columns constructs the correct file path."""
    # Mock the expected columns from JSON file
    expected_columns = ["col1", "col2"]
    mock_json_load.return_value = expected_columns

    # Mock the file path chain - files().joinpath() is called with all args at once
    mock_json_path = MagicMock()
    mock_json_path.open.return_value = MagicMock()
    mock_files.return_value.joinpath.return_value = mock_json_path

    # Create minimal test DataFrame
    test_df = pd.DataFrame({"existing": [1]})

    # Call the function
    insert_missing_columns(test_df)

    # Verify the file path was constructed correctly
    mock_files.assert_called_once_with("b2aiprep")
    mock_files.return_value.joinpath.assert_called_once_with(
        "prepare", "resources", "all_columns.json"
    )
    mock_json_path.open.assert_called_once()
    mock_json_load.assert_called_once()


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

    # Manually call get_df_of_repeat_instrument() to mimic what
    # get_recordings_for_acoustic_task() does internally
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
    if not csv_file_path.exists():
        raise FileNotFoundError(f"CSV file not found at: {csv_file_path}")

    # Use TemporaryDirectory for the output directory
    with TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "bids_output"
        initialize_data_directory(output_dir)
        # Call the actual redcap_to_bids function with the real CSV and temporary output dir
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


def test_get_paths_subject_extraction_edge_cases():
    """Test edge cases in subject ID extraction from filenames."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create BIDS-like directory structure
        subject_audio = temp_path / "sub-123" / "ses-001" / "audio"
        subject_audio.mkdir(parents=True)

        # Test various filename patterns
        test_cases = [
            ("sub-123_task-reading.wav", "123"),
            ("sub-123_ses-001_task-reading.wav", "123"),
            ("sub-123_ses-001_task-reading_run-01.wav", "123"),
        ]

        for filename, expected_subject in test_cases:
            # Create the test file
            test_file = subject_audio / filename
            test_file.write_text("test content")

            # Call get_paths
            result = get_paths(temp_path, ".wav")

            # Verify subject extraction
            assert len(result) == 1
            assert result[0]["subject"] == expected_subject

            # Clean up for next iteration
            test_file.unlink()


def test_get_paths_problematic_filename():
    """Test that get_paths handles files that don't follow BIDS naming convention."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # This creates a directory that starts with "sub" but isn't properly formatted
        # The directory name "subject-001" starts with "sub" so it gets processed
        subject_dir = temp_path / "subject-001"
        session_dir = subject_dir / "ses-001"
        audio_dir = session_dir / "audio"
        audio_dir.mkdir(parents=True)

        # Create a file that doesn't have "sub-" in the expected position
        problem_file = audio_dir / "subject-001_task-reading.wav"
        problem_file.write_text("test content")

        # With the fixed get_paths function, this should skip malformed files
        result = get_paths(temp_path, ".wav")

        # Should return empty list since the file doesn't follow BIDS naming convention
        assert result == []


def test_get_audio_paths():
    """Test get_audio_paths function specifically for .wav files."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create BIDS-like directory structure
        subject1_audio = temp_path / "sub-001" / "ses-001" / "audio"
        subject1_audio.mkdir(parents=True)

        subject2_audio = temp_path / "sub-002" / "ses-001" / "audio"
        subject2_audio.mkdir(parents=True)

        # Create .wav audio files
        wav_file1 = subject1_audio / "sub-001_task-reading.wav"
        wav_file1.write_text("audio content 1")

        wav_file2 = subject2_audio / "sub-002_task-speaking.wav"
        wav_file2.write_text("audio content 2")

        # Create non-audio files that should be ignored
        txt_file = subject1_audio / "sub-001_transcript.txt"
        txt_file.write_text("transcript content")

        json_file = subject1_audio / "sub-001_metadata.json"
        json_file.write_text('{"key": "value"}')

        mp3_file = subject2_audio / "sub-002_recording.mp3"
        mp3_file.write_text("mp3 audio content")

        # Call get_audio_paths
        result = get_audio_paths(temp_path)

        # Verify only .wav files are returned
        assert len(result) == 2

        # Sort results by subject for consistent testing
        result.sort(key=lambda x: x["subject"])

        # Check first audio file
        assert result[0]["path"] == wav_file1.absolute()
        assert result[0]["subject"] == "001"
        assert result[0]["size"] == len("audio content 1")

        # Check second audio file
        assert result[1]["path"] == wav_file2.absolute()
        assert result[1]["subject"] == "002"
        assert result[1]["size"] == len("audio content 2")


def test_get_audio_paths_empty_directory():
    """Test get_audio_paths with an empty directory."""
    with TemporaryDirectory() as temp_dir:
        result = get_audio_paths(temp_dir)
        assert result == []


def test_get_audio_paths_no_audio_files():
    """Test get_audio_paths when directory contains no .wav files."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create BIDS-like directory structure
        subject_audio = temp_path / "sub-001" / "ses-001" / "audio"
        subject_audio.mkdir(parents=True)

        # Create non-audio files
        txt_file = subject_audio / "sub-001_transcript.txt"
        txt_file.write_text("transcript content")

        json_file = subject_audio / "sub-001_metadata.json"
        json_file.write_text('{"key": "value"}')

        # Call get_audio_paths
        result = get_audio_paths(temp_path)

        # Should return empty list as no .wav files exist
        assert result == []


def test_validate_redcap_df_column_names_all_coded():
    """Test validate_redcap_df_column_names with all coded headers - should pass without error."""
    # Create DataFrame with coded column names
    mock_data = pd.DataFrame(
        {
            "record_id": [1, 2, 3],
            "selected_language": ["English", "Spanish", "French"],
            "consent_status": ["Consented", "Pending", "Declined"],
        }
    )

    # Mock the column mapping to include these columns
    mock_column_mapping = {
        "record_id": "Record ID",
        "selected_language": "Language",
        "consent_status": "Consent Status",
    }

    with (
        patch("b2aiprep.prepare.bids.files") as mock_files,
        patch("json.loads", return_value=mock_column_mapping),
    ):
        mock_resource = MagicMock()
        mock_resource.joinpath.return_value.read_text.return_value = "{}"
        mock_files.return_value.joinpath.return_value = mock_resource

        # Should not raise an exception
        validate_redcap_df_column_names(mock_data)


def test_validate_redcap_df_column_names_no_coded_headers():
    """Test validate_redcap_df_column_names with no coded headers - should raise ValueError."""
    # Create DataFrame with label column names only
    mock_data = pd.DataFrame(
        {
            "Random Column": [1, 2, 3],
            "Another Random": ["A", "B", "C"],
            "Third Random": ["X", "Y", "Z"],
        }
    )

    mock_column_mapping = {
        "record_id": "Record ID",
        "selected_language": "Language",
        "consent_status": "Consent Status",
    }

    with (
        patch("b2aiprep.prepare.bids.files") as mock_files,
        patch("json.loads", return_value=mock_column_mapping),
    ):
        mock_resource = MagicMock()
        mock_resource.joinpath.return_value.read_text.return_value = "{}"
        mock_files.return_value.joinpath.return_value = mock_resource

        with pytest.raises(ValueError, match="Dataframe has no coded headers"):
            validate_redcap_df_column_names(mock_data)


def test_validate_redcap_df_column_names_majority_label_headers():
    """Test validate_redcap_df_column_names with majority label headers - raises ValueError."""
    # Create DataFrame with mostly label column names
    mock_data = pd.DataFrame(
        {
            "Record ID": [1, 2, 3],
            "Language": ["English", "Spanish", "French"],
            "Consent Status": ["Consented", "Pending", "Declined"],
            "record_id": [1, 2, 3],  # Only one coded header
        }
    )

    mock_column_mapping = {
        "record_id": "Record ID",
        "selected_language": "Language",
        "consent_status": "Consent Status",
    }

    with (
        patch("b2aiprep.prepare.bids.files") as mock_files,
        patch("json.loads", return_value=mock_column_mapping),
    ):
        mock_resource = MagicMock()
        mock_resource.joinpath.return_value.read_text.return_value = "{}"
        mock_files.return_value.joinpath.return_value = mock_resource

        with pytest.raises(
            ValueError, match="Dataframe has label headers rather than coded headers"
        ):
            validate_redcap_df_column_names(mock_data)


def test_validate_redcap_df_column_names_mixed_headers_warning():
    """Test validate_redcap_df_column_names with mixed headers - should log warning."""
    # Create DataFrame with mix of coded and label headers (but majority coded)
    mock_data = pd.DataFrame(
        {
            "record_id": [1, 2, 3],
            "selected_language": ["English", "Spanish", "French"],
            "Language": ["English", "Spanish", "French"],  # One label header
        }
    )

    mock_column_mapping = {
        "record_id": "Record ID",
        "selected_language": "Language",
        "consent_status": "Consent Status",
    }

    with (
        patch("b2aiprep.prepare.bids.files") as mock_files,
        patch("json.loads", return_value=mock_column_mapping),
        patch("b2aiprep.prepare.bids._LOGGER") as mock_logger,
    ):
        mock_resource = MagicMock()
        mock_resource.joinpath.return_value.read_text.return_value = "{}"
        mock_files.return_value.joinpath.return_value = mock_resource

        validate_redcap_df_column_names(mock_data)

        # Should have logged a warning about mixed headers
        mock_logger.warning.assert_called()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "mix of label and coded headers" in warning_call


def test_validate_redcap_df_column_names_partial_coded_headers_warning():
    """Test validate_redcap_df_column_names with some coded headers but not all - logs warning."""
    # Create DataFrame with some coded headers and some unknown columns
    mock_data = pd.DataFrame(
        {
            "record_id": [1, 2, 3],
            "selected_language": ["English", "Spanish", "French"],
            "unknown_column": ["A", "B", "C"],
            "another_unknown": ["X", "Y", "Z"],
        }
    )

    mock_column_mapping = {
        "record_id": "Record ID",
        "selected_language": "Language",
        "consent_status": "Consent Status",
    }

    with (
        patch("b2aiprep.prepare.bids.files") as mock_files,
        patch("json.loads", return_value=mock_column_mapping),
        patch("b2aiprep.prepare.bids._LOGGER") as mock_logger,
    ):
        mock_resource = MagicMock()
        mock_resource.joinpath.return_value.read_text.return_value = "{}"
        mock_files.return_value.joinpath.return_value = mock_resource

        validate_redcap_df_column_names(mock_data)

        # Should have logged a warning about partial coded headers
        mock_logger.warning.assert_called()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "coded headers" in warning_call and "/" in warning_call


@patch("b2aiprep.prepare.bids.files")
@patch("json.loads")
def test_validate_redcap_df_column_names_file_loading(mock_json_loads, mock_files):
    """Test that validate_redcap_df_column_names correctly loads the column mapping file."""
    mock_data = pd.DataFrame(
        {"record_id": [1, 2, 3], "selected_language": ["English", "Spanish", "French"]}
    )

    mock_column_mapping = {"record_id": "Record ID", "selected_language": "Language"}
    mock_json_loads.return_value = mock_column_mapping

    # Mock the chained file system access: files().joinpath().joinpath()
    mock_final_resource = MagicMock()
    mock_final_resource.read_text.return_value = "{}"

    mock_resources_path = MagicMock()
    mock_resources_path.joinpath.return_value = mock_final_resource

    mock_prepare_path = MagicMock()
    mock_prepare_path.joinpath.return_value = mock_resources_path

    mock_files.return_value.joinpath.return_value = mock_prepare_path

    validate_redcap_df_column_names(mock_data)

    # Verify the correct file path was accessed
    mock_files.assert_called_with("b2aiprep")
    mock_files.return_value.joinpath.assert_called_with("prepare")
    mock_prepare_path.joinpath.assert_called_with("resources")
    mock_resources_path.joinpath.assert_called_with("column_mapping.json")
    mock_final_resource.read_text.assert_called_once()


def test_output_participant_data_to_fhir_basic():
    """Test output_participant_data_to_fhir with basic participant data."""
    with TemporaryDirectory() as temp_dir:
        outdir = Path(temp_dir)

        # Create minimal test participant data
        participant = {
            "record_id": "001",
            "sessions": [
                {
                    "session_id": "session1",
                    "acoustic_tasks": [
                        {
                            "acoustic_task_name": "Test Task",
                            "acoustic_task_id": "task1",
                            "recordings": [
                                {
                                    "recording_id": "rec1",
                                    "recording_name": "Test Recording",
                                }
                            ],
                        }
                    ],
                }
            ],
        }

        # Mock the required functions
        with (
            patch("b2aiprep.prepare.bids.get_instrument_for_name") as mock_get_instrument,
            patch("b2aiprep.prepare.bids.convert_response_to_fhir") as mock_convert_fhir,
            patch("b2aiprep.prepare.bids.write_pydantic_model_to_bids_file") as mock_write_file,
            patch("pandas.DataFrame.to_csv") as mock_to_csv,
        ):
            # Set up mock instruments
            mock_session_instrument = MagicMock()
            mock_session_instrument.columns = ["session_id"]
            mock_task_instrument = MagicMock()
            mock_task_instrument.name = "acoustic_tasks"
            mock_task_instrument.schema_name = "acoustictaskschema"
            mock_task_instrument.columns = ["acoustic_task_id", "acoustic_task_name"]
            mock_recording_instrument = MagicMock()
            mock_recording_instrument.name = "recordings"
            mock_recording_instrument.schema_name = "recordingschema"
            mock_recording_instrument.columns = ["recording_id", "recording_name"]

            mock_get_instrument.side_effect = [
                mock_session_instrument,
                mock_task_instrument,
                mock_recording_instrument,
            ]

            # Mock FHIR conversion
            mock_fhir_data = MagicMock()
            mock_convert_fhir.return_value = mock_fhir_data

            # Call the function
            output_participant_data_to_fhir(participant, outdir)

            # Verify directory structure was created
            subject_dir = outdir / "sub-001"
            session_dir = subject_dir / "ses-session1"
            audio_dir = session_dir / "audio"
            assert audio_dir.exists()

            # Verify FHIR conversion was called
            assert mock_convert_fhir.call_count == 2  # Once for task, once for recording

            # Verify files were written
            assert mock_write_file.call_count == 2  # Once for task, once for recording

            # Verify sessions.tsv was created
            mock_to_csv.assert_called_once()
            sessions_tsv_path = subject_dir / "sessions.tsv"
            mock_to_csv.assert_called_with(sessions_tsv_path, sep="\t", index=False)


def test_output_participant_data_to_fhir_with_audio_files():
    """Test output_participant_data_to_fhir with audio file copying."""
    with TemporaryDirectory() as temp_dir, TemporaryDirectory() as audio_temp_dir:
        outdir = Path(temp_dir)
        audiodir = Path(audio_temp_dir)

        # Create a test audio file
        test_audio_file = audiodir / "site1" / "rec1_test.wav"
        test_audio_file.parent.mkdir(parents=True)
        test_audio_file.write_text("fake audio content")

        participant = {
            "record_id": "001",
            "sessions": [
                {
                    "session_id": "session1",
                    "acoustic_tasks": [
                        {
                            "acoustic_task_name": "Test Task",
                            "acoustic_task_id": "task1",
                            "recordings": [
                                {
                                    "recording_id": "rec1",
                                    "recording_name": "Test Recording",
                                }
                            ],
                        }
                    ],
                }
            ],
        }

        with (
            patch("b2aiprep.prepare.bids.get_instrument_for_name") as mock_get_instrument,
            patch("b2aiprep.prepare.bids.convert_response_to_fhir") as mock_convert_fhir,
            patch("b2aiprep.prepare.bids.write_pydantic_model_to_bids_file"),
            patch("pandas.DataFrame.to_csv"),
        ):
            # Set up mock instruments (similar to previous test)
            mock_session_instrument = MagicMock()
            mock_session_instrument.columns = ["session_id"]
            mock_task_instrument = MagicMock()
            mock_task_instrument.name = "acoustic_tasks"
            mock_task_instrument.schema_name = "acoustictaskschema"
            mock_task_instrument.columns = ["acoustic_task_id", "acoustic_task_name"]
            mock_recording_instrument = MagicMock()
            mock_recording_instrument.name = "recordings"
            mock_recording_instrument.schema_name = "recordingschema"
            mock_recording_instrument.columns = ["recording_id", "recording_name"]

            mock_get_instrument.side_effect = [
                mock_session_instrument,
                mock_task_instrument,
                mock_recording_instrument,
            ]

            mock_fhir_data = MagicMock()
            mock_convert_fhir.return_value = mock_fhir_data

            # Call the function with audiodir
            output_participant_data_to_fhir(participant, outdir, audiodir)

            # Verify audio file was copied
            expected_audio_dest = (
                outdir
                / "sub-001"
                / "ses-session1"
                / "audio"
                / "sub-001_ses-session1_task-Test-Recording.wav"
            )
            assert expected_audio_dest.exists()
            assert expected_audio_dest.read_text() == "fake audio content"


def test_output_participant_data_to_fhir_nonexistent_audiodir():
    """Test output_participant_data_to_fhir with non-existent audio directory."""
    with TemporaryDirectory() as temp_dir:
        outdir = Path(temp_dir)
        nonexistent_audiodir = Path(temp_dir) / "nonexistent"

        participant = {
            "record_id": "001",
            "sessions": [
                {
                    "session_id": "session1",
                    "acoustic_tasks": [
                        {
                            "acoustic_task_name": "Test Task",
                            "acoustic_task_id": "task1",
                            "recordings": [
                                {
                                    "recording_id": "rec1",
                                    "recording_name": "Test Recording",
                                }
                            ],
                        }
                    ],
                }
            ],
        }

        with (
            patch("b2aiprep.prepare.bids.get_instrument_for_name") as mock_get_instrument,
            patch("b2aiprep.prepare.bids.convert_response_to_fhir") as mock_convert_fhir,
            patch("b2aiprep.prepare.bids.write_pydantic_model_to_bids_file"),
            patch("pandas.DataFrame.to_csv"),
        ):
            # Set up mock instruments
            mock_session_instrument = MagicMock()
            mock_session_instrument.columns = ["session_id"]
            mock_task_instrument = MagicMock()
            mock_task_instrument.name = "acoustic_tasks"
            mock_task_instrument.schema_name = "acoustictaskschema"
            mock_task_instrument.columns = ["acoustic_task_id", "acoustic_task_name"]
            mock_recording_instrument = MagicMock()
            mock_recording_instrument.name = "recordings"
            mock_recording_instrument.schema_name = "recordingschema"
            mock_recording_instrument.columns = ["recording_id", "recording_name"]

            mock_get_instrument.side_effect = [
                mock_session_instrument,
                mock_task_instrument,
                mock_recording_instrument,
            ]

            mock_fhir_data = MagicMock()
            mock_convert_fhir.return_value = mock_fhir_data

            # Should not raise error even with non-existent audiodir
            output_participant_data_to_fhir(participant, outdir, nonexistent_audiodir)

            # Verify function completed successfully
            subject_dir = outdir / "sub-001"
            assert subject_dir.exists()


def test_output_participant_data_to_fhir_multiple_sessions():
    """Test output_participant_data_to_fhir with multiple sessions and tasks."""
    with TemporaryDirectory() as temp_dir:
        outdir = Path(temp_dir)

        participant = {
            "record_id": "001",
            "sessions": [
                {
                    "session_id": "session1",
                    "acoustic_tasks": [
                        {
                            "acoustic_task_name": "Task 1",
                            "acoustic_task_id": "task1",
                            "recordings": [
                                {
                                    "recording_id": "rec1",
                                    "recording_name": "Recording 1",
                                }
                            ],
                        },
                        {
                            "acoustic_task_name": "Task 2",
                            "acoustic_task_id": "task2",
                            "recordings": [
                                {
                                    "recording_id": "rec2",
                                    "recording_name": "Recording 2",
                                }
                            ],
                        },
                    ],
                },
                {
                    "session_id": "session2",
                    "acoustic_tasks": [
                        {
                            "acoustic_task_name": "Task 3",
                            "acoustic_task_id": "task3",
                            "recordings": [
                                {
                                    "recording_id": "rec3",
                                    "recording_name": "Recording 3",
                                }
                            ],
                        }
                    ],
                },
            ],
        }

        with (
            patch("b2aiprep.prepare.bids.get_instrument_for_name") as mock_get_instrument,
            patch("b2aiprep.prepare.bids.convert_response_to_fhir") as mock_convert_fhir,
            patch("b2aiprep.prepare.bids.write_pydantic_model_to_bids_file") as mock_write_file,
            patch("pandas.DataFrame.to_csv"),
        ):
            # Set up mock instruments
            mock_session_instrument = MagicMock()
            mock_session_instrument.columns = ["session_id"]
            mock_task_instrument = MagicMock()
            mock_task_instrument.name = "acoustic_tasks"
            mock_task_instrument.schema_name = "acoustictaskschema"
            mock_task_instrument.columns = ["acoustic_task_id", "acoustic_task_name"]
            mock_recording_instrument = MagicMock()
            mock_recording_instrument.name = "recordings"
            mock_recording_instrument.schema_name = "recordingschema"
            mock_recording_instrument.columns = ["recording_id", "recording_name"]

            mock_get_instrument.side_effect = [
                mock_session_instrument,
                mock_task_instrument,
                mock_recording_instrument,
            ]

            mock_fhir_data = MagicMock()
            mock_convert_fhir.return_value = mock_fhir_data

            output_participant_data_to_fhir(participant, outdir)

            # Verify directory structure for multiple sessions
            session1_dir = outdir / "sub-001" / "ses-session1" / "audio"
            session2_dir = outdir / "sub-001" / "ses-session2" / "audio"
            assert session1_dir.exists()
            assert session2_dir.exists()

            # Verify FHIR conversion called for all tasks and recordings
            # 3 tasks + 3 recordings = 6 calls
            assert mock_convert_fhir.call_count == 6

            # Verify write calls for all tasks and recordings
            assert mock_write_file.call_count == 6


def test_output_participant_data_to_fhir_none_task():
    """Test output_participant_data_to_fhir handles None tasks gracefully."""
    with TemporaryDirectory() as temp_dir:
        outdir = Path(temp_dir)

        participant = {
            "record_id": "001",
            "sessions": [
                {
                    "session_id": "session1",
                    "acoustic_tasks": [
                        None,  # This should be skipped
                        {
                            "acoustic_task_name": "Valid Task",
                            "acoustic_task_id": "task1",
                            "recordings": [
                                {
                                    "recording_id": "rec1",
                                    "recording_name": "Valid Recording",
                                }
                            ],
                        },
                    ],
                }
            ],
        }

        with (
            patch("b2aiprep.prepare.bids.get_instrument_for_name") as mock_get_instrument,
            patch("b2aiprep.prepare.bids.convert_response_to_fhir") as mock_convert_fhir,
            patch("b2aiprep.prepare.bids.write_pydantic_model_to_bids_file") as mock_write_file,
            patch("pandas.DataFrame.to_csv"),
        ):
            # Set up mock instruments
            mock_session_instrument = MagicMock()
            mock_session_instrument.columns = ["session_id"]
            mock_task_instrument = MagicMock()
            mock_task_instrument.name = "acoustic_tasks"
            mock_task_instrument.schema_name = "acoustictaskschema"
            mock_task_instrument.columns = ["acoustic_task_id", "acoustic_task_name"]
            mock_recording_instrument = MagicMock()
            mock_recording_instrument.name = "recordings"
            mock_recording_instrument.schema_name = "recordingschema"
            mock_recording_instrument.columns = ["recording_id", "recording_name"]

            mock_get_instrument.side_effect = [
                mock_session_instrument,
                mock_task_instrument,
                mock_recording_instrument,
            ]

            mock_fhir_data = MagicMock()
            mock_convert_fhir.return_value = mock_fhir_data

            output_participant_data_to_fhir(participant, outdir)

            # Should only process the valid task (1 task + 1 recording = 2 calls)
            assert mock_convert_fhir.call_count == 2
            assert mock_write_file.call_count == 2


@patch("logging.warning")
def test_output_participant_data_to_fhir_missing_audio_file(mock_warning):
    """Test output_participant_data_to_fhir logs warning when audio file not found."""
    with TemporaryDirectory() as temp_dir, TemporaryDirectory() as audio_temp_dir:
        outdir = Path(temp_dir)
        audiodir = Path(audio_temp_dir)

        participant = {
            "record_id": "001",
            "sessions": [
                {
                    "session_id": "session1",
                    "acoustic_tasks": [
                        {
                            "acoustic_task_name": "Test Task",
                            "acoustic_task_id": "task1",
                            "recordings": [
                                {
                                    "recording_id": "nonexistent_rec",
                                    "recording_name": "Missing Recording",
                                }
                            ],
                        }
                    ],
                }
            ],
        }

        with (
            patch("b2aiprep.prepare.bids.get_instrument_for_name") as mock_get_instrument,
            patch("b2aiprep.prepare.bids.convert_response_to_fhir") as mock_convert_fhir,
            patch("b2aiprep.prepare.bids.write_pydantic_model_to_bids_file"),
            patch("pandas.DataFrame.to_csv"),
        ):
            # Set up mock instruments
            mock_session_instrument = MagicMock()
            mock_session_instrument.columns = ["session_id"]
            mock_task_instrument = MagicMock()
            mock_task_instrument.name = "acoustic_tasks"
            mock_task_instrument.schema_name = "acoustictaskschema"
            mock_task_instrument.columns = ["acoustic_task_id", "acoustic_task_name"]
            mock_recording_instrument = MagicMock()
            mock_recording_instrument.name = "recordings"
            mock_recording_instrument.schema_name = "recordingschema"
            mock_recording_instrument.columns = ["recording_id", "recording_name"]

            mock_get_instrument.side_effect = [
                mock_session_instrument,
                mock_task_instrument,
                mock_recording_instrument,
            ]

            mock_fhir_data = MagicMock()
            mock_convert_fhir.return_value = mock_fhir_data

            output_participant_data_to_fhir(participant, outdir, audiodir)

            # Verify warning was logged for missing audio file
            mock_warning.assert_called()
            warning_call = mock_warning.call_args[0][0]
            assert "No audio file found for recording" in warning_call


def test_validate_bids_folder_all_files_present():
    """Test validate_bids_folder when all files have features and transcripts."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create BIDS directory structure
        audio_dir = temp_path / "sub-001" / "ses-001" / "audio"
        audio_dir.mkdir(parents=True)

        # Create audio file
        audio_file = audio_dir / "sub-001_ses-001_task-reading.wav"
        audio_file.write_text("audio content")

        # Create corresponding feature and transcript files
        feature_file = audio_dir / "sub-001_ses-001_task-reading.pt"
        feature_file.write_text("feature content")

        transcript_file = audio_dir / "sub-001_ses-001_task-reading.txt"
        transcript_file.write_text("transcript content")

        with patch("b2aiprep.prepare.bids._LOGGER") as mock_logger:
            validate_bids_folder(temp_path)

            # Should log success message
            mock_logger.info.assert_called_with("All audio files have been processed.")
            mock_logger.warning.assert_not_called()


def test_validate_bids_folder_missing_features():
    """Test validate_bids_folder when audio files are missing feature files."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create BIDS directory structure
        audio_dir1 = temp_path / "sub-001" / "ses-001" / "audio"
        audio_dir1.mkdir(parents=True)
        audio_dir2 = temp_path / "sub-002" / "ses-001" / "audio"
        audio_dir2.mkdir(parents=True)

        # Create audio files
        audio_file1 = audio_dir1 / "sub-001_ses-001_task-reading.wav"
        audio_file1.write_text("audio content 1")
        audio_file2 = audio_dir2 / "sub-002_ses-001_task-speaking.wav"
        audio_file2.write_text("audio content 2")

        # Create transcript files but no feature files
        transcript_file1 = audio_dir1 / "sub-001_ses-001_task-reading.txt"
        transcript_file1.write_text("transcript content 1")
        transcript_file2 = audio_dir2 / "sub-002_ses-001_task-speaking.txt"
        transcript_file2.write_text("transcript content 2")

        with patch("b2aiprep.prepare.bids._LOGGER") as mock_logger:
            validate_bids_folder(temp_path)

            # Should log warning about missing features
            mock_logger.warning.assert_called()
            warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
            feature_warning = next(
                (call for call in warning_calls if "Missing features" in call), None
            )
            assert feature_warning is not None
            assert "Missing features for 2 / 2 audio files" in feature_warning


def test_validate_bids_folder_missing_transcriptions():
    """Test validate_bids_folder when audio files are missing transcription files."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create BIDS directory structure
        audio_dir = temp_path / "sub-001" / "ses-001" / "audio"
        audio_dir.mkdir(parents=True)

        # Create audio file
        audio_file = audio_dir / "sub-001_ses-001_task-reading.wav"
        audio_file.write_text("audio content")

        # Create feature file but no transcript file
        feature_file = audio_dir / "sub-001_ses-001_task-reading.pt"
        feature_file.write_text("feature content")

        with patch("b2aiprep.prepare.bids._LOGGER") as mock_logger:
            validate_bids_folder(temp_path)

            # Should log warning about missing transcriptions
            mock_logger.warning.assert_called()
            warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]
            transcript_warning = next(
                (call for call in warning_calls if "Missing transcriptions" in call), None
            )
            assert transcript_warning is not None
            assert "Missing transcriptions for 1 / 1 audio files" in transcript_warning


def test_validate_bids_folder_missing_both():
    """Test validate_bids_folder when audio files are missing both features and transcriptions."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create BIDS directory structure
        audio_dir = temp_path / "sub-001" / "ses-001" / "audio"
        audio_dir.mkdir(parents=True)

        # Create audio file only (no features or transcripts)
        audio_file = audio_dir / "sub-001_ses-001_task-reading.wav"
        audio_file.write_text("audio content")

        with patch("b2aiprep.prepare.bids._LOGGER") as mock_logger:
            validate_bids_folder(temp_path)

            # Should log warnings for both missing features and transcriptions
            mock_logger.warning.assert_called()
            warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]

            feature_warning = next(
                (call for call in warning_calls if "Missing features" in call), None
            )
            transcript_warning = next(
                (call for call in warning_calls if "Missing transcriptions" in call), None
            )

            assert feature_warning is not None
            assert transcript_warning is not None
            assert "Missing features for 1 / 1 audio files" in feature_warning
            assert "Missing transcriptions for 1 / 1 audio files" in transcript_warning


def test_validate_bids_folder_partial_missing():
    """Test validate_bids_folder with mixed scenarios - some files complete, some missing features/transcripts."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create BIDS directory structure
        audio_dir1 = temp_path / "sub-001" / "ses-001" / "audio"
        audio_dir1.mkdir(parents=True)
        audio_dir2 = temp_path / "sub-002" / "ses-001" / "audio"
        audio_dir2.mkdir(parents=True)
        audio_dir3 = temp_path / "sub-003" / "ses-001" / "audio"
        audio_dir3.mkdir(parents=True)

        # Create audio files
        audio_file1 = audio_dir1 / "sub-001_ses-001_task-reading.wav"
        audio_file1.write_text("audio content 1")
        audio_file2 = audio_dir2 / "sub-002_ses-001_task-speaking.wav"
        audio_file2.write_text("audio content 2")
        audio_file3 = audio_dir3 / "sub-003_ses-001_task-counting.wav"
        audio_file3.write_text("audio content 3")

        # File 1: Complete (has both feature and transcript)
        feature_file1 = audio_dir1 / "sub-001_ses-001_task-reading.pt"
        feature_file1.write_text("feature content 1")
        transcript_file1 = audio_dir1 / "sub-001_ses-001_task-reading.txt"
        transcript_file1.write_text("transcript content 1")

        # File 2: Missing feature only
        transcript_file2 = audio_dir2 / "sub-002_ses-001_task-speaking.txt"
        transcript_file2.write_text("transcript content 2")

        # File 3: Missing transcript only
        feature_file3 = audio_dir3 / "sub-003_ses-001_task-counting.pt"
        feature_file3.write_text("feature content 3")

        with patch("b2aiprep.prepare.bids._LOGGER") as mock_logger:
            validate_bids_folder(temp_path)

            # Should log warnings for both missing features and transcriptions
            mock_logger.warning.assert_called()
            warning_calls = [call[0][0] for call in mock_logger.warning.call_args_list]

            feature_warning = next(
                (call for call in warning_calls if "Missing features" in call), None
            )
            transcript_warning = next(
                (call for call in warning_calls if "Missing transcriptions" in call), None
            )

            assert feature_warning is not None
            assert transcript_warning is not None
            assert "Missing features for 1 / 3 audio files" in feature_warning
            assert "Missing transcriptions for 1 / 3 audio files" in transcript_warning


def test_validate_bids_folder_no_audio_files():
    """Test validate_bids_folder when there are no audio files in the directory."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create BIDS directory structure but no audio files
        audio_dir = temp_path / "sub-001" / "ses-001" / "audio"
        audio_dir.mkdir(parents=True)

        # Create some non-audio files
        feature_file = audio_dir / "sub-001_ses-001_task-reading.pt"
        feature_file.write_text("feature content")
        transcript_file = audio_dir / "sub-001_ses-001_task-reading.txt"
        transcript_file.write_text("transcript content")

        with patch("b2aiprep.prepare.bids._LOGGER") as mock_logger:
            validate_bids_folder(temp_path)

            # Should log success since there are no audio files to validate
            mock_logger.info.assert_called_with("All audio files have been processed.")
            mock_logger.warning.assert_not_called()


def test_validate_bids_folder_complex_filenames():
    """Test validate_bids_folder with complex BIDS filenames."""
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create BIDS directory structure
        audio_dir = temp_path / "sub-001" / "ses-001" / "audio"
        audio_dir.mkdir(parents=True)

        # Create audio file with complex naming
        audio_file = audio_dir / "sub-001_ses-001_task-reading_run-01.wav"
        audio_file.write_text("audio content")

        # Create corresponding feature and transcript files
        feature_file = audio_dir / "sub-001_ses-001_task-reading_run-01.pt"
        feature_file.write_text("feature content")
        transcript_file = audio_dir / "sub-001_ses-001_task-reading_run-01.txt"
        transcript_file.write_text("transcript content")

        with patch("b2aiprep.prepare.bids._LOGGER") as mock_logger:
            validate_bids_folder(temp_path)

            # Should log success message
            mock_logger.info.assert_called_with("All audio files have been processed.")
            mock_logger.warning.assert_not_called()
