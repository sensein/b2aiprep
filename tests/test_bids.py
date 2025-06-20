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
    load_redcap_csv,
    questionnaire_mapping,
    redcap_to_bids,
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
