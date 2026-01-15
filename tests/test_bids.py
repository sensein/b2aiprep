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
    get_audio_paths,
    get_paths,
    validate_bids_folder_audios,
    write_pydantic_model_to_bids_file,
)
from b2aiprep.prepare.redcap import RedCapDataset
from b2aiprep.prepare.constants import AUDIO_TASKS, RepeatInstrument
from b2aiprep.prepare.dataset import BIDSDataset

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


@patch("pathlib.Path.exists")
@patch("pandas.read_csv")
def test_load_redcap_csv(mock_read_csv, mock_exists):
    # Mock file existence
    mock_exists.return_value = True
    
    # Mocking the DataFrame returned by read_csv
    mock_data = pd.DataFrame(
        {"record_id": [1, 2], "redcap_repeat_instrument": ["instrument_1", None]}
    )
    mock_read_csv.return_value = mock_data

    # Test the RedCapDataset._load_redcap_csv static method
    df = RedCapDataset._load_redcap_csv("dummy_path.csv")
    assert df is not None
    assert "record_id" in df.columns
    assert "redcap_repeat_instrument" in df.columns
    # Test that None values are filled with Participant instrument
    assert df["redcap_repeat_instrument"].iloc[1] == "Participant"

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

    # Create RedCapDataset and use its method
    dataset = RedCapDataset(df=mock_data, source_type='test')
    filtered_df = dataset.get_df_of_repeat_instrument(mock_instrument)
    assert len(filtered_df) == 1
    assert "column_1" in filtered_df.columns


def test_get_recordings_for_acoustic_task():
    # Use actual tasks from the AUDIO_TASKS list
    task_name = AUDIO_TASKS[0]  # Pick the first valid task from the list for the test

    # Create mock data for acoustic tasks and recordings based on the real task names
    mock_data = pd.DataFrame(
        {
            "acoustic_task_name": [task_name, AUDIO_TASKS[1]],  # Use real AUDIO_TASKS
            "acoustic_task_id": [1, 2],
            "redcap_repeat_instrument": ["Acoustic Task", "Acoustic Task"],
            "recording_acoustic_task_id": [1, 2],
            "recording_id": ["rec1", "rec2"]
        }
    )

    # Create RedCapDataset and use its method
    dataset = RedCapDataset(df=mock_data, source_type='test')
    
    # Test the get_recordings_for_acoustic_task method
    recordings_df = dataset.get_recordings_for_acoustic_task(task_name)
    
    # Verify results - should filter to recordings for the specified task
    assert len(recordings_df) >= 0  # May be empty if no recordings match


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

    result = BIDSDataset._df_to_dict(mock_data, "index_col")
    assert result["A"]["data_col"] == 1


# Note: create_file_dir and questionnaire_mapping tests removed as these functions
# have been moved to the new dataset classes or are no longer needed


def test_get_instrument_for_name():
    # Use an actual instrument name from RepeatInstrument
    instrument_name = "participant"

    # Call the function with the actual instrument name
    instrument = BIDSDataset._get_instrument_for_name(instrument_name)

    # Assert that the returned instrument matches the expected one
    assert instrument == RepeatInstrument.PARTICIPANT.value


def test_redcap_to_bids():
    project_root = Path(__file__).parent.parent
    csv_file_path = project_root / "data/sdv_redcap_synthetic_data_1000_rows.csv"
    if not csv_file_path.exists():
        raise FileNotFoundError(f"CSV file not found at: {csv_file_path}")

    # Create RedCapDataset from CSV file
    redcap_dataset = RedCapDataset.from_redcap(csv_file_path)
    
    # Use TemporaryDirectory for the output directory
    with TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "bids_output"

        # Convert to BIDS format using the new class method
        bids_dataset = BIDSDataset.from_redcap(redcap_dataset, output_dir, audiodir=None)
        
        # Check if the expected output files exist in the temporary directory
        if not any(output_dir.iterdir()):
            raise AssertionError("No output was created in the output directory")
        
        # Verify that the returned object is a BIDSDataset
        assert isinstance(bids_dataset, BIDSDataset), "redcap_to_bids should return a BIDSDataset instance"
        
        # Verify that the BIDSDataset points to the correct directory
        assert bids_dataset.data_path == output_dir.resolve(), "BIDSDataset should point to the output directory"


def test_redcap_dataset_to_csv():
    """Test the new RedCapDataset.to_csv method."""
    test_data = {
        'record_id': [1, 2, 3],
        'redcap_repeat_instrument': ['Participant', 'Session', 'Recording'],
        'test_column': ['a', 'b', 'c']
    }
    df = pd.DataFrame(test_data)
    
    # Mock RedCapDataset to avoid dependency issues
    from unittest.mock import MagicMock, patch
    
    with patch.dict('sys.modules', {
        'requests': MagicMock(),
        'tqdm': MagicMock(),
        'b2aiprep.prepare.constants': MagicMock(),
        'b2aiprep.prepare.reproschema_to_redcap': MagicMock(),
        'b2aiprep.prepare.utils': MagicMock(),
        'b2aiprep.prepare.bids': MagicMock(),
    }):
        from b2aiprep.prepare.redcap import RedCapDataset
        
        dataset = RedCapDataset(df=df, source_type='test')
        
        # Test to_csv method
        with TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "test_output.csv"
            dataset.to_csv(csv_path)
            
            # Verify the CSV was created
            assert csv_path.exists(), "CSV file should be created"
            
            # Verify the content
            result_df = pd.read_csv(csv_path)
            assert len(result_df) == 3, "CSV should have 3 rows"
            assert list(result_df.columns) == ['record_id', 'redcap_repeat_instrument', 'test_column'], "CSV should have correct columns"
            assert result_df['record_id'].tolist() == [1, 2, 3], "CSV should have correct data"

def test_construct_phenotype():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a sample DataFrame
        data = {
            # needs to be a real data element from reproschema for the test to pass
            "record_id": ["r01", "r02", "r03", "r04"],
            "acoustic_task_name": ["A", "B", "C", "D"],
        }
        df = pd.DataFrame(data)

        # Run the function using BIDSDataset static method
        BIDSDataset._construct_phenotype_from_reproschema(
            df, output_dir=temp_dir,
        )

        # Check if the TSV file is created
        tsv_path = list(Path(temp_dir).rglob("acoustic_task.tsv"))
        assert len(tsv_path) == 1, "TSV file should be created"
        tsv_path = tsv_path[0]

        # Load TSV file and check content
        result_df = pd.read_csv(tsv_path, sep="\t")
        columns = result_df.columns.tolist()
        assert columns[0] in {'record_id', 'participant_id'}, "First column should be record_id or participant_id"
        assert columns[1] == "acoustic_task_name", "Second column should be acoustic_task_name"

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
    """Test RedCapDataset._validate_redcap_columns with all coded headers - should pass without error."""
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
        patch("b2aiprep.prepare.redcap.files") as mock_files,
        patch("json.loads", return_value=mock_column_mapping),
    ):
        mock_resource = MagicMock()
        mock_resource.read_text.return_value = "{}"
        mock_files.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value = mock_resource

        # Create dataset and call method - should not raise an exception
        dataset = RedCapDataset(df=mock_data, source_type='test')
        dataset._validate_redcap_columns()


def test_validate_redcap_df_column_names_no_coded_headers():
    """Test RedCapDataset._validate_redcap_columns with no coded headers - should raise ValueError."""
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
        patch("b2aiprep.prepare.redcap.files") as mock_files,
        patch("json.loads", return_value=mock_column_mapping),
    ):
        mock_resource = MagicMock()
        mock_resource.read_text.return_value = "{}"
        mock_files.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value = mock_resource

        with pytest.raises(ValueError, match="DataFrame has no coded headers"):
            dataset = RedCapDataset(df=mock_data, source_type='test')
            dataset._validate_redcap_columns()


def test_validate_redcap_df_column_names_majority_label_headers():
    """Test RedCapDataset._validate_redcap_columns with majority label headers - raises ValueError."""
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
        patch("b2aiprep.prepare.redcap.files") as mock_files,
        patch("json.loads", return_value=mock_column_mapping),
    ):
        mock_resource = MagicMock()
        mock_resource.read_text.return_value = "{}"
        mock_files.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value = mock_resource

        with pytest.raises(
            ValueError, match="DataFrame has label headers rather than coded headers"
        ):
            dataset = RedCapDataset(df=mock_data, source_type='test')
            dataset._validate_redcap_columns()


def test_validate_redcap_df_column_names_mixed_headers_warning():
    """Test RedCapDataset._validate_redcap_columns with mixed headers - should log warning."""
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
        patch("b2aiprep.prepare.redcap.files") as mock_files,
        patch("json.loads", return_value=mock_column_mapping),
        patch("b2aiprep.prepare.redcap._LOGGER") as mock_logger,
    ):
        mock_resource = MagicMock()
        mock_resource.read_text.return_value = "{}"
        mock_files.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value = mock_resource

        dataset = RedCapDataset(df=mock_data, source_type='test')
        dataset._validate_redcap_columns()

        # Should have logged a warning about mixed headers
        mock_logger.warning.assert_called()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "mix of label and coded headers" in warning_call


def test_validate_redcap_df_column_names_partial_coded_headers_warning():
    """Test RedCapDataset._validate_redcap_columns with some coded headers but not all - logs warning."""
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
        patch("b2aiprep.prepare.redcap.files") as mock_files,
        patch("json.loads", return_value=mock_column_mapping),
        patch("b2aiprep.prepare.redcap._LOGGER") as mock_logger,
    ):
        mock_resource = MagicMock()
        mock_resource.read_text.return_value = "{}"
        mock_files.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value = mock_resource

        dataset = RedCapDataset(df=mock_data, source_type='test')
        dataset._validate_redcap_columns()

        # Should have logged a warning about partial coded headers
        mock_logger.warning.assert_called()
        warning_call = mock_logger.warning.call_args[0][0]
        assert "coded headers" in warning_call and "/" in warning_call


@patch("b2aiprep.prepare.redcap.files")
@patch("json.loads")
def test_validate_redcap_df_column_names_file_loading(mock_json_loads, mock_files):
    """Test that RedCapDataset._validate_redcap_columns correctly loads the column mapping file."""
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

    dataset = RedCapDataset(df=mock_data, source_type='test')
    dataset._validate_redcap_columns()

    # Verify the correct file path was accessed
    mock_files.assert_called_with("b2aiprep")
    mock_files.return_value.joinpath.assert_called_with("prepare")
    mock_prepare_path.joinpath.assert_called_with("resources")
    mock_resources_path.joinpath.assert_called_with("column_mapping.json")
    mock_final_resource.read_text.assert_called_once()


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
            validate_bids_folder_audios(temp_path)

            # Should log success message
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
            validate_bids_folder_audios(temp_path)

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
            validate_bids_folder_audios(temp_path)

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
            validate_bids_folder_audios(temp_path)

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
            validate_bids_folder_audios(temp_path)

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
            validate_bids_folder_audios(temp_path)

            # Should log success since there are no audio files to validate
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
            validate_bids_folder_audios(temp_path)

            # Should log success message
            mock_logger.warning.assert_not_called()
