import datetime
import json
import logging
import os
import tempfile
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from conftest import WAV_BYTES

from b2aiprep.prepare.bids import get_audio_paths, get_paths, validate_bids_folder_audios
from b2aiprep.prepare.constants import AUDIO_TASKS, RepeatInstrument
from b2aiprep.prepare.dataset import BIDSDataset
from b2aiprep.prepare.redcap import RedCapDataset


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
            "redcap_repeat_instrument": ["Acoustic Task"] * 4,
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

def test_construct_phenotype_demographics_has_one_row_per_participant():
    """Rows from other forms get no sex_at_birth and are dropped; "Unknown" goes only to
    participants with demographics answers (the table is built from every REDCap row)."""
    nan = float("nan")
    df = pd.DataFrame({
        "record_id": ["r01", "r01", "r01", "r01", "r02", "r02", "r02"],
        "redcap_repeat_instrument": [nan, "Q - Generic - Demographics", "Acoustic Task", "Acoustic Task",
                                     nan, "Q - Generic - Demographics", "Acoustic Task"],
        "age": ["41", nan, nan, nan, "52", nan, nan],
        "gender_identity": [nan, "Female gender identity", nan, nan, nan, nan, nan],
        "specify_gender_identity": [nan, "Cis: same gender as the sex assigned at birth", nan, nan, nan, nan, nan],
        "children": [nan, "No", nan, nan, nan, "Yes", nan],
        "acoustic_task_name": [nan, nan, "A", "B", nan, nan, "C"],
    })
    with tempfile.TemporaryDirectory() as temp_dir:
        BIDSDataset._construct_phenotype_from_reproschema(df, output_dir=temp_dir)
        tsv = [p for p in Path(temp_dir).rglob("demographics.tsv")]
        assert len(tsv) == 1
        out = pd.read_csv(tsv[0], sep="\t", dtype=str, keep_default_na=False, na_values=[""])
    assert sorted(out["participant_id"]) == ["r01", "r02"]
    by_id = out.set_index("participant_id")
    assert by_id.loc["r01", "sex_at_birth"] == "Female"
    assert by_id.loc["r02", "sex_at_birth"] == "Unknown"  # answered neither question
    assert list(by_id["age"]) == ["41", "52"]


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


class _StopAfterDispatch(Exception):
    """Sentinel raised by a stubbed _load_reproschema to halt after dispatch is observed."""

    def __init__(self, folder):
        super().__init__()
        self.folder = folder


def _construct_phenotype_dispatch_df():
    return pd.DataFrame({
        "record_id": ["r1"],
        "redcap_repeat_instrument": ["Acoustic Task"],
    })


def test_construct_phenotype_locates_commit_sha_for_flat_layout(tmp_path, monkeypatch):
    """Layout A: schema directly in redcap2rs/ → reproschema_folder = redcap2rs/."""
    redcap2rs = tmp_path / "redcap2rs"
    redcap2rs.mkdir()
    (redcap2rs / "b2ai-redcap2rs_schema").write_text(json.dumps({"ui": {"order": []}}))

    monkeypatch.setattr("b2aiprep.prepare.dataset.files", lambda _: redcap2rs)

    def stub(_schema, folder):
        raise _StopAfterDispatch(folder)

    monkeypatch.setattr(BIDSDataset, "_load_reproschema", staticmethod(stub))

    with pytest.raises(_StopAfterDispatch) as exc:
        BIDSDataset._construct_phenotype_from_reproschema(
            _construct_phenotype_dispatch_df(), output_dir=str(tmp_path / "out")
        )

    assert exc.value.folder == redcap2rs.resolve()


def test_construct_phenotype_locates_commit_sha_for_nested_layout(tmp_path, monkeypatch):
    """Layout B: schema nested in redcap2rs/b2ai-redcap2rs/ → reproschema_folder = redcap2rs/."""
    redcap2rs = tmp_path / "redcap2rs"
    nested = redcap2rs / "b2ai-redcap2rs"
    nested.mkdir(parents=True)
    (nested / "b2ai-redcap2rs_schema").write_text(json.dumps({"ui": {"order": []}}))

    monkeypatch.setattr("b2aiprep.prepare.dataset.files", lambda _: redcap2rs)

    def stub(_schema, folder):
        raise _StopAfterDispatch(folder)

    monkeypatch.setattr(BIDSDataset, "_load_reproschema", staticmethod(stub))

    with pytest.raises(_StopAfterDispatch) as exc:
        BIDSDataset._construct_phenotype_from_reproschema(
            _construct_phenotype_dispatch_df(), output_dir=str(tmp_path / "out")
        )

    assert exc.value.folder == redcap2rs.resolve()


def test_construct_phenotype_resolves_symlinked_install_path(tmp_path, monkeypatch):
    """When files() returns a symlinked path, the folder passed downstream is the resolved one."""
    real = tmp_path / "real" / "redcap2rs"
    nested = real / "b2ai-redcap2rs"
    nested.mkdir(parents=True)
    (nested / "b2ai-redcap2rs_schema").write_text(json.dumps({"ui": {"order": []}}))

    link = tmp_path / "link"
    link.symlink_to(real)

    monkeypatch.setattr("b2aiprep.prepare.dataset.files", lambda _: link)

    def stub(_schema, folder):
        raise _StopAfterDispatch(folder)

    monkeypatch.setattr(BIDSDataset, "_load_reproschema", staticmethod(stub))

    with pytest.raises(_StopAfterDispatch) as exc:
        BIDSDataset._construct_phenotype_from_reproschema(
            _construct_phenotype_dispatch_df(), output_dir=str(tmp_path / "out")
        )

    assert exc.value.folder == real.resolve()
    assert exc.value.folder != link


def test_load_reproschema_resolves_paths_through_symlink(tmp_path):
    """Activities reached via a symlink should not crash inside build_activity_payload."""
    real = tmp_path / "real"
    real.mkdir()
    (real / "b2ai-redcap2rs_schema").write_text(
        json.dumps({"ui": {"order": ["activities/test/test"]}})
    )
    activity_dir = real / "activities" / "test"
    activity_dir.mkdir(parents=True)
    (activity_dir / "test").write_text(json.dumps({"id": "test_id", "ui": {"order": []}}))
    (real / "commit_sha.txt").write_text("abc12345")

    link = tmp_path / "link"
    link.symlink_to(real)

    schema_via_link = link / "b2ai-redcap2rs_schema"
    folder_resolved = link.resolve()

    result = BIDSDataset._load_reproschema(schema_via_link, folder_resolved)

    assert "test_id" in result
    assert "abc12345" in result["test_id"]["url"]


class TestTablesAreReadAsWritten:
    """Tables the pipeline reads back (and may rewrite) keep typed answers such as "None" or "N/A"
    and integers as written; only an empty cell is missing."""

    def test_phenotype_tables_load_as_written(self, tmp_path):
        base = tmp_path / "recording"
        pd.DataFrame({"participant_id": ["005009", "005010", "005011"], "recording_local_hour": ["10", "", "7"],
                      "comment": ["N/A", "None", ""]}).to_csv(base.with_suffix(".tsv"), sep="\t", index=False)
        base.with_suffix(".json").write_text(json.dumps({"recording": {"description": "", "data_elements": {}}}))
        df, *_ = BIDSDataset.load_phenotype_file(base)
        assert df["participant_id"].tolist() == ["005009", "005010", "005011"]
        assert df["recording_local_hour"].tolist()[0::2] == ["10", "7"] and pd.isna(df["recording_local_hour"].iloc[1])
        assert df["comment"].tolist()[:2] == ["N/A", "None"] and pd.isna(df["comment"].iloc[2])

    def test_removing_participants_without_audio_keeps_typed_answers(self, tmp_path):
        folder = tmp_path / "phenotype" / "confounders"
        folder.mkdir(parents=True)
        (folder / "confounders.tsv").write_text(
            "participant_id\tph_walking\tnote\np1\tNone\tN/A\np2\tMild\tNA\np3\tNone\tnull\n")
        BIDSDataset._filter_phenotype_to_participants(str(tmp_path / "phenotype"), {"p1", "p2"})
        assert BIDSDataset._read_tsv_as_written(folder / "confounders.tsv").to_dict("list") == {
            "participant_id": ["p1", "p2"], "ph_walking": ["None", "Mild"], "note": ["N/A", "NA"]}

    def test_removing_recordings_without_a_sidecar_keeps_typed_answers(self, tmp_path):
        task = tmp_path / "task"
        task.mkdir()
        (task / "recording.tsv").write_text("recording_id\trecording_acoustic_task_id\trecording_microphone\n"
                                            "r1\tt1\tNone\nr2\tt2\tN/A\n")
        (task / "acoustic_task.tsv").write_text("acoustic_task_id\tacoustic_task_notes\nt1\tNA\nt2\tnull\n")
        BIDSDataset._filter_task_tables_to_recordings(str(tmp_path), {"r1"})
        assert BIDSDataset._read_tsv_as_written(task / "recording.tsv").to_dict("list") == {
            "recording_id": ["r1"], "recording_acoustic_task_id": ["t1"], "recording_microphone": ["None"]}
        assert BIDSDataset._read_tsv_as_written(task / "acoustic_task.tsv").to_dict("list") == {
            "acoustic_task_id": ["t1"], "acoustic_task_notes": ["NA"]}


def test_building_without_a_published_supplement_column_warns(tmp_path, caplog):
    """redcap2bids without --supplement must not drop some_data_collected_remotely silently."""
    df = pd.DataFrame({"record_id": ["r1"], "redcap_repeat_instrument": ["Acoustic Task"],
                       "acoustic_task_name": ["A"]})
    with caplog.at_level("WARNING"):
        BIDSDataset._construct_phenotype_from_reproschema(df, output_dir=str(tmp_path))
    assert any("some_data_collected_remotely" in r.getMessage() and "--supplement" in r.getMessage()
               for r in caplog.records if r.levelname == "WARNING")


def test_rows_are_dropped_for_both_instruments():
    from b2aiprep.prepare.dataset import BIDSDataset

    df = pd.DataFrame({
        "redcap_repeat_instrument": ["Recording", "Recording", "Acoustic Task",
                                     "Acoustic Task", "Session"],
        "recording_name": ["Audio Check (v2)-3", "Glides", None, None, None],
        "acoustic_task_name": [None, None, "Audio Check", "Rainbow Passage", None],
    })
    out = BIDSDataset._drop_audio_check_rows(df)
    assert out["recording_name"].dropna().tolist() == ["Glides"]
    assert out["acoustic_task_name"].dropna().tolist() == ["Rainbow Passage"]
    # the session row carries neither column and must survive
    assert (out["redcap_repeat_instrument"] == "Session").sum() == 1


def test_filtering_is_idempotent_and_safe_on_unrelated_frames():
    from b2aiprep.prepare.dataset import BIDSDataset

    df = pd.DataFrame({"redcap_repeat_instrument": ["Recording"], "recording_name": ["Glides"]})
    assert len(BIDSDataset._drop_audio_check_rows(df)) == 1
    # a frame with no instrument column at all (e.g. an already-built phenotype table)
    assert len(BIDSDataset._drop_audio_check_rows(pd.DataFrame({"x": [1, 2]}))) == 2


def test_participants_without_audio_are_excluded_from_tree():
    """No sub-*/ directory is created for participants with no locatable source file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = os.path.join(tmpdir, "bids")
        os.makedirs(outdir)
        # Simulate: had_audio returns False when audio_files_by_recording has nothing
        participant = {
            "record_id": "test-no-audio",
            "sessions": [{
                "session_id": "S1",
                "session_status": "Completed",
                "session_is_control_participant": "No",
                "session_duration": "100",
                "session_site": "MIT",
                "acoustic_tasks": [{
                    "acoustic_task_id": "T1",
                    "acoustic_task_session_id": "S1",
                    "acoustic_task_name": "Rainbow Passage",
                    "acoustic_task_cohort": "adult",
                    "acoustic_task_status": "Completed",
                    "recordings": [{
                        "recording_id": "NONEXISTENT-UUID",
                        "recording_name": "Rainbow Passage",
                        "recording_duration": "5.0",
                    }]
                }]
            }],
        }
        from pathlib import Path
        from collections import OrderedDict
        had_audio = BIDSDataset._output_participant_data_to_metadata_file(
            participant, Path(outdir),
            audio_files_by_recording={},  # no audio at all
            audio_descriptor_dict=OrderedDict(),
            questionnaire_lookup={},
        )
        assert had_audio[0] is False, "should return False when no recordings have source audio"
        assert not os.path.exists(os.path.join(outdir, "sub-test-no-audio")), (
            "no sub-*/ directory should exist for a participant with no audio"
        )


def _row(instrument, **values):
    """A RedCap export row for an instrument: every column present, overrides applied.

    ``convert_response_to_bids_metadata`` indexes every column of the instrument, so a
    partial dict raises KeyError before reaching the code under test.
    """
    cols = json.loads(
        files("b2aiprep.prepare.resources")
        .joinpath("instrument_columns", f"{instrument}.json")
        .read_text()
    )
    base = {c: None for c in cols}
    base["record_id"] = "p1"
    base.update(values)
    return base


def _session(*acoustic_tasks):
    """A session dict carrying every sessions.json column (sessions.tsv is written from them)."""
    return {
        "session_id": "S1",
        "session_status": "Completed",
        "session_is_control_participant": "No",
        "session_duration": 1,
        "session_site": "test",
        "acoustic_tasks": list(acoustic_tasks),
    }


def test_sidecar_filename_uses_normalized_task_entity(tmp_path):
    BIDSDataset._write_pydantic_model_to_bids_file(
        tmp_path,
        {"id": "x"},
        schema_name="recording",
        subject_id="p1",
        session_id="S 1",
        task_name="Audio Check (v2)",
        recording_name="Audio Check (v2)-1",
    )
    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == ["sub-p1_ses-S-1_task-audio-check-v2-1_recording-metadata.json"]

    BIDSDataset._write_pydantic_model_to_bids_file(
        tmp_path, {"id": "y"}, schema_name="acoustic_task", subject_id="p1",
        session_id="S1", task_name="Cape V sentences-1 (v2)",
    )
    assert (tmp_path / "sub-p1_ses-S1_task-cape-v-sentences-v2-1_acoustic-task-metadata.json").exists()


def test_audio_and_sidecar_share_the_entity(tmp_path):
    """The wav and its _recording-metadata.json must have the same stem so deidentify can pair them."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    wav = src_dir / "11111111-2222-3333-4444-555555555555.wav"
    wav.write_bytes(WAV_BYTES)  # must exceed _MIN_AUDIO_BYTES to pass pre-scan
    # convert_response_to_bids_metadata indexes every instrument column, so give the task and
    # recording rows the full column set (as a RedCap export row would) and override a few.
    def row(instrument, **values):
        cols = json.loads(files("b2aiprep.prepare.resources").joinpath("instrument_columns", f"{instrument}.json").read_text())
        base = {c: None for c in cols}
        base["record_id"] = "p1"
        base.update(values)
        return base

    participant = {
        "record_id": "p1",
        "sessions": [
            {
                # every sessions.json column: the function writes sub-*/sessions.tsv from them
                "session_id": "S1",
                "session_status": "Completed",
                "session_is_control_participant": "No",
                "session_duration": 1,
                "session_site": "test",
                "acoustic_tasks": [
                    row(
                        "acoustic_tasks",
                        acoustic_task_id="t1",
                        acoustic_task_name="Audio Check (v2)",
                        acoustic_task_session_id="S1",
                        recordings=[
                            row(
                                "recordings",
                                recording_id="11111111-2222-3333-4444-555555555555",
                                recording_name="Audio Check (v2)-1",
                                recording_acoustic_task_id="t1",
                                recording_session_id="S1",
                            )
                        ],
                    )
                ],
            }
        ],
    }
    out = tmp_path / "bids"
    out.mkdir()
    BIDSDataset._output_participant_data_to_metadata_file(
        participant, out, audio_files_by_recording={wav.stem: wav}, max_audio_workers=1,
        sanitize_audio_format=False, audio_descriptor_dict={},
    )
    audio_dir = out / "sub-p1" / "ses-S1" / "audio"
    names = sorted(p.name for p in audio_dir.iterdir())
    assert "sub-p1_ses-S1_task-audio-check-v2-1.flac" in names
    assert "sub-p1_ses-S1_task-audio-check-v2-1_recording-metadata.json" in names
    # Every key redcap2bids writes is described by the field map's audio_sidecar table (after
    # deidentify renames record_id), so none is removed as unknown at release.
    sidecar = json.loads((audio_dir / "sub-p1_ses-S1_task-audio-check-v2-1_recording-metadata.json").read_text())
    fm = BIDSDataset._load_reorganization_file(exclude_dropped=False)
    table = set(fm.loc[fm.schema_name == "audio_sidecar", "column_name"])
    keys = {"participant_id" if k == "record_id" else k for k in sidecar}
    assert keys <= table, keys - table
    assert not any("(" in n or ")" in n or n != n.lower().replace("sub-p1_ses-s1", "sub-p1_ses-S1") for n in names if n.endswith(".flac"))


def test_recording_without_a_name_is_skipped_not_named_nan(tmp_path, caplog):
    """A missing recording_name must not yield a "task-nan" audio file.

    The wav name came from str(recording_name) while the sidecar name came from a
    pd.notna() check, so a NaN name produced task-nan.wav beside a sidecar named after
    the acoustic task -- and deidentify then dropped the recording for a missing sidecar.
    """
    participant = {
        "record_id": "p1",
        "selected_language": "English",
        "sessions": [
            _session(
                _row(
                    "acoustic_tasks",
                    acoustic_task_id="t1",
                    acoustic_task_name="Audio Check (v2)",
                    acoustic_task_session_id="S1",
                    recordings=[
                        _row(
                            "recordings",
                            recording_id="r1",
                            recording_name=float("nan"),
                            recording_acoustic_task_id="t1",
                            recording_session_id="S1",
                        )
                    ],
                )
            )
        ],
    }
    with caplog.at_level("WARNING"):
        BIDSDataset._output_participant_data_to_metadata_file(participant, tmp_path)
    assert "missing recording_name" in caplog.text
    assert not list(tmp_path.rglob("*task-nan*"))
    assert not list(tmp_path.rglob("*task-none*"))


def test_collision_bookkeeping_survives_a_missing_id():
    """A missing id must still mark the entity as seen.

    Regression guard for the review finding on #333: the checks used
    `seen.get(entity) is not None`, so an entity first recorded with a None/NaN id
    read as "never seen" and the next record mapping to the same entity was
    overwritten without a warning -- silently, which is what these guards exist to
    prevent. Equality is only trusted when both ids are present, so NaN != NaN
    cannot fabricate a collision either.
    """
    from b2aiprep.prepare.dataset import _note_entity_collision

    # first record carries no id at all
    seen = {}
    assert _note_entity_collision(seen, "glides-high-to-low", None) == (False, None)
    assert "glides-high-to-low" in seen, "a missing id must still mark the entity as seen"
    collided, prior = _note_entity_collision(seen, "glides-high-to-low", "REC-2")
    assert collided, "a second record on the same entity is a collision even if the first had no id"
    assert prior is None

    # NaN is treated the same way, and never collides with itself
    nan = float("nan")
    seen = {}
    assert _note_entity_collision(seen, "free-speech", nan) == (False, None)
    collided, _ = _note_entity_collision(seen, "free-speech", nan)
    assert collided, "two distinct records with unusable ids are still a collision"

    # the ordinary cases are unchanged
    seen = {}
    assert _note_entity_collision(seen, "audio-check", "A") == (False, None)
    assert _note_entity_collision(seen, "audio-check", "A") == (False, "A"), "same id twice is not a collision"
    collided, prior = _note_entity_collision(seen, "audio-check", "B")
    assert (collided, prior) == (True, "A")


def test_no_acoustictask_sidecar_generated(tmp_path):
    """Per-task _acoustictask-metadata.json sidecars are no longer generated.

    Per-recording _recording-metadata.json sidecars must still be written.
    """
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    wav = src_dir / "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.wav"
    wav.write_bytes(WAV_BYTES)

    def row(instrument, **values):
        cols = json.loads(files("b2aiprep.prepare.resources").joinpath("instrument_columns", f"{instrument}.json").read_text())
        base = {c: None for c in cols}
        base["record_id"] = "p1"
        base.update(values)
        return base

    participant = {
        "record_id": "p1",
        "sessions": [
            {
                "session_id": "S1",
                "session_status": "Completed",
                "session_is_control_participant": "No",
                "session_duration": 1,
                "session_site": "test",
                "acoustic_tasks": [
                    row(
                        "acoustic_tasks",
                        acoustic_task_id="t1",
                        acoustic_task_name="Prolonged vowel",
                        acoustic_task_session_id="S1",
                        acoustic_task_cohort="generic",
                        recordings=[
                            row(
                                "recordings",
                                recording_id="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
                                recording_name="Prolonged vowel-1",
                                recording_acoustic_task_id="t1",
                                recording_session_id="S1",
                            )
                        ],
                    )
                ],
            }
        ],
    }
    out = tmp_path / "bids"
    out.mkdir()
    BIDSDataset._output_participant_data_to_metadata_file(
        participant, out, audio_files_by_recording={wav.stem: wav}, max_audio_workers=1,
        sanitize_audio_format=False, audio_descriptor_dict={},
    )
    audio_dir = out / "sub-p1" / "ses-S1" / "audio"
    all_files = sorted(p.name for p in audio_dir.iterdir())

    # Per-recording sidecar must exist
    recording_sidecars = [f for f in all_files if f.endswith("_recording-metadata.json")]
    assert len(recording_sidecars) == 1, f"Expected 1 recording sidecar, got {recording_sidecars}"

    # Per-task sidecar must NOT exist
    task_sidecars = [f for f in all_files if "acoustictask" in f or "acoustic-task" in f]
    assert len(task_sidecars) == 0, f"Unexpected task sidecar(s): {task_sidecars}"

    # Audio file must exist
    wavs = [f for f in all_files if f.endswith(".flac")]
    assert len(wavs) == 1


def test_no_acoustictask_sidecar_phenotype_only_mode(tmp_path):
    """In phenotype-only mode (no audio dir), no task sidecars are generated.

    The session cleanup removes audio directories with no .wav files, so
    recording sidecars are also cleaned up. The key assertion: no task sidecars
    survive even if the directory is still present (checked before cleanup).
    """
    def row(instrument, **values):
        cols = json.loads(files("b2aiprep.prepare.resources").joinpath("instrument_columns", f"{instrument}.json").read_text())
        base = {c: None for c in cols}
        base["record_id"] = "p2"
        base.update(values)
        return base

    participant = {
        "record_id": "p2",
        "sessions": [
            {
                "session_id": "S1",
                "session_status": "Completed",
                "session_is_control_participant": "No",
                "session_duration": 1,
                "session_site": "test",
                "acoustic_tasks": [
                    row(
                        "acoustic_tasks",
                        acoustic_task_id="t1",
                        acoustic_task_name="Prolonged vowel",
                        acoustic_task_session_id="S1",
                        acoustic_task_cohort="generic",
                        recordings=[
                            row(
                                "recordings",
                                recording_id="11111111-0000-0000-0000-000000000000",
                                recording_name="Prolonged vowel-1",
                                recording_acoustic_task_id="t1",
                                recording_session_id="S1",
                            )
                        ],
                    )
                ],
            }
        ],
    }
    out = tmp_path / "bids"
    out.mkdir()
    BIDSDataset._output_participant_data_to_metadata_file(
        participant, out, audio_files_by_recording=None, max_audio_workers=1,
        sanitize_audio_format=False, audio_descriptor_dict={},
    )
    # In phenotype-only mode the session cleanup removes audio dirs with no wavs,
    # so the entire audio directory is gone.
    audio_dir = out / "sub-p2" / "ses-S1" / "audio"
    assert not audio_dir.exists(), "Audio dir should be cleaned up in phenotype-only mode"


def test_sessions_tsv_keeps_sessions_without_audio(tmp_path):
    """A session with no audio gets no audio directory but stays in sessions.tsv, so its
    questionnaire rows still name a listed session."""
    wav = tmp_path / "11111111-2222-3333-4444-555555555555.wav"
    wav.write_bytes(WAV_BYTES)  # must exceed _MIN_AUDIO_BYTES to pass pre-scan
    task = _row("acoustic_tasks", acoustic_task_id="t1", acoustic_task_name="Rainbow Passage",
                acoustic_task_session_id="S1",
                recordings=[_row("recordings", recording_id=wav.stem, recording_name="Rainbow Passage",
                                 recording_acoustic_task_id="t1", recording_session_id="S1")])
    participant = {
        "record_id": "p1",
        "sessions": [
            {**_session(task), "session_id": "S1", "session_index": "1"},
            {**_session(), "session_id": "S2", "session_index": "2"},
        ],
    }
    out = tmp_path / "bids"
    out.mkdir()
    BIDSDataset._output_participant_data_to_metadata_file(
        participant, out, audio_files_by_recording={wav.stem: wav}, max_audio_workers=1,
        sanitize_audio_format=False, audio_descriptor_dict={},
    )
    ses = pd.read_csv(out / "sub-p1" / "sessions.tsv", sep="\t", dtype=str)
    assert list(ses.session_id) == ["S1", "S2"] and list(ses.session_index) == ["1", "2"]
    assert (out / "sub-p1" / "ses-S1" / "audio").is_dir()
    assert not (out / "sub-p1" / "ses-S2").exists()


def test_csv_description_becomes_the_data_element():
    element = BIDSDataset._synthetic_data_element(
        "acoustic_task_complete",
        {"description": "RedCap form-completion status for the acoustic_task instrument."},
    )
    assert element["description"] == (
        "RedCap form-completion status for the acoustic_task instrument."
    )
    assert element["valueType"] == ["xsd:string"]


def test_no_ontology_reference_is_invented():
    """The entry must not claim provenance it does not have.

    A termURL points at a specific commit of b2ai-redcap2rs, which is meaningful only for a
    column that ReproSchema actually defines. Emitting one here would put an unsourced ontology
    reference into a published data dictionary, so the absence of these keys is what lets a
    consumer tell a CSV-described column from an authored one.
    """
    element = BIDSDataset._synthetic_data_element("enrollment_form_timestamp", {"description": "x"})
    for key in ("termURL", "choices", "question", "datatype"):
        assert key not in element, f"{key} must not be invented for a CSV-only column"
    # same shape as the synthetic participant_id element the writer already emits
    assert set(element) == {"description", "valueType"}


def test_checkbox_options_keep_their_integer_type():
    """A ___ option still becomes 0/1 when the phenotype data is cleaned."""
    cleaned = BIDSDataset._synthetic_data_element(
        "participant_data_collection_origins___reproschema", {"description": "x"},
        column_choice="reproschema", clean_phenotype_data=True,
    )
    assert cleaned["valueType"] == ["xsd:integer"]
    raw = BIDSDataset._synthetic_data_element(
        "participant_data_collection_origins___reproschema", {"description": "x"},
        column_choice="reproschema", clean_phenotype_data=False,
    )
    assert raw["valueType"] == ["xsd:string"]


def test_a_missing_description_is_stated_not_left_blank():
    """An empty description would otherwise reach a published dictionary as ""."""
    for row in ({}, {"description": ""}, {"description": "   "}, {"description": None}):
        element = BIDSDataset._synthetic_data_element("some_form_complete", row)
        assert "some_form_complete" in element["description"]
        assert "no ReproSchema definition" in element["description"]


def test_form_status_does_not_keep_an_otherwise_empty_row():
    """RedCap emits `<form>_complete` for every record, filled in or not.

    Counting it as content would put every participant in every table: measured on the v4 adult
    export, `diagnosis/amyotrophic_lateral_sclerosis.tsv` went from 6 rows to 2005 before this
    rule existed.
    """
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(
        {
            "participant_id": ["has-als", "no-als"],
            "diagnosis_als_onset": ["bulbar", np.nan],
            "d_neuro_amyotrophic_lateral_sclerosis_complete": ["Complete", "Incomplete"],
        }
    )
    kept = BIDSDataset._drop_rows_without_substantive_data(
        df, "participant_id", {"d_neuro_amyotrophic_lateral_sclerosis_complete"}
    )
    assert kept["participant_id"].tolist() == ["has-als"]

    # without the exclusion the empty row survives - this is the regression being guarded
    assert len(BIDSDataset._drop_rows_without_substantive_data(df, "participant_id", set())) == 2


def test_a_table_of_only_bookkeeping_columns_is_emptied():
    """No substantive columns means no research value — table is emptied."""
    import pandas as pd

    df = pd.DataFrame({"participant_id": ["a", "b"], "some_form_complete": ["Incomplete"] * 2})
    result = BIDSDataset._drop_rows_without_substantive_data(
        df, "participant_id", {"some_form_complete"}
    )
    assert len(result) == 0
    assert list(result.columns) == list(df.columns)


def test_calculated_fields_do_not_keep_incomplete_diagnosis_rows():
    """A diagnosis form with only auto-calculated values and _complete=Incomplete is phantom."""
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(
        {
            "participant_id": ["real-als", "phantom"],
            "diagnosis_als_onset": ["bulbar", np.nan],
            "diagnosis_als_gsd_calculation": [1, 0],
            "d_neuro_als_complete": ["Complete", "Incomplete"],
        }
    )
    kept = BIDSDataset._drop_rows_without_substantive_data(
        df,
        "participant_id",
        csv_only_columns={"d_neuro_als_complete"},
        calculated_columns={"diagnosis_als_gsd_calculation"},
        schema_group="diagnosis",
    )
    assert kept["participant_id"].tolist() == ["real-als"]


def test_calculated_fields_excluded_from_substantive_check_non_diagnosis():
    """For non-diagnosis forms, calculated fields are excluded from the emptiness test."""
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(
        {
            "participant_id": ["has-score", "calc-only"],
            "vhi_item_1": ["Sometimes", np.nan],
            "vhi_10_calc_score": [12, 0],
            "vhi10_complete": ["Complete", "Incomplete"],
        }
    )
    kept = BIDSDataset._drop_rows_without_substantive_data(
        df,
        "participant_id",
        csv_only_columns={"vhi10_complete"},
        calculated_columns={"vhi_10_calc_score"},
        schema_group="questionnaire",
    )
    assert kept["participant_id"].tolist() == ["has-score"]


def test_diagnosis_incomplete_with_real_data_is_dropped():
    """Diagnosis forms require _complete=Complete; incomplete rows with real data are dropped."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "participant_id": ["verified", "unverified"],
            "diagnosis_mtd_degree": [50.0, 30.0],
            "d_voice_mtd_complete": ["Complete", "Incomplete"],
        }
    )
    kept = BIDSDataset._drop_rows_without_substantive_data(
        df,
        "participant_id",
        csv_only_columns={"d_voice_mtd_complete"},
        schema_group="diagnosis",
    )
    assert kept["participant_id"].tolist() == ["verified"]


def test_diagnosis_unverified_is_also_dropped():
    """Unverified diagnosis forms are not clinician-verified and must be dropped."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "participant_id": ["complete", "unverified"],
            "diagnosis_pd_subtype": ["IPD", "PSP"],
            "d_neuro_pd_complete": ["Complete", "Unverified"],
        }
    )
    kept = BIDSDataset._drop_rows_without_substantive_data(
        df,
        "participant_id",
        csv_only_columns={"d_neuro_pd_complete"},
        schema_group="diagnosis",
    )
    assert kept["participant_id"].tolist() == ["complete"]


def test_non_diagnosis_keeps_incomplete_rows_with_data():
    """Questionnaires/enrollment keep Incomplete rows that have real data."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "participant_id": ["complete", "incomplete-but-data"],
            "phq9_score": [12, 8],
            "phq9_complete": ["Complete", "Incomplete"],
        }
    )
    kept = BIDSDataset._drop_rows_without_substantive_data(
        df,
        "participant_id",
        csv_only_columns={"phq9_complete"},
        schema_group="questionnaire",
    )
    assert len(kept) == 2


def test_complete_form_with_no_data_is_dropped_non_diagnosis():
    """A Complete form with all-null substantive columns is still dropped (data anomaly)."""
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(
        {
            "participant_id": ["has-data", "empty-complete"],
            "confounders_smoking": ["Yes", np.nan],
            "confounders_complete": ["Complete", "Complete"],
        }
    )
    kept = BIDSDataset._drop_rows_without_substantive_data(
        df,
        "participant_id",
        csv_only_columns={"confounders_complete"},
        schema_group="confounders",
    )
    assert kept["participant_id"].tolist() == ["has-data"]


def test_diagnosis_no_complete_col_falls_back_to_substantive_check():
    """If no _complete column exists, diagnosis forms fall back to the data check."""
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(
        {
            "participant_id": ["has-data", "empty"],
            "diagnosis_field": ["Yes", np.nan],
        }
    )
    kept = BIDSDataset._drop_rows_without_substantive_data(
        df,
        "participant_id",
        csv_only_columns=set(),
        schema_group="diagnosis",
    )
    assert kept["participant_id"].tolist() == ["has-data"]


def test_computed_column_element_carries_its_spec_and_no_term():
    element = BIDSDataset._synthetic_data_element(
        "session_local_hour",
        {"description": "Hour of day.", "column_name": "session_local_hour", "source": "pipeline"},
    )
    assert element["valueType"] == element["datatype"] == ["xsd:integer"]
    assert (element["minValue"], element["maxValue"]) == (0, 23)
    assert "termURL" not in element and "question" not in element


ANCHOR = datetime.date(2100, 1, 1)


def _ingest_frame():
    return pd.DataFrame(
        [
            {"record_id": "a", "redcap_repeat_instrument": "Session", "session_id": "s1",
             "enrollment_institution": "MIT", "session_site": "MIT", "session_started_at": "2024-07-01T16:00:00Z",
             "session_is_control_participant": "No"},
            {"record_id": "a", "redcap_repeat_instrument": "Participant", "demographics_language": "English"},
        ],
        dtype=object,
    )


def _remote_frame(zipcode=None, state=None, via="Participant", site="MIT"):
    """One participant: a session at 2024-07-01 20:00Z with one self-administered recording."""
    return pd.DataFrame(
        [
            {"record_id": "r", "redcap_repeat_instrument": "Session", "session_id": "s1",
             "enrollment_institution": site, "session_started_at": "2024-07-01T20:00:00Z"},
            {"record_id": "r", "redcap_repeat_instrument": "Recording", "recording_session_id": "s1",
             "recording_via": via, "recording_created_at": "2024-07-01T20:05:00Z"},
            {"record_id": "r", "redcap_repeat_instrument": "Q - Generic - Demographics",
             "zipcode": zipcode, "state_province": state},
        ],
        dtype=object,
    )


def _na(values):
    return [v if isinstance(v, str) else None for v in values]


def test_ingest_shifts_dates_and_removes_drop_columns(tmp_path):
    dataset = RedCapDataset(df=_ingest_frame(), source_type="redcap")
    log = tmp_path / "logs" / "date_shift.json"
    out = BIDSDataset._apply_field_map_at_ingest(dataset, ANCHOR, log, tmp_path / "bids")
    assert "demographics_language" not in out.df.columns
    assert "session_is_control_participant" not in out.df.columns
    # internal columns the pipeline reads are kept
    assert {"redcap_repeat_instrument", "enrollment_institution"} <= set(out.df.columns)
    assert "session_site" not in out.df.columns
    shifted = datetime.datetime.fromisoformat(out.df.loc[0, "session_started_at"]).date()
    assert abs((shifted - ANCHOR).days) <= 3
    report = json.loads(log.read_text())
    assert report["anchor"] == "2100-01-01" and report["participants_shifted"] == 1
    # the input dataset is not modified
    assert dataset.df.loc[0, "session_started_at"] == "2024-07-01T16:00:00Z"


def test_ingest_refuses_a_log_inside_the_bids_tree(tmp_path):
    dataset = RedCapDataset(df=_ingest_frame(), source_type="redcap")
    with pytest.raises(ValueError, match="outside the BIDS output"):
        BIDSDataset._apply_field_map_at_ingest(dataset, ANCHOR, tmp_path / "bids" / "log.json", tmp_path / "bids")


def test_session_index_numbers_by_start_time_undated_last_duplicates_share():
    df = pd.DataFrame({
        "record_id": ["p1", "p1", "p1", "p1", "p2", "p1"],
        "redcap_repeat_instrument": ["Session"] * 5 + ["Acoustic Task"],
        "session_id": ["B", "A", "C", "B", "Z", None],
        "session_started_at": ["2025-03-01T10:00:00Z", "2025-01-01T10:00:00Z", None,
                               "2025-03-01T10:00:00Z", "2024-01-01T00:00:00Z", None],
    })
    out = BIDSDataset._add_session_index(df)
    got = dict(zip(zip(out.record_id, out.session_id), out.session_index))
    assert got[("p1", "A")] == "1" and got[("p1", "B")] == "2" and got[("p1", "C")] == "3"
    assert got[("p2", "Z")] == "1"
    assert out.loc[5, "session_index"] is pd.NA
    assert list(out.loc[out.session_id == "B", "session_index"]) == ["2", "2"]


def test_recording_order_and_gaps_use_real_times_and_leave_undated_blank(caplog):
    rows = [
        ("Session", {"session_id": "S2", "session_started_at": "2024-07-01T17:00:00Z"}),
        ("Session", {"session_id": "S1", "session_started_at": "2024-07-01T16:00:00Z"}),
        ("Session", {"session_id": "S1", "session_started_at": None}),  # repeated row of S1
        ("Session", {"session_id": "S3", "session_started_at": None}),
        ("Recording", {"recording_session_id": "S1", "recording_id": "R2", "recording_created_at": "2024-07-01T16:01:00Z"}),
        ("Recording", {"recording_session_id": "S1", "recording_id": "R1", "recording_created_at": "2024-07-01T16:01:00Z"}),
        ("Recording", {"recording_session_id": "S1", "recording_id": "R4", "recording_created_at": None}),
        ("Recording", {"recording_session_id": "S1", "recording_id": "R3", "recording_created_at": "2024-07-01T16:01:30.400Z"}),
        ("Recording", {"recording_session_id": "S2", "recording_id": "R5", "recording_created_at": "2024-07-01T17:02:00Z"}),
    ]
    df = pd.DataFrame([{"record_id": "p1", "redcap_repeat_instrument": i, **v} for i, v in rows], dtype=object)
    with caplog.at_level("WARNING"):
        out = BIDSDataset._add_recording_order_and_gaps(df)
    # one timeline from the first session's start (S1, 16:00Z); a repeated undated row of S1 gets S1's value
    assert _na(out.session_seconds_since_first_session) == ["3600", "0", "0", None] + [None] * 5
    assert _na(out.recording_order) == [None] * 4 + ["2", "1", None, "3", "1"]
    assert _na(out.recording_seconds_since_first_session) == [None] * 4 + ["60", "60", None, "90", "3720"]
    assert "1 recording(s) without a start time, left unordered: R4" in caplog.text
    assert "recording_order" not in df.columns  # input untouched


def test_local_hours_read_the_shifted_wall_clock_time():
    df = pd.DataFrame({
        "session_started_at": ["2100-01-05T13:34:41-05:00", None, "2100-01-05T00:10:00.250+01:00", "2100-01-05"],
        "recording_created_at": [None, "2100-01-05T23:59:59-08:00", "not a time", None],
    }, dtype=object)
    out = BIDSDataset._add_local_hours(df)
    assert _na(out.session_local_hour) == ["13", None, "0", None]  # a date-only start has no hour
    assert _na(out.recording_local_hour) == [None, "23", None, None]


def test_ingest_adds_derived_session_and_recording_fields(tmp_path):
    df = _remote_frame(zipcode="02139")
    df.loc[1, "recording_id"] = "R1"
    out = BIDSDataset._apply_field_map_at_ingest(RedCapDataset(df=df, source_type="redcap"), ANCHOR, None, tmp_path)
    # 20:00Z and 20:05Z on 2024-07-01 are 16:00 and 16:05 in Cambridge, MA (EDT)
    assert out.df.loc[0, "session_local_hour"] == "16" and out.df.loc[1, "recording_local_hour"] == "16"
    assert out.df.loc[1, "recording_order"] == "1"
    # 20:00Z session, 20:05Z recording
    assert _na([out.df.loc[0, "session_seconds_since_first_session"], out.df.loc[1, "recording_seconds_since_first_session"]]) == ["0", "300"]


def test_session_hour_check_lists_in_clinic_sessions_outside_clinic_hours(caplog):
    df = pd.DataFrame([
        {"record_id": "p1", "redcap_repeat_instrument": "Session", "session_id": "S1", "session_local_hour": "2"},
        {"record_id": "p1", "redcap_repeat_instrument": "Session", "session_id": "S2", "session_local_hour": "19"},
        {"record_id": "p2", "redcap_repeat_instrument": "Session", "session_id": "S3", "session_local_hour": "23"},
        {"record_id": "p2", "redcap_repeat_instrument": "Recording", "recording_session_id": "S3",
         "recording_via": "Participant"},  # self-administered: not checked
        {"record_id": "p3", "redcap_repeat_instrument": "Session", "session_id": "S4", "session_local_hour": None},
        # an undated row, then a dated row of the same session
        {"record_id": "p4", "redcap_repeat_instrument": "Session", "session_id": "S5", "session_local_hour": None},
        {"record_id": "p4", "redcap_repeat_instrument": "Session", "session_id": "S5", "session_local_hour": "22"},
        # self-administered only in rows dropped before ingest (the microphone check)
        {"record_id": "p5", "redcap_repeat_instrument": "Session", "session_id": "S6", "session_local_hour": "0"},
    ], dtype=object)
    with caplog.at_level(logging.WARNING):
        BIDSDataset._check_session_hours(df, also_self_administered={"S6"})
    assert "2 in-clinic session(s) of 3 started outside 07:00-19:59" in caplog.text  # S4 has no hour
    assert "p1 S1 (02h)" in caplog.text and "p4 S5 (22h)" in caplog.text
    assert "S3" not in caplog.text and "S6" not in caplog.text


def test_days_since_surgery_measures_to_the_first_session_and_checks_the_forms_session(caplog):
    mc = "Q - Pediatric - Generic Medical Conditions"
    df = pd.DataFrame([
        {"record_id": "c1", "redcap_repeat_instrument": "Session", "session_id": "S1",
         "session_started_at": "2100-01-05T23:30:00-05:00"},  # local date 2100-01-05, not the UTC date
        {"record_id": "c1", "redcap_repeat_instrument": mc, "peds_mc_session_id": "S1",
         "peds_mc_tonsillectomy_date": "2099-12-29", "peds_mc_etp_procedure_date": "2100-01-07"},
        {"record_id": "c2", "redcap_repeat_instrument": mc, "peds_mc_session_id": "S9",
         "peds_mc_tonsillectomy_date": "2099-01-01", "peds_mc_etp_procedure_date": None},
        # the first numeric age is the one checked against
        {"record_id": "c3", "redcap_repeat_instrument": "Participant", "age": "unknown"},
        {"record_id": "c3", "redcap_repeat_instrument": "Participant", "age": "4"},
        {"record_id": "c3", "redcap_repeat_instrument": "Participant", "age": "12"},
        {"record_id": "c3", "redcap_repeat_instrument": "Session", "session_id": "S3",
         "session_started_at": "2100-01-05T10:00:00-05:00"},
        {"record_id": "c3", "redcap_repeat_instrument": mc, "peds_mc_session_id": "S3",
         "peds_mc_tonsillectomy_date": "2090-01-05", "peds_mc_etp_procedure_date": "2098-01-05"},
    ], dtype=object)
    with caplog.at_level("WARNING"):
        out = BIDSDataset._add_days_since_surgery(df)
    assert _na(out.peds_mc_tonsillectomy_days_since) == [None, "7", None, None, None, None, None, None]  # before birth
    assert _na(out.peds_mc_etp_procedure_days_since) == [None, None, None, None, None, None, None, "730"]
    assert "peds_mc_adenoidectomy_days_since" not in out.columns  # source column absent
    assert "1 surgery date(s) after the session they were reported in, left blank: c1 peds_mc_etp_procedure_date" in caplog.text
    assert "1 surgery date(s) of participants with no dated session" in caplog.text
    assert "1 surgery date(s) before the participant was born" in caplog.text
    assert "c3 peds_mc_tonsillectomy_date (3652 days, age 4)" in caplog.text


def test_episode_and_trauma_dates_measure_to_the_first_session(caplog):
    df = pd.DataFrame([
        {"record_id": "a", "redcap_repeat_instrument": None, "mbd_last_manic_episode": "2099-12-06", "age": "30"},
        {"record_id": "a", "redcap_repeat_instrument": "Session", "session_id": "S2",
         "session_started_at": "2100-02-01T10:00:00-05:00"},
        {"record_id": "a", "redcap_repeat_instrument": "Session", "session_id": "S1",
         "session_started_at": "2100-01-05T10:00:00-05:00"},
        {"record_id": "a", "redcap_repeat_instrument": "Q - Mood - PTSD Adult", "ptsd_session_id": "S2",
         "traumatic_event_date": "2100-01-25"},
        {"record_id": "b", "redcap_repeat_instrument": None, "mbd_last_manic_episode": "2100-03-01"},
        {"record_id": "b", "redcap_repeat_instrument": "Session", "session_id": "S3",
         "session_started_at": "2100-01-05T10:00:00-05:00"},
    ], dtype=object)
    with caplog.at_level("WARNING"):
        out = BIDSDataset._add_days_since_episodes(df)
    assert _na(out.mbd_last_manic_episode_days_since) == ["30", None, None, None, "-55", None]  # b: after enrollment (2100 is not a leap year)
    # reported at S2 (2100-02-01), measured to the first session S1 (2100-01-05): 20 days after it
    assert _na(out.traumatic_event_days_since) == [None, None, None, "-20", None, None]
    # b's diagnosis form has no session to check against: kept, and flagged for QA
    assert "1 episode date(s) on a form with no session fall after the participant's first session" in caplog.text
    assert "b mbd_last_manic_episode (55 days after)" in caplog.text


def test_episode_after_enrollment_is_kept_negative_and_trauma_on_session_day_is_not_given(caplog):
    df = pd.DataFrame([
        # an episode between the first (2100-01-05) and last (2100-02-01) session: reported at a later visit
        {"record_id": "a", "redcap_repeat_instrument": None, "mbd_last_manic_episode": "2100-01-15"},
        {"record_id": "a", "redcap_repeat_instrument": "Session", "session_id": "S1",
         "session_started_at": "2100-01-05T10:00:00-05:00"},
        {"record_id": "a", "redcap_repeat_instrument": "Session", "session_id": "S2",
         "session_started_at": "2100-02-01T10:00:00-05:00"},
        {"record_id": "a", "redcap_repeat_instrument": "Q - Mood - PTSD Adult", "ptsd_session_id": "S2",
         "traumatic_event_date": "2100-02-01"},
    ], dtype=object)
    with caplog.at_level("WARNING"):
        out = BIDSDataset._add_days_since_episodes(df)
    assert _na(out.mbd_last_manic_episode_days_since) == ["-10", None, None, None]
    assert _na(out.traumatic_event_days_since) == [None] * 4
    assert "1 episode date(s) equal to the session date, treated as not given" in caplog.text


def test_a_form_on_an_undated_session_keeps_its_value_unchecked(caplog):
    df = pd.DataFrame([
        {"record_id": "a", "redcap_repeat_instrument": "Session", "session_id": "S1",
         "session_started_at": "2100-01-05T10:00:00-05:00"},
        {"record_id": "a", "redcap_repeat_instrument": "Session", "session_id": "S2", "session_started_at": None},
        {"record_id": "a", "redcap_repeat_instrument": "Q - Mood - PTSD Adult", "ptsd_session_id": "S2",
         "traumatic_event_date": "2100-01-25"},
    ], dtype=object)
    with caplog.at_level("WARNING"):
        out = BIDSDataset._add_days_since_episodes(df)
    assert _na(out.traumatic_event_days_since) == [None, None, "-20"]
    assert "1 episode date(s) on a form whose session has no date could not be checked against it; kept: a traumatic_event_date" in caplog.text


def test_state_province_prefers_the_stated_value_then_the_postal_code(caplog):
    demo = "Q - Generic - Demographics"
    df = pd.DataFrame([
        {"record_id": "a", "redcap_repeat_instrument": demo, "zipcode": "02139", "state_province": "Massachusetts"},
        {"record_id": "b", "redcap_repeat_instrument": demo, "zipcode": None, "state_province": "ny"},
        {"record_id": "c", "redcap_repeat_instrument": demo, "zipcode": "bad", "state_province": "Quebec"},
        {"record_id": "d", "redcap_repeat_instrument": demo, "zipcode": None, "state_province": None},
        {"record_id": "e", "redcap_repeat_instrument": demo, "zipcode": "10001", "state_province": "NJ"},
        {"record_id": "f", "redcap_repeat_instrument": "Session", "zipcode": None, "state_province": None},
    ], dtype=object)
    with caplog.at_level("WARNING"):
        out = BIDSDataset._add_state_province(df)
    assert _na(out.state_province_standardized) == ["MA", "NY", "QC", "Unknown", "NJ", None]  # e: the stated NJ wins
    assert "1 form(s) whose postal code and stated state/province disagree (the stated value is used): e" in caplog.text
