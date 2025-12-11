"""Tests for the RedCapDataset class."""

import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from b2aiprep.prepare.redcap import RedCapDataset


class TestRedCapDataset:
    """Test suite for the new RedCapDataset class."""

    def test_redcap_dataset_creation(self):
        """Test basic RedCapDataset creation."""
        test_data = {
            'record_id': [1, 2, 3],
            'redcap_repeat_instrument': ['Participant', 'Session', 'Recording'],
            'redcap_repeat_instance': [1, 1, 1],
            'session_id': ['session1', 'session1', 'session1']
        }
        
        df = pd.DataFrame(test_data)
        dataset = RedCapDataset(df=df, source_type='test')
        
        assert len(dataset.df) == 3
        assert dataset.source_type == 'test'
        assert 'record_id' in dataset.df.columns

    def test_redcap_dataset_summary(self):
        """Test RedCapDataset get_summary method."""
        test_data = {
            'record_id': [1, 1, 2, 2],
            'redcap_repeat_instrument': ['Participant', 'Session', 'Participant', 'Recording'],
            'redcap_repeat_instance': [1, 1, 1, 1]
        }
        
        df = pd.DataFrame(test_data)
        dataset = RedCapDataset(df=df, source_type='test')
        
        summary = dataset.get_summary()
        
        assert summary['source_type'] == 'test'
        assert summary['total_rows'] == 4
        assert summary['unique_participants'] == 2
        assert 'Participant' in summary['instruments']
        assert 'Session' in summary['instruments']

    def test_redcap_dataset_validation(self):
        """Test RedCapDataset validate method."""
        # Valid data
        test_data = {
            'record_id': [1, 2, 3],
            'redcap_repeat_instrument': ['Participant', 'Session', 'Recording']
        }
        
        df = pd.DataFrame(test_data)
        dataset = RedCapDataset(df=df, source_type='test')
        
        validation = dataset.validate()
        
        assert validation['valid'] == True
        assert validation['stats']['total_rows'] == 3

    def test_redcap_dataset_validation_missing_columns(self):
        """Test RedCapDataset validation with missing required columns."""
        # Missing required columns
        test_data = {
            'some_column': [1, 2, 3]
        }
        
        df = pd.DataFrame(test_data)
        dataset = RedCapDataset(df=df, source_type='test')
        
        validation = dataset.validate()
        
        assert validation['valid'] == False
        assert len(validation['issues']) > 0

    @patch("pandas.read_csv")
    @patch("pathlib.Path.exists")
    def test_redcap_dataset_from_redcap(self, mock_exists, mock_read_csv):
        """Test RedCapDataset.from_redcap class method."""
        mock_exists.return_value = True
        mock_data = pd.DataFrame({
            'record_id': [1, 2],
            'redcap_repeat_instrument': ['Participant', 'Session']
        })
        mock_read_csv.return_value = mock_data
    
        with patch.object(RedCapDataset, '_validate_redcap_columns'):
            dataset = RedCapDataset.from_redcap("dummy_path.csv")
            
            assert dataset.source_type == 'redcap'
            assert len(dataset.df) == 2
            assert 'record_id' in dataset.df.columns

    @patch("pathlib.Path.iterdir")
    @patch("pathlib.Path.is_dir")
    @patch("pathlib.Path.rglob")
    def test_redcap_dataset_from_reproschema(self, mock_rglob, mock_is_dir, mock_iterdir):
        """Test RedCapDataset.from_reproschema class method."""
        # Mock directory structure
        mock_is_dir.return_value = True
        mock_iterdir.return_value = [Path("subject1")]
        mock_rglob.return_value = [Path("audio1.wav")]
        
        with patch.object(RedCapDataset, '_convert_reproschema_to_redcap') as mock_convert:
            mock_convert.return_value = pd.DataFrame({
                'record_id': [1],
                'redcap_repeat_instrument': ['Participant']
            })

            dataset = RedCapDataset.from_reproschema("audio_dir", "survey_dir")
            
            assert dataset.source_type == 'reproschema'
            assert len(dataset.df) == 1

    def test_redcap_dataset_get_df_of_repeat_instrument(self):
        """Test RedCapDataset get_df_of_repeat_instrument method."""
        test_data = {
            'record_id': [1, 1, 2, 2],
            'redcap_repeat_instrument': ['Participant', 'Session', 'Participant', 'Recording']
        }
        
        df = pd.DataFrame(test_data)
        dataset = RedCapDataset(df=df, source_type='test')
        
        # Mock instrument
        mock_instrument = MagicMock()
        mock_instrument.text = 'Participant'
        mock_instrument.get_columns.return_value = ['record_id', 'redcap_repeat_instrument']
        
        result_df = dataset.get_df_of_repeat_instrument(mock_instrument)
        
        assert len(result_df) == 2  # Two 'Participant' entries
        assert all(result_df['redcap_repeat_instrument'] == 'Participant')

    def test_redcap_dataset_str_representation(self):
        """Test RedCapDataset string representation."""
        test_data = {
            'record_id': [1, 2, 3],
            'redcap_repeat_instrument': ['Participant', 'Session', 'Recording']
        }
        
        df = pd.DataFrame(test_data)
        dataset = RedCapDataset(df=df, source_type='test')
        
        str_repr = str(dataset)
        
        assert 'RedCapDataset' in str_repr
        assert 'test' in str_repr
        assert 'rows=3' in str_repr
        assert 'columns=2' in str_repr

    @patch("pandas.read_csv")
    def test_load_redcap_csv_static_method(self, mock_read_csv):
        """Test RedCapDataset._load_redcap_csv static method."""
        # Mock the DataFrame returned by read_csv
        mock_data = pd.DataFrame({
            'record_id': [1, 2], 
            'redcap_repeat_instrument': [None, 'Session']
        })
        mock_read_csv.return_value = mock_data
        
        with patch("pathlib.Path.exists", return_value=True):
            df = RedCapDataset._load_redcap_csv("dummy_path.csv")
        
        assert df is not None
        assert 'record_id' in df.columns
        assert 'redcap_repeat_instrument' in df.columns
        # Check that None values were filled with 'Participant'
        assert df['redcap_repeat_instrument'].iloc[0] == 'Participant'

    def test_load_redcap_csv_file_not_found(self):
        """Test RedCapDataset._load_redcap_csv with non-existent file."""
        with pytest.raises(FileNotFoundError):
            RedCapDataset._load_redcap_csv("non_existent_file.csv")

    @patch("b2aiprep.prepare.redcap.files")
    @patch("json.loads")
    def test_validate_redcap_columns(self, mock_json_loads, mock_files):
        """Test RedCapDataset._validate_redcap_columns method."""
        # Mock column mapping
        column_mapping = {
            'record_id': 'Record ID',
            'session_id': 'Session ID'
        }
        mock_json_loads.return_value = column_mapping
        
        # Mock file reading
        mock_path = MagicMock()
        mock_path.read_text.return_value = '{"record_id": "Record ID"}'
        mock_files.return_value.joinpath.return_value.joinpath.return_value.joinpath.return_value = mock_path
        
        # Test with coded headers (should pass)
        test_df = pd.DataFrame({
            'record_id': [1, 2],
            'session_id': ['s1', 's2']
        })
        
        dataset = RedCapDataset(df=test_df, source_type='test')
        # Should not raise an exception
        dataset._validate_redcap_columns()

    def test_get_recordings_for_acoustic_task_invalid_task(self):
        """Test get_recordings_for_acoustic_task with invalid task."""
        test_df = pd.DataFrame({
            'record_id': [1, 2],
            'redcap_repeat_instrument': ['Participant', 'Session']
        })
        
        dataset = RedCapDataset(df=test_df, source_type='test')
        
        with pytest.raises(ValueError, match="Unrecognized"):
            dataset.get_recordings_for_acoustic_task("invalid_task")
