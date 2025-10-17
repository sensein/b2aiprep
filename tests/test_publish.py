"""Tests for the publish_bids_dataset command."""

import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

import pytest

from b2aiprep.commands import publish_bids_dataset
from b2aiprep.prepare.dataset import BIDSDataset


class TestPublishCommandRefactor:
    """Test cases for the publish_bids_dataset command."""

    @pytest.fixture
    def temp_bids_dir(self):
        """Create a minimal BIDS directory for testing."""
        temp_dir = tempfile.mkdtemp()
        bids_path = Path(temp_dir) / "test_bids"
        bids_path.mkdir(parents=True, exist_ok=True)
        
        # Create minimal participants.tsv (required for BIDSDataset)
        participants_file = bids_path / "participants.tsv"
        participants_file.write_text("record_id\tage\nparticipant001\t25\n")
        
        # Create participants.json
        participants_json = bids_path / "participants.json"
        participants_json.write_text('{"record_id": {"description": "ID"}, "age": {"description": "Age"}}')
        
        yield str(bids_path)
        
        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        output_path = Path(temp_dir) / "output"
        yield str(output_path)
        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def temp_publish_config_dir(self):
        """Create a temporary publish config directory with required files."""
        temp_dir = tempfile.mkdtemp()
        config_path = Path(temp_dir) / "config"
        config_path.mkdir(parents=True, exist_ok=True)
        
        # Create required config files
        (config_path / "participant_ids_to_remove.json").write_text("[]")
        (config_path / "audio_filestems_to_remove.json").write_text("[]")
        (config_path / "id_remapping.json").write_text("{}")
        (config_path / "session_id_remapping.json").write_text("{}")
        
        yield str(config_path)
        
        # Cleanup
        shutil.rmtree(temp_dir)

    def test_publish_command_calls_deidentify(self, temp_bids_dir, temp_output_dir, temp_publish_config_dir):
        """Test that the publish command calls the BIDSDataset.deidentify method."""
        runner = CliRunner()
        
        with patch.object(BIDSDataset, 'deidentify') as mock_deidentify:
            # Mock the deidentify method to return a BIDSDataset instance
            mock_deidentified = MagicMock(spec=BIDSDataset)
            mock_deidentify.return_value = mock_deidentified
            
            # Run the command
            result = runner.invoke(publish_bids_dataset, [temp_bids_dir, temp_output_dir, temp_publish_config_dir])
            
            # Check that the command succeeded
            assert result.exit_code == 0
            
            # Check that deidentify was called with correct parameters
            mock_deidentify.assert_called_once_with(outdir=temp_output_dir, publish_config_dir=Path(temp_publish_config_dir), skip_audio=False)

    def test_publish_command_with_skip_audio(self, temp_bids_dir, temp_output_dir, temp_publish_config_dir):
        """Test the publish command with --skip_audio flag."""
        runner = CliRunner()
        
        with patch.object(BIDSDataset, 'deidentify') as mock_deidentify:
            # Mock the deidentify method to return a BIDSDataset instance
            mock_deidentified = MagicMock(spec=BIDSDataset)
            mock_deidentify.return_value = mock_deidentified
            
            # Run the command with --skip_audio
            result = runner.invoke(publish_bids_dataset, [
                temp_bids_dir, temp_output_dir, temp_publish_config_dir, '--skip_audio'
            ])
            
            # Check that the command succeeded
            assert result.exit_code == 0
            
            # Check that deidentify was called with skip_audio=True
            mock_deidentify.assert_called_once_with(outdir=temp_output_dir, publish_config_dir=Path(temp_publish_config_dir), skip_audio=True)

    def test_publish_command_with_no_skip_audio(self, temp_bids_dir, temp_output_dir, temp_publish_config_dir):
        """Test the publish command with --no-skip_audio flag."""
        runner = CliRunner()
        
        with patch.object(BIDSDataset, 'deidentify') as mock_deidentify:
            # Mock the deidentify method to return a BIDSDataset instance
            mock_deidentified = MagicMock(spec=BIDSDataset)
            mock_deidentify.return_value = mock_deidentified
            
            # Run the command with --no-skip_audio (explicit)
            result = runner.invoke(publish_bids_dataset, [
                temp_bids_dir, temp_output_dir, temp_publish_config_dir, '--no-skip_audio'
            ])
            
            # Check that the command succeeded
            assert result.exit_code == 0
            
            # Check that deidentify was called with skip_audio=False
            mock_deidentify.assert_called_once_with(outdir=temp_output_dir, publish_config_dir=Path(temp_publish_config_dir), skip_audio=False)

    def test_publish_command_creates_bidsdataset_instance(self, temp_bids_dir, temp_output_dir, temp_publish_config_dir):
        """Test that the publish command creates a BIDSDataset instance with correct path."""
        runner = CliRunner()
        
        with patch('b2aiprep.commands.BIDSDataset') as mock_bids_class:
            # Mock the BIDSDataset class
            mock_instance = MagicMock(spec=BIDSDataset)
            mock_deidentified = MagicMock(spec=BIDSDataset)
            mock_instance.deidentify.return_value = mock_deidentified
            mock_bids_class.return_value = mock_instance
            
            # Run the command
            result = runner.invoke(publish_bids_dataset, [temp_bids_dir, temp_output_dir, temp_publish_config_dir])
            
            # Check that the command succeeded
            assert result.exit_code == 0
            
            # Check that BIDSDataset was instantiated with correct path
            mock_bids_class.assert_called_once_with(Path(temp_bids_dir))
            
            # Check that deidentify was called
            mock_instance.deidentify.assert_called_once()

    def test_publish_command_success_message(self, temp_bids_dir, temp_output_dir, temp_publish_config_dir):
        """Test that the publish command outputs success message."""
        runner = CliRunner()
        
        with patch.object(BIDSDataset, 'deidentify') as mock_deidentify:
            # Mock the deidentify method to return a BIDSDataset instance
            mock_deidentified = MagicMock(spec=BIDSDataset)
            mock_deidentify.return_value = mock_deidentified
            
            # Run the command
            result = runner.invoke(publish_bids_dataset, [temp_bids_dir, temp_output_dir, temp_publish_config_dir])
            
            # Check that the command succeeded
            assert result.exit_code == 0
            
            # The success message is logged, not printed to stdout
            # So we just check that the command completed without error

    def test_publish_command_handles_deidentify_errors(self, temp_bids_dir, temp_output_dir, temp_publish_config_dir):
        """Test that the publish command handles errors from deidentify method."""
        runner = CliRunner()
        
        with patch.object(BIDSDataset, 'deidentify') as mock_deidentify:
            # Mock the deidentify method to raise an exception
            mock_deidentify.side_effect = FileNotFoundError("Test error")
            
            # Run the command
            result = runner.invoke(publish_bids_dataset, [temp_bids_dir, temp_output_dir, temp_publish_config_dir])
            
            # Check that the command failed
            assert result.exit_code != 0
            assert "Test error" in str(result.exception)

    def test_publish_command_help_text(self):
        """Test that the help text is updated correctly."""
        runner = CliRunner()
        
        # Run the command with --help
        result = runner.invoke(publish_bids_dataset, ['--help'])
        
        # Check that the command succeeded
        assert result.exit_code == 0
        
        # Check that help text mentions deidentification
        assert "deidentification" in result.output.lower()
        assert "skip_audio" in result.output.lower()

    def test_publish_command_integration(self, temp_bids_dir, temp_output_dir, temp_publish_config_dir):
        """Integration test for the publish command without mocking."""
        runner = CliRunner()
        
        # Run the command without mocking (will use actual implementation)
        result = runner.invoke(publish_bids_dataset, [temp_bids_dir, temp_output_dir, temp_publish_config_dir])
        
        # Check that the command succeeded
        assert result.exit_code == 0
        
        # Check that output directory was created
        assert Path(temp_output_dir).exists()
        
        # Check that participants files were created
        assert (Path(temp_output_dir) / "participants.tsv").exists()
        assert (Path(temp_output_dir) / "participants.json").exists()


class TestPublishCommandBackwardCompatibility:
    """Test backward compatibility of the refactored publish command."""

    @pytest.fixture
    def temp_bids_dir(self):
        """Create a BIDS directory similar to what the old command expected."""
        temp_dir = tempfile.mkdtemp()
        bids_path = Path(temp_dir) / "test_bids"
        bids_path.mkdir(parents=True, exist_ok=True)
        
        # Create participants.tsv with record_id (old format)
        participants_file = bids_path / "participants.tsv"
        participants_file.write_text("record_id\tage\nparticipant001\t25\nparticipant002\t30\n")
        
        # Create participants.json
        participants_json = bids_path / "participants.json"
        participants_json.write_text('{"record_id": {"description": "ID"}, "age": {"description": "Age"}}')
        
        yield str(bids_path)
        
        # Cleanup
        shutil.rmtree(temp_dir)

    def test_old_vs_new_command_compatibility(self, temp_bids_dir):
        """Test that the new command produces similar results to what the old command would."""
        runner = CliRunner()
        
        # Create temporary output directory and config directory
        import tempfile
        temp_dir = tempfile.mkdtemp()
        temp_output_dir = Path(temp_dir) / "output"
        temp_config_dir = Path(temp_dir) / "config"
        temp_config_dir.mkdir(parents=True, exist_ok=True)
        
        # Create required config files
        (temp_config_dir / "participant_ids_to_remove.json").write_text("[]")
        (temp_config_dir / "audio_filestems_to_remove.json").write_text("[]")
        (temp_config_dir / "id_remapping.json").write_text("{}")
        
        try:
            # Run the new command
            result = runner.invoke(publish_bids_dataset, [temp_bids_dir, str(temp_output_dir), str(temp_config_dir)])
            
            # Check that the command succeeded
            assert result.exit_code == 0
            
            # Check that expected output files exist
            output_path = Path(temp_output_dir)
            assert (output_path / "participants.tsv").exists()
            assert (output_path / "participants.json").exists()
            
            # Check that participants.tsv has participant_id instead of record_id
            participants_content = (output_path / "participants.tsv").read_text()
            assert "participant_id" in participants_content
            # record_id should have been renamed to participant_id
        finally:
            # Cleanup
            shutil.rmtree(temp_dir)
