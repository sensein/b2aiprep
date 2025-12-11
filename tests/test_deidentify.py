"""Tests for the deidentify_bids_dataset command."""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

import pytest

from b2aiprep.commands import deidentify_bids_dataset
from b2aiprep.prepare.dataset import BIDSDataset


class TestDeidentifyCommand:
    """Test cases for the deidentify_bids_dataset command."""

    @pytest.fixture
    def temp_bids_dir(self):
        """Create a minimal BIDS directory for testing."""
        temp_dir = tempfile.mkdtemp()
        bids_path = Path(temp_dir) / "test_bids"
        bids_path.mkdir(parents=True, exist_ok=True)
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

    def test_deidentify_command_calls_deidentify(self, temp_bids_dir, temp_output_dir, setup_publish_config):
        """Test that the deidentify command calls the BIDSDataset.deidentify method."""
        runner = CliRunner()
        setup_publish_config = setup_publish_config.as_posix()
        
        with patch.object(BIDSDataset, 'deidentify') as mock_deidentify:
            # Mock the deidentify method to return a BIDSDataset instance
            mock_deidentified = MagicMock(spec=BIDSDataset)
            mock_deidentify.return_value = mock_deidentified
            
            # Run the command
            result = runner.invoke(deidentify_bids_dataset, [temp_bids_dir, temp_output_dir, setup_publish_config])
            
            # Check that the command succeeded
            assert result.exit_code == 0
            
            # Check that deidentify was called with correct parameters
            mock_deidentify.assert_called_once_with(outdir=temp_output_dir, deidentify_config_dir=Path(setup_publish_config), skip_audio=False, skip_audio_features=False, max_workers=16)

    def test_deidentify_command_help_text(self):
        """Test that the help text is updated correctly."""
        runner = CliRunner()
        
        # Run the command with --help
        result = runner.invoke(deidentify_bids_dataset, ['--help'])
        
        # Check that the command succeeded
        assert result.exit_code == 0
        
        # Check that help text mentions deidentification
        assert "deidentification" in result.output.lower()
        assert "skip_audio" in result.output.lower()

    def test_deidentify_command_integration(self, temp_bids_dir, temp_output_dir, setup_publish_config):
        """Integration test for the deidentify command without mocking."""
        runner = CliRunner()
        setup_publish_config = setup_publish_config.as_posix()
        
        # Run the command without mocking (will use actual implementation)
        result = runner.invoke(deidentify_bids_dataset, [temp_bids_dir, temp_output_dir, setup_publish_config])
        
        # Check that the command succeeded
        assert result.exit_code == 0
        
        # Check that output directory was created
        assert Path(temp_output_dir).exists()
