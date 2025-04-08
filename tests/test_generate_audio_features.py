import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def setup_temp_files():
    """Fixture to set up temporary directories and files for testing."""
    # Paths to required data files
    project_root = Path(__file__).parent.parent
    redcap_csv_source = project_root / "data/sdv_redcap_synthetic_data_1000_rows.csv"

    with (
        tempfile.TemporaryDirectory() as audio_dir,
        tempfile.TemporaryDirectory() as bids_dir,
        tempfile.TemporaryDirectory() as tar_dir,
    ):

        # Copy the RedCap CSV file into the temporary directory
        redcap_csv_path = os.path.join(tempfile.gettempdir(), "sdv_redcap_synthetic_data.csv")
        shutil.copy(redcap_csv_source, redcap_csv_path)

        # Create a dummy audio file in the audio directory
        audio_file_path = os.path.join(audio_dir, "sample.wav")
        with open(audio_file_path, "w") as f:
            f.write("dummy audio data")

        # Create paths for bids_dir and tar_file
        bids_dir_path = os.path.join(bids_dir, "bids_data")
        tar_file_path = os.path.join(tar_dir, "output.tar.gz")

        yield redcap_csv_path, audio_dir, bids_dir_path, tar_file_path





def test_generate_audio_features_cli_update(setup_temp_files):
    """Test the 'b2aiprep-cli prepbidslikedata' command using subprocess."""
    redcap_csv_path, audio_dir, bids_dir_path, tar_file_path = setup_temp_files
    update = "True"
    shadow_path = f"{Path(bids_dir_path)}_update"
    # Define the CLI command
    command = [
        "b2aiprep-cli",
        "generate-audio-features",
        bids_dir_path,
        shadow_path,
        "--tar_file_path",
        tar_file_path,
        "-t",
        "tiny",  # transcription_model_size
        "--n_cores",
        "2",
        "--with_sensitive",
        "--overwrite",
        "--validate",
        "--update",
        update
    ]
   
    # Run the CLI command
    result = subprocess.run(command, capture_output=True, text=True)

    # Check if the command was successful
    assert result.returncode == 0, f"CLI command failed: {result.stderr}"

    # Additional assertions can be added to check output files
    assert os.path.exists(bids_dir_path), "BIDS directory was not created"
    assert os.path.exists(tar_file_path), "Tar file was not created"
    assert os.path.exists(shadow_path), "shadow tree was not created"
