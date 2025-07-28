import logging
import pandas as pd
import csv
import os
import shutil
import struct
import subprocess
import tempfile
import wave
from pathlib import Path
import uuid
import pytest
import torch

from b2aiprep.prepare.prepare import (
    extract_features_workflow,
    extract_single,
    wav_to_features,
    extract_features_sequentially,
    extract_features_workflow,
    validate_bids_data,
    reduce_id_length,
    reduce_length_of_id,
    get_value_from_metadata,
    update_metadata_record_and_session_id,
    filter_audio_paths,
    is_audio_sensitive
)

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    parser.addoption("--bids-files-path", action="store", default=None, help="Path to BIDS files")


def create_dummy_wav_file(filepath, duration_seconds=1.0, sample_rate=16000):
    """Create a minimal valid WAV file for testing."""
    with wave.open(filepath, "wb") as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 2 bytes per sample
        wav_file.setframerate(sample_rate)

        # Generate simple sine wave
        num_samples = int(duration_seconds * sample_rate)
        samples = []
        for i in range(num_samples):
            # Simple sine wave
            value = int(32767 * 0.1)  # Low amplitude to avoid clipping
            samples.append(struct.pack("<h", value))

        wav_file.writeframes(b"".join(samples))


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

        # Create a proper dummy audio file in the audio directory
        audio_file_path = os.path.join(audio_dir, "sample.wav")
        create_dummy_wav_file(audio_file_path)

        # Create paths for bids_dir and tar_file
        bids_dir_path = os.path.join(bids_dir, "bids_data")
        tar_file_path = os.path.join(tar_dir, "output.tar.gz")

        yield redcap_csv_path, audio_dir, bids_dir_path, tar_file_path


@pytest.fixture
def setup_bids_structure():
    """Fixture to create a basic BIDS structure for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        bids_dir = Path(temp_dir) / "bids"
        bids_dir.mkdir()

        # Create basic BIDS structure
        subject_dir = bids_dir / "sub-001" / "ses-001" / "audio"
        subject_dir.mkdir(parents=True)

        # Create proper dummy audio file
        audio_file = subject_dir / "sub-001_ses-001_task-reading.wav"
        create_dummy_wav_file(str(audio_file))

        # Create corresponding JSON metadata file for the audio
        audio_json = subject_dir / "sub-001_ses-001_task-reading.json"
        audio_json.write_text(
            """{
            "item": [
                {
                    "linkId": "record_id",
                    "answer": [{"valueString": "rec-001"}]
                },
                {
                    "linkId": "session_id",
                    "answer": [{"valueString": "ses-001"}]
                }
            ]
        }"""
        )

        # Create proper dummy feature file (PyTorch tensor)
        feature_file = subject_dir / "sub-001_ses-001_task-reading_features.pt"
        dummy_features = {
            "torchaudio": {
                "spectrogram": torch.randn(10, 100),  # dummy spectrogram
                "mfcc": torch.randn(13, 100),  # dummy MFCC features
            }
        }
        torch.save(dummy_features, feature_file)

        # Create participants.tsv
        participants_file = bids_dir / "participants.tsv"
        participants_file.write_text("participant_id\trecord_id\tsex\tage\nsub-001\trec-001\tF\t25")

        # Create participants.json (required for some commands)
        participants_json = bids_dir / "participants.json"
        participants_json.write_text(
            '{"participant_id": {"Description": "Unique participant identifier"}, '
            '"record_id": {"Description": "Unique record identifier"}, '
            '"sex": {"Description": "Sex of participant"}, '
            '"age": {"Description": "Age of participant"}}'
        )

        # Create dataset_description.json
        dataset_desc = bids_dir / "dataset_description.json"
        dataset_desc.write_text('{"Name": "Test Dataset", "BIDSVersion": "1.8.0"}')

        # Create README.md and CHANGES.md
        (bids_dir / "README.md").write_text("Test dataset")
        (bids_dir / "CHANGES.md").write_text("Version 1.0")

        yield bids_dir


@pytest.fixture
def bids_files_path(request):
    return request.config.getoption("--bids-files-path")


def test_extract_single(setup_temp_files):
    redcap_csv_path, audio_dir, bids_dir_path, tar_file_path = setup_temp_files

    wav_path = os.path.join(audio_dir, "sample.wav")
    save_to = extract_single(wav_path=wav_path,
                             transcription_model_size="tiny",
                             with_sensitive=False,
                             update=False)

    assert os.path.exists(save_to), ".pt file was not generated"


def test_wav_to_features(setup_temp_files):
    redcap_csv_path, audio_dir, bids_dir_path, tar_file_path = setup_temp_files

    wav_path = [os.path.join(audio_dir, "sample.wav")]
    save_to = wav_to_features(wav_paths=wav_path,
                              transcription_model_size="tiny",
                              with_sensitive=False)

    for paths in save_to:
        assert os.path.exists(paths), ".pt file was not generated"


def test_extract_features_sequentially(setup_bids_structure):
    bids_dir = setup_bids_structure
    extract_features_sequentially(bids_dir_path=bids_dir)
    pt_files = list(bids_dir.rglob("*.pt"))
    assert pt_files


def test_extract_features_workflow(setup_bids_structure):
    bids_dir = setup_bids_structure
    extract_features_workflow(bids_dir_path=bids_dir)
    pt_files = list(bids_dir.rglob("*.pt"))
    assert pt_files, ".pt files were not generated"


def test_validate_bids_data(setup_bids_structure):
    bids_dir = setup_bids_structure
    assert validate_bids_data(bids_dir_path=bids_dir) is None

def test_is_audio_sensitive():
    file_path = Path("file_one_audio-check.wav")
    assert is_audio_sensitive(filepath=file_path) == False
    

def test_filter_audio_paths():
    audio_paths = [
        Path("file_one_audio-check.wav"),
        Path("file_two_sample_data.wav"),
        Path("file_three_audio-check.wav"),
        Path("file_four_normal_case.wav"),
    ]
    actual = filter_audio_paths(audio_paths=audio_paths)
    expected = [
        Path("file_two_sample_data.wav"),
        Path("file_four_normal_case.wav")]
    assert actual == expected

def test_reduce_id_length():
    example_id = "5f0c5b34-b634-4564-b97c-b44435a3e0ff"
    actual = reduce_id_length(example_id)
    expected = "5f0c5b34"
    assert actual == expected


def test_reduce_length_of_id():
    uuid_list = [
        '8a7d5b46-84c0-4f36-9ae0-e84c21f0c9f7',
        'd6b1a71e-9313-4141-9a8b-2170169ab74a',
        'b3cf2bd4-4c83-430e-8238-c22a770f828e',
        '7f16962f-2556-4be7-b6cf-7db062254646',
        'f6832f4d-369c-4a7b-bde6-3a84f6e68280',
        'fc9f9939-9d44-4743-a9c7-c2c8d8f6f14e',
        'c9f4e3d5-5087-4d96-9e00-7c8584b0ff49',
        '1ff15a6b-4ef0-40d5-bb4e-b5ec9d40d5f2',
        'cd9ac2b3-1879-4214-95a0-bf8943bc1993',
        '8fa44e90-9201-4f94-847b-32bfb285b747'
    ]

    # Create the DataFrame
    df = pd.DataFrame({
        "record_id": uuid_list
    })
    list_of_expected_id = []
    for record_id in df["record_id"]:
        list_of_expected_id.append(record_id[:record_id.find("-")])

    df_modified = reduce_length_of_id(df, "record_id")
    list_of_actual_id = []
    for record_id in df_modified["record_id"]:
        list_of_actual_id.append(record_id)

    assert list_of_expected_id == list_of_actual_id


def test_get_value_from_metadata():
    metadata = {
        "item": [
            {"linkId": "abc123", "answer": [{"valueString": "value1"}]},
            {"linkId": "def456", "answer": [{"valueString": "value2"}]},
            {"linkId": "xyz789", "answer": [{"valueString": "value3"}]},
        ]
    }
    link_id = "abc123"
    actual = get_value_from_metadata(metadata=metadata, linkid=link_id)
    expected = "value1"
    assert actual == expected


def test_update_metadata_record_and_session_id():
    metadata = {
        "item": [
            {"linkId": "record_id", "answer": [
                {"valueString": "1a3f7e22-8d4b-4e3a-b36d-12a5c2e5b9d8"}]},
            {"linkId": "session_id", "answer": [
                {"valueString": "5d9e34a7-2c90-44bf-8b53-1bde7a67e3f2"}]},
            {"linkId": "record_id", "answer": [
                {"valueString": "c7f3a128-e10a-4d0d-9c7e-f1a4b5767d6b"}]},
        ]
    }
    expected_metadata = {
        "item": [
            {"linkId": "participant_id", "answer": [
                {"valueString": "1a3f7e22"}]},
            {"linkId": "session_id", "answer": [
                {"valueString": "5d9e34a7"}]},
            {"linkId": "participant_id", "answer": [
                {"valueString": "c7f3a128"}]},
        ]
    }
    update_metadata_record_and_session_id(metadata=metadata)
    assert metadata == expected_metadata


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Skipping benchmarking test in CI environment"
)
def test_extract_features_timing(benchmark, caplog, bids_files_path):
    caplog.set_level(logging.INFO)
    if bids_files_path is None:
        _logger.error("Please provide the path to BIDS files using --bids-files-path")
        return
    result = benchmark(extract_features_workflow, bids_files_path)
    _logger.info(str(result))
    assert result is not None, "Benchmark failed"
    assert len(result["features"]) > 0, "No features extracted"
