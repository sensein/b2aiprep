"""Tests for the CLI of b2aiprep."""
import os
import shutil
import struct
import subprocess
import tempfile
import wave
from pathlib import Path
import json

import pytest
from conftest import WAV_BYTES
import torch
import pandas as pd
import pyarrow.parquet as pq
from unittest.mock import patch, MagicMock
from click.testing import CliRunner

from b2aiprep.commands import (
    _copy_recording_tables_to_bundle,
    _unreleasable_columns,
    create_subject_splits,
    deidentify_bids_dataset,
    redcap2bids,
    validate_bundled_dataset,
)
from b2aiprep.prepare.dataset import AccessTier, BIDSDataset, DispositionLevel, SessionLabels
from b2aiprep.prepare import field_map


@pytest.fixture(autouse=True)
def adult_field_map(monkeypatch):
    """Overrides conftest's: CLI tests start with no cohort selected, so a command that reads the
    field map without selecting its --cohort fails here."""
    monkeypatch.setattr(field_map, "_selected", None)

class TestDeidentifyCommand:
    """Test cases for the deidentify_bids_dataset command."""

    @pytest.fixture
    def temp_bids_dir(self):
        """Create a minimal BIDS directory for testing."""
        temp_dir = tempfile.mkdtemp()
        bids_path = Path(temp_dir) / "test_bids"
        bids_path.mkdir(parents=True, exist_ok=True)

        # Create a minimal participant so the allowlist is non-empty
        audio_dir = bids_path / "sub-p1" / "ses-s1" / "audio"
        audio_dir.mkdir(parents=True)
        (audio_dir / "sub-p1_ses-s1_task-test.wav").write_bytes(WAV_BYTES)
        (audio_dir / "sub-p1_ses-s1_task-test_recording-metadata.json").write_text(
            json.dumps({"item": [
                {"linkId": "record_id", "answer": [{"valueString": "p1"}]},
                {"linkId": "session_id", "answer": [{"valueString": "s1"}]},
            ]})
        )
        pd.DataFrame({"record_id": ["p1"], "session_id": ["s1"], "session_index": ["1"]}).to_csv(
            bids_path / "sub-p1" / "sessions.tsv", sep="\t", index=False
        )

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
            result = runner.invoke(deidentify_bids_dataset, ["--cohort", "adult", temp_bids_dir, temp_output_dir, setup_publish_config])
            
            # Check that the command succeeded
            assert result.exit_code == 0
            
            # Check that deidentify was called with correct parameters
            mock_deidentify.assert_called_once_with(outdir=temp_output_dir, deidentify_config_dir=Path(setup_publish_config), skip_audio=False, skip_audio_features=False, max_workers=16, disposition_level=None, keep_shifted_dates=False, session_labels=SessionLabels.ORDINAL, session_id_map=None, cohort="adult")

    def test_deidentify_command_passes_qa_and_session_options(self, temp_bids_dir, temp_output_dir,
                                                              setup_publish_config, tmp_path):
        session_map = tmp_path / "map.json"
        with patch.object(BIDSDataset, "deidentify") as mock_deidentify:
            result = CliRunner().invoke(deidentify_bids_dataset, ["--cohort", "adult",
                temp_bids_dir, temp_output_dir, str(setup_publish_config), "--disposition-level", "internal",
                "--keep-shifted-dates", "--session-labels", "index", "--session-id-map", str(session_map)])
        assert result.exit_code == 0, result.output
        kwargs = mock_deidentify.call_args.kwargs
        assert kwargs["disposition_level"] is DispositionLevel.INTERNAL and kwargs["keep_shifted_dates"] is True
        assert kwargs["session_labels"] is SessionLabels.INDEX and Path(kwargs["session_id_map"]) == session_map
        assert kwargs["cohort"] == "adult"

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
        # deidentify refuses an allowlisted participant without a pseudonym
        (setup_publish_config / "id_remapping.json").write_text(json.dumps({"p1": "p1"}))
        setup_publish_config = setup_publish_config.as_posix()
        
        # Run the command without mocking (will use actual implementation)
        result = runner.invoke(deidentify_bids_dataset, ["--cohort", "adult", temp_bids_dir, temp_output_dir, setup_publish_config])
        
        # Check that the command succeeded
        assert result.exit_code == 0
        
        out = Path(temp_output_dir) / "sub-p1"
        assert (out / "ses-01" / "audio" / "sub-p1_ses-01_task-test.flac").exists()
        assert (out / "ses-01" / "audio" / "sub-p1_ses-01_task-test.json").exists()
        assert (out / "sub-p1_sessions.tsv").exists()


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

    # write associated metadata file
    mock_metedata_recording_json = {
            "item": [
                {
                    "linkId": "record_id",
                    "answer": [{"valueString": "001"}]
                },
                {
                    "linkId": "session_id",
                    "answer": [{"valueString": "001"}]
                },
                {
                    "linkId": "recording_id",
                    "answer": [{"valueString": "rec-001"}]
                }
            ]
        }

    metadata_path = filepath[:].replace(".wav", "_recording-metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(mock_metedata_recording_json, f, indent=2)


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
              
        session_data = {
            "record_id": ["001"],
            "session_id": ["001"],
            "session_index": ["1"],
        }
        session_df = pd.DataFrame(session_data)

        session_tsv = bids_dir / "sub-001"/ f"sessions.tsv"
        session_df.to_csv(session_tsv, sep="\t", index=False)

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
                    "answer": [{"valueString": "001"}]
                },
                {
                    "linkId": "session_id",
                    "answer": [{"valueString": "001"}]
                },
                {
                    "linkId": "recording_id",
                    "answer": [{"valueString": "rec-001"}]
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
                "mel_filter_bank": torch.randn(13,100),
                "mel_spectrogram": torch.randn(13,100),
                "pitch": torch.randn(100)
            },
            "ppgs": torch.randn(40,100),
            "sparc": {
                "ema": torch.randn(100,12),
                "loudness": torch.randn(100,1),
                "periodicity": torch.randn(100,1),
                "pitch_stats": torch.randn(2),
                "pitch": torch.randn(100,1)
            }

        }
        torch.save(dummy_features, feature_file)

        # Create participants.tsv
        participants_file = bids_dir / "participants.tsv"
        participants_file.write_text("participant_id\trecord_id\tsex\tage\n001\t001\tF\t25")

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
        (bids_dir / "CHANGELOG.md").write_text("Version 1.0")

        yield bids_dir


@pytest.fixture
def setup_bids_structure_with_nan_feature():
    """Fixture to create a basic BIDS structure for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        bids_dir = Path(temp_dir) / "bids"
        bids_dir.mkdir()

        # Create basic BIDS structure
        subject_dir = bids_dir / "sub-001" / "ses-001" / "audio"
        subject_dir.mkdir(parents=True)
              
        session_data = {
            "record_id": ["001"],
            "session_id": ["001"],
            "session_index": ["1"],
        }
        session_df = pd.DataFrame(session_data)

        session_tsv = bids_dir / "sub-001"/ f"sessions.tsv"
        session_df.to_csv(session_tsv, sep="\t", index=False)

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
                    "answer": [{"valueString": "001"}]
                },
                {
                    "linkId": "session_id",
                    "answer": [{"valueString": "001"}]
                },
                {
                    "linkId": "recording_id",
                    "answer": [{"valueString": "rec-001"}]
                }
            ]
        }"""
        )

        # Create proper dummy feature file (PyTorch tensor)
        feature_file = subject_dir / "sub-001_ses-001_task-reading_features.pt"
        dummy_features = {
            "torchaudio": {
                "spectrogram": torch.nan,  # dummy spectrogram
                "mfcc": torch.randn(13, 100),  # dummy MFCC features
                "mel_filter_bank": torch.randn(13,100),
                "mel_spectrogram": torch.randn(13,100),
                "pitch": torch.randn(100)
            },
            "ppgs": torch.randn(40,100),
            "sparc": {
                "ema": torch.randn(100,12),
                "loudness": torch.randn(100,1),
                "periodicity": torch.randn(100,1),
                "pitch_stats": torch.randn(2),
                "pitch": torch.randn(100,1)
            }

        }
        torch.save(dummy_features, feature_file)

        # Create participants.tsv
        participants_file = bids_dir / "participants.tsv"
        participants_file.write_text("participant_id\trecord_id\tsex\tage\n001\t001\tF\t25")

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
        (bids_dir / "CHANGELOG.md").write_text("Version 1.0")

        yield bids_dir


@pytest.fixture
def setup_bids_structure_after_deidentify():
    """Fixture to create a basic BIDS structure for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        bids_dir = Path(temp_dir) / "bids"
        bids_dir.mkdir()

        # Create basic BIDS structure
        subject_dir = bids_dir / "sub-001" / "ses-001" / "audio"
        subject_dir.mkdir(parents=True)
              
        session_data = {
            "record_id": ["001"],
            "session_id": ["001"],
            "session_index": ["1"],
        }
        session_df = pd.DataFrame(session_data)

        session_tsv = bids_dir / "sub-001"/ f"sessions.tsv"
        session_df.to_csv(session_tsv, sep="\t", index=False)

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
                    "answer": [{"valueString": "001"}]
                },
                {
                    "linkId": "session_id",
                    "answer": [{"valueString": "001"}]
                },
                {
                    "linkId": "recording_id",
                    "answer": [{"valueString": "rec-001"}]
                }
            ]
        }"""
        )

        # Create proper dummy feature file (PyTorch tensor)
        feature_file = subject_dir / "sub-001_ses-001_task-reading_features.pt"
        dummy_features = {
            "torchaudio": {
                "spectrogram": None,
                "mfcc": None,  # dummy MFCC features
                "mel_filter_bank": torch.randn(13,100),
                "mel_spectrogram": None,
                "pitch": torch.randn(100)
            },
            "ppgs": None,
            "sparc": {
                "ema": torch.randn(100,12),
                "loudness": torch.randn(100,1),
                "periodicity": torch.randn(100,1),
                "pitch_stats": torch.randn(2),
                "pitch": torch.randn(100,1)
            }

        }
        torch.save(dummy_features, feature_file)

        # Create participants.tsv
        participants_file = bids_dir / "participants.tsv"
        participants_file.write_text("participant_id\trecord_id\tsex\tage\n001\t001\tF\t25")

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
        (bids_dir / "CHANGELOG.md").write_text("Version 1.0")

        yield bids_dir

def test_dashboard_cli_help():
    """Test the 'b2aiprep-cli dashboard' command help (avoid launching actual dashboard)."""
    command = ["b2aiprep-cli", "dashboard", "--help"]

    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI help command failed: {result.stderr}"
    assert "Launches a dashboard" in result.stdout, "Dashboard help text not found"


def test_redcap2bids_cli(setup_temp_files):
    """Test the 'b2aiprep-cli redcap2bids' command using subprocess."""
    redcap_csv_path, audio_dir, _, _ = setup_temp_files

    with tempfile.TemporaryDirectory() as outdir, tempfile.TemporaryDirectory() as logdir:
        command = [
            "b2aiprep-cli",
            "redcap2bids",
            "--cohort",
            "adult",
            redcap_csv_path,
            "--outdir",
            outdir,
            "--audiodir",
            audio_dir,
            "--date-shift-anchor",
            "2100-01-01",
            "--date-shift-log",
            os.path.join(logdir, "date_shift.json"),
        ]

        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, f"CLI command failed: {result.stderr}"
        assert os.path.exists(outdir), "Output directory was not created"
        with open(os.path.join(logdir, "date_shift.json")) as f:
            assert json.load(f)["anchor"] == "2100-01-01"


def test_redcap2bids_fails_without_a_date_shift_anchor(setup_temp_files, tmp_path):
    redcap_csv_path, _, _, _ = setup_temp_files
    result = CliRunner().invoke(redcap2bids, ["--cohort", "adult", redcap_csv_path, "--outdir", str(tmp_path / "out")])
    assert result.exit_code == 2 and "--date-shift-anchor" in result.output


def test_redcap_stats_needs_no_cohort(setup_temp_files):
    """It summarizes a raw export: nothing was removed at ingest, so the field map is not read."""
    from b2aiprep.commands import redcap_stats
    redcap_csv_path, _, _, _ = setup_temp_files
    result = CliRunner().invoke(redcap_stats, [redcap_csv_path])
    assert result.exit_code == 0, result.output
    assert "Total unique participant IDs" in result.output or "No session data" in result.output


@pytest.mark.parametrize("command", ["redcap2bids", "deidentify_bids_dataset", "validate_bundled_dataset"])
def test_commands_that_read_the_field_map_require_a_cohort(command, tmp_path):
    """No default cohort: a build must say whose field-map rows it uses."""
    import b2aiprep.commands as commands
    args = {"redcap2bids": [str(tmp_path), "--date-shift-anchor", "2100-01-01"],
            "deidentify_bids_dataset": [str(tmp_path), str(tmp_path / "out"), str(tmp_path)],
            "validate_bundled_dataset": [str(tmp_path), str(tmp_path)]}[command]
    result = CliRunner().invoke(getattr(commands, command), args)
    assert result.exit_code == 2 and "--cohort" in result.output


def test_create_bundled_dataset_cli(setup_bids_structure):
    """Test the 'b2aiprep-cli create-bundled-dataset' command using subprocess."""
    bids_dir = setup_bids_structure

    with tempfile.TemporaryDirectory() as outdir:
        command = ["b2aiprep-cli", "create-bundled-dataset", str(bids_dir), outdir]

        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, f"CLI command failed: {result.stderr}"
        assert os.path.exists(outdir), "Output directory was not created"
        parquets = list(Path(outdir).rglob("*.parquet"))
        assert parquets, "no feature parquet was bundled"
        for path in parquets:
            assert '"List"' not in pq.read_schema(path).metadata[b"huggingface"].decode(), path.name
            assert pq.ParquetFile(path).metadata.row_group(0).column(0).compression == "ZSTD", path.name
        for path in Path(outdir).rglob("*.parquet"):
            sidecar_path = path.with_suffix(".json")
            if not sidecar_path.exists():
                continue  # parquets that are not array features (e.g. transcripts) have no tensor sidecar
            sidecar = json.loads(sidecar_path.read_text())
            tensors = [f for f in sidecar["fields"] if "tensor" in f.get("extras", {})]
            assert len(tensors) == 1, path.name
            declared = tensors[0]["extras"]["tensor"]["dtype"]
            assert _leaf_dtype(pq.read_schema(path).field(tensors[0]["name"]).type) == declared, path.name
            assert "senselab_version" in tensors[0]["extras"]["tensor"]["provenance"], path.name
        static = Path(outdir) / "features" / "static_features.tsv"
        if static.exists():
            assert "windows" not in pd.read_csv(static, sep="\t", nrows=1).columns


def _leaf_dtype(arrow_type):
    """numpy name of the innermost value type of a (nested) arrow list type."""
    import pyarrow as pa
    while pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type) or pa.types.is_fixed_size_list(arrow_type):
        arrow_type = arrow_type.value_type
    return arrow_type.to_pandas_dtype().__name__


def test_every_feature_sidecar_declares_the_dtype_the_bundle_stores():
    """#363: the bundle writes the sidecar's dtype; posteriorgrams and the mel spectrogram stay float32."""
    import numpy as np
    from b2aiprep.commands import _feature_sidecar, _sidecar_array_dtype
    schemas = Path(__file__).resolve().parents[1] / "src/b2aiprep/prepare/resources/feature_schemas"
    declared = {p.stem: _sidecar_array_dtype(_feature_sidecar(p.stem), p.stem) for p in sorted(schemas.glob("*.json"))}
    assert declared["ppgs"] is np.float32 and declared["torchaudio_mel_spectrogram"] is np.float32
    assert all(v is np.float16 for k, v in declared.items() if k not in ("ppgs", "torchaudio_mel_spectrogram"))


def test_sidecar_without_a_declared_dtype_is_refused():
    from b2aiprep.commands import _sidecar_array_dtype
    with pytest.raises(ValueError, match="must declare exactly one tensor"):
        _sidecar_array_dtype({"fields": [{"name": "x", "extras": {"tensor": {"ndim": 2}}}]}, "x")
    with pytest.raises(ValueError, match="must declare exactly one tensor"):
        _sidecar_array_dtype(None, "missing")


def test_sidecar_records_the_senselab_versions_of_its_feature_files():
    from b2aiprep.commands import _sidecar_with_software_versions
    sidecar = {"fields": [{"name": "ppgs", "extras": {"tensor": {"dtype": "float32", "provenance": {"software": "ppgs"}}}}]}
    one = _sidecar_with_software_versions(sidecar, {"1.4.0a14"})
    assert one["fields"][0]["extras"]["tensor"]["provenance"] == {"software": "ppgs", "senselab_version": "1.4.0a14"}
    two = _sidecar_with_software_versions(sidecar, {"unknown", "1.4.0a14"})
    assert two["fields"][0]["extras"]["tensor"]["provenance"]["senselab_version"] == ["1.4.0a14", "unknown"]
    assert "senselab_version" not in sidecar["fields"][0]["extras"]["tensor"]["provenance"]  # input untouched


def test_static_feature_values_keep_scalars_only():
    """senselab 1.4's SQUIM `windows` list stays in the feature file, out of static_features.tsv."""
    from b2aiprep.commands import _static_feature_values
    squim = {"stoi": 0.9, "pesq": 2.1, "si_sdr": 14.0, "span_s": 12.5, "windows_n": 2,
             "windows": [{"start_s": 0.0, "end_s": 10.0, "stoi": 0.9}]}
    assert _static_feature_values(squim) == {"stoi": 0.9, "pesq": 2.1, "si_sdr": 14.0, "span_s": 12.5, "windows_n": 2}


def _list_feature_dataset(n_rows):
    from datasets import Dataset, Features, Sequence, Value
    features = Features({
        "participant_id": Value("string"), "session_id": Value("string"), "task_name": Value("string"),
        "ppgs": Sequence(Sequence(Value("float32"))), "n_frames": Value("int64"),
    })
    rows = {
        "participant_id": [f"p{i // 10:03d}" for i in range(n_rows)],
        "session_id": ["s1"] * n_rows,
        "task_name": [f"t{i % 10}" for i in range(n_rows)],
        "ppgs": [[[float(i), 0.5], [1e-6, float(-i)]] for i in range(n_rows)],
        "n_frames": [2] * n_rows,
    }
    return Dataset.from_dict(rows, features=features)


_BUNDLE_SETTINGS = dict(
    version="2.6", compression="zstd", compression_level=3,
    use_dictionary=["participant_id", "session_id", "task_name"], write_statistics=True,
    data_page_size=1_048_576, write_page_index=True, use_byte_stream_split=True,
)


def test_write_bundle_parquet_keeps_the_v4_layout_and_loads_in_datasets(tmp_path):
    """Bundle settings reach the file under any datasets version, which a datasets-3-readable label describes."""
    from datasets import Dataset
    from b2aiprep.commands import _write_bundle_parquet
    ds = _list_feature_dataset(2500)
    path = tmp_path / "ppgs.parquet"
    _write_bundle_parquet(ds, path, **_BUNDLE_SETTINGS,
                          sorting_columns=(pq.SortingColumn(0), pq.SortingColumn(2)))

    meta = pq.ParquetFile(path).metadata
    assert [meta.row_group(i).num_rows for i in range(meta.num_row_groups)] == [1000, 1000, 500]
    column = meta.row_group(0).column(3)
    assert column.compression == "ZSTD" and "BYTE_STREAM_SPLIT" in column.encodings and column.has_offset_index
    assert meta.row_group(0).sorting_columns
    label = json.loads(pq.read_schema(path).metadata[b"huggingface"])["info"]["features"]["ppgs"]
    assert label["_type"] == "Sequence" and label["feature"]["_type"] == "Sequence" and "length" not in label

    loaded = Dataset.from_parquet(str(path), cache_dir=str(tmp_path / "cache"))
    assert loaded.features == ds.features
    assert loaded[2499]["ppgs"] == ds[2499]["ppgs"] and loaded[7]["participant_id"] == ds[7]["participant_id"]


def test_write_bundle_parquet_matches_datasets_to_parquet_where_it_still_works(tmp_path):
    """Under datasets 3.x, whose to_parquet accepts the settings, the two files are the same bytes."""
    import datasets
    from b2aiprep.commands import _write_bundle_parquet
    if int(datasets.__version__.split(".")[0]) >= 4:
        pytest.skip("datasets>=4 to_parquet rejects the bundle settings (its own compression collides)")
    ds = _list_feature_dataset(2500)
    _write_bundle_parquet(ds, tmp_path / "own.parquet", **_BUNDLE_SETTINGS)
    ds.to_parquet(str(tmp_path / "hf.parquet"), **_BUNDLE_SETTINGS)
    assert (tmp_path / "own.parquet").read_bytes() == (tmp_path / "hf.parquet").read_bytes()


def test_datasets3_feature_metadata_relabels_lists_only():
    from b2aiprep.commands import _datasets3_feature_metadata
    value = {"dtype": "float32", "_type": "Value"}
    meta = {"info": {"features": {
        "x": {"feature": {"feature": value, "_type": "List"}, "_type": "List"},
        "fixed": {"feature": value, "length": 3, "_type": "List"},
        "arr": {"shape": [2, 3], "dtype": "float32", "_type": "Array2D"},
        "id": {"dtype": "string", "_type": "Value"},
    }}}
    out = _datasets3_feature_metadata(meta)["info"]["features"]
    assert out["x"] == {"feature": {"feature": value, "_type": "Sequence"}, "_type": "Sequence"}
    assert out["fixed"] == {"feature": value, "length": 3, "_type": "Sequence"}
    assert out["arr"] == meta["info"]["features"]["arr"] and out["id"] == meta["info"]["features"]["id"]
    with pytest.raises(ValueError, match="List of a dict"):
        _datasets3_feature_metadata({"_type": "List", "feature": {"a": value}})


def test_create_bundled_dataset_cli_with_nans(setup_bids_structure_with_nan_feature):
    """Test the 'b2aiprep-cli create-bundled-dataset' command using subprocess with nans."""
    bids_dir = setup_bids_structure_with_nan_feature

    with tempfile.TemporaryDirectory() as outdir:
        command = ["b2aiprep-cli", "create-bundled-dataset", str(bids_dir), outdir]

        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, f"CLI command failed: {result.stderr}"
        assert os.path.exists(outdir), "Output directory was not created"


def test_create_bundled_dataset_cli_with_sensitive(setup_bids_structure_after_deidentify):
    """Test the 'b2aiprep-cli create-bundled-dataset' command using subprocess with deidentified."""
    bids_dir = setup_bids_structure_after_deidentify

    with tempfile.TemporaryDirectory() as outdir:
        command = ["b2aiprep-cli", "create-bundled-dataset", str(bids_dir), outdir]

        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, f"CLI command failed: {result.stderr}"
        assert os.path.exists(outdir), "Output directory was not created"


def test_deidentify_bids_dataset_cli_id_rename(
    setup_bids_structure, setup_publish_config, tmp_path
):
    """Test the 'b2aiprep-cli deidentify-bids-dataset' command using subprocess."""
    bids_dir = setup_bids_structure
    config_dir = setup_publish_config
    outdir = tmp_path / "output_dataset"

    # Create phenotype directory structure
    phenotype_dir = bids_dir / "phenotype"
    phenotype_dir.mkdir()
    (phenotype_dir / "confounders.tsv").write_text(
        "participant_id\trecord_id\ntest\trec-test"
    )
    (phenotype_dir / "confounders.json").write_text(
        '{"participant_id": {"Description": "Participant identifier"}, '
        '"record_id": {"Description": "Record identifier"}}'
    )

    # Modify id_remapping.json
    id_remapping_path = config_dir / "id_remapping.json"
    with open(id_remapping_path, "w") as f:
        json.dump({"001": "P001"}, f, indent=2)

    # Match the task name used in setup_bids_structure
    with open(config_dir / "audio_tasks_to_include.json", "w") as f:
        json.dump(["reading"], f)

    command = [
        "b2aiprep-cli",
        "deidentify-bids-dataset",
        "--cohort",
        "adult",
        str(bids_dir),
        str(outdir),
        str(config_dir),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI command failed: {result.stderr}"
    assert outdir.exists(), "Output directory was not created"
    # checked if participant_id was changed
    assert (outdir / "sub-P001").exists()
    assert not (outdir / "audio_quality_metrics.json").exists()
    assert not (outdir / "audio_quality_metrics.tsv").exists()
        
def test_deidentify_bids_dataset_cli_remove_audio(
    setup_bids_structure, setup_publish_config, tmp_path
):
    """Test the 'b2aiprep-cli deidentify-bids-dataset' command using subprocess."""
    bids_dir = setup_bids_structure
    config_dir = setup_publish_config
    outdir = tmp_path / "output_dataset"

    # Create phenotype directory structure
    phenotype_dir = bids_dir / "phenotype"
    phenotype_dir.mkdir()
    (phenotype_dir / "confounders.tsv").write_text(
        "participant_id\trecord_id\ntest\trec-test"
    )
    (phenotype_dir / "confounders.json").write_text(
        '{"participant_id": {"Description": "Participant identifier"}, '
        '"record_id": {"Description": "Record identifier"}}'
    )

    # Create audio quality metrics
    quality_metrics_file = bids_dir / "audio_quality_metrics.tsv"
    quality_metrics_file.write_text("participant_id\tsession_id\ttask_name\n001\t001\treading\n001\t001\tnot-reading")

    # Create participants.json (required for some commands)
    participants_json = bids_dir / "participants.json"
    participants_json.write_text(
        '{"participant_id": {"Description": "Unique participant identifier"}, '
        '"record_id": {"Description": "Unique record identifier"}, '
        '"sex": {"Description": "Sex of participant"}, '
        '"age": {"Description": "Age of participant"}}'
    )

    # Modify audio_filestems_to_remove.json
    audio_to_remove_path = config_dir / "audio_filestems_to_remove.json"
    with open(audio_to_remove_path, "w") as f:
        json.dump(["sub-001_ses-001_task-reading"], f, indent=2)
    (config_dir / "id_remapping.json").write_text(json.dumps({"001": "001"}))  # deidentify requires a pseudonym
    (config_dir / "deidentify_settings.json").write_text(json.dumps({"access_tier": "registered"}))

    # Match the task names used in setup_bids_structure + extra task below
    with open(config_dir / "audio_tasks_to_include.json", "w") as f:
        json.dump(["reading", "not-reading"], f)

    # Add a second audio file so the participant still has output after removing reading
    extra_audio = bids_dir / "sub-001" / "ses-001" / "audio" / "sub-001_ses-001_task-not-reading.wav"
    create_dummy_wav_file(str(extra_audio))
    extra_sidecar = bids_dir / "sub-001" / "ses-001" / "audio" / "sub-001_ses-001_task-not-reading_recording-metadata.json"
    extra_sidecar.write_text(json.dumps({
        "item": [
            {"linkId": "record_id", "answer": [{"valueString": "001"}]},
            {"linkId": "session_id", "answer": [{"valueString": "001"}]},
        ]
    }))

    command = [
        "b2aiprep-cli",
        "deidentify-bids-dataset",
        "--cohort",
        "adult",
        str(bids_dir),
        str(outdir),
        str(config_dir),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI command failed: {result.stderr}"
    assert outdir.exists(), "Output directory was not created"
    # check if file was removed
    assert not (
        outdir / "sub-001" / "ses-1" / "audio" / "sub-001_ses-1_task-reading.json"
    ).exists()
    assert not (
        outdir / "sub-001" / "ses-1" / "audio" / "sub-001_ses-1_task-reading.flac"
    ).exists()
    assert not (
        outdir
        / "sub-001"
        / "ses-1"
        / "audio"
        / "sub-001_ses-1_task-reading_features.pt"
    ).exists()
    assert (outdir / "audio_quality_metrics.json").exists()
    assert (outdir / "audio_quality_metrics.tsv").exists()
    
    quality_deidentified = pd.read_csv((outdir / "audio_quality_metrics.tsv"),sep='\t')
    assert len(quality_deidentified)==1, f"deidentified audio_quality_metrics.tsv should have 1 entry but found {len(quality_deidentified)}"


def test_run_quality_control(
    setup_bids_structure
):
    """Test the 'b2aiprep-cli run-quality-control' command using subprocess."""
    bids_dir = setup_bids_structure

    command = [
        "b2aiprep-cli",
        "run-quality-control-on-audios",
        str(bids_dir),
    ]

    result_file = bids_dir / "audio_quality_metrics.tsv"
    result_data_descriptor = bids_dir / "audio_quality_metrics.json"

    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, f"CLI command failed: {result.stderr}"
    assert result_file.exists(), "Quality metrics TSV was not created"
    assert result_data_descriptor.exists(), "Quality metrics JSON was not created"


def test_reproschema_to_redcap_cli():
    """Test the 'b2aiprep-cli reproschema-to-redcap' command using subprocess."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create survey structure
        survey_dir = Path(temp_dir) / "survey"
        subject_dir = survey_dir / "subject_001"
        session_dir = subject_dir / "session_001"
        session_dir.mkdir(parents=True)

        # Create dummy survey file
        survey_file = subject_dir / "questionnaire.json"
        survey_file.write_text('{"test": "data"}')

        # Create audio structure
        audio_dir = Path(temp_dir) / "audio"
        audio_subject_dir = audio_dir / "subject_001" / "session_001"
        audio_subject_dir.mkdir(parents=True)

        # Create proper dummy audio file with UUID-like name
        audio_file = audio_subject_dir / "ready_for_school_12345678-1234-1234-1234-123456789abc.wav"
        create_dummy_wav_file(str(audio_file))
        
        age_file = audio_subject_dir/"activity_0.jsonld" 
        
        fake_jsonld = """[
            {
                "@context": "https://fake.context.com",
                "@type": "reproschema:ResponseActivity",
                "@id": "uuid:aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "used": [
                    "https://fake.url.com/activities/contact_information/items/age"
                ],
                "startedAtTime": "2025-01-01T10:00:00.000Z",
                "endedAtTime": "2025-01-01T10:00:05.000Z"
            },
            {
                "@context": "https://fake.context.com",
                "@type": "reproschema:Response",
                "@id": "uuid:bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "isAbout": "https://fake.url.com/activities/contact_information/items/age",
                "value": "7"
            }
        ]"""
        age_file.write_text(fake_jsonld)


        output_dir = Path(temp_dir) / "output"

        command = [
            "b2aiprep-cli",
            "reproschema-to-redcap",
            str(audio_dir),
            str(survey_dir),
            str(output_dir),
        ]

        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, f"CLI command failed: {result.stderr}"
        assert output_dir.exists(), "Output directory was not created"


def test_generate_id_lookup_table_cli():
    """Test the 'b2aiprep-cli generate-id-lookup-table' command using subprocess."""
    with tempfile.TemporaryDirectory() as temp_dir:
        ids = {"participant_id": ["P001", "P002"]}
        id_df = pd.DataFrame(ids)
        id_dir = Path(temp_dir) / "ids.tsv"

        id_df.to_csv(id_dir, sep='\t', index=False)

        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "b2aiprep-cli",
            "generate-id-lookup-table",
            id_dir,
            output_dir
        ]

        output_file = output_dir / "id_lookup_table.json"
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, f"CLI command failed: {result.stderr}"
        assert output_dir.exists(), "Output directory was not created"
        assert output_file.exists(), "Output file was not created"

def test_generate_id_lookup_table_existing_remap_cli():
    """Test the 'b2aiprep-cli generate-id-lookup-table' command with --load_lookup."""
    with tempfile.TemporaryDirectory() as temp_dir:
        ids = {"participant_id": ["P001", "P002"]}
        id_df = pd.DataFrame(ids)
        id_dir = Path(temp_dir) / "ids.tsv"
        remap = {"P003": "123456", "P004": "678910"}
        remap_file_path = Path(temp_dir) / "id_remap.json"
        with open(remap_file_path, "w") as file:
            json.dump(remap, file)
        id_df.to_csv(id_dir, sep='\t', index=False)

        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            "b2aiprep-cli",
            "generate-id-lookup-table",
            id_dir,
            output_dir,
            "--load_lookup",
            remap_file_path
        ]

        output_file = output_dir / "id_lookup_table.json"
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, f"CLI command failed: {result.stderr}"
        assert output_dir.exists(), "Output directory was not created"
        assert output_file.exists(), "Output file was not created"
        
        with open(output_file, 'r') as file:
            data = json.load(file)

        assert "P003" in data, "Missing Participant in remap"
        assert data["P003"] == "123456", "Remap is incorrect"
        assert "P004" in data, "Missing Participant in remap"
        assert data["P004"] == "678910", "Remap is incorrect"
        assert "P001" in data, "P001 is missing in remap file"
        assert "P002" in data, "P002 is missing in remap file"


def _write_ids_tsv(path, ids, id_column="record_id"):
    pd.DataFrame({id_column: ids}).to_csv(path, sep="\t", index=False)


def _read_split_files(output_dir):
    files = sorted(Path(output_dir).glob("subject_ids_*.txt"))
    return [f.read_text().strip().splitlines() for f in files]


class TestCreateSubjectSplits:
    """Tests for the create_subject_splits command."""

    def test_splits_default_chunk_size(self, tmp_path):
        input_file = tmp_path / "ids.tsv"
        _write_ids_tsv(input_file, [f"R{i:03d}" for i in range(25)])
        output_dir = tmp_path / "out"

        runner = CliRunner()
        result = runner.invoke(create_subject_splits, [str(input_file), str(output_dir)])
        assert result.exit_code == 0, result.output

        chunks = _read_split_files(output_dir)
        assert [len(c) for c in chunks] == [10, 10, 5]
        # Round-trip covers every unique ID exactly once.
        flat = [x for chunk in chunks for x in chunk]
        assert sorted(flat) == sorted({f"R{i:03d}" for i in range(25)})

    def test_splits_exact_multiple(self, tmp_path):
        input_file = tmp_path / "ids.tsv"
        _write_ids_tsv(input_file, [f"R{i}" for i in range(20)])
        output_dir = tmp_path / "out"

        result = CliRunner().invoke(
            create_subject_splits,
            [str(input_file), str(output_dir), "--num_participants_per_file", "5"],
        )
        assert result.exit_code == 0, result.output
        chunks = _read_split_files(output_dir)
        assert [len(c) for c in chunks] == [5, 5, 5, 5]

    def test_single_participant_creates_one_file(self, tmp_path):
        input_file = tmp_path / "ids.tsv"
        _write_ids_tsv(input_file, ["only_one"])
        output_dir = tmp_path / "out"

        result = CliRunner().invoke(create_subject_splits, [str(input_file), str(output_dir)])
        assert result.exit_code == 0, result.output
        chunks = _read_split_files(output_dir)
        assert chunks == [["only_one"]]

    def test_deduplicates_ids(self, tmp_path):
        input_file = tmp_path / "ids.tsv"
        _write_ids_tsv(input_file, ["A", "B", "A", "C", "B"])
        output_dir = tmp_path / "out"

        result = CliRunner().invoke(create_subject_splits, [str(input_file), str(output_dir)])
        assert result.exit_code == 0, result.output
        chunks = _read_split_files(output_dir)
        assert chunks == [["A", "B", "C"]]

    def test_accepts_csv_input(self, tmp_path):
        # RedCap exports are comma-separated; the command should dispatch on extension.
        input_file = tmp_path / "redcap.csv"
        pd.DataFrame({"record_id": ["A", "B", "C"]}).to_csv(input_file, index=False)
        output_dir = tmp_path / "out"

        result = CliRunner().invoke(create_subject_splits, [str(input_file), str(output_dir)])
        assert result.exit_code == 0, result.output
        assert _read_split_files(output_dir) == [["A", "B", "C"]]

    def test_custom_id_column(self, tmp_path):
        input_file = tmp_path / "ids.tsv"
        _write_ids_tsv(input_file, ["P1", "P2", "P3"], id_column="participant_id")
        output_dir = tmp_path / "out"

        result = CliRunner().invoke(
            create_subject_splits,
            [str(input_file), str(output_dir), "--id_column", "participant_id"],
        )
        assert result.exit_code == 0, result.output
        chunks = _read_split_files(output_dir)
        assert chunks == [["P1", "P2", "P3"]]

    def test_creates_output_dir_when_missing(self, tmp_path):
        input_file = tmp_path / "ids.tsv"
        _write_ids_tsv(input_file, ["A"])
        output_dir = tmp_path / "nested" / "not_yet_created"
        assert not output_dir.exists()

        result = CliRunner().invoke(create_subject_splits, [str(input_file), str(output_dir)])
        assert result.exit_code == 0, result.output
        assert output_dir.is_dir()
        assert (output_dir / "subject_ids_0.txt").exists()

    def test_missing_id_column_errors(self, tmp_path):
        input_file = tmp_path / "ids.tsv"
        _write_ids_tsv(input_file, ["A"], id_column="participant_id")
        output_dir = tmp_path / "out"

        result = CliRunner().invoke(create_subject_splits, [str(input_file), str(output_dir)])
        assert result.exit_code != 0
        assert "record_id" in result.output

    def test_invalid_num_per_file_errors(self, tmp_path):
        input_file = tmp_path / "ids.tsv"
        _write_ids_tsv(input_file, ["A"])
        output_dir = tmp_path / "out"

        result = CliRunner().invoke(
            create_subject_splits,
            [str(input_file), str(output_dir), "--num_participants_per_file", "0"],
        )
        assert result.exit_code != 0

    def test_help_documents_options(self):
        result = CliRunner().invoke(create_subject_splits, ["--help"])
        assert result.exit_code == 0
        assert "num_participants_per_file" in result.output
        assert "id_column" in result.output


# --- validate-bundled-dataset ---------------------------------------------------------------

def _bundle(root, phenotype_participants=("005009",), session_columns=(), sessions=(("005009", "01"),),
            tables=None, omit=()):
    """A minimal bundle in v4 naming (zero-padded pseudonyms, ordinal session labels).

    Features hold one row for 005009 / 01. *tables*: {"group/name": (DataFrame, sidecar or None)}.
    """
    features, task = root / "features", root / "phenotype" / "task"
    features.mkdir(parents=True)
    task.mkdir(parents=True)
    df = pd.DataFrame({"participant_id": ["005009"], "task_name": ["test"], "session_id": ["01"]})
    df.to_parquet(features / "torchaudio_spectrogram.parquet")
    df.to_parquet(features / "torchaudio_mfcc.parquet")
    if "static_features.tsv" not in omit:
        (features / "static_features.tsv").write_text("participant_id\tsession_id\n005009\t01\n")
    (features / "static_features.json").write_text("{}")
    pd.DataFrame({"participant_id": [p for p, _ in sessions], "session_id": [s for _, s in sessions],
                  **{c: ["x"] * len(sessions) for c in session_columns}}).to_csv(
        task / "session.tsv", sep="\t", index=False)
    pd.DataFrame({"participant_id": list(phenotype_participants)}).to_csv(
        root / "phenotype" / "demographics.tsv", sep="\t", index=False)
    for path, (table, sidecar) in (tables or {}).items():
        target = root / "phenotype" / f"{path}.tsv"
        target.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(target, sep="\t", index=False)
        if sidecar is not None:
            target.with_suffix(".json").write_text(json.dumps(sidecar))
    return root


def _allowlist_config(root, verdicts=None):
    """A v4 deidentify config: an allowlist, no participants_to_remove.json."""
    root.mkdir()
    (root / "participants_to_include.json").write_text(json.dumps(["p1", "p2"]))
    (root / "id_remapping.json").write_text(json.dumps({"p1": "005009", "p2": "005010"}))
    (root / "deidentify_settings.json").write_text(json.dumps({"access_tier": "registered"}))
    (root / "audio_tasks_to_include.json").write_text(json.dumps(["test"]))
    if verdicts is not None:
        (root / "column_value_reviews.json").write_text(json.dumps({"verdicts": verdicts}))
    return root


def _validate(bundle, config):
    return CliRunner().invoke(validate_bundled_dataset, ["--cohort", "adult", str(bundle), str(config)])


def test_validate_bundled_dataset_flags_tables_of_another_cohort(tmp_path):
    """The bundle has no dataset_description.json, so a wrong --cohort shows as unknown tables."""
    bundle = _bundle(tmp_path / "bundle", tables={
        "pediatric/pediatric_demographics": (pd.DataFrame({"participant_id": ["005009"]}), None)})
    result = _validate(bundle, _allowlist_config(tmp_path / "cfg"))
    assert result.exit_code != 0
    assert "not in the adult field map" in result.output and "pediatric_demographics" in result.output


def test_validate_bundled_dataset_passes_a_release_bundle(tmp_path):
    bundle = _bundle(tmp_path / "bundle", ["005009", "005010"],
                     session_columns=["session_status", "session_index", "session_local_hour"])
    result = _validate(bundle, _allowlist_config(tmp_path / "cfg"))
    assert result.exit_code == 0, result.output
    assert "PASSED" in result.output


def test_validate_bundled_dataset_with_a_removal_list_config(tmp_path, setup_publish_config):
    """The older participants_to_remove.json config (no allowlist) still validates."""
    result = _validate(_bundle(tmp_path / "bundle"), setup_publish_config)
    assert result.exit_code == 0, result.output


def test_validate_bundled_dataset_flags_a_pseudonym_not_on_the_allowlist(tmp_path):
    """Leading-zero pseudonyms such as 005009 and '01' session labels are read as text and still match."""
    result = _validate(_bundle(tmp_path / "bundle", ["005009", "089247"]), _allowlist_config(tmp_path / "cfg"))
    assert result.exit_code != 0
    assert "should not be released in demographics.tsv: 1" in result.output
    assert "not found in" not in result.output


def test_validate_bundled_dataset_fails_without_static_features(tmp_path):
    result = _validate(_bundle(tmp_path / "bundle", omit=("static_features.tsv",)), _allowlist_config(tmp_path / "cfg"))
    assert result.exit_code != 0
    assert "Missing required file in features/: static_features.tsv" in result.output


def test_validate_bundled_dataset_fails_without_allowlist_or_removal_list(tmp_path):
    cfg = tmp_path / "cfg"
    cfg.mkdir()
    (cfg / "id_remapping.json").write_text("{}")
    (cfg / "deidentify_settings.json").write_text(json.dumps({"access_tier": "registered"}))
    (cfg / "audio_tasks_to_include.json").write_text(json.dumps(["test"]))
    result = _validate(_bundle(tmp_path / "bundle"), cfg)
    assert result.exit_code != 0 and "config file not found" in result.output
    assert "participants_to_remove.json" in result.output


def _recording_tables(bundle, annotations=None, transcripts=None, quality=None):
    """Per-recording tables in features/ for the one released recording r1 (005009 / 01 / test).
    Each argument overrides that table's columns; its JSON describes the default columns."""
    features = bundle / "features"
    pd.DataFrame({"recording_id": ["r1"]}).to_csv(bundle / "phenotype" / "task" / "recording.tsv", sep="\t", index=False)
    keys = {"participant_id": ["005009"], "session_id": ["01"], "task_name": ["test"]}
    for name, ext, cols, extra in (
        ("audio_quality_annotations", ".tsv", annotations, {"recording_id": ["r1"], "snr_db": ["20"]}),
        ("consensus_transcripts", ".parquet", transcripts, {"recording_id": ["r1"], "word_index": [0], "text": ["a"]}),
        ("audio_quality_metrics", ".tsv", quality, {"snr": ["1"]}),
    ):
        df = pd.DataFrame({**keys, **extra})
        (features / f"{name}.json").write_text(json.dumps({c: {"description": c} for c in df.columns}))
        df = pd.DataFrame({**keys, **extra, **(cols or {})})
        df.to_csv(features / f"{name}{ext}", sep="\t", index=False) if ext == ".tsv" else df.to_parquet(features / f"{name}{ext}")
    (features / "static_features.tsv").write_text("participant_id\tsession_id\ttask_name\n005009\t01\ttest\n")
    return bundle


def test_validate_bundled_dataset_passes_per_recording_tables(tmp_path):
    result = _validate(_recording_tables(_bundle(tmp_path / "bundle")), _allowlist_config(tmp_path / "cfg"))
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("tables, message", [
    ({"annotations": {"participant_id": ["005099"]}}, "should not be released in audio_quality_annotations.tsv"),
    ({"annotations": {"recording_id": ["r9"]}}, "Recordings in audio_quality_annotations.tsv not in phenotype/task/recording.tsv"),
    ({"annotations": {"session_id": ["02"]}}, "Sessions in audio_quality_annotations.tsv not found"),
    ({"annotations": {"extra_column": ["x"]}}, "not described in audio_quality_annotations.json: ['extra_column']"),
    ({"transcripts": {"task_name": ["free-speech-1"]}}, "unexpected audio tasks in consensus_transcripts.parquet"),
    ({"transcripts": {"participant_id": ["p1"]}}, "non-remapped) participant IDs in consensus_transcripts.parquet"),
    ({"quality": {"task_name": ["other-task"]}}, "Rows in audio_quality_metrics.tsv for recordings without released features: 1"),
], ids=["participant", "recording", "session", "undescribed-column", "sensitive-transcript", "unmapped-id",
        "quality-metrics-without-features"])
def test_validate_bundled_dataset_flags_per_recording_tables(tmp_path, tables, message):
    bundle = _recording_tables(_bundle(tmp_path / "bundle"), **tables)
    result = _validate(bundle, _allowlist_config(tmp_path / "cfg"))
    assert result.exit_code != 0
    assert message in result.output, result.output


def test_bundle_copies_per_recording_tables_present(tmp_path):
    bids, features = tmp_path / "bids", tmp_path / "features"
    bids.mkdir()
    features.mkdir()
    pd.DataFrame({"participant_id": ["005009"], "recording_id": ["r1"], "task_name": ["test"]}).to_parquet(
        bids / "consensus_transcripts.parquet")
    (bids / "consensus_transcripts.json").write_text("{}")
    stats = {}
    _copy_recording_tables_to_bundle(bids, features, stats)
    assert sorted(p.name for p in features.iterdir()) == ["consensus_transcripts.json", "consensus_transcripts.parquet"]
    assert stats["consensus_transcripts.parquet"]["examples"] == 1


@pytest.mark.parametrize("column", ["session_started_at", "session_complete", "session_site", "not_a_field",
                                    "record_id"])
def test_validate_bundled_dataset_fails_on_columns_a_release_must_not_have(tmp_path, column):
    """Internal, drop, date-shifted, unknown, and the original participant ID."""
    result = _validate(_bundle(tmp_path / "bundle", session_columns=[column]), _allowlist_config(tmp_path / "cfg"))
    assert result.exit_code != 0 and column in result.output


@pytest.mark.parametrize("verdicts, passes", [
    (None, False),
    ([{"participant_id": "p1", "column_name": "other_voice_activity", "verdict": "safe"}], True),
], ids=["unreviewed", "reviewed"])
def test_validate_bundled_dataset_allows_a_review_column_only_with_verdicts(tmp_path, verdicts, passes):
    """other_voice_activity is a review column (shipped field map)."""
    bundle = _bundle(tmp_path / "bundle", tables={"confounders/confounders": (pd.DataFrame(
        {"participant_id": ["005009"], "other_voice_activity": ["x"]}), None)})
    result = _validate(bundle, _allowlist_config(tmp_path / "cfg", verdicts))
    assert (result.exit_code == 0) is passes, result.output
    if not passes:
        assert "other_voice_activity" in result.output


def test_unreleasable_columns_flags_controlled_only_columns_outside_the_controlled_tier(tmp_path):
    fm = pd.DataFrame({"schema_name": ["t", "t"], "column_name": ["a", "b"], "disposition": ["release", "release"],
                       "date_shift": ["NO", "NO"], "access_tier": ["", "controlled"]})
    tsv = tmp_path / "t.tsv"
    assert any("controlled-only" in i and "b" in i for i in _unreleasable_columns(tsv, ["participant_id", "a", "b"], fm))
    assert _unreleasable_columns(tsv, ["participant_id", "a", "b"], fm, AccessTier.CONTROLLED) == []


def test_validate_bundled_dataset_compares_sessions_per_participant(tmp_path):
    """Ordinal labels repeat across participants: 005009's session 01 is not 005010's."""
    bundle = _bundle(tmp_path / "bundle", ["005009", "005010"], sessions=(("005010", "01"),))
    result = _validate(bundle, _allowlist_config(tmp_path / "cfg"))
    assert result.exit_code != 0 and "torchaudio_mfcc.parquet not found in session.tsv" in result.output


def test_validate_bundled_dataset_needs_participant_id_in_session_tsv(tmp_path):
    bundle = _bundle(tmp_path / "bundle")
    (bundle / "phenotype" / "task" / "session.tsv").write_text("session_id\n01\n")
    result = _validate(bundle, _allowlist_config(tmp_path / "cfg"))
    assert result.exit_code != 0 and "missing required column(s) participant_id" in result.output


# ph_walking has answer choices; other_voice_activity is free text (a review column)
@pytest.mark.parametrize("ph_walking, free_text, passes", [
    ("None", "Mild", False),  # an answer choice that readers drop as missing
    ("Mild", "N/A", True),    # typed free text is left as written
], ids=["answer-choice", "free-text"])
def test_validate_bundled_dataset_flags_answer_choices_readers_take_for_missing(tmp_path, ph_walking, free_text,
                                                                                passes):
    sidecar = {"confounders": {"data_elements": {
        "ph_walking": {"choices": [{"name": {"en": "None"}, "value": "none"}, {"name": {"en": "Mild"}, "value": "mild"}]},
        "other_voice_activity": {"description": "free text"}}}}
    bundle = _bundle(tmp_path / "bundle", tables={"confounders/confounders": (pd.DataFrame(
        {"participant_id": ["005009"], "ph_walking": [ph_walking], "other_voice_activity": [free_text]}), sidecar)})
    verdicts = [{"participant_id": "p1", "column_name": "other_voice_activity", "verdict": "safe"}]
    result = _validate(bundle, _allowlist_config(tmp_path / "cfg", verdicts))
    assert (result.exit_code == 0) is passes, result.output
    if not passes:
        assert "ph_walking (1)" in result.output and "other_voice_activity (" not in result.output

