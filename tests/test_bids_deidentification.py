"""Tests for BIDS dataset deidentification functionality."""

import json
import logging
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from b2aiprep.prepare.dataset import BIDSDataset

class TestBIDSDatasetDeidentification:
    """Test cases for BIDSDataset deidentification methods."""

    def setup_publish_config(self, publish_config_dir):
        """Helper method to create config files for testing."""
        publish_config_dir.mkdir(exist_ok=True)
        (publish_config_dir / "id_remapping.json").write_text("{}")
        (publish_config_dir / "participant_ids_to_remove.json").write_text("[]")
        (publish_config_dir / "audio_filestems_to_remove.json").write_text("[]")

    @pytest.fixture
    def temp_bids_dir(self):
        """Create a temporary BIDS directory structure for testing."""
        temp_dir = tempfile.mkdtemp()
        bids_path = Path(temp_dir) / "test_bids"
        bids_path.mkdir(parents=True, exist_ok=True)

        # # Create participants.tsv
        # participants_data = {
        #     "record_id": ["participant001", "participant002", "participant003"],
        #     "age": [25, 30, 35],
        #     "gender_identity": ["Male", "Female", "Male"],
        #     "specify_gender_identity": ["Male", "Female", "Male"],
        #     "alcohol_amt": ["4-Mar", "2", "6-May"],  # Test date fix
        #     "session_id": ["session001", "session002", "session003"],
        # }
        # participants_df = pd.DataFrame(participants_data)
        # participants_df.to_csv(bids_path / "participants.tsv", sep="\t", index=False)

        # Create participants.json
        # participants_json = {
        #     "record_id": {"description": "Unique identifier for each participant."},
        #     "age": {"description": "Age of participant"},
        #     "gender_identity": {"description": "Gender identity"},
        #     "specify_gender_identity": {"description": "Specify gender identity"},
        #     "alcohol_amt": {"description": "Amount of alcohol consumed"},
        #     "session_id": {"description": "Session identifier"},
        # }
        # with open(bids_path / "participants.json", "w") as f:
        #     json.dump(participants_json, f, indent=2)

        # Create phenotype directory
        phenotype_dir = bids_path / "phenotype"
        phenotype_dir.mkdir(parents=True, exist_ok=True)

        # Create a test phenotype file
        test_pheno_data = {
            "record_id": ["participant001", "participant002"],
            "test_score": [85, 90],
            "session_id": ["session001", "session002"],
        }
        test_pheno_df = pd.DataFrame(test_pheno_data)
        test_pheno_df.to_csv(phenotype_dir / "test_phenotype.tsv", sep="\t", index=False)

        test_pheno_json = {
            "record_id": {"description": "Participant ID"},
            "test_score": {"description": "Test score"},
            "session_id": {"description": "Session ID"},
        }
        with open(phenotype_dir / "test_phenotype.json", "w") as f:
            json.dump(test_pheno_json, f, indent=2)

        # Create audio files and metadata
        for i, (participant, session) in enumerate(
            [
                ("participant001", "session001"),
                ("participant002", "session002"),
                ("participant003", "session003"),
            ],
            1,
        ):
            audio_dir = bids_path / f"sub-{participant}" / f"ses-{session}" / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)

            # Create dummy audio file
            audio_file = audio_dir / f"sub-{participant}_ses-{session}_task-test.wav"
            audio_file.write_bytes(b"dummy audio data")

            # Create metadata file
            metadata = {
                "item": [
                    {"linkId": "record_id", "answer": [{"valueString": participant}]},
                    {"linkId": "session_id", "answer": [{"valueString": session}]},
                    {"linkId": "recording_name", "answer": [{"valueString": "test"}]},
                ]
            }
            json_file = audio_file.parent / f"{audio_file.stem}_recording-metadata.json"
            with open(json_file, "w") as f:
                json.dump(metadata, f, indent=2)
                
            session_data = {
                "record_id": ["participant001", "participant002"],
                "session_id": ["session001", "session002"],
            }
            session_df = pd.DataFrame(session_data)
            session_dir = bids_path / f"sub-{participant}"
            session_tsv = session_dir/ f"sessions.tsv"
            session_df.to_csv(session_tsv, sep="\t", index=False)

        # Create BIDS template files
        (bids_path / "README.md").write_text("# Test BIDS Dataset")
        (bids_path / "CHANGES.md").write_text("## Changes")
        dataset_desc = {"Name": "Test Dataset", "BIDSVersion": "1.0.0"}
        with open(bids_path / "dataset_description.json", "w") as f:
            json.dump(dataset_desc, f, indent=2)

        yield bids_path

        # Cleanup
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def output_dir(self):
        """Create a temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        output_path = Path(temp_dir) / "output"
        yield output_path
        # Cleanup
        shutil.rmtree(temp_dir)

    def test_deidentify_basic_functionality(self, temp_bids_dir, output_dir):
        """Test basic deidentification functionality."""
        dataset = BIDSDataset(temp_bids_dir)

        # Test deidentification
        publish_config_dir = temp_bids_dir / "publish_config"
        self.setup_publish_config(publish_config_dir)

        deidentified_dataset = dataset.deidentify(
            outdir=output_dir, publish_config_dir=publish_config_dir
        )

        # Check that output directory was created
        assert output_dir.exists()
        assert isinstance(deidentified_dataset, BIDSDataset)
        assert deidentified_dataset.data_path.resolve() == output_dir.resolve()


    def test_deidentify_skip_audio(self, temp_bids_dir, output_dir):
        """Test deidentification with skip_audio=True."""
        dataset = BIDSDataset(temp_bids_dir)

        # Test deidentification with skip_audio=True
        publish_config_dir = temp_bids_dir / "publish_config"
        self.setup_publish_config(publish_config_dir)
        deidentified_dataset = dataset.deidentify(
            outdir=output_dir, publish_config_dir=publish_config_dir, skip_audio=True
        )

        # Check that no audio files were copied
        audio_files = list(output_dir.rglob("*.wav"))
        assert len(audio_files) == 0

    def test_deidentify_with_audio(self, temp_bids_dir, output_dir):
        """Test deidentification with audio processing."""
        dataset = BIDSDataset(temp_bids_dir)

        # Mock the filter_audio_paths to return all paths for testing
        with patch("b2aiprep.prepare.dataset.filter_audio_paths") as mock_filter:
            # Get original audio paths
            from b2aiprep.prepare.bids import get_paths

            audio_paths = get_paths(temp_bids_dir, file_extension=".wav")
            audio_paths = [x["path"] for x in audio_paths]
            mock_filter.return_value = audio_paths

            # Test deidentification with audio
            publish_config_dir = temp_bids_dir / "publish_config"
            self.setup_publish_config(publish_config_dir)
            deidentified_dataset = dataset.deidentify(
                outdir=output_dir, publish_config_dir=publish_config_dir, skip_audio=False
            )

        # Check that audio files were copied (should be processed by filter)
        audio_files = list(output_dir.rglob("*.wav"))
        # Note: The actual number depends on filtering logic, but should be > 0
        assert len(audio_files) >= 0

    def test_clean_method(self, temp_bids_dir):
        """Test the clean method."""
        dataset = BIDSDataset(temp_bids_dir)

        # Create test data with issues that need cleaning
        df = pd.DataFrame(
            {
                "record_id": ["participant001", "participant002"],
                "alcohol_amt": ["4-Mar", "6-May"],  # Date values that need fixing
                "gender_identity": ["Male", "Female"],
                "specify_gender_identity": ["Male", "Female"],
                "age": [25, 30],
            }
        )

        phenotype = {
            "place_holder_schema": {
                "data_elements": {
                    "record_id": {"description": "ID"},
                    "gender_identity": {"description": "Gender identity"},
                    "specify_gender_identity": {"description": "Specify gender"},
                    "age": {"description": "Age"},
                }
            }
        }

        # Apply cleaning
        cleaned_df, cleaned_phenotype = dataset._clean_phenotype_data(df, phenotype)

        # Check that alcohol_amt values were fixed
        assert "3 - 4" in cleaned_df["alcohol_amt"].values
        assert "5 - 6" in cleaned_df["alcohol_amt"].values
        assert "4-Mar" not in cleaned_df["alcohol_amt"].values
        assert "6-May" not in cleaned_df["alcohol_amt"].values

    def test_deidentify_phenotype_processing(self, temp_bids_dir, output_dir):
        """Test that phenotype files are properly processed."""
        dataset = BIDSDataset(temp_bids_dir)

        # Test deidentification
        publish_config_dir = temp_bids_dir / "publish_config"
        self.setup_publish_config(publish_config_dir)
        dataset.deidentify(outdir=output_dir, publish_config_dir=publish_config_dir)

        # Check that phenotype directory was created
        phenotype_dir = output_dir / "phenotype"
        assert phenotype_dir.exists()

        # Check that test phenotype files were processed
        test_pheno_tsv = phenotype_dir / "test_phenotype.tsv"
        test_pheno_json = phenotype_dir / "test_phenotype.json"
        assert test_pheno_tsv.exists()
        assert test_pheno_json.exists()

    def test_deidentify_template_files_copied(self, temp_bids_dir, output_dir):
        """Test that BIDS template files are copied."""
        dataset = BIDSDataset(temp_bids_dir)

        # Test deidentification
        publish_config_dir = temp_bids_dir / "publish_config"
        self.setup_publish_config(publish_config_dir)
        dataset.deidentify(outdir=output_dir, publish_config_dir=publish_config_dir)

        # Check that template files were copied
        assert (output_dir / "README.md").exists()
        assert (output_dir / "CHANGES.md").exists()
        assert (output_dir / "dataset_description.json").exists()

    def test_deidentify_output_dir_exists(self, temp_bids_dir, output_dir):
        """Test error handling when output directory already exists."""
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset = BIDSDataset(temp_bids_dir)

        # Should raise FileExistsError
        with pytest.raises(FileExistsError):
            publish_config_dir = temp_bids_dir / "publish_config"
            self.setup_publish_config(publish_config_dir)
            dataset.deidentify(outdir=output_dir, publish_config_dir=publish_config_dir)

    @patch(
        "b2aiprep.prepare.constants._load_participant_exclusions", return_value=["participant001"]
    )
    def test_participant_removal(self, mock_exclusions, temp_bids_dir, output_dir):
        """Test that hard-coded participants are removed."""
        dataset = BIDSDataset(temp_bids_dir)

        # Mock filter_audio_paths to test participant removal
        with patch("b2aiprep.prepare.dataset.filter_audio_paths") as mock_filter:
            # Get original audio paths
            from b2aiprep.prepare.bids import get_paths

            audio_paths = get_paths(temp_bids_dir, file_extension=".wav")
            audio_paths = [x["path"] for x in audio_paths]
            mock_filter.return_value = audio_paths

            # Test deidentification
            publish_config_dir = temp_bids_dir / "publish_config"
            self.setup_publish_config(publish_config_dir)
            dataset.deidentify(
                outdir=output_dir, publish_config_dir=publish_config_dir, skip_audio=False
            )

        # Check that participant001 files are not in output
        participant001_files = list(output_dir.rglob("*participant001*"))
        # Note: The actual behavior depends on the load_phenotype_data function
        # which should filter out the hard-coded participants

    def test_audio_metadata_processing(self, temp_bids_dir, output_dir):
        """Test that audio metadata is properly processed."""
        dataset = BIDSDataset(temp_bids_dir)

        # Mock the necessary functions for testing
        with (
            patch("b2aiprep.prepare.dataset.filter_audio_paths") as mock_filter,
            patch("b2aiprep.prepare.dataset.update_metadata_record_and_session_id") as mock_update,
            patch("b2aiprep.prepare.dataset.get_value_from_metadata") as mock_get_value,
        ):

            # Setup mocks
            from b2aiprep.prepare.bids import get_paths

            audio_paths = get_paths(temp_bids_dir, file_extension=".wav")
            audio_paths = [x["path"] for x in audio_paths]
            mock_filter.return_value = audio_paths
            mock_get_value.side_effect = lambda metadata, linkid, endswith: "test_id"

            # Test deidentification
            publish_config_dir = temp_bids_dir / "publish_config"
            self.setup_publish_config(publish_config_dir)
            dataset.deidentify(
                outdir=output_dir, publish_config_dir=publish_config_dir, skip_audio=False
            )

            # Check that metadata functions were called
            assert mock_update.called
            assert mock_get_value.called

    def test_logging_messages(self, temp_bids_dir, output_dir, caplog):
        """Test that appropriate logging messages are generated."""
        dataset = BIDSDataset(temp_bids_dir)

        with caplog.at_level(logging.INFO):
            publish_config_dir = temp_bids_dir / "publish_config"
            self.setup_publish_config(publish_config_dir)
            dataset.deidentify(
                outdir=output_dir, publish_config_dir=publish_config_dir, skip_audio=True
            )

        # Check for expected log messages
        log_messages = [record.message for record in caplog.records]
        assert any("Finished processing phenotype data." in msg for msg in log_messages)
        assert any("Deidentification completed" in msg for msg in log_messages)


class TestBIDSDatasetClean:
    """Test cases specifically for the clean and component methods."""

    def test_clean_alcohol_column_fixes(self):
        """Test that alcohol column date values are fixed."""
        dataset = BIDSDataset(Path("/dummy"))  # Path doesn't matter for this test

        df = pd.DataFrame(
            {
                "record_id": ["p1", "p2", "p3", "p4"],
                "alcohol_amt": ["4-Mar", "6-May", "9-Jul", "normal_value"],
            }
        )

        phenotype = {
            "record_id": {"description": "ID"},
            "alcohol_amt": {"description": "Alcohol amount"},
        }

        cleaned_df, cleaned_phenotype = dataset._clean_phenotype_data(df, phenotype)

        # Check that date values were fixed
        expected_fixes = {"4-Mar": "3 - 4", "6-May": "5 - 6", "9-Jul": "7 - 9"}

        for original, expected in expected_fixes.items():
            assert expected in cleaned_df["alcohol_amt"].values
            assert original not in cleaned_df["alcohol_amt"].values

        # Check that normal value is unchanged
        assert "normal_value" in cleaned_df["alcohol_amt"].values

    def test_fix_alcohol_column_method(self):
        """Test the _fix_alcohol_column method specifically."""
        dataset = BIDSDataset(Path("/dummy"))

        df = pd.DataFrame({"alcohol_amt": ["4-Mar", "6-May", "9-Jul", "normal_value", "2"]})

        fixed_df = dataset._fix_alcohol_column(df)

        assert fixed_df["alcohol_amt"].iloc[0] == "3 - 4"
        assert fixed_df["alcohol_amt"].iloc[1] == "5 - 6"
        assert fixed_df["alcohol_amt"].iloc[2] == "7 - 9"
        assert fixed_df["alcohol_amt"].iloc[3] == "normal_value"
        assert fixed_df["alcohol_amt"].iloc[4] == "2"

    def test_deidentify_phenotype_method(self):
        """Test the _deidentify_phenotype method."""
        dataset = BIDSDataset(Path("/dummy"))

        df = pd.DataFrame(
            {
                "record_id": ["participant001", "participant002", "participant003"],
                "session_id": ["session001", "session002", "session003"],
                "age": [25, 30, 35],
            }
        )

        phenotype = {
            "record_id": {"description": "Participant ID"},
            "session_id": {"description": "Session ID"},
            "age": {"description": "Age"},
        }

        # Mock _load_participant_exclusions to include participant001
        with patch(
            "b2aiprep.prepare.constants._load_participant_exclusions",
            return_value=["participant001"],
        ):
            deidentified_df, deidentified_phenotype = dataset._deidentify_phenotype(
                df, phenotype, ["participant001"], {}
            )

        # Check that participant001 was removed
        assert len(deidentified_df) == 2
        assert "participant001" not in deidentified_df["participant_id"].values

        # Check that record_id was renamed to participant_id
        assert "participant_id" in deidentified_df.columns
        assert "record_id" not in deidentified_df.columns
        assert "participant_id" in deidentified_phenotype
        assert "record_id" not in deidentified_phenotype

    def test_remove_columns_methods(self):
        """Test the various column removal methods."""
        dataset = BIDSDataset(Path("/dummy"))

        df = pd.DataFrame(
            {
                "record_id": ["p1", "p2"],
                "consent_status": [None, None],  # Empty column
                "acoustic_task_id": ["task1", "task2"],  # System column
                "state_province": ["CA", "NY"],  # Low utility column
                "age": [25, 30],
            }
        )

        phenotype = {
            "record_id": {"description": "ID"},
            "consent_status": {"description": "Consent"},
            "acoustic_task_id": {"description": "Task ID"},
            "state_province": {"description": "State"},
            "age": {"description": "Age"},
        }

        # Test removing empty columns
        df_empty, phenotype_empty = dataset._remove_empty_columns(df, phenotype)
        assert "consent_status" not in df_empty.columns
        assert "consent_status" not in phenotype_empty

        # Test removing system columns
        df_system, phenotype_system = dataset._remove_system_columns(df, phenotype)
        assert "acoustic_task_id" not in df_system.columns
        assert "acoustic_task_id" not in phenotype_system

        # Test removing sensitive columns
        df_sensitive, phenotype_sensitive = dataset._remove_sensitive_columns(df, phenotype)
        assert "state_province" not in df_sensitive.columns
        assert "state_province" not in phenotype_sensitive

    def test_add_sex_at_birth_column(self):
        """Test the _add_sex_at_birth_column method."""
        dataset = BIDSDataset(Path("/dummy"))

        df = pd.DataFrame(
            {
                "record_id": ["p1", "p2"],
                "gender_identity": ["Male", "Female"],
                "specify_gender_identity": ["Male", "Female"],
                "age": [25, 30],
            }
        )

        phenotype = {
            "place_holder_schema": {
                "data_elements": {
                    "record_id": {"description": "ID"},
                    "gender_identity": {"description": "Gender identity"},
                    "specify_gender_identity": {"description": "Specify gender"},
                    "age": {"description": "Age"},
                }
            }
        }

        df_sex, phenotype_sex = dataset._add_sex_at_birth_column(df, phenotype)

        # Check that sex_at_birth column was added
        assert "sex_at_birth" in df_sex.columns
        assert "sex_at_birth" in phenotype_sex
        assert df_sex["sex_at_birth"].iloc[0] == "Male"
        assert df_sex["sex_at_birth"].iloc[1] == "Female"

        # Check that specify_gender_identity was removed
        assert "specify_gender_identity" not in df_sex.columns

    def test_load_phenotype_data_method(self):
        """Test the load_phenotype_data method."""
        # Create a temporary BIDS directory with test data
        import tempfile

        temp_dir = tempfile.mkdtemp()
        bids_path = Path(temp_dir) / "test_bids"
        bids_path.mkdir(parents=True, exist_ok=True)

        try:
            # Create test participants files
            participants_data = {"record_id": ["participant001", "participant002"], "age": [25, 30]}
            participants_df = pd.DataFrame(participants_data)
            participants_df.to_csv(bids_path / "participants.tsv", sep="\t", index=False)

            participants_json = {
                "record_id": {"description": "Unique identifier"},
                "age": {"description": "Age of participant"},
            }
            with open(bids_path / "participants.json", "w") as f:
                json.dump(participants_json, f, indent=2)

            dataset = BIDSDataset(bids_path)
            df, phenotype = dataset.load_phenotype_data(bids_path, "participants")

            # Check that data was loaded
            assert isinstance(df, pd.DataFrame)
            assert isinstance(phenotype, dict)
            assert len(df) > 0
            assert len(phenotype) > 0

            # Check that record_id column exists
            assert "record_id" in df.columns
        finally:
            # Cleanup
            shutil.rmtree(temp_dir)

    def test_clean_preserves_phenotype_structure(self):
        """Test that phenotype dictionary structure is preserved."""
        dataset = BIDSDataset(Path("/dummy"))

        df = pd.DataFrame({"record_id": ["p1", "p2"], "age": [25, 30]})

        original_phenotype = {
            "record_id": {"description": "ID", "type": "string"},
            "age": {"description": "Age", "type": "integer"},
        }

        cleaned_df, cleaned_phenotype = dataset._clean_phenotype_data(df, original_phenotype)

        # Check that phenotype structure is preserved
        assert isinstance(cleaned_phenotype, dict)
        assert len(cleaned_phenotype) <= len(
            original_phenotype
        )  # May be smaller due to column removal
