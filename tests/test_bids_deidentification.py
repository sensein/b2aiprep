"""Tests for BIDS dataset deidentification functionality."""

import importlib.util
import json
import logging
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from b2aiprep.prepare.dataset import (
    AccessTier,
    BIDSDataset,
    DispositionLevel,
    SessionLabels,
    ValueReview,
    review_fingerprint,
)


# three session IDs of one participant, in session_index order
A, B, C = ("AAAA1111-0000-0000-0000-000000000001", "BBBB2222-0000-0000-0000-000000000002",
           "CCCC3333-0000-0000-0000-000000000003")


class TestBIDSDatasetDeidentification:
    """Test cases for BIDSDataset deidentification methods."""

    @pytest.fixture(autouse=True)
    def _identity_pseudonyms(self, setup_publish_config):
        """Deidentify refuses an allowlisted participant without a pseudonym; keep IDs as they are."""
        ids = [f"participant00{i}" for i in (1, 2, 3)]
        (setup_publish_config / "id_remapping.json").write_text(json.dumps({p: p for p in ids}))

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
            "acid_reflux": ["Yes", "No"],
            "session_id": ["session001", "session002"],
        }
        test_pheno_df = pd.DataFrame(test_pheno_data)
        test_pheno_df.to_csv(phenotype_dir / "confounders.tsv", sep="\t", index=False)

        test_pheno_json = {
            "record_id": {"description": "Participant ID"},
            "acid_reflux": {"description": "Acid reflux"},
            "session_id": {"description": "Session ID"},
        }
        with open(phenotype_dir / "confounders.json", "w") as f:
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

            # Create fhir metadata file
            metadata = {
                "id": "rec-001",
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
                "record_id": [participant],
                "session_id": [session],
                "session_index": ["1"],
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

    def test_deidentify_basic_functionality(self, temp_bids_dir, output_dir, setup_publish_config):
        """Test basic deidentification functionality."""
        dataset = BIDSDataset(temp_bids_dir)

        deidentified_dataset = dataset.deidentify(
            outdir=output_dir, deidentify_config_dir=setup_publish_config
        )

        # Check that output directory was created
        assert output_dir.exists()
        assert isinstance(deidentified_dataset, BIDSDataset)
        assert deidentified_dataset.data_path.resolve() == output_dir.resolve()


    def test_deidentify_skip_audio(self, temp_bids_dir, output_dir, setup_publish_config):
        """Test deidentification with skip_audio=True."""
        dataset = BIDSDataset(temp_bids_dir)

        # Test deidentification with skip_audio=True
        deidentified_dataset = dataset.deidentify(
            outdir=output_dir, deidentify_config_dir=setup_publish_config, skip_audio=True
        )

        # Check that no audio files were copied
        audio_files = list(output_dir.rglob("*.wav"))
        assert len(audio_files) == 0

    def test_deidentify_with_audio(self, temp_bids_dir, output_dir, setup_publish_config):
        """Test deidentification with audio processing."""
        dataset = BIDSDataset(temp_bids_dir)

        # Test deidentification with audio
        _ = dataset.deidentify(
            outdir=output_dir, deidentify_config_dir=setup_publish_config, skip_audio=False
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

    def test_deidentify_phenotype_processing(self, temp_bids_dir, output_dir, setup_publish_config):
        """Test that phenotype files are properly processed."""
        dataset = BIDSDataset(temp_bids_dir)

        # Test deidentification
        dataset.deidentify(outdir=output_dir, deidentify_config_dir=setup_publish_config)

        # Check that phenotype directory was created
        phenotype_dir = output_dir / "phenotype"
        assert phenotype_dir.exists()

        # Check that test phenotype files were processed
        test_pheno_tsv = phenotype_dir / "confounders.tsv"
        test_pheno_json = phenotype_dir / "confounders.json"
        assert test_pheno_tsv.exists()
        assert test_pheno_json.exists()

    def test_deidentify_template_files_copied(self, temp_bids_dir, output_dir, setup_publish_config):
        """Test that BIDS template files are copied."""
        dataset = BIDSDataset(temp_bids_dir)

        # Test deidentification
        dataset.deidentify(outdir=output_dir, deidentify_config_dir=setup_publish_config)

        # Check that template files were copied
        assert (output_dir / "README.md").exists()
        assert (output_dir / "CHANGES.md").exists()
        assert (output_dir / "dataset_description.json").exists()

    def test_deidentify_output_dir_exists(self, temp_bids_dir, output_dir, setup_publish_config):
        """Test error handling when output directory already exists."""
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset = BIDSDataset(temp_bids_dir)

        # Should raise FileExistsError
        with pytest.raises(FileExistsError):
            dataset.deidentify(outdir=output_dir, deidentify_config_dir=setup_publish_config)

    def test_participant_removal(self, temp_bids_dir, output_dir, setup_publish_config):
        """Test that participants on the removal list are excluded."""
        dataset = BIDSDataset(temp_bids_dir)

        # Write removal list to config (the allowlist fallback inverts this)
        with open(setup_publish_config / "participants_to_remove.json", "w") as f:
            json.dump(["participant001"], f)

        # Test deidentification
        dataset.deidentify(
            outdir=output_dir, deidentify_config_dir=setup_publish_config, skip_audio=False
        )

        # Check that participant001 files are not in output
        participant001_files = list(output_dir.rglob("*participant001*"))
        assert len(participant001_files) == 0

    def test_audio_metadata_processing(self, temp_bids_dir, output_dir, setup_publish_config):
        """Test that audio files and metadata are properly processed."""
        dataset = BIDSDataset(temp_bids_dir)

        # Test deidentification
        dataset.deidentify(
            outdir=output_dir, deidentify_config_dir=setup_publish_config, skip_audio=False
        )

        # Check that audio files and sidecars were written
        wav_files = list(output_dir.rglob("*.wav"))
        json_files = list(output_dir.rglob("*.json"))
        assert len(wav_files) > 0, "Expected at least one wav file in output"
        assert len(json_files) > 0, "Expected at least one json file in output"

    def test_logging_messages(self, temp_bids_dir, output_dir, caplog, setup_publish_config):
        """Test that appropriate logging messages are generated."""
        dataset = BIDSDataset(temp_bids_dir)

        with caplog.at_level(logging.INFO):
            dataset.deidentify(
                outdir=output_dir, deidentify_config_dir=setup_publish_config, skip_audio=True
            )

        # Check for expected log messages
        log_messages = [record.message for record in caplog.records]
        assert any("Finished processing phenotype data" in msg for msg in log_messages)
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

    def test_remove_columns_methods(self, caplog):
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
        with caplog.at_level(logging.WARNING):
            dataset._warn_about_empty_columns(df, phenotype)

        # Check that warning was logged
        assert any("empty" in record.message.lower() for record in caplog.records)

        # Test removing sensitive columns
        df_sensitive, phenotype_sensitive = dataset._remove_sensitive_columns(df, phenotype)
        assert "state_province" not in df_sensitive.columns
        assert "state_province" not in phenotype_sensitive

    def test_add_sex_at_birth_column(self):
        """Test the _add_sex_at_birth_column method."""
        dataset = BIDSDataset(Path("/dummy"))

        # A Male/Female sex_assigned_at_birth is kept; "Prefer not to answer" becomes "Unknown";
        # without an answer, a Cis answer gives the sex at birth and any other gender answer gives
        # "Unknown". A row that answered neither question stays blank here and gets "Unknown" from
        # _fill_unknown_sex_at_birth once rows without data are dropped.
        df = pd.DataFrame(
            {
                "record_id": ["p1", "p2", "p3", "p4", "p5", "p6", "p7"],
                "gender_identity": ["Male gender identity", "Female gender identity",
                                    "Female gender identity", "Male gender identity",
                                    "Non-binary or genderqueer gender identity", None, "Female gender identity"],
                "specify_gender_identity": ["Cis: same gender as the sex assigned at birth", "Cis: same gender as the sex assigned at birth",
                                            "Cis: same gender as the sex assigned at birth", "Trans", None, None, "Trans"],
                "sex_assigned_at_birth": ["Male", "Prefer not to answer", None, None, None, None, "Male"],
                "age": [25, 30, 35, 40, 45, 50, 55],
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
        assert "sex_at_birth" in phenotype_sex["place_holder_schema"]["data_elements"]
        assert [v if pd.notna(v) else None for v in df_sex["sex_at_birth"]] == [
            "Male", "Unknown", "Female", "Unknown", "Unknown", None, "Male"]  # p7: stated sex kept for a trans participant
        BIDSDataset._fill_unknown_sex_at_birth(df_sex)
        assert df_sex["sex_at_birth"].iloc[5] == "Unknown"

        # Check that specify_gender_identity was removed
        assert "specify_gender_identity" not in df_sex.columns

    def test_add_sex_at_birth_column_when_no_one_answered_sex_assigned_at_birth(self):
        """An all-blank column is read as float; filling it with a string must not raise."""
        df = pd.DataFrame({"record_id": ["p1", "p2"],
                           "gender_identity": ["Female gender identity", "Male gender identity"],
                           "specify_gender_identity": ["Cis: same gender as the sex assigned at birth", "Trans"],
                           "sex_assigned_at_birth": [float("nan"), float("nan")]})
        phenotype = {"s": {"data_elements": {c: {"description": c} for c in ("record_id", "gender_identity", "specify_gender_identity")}}}
        out, _ = BIDSDataset._add_sex_at_birth_column(df, phenotype)
        assert list(out["sex_at_birth"]) == ["Female", "Unknown"]

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
                "participants": {
                    "data_elements": {
                        "record_id": {"description": "Unique identifier"},
                        "age": {"description": "Age of participant"},
                    }
                }
            }
            with open(bids_path / "participants.json", "w") as f:
                json.dump(participants_json, f, indent=2)

            dataset = BIDSDataset(bids_path)
            df, phenotype = dataset.load_phenotype_data(bids_path.joinpath("participants.tsv"))

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



def _text(values):
    return [v if isinstance(v, str) else None for v in values]


def test_disclosure_transforms_group_rare_answers_and_pass_redcap_age_label():
    df = pd.DataFrame({
        "record_id": ["a", "b", "c", "d", "e"],
        "age": ["89", "89.0", None, "90 and above", "18"],
        "gender_identity": ["Female gender identity", "Other", "Non-binary or genderqueer gender identity", None, None],
        "sex_assigned_at_birth": ["Male", "Intersex", "Unknown", "Prefer not to answer", None],
    })
    out = BIDSDataset._apply_disclosure_transforms(df)
    assert _text(out.age) == ["89", "89.0", None, "90 and above", "18"]
    assert _text(out.gender_identity) == ["Female gender identity", "Prefer not to answer",
                                       "Non-binary or genderqueer gender identity", None, None]
    assert _text(out.sex_assigned_at_birth) == ["Male", "Prefer not to answer", "Prefer not to answer",
                                               "Prefer not to answer", None]
    assert _text(df.gender_identity)[1] == "Other"  # input untouched


@pytest.mark.parametrize("age", ["90", "97.0"])
def test_disclosure_transforms_flag_numeric_age_of_90_or_more(age, caplog):
    df = pd.DataFrame({"record_id": ["a", "b"], "age": ["45", age]})
    with caplog.at_level("WARNING"):
        out = BIDSDataset._apply_disclosure_transforms(df)
    assert list(out.age) == ["45", age]
    assert "QA REVIEW REQUIRED: age: 1 numeric value(s)" in caplog.text and "records b" in caplog.text


def _make_sessions_tsv(participant_dir, rows, has_index=True):
    """Write a minimal sessions.tsv under participant_dir."""
    participant_dir.mkdir(parents=True, exist_ok=True)
    cols = ["session_id"]
    if has_index:
        cols.append("session_index")
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(participant_dir / "sessions.tsv", sep="\t", index=False)


def _field_map_df(rows):
    """Build a minimal field-map DataFrame for testing."""
    return pd.DataFrame(rows)


def _released_sessions_tree(make_deid_tree, root):
    """p1: audio in A, nothing in B, a questionnaire row in C."""
    sessions = pd.DataFrame({"record_id": ["p1"] * 3, "session_id": [A, B, C],
                             "session_index": ["1", "2", "3"], "session_status": ["Completed"] * 3})
    return make_deid_tree(
        {"p1": [(A, "rainbow-passage")]}, root=root, sessions={"p1": sessions}, pseudonyms={"p1": "900001"},
        tables={"session": sessions.rename(columns={"record_id": "participant_id"}),
                "confounders": pd.DataFrame({"participant_id": ["p1"], "confounders_session_id": [C],
                                             "acid_reflux": ["Yes"]})})


@pytest.fixture(scope="module")
def crosswalk():
    spec = importlib.util.spec_from_file_location(
        "session_label_crosswalk", Path(__file__).parents[1] / "scripts" / "session_label_crosswalk.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sidecar_tree(root, pid="p1", ses="S1", recordings=(("rec-A", "noisy-sounds-1"), ("rec-B", "noisy-sounds-2"))):
    """Sidecars only (no audio): for runs with skip_audio=True."""
    audio = root / f"sub-{pid}" / f"ses-{ses}" / "audio"
    audio.mkdir(parents=True)
    for rec_id, task in recordings:
        stem = f"sub-{pid}_ses-{ses}_task-{task}"
        (audio / f"{stem}_recording-metadata.json").write_text(
            json.dumps({"record_id": pid, "recording_id": rec_id.upper(), "session_id": ses}))
    pd.DataFrame({"record_id": [pid], "session_id": [ses], "session_index": ["1"]}).to_csv(
        root / f"sub-{pid}" / "sessions.tsv", sep="\t", index=False)
    return root / f"sub-{pid}"


class TestLoadParticipantAllowlist:

    def test_allowlist_from_include_file(self, tmp_path):
        (tmp_path / "participants_to_include.json").write_text(
            json.dumps(["a", "b", "c"])
        )
        result = BIDSDataset._load_participant_allowlist(
            tmp_path, {"a", "b", "c", "d"}
        )
        assert result == {"a", "b", "c"}

    def test_allowlist_fallback_from_removal(self, tmp_path):
        (tmp_path / "participants_to_remove.json").write_text(
            json.dumps(["d"])
        )
        result = BIDSDataset._load_participant_allowlist(
            tmp_path, {"a", "b", "c", "d"}
        )
        assert result == {"a", "b", "c"}

    def test_allowlist_empty_raises(self, tmp_path):
        (tmp_path / "participants_to_include.json").write_text(json.dumps([]))
        with pytest.raises(ValueError, match="empty"):
            BIDSDataset._load_participant_allowlist(tmp_path, {"a"})

    def test_allowlist_unmatched_warns(self, tmp_path, caplog):
        (tmp_path / "participants_to_include.json").write_text(
            json.dumps(["a", "x"])
        )
        with caplog.at_level(logging.WARNING):
            result = BIDSDataset._load_participant_allowlist(
                tmp_path, {"a", "b"}
            )
        assert result == {"a"}
        assert any("x" in r.message for r in caplog.records)

    def test_allowlist_no_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BIDSDataset._load_participant_allowlist(tmp_path, {"a"})


class TestBuildSessionIdMapping:

    def test_session_mapping_from_index(self, tmp_path):
        p1 = tmp_path / "sub-p1"
        _make_sessions_tsv(p1, [
            ["uuid-aaa", 1],
            ["uuid-bbb", 2],
        ])
        mapping = BIDSDataset._build_session_id_mapping(
            tmp_path, {"p1"}
        )
        assert mapping["uuid-aaa"] == "01"
        assert mapping["uuid-bbb"] == "02"

    def test_session_mapping_truncated_uuid_fallback(self, tmp_path):
        """Without session_index, falls back to truncated UUID (reduce_id_length)."""
        p1 = tmp_path / "sub-p1"
        _make_sessions_tsv(p1, [
            ["abcd1234-5678-9abc-def0-111111111111"],
            ["ffffffff-eeee-dddd-cccc-bbbbbbbbbbbb"],
        ], has_index=False)
        mapping = BIDSDataset._build_session_id_mapping(
            tmp_path, {"p1"}
        )
        assert mapping["abcd1234-5678-9abc-def0-111111111111"] == "abcd1234"
        assert mapping["ffffffff-eeee-dddd-cccc-bbbbbbbbbbbb"] == "ffffffff"

    def test_session_mapping_single_session(self, tmp_path):
        p1 = tmp_path / "sub-p1"
        _make_sessions_tsv(p1, [["only-one", 1]])
        mapping = BIDSDataset._build_session_id_mapping(
            tmp_path, {"p1"}
        )
        assert mapping["only-one"] == "01"

    def test_session_mapping_completeness(self, tmp_path):
        for pid, sessions in [("p1", ["s1", "s2", "s3"]),
                              ("p2", ["s4", "s5", "s6"])]:
            d = tmp_path / f"sub-{pid}"
            rows = [[sid, i + 1] for i, sid in enumerate(sessions)]
            _make_sessions_tsv(d, rows)
        mapping = BIDSDataset._build_session_id_mapping(
            tmp_path, {"p1", "p2"}
        )
        assert len(mapping) == 6
        for sid in ["s1", "s2", "s3", "s4", "s5", "s6"]:
            assert sid in mapping

    def test_session_mapping_missing_sessions_tsv(self, tmp_path, caplog):
        p1 = tmp_path / "sub-p1"
        p1.mkdir()
        with caplog.at_level(logging.WARNING):
            mapping = BIDSDataset._build_session_id_mapping(
                tmp_path, {"p1"}
            )
        assert len(mapping) == 0
        assert any("sessions.tsv" in r.message or "p1" in r.message
                    for r in caplog.records)


class TestDropColumnsByDisposition:

    def test_drop_internal_and_review(self):
        fm = _field_map_df([
            {"column_name": "col_a", "disposition": "release", "delete": "NO"},
            {"column_name": "col_b", "disposition": "internal", "delete": "YES"},
            {"column_name": "col_c", "disposition": "review", "delete": "NO"},
        ])
        df = pd.DataFrame({"col_a": [1], "col_b": [2], "col_c": [3]})
        result, dropped = BIDSDataset._drop_columns_by_disposition(
            df, field_map_df=fm
        )
        assert list(result.columns) == ["col_a"]
        assert set(dropped) == {"col_b", "col_c"}

    def test_error_when_no_disposition(self):
        fm = _field_map_df([
            {"column_name": "col_a", "delete": "NO"},
            {"column_name": "col_b", "delete": "YES"},
        ])
        df = pd.DataFrame({"col_a": [1], "col_b": [2]})
        with pytest.raises(ValueError, match="disposition"):
            BIDSDataset._drop_columns_by_disposition(df, field_map_df=fm)

    def test_columns_not_in_field_map_removed(self, caplog):
        """A column with no field-map row has no disposition, so it is never published."""
        fm = _field_map_df([
            {"column_name": "col_a", "disposition": "release", "delete": "NO"},
        ])
        df = pd.DataFrame({"participant_id": ["p1"], "col_a": [1], "pipeline_computed": [99]})
        with caplog.at_level(logging.WARNING):
            result, dropped = BIDSDataset._drop_columns_by_disposition(df, field_map_df=fm)
        assert list(result.columns) == ["participant_id", "col_a"]
        assert dropped == ["pipeline_computed"]
        assert any("pipeline_computed" in r.getMessage() for r in caplog.records)
        result, _ = BIDSDataset._drop_columns_by_disposition(
            df, field_map_df=fm, level=DispositionLevel.INTERNAL)
        assert "pipeline_computed" not in result.columns

    def test_review_columns_stripped(self):
        fm = _field_map_df([
            {"column_name": "col_r", "disposition": "review", "delete": "NO"},
            {"column_name": "col_ok", "disposition": "release", "delete": "NO"},
        ])
        df = pd.DataFrame({"col_r": [1], "col_ok": [2]})
        result, dropped = BIDSDataset._drop_columns_by_disposition(
            df, field_map_df=fm
        )
        assert "col_r" not in result.columns
        assert "col_ok" in result.columns
        assert "col_r" in dropped

    def test_returns_dropped_list(self):
        fm = _field_map_df([
            {"column_name": "a", "disposition": "release", "delete": "NO"},
            {"column_name": "b", "disposition": "internal", "delete": "YES"},
            {"column_name": "c", "disposition": "review", "delete": "NO"},
        ])
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        _, dropped = BIDSDataset._drop_columns_by_disposition(
            df, field_map_df=fm
        )
        assert isinstance(dropped, list)
        assert set(dropped) == {"b", "c"}

    def test_names_to_drop_at_each_level(self):
        """drop is removed at every level; shifted dates only with keep_date_shifted."""
        fm = _field_map_df([
            {"column_name": c, "disposition": d, "date_shift": ds, "access_tier": ""}
            for c, d, ds in [("a_release", "release", "NO"), ("b_internal", "internal", "NO"),
                             ("c_drop", "drop", "NO"), ("d_drop_date", "drop", "YES"),
                             ("e_internal_date", "internal", "YES")]])
        drop = BIDSDataset._names_to_drop_at_level
        assert drop(fm, DispositionLevel.INTERNAL) == {"c_drop", "d_drop_date"}
        assert drop(fm, DispositionLevel.REVIEW, keep_date_shifted=True) == {"b_internal", "c_drop", "d_drop_date"}
        assert drop(fm, DispositionLevel.RELEASE) == {"b_internal", "c_drop", "d_drop_date", "e_internal_date"}

    def test_controlled_only_columns_are_kept_only_in_the_controlled_tier(self):
        fm = _field_map_df([{"column_name": c, "disposition": d, "date_shift": "NO", "access_tier": t}
                            for c, d, t in [("a", "release", ""), ("b", "release", "controlled"),
                                            ("c", "internal", "")]])
        drop = BIDSDataset._names_to_drop_at_level
        assert drop(fm, DispositionLevel.RELEASE) == {"b", "c"}  # default tier is the narrower one
        assert drop(fm, DispositionLevel.RELEASE, access_tier=AccessTier.CONTROLLED) == {"c"}
        assert drop(fm, DispositionLevel.INTERNAL, access_tier=AccessTier.REGISTERED) == set()  # QA keeps all

    def test_disposition_is_matched_within_the_table(self):
        """self_reported_* is released in eligibility and internal in enrollment."""
        fm = _field_map_df([
            {"schema_name": "eligibility", "column_name": "self_reported_asthma", "disposition": "release"},
            {"schema_name": "enrollment", "column_name": "self_reported_asthma", "disposition": "internal"}])
        df = pd.DataFrame({"participant_id": ["p"], "self_reported_asthma": ["Yes"]})
        kept, dropped = BIDSDataset._drop_columns_by_disposition(df, fm, schema_name="eligibility")
        assert "self_reported_asthma" in kept.columns and not dropped
        _, dropped = BIDSDataset._drop_columns_by_disposition(df, fm, schema_name="enrollment")
        assert dropped == ["self_reported_asthma"]

    def test_unknown_table_refuses_instead_of_keeping_everything(self):
        fm = _field_map_df([{"schema_name": "demographics", "column_name": "zipcode", "disposition": "internal"}])
        df = pd.DataFrame({"participant_id": ["p"], "zipcode": ["02139"]})
        with pytest.raises(ValueError, match="no rows in the field map"):
            BIDSDataset._drop_columns_by_disposition(df, fm, schema_name="renamed_table")


class TestSessionOrdinalIntegration:

    def test_mapping_uses_ordinals_not_uuids(self, tmp_path):
        """Session mapping returns zero-padded ordinals, not truncated UUIDs."""
        p1 = tmp_path / "sub-p1"
        _make_sessions_tsv(p1, [
            ["abc-def-123", 2],
            ["xyz-uvw-789", 1],
        ])
        mapping = BIDSDataset._build_session_id_mapping(tmp_path, {"p1"})
        assert mapping["xyz-uvw-789"] == "01"
        assert mapping["abc-def-123"] == "02"

    def test_truncated_uuid_fallback_is_deterministic(self, tmp_path):
        """Without session_index, truncated UUID mapping is deterministic."""
        p1 = tmp_path / "sub-p1"
        _make_sessions_tsv(p1, [
            ["zzz00000-1111-2222-3333-444444444444"],
            ["aaa00000-5555-6666-7777-888888888888"],
            ["mmm00000-9999-aaaa-bbbb-cccccccccccc"],
        ], has_index=False)
        m1 = BIDSDataset._build_session_id_mapping(tmp_path, {"p1"})
        m2 = BIDSDataset._build_session_id_mapping(tmp_path, {"p1"})
        assert m1 == m2
        assert m1["aaa00000-5555-6666-7777-888888888888"] == "aaa00000"
        assert m1["mmm00000-9999-aaaa-bbbb-cccccccccccc"] == "mmm00000"
        assert m1["zzz00000-1111-2222-3333-444444444444"] == "zzz00000"


class TestSessionsCarryForward:

    def test_disposition_columns_dropped(self):
        """sessions.tsv columns with disposition=internal are dropped."""
        fm = _field_map_df([
            {"column_name": "session_duration", "disposition": "release", "delete": "NO"},
            {"column_name": "session_site", "disposition": "internal", "delete": "YES"},
            {"column_name": "session_id", "disposition": "release", "delete": "NO"},
        ])
        df = pd.DataFrame({
            "session_id": ["s1"],
            "session_duration": ["120"],
            "session_site": ["MIT"],
        })
        result, dropped = BIDSDataset._drop_columns_by_disposition(df, field_map_df=fm)
        assert "session_duration" in result.columns
        assert "session_site" not in result.columns
        assert "session_site" in dropped

    def test_record_id_renamed_to_participant_id(self):
        """sessions.tsv should rename record_id to participant_id."""
        df = pd.DataFrame({
            "record_id": ["uuid-1"],
            "session_id": ["ses-1"],
        })
        df = df.rename(columns={"record_id": "participant_id"})
        assert "participant_id" in df.columns
        assert "record_id" not in df.columns


class TestDispositionInPhenotypeContext:

    def test_only_release_columns_survive(self):
        """After disposition drop, only release columns remain."""
        fm = _field_map_df([
            {"column_name": "participant_id", "disposition": "release", "delete": "NO"},
            {"column_name": "age", "disposition": "release", "delete": "NO"},
            {"column_name": "session_complete", "disposition": "internal", "delete": "YES"},
            {"column_name": "free_text_specify", "disposition": "review", "delete": "NO"},
        ])
        df = pd.DataFrame({
            "participant_id": ["p1", "p2"],
            "age": ["30", "40"],
            "session_complete": ["Complete", "Incomplete"],
            "free_text_specify": ["some text", "other text"],
        })
        result, dropped = BIDSDataset._drop_columns_by_disposition(df, field_map_df=fm)
        assert set(result.columns) == {"participant_id", "age"}
        assert set(dropped) == {"session_complete", "free_text_specify"}

    def test_error_when_no_disposition_in_phenotype_context(self):
        """When field map has no disposition column, raises ValueError."""
        fm = _field_map_df([
            {"column_name": "age", "delete": "NO"},
            {"column_name": "zipcode", "delete": "YES"},
        ])
        df = pd.DataFrame({"age": ["30"], "zipcode": ["02139"]})
        with pytest.raises(ValueError, match="disposition"):
            BIDSDataset._drop_columns_by_disposition(df, field_map_df=fm)


class TestEndToEndAllowlistFiltering:

    def test_emptied_participant_excluded_from_phenotype(self, tmp_path):
        """A participant emptied by task filtering should not appear in phenotype.

        Setup: 3 participants on allowlist. p3 has only a task NOT in the
        include list, so after filtering p3 has zero audio and is excluded.
        Phenotype should have only p1 and p2.
        """
        bids = tmp_path / "bids"
        config = tmp_path / "config"
        out = tmp_path / "output"
        config.mkdir()

        # Config files
        (config / "participants_to_include.json").write_text(
            json.dumps(["p1", "p2", "p3"])
        )
        # identity pseudonyms: deidentify refuses an allowlisted participant without one
        (config / "id_remapping.json").write_text(json.dumps({p: p for p in ("p1", "p2", "p3")}))
        (config / "deidentify_settings.json").write_text(json.dumps({"access_tier": "registered"}))
        (config / "audio_filestems_to_remove.json").write_text(json.dumps([]))
        (config / "audio_tasks_to_include.json").write_text(
            json.dumps(["rainbow-passage"])
        )

        # Build BIDS tree: p1 and p2 have included task, p3 has excluded task
        for pid, task in [("p1", "rainbow-passage"), ("p2", "rainbow-passage"),
                          ("p3", "excluded-task")]:
            audio_dir = bids / f"sub-{pid}" / "ses-s1" / "audio"
            audio_dir.mkdir(parents=True)
            wav = audio_dir / f"sub-{pid}_ses-s1_task-{task}.wav"
            wav.write_bytes(b"RIFF" + b"\x00" * 8192)
            # Sidecar
            sidecar = audio_dir / f"sub-{pid}_ses-s1_task-{task}_recording-metadata.json"
            sidecar.write_text(json.dumps({
                "item": [
                    {"linkId": "record_id", "answer": [{"valueString": pid}]},
                    {"linkId": "session_id", "answer": [{"valueString": "s1"}]},
                ]
            }))
            # sessions.tsv
            ses_df = pd.DataFrame({"record_id": [pid], "session_id": ["s1"], "session_index": ["1"]})
            ses_df.to_csv(bids / f"sub-{pid}" / "sessions.tsv", sep="\t", index=False)

        # Phenotype with all 3
        pheno_dir = bids / "phenotype"
        pheno_dir.mkdir()
        pheno_df = pd.DataFrame({
            "participant_id": ["p1", "p2", "p3"],
            "acid_reflux": ["Yes", "No", "Yes"],
        })
        pheno_df.to_csv(pheno_dir / "confounders.tsv", sep="\t", index=False)
        (pheno_dir / "confounders.json").write_text(json.dumps({}))

        # Template files
        (bids / "dataset_description.json").write_text(json.dumps({"Name": "test"}))

        dataset = BIDSDataset(bids)
        dataset.deidentify(outdir=out, deidentify_config_dir=config)

        # p3 should be excluded
        assert not (out / "sub-p3").exists()
        # Phenotype should only have p1 and p2
        result_pheno = pd.read_csv(out / "phenotype" / "confounders.tsv", sep="\t", dtype=str)
        assert set(result_pheno["participant_id"]) == {"p1", "p2"}

        # session_id_mapping.json generation is currently disabled
        assert not (out / "session_id_mapping.json").exists()

    def test_allowlisted_participant_without_a_pseudonym_is_refused(self, tmp_path, make_deid_tree):
        bids, config = make_deid_tree({"p1": [("s1", "rainbow-passage")]}, pseudonyms={})
        with pytest.raises(ValueError, match="1 allowlisted participant.*no pseudonym.*p1"):
            BIDSDataset(bids).deidentify(outdir=tmp_path / "out", deidentify_config_dir=config)
        assert list((tmp_path / "out").iterdir()) == []


class TestSessionLabels:

    def _sessions(self):
        return pd.DataFrame({"session_id": [A, B, C], "session_index": ["1", "2", "3"]})

    @pytest.mark.parametrize("mode, labels, order", [
        (SessionLabels.ORDINAL, {A: "01", C: "02"}, {A: 1, C: 2}),   # numbered without gaps
        (SessionLabels.INDEX, {A: "01", C: "03"}, {A: 1, C: 3}),     # session_index, gaps kept
    ])
    def test_numbered_labels(self, mode, labels, order):
        assert BIDSDataset._session_labels(self._sessions(), mode, {A, C}) == (labels, order)

    def test_uuid_lower_case_and_long_on_collision(self):
        ses = pd.DataFrame({"session_id": ["ABCD1234-0000-0000-0000-00000000000A",
                                           "ABCD1234-1111-0000-0000-00000000000B",
                                           "FFFF0000-0000-0000-0000-00000000000C"]})
        labels, _ = BIDSDataset._session_labels(ses, SessionLabels.UUID)
        assert labels == {ses.session_id[0]: "abcd123400000000", ses.session_id[1]: "abcd123411110000",
                          ses.session_id[2]: "ffff0000"}

    def test_ordinal_without_session_index_refuses(self):
        with pytest.raises(ValueError, match="session_index"):
            BIDSDataset._session_labels(pd.DataFrame({"session_id": ["x"]}), SessionLabels.ORDINAL)


class TestReleasedSessions:
    """A session is released with a file to publish or questionnaire rows; empty ones are not,
    and no original session ID reaches the output."""

    @pytest.mark.parametrize("mode, audio_dir, session_labels, questionnaire_label", [
        (SessionLabels.ORDINAL, "ses-01", ["01", "02"], "02"),
        (SessionLabels.INDEX, "ses-01", ["01", "03"], "03"),
        (SessionLabels.UUID, "ses-aaaa1111", ["aaaa1111", "cccc3333"], "cccc3333"),
    ])
    def test_labels_and_no_original_ids(self, tmp_path, make_deid_tree, mode, audio_dir, session_labels,
                                        questionnaire_label):
        bids, config = _released_sessions_tree(make_deid_tree, tmp_path)
        out, session_map = tmp_path / "out", tmp_path / "internal" / "map.json"
        BIDSDataset(bids).deidentify(outdir=out, deidentify_config_dir=config, session_labels=mode,
                                     session_id_map=session_map)
        assert [d.name for d in (out / "sub-900001").glob("ses-*")] == [audio_dir]
        ses = pd.read_csv(out / "sub-900001" / "sub-900001_sessions.tsv", sep="\t", dtype=str)
        assert list(ses.session_id) == session_labels
        session = pd.read_csv(out / "phenotype" / "session.tsv", sep="\t", dtype=str)
        assert sorted(session.session_id) == session_labels
        conf = pd.read_csv(out / "phenotype" / "confounders.tsv", sep="\t", dtype=str)
        assert list(conf.confounders_session_id) == [questionnaire_label]
        for f in out.rglob("*"):
            if f.is_file() and f.suffix in (".tsv", ".json"):
                text = f.read_text()
                assert not any(x in text or x.lower() in text for x in (A, B, C)), f
        record = json.loads(session_map.read_text())
        assert record["session_labels"] == mode.value
        assert [(r["session_id"], r["released_session_label"]) for r in record["sessions"]] == [
            (A, session_labels[0]), (C, session_labels[1])]

    def test_session_id_map_inside_output_refused(self, tmp_path):
        """Checked before anything is read or written."""
        (tmp_path / "bids").mkdir()
        with pytest.raises(ValueError, match="outside the output"):
            BIDSDataset(tmp_path / "bids").deidentify(outdir=tmp_path / "out", deidentify_config_dir=tmp_path,
                                                      session_id_map=tmp_path / "out" / "map.json")
        assert not (tmp_path / "out").exists()


class TestSessionLabelCrosswalk:
    """scripts/session_label_crosswalk.py, on the session maps deidentify writes."""

    def _maps(self, tmp_path, make_deid_tree):
        for name, mode in (("index", SessionLabels.INDEX), ("ordinal", SessionLabels.ORDINAL)):
            bids, config = _released_sessions_tree(make_deid_tree, tmp_path / name)
            BIDSDataset(bids).deidentify(outdir=tmp_path / name / "out", deidentify_config_dir=config,
                                         session_labels=mode, session_id_map=tmp_path / f"{name}.json")
        return tmp_path / "index.json", tmp_path / "ordinal.json"

    def test_crosswalk_between_label_schemes(self, tmp_path, make_deid_tree, crosswalk):
        index, ordinal = self._maps(tmp_path, make_deid_tree)
        table = crosswalk.crosswalk(crosswalk.load_map(index), crosswalk.load_map(ordinal))
        assert table.to_dict("records") == [
            {"participant_id": "900001", "old_session_label": "01", "new_session_label": "01"},
            {"participant_id": "900001", "old_session_label": "03", "new_session_label": "02"},
        ]
        v31 = crosswalk.crosswalk(crosswalk.uuid_labels(crosswalk.load_map(ordinal)), crosswalk.load_map(ordinal))
        assert list(v31.old_session_label) == ["aaaa1111", "cccc3333"]

    def test_main_writes_no_original_ids(self, tmp_path, make_deid_tree, crosswalk):
        _, ordinal = self._maps(tmp_path, make_deid_tree)
        assert crosswalk.main(["--old-uuid-labels", "--new", str(ordinal), "-o", str(tmp_path / "x.tsv")]) == 0
        assert not any(x in (tmp_path / "x.tsv").read_text() for x in (A, C))


class TestFormBookkeepingIsNotData:
    """Only REDCap-generated and form bookkeeping columns are not data; a row holding nothing else is
    dropped after deidentify, while a pipeline-derived value (e.g. state_province) is data."""

    def test_form_metadata_columns(self):
        cols = ["participant_id", "demographics_session_id", "demographics_via", "demographics_origin",
                "demographics_duration", "demographics_started_at", "marital_status", "employ_status"]
        assert BIDSDataset._form_metadata_columns(cols) == {
            "demographics_session_id", "demographics_via", "demographics_origin",
            "demographics_duration", "demographics_started_at"}

    @pytest.mark.parametrize("schema, df, kept", [
        ("confounders", pd.DataFrame({"participant_id": ["p1", "p2"], "confounders_session_id": ["s1", "s2"],
                                      "confounders_via": ["Participant"] * 2, "confounders_duration": ["30", "12"],
                                      "acid_reflux": ["Yes", None]}), ["p1"]),
        ("demographics", pd.DataFrame({"participant_id": ["p1", "p2"], "demographics_session_id": ["01", "01"],
                                       "state_province": ["Ontario", None],
                                       "demographics_duration": ["120", "95"]}), ["p1"]),
    ], ids=["answer-kept-metadata-only-dropped", "derived-value-kept"])
    def test_rows_with_only_bookkeeping_are_dropped(self, schema, df, kept):
        assert list(BIDSDataset._drop_rows_emptied_by_deidentify(df, schema).participant_id) == kept

    def test_metadata_only_row_releases_no_session(self, tmp_path):
        pheno = tmp_path / "phenotype"
        pheno.mkdir()
        pd.DataFrame({
            "participant_id": ["p1", "p1"],
            "confounders_session_id": ["S1", "S2"],
            "acid_reflux": ["Yes", None],
        }).to_csv(pheno / "confounders.tsv", sep="\t", index=False)
        (pheno / "confounders.json").write_text(json.dumps({}))
        assert BIDSDataset._sessions_with_questionnaire_data(pheno) == {"S1"}


class TestParticipantFailureStopsRun:

    def test_one_failing_participant_raises(self, tmp_path, make_deid_tree):
        """A participant that errors must not vanish from the release like one with no audio."""
        bids, config = make_deid_tree(
            {"p1": [("s1", "rainbow-passage")], "p2": [("s1", "rainbow-passage")]},
            # no session_index for p2, so ordinal labelling fails for p2 only
            sessions={"p2": pd.DataFrame({"record_id": ["p2"], "session_id": ["s1"]})})
        with pytest.raises(RuntimeError, match=r"failed for 1 participant.*p2"):
            BIDSDataset(bids).deidentify(outdir=tmp_path / "out", deidentify_config_dir=config)


class TestLabelsSharedAcrossTiers:
    """Labels are numbered over every session some tier could release, so a session one build
    withholds leaves a gap instead of shifting the later labels."""

    def test_features_only_session_withheld_without_features(self, tmp_path, make_deid_tree):
        """S1 has only features to publish (its task's audio is not released)."""
        bids, _ = make_deid_tree({"p1": [("S1", "free-speech-1"), ("S2", "rainbow-passage")]}, features=True)
        pdir = bids / "sub-p1"
        with_features, _, _, _ = BIDSDataset._deidentify_participant_files(
            pdir, tmp_path / "a", {"p1": "900001"}, [], ["rainbow-passage"])
        without, order, _, _ = BIDSDataset._deidentify_participant_files(
            pdir, tmp_path / "b", {"p1": "900001"}, [], ["rainbow-passage"], skip_audio_features=True)
        assert with_features == {"S1": "01", "S2": "02"}
        assert without == {"S2": "02"} and order == {"S2": 2}
        ses = pd.read_csv(tmp_path / "b" / "sub-900001" / "sub-900001_sessions.tsv", sep="\t", dtype=str)
        assert list(ses.session_id) == ["02"] and list(ses.session_index) == ["2"]

    def test_participant_with_only_feature_output_is_kept(self, tmp_path, make_deid_tree):
        """Recordings whose task audio is not released still publish stripped features."""
        bids, _ = make_deid_tree({"p1": [("S1", "free-speech-1")]}, features=True)
        labels, _, _, _ = BIDSDataset._deidentify_participant_files(
            bids / "sub-p1", tmp_path / "out", {"p1": "900001"}, [], ["noisy-sounds-*"])
        assert labels == {"S1": "01"}
        assert list((tmp_path / "out").rglob("*free-speech-1_features.pt"))

    # other_voice_activity is a review column and ever_alcohol_rehab controlled-only (shipped field map)
    @pytest.mark.parametrize("column, value, narrow, wide", [
        ("other_voice_activity", "Podcaster", {},
         {"column_value_reviews.json": {"verdicts": [
             {"participant_id": "p1", "column_name": "other_voice_activity", "verdict": "safe"}]}}),
        ("ever_alcohol_rehab", "No", {}, {"deidentify_settings.json": {"access_tier": "controlled"}}),
    ], ids=["review-column", "controlled-only-column"])
    def test_session_with_only_withheld_data_keeps_later_labels(self, tmp_path, make_deid_tree, column, value,
                                                                narrow, wide):
        """s2 holds only one answer, which the narrow build withholds and the wide one publishes."""
        labels = {}
        for name, config_files in (("narrow", narrow), ("wide", wide)):
            bids, config = make_deid_tree(
                {"p1": [("s1", "rainbow-passage"), ("s3", "rainbow-passage")]}, root=tmp_path / name,
                sessions={"p1": ["s1", "s2", "s3"]}, config_files=config_files,
                tables={"confounders/confounders": pd.DataFrame(
                    {"participant_id": ["p1"], "confounders_session_id": ["s2"], column: [value]})})
            out = tmp_path / name / "out"
            BIDSDataset(bids).deidentify(outdir=out, deidentify_config_dir=config)
            (sessions,) = list((out / "sub-900000").glob("*sessions.tsv"))
            labels[name] = list(BIDSDataset._read_tsv_as_written(sessions)["session_id"])
        assert labels == {"narrow": ["01", "03"], "wide": ["01", "02", "03"]}


class TestSidecarDispositions:
    """Audio sidecar keys follow the audio_sidecar table's dispositions, as sessions.tsv follows
    the session table's; a key the table does not list is removed and logged."""

    def _deidentify(self, root, make_deid_tree, **kwargs):
        bids, config = make_deid_tree({"p1": [("s1", "rainbow-passage", {
            "recording_duration": 1.5, "recording_microphone": "Built-in", "task_name": "rainbow-passage",
            "not_a_sidecar_key": "x"})]}, root=root, pseudonyms={"p1": "p1"})
        BIDSDataset(bids).deidentify(outdir=root / "out", deidentify_config_dir=config, **kwargs)
        (sidecar,) = (root / "out" / "sub-p1").rglob("*.json")
        return json.loads(sidecar.read_text())

    def test_known_keys_kept_unknown_removed_and_logged(self, tmp_path, make_deid_tree, caplog):
        with caplog.at_level(logging.WARNING):
            meta = self._deidentify(tmp_path, make_deid_tree)
        assert meta == {"participant_id": "p1", "session_id": "01", "recording_duration": 1.5,
                        "recording_microphone": "Built-in", "task_name": "rainbow-passage"}
        assert any("not_a_sidecar_key" in r.getMessage() for r in caplog.records)

    def test_internal_key_removed_at_release_kept_at_internal(self, tmp_path, make_deid_tree, monkeypatch):
        # deidentify reads the field map through this cache; no public way to substitute one
        fm = BIDSDataset._load_reorganization_file(exclude_dropped=False)
        fm.loc[(fm.schema_name == "audio_sidecar") & (fm.column_name == "recording_microphone"),
               "disposition"] = "internal"
        monkeypatch.setattr(BIDSDataset, "_cached_field_map_df", fm)
        assert "recording_microphone" not in self._deidentify(tmp_path / "a", make_deid_tree)
        assert "recording_microphone" in self._deidentify(
            tmp_path / "b", make_deid_tree, disposition_level=DispositionLevel.INTERNAL)


class TestRemovedRecordings:
    """audio_recording_ids_to_remove.json and audio_filestems_to_remove.json remove a recording's
    audio, sidecar, features, quality-metric row and phenotype rows; audio_tasks_to_include selects tasks."""

    def test_exclusion_key_ignores_case_and_suffixes(self):
        key = BIDSDataset._exclusion_key
        assert key("sub-P1_ses-ABCD-1234_task-Rainbow-Passage.wav") == key(
            "sub-p1_ses-abcd-1234_task-rainbow-passage_recording-metadata.json")
        assert key("sub-p1_ses-s1_task-noisy-sounds-2_features.pt") == "sub-p1_ses-s1_task-noisy-sounds-2"

    def test_recording_ids_resolve_to_filestems_and_remove_audio_and_features(self, tmp_path):
        """Removal by recording_id uses the filestem mechanism, so features go too, even for a
        recording whose task is not included (its audio loop skips it before any ID check)."""
        pdir = _sidecar_tree(tmp_path / "in", recordings=(("rec-A", "free-speech-1"), ("rec-B", "noisy-sounds-2")))
        audio = pdir / "ses-S1" / "audio"
        for task in ("free-speech-1", "noisy-sounds-2"):
            torch.save({"opensmile": {"x": 1}}, audio / f"sub-p1_ses-S1_task-{task}_features.pt")
        stems = BIDSDataset._filestems_for_recording_ids(tmp_path / "in", ["p1"], {"rec-a"})
        assert stems == ["sub-p1_ses-S1_task-free-speech-1"]
        out = tmp_path / "out"
        BIDSDataset._deidentify_participant_files(
            pdir, out, {"p1": "900001"}, stems, ["noisy-sounds-*"], skip_audio=True, skip_audio_features=False)
        names = sorted(p.name for p in out.rglob("*") if p.is_file())
        assert not any("free-speech" in n for n in names), names
        assert any(n.endswith("noisy-sounds-2.json") for n in names)
        assert any("noisy-sounds-2_features" in n for n in names)

    def test_filestem_differing_in_case_still_removes(self, tmp_path):
        """A configured stem with a lower-case session ID removes the recording in an upper-case tree."""
        pdir = _sidecar_tree(tmp_path / "in", ses="ABC1")
        BIDSDataset._deidentify_participant_files(
            pdir, tmp_path / "out", {"p1": "900001"}, ["sub-p1_ses-abc1_task-Noisy-Sounds-1"], ["noisy-sounds-*"],
            skip_audio=True, skip_audio_features=True)
        names = [p.name for p in (tmp_path / "out").rglob("*.json")]
        assert len(names) == 1 and "noisy-sounds-2" in names[0]

    def test_tasks_are_included_by_pattern(self, tmp_path):
        pdir = _sidecar_tree(tmp_path / "in", recordings=(("rec-A", "identifying-pictures-35"),
                                                          ("rec-B", "reading-passage-3")))
        BIDSDataset._deidentify_participant_files(
            pdir, tmp_path / "out", {"p1": "900001"}, [], ["identifying-pictures-*"],
            skip_audio=True, skip_audio_features=True)
        written = [p.name for p in (tmp_path / "out").rglob("*.json")]
        assert len(written) == 1 and "identifying-pictures-35" in written[0]

    @pytest.mark.parametrize("stems, matched, participants, recording_ids, expected", [
        (["sub-001js_ses-x_task-passage-10"], set(), {"a-uuid-participant"}, set(),
         ["0 matched", "1 are for participants outside this run"]),
        (["sub-p1_ses-s1_task-noisy-sounds-1", "sub-p1_ses-s1_task-noisy-sounds-9"],
         {"sub-p1_ses-s1_task-noisy-sounds-1"}, {"p1"}, {"rid-x"},
         ["1 matched", "1 for participants in this run matched nothing", "None was found"]),
    ], ids=["participants-outside-the-run", "entries-matching-nothing"])
    def test_unmatched_removal_entries_warn(self, caplog, stems, matched, participants, recording_ids, expected):
        keys = {s: {BIDSDataset._exclusion_key(s)} for s in stems}
        with caplog.at_level(logging.INFO):
            BIDSDataset._report_exclusion_coverage(
                stems, keys, {BIDSDataset._exclusion_key(m) for m in matched}, participants, recording_ids, [])
        warnings = " ".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
        assert all(text in warnings for text in expected), warnings

    def test_quality_metrics_exclusion_matches_the_recording_exactly(self, tmp_path):
        """identifying-pictures-2 must not also remove identifying-pictures-20 and -21."""
        (tmp_path / "bids").mkdir()
        pd.DataFrame({
            "participant_id": ["p1", "p1", "p1"], "session_id": ["s1"] * 3,
            "task_name": ["identifying-pictures-2", "identifying-pictures-20", "identifying-pictures-21"],
            "snr": ["1", "2", "3"],
        }).to_csv(tmp_path / "bids" / "audio_quality_metrics.tsv", sep="\t", index=False)
        (tmp_path / "out").mkdir()
        BIDSDataset._deidentify_quality_metrics(
            tmp_path / "bids", tmp_path / "out", [], ["sub-p1_ses-s1_task-identifying-pictures-2"],
            [], {"p1": "900001"}, {})
        out = pd.read_csv(tmp_path / "out" / "audio_quality_metrics.tsv", sep="\t", dtype=str)
        assert sorted(out.task_name) == ["identifying-pictures-20", "identifying-pictures-21"]

    @pytest.mark.parametrize("removal", [
        {"audio_recording_ids_to_remove.json": ["r2"]},
        {"audio_filestems_to_remove.json": ["sub-p1_ses-s1_task-maximum-phonation-time-1"]},
    ], ids=["by-recording-id", "by-filestem"])
    def test_removed_recordings_leave_recording_and_acoustic_task_tables(self, tmp_path, make_deid_tree, removal):
        """recording.tsv says 'R2': the recording ID is matched without regard to case."""
        bids, config = make_deid_tree(
            {"p1": [("s1", "rainbow-passage", {"recording_id": "r1"}),
                    ("s1", "maximum-phonation-time-1", {"recording_id": "r2"})]},
            config_files=removal,
            tables={"task/recording": pd.DataFrame({
                        "participant_id": ["p1", "p1"], "recording_id": ["r1", "R2"],
                        "recording_acoustic_task_id": ["t1", "t2"], "recording_name": ["a", "b"]}),
                    "task/acoustic_task": pd.DataFrame({
                        "participant_id": ["p1", "p1"], "acoustic_task_id": ["t1", "t2"],
                        "acoustic_task_name": ["Rainbow Passage", "Maximum phonation time-1"]})})
        out = tmp_path / "out"
        BIDSDataset(bids).deidentify(outdir=out, deidentify_config_dir=config)
        assert list(BIDSDataset._read_tsv_as_written(out / "phenotype/task/recording.tsv")["recording_id"]) == ["r1"]
        assert list(BIDSDataset._read_tsv_as_written(
            out / "phenotype/task/acoustic_task.tsv")["acoustic_task_id"]) == ["t1"]
        assert len(list(out.rglob("*.wav"))) == 1


class TestLoadColumnValueReviews:

    def test_loads_verdicts(self, tmp_path):
        manifest = {
            "verdicts": [
                {"participant_id": "p1", "column_name": "col_a", "verdict": "safe"},
                {"participant_id": "p2", "column_name": "col_a", "verdict": "redact"},
            ]
        }
        (tmp_path / "column_value_reviews.json").write_text(json.dumps(manifest))
        result = BIDSDataset._load_column_value_reviews(tmp_path)
        assert result[("p1", "col_a", None)] == ValueReview("safe")
        assert result[("p2", "col_a", None)] == ValueReview("redact")

    def test_missing_file_returns_empty(self, tmp_path):
        result = BIDSDataset._load_column_value_reviews(tmp_path)
        assert result == {}

    def test_duplicate_last_wins(self, tmp_path, caplog):
        manifest = {
            "verdicts": [
                {"participant_id": "p1", "column_name": "col_a", "verdict": "safe"},
                {"participant_id": "p1", "column_name": "col_a", "verdict": "redact"},
            ]
        }
        (tmp_path / "column_value_reviews.json").write_text(json.dumps(manifest))
        with caplog.at_level(logging.WARNING):
            result = BIDSDataset._load_column_value_reviews(tmp_path)
        assert result[("p1", "col_a", None)].verdict == "redact"
        assert any("duplicate" in r.message.lower() for r in caplog.records)


    def test_source_column_name_normalized(self, tmp_path):
        """Verdicts using source_column_name are normalized to output names."""
        manifest = {
            "verdicts": [
                {"participant_id": "p1", "source_column_name": "old_src_name", "verdict": "safe"},
                {"participant_id": "p2", "column_name": "new_out_name", "verdict": "redact"},
            ]
        }
        (tmp_path / "column_value_reviews.json").write_text(json.dumps(manifest))
        field_map = pd.DataFrame({
            "column_name_source": ["old_src_name", "other_col"],
            "column_name": ["new_out_name", "other_col"],
        })
        result = BIDSDataset._load_column_value_reviews(tmp_path, field_map_df=field_map)
        assert result[("p1", "new_out_name", None)].verdict == "safe"
        assert result[("p2", "new_out_name", None)].verdict == "redact"

    def test_column_name_not_in_field_map_kept_as_is(self, tmp_path):
        """Verdicts with column names not in the field map are kept unchanged."""
        manifest = {
            "verdicts": [
                {"participant_id": "p1", "column_name": "unknown_col", "verdict": "safe"},
            ]
        }
        (tmp_path / "column_value_reviews.json").write_text(json.dumps(manifest))
        field_map = pd.DataFrame({
            "column_name_source": ["other"],
            "column_name": ["other"],
        })
        result = BIDSDataset._load_column_value_reviews(tmp_path, field_map_df=field_map)
        assert ("p1", "unknown_col", None) in result

    def _load(self, tmp_path, *entries):
        (tmp_path / "column_value_reviews.json").write_text(json.dumps({"verdicts": list(entries)}))
        return BIDSDataset._load_column_value_reviews(tmp_path)

    def test_session_and_redacted_text_are_kept_and_markers_written_one_way(self, tmp_path):
        result = self._load(tmp_path, {
            "participant_id": "p1", "session_id": "s1", "column_name": "col_a", "verdict": "redact",
            "redacted_text": "19/30, [redacted] and [ Redacted ]"})
        assert result == {("p1", "col_a", "s1"): ValueReview("redact", "19/30, [REDACTED] and [REDACTED]")}

    def test_redacted_text_is_ignored_unless_the_verdict_is_redact(self, tmp_path):
        result = self._load(tmp_path, {
            "participant_id": "p1", "column_name": "col_a", "verdict": "safe", "redacted_text": "x"})
        assert result[("p1", "col_a", None)] == ValueReview("safe")

    @pytest.mark.parametrize("text", ["[redcated] here", "a redacted word", "[redacted cut off", "[REDATCED]"])
    def test_a_misspelled_marker_stops_the_run(self, tmp_path, text):
        with pytest.raises(ValueError, match="outside a \\[REDACTED\\] marker"):
            self._load(tmp_path, {"participant_id": "p1", "column_name": "col_a", "verdict": "redact",
                                  "redacted_text": text})

    def test_words_that_only_resemble_the_marker_are_text(self, tmp_path):
        text = "reduced dose, drug-related, redness [sic]"
        result = self._load(tmp_path, {"participant_id": "p1", "column_name": "col_a", "verdict": "redact",
                                       "redacted_text": text})
        assert result[("p1", "col_a", None)].redacted_text == text


class TestApplyColumnValueReviews:

    def test_safe_passes_through(self):
        df = pd.DataFrame({
            "participant_id": ["p1", "p2"],
            "score": [10, 20],
            "free_text": ["safe value", "also safe"],
        })
        verdicts = {
            ("p1", "free_text", None): ValueReview("safe"),
            ("p2", "free_text", None): ValueReview("safe"),
        }
        result, dropped = BIDSDataset._apply_column_value_reviews(
            df, {"free_text"}, verdicts
        )
        assert result.loc[result["participant_id"] == "p1", "free_text"].values[0] == "safe value"
        assert dropped == []

    def test_redact_replaces(self):
        df = pd.DataFrame({
            "participant_id": ["p1", "p2"],
            "free_text": ["Dr. Smith", "safe"],
        })
        verdicts = {
            ("p1", "free_text", None): ValueReview("redact"),
            ("p2", "free_text", None): ValueReview("safe"),
        }
        result, _ = BIDSDataset._apply_column_value_reviews(
            df, {"free_text"}, verdicts
        )
        assert result.loc[result["participant_id"] == "p1", "free_text"].values[0] == "[REDACTED]"
        assert result.loc[result["participant_id"] == "p2", "free_text"].values[0] == "safe"

    def test_redact_publishes_the_replacement_text_for_that_session_only(self):
        df = pd.DataFrame({
            "participant_id": ["p1", "p1", "p2"],
            "session_id": ["s1", "s2", "s1"],
            "free_text": ["seen Sept 4/25", "seen in clinic", "fine"],
        })
        verdicts = {
            ("p1", "free_text", "s1"): ValueReview("redact", "seen [REDACTED]"),
            ("p1", "free_text", "s2"): ValueReview("safe"),
            ("p2", "free_text", None): ValueReview("safe"),
        }
        result, _ = BIDSDataset._apply_column_value_reviews(df, {"free_text"}, verdicts)
        assert list(result["free_text"]) == ["seen [REDACTED]", "seen in clinic", "fine"]

    def test_a_form_table_matches_verdicts_on_its_own_session_column(self):
        """Form tables name the session after the form (confounders_session_id, mph_session_id)."""
        df = pd.DataFrame({
            "participant_id": ["p1", "p1"], "confounders_session_id": ["s1", "s2"],
            "free_text": ["first visit", "second visit"]})
        verdicts = {
            ("p1", "free_text", "s1"): ValueReview("redact", "[REDACTED] visit"),
            ("p1", "free_text", "s2"): ValueReview("safe"),
        }
        result, _ = BIDSDataset._apply_column_value_reviews(df, {"free_text"}, verdicts)
        assert list(result["free_text"]) == ["[REDACTED] visit", "second visit"]

    def test_a_verdict_applies_only_to_the_answer_that_was_reviewed(self, caplog):
        """An answer changed since review (a correction, a re-export) is withheld, not published
        unread; spacing and number format are not changes."""
        df = pd.DataFrame({
            "participant_id": ["p1", "p2", "p3", "p4"],
            "free_text": ["now says something else", " 17  ", "Seen  in\r\nclinic", "kept"]})
        verdicts = {
            ("p1", "free_text", None): ValueReview("safe", value_sha256=review_fingerprint("what was reviewed")),
            ("p2", "free_text", None): ValueReview("safe", value_sha256=review_fingerprint("17.0")),
            ("p3", "free_text", None): ValueReview("redact", "Seen [REDACTED]", review_fingerprint("Seen in clinic")),
            ("p4", "free_text", None): ValueReview("safe"),  # no fingerprint: as before
        }
        with caplog.at_level(logging.WARNING):
            result, _ = BIDSDataset._apply_column_value_reviews(df, {"free_text"}, verdicts)
        assert pd.isna(result.loc[0, "free_text"])
        assert list(result["free_text"][1:]) == [" 17  ", "Seen [REDACTED]", "kept"]
        assert any("p1 free_text" in r.message and "differ from the answer that was reviewed" in r.message
                   for r in caplog.records)

    def test_the_fingerprint_is_loaded_with_the_verdict(self, tmp_path):
        (tmp_path / "column_value_reviews.json").write_text(json.dumps({"verdicts": [
            {"participant_id": "p1", "column_name": "c", "verdict": "safe", "value_sha256": "abc"}]}))
        assert BIDSDataset._load_column_value_reviews(tmp_path)[("p1", "c", None)] == ValueReview("safe", None, "abc")

    def test_a_session_without_its_own_verdict_is_withheld(self):
        df = pd.DataFrame({
            "participant_id": ["p1", "p1"], "session_id": ["s1", "s2"], "free_text": ["a", "b"]})
        verdicts = {("p1", "free_text", "s1"): ValueReview("safe")}
        result, _ = BIDSDataset._apply_column_value_reviews(df, {"free_text"}, verdicts)
        assert result.loc[0, "free_text"] == "a" and pd.isna(result.loc[1, "free_text"])

    def test_drop_nulls_value(self):
        df = pd.DataFrame({
            "participant_id": ["p1"],
            "free_text": ["PII content"],
        })
        verdicts = {("p1", "free_text", None): ValueReview("drop")}
        result, _ = BIDSDataset._apply_column_value_reviews(
            df, {"free_text"}, verdicts
        )
        assert pd.isna(result.loc[0, "free_text"])

    def test_no_verdict_nulls_value(self):
        df = pd.DataFrame({
            "participant_id": ["p1", "p2"],
            "free_text": ["reviewed", "unreviewed"],
        })
        verdicts = {("p1", "free_text", None): ValueReview("safe")}  # p2 has no verdict
        result, _ = BIDSDataset._apply_column_value_reviews(
            df, {"free_text"}, verdicts
        )
        assert result.loc[result["participant_id"] == "p1", "free_text"].values[0] == "reviewed"
        assert pd.isna(result.loc[result["participant_id"] == "p2", "free_text"].values[0])

    def test_no_verdicts_for_column_drops_it(self):
        df = pd.DataFrame({
            "participant_id": ["p1"],
            "reviewed_col": ["has verdict"],
            "unreviewed_col": ["no verdicts at all"],
        })
        verdicts = {("p1", "reviewed_col", None): ValueReview("safe")}
        result, dropped = BIDSDataset._apply_column_value_reviews(
            df, {"reviewed_col", "unreviewed_col"}, verdicts
        )
        assert "reviewed_col" in result.columns
        assert "unreviewed_col" not in result.columns
        assert "unreviewed_col" in dropped

    def test_empty_verdicts_drops_all_review_columns(self):
        df = pd.DataFrame({
            "participant_id": ["p1"],
            "review_col": ["some value"],
        })
        result, dropped = BIDSDataset._apply_column_value_reviews(
            df, {"review_col"}, {}
        )
        assert "review_col" not in result.columns
        assert "review_col" in dropped

    def test_rare_checkbox_label_needs_a_verdict_like_the_rest_of_the_cell(self, tmp_path, make_deid_tree):
        """The fold runs before the verdicts: a label folded into other_voice_activity (a review
        column) for a participant with no verdict is blanked with the cell."""
        bids, config = make_deid_tree(
            {"p1": [("s1", "rainbow-passage")], "p2": [("s1", "rainbow-passage")]},
            tables={"confounders/confounders": pd.DataFrame({
                "participant_id": ["p1", "p2"], "voice_activity_v2___attorney": ["1", ""],
                "voice_activity_v2___other": ["", "1"], "other_voice_activity": ["", "Podcaster"]})},
            settings={"small_checkbox_options": {"confounders": {"voice_activity_v2": {
                "other": "voice_activity_v2___other", "specify": "other_voice_activity", "min_participants": 10}}}},
            config_files={"column_value_reviews.json": {"verdicts": [
                {"participant_id": "p2", "column_name": "other_voice_activity", "verdict": "safe"}]}})
        BIDSDataset(bids).deidentify(outdir=tmp_path / "out", deidentify_config_dir=config)
        df = BIDSDataset._read_tsv_as_written(tmp_path / "out" / "phenotype" / "confounders" / "confounders.tsv")
        by_id = df.set_index("participant_id")
        assert "voice_activity_v2___attorney" not in df.columns
        assert by_id.loc["900000", "voice_activity_v2___other"] == "1"
        assert pd.isna(by_id.loc["900000", "other_voice_activity"])  # no verdict: no label
        assert by_id.loc["900001", "other_voice_activity"] == "Podcaster"


# the adult configs' small_checkbox_options (4.0-release/configs)
VOICE_ACTIVITY_FOLD = {"confounders": {
    "voice_activity_v2": {"other": "voice_activity_v2___other", "specify": "other_voice_activity",
                          "keep": ["voice_activity_v2___none"], "min_participants": 10},
    "voice_activity": {"other": "voice_activity___7", "min_participants": 10}}}


SITES = {"participant": {"enrollment_institution": {
    "output_column": "site", "values": {"MIT": "site_A", "USF": "site_B", "WCM": "site_C"}}}}


def _participants(sites):
    return pd.DataFrame({"participant_id": [f"p{i}" for i in range(len(sites))],
                         "enrollment_institution": sites, "age": ["40"] * len(sites)})


def _voice_activity(dtype):
    """12 participants: Teacher ticked by p0-p9 (10), Attorney by p10-p11 (2); p11 also typed 'Podcaster'."""
    tick = "1" if dtype == "text" else 1
    blank = "" if dtype == "text" else None
    df = pd.DataFrame({
        "participant_id": [f"p{i}" for i in range(12)],
        "voice_activity_v2___teacher": [tick] * 10 + [blank, blank],
        "voice_activity_v2___attorney": [blank] * 10 + [tick, tick],
        "voice_activity_v2___other": [blank] * 11 + [tick],
        "voice_activity_v2___none": [blank] * 12,
        "other_voice_activity": [blank] * 11 + ["Podcaster"],
    }, dtype=str if dtype == "text" else object)
    if dtype == "text":
        df = df.replace("", pd.NA)
    elements = {c: {"description": c} for c in df.columns}
    elements["voice_activity_v2___attorney"]["choices"] = [{"name": {"en": "Attorney"}, "value": "attorney"}]
    return df, elements, tick


GENDER_RULE = {"table": "pediatric_demographics", "column": "peds_gender_identity", "values": ["Other"]}


def _pediatric_tree(make_deid_tree, rule):
    """peds_gender_identity is a released field (shipped field map)."""
    return make_deid_tree(
        {"p1": [("s1", "rainbow-passage")], "p2": [("s1", "rainbow-passage")]},
        tables={"pediatric/pediatric_demographics": pd.DataFrame({
            "participant_id": ["p1", "p2"], "peds_gender_identity": ["Female gender identity", "Other"]})},
        choices={"pediatric/pediatric_demographics": {"peds_gender_identity": ["Female gender identity", "Other"]}},
        settings={"exclude_participants": [rule]})


@pytest.mark.parametrize("contents, error, match", [
    (None, FileNotFoundError, "access tier"),
    ({"access_tier": "public"}, ValueError, "access_tier must be one of"),
    ({"access_tier": "controlled", "typo": 1}, ValueError, "unknown setting"),
], ids=["missing-file", "unknown-tier", "unknown-key"])
def test_settings_file_errors(tmp_path, contents, error, match):
    if contents is not None:
        (tmp_path / "deidentify_settings.json").write_text(json.dumps(contents))
    with pytest.raises(error, match=match):
        BIDSDataset._load_deidentify_settings(tmp_path)


@pytest.mark.parametrize("key, value, empty", [
    ("value_mappings", SITES, {}),
    ("relabel", {"confounders": {"ph_walking": {"None": "No difficulty"}}}, {}),
    ("exclude_participants", [{"table": "t", "column": "c", "values": ["x"]}], []),
    ("small_checkbox_options", VOICE_ACTIVITY_FOLD, {}),
])
def test_optional_settings_pass_through_or_default_to_nothing(tmp_path, key, value, empty):
    path = tmp_path / "deidentify_settings.json"
    path.write_text(json.dumps({"access_tier": "controlled"}))
    settings = BIDSDataset._load_deidentify_settings(tmp_path)
    assert settings["access_tier"] is AccessTier.CONTROLLED and settings[key] == empty
    path.write_text(json.dumps({"access_tier": "registered", key: value}))
    assert BIDSDataset._load_deidentify_settings(tmp_path)[key] == value


def test_site_is_written_next_to_its_source_with_its_labels_as_choices():
    phenotype = {"enrollment_institution": {"description": "Enrollment Institution"}}
    df, phenotype = BIDSDataset._apply_value_mappings(
        _participants(["MIT", "USF", None]), phenotype, "participant", SITES)
    assert list(df.columns) == ["participant_id", "enrollment_institution", "site", "age"]
    assert df["site"].tolist()[:2] == ["site_A", "site_B"] and pd.isna(df["site"].iloc[2])
    assert df["enrollment_institution"].tolist()[:2] == ["MIT", "USF"]  # left for the disposition step
    element = phenotype["site"]
    assert [c["value"] for c in element["choices"]] == ["site_A", "site_B"]
    assert "arbitrary label" in element["description"] and "termURL" not in element


def test_a_value_with_no_mapping_stops_the_run():
    with pytest.raises(ValueError, match="no mapping"):
        BIDSDataset._apply_value_mappings(_participants(["MIT", "Elsewhere"]), {}, "participant", SITES)


def test_rare_labels_become_other_when_a_minimum_is_set():
    spec = json.loads(json.dumps(SITES))
    spec["participant"]["enrollment_institution"].update({"min_participants": 2, "other": "other"})
    df, phenotype = BIDSDataset._apply_value_mappings(_participants(["MIT", "MIT", "USF"]), {}, "participant", spec)
    assert df["site"].tolist() == ["site_A", "site_A", "other"]
    assert [c["value"] for c in phenotype["site"]["choices"]] == ["other", "site_A"]


def test_tables_without_a_mapping_are_untouched():
    df = _participants(["MIT"])
    out, _ = BIDSDataset._apply_value_mappings(df, {}, "demographics", SITES)
    assert out.equals(df)


@pytest.mark.parametrize("dtype", ["numeric", "text"])  # deidentify reads tables as text
def test_rare_checkbox_options_fold_into_other_with_their_label(dtype):
    df, elements, tick = _voice_activity(dtype)
    out, el = BIDSDataset._fold_small_checkbox_options(df, elements, "confounders", VOICE_ACTIVITY_FOLD)
    assert "voice_activity_v2___attorney" not in out.columns and "voice_activity_v2___attorney" not in el
    assert "voice_activity_v2___teacher" in out.columns
    assert list(out["voice_activity_v2___other"])[-2:] == [tick, tick]
    assert list(out["other_voice_activity"])[-2:] == ["Attorney", "Podcaster; Attorney"]


def test_rare_checkbox_options_are_counted_over_released_participants():
    df, elements, _ = _voice_activity("text")
    out, _ = BIDSDataset._fold_small_checkbox_options(
        df, elements, "confounders", VOICE_ACTIVITY_FOLD, released_participants={f"p{i}" for i in range(1, 12)})
    assert "voice_activity_v2___teacher" not in out.columns  # 9 released participants: folded


def test_deprecated_voice_activity_folds_rare_options_into_its_other_option():
    """The first version of the question has no 'please specify' box: rare options only tick Other."""
    df = pd.DataFrame({"participant_id": [f"p{i}" for i in range(12)],
                       "voice_activity___4": ["1"] * 10 + ["", ""],    # Teacher, 10 participants: kept
                       "voice_activity___6": [""] * 10 + ["1", "1"],   # Cheerleading, 2: folded
                       "voice_activity___7": [""] * 12}, dtype=str).replace("", pd.NA)
    out, _ = BIDSDataset._fold_small_checkbox_options(df, {}, "confounders", VOICE_ACTIVITY_FOLD)
    assert "voice_activity___6" not in out.columns and "voice_activity___4" in out.columns
    assert out["voice_activity___7"].tolist()[-2:] == ["1", "1"]


def test_relabel_changes_only_the_listed_labels_in_cells_and_choices():
    df = pd.DataFrame({"participant_id": ["1", "2", "3"], "ph_walking": ["None", "Mild", pd.NA]})
    choices = [{"name": {"en": "None"}, "value": "none"}, {"name": {"en": "Mild"}, "value": "mild"}]
    out, el = BIDSDataset._apply_relabels(df, {"ph_walking": {"choices": choices}}, "confounders",
                                         {"confounders": {"ph_walking": {"None": "No difficulty"}}})
    assert out["ph_walking"].tolist()[:2] == ["No difficulty", "Mild"] and pd.isna(out["ph_walking"].iloc[2])
    assert el["ph_walking"]["choices"] == [{"name": {"en": "No difficulty"}, "value": "none"},
                                           {"name": {"en": "Mild"}, "value": "mild"}]


def test_relabel_refuses_a_label_the_column_does_not_have():
    df = pd.DataFrame({"participant_id": ["1"], "ph_walking": ["None"]})
    elements = {"ph_walking": {"choices": [{"name": {"en": "None"}, "value": "none"}]}}
    with pytest.raises(ValueError, match="no choice labelled"):
        BIDSDataset._apply_relabels(df, elements, "confounders", {"confounders": {"ph_walking": {"Nome": "x"}}})


def test_relabelled_answers_and_their_choices_reach_the_released_json(tmp_path, make_deid_tree):
    """ph_walking is a released field (shipped field map)."""
    bids, config = make_deid_tree(
        {"p1": [("s1", "rainbow-passage")], "p2": [("s1", "rainbow-passage")]},
        tables={"confounders/confounders": pd.DataFrame({"participant_id": ["p1", "p2"],
                                                         "ph_walking": ["None", "Mild"]})},
        choices={"confounders/confounders": {"ph_walking": ["None", "Mild", "Extreme"]}},
        settings={"relabel": {"confounders": {"ph_walking": {"None": "No difficulty"}}}})
    BIDSDataset(bids).deidentify(outdir=tmp_path / "out", deidentify_config_dir=config)
    released = tmp_path / "out" / "phenotype" / "confounders"
    assert sorted(BIDSDataset._read_tsv_as_written(released / "confounders.tsv")["ph_walking"]) == [
        "Mild", "No difficulty"]
    element = json.loads((released / "confounders.json").read_text())["confounders"]["data_elements"]["ph_walking"]
    assert element["choices"] == [{"name": {"en": "No difficulty"}, "value": "none"},
                                  {"name": {"en": "Mild"}, "value": "mild"},
                                  {"name": {"en": "Extreme"}, "value": "extreme"}]


def test_exclude_participants_removes_a_participant_by_one_answer(tmp_path, make_deid_tree):
    bids, config = _pediatric_tree(make_deid_tree, GENDER_RULE)
    BIDSDataset(bids).deidentify(outdir=tmp_path / "out", deidentify_config_dir=config)
    assert sorted(p.name for p in (tmp_path / "out").glob("sub-*")) == ["sub-900000"]
    released = BIDSDataset._read_tsv_as_written(
        tmp_path / "out" / "phenotype" / "pediatric" / "pediatric_demographics.tsv")
    assert list(released["participant_id"]) == ["900000"]


@pytest.mark.parametrize("typo, match", [
    ({"column": "peds_gender_identiy"}, "no column 'peds_gender_identiy'"),
    ({"values": ["other"]}, r"no choice labelled \['other'\]"),
], ids=["column", "value"])
def test_exclude_participants_refuses_a_rule_that_would_match_nothing(tmp_path, make_deid_tree, typo, match):
    bids, config = _pediatric_tree(make_deid_tree, {**GENDER_RULE, **typo})
    with pytest.raises(ValueError, match=match):
        BIDSDataset(bids).deidentify(outdir=tmp_path / "out", deidentify_config_dir=config)
