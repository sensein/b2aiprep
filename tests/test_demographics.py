from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from b2aiprep.prepare.demographics import load_features_for_recordings


class TestLoadFeaturesForRecordings:
    """Test cases for the load_features_for_recordings function."""

    def test_load_features_basic(self):
        """Test basic functionality without feature filtering."""
        # Create test DataFrame
        df = pd.DataFrame(
            {
                "recording_id": ["rec1", "rec2", "rec1", "rec3"],  # rec1 appears twice
                "other_column": ["a", "b", "c", "d"],
            }
        )

        # Mock torch tensors
        mock_tensor1 = {"feature1": torch.tensor([1, 2, 3]), "feature2": torch.tensor([4, 5, 6])}
        mock_tensor2 = {"feature1": torch.tensor([7, 8, 9]), "feature2": torch.tensor([10, 11, 12])}
        mock_tensor3 = {
            "feature1": torch.tensor([13, 14, 15]),
            "feature2": torch.tensor([16, 17, 18]),
        }

        with TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            with patch("torch.load") as mock_torch_load:
                # Configure mock to return different tensors for different files
                def side_effect(filepath, weights_only=False):
                    if "rec1_features.pt" in str(filepath):
                        return mock_tensor1
                    elif "rec2_features.pt" in str(filepath):
                        return mock_tensor2
                    elif "rec3_features.pt" in str(filepath):
                        return mock_tensor3
                    else:
                        raise FileNotFoundError(f"No such file: {filepath}")

                mock_torch_load.side_effect = side_effect

                result = load_features_for_recordings(df, data_path)

                # Should have loaded 3 unique recordings
                assert len(result) == 3
                assert "rec1" in result
                assert "rec2" in result
                assert "rec3" in result

                # Verify correct tensors were returned
                assert result["rec1"] == mock_tensor1
                assert result["rec2"] == mock_tensor2
                assert result["rec3"] == mock_tensor3

                # Verify torch.load was called with correct paths and weights_only=False
                expected_calls = [
                    ((data_path / "rec1_features.pt",), {"weights_only": False}),
                    ((data_path / "rec2_features.pt",), {"weights_only": False}),
                    ((data_path / "rec3_features.pt",), {"weights_only": False}),
                ]

                # Check that all expected calls were made (order might vary)
                actual_calls = [(call.args, call.kwargs) for call in mock_torch_load.call_args_list]
                for expected_call in expected_calls:
                    assert expected_call in actual_calls

    def test_load_features_with_feature_selection(self):
        """Test loading features with specific feature selection."""
        df = pd.DataFrame({"recording_id": ["rec1", "rec2"], "other_column": ["a", "b"]})

        # Mock torch tensors with multiple features
        mock_tensor1 = {
            "specgram": torch.tensor([1, 2, 3]),
            "mfcc": torch.tensor([4, 5, 6]),
            "opensmile": torch.tensor([7, 8, 9]),
        }
        mock_tensor2 = {
            "specgram": torch.tensor([10, 11, 12]),
            "mfcc": torch.tensor([13, 14, 15]),
            "opensmile": torch.tensor([16, 17, 18]),
        }

        with TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            with patch("torch.load") as mock_torch_load:

                def side_effect(filepath, weights_only=False):
                    if "rec1_features.pt" in str(filepath):
                        return mock_tensor1
                    elif "rec2_features.pt" in str(filepath):
                        return mock_tensor2
                    else:
                        raise FileNotFoundError(f"No such file: {filepath}")

                mock_torch_load.side_effect = side_effect

                # Test with 'mfcc' feature selection
                result = load_features_for_recordings(df, data_path, feature="mfcc")

                # Should have extracted only the 'mfcc' feature
                assert len(result) == 2
                assert torch.equal(result["rec1"], torch.tensor([4, 5, 6]))
                assert torch.equal(result["rec2"], torch.tensor([13, 14, 15]))

    def test_load_features_all_valid_features(self):
        """Test that all defined feature options work correctly."""
        df = pd.DataFrame({"recording_id": ["rec1"]})

        feature_options = [
            "specgram",
            "melfilterbank",
            "mfcc",
            "opensmile",
            "sample_rate",
            "checksum",
            "transcription",
        ]

        mock_tensor = {option: torch.tensor([1, 2, 3]) for option in feature_options}

        with TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            with patch("torch.load", return_value=mock_tensor):
                # Test each valid feature option
                for feature in feature_options:
                    result = load_features_for_recordings(df, data_path, feature=feature)
                    assert len(result) == 1
                    assert "rec1" in result
                    assert torch.equal(result["rec1"], torch.tensor([1, 2, 3]))

    def test_load_features_invalid_feature(self):
        """Test that ValueError is raised for invalid feature names."""
        df = pd.DataFrame({"recording_id": ["rec1"]})

        with TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            # Test with invalid feature
            with pytest.raises(ValueError, match="Unrecognized feature invalid_feature"):
                load_features_for_recordings(df, data_path, feature="invalid_feature")

            # Test with another invalid feature
            with pytest.raises(ValueError, match="Unrecognized feature xyz"):
                load_features_for_recordings(df, data_path, feature="xyz")

    def test_load_features_empty_dataframe(self):
        """Test behavior with empty DataFrame."""
        df = pd.DataFrame({"recording_id": []})

        with TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            with patch("torch.load") as mock_torch_load:
                result = load_features_for_recordings(df, data_path)

                # Should return empty dict
                assert result == {}
                # torch.load should not be called
                mock_torch_load.assert_not_called()

    def test_load_features_single_recording(self):
        """Test with DataFrame containing single recording."""
        df = pd.DataFrame({"recording_id": ["single_rec"], "session": ["session1"]})

        mock_tensor = {"feature": torch.tensor([100, 200, 300])}

        with TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            with patch("torch.load", return_value=mock_tensor) as mock_torch_load:
                result = load_features_for_recordings(df, data_path)

                assert len(result) == 1
                assert result["single_rec"] == mock_tensor

                # Verify correct file path was used
                mock_torch_load.assert_called_once_with(
                    data_path / "single_rec_features.pt", weights_only=False
                )

    def test_load_features_duplicate_recording_ids(self):
        """Test that duplicate recording IDs are handled correctly (loaded only once)."""
        df = pd.DataFrame(
            {
                "recording_id": ["dup_rec", "dup_rec", "dup_rec", "other_rec"],
                "session": ["s1", "s2", "s3", "s4"],
            }
        )

        mock_tensor1 = {"feature": torch.tensor([1, 1, 1])}
        mock_tensor2 = {"feature": torch.tensor([2, 2, 2])}

        with TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            with patch("torch.load") as mock_torch_load:

                def side_effect(filepath, weights_only=False):
                    if "dup_rec_features.pt" in str(filepath):
                        return mock_tensor1
                    elif "other_rec_features.pt" in str(filepath):
                        return mock_tensor2
                    else:
                        raise FileNotFoundError(f"No such file: {filepath}")

                mock_torch_load.side_effect = side_effect

                result = load_features_for_recordings(df, data_path)

                # Should have 2 unique recordings
                assert len(result) == 2
                assert result["dup_rec"] == mock_tensor1
                assert result["other_rec"] == mock_tensor2

                # torch.load should be called exactly twice (once per unique recording)
                assert mock_torch_load.call_count == 2

    def test_load_features_file_not_found(self):
        """Test error handling when feature file doesn't exist."""
        df = pd.DataFrame({"recording_id": ["missing_rec"]})

        with TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            with patch("torch.load", side_effect=FileNotFoundError("File not found")):
                # Should propagate the FileNotFoundError
                with pytest.raises(FileNotFoundError, match="File not found"):
                    load_features_for_recordings(df, data_path)

    def test_load_features_torch_load_error(self):
        """Test error handling when torch.load fails for other reasons."""
        df = pd.DataFrame({"recording_id": ["corrupt_rec"]})

        with TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            with patch("torch.load", side_effect=RuntimeError("Corrupted file")):
                # Should propagate the RuntimeError
                with pytest.raises(RuntimeError, match="Corrupted file"):
                    load_features_for_recordings(df, data_path)

    def test_load_features_no_recording_id_column(self):
        """Test error handling when DataFrame doesn't have recording_id column."""
        df = pd.DataFrame({"other_column": ["value1", "value2"]})

        with TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            # Should raise KeyError for missing column
            with pytest.raises(KeyError, match="recording_id"):
                load_features_for_recordings(df, data_path)

    def test_load_features_feature_not_in_tensor(self):
        """Test error handling when requested feature is not in loaded tensor."""
        df = pd.DataFrame({"recording_id": ["rec1"]})

        # Mock tensor without the requested feature
        mock_tensor = {"other_feature": torch.tensor([1, 2, 3])}

        with TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            with patch("torch.load", return_value=mock_tensor):
                # Should raise KeyError when trying to access missing feature
                with pytest.raises(KeyError):
                    load_features_for_recordings(df, data_path, feature="mfcc")

    def test_load_features_complex_scenario(self):
        """Test a complex scenario with multiple recordings and feature selection."""
        df = pd.DataFrame(
            {
                "recording_id": ["audio_001", "audio_002", "audio_001", "audio_003"],
                "subject_id": ["sub1", "sub2", "sub1", "sub3"],
                "session": ["ses1", "ses1", "ses2", "ses1"],
            }
        )

        # Create realistic mock tensors
        mock_tensors = {
            "audio_001": {
                "specgram": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "mfcc": torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
                "opensmile": torch.tensor([10.5, 11.5, 12.5]),
            },
            "audio_002": {
                "specgram": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
                "mfcc": torch.tensor([[0.5, 0.6], [0.7, 0.8]]),
                "opensmile": torch.tensor([20.5, 21.5, 22.5]),
            },
            "audio_003": {
                "specgram": torch.tensor([[9.0, 10.0], [11.0, 12.0]]),
                "mfcc": torch.tensor([[0.9, 1.0], [1.1, 1.2]]),
                "opensmile": torch.tensor([30.5, 31.5, 32.5]),
            },
        }

        with TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            with patch("torch.load") as mock_torch_load:

                def side_effect(filepath, weights_only=False):
                    filename = filepath.name
                    for rec_id, tensor in mock_tensors.items():
                        if filename == f"{rec_id}_features.pt":
                            return tensor
                    raise FileNotFoundError(f"No such file: {filepath}")

                mock_torch_load.side_effect = side_effect

                # Test with specgram feature selection
                result = load_features_for_recordings(df, data_path, feature="specgram")

                assert len(result) == 3
                assert torch.equal(result["audio_001"], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
                assert torch.equal(result["audio_002"], torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
                assert torch.equal(result["audio_003"], torch.tensor([[9.0, 10.0], [11.0, 12.0]]))

                # Verify exactly 3 files were loaded
                assert mock_torch_load.call_count == 3
