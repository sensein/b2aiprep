from pathlib import Path

import torch

from b2aiprep.prepare.dataset import BIDSDataset
from b2aiprep.prepare.utils import sanitize_task_entity_in_bids_stem


def test_deidentify_feature_files_removes_sparc_ema_for_sensitive_tasks(tmp_path: Path) -> None:
    bids_root = tmp_path / "bids"
    out_root = tmp_path / "out"

    in_audio_dir = bids_root / "sub-001" / "ses-001" / "audio"
    in_audio_dir.mkdir(parents=True, exist_ok=True)

    sensitive_pt = in_audio_dir / (
        "sub-001_ses-001_task-Free Speech_rec-test_features.pt"
    )
    nonsensitive_pt = in_audio_dir / (
        "sub-001_ses-001_task-NonSensitive_rec-test_features.pt"
    )

    payload = {
        "sparc": {"ema": [1, 2, 3], "pitch": [4, 5, 6]},
        "torchaudio": {"spectrogram": [[0.0]]},
    }
    torch.save(payload, sensitive_pt)
    torch.save(payload, nonsensitive_pt)

    BIDSDataset._deidentify_feature_files(
        bids_root,
        out_root,
        sensitive_audio_task_list=["free-speech"],
    )

    def _expected_out_name(input_pt: Path) -> str:
        features_ending = "_features"
        audio_path_stem = sanitize_task_entity_in_bids_stem(
            input_pt.stem.replace(features_ending, "")
        )
        path_stem_ending = "-".join(audio_path_stem.split("_")[2:]) + features_ending
        return f"sub-001_ses-001_{path_stem_ending}{input_pt.suffix}"

    out_sensitive_pt = out_root / "sub-001" / "ses-001" / "audio" / _expected_out_name(sensitive_pt)
    out_nonsensitive_pt = out_root / "sub-001" / "ses-001" / "audio" / _expected_out_name(nonsensitive_pt)

    assert out_sensitive_pt.exists()
    assert out_nonsensitive_pt.exists()

    sensitive_out = torch.load(out_sensitive_pt, weights_only=False, map_location=torch.device("cpu"))
    nonsensitive_out = torch.load(out_nonsensitive_pt, weights_only=False, map_location=torch.device("cpu"))

    assert "ema" not in sensitive_out.get("sparc", {})
    assert "pitch" in sensitive_out.get("sparc", {})
    assert "ema" in nonsensitive_out.get("sparc", {})


def test_deidentify_feature_files_sanitizes_task_in_output_filename(tmp_path: Path) -> None:
    bids_root = tmp_path / "bids"
    out_root = tmp_path / "out"

    in_audio_dir = bids_root / "sub-001" / "ses-001" / "audio"
    in_audio_dir.mkdir(parents=True, exist_ok=True)

    pt = in_audio_dir / "sub-001_ses-001_task-Free Speech_rec-test_features.pt"
    torch.save({"sparc": {"ema": [1, 2, 3]}}, pt)

    BIDSDataset._deidentify_feature_files(
        bids_root,
        out_root,
        sensitive_audio_task_list=[],
    )

    out_files = list((out_root / "sub-001" / "ses-001" / "audio").glob("*.pt"))
    assert len(out_files) == 1
    assert "task-free-speech" in out_files[0].name


def test_deidentify_phenotype_sanitizes_acoustic_task_name(tmp_path: Path) -> None:
    import pandas as pd

    df = pd.DataFrame(
        {
            "participant_id": ["001"],
            "acoustic_task_name": ["Free Speech"],
        }
    )
    out_df, _ = BIDSDataset._deidentify_phenotype(df, phenotype={})
    assert out_df.loc[0, "acoustic_task_name"] == "free-speech"
