from pathlib import Path
from unittest.mock import patch

import json

import pandas as pd
import pytest
import torch
from conftest import WAV_BYTES

from b2aiprep.prepare import quality_control
from b2aiprep.prepare.quality_control import quality_control_wrapper


def _no_snorkel(name, *args, **kwargs):
    return None if name == "snorkel" else object()


def test_deep_checks_without_an_output_directory_is_refused_before_any_metric(tmp_path):
    with patch.object(quality_control, "check_quality") as check_quality:
        with pytest.raises(ValueError, match="output directory"):
            quality_control_wrapper(audio_paths=[], bids_path=tmp_path, deep_checks=True)
    check_quality.assert_not_called()


def test_deep_checks_without_snorkel_is_refused_before_any_metric(tmp_path):
    with patch.object(quality_control.importlib.util, "find_spec", side_effect=_no_snorkel), \
            patch.object(quality_control, "check_quality") as check_quality:
        with pytest.raises(ImportError, match=r"b2aiprep\[qc\]"):
            quality_control_wrapper(audio_paths=[], bids_path=tmp_path, outdir=tmp_path / "qc", deep_checks=True)
    check_quality.assert_not_called()


def test_deep_checks_accept_a_string_output_directory_and_write_there(tmp_path):
    outdir = tmp_path / "qc"
    outdir.mkdir()
    bids = tmp_path / "bids"
    bids.mkdir()
    with patch.object(quality_control.importlib.util, "find_spec", return_value=object()), \
            patch.object(quality_control, "check_quality", return_value=pd.DataFrame({"path": []})), \
            patch.object(quality_control, "review_files") as review_files:
        quality_control_wrapper(audio_paths=[], bids_path=bids, outdir=str(outdir), deep_checks=True)
    assert review_files.call_args.kwargs["df_path"] == outdir / "quality_control_results_non_windowed.csv"
    assert Path(review_files.call_args.kwargs["output_dir"]) == outdir
    assert (outdir / "silence_removal.csv").exists()
    assert (outdir / "diarization_check.csv").exists()
    assert (bids / "audio_quality_metrics.tsv").exists()


def test_review_files_runs_when_snorkel_is_installed():
    pytest.importorskip("snorkel")
    from senselab.audio.tasks.quality_control import review

    assert review.SNORKEL_AVAILABLE


def test_missing_speakers_are_flagged_only_for_lexical_tasks(tmp_path):
    """The diarization check reads the registry's speech_type from the sidecar, so a pediatric
    speech task is checked and a prolonged vowel is not."""
    audio = tmp_path / "bids" / "sub-p" / "ses-s" / "audio"
    audio.mkdir(parents=True)
    paths = []
    for task, speech_type in [("repeating-words", "read"), ("prolonged-vowel", "non-lexical"), ("no-sidecar", None)]:
        stem = f"sub-p_ses-s_task-{task}"
        (audio / f"{stem}.wav").write_bytes(WAV_BYTES)
        torch.save({"diarization": []}, audio / f"{stem}_features.pt")
        if speech_type:
            (audio / f"{stem}_recording-metadata.json").write_text(json.dumps({"speech_type": speech_type}))
        paths.append(audio / f"{stem}.wav")
    outdir = tmp_path / "qc"
    with patch.object(quality_control.importlib.util, "find_spec", return_value=object()), \
            patch.object(quality_control, "check_quality", return_value=pd.DataFrame({"path": []})), \
            patch.object(quality_control, "review_files"), \
            patch.object(quality_control, "primary_speaker_ratio_metric", return_value=0.0):
        quality_control_wrapper(audio_paths=paths, bids_path=tmp_path / "bids", outdir=outdir, deep_checks=True)
    flags = pd.read_csv(outdir / "diarization_check.csv").set_index("task")["flag"]
    assert flags["task-repeating-words"] == "no_speakers_found"
    assert pd.isna(flags["task-prolonged-vowel"])
    assert pd.isna(flags["task-no-sidecar"])
