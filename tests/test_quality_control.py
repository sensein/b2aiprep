from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

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
