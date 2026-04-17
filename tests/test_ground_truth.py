"""T035 — SC-006 ground-truth classification tests.

Uses synthetic WAV fixtures with known properties to verify that the pipeline
produces the expected classifications:
- clean single-speaker on-task audio → PASS or NEEDS_REVIEW (not FAIL)
- clipped/silent/noisy audio → FAIL or NEEDS_REVIEW (not PASS)
"""

import csv
from pathlib import Path

import pytest
from click.testing import CliRunner

from b2aiprep.cli import cli


def _run_pipeline(bids_dir: Path, output_dir: Path) -> dict[str, str]:
    """Run qa-run and return {task_name: final_classification} from composite TSV."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "qa-run",
            str(bids_dir),
            str(output_dir),
            "--skip-pii",
            "--skip-task-compliance",
        ],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output

    tsv = output_dir / "qa_composite_scores.tsv"
    assert tsv.exists()
    classifications = {}
    with open(tsv, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            classifications[row["task_name"]] = row["final_classification"]
    return classifications


def _make_bids(tmp_path, wav_factory, pid, wav_type, task="harvard-sentences-list-1-1"):
    sid = "ses-01"
    bids = tmp_path / "bids"
    audio_dir = bids / pid / sid / "audio"
    audio_dir.mkdir(parents=True)
    wav_factory(
        wav_type=wav_type,
        filename=f"{pid}_{sid}_task-{task}_audio.wav",
        target_dir=audio_dir,
    )
    pt = audio_dir / f"{pid}_{sid}_task-{task}_features.pt"
    pt.write_bytes(b"")
    return bids


@pytest.fixture
def _bids_clean(tmp_path, wav_factory):
    return _make_bids(tmp_path, wav_factory, "sub-gtclean", "clean")


@pytest.fixture
def _bids_clipped(tmp_path, wav_factory):
    return _make_bids(tmp_path, wav_factory, "sub-gtclip", "clipped")


@pytest.fixture
def _bids_silent(tmp_path, wav_factory):
    return _make_bids(tmp_path, wav_factory, "sub-gtsilent", "silent")


class TestGroundTruth:
    def test_clean_audio_not_fail(self, _bids_clean, tmp_path):
        """SC-006: clean audio must not auto-fail on audio quality."""
        out = tmp_path / "out"
        classifications = _run_pipeline(_bids_clean, out)
        assert classifications, "no rows in composite TSV"
        for task, cls in classifications.items():
            assert cls != "fail", (
                f"Clean audio task {task!r} unexpectedly classified as fail"
            )

    def test_clipped_audio_not_pass(self, _bids_clipped, tmp_path):
        """SC-006: heavily clipped audio must not auto-pass."""
        out = tmp_path / "out"
        classifications = _run_pipeline(_bids_clipped, out)
        assert classifications, "no rows in composite TSV"
        for task, cls in classifications.items():
            assert cls != "pass", (
                f"Clipped audio task {task!r} unexpectedly classified as pass"
            )

    def test_silent_audio_not_pass(self, _bids_silent, tmp_path):
        """SC-006: silence-only audio must not auto-pass."""
        out = tmp_path / "out"
        classifications = _run_pipeline(_bids_silent, out)
        assert classifications, "no rows in composite TSV"
        for task, cls in classifications.items():
            assert cls != "pass", (
                f"Silent audio task {task!r} unexpectedly classified as pass"
            )
