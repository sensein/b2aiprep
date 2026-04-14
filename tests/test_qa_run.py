"""T013 — End-to-end integration test for the ``qa-run`` CLI command.

Tests using Click's CliRunner on a synthetic batch of WAV files with known
ground truth.  Synthetic WAVs come from the ``wav_factory`` fixture defined in
conftest.py — no import needed; pytest injects it automatically.

All tests skip automatically if:
- ``qa-run`` is not yet registered in the CLI (T022/T023), OR
- required modules (quality_control extension, etc.) are not yet implemented.

Assertions:
- qa_check_results.tsv, qa_composite_scores.tsv, needs_review_queue.tsv,
  and qa_pipeline_config_*.json are written to OUTPUT_DIR
- Pipeline is deterministic: running twice with the same config produces
  byte-for-byte identical TSV outputs
"""

from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Guard — skip entire file if CLI command or core modules aren't ready
# ---------------------------------------------------------------------------

try:
    from click.testing import CliRunner
    from b2aiprep.cli import cli

    _commands = set(cli.commands.keys()) if hasattr(cli, "commands") else set()
    if "qa-run" not in _commands:
        pytest.skip("qa-run not yet registered in CLI (T023)", allow_module_level=True)
except (ImportError, AttributeError):
    pytest.skip("CLI not importable", allow_module_level=True)

try:
    import b2aiprep.prepare.quality_control as _qc
    if not hasattr(_qc, "check_audio_quality"):
        pytest.skip("check_audio_quality not yet implemented (T014)", allow_module_level=True)
except ImportError:
    pytest.skip("quality_control extension not yet implemented", allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bids_dir(tmp_path, wav_factory):
    """Minimal synthetic BIDS dataset with 3 audio files."""
    bids = tmp_path / "bids"
    bids.mkdir()

    subjects = [
        ("sub-001", "ses-01", "harvard-sentences-list-1-1"),
        ("sub-002", "ses-01", "phonation"),
        ("sub-003", "ses-01", "conversational-storytelling"),
    ]

    for pid, sid, task in subjects:
        voice_dir = bids / pid / sid / "voice"
        voice_dir.mkdir(parents=True)

        wav_factory(
            wav_type="clean",
            filename=f"{pid}_{sid}_task-{task}_audio.wav",
            target_dir=voice_dir,
        )

        # Minimal .pt placeholder; pipeline mocks feature loading during tests
        pt_file = voice_dir / f"{pid}_{sid}_task-{task}_features.pt"
        pt_file.write_bytes(b"")

    return bids


@pytest.fixture
def output_dir(tmp_path):
    out = tmp_path / "qa_output"
    out.mkdir()
    return out


# ---------------------------------------------------------------------------
# Output file presence tests
# ---------------------------------------------------------------------------


class TestOutputFiles:
    def test_check_results_tsv_written(self, bids_dir, output_dir):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "qa-run", str(bids_dir), str(output_dir),
            "--skip-pii", "--skip-task-compliance",
        ])
        assert result.exit_code == 0, f"qa-run failed:\n{result.output}"
        assert (output_dir / "qa_check_results.tsv").exists()

    def test_composite_scores_tsv_written(self, bids_dir, output_dir):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "qa-run", str(bids_dir), str(output_dir),
            "--skip-pii", "--skip-task-compliance",
        ])
        assert result.exit_code == 0
        assert (output_dir / "qa_composite_scores.tsv").exists()

    def test_needs_review_queue_tsv_written(self, bids_dir, output_dir):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "qa-run", str(bids_dir), str(output_dir),
            "--skip-pii", "--skip-task-compliance",
        ])
        assert result.exit_code == 0
        assert (output_dir / "needs_review_queue.tsv").exists()

    def test_config_snapshot_written(self, bids_dir, output_dir):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "qa-run", str(bids_dir), str(output_dir),
            "--skip-pii", "--skip-task-compliance",
        ])
        assert result.exit_code == 0
        config_files = list(output_dir.glob("qa_pipeline_config_*.json"))
        assert len(config_files) == 1, f"Expected exactly one config snapshot, found: {config_files}"

    def test_exit_code_zero_on_success(self, bids_dir, output_dir):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "qa-run", str(bids_dir), str(output_dir),
            "--skip-pii", "--skip-task-compliance",
        ])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# FR-011 — Corrupt/unreadable files skipped, not halting, absent from TSVs
# ---------------------------------------------------------------------------


class TestFR011Integration:
    def test_corrupt_file_does_not_halt_pipeline(self, tmp_path, wav_factory):
        """A zero-byte WAV in the batch must not cause the pipeline to fail."""
        bids = tmp_path / "bids"
        voice = bids / "sub-001" / "ses-01" / "voice"
        voice.mkdir(parents=True)

        wav_factory(
            wav_type="clean",
            filename="sub-001_ses-01_task-harvard-sentences-list-1-1_audio.wav",
            target_dir=voice,
        )

        corrupt = voice / "sub-001_ses-01_task-phonation_audio.wav"
        corrupt.write_bytes(b"")

        out = tmp_path / "out"
        out.mkdir()

        runner = CliRunner()
        result = runner.invoke(cli, [
            "qa-run", str(bids), str(out),
            "--skip-pii", "--skip-task-compliance",
        ])
        assert result.exit_code == 0, f"Pipeline halted on corrupt file:\n{result.output}"

    def test_corrupt_file_absent_from_composite_scores_tsv(self, tmp_path, wav_factory):
        bids = tmp_path / "bids"
        voice = bids / "sub-001" / "ses-01" / "voice"
        voice.mkdir(parents=True)

        wav_factory(
            wav_type="clean",
            filename="sub-001_ses-01_task-harvard-sentences-list-1-1_audio.wav",
            target_dir=voice,
        )

        corrupt = voice / "sub-001_ses-01_task-corrupt_audio.wav"
        corrupt.write_bytes(b"")

        out = tmp_path / "out"
        out.mkdir()

        runner = CliRunner()
        runner.invoke(cli, [
            "qa-run", str(bids), str(out),
            "--skip-pii", "--skip-task-compliance",
        ])

        composite_tsv = out / "qa_composite_scores.tsv"
        if composite_tsv.exists():
            content = composite_tsv.read_text()
            assert "task-corrupt" not in content


# ---------------------------------------------------------------------------
# Determinism test (SC-002)
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_two_runs_produce_identical_composite_scores(self, bids_dir, tmp_path):
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        out1.mkdir()
        out2.mkdir()

        runner = CliRunner()
        r1 = runner.invoke(cli, [
            "qa-run", str(bids_dir), str(out1),
            "--skip-pii", "--skip-task-compliance",
        ])
        r2 = runner.invoke(cli, [
            "qa-run", str(bids_dir), str(out2),
            "--skip-pii", "--skip-task-compliance",
        ])

        assert r1.exit_code == 0
        assert r2.exit_code == 0

        tsv1 = (out1 / "qa_composite_scores.tsv").read_text()
        tsv2 = (out2 / "qa_composite_scores.tsv").read_text()
        assert tsv1 == tsv2, "qa_composite_scores.tsv is not deterministic across runs"

    def test_two_runs_produce_identical_check_results(self, bids_dir, tmp_path):
        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        out1.mkdir()
        out2.mkdir()

        runner = CliRunner()
        runner.invoke(cli, [
            "qa-run", str(bids_dir), str(out1),
            "--skip-pii", "--skip-task-compliance",
        ])
        runner.invoke(cli, [
            "qa-run", str(bids_dir), str(out2),
            "--skip-pii", "--skip-task-compliance",
        ])

        tsv1 = (out1 / "qa_check_results.tsv").read_text()
        tsv2 = (out2 / "qa_check_results.tsv").read_text()
        assert tsv1 == tsv2, "qa_check_results.tsv is not deterministic across runs"
