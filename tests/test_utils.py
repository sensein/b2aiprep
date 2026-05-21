import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
import pandas as pd
import pytest

from b2aiprep.prepare.utils import (
    copy_package_resource,
    get_commit_sha,
    make_tsv_files,
    reformat_resources,
    remove_files_by_pattern,
    get_wav_duration,
)


def test_reformat_resources():
    # Create temporary directories for input and output
    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        # Create a sample input JSON file with a list
        sample_json = ["key1", "key2", "key3"]
        sample_json_path = os.path.join(input_dir, "sample.json")
        with open(sample_json_path, "w") as f:
            json.dump(sample_json, f)

        # Run the function
        reformat_resources(input_dir, output_dir)

        # Check if the output JSON file is created correctly
        output_json_path = os.path.join(output_dir, "sample.json")
        assert os.path.exists(output_json_path)

        with open(output_json_path, "r") as f:
            result = json.load(f)

        # Verify the format
        expected = {
            "key1": {"description": ""},
            "key2": {"description": ""},
            "key3": {"description": ""},
        }
        assert result == expected


def test_make_tsv_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a sample JSON file
        json_data = {
            "col1": {"description": ""},
            "col2": {"description": ""},
            "col3": {"description": ""},
        }
        json_path = os.path.join(temp_dir, "sample.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f)

        # Run the function
        make_tsv_files(temp_dir)

        # Check if the TSV file is created
        tsv_path = os.path.join(temp_dir, "sample.tsv")
        assert os.path.exists(tsv_path)

        # Load TSV file and check columns
        df = pd.read_csv(tsv_path, sep="\t")
        assert list(df.columns) == ["col1", "col2", "col3"]

def test_remove_files_by_pattern():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some test files
        txt_file_path = os.path.join(temp_dir, "test.txt")
        json_file_path = os.path.join(temp_dir, "test.json")
        with open(txt_file_path, "w") as f:
            f.write("test file")
        with open(json_file_path, "w") as f:
            f.write("test file")

        # Run the function to remove .txt files
        remove_files_by_pattern(temp_dir, "*.txt")

        # Check results
        assert not os.path.exists(txt_file_path)
        assert os.path.exists(json_file_path)

def test_get_wav_duration():
    b2ai_wav_path = "data/test_audio2.wav"
    actual = get_wav_duration(b2ai_wav_path)
    expected  = 2.9257142857142857
    assert actual == expected


def test_get_commit_sha_reads_file(tmp_path):
    sha = "abcdef1234567890abcdef1234567890abcdef12"
    (tmp_path / "commit_sha.txt").write_text(sha + "\n")
    assert get_commit_sha(tmp_path) == sha


def test_get_commit_sha_falls_back_to_git(tmp_path, monkeypatch):
    expected_sha = "a" * 40
    captured = {}

    def fake_run(cmd, capture_output, check, text):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=expected_sha + "\n", stderr="")

    monkeypatch.setattr("b2aiprep.prepare.utils.subprocess.run", fake_run)
    assert get_commit_sha(tmp_path) == expected_sha
    assert captured["cmd"] == ["git", "-C", str(tmp_path), "rev-parse", "HEAD"]


@pytest.mark.parametrize(
    "exc",
    [
        subprocess.CalledProcessError(128, ["git"]),
        FileNotFoundError("git"),
    ],
    ids=["git_not_a_repo", "git_not_installed"],
)
def test_get_commit_sha_returns_empty_when_unrecoverable(tmp_path, monkeypatch, caplog, exc):
    def fake_run(*args, **kwargs):
        raise exc

    monkeypatch.setattr("b2aiprep.prepare.utils.subprocess.run", fake_run)
    caplog.set_level(logging.WARNING, logger="b2aiprep.prepare.utils")
    assert get_commit_sha(tmp_path) == ""
    assert "Could not determine reproschema commit SHA" in caplog.text


if __name__ == "__main__":
    pytest.main()
