import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from b2aiprep.prepare.utils import (
    TaskMatcher,
    canonical_task_entity,
    copy_package_resource,
    get_commit_sha,
    get_wav_duration,
    is_audio_check,
    load_recording_name_aliases,
    make_tsv_files,
    reformat_resources,
    remove_files_by_pattern,
    sanitize_task_entity_in_bids_stem,
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


def test_task_matcher_exact_glob_and_regex():
    m = TaskMatcher(["Noisy-Sounds-1", "identifying-pictures-*", "picture-description",
                     "Repeating Words *", "re:role-naming-tasks-sounds-(days|months)", "conversation-*"])
    assert "noisy-sounds-1" in m and "Noisy Sounds 1" in m
    assert "Identifying-Pictures-35" in m and "identifying-pictures" not in m
    assert "picture-description" in m and "picture-description-2" not in m and "picture-28" not in m
    assert "repeating-words-bad" in m
    assert "Role-Naming-Tasks-Sounds-Days" in m and "role-naming-tasks-sounds-numbers" not in m
    assert "Conversation-(6-plus)-favorite-food" in m
    assert not TaskMatcher([]) and TaskMatcher(["re:x"])


@pytest.mark.parametrize(
    "name",
    ["Audio Check", "Audio Check-1", "Audio Check (v2)-5", "audio-check-v2-3",
     "audio-check-(v2)", "AUDIO CHECK", "audio_check"],
)
def test_every_collected_spelling_is_recognised(name):
    """All of these appear in the v4 exports; they must not need separate handling."""
    assert is_audio_check(name)


@pytest.mark.parametrize(
    "name",
    ["Glides", "free-speech-1", "Rainbow Passage", "Cape V sentences", "Word-color Stroop"],
)
def test_real_tasks_are_untouched(name):
    assert not is_audio_check(name)


@pytest.mark.parametrize("empty", [None, "", "   ", "nan", "NaN", "None", float("nan")])
def test_missing_names_are_not_audio_checks(empty):
    """Callers pass raw RedCap cells, which are freely NaN; that is not an audio check."""
    assert not is_audio_check(empty)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Audio Check (v2)-1", "audio-check-v2-1"),                       # parentheses
        ("Diadochokinesis (v2)-puhtuhkuh", "diadochokinesis-v2-puhtuhkuh"),
        ("Conversation (6 plus)-favorite_food", "conversation-6-plus-favorite-food"),  # underscores
        ("Prolonged Vowel", "prolonged-vowel"),                           # case
        ("Prolonged vowel", "prolonged-vowel"),
        ("Harvard Sentences-List 38-1", "harvard-sentences-list-38-1"),
        ("Cape V sentences (v2)-1", "cape-v-sentences-v2-1"),             # canonical order
        ("Cape V sentences-1 (v2)", "cape-v-sentences-v2-1"),             # curated alias
        ("  Loudness (v2) ", "loudness-v2"),                              # surrounding whitespace
    ],
)
def test_canonical_task_entity(raw, expected):
    assert canonical_task_entity(raw) == expected


def test_canonical_task_entity_is_idempotent_and_deidentify_safe():
    for raw in ["Audio Check (v2)-1", "Cape V sentences-1 (v2)", "Respiration and cough-FiveBreaths-4"]:
        entity = canonical_task_entity(raw)
        assert canonical_task_entity(entity) == entity
        stem = f"sub-abc_ses-DEF-123_task-{entity}"
        # deidentify sanitizes the task entity of every stem; an ingest-normalized stem is a no-op
        assert sanitize_task_entity_in_bids_stem(stem) == stem


def test_entity_has_no_forbidden_characters():
    for raw in ["Audio Check (v2)-1", "Conversation (6 plus)-favorite_food", "Cape V sentences-1 (v2)"]:
        entity = canonical_task_entity(raw)
        assert entity == entity.lower()
        assert not any(ch in entity for ch in "() _"), entity
        assert "--" not in entity and not entity.startswith("-") and not entity.endswith("-")


def test_missing_alias_resource_raises(monkeypatch):
    """A missing packaged alias file must fail, not silently rename every file."""
    import b2aiprep.prepare.utils as utils

    load_recording_name_aliases.cache_clear()
    monkeypatch.setattr(
        utils, "_RECORDING_NAME_ALIASES", ("prepare", "resources", "task_registry", "absent.json")
    )
    try:
        with pytest.raises(FileNotFoundError):
            load_recording_name_aliases()
    finally:
        load_recording_name_aliases.cache_clear()
