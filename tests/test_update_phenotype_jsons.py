import json

import pytest
import requests
import responses

from b2aiprep.prepare.update_phenotype_jsons import (
    get_reproschema_raw_url,
    is_url_resolvable,
    search_string_in_json_files,
)

CHECKSUM = "24cbb461c2ff6556f047dd4bc1275b4b08d52eb8"


@pytest.fixture
def temp_directory_with_files(tmp_path):
    """Create a temporary directory with test files."""
    # JSON file with a matching string
    file_with_match = tmp_path / "file_with_match.json"
    file_with_match.write_text(json.dumps({"key": "search_string"}), encoding="utf-8")

    # JSON file without a matching string
    file_without_match = tmp_path / "file_without_match.json"
    file_without_match.write_text(json.dumps({"key": "other_value"}), encoding="utf-8")

    # Non-JSON file
    non_json_file = tmp_path / "not_a_json.txt"
    non_json_file.write_text("This is not a JSON file.", encoding="utf-8")

    # JSON file where the search string is in the file name
    file_with_match_in_name = tmp_path / "search_string_in_name.json"
    file_with_match_in_name.write_text(json.dumps({"key": "value"}), encoding="utf-8")

    return tmp_path


def test_search_string_found_in_json(temp_directory_with_files):
    """Test that a matching string in a JSON file is found."""
    result = search_string_in_json_files(str(temp_directory_with_files), "search_string")
    assert str(temp_directory_with_files / "file_with_match.json") in result
    assert str(temp_directory_with_files / "file_without_match.json") not in result


def test_search_string_not_found(temp_directory_with_files):
    """Test that no files are returned if the string is not present."""
    result = search_string_in_json_files(str(temp_directory_with_files), "nonexistent_string")
    assert result == []


def test_non_json_file_ignored(temp_directory_with_files):
    """Test that non-JSON files are ignored."""
    result = search_string_in_json_files(str(temp_directory_with_files), "search_string")
    assert str(temp_directory_with_files / "not_a_json.txt") not in result


def test_search_string_in_filename(temp_directory_with_files):
    """Test that a matching string in the file name is found."""
    result = search_string_in_json_files(str(temp_directory_with_files), "search_string")
    assert str(temp_directory_with_files / "search_string_in_name.json") in result


def test_invalid_json_ignored(tmp_path):
    """Test that invalid JSON files are ignored."""
    invalid_json_file = tmp_path / "invalid.json"
    invalid_json_file.write_text("{invalid json", encoding="utf-8")  # Write invalid JSON

    result = search_string_in_json_files(str(tmp_path), "search_string")
    assert str(invalid_json_file) not in result


# Test for is_url_resolvable
@responses.activate
def test_is_url_resolvable_success():
    """Test that is_url_resolvable returns True for a valid URL."""
    test_url = "https://example.com/test"
    responses.add(responses.GET, test_url, status=200)

    assert is_url_resolvable(test_url) is True


@responses.activate
def test_is_url_resolvable_failure():
    """Test that is_url_resolvable returns False for an invalid URL."""
    test_url = "asdfasdf"
    responses.add(responses.GET, test_url, status=404)

    assert is_url_resolvable(test_url) is False


@responses.activate
def test_is_url_resolvable_exception():
    """Test that is_url_resolvable returns False when an exception occurs."""
    test_url = "lkjasdkflaskl"
    responses.add(responses.GET, test_url, body=requests.exceptions.ConnectionError())

    assert is_url_resolvable(test_url) is False


# Test for get_reproschema_raw_url
@responses.activate
def test_get_reproschema_raw_url_success():
    """Test that get_reproschema_raw_url returns a valid URL if it is resolvable."""
    path = "/b2ai-redcap2rs/example_file.json"
    expected_url = (
        f"https://raw.githubusercontent.com/sensein/b2ai-redcap2rs/{CHECKSUM}/example_file.json"
    )
    responses.add(responses.GET, expected_url, status=200)

    assert get_reproschema_raw_url(path, checksum=CHECKSUM) == expected_url


@responses.activate
def test_get_reproschema_raw_url_failure():
    """Test that get_reproschema_raw_url returns False if the URL is not resolvable."""
    path = "/b2ai-redcap2rs/example_file.json"
    expected_url = (
        f"https://raw.githubusercontent.com/sensein/b2ai-redcap2rs/{CHECKSUM}/example_file.json"
    )
    responses.add(responses.GET, expected_url, status=404)

    assert get_reproschema_raw_url(path, checksum=CHECKSUM) is False
