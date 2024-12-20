import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
import requests
import responses

from b2aiprep.prepare.phenotype import (
    get_activity_schema_path,
    get_all_schema_paths,
    get_reproschema_raw_url,
    is_url_resolvable,
    populate_data_element,
    search_string_in_json_files,
    update_phenotype_jsons,
)
from b2aiprep.prepare.resources.phenotype_json_descriptions import (
    PHENOTYPE_JSON_FILE_DESCRIPTIONS,
)

B2AI_REDCAP2RS_ACTIVITIES_DIR = (
    "/Users/isaacbevers/sensein/reproschema-wrapper/b2ai-redcap2rs/activities"
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


def test_update_phenotype_jsons():
    """Test update_phenotype_jsons to ensure JSON files are generated correctly."""
    # Set up a temporary phenotype directory with test data
    real_phenotype_dir = os.path.abspath(
        "src/b2aiprep/prepare/resources/b2ai-data-bids-like-template/phenotype"
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        phenotype_dir = os.path.join(temp_dir, "phenotype")
        # Ensure destination directory exists
        os.makedirs(phenotype_dir)

        activities_dir = B2AI_REDCAP2RS_ACTIVITIES_DIR
        file_descriptions = PHENOTYPE_JSON_FILE_DESCRIPTIONS

        # Copy structure and create empty files
        for item in Path(real_phenotype_dir).iterdir():
            if ".json" in str(item) and "<measurement_tool_name>.json" not in str(item):
                item = Path(item)  # Ensure item is a Path object
                if item.is_dir():
                    # Create the directory structure in the destination
                    Path(phenotype_dir).joinpath(item.name).mkdir(parents=True, exist_ok=True)
                else:
                    # Create a valid JSON file with the same name in the destination
                    json_file_path = Path(phenotype_dir).joinpath(item.name)
                    with json_file_path.open("w") as json_file:
                        json.dump({}, json_file)  # Write an empty JSON object

        update_phenotype_jsons(activities_dir, file_descriptions, phenotype_dir=phenotype_dir)

        generated_phenotype_json_paths = [
            os.path.join(phenotype_dir, item) for item in os.listdir(phenotype_dir)
        ]

        generated_phenotype_json_dicts = []
        for json_file_path in generated_phenotype_json_paths:
            with Path(json_file_path).open("r") as json_file:
                generated_phenotype_json_dicts.append(json.load(json_file))

        for i, generated_phenotype_json_dict in enumerate(generated_phenotype_json_dicts):
            assert "" in generated_phenotype_json_dict, generated_phenotype_json_paths[i]
            generated_phenotype_json_dict = generated_phenotype_json_dict[""]
            assert "data_elements" in generated_phenotype_json_dict
            assert "description" in generated_phenotype_json_dict
            assert generated_phenotype_json_dict["description"] != ""
            assert "url" in generated_phenotype_json_dict

        # Check all real JSON file names exist in the phenotype directory
        # generated_files = os.listdir(phenotype_dir)
        # for file_name in real_json_files:
        #     assert file_name in generated_files, f"Missing file: {file_name}"

        #     # Validate contents of each JSON file
        #     generated_path = os.path.join(phenotype_dir, file_name)
        #     with open(generated_path, "r", encoding="utf-8") as generated_file:
        #         generated_content = json.load(generated_file)
        #         assert "description" in generated_content
        #         assert generated_content["description"] == file_descriptions[file_name]
        #         assert "key" in generated_content["data_elements"]


class TestPopulateDataElement(unittest.TestCase):
    def setUp(self):
        # Create a temporary reproschema item file for testing
        self.temp_file = tempfile.NamedTemporaryFile(
            delete=False, mode="w", encoding="utf-8", suffix=".json"
        )
        self.reproschema_item = {
            "question": "What is your age?",
            "responseOptions": {
                "valueType": "integer",
                "maxValue": 120,
                "minValue": 0,
                "choices": [{"value": 1, "label": "Option 1"}, {"value": 2, "label": "Option 2"}],
            },
        }
        json.dump(self.reproschema_item, self.temp_file)
        self.temp_file.close()
        self.output_phenotype_dict = {}

    def tearDown(self):
        # Clean up temporary file
        try:
            os.remove(self.temp_file.name)
        except FileNotFoundError:
            pass

    @patch(
        "b2aiprep.prepare.phenotype.get_reproschema_raw_url",
        return_value="http://example.com/reproschema",
    )
    def test_populate_data_element(self, mock_get_reproschema_raw_url):
        # Import the function to test

        # Call the function with the test data
        key = "test_key"
        phenotype_file_name = "test_phenotype.json"
        populate_data_element(
            self.output_phenotype_dict, key, self.temp_file.name, phenotype_file_name
        )

        # Verify the output
        self.assertIn(key, self.output_phenotype_dict)
        entry = self.output_phenotype_dict[key]
        self.assertEqual(entry["question"], self.reproschema_item["question"])
        self.assertEqual(entry["datatype"], self.reproschema_item["responseOptions"]["valueType"])
        self.assertEqual(entry["maxValue"], self.reproschema_item["responseOptions"]["maxValue"])
        self.assertEqual(entry["minValue"], self.reproschema_item["responseOptions"]["minValue"])
        self.assertEqual(entry["choices"], self.reproschema_item["responseOptions"]["choices"])
        self.assertEqual(entry["termURL"], "http://example.com/reproschema")


def test_get_all_schema_paths():
    """Test retrieving all schema file paths within a directory."""
    # Create a temporary directory structure
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test files
        schema_file_1 = os.path.join(temp_dir, "file1.schema")
        schema_file_2 = os.path.join(temp_dir, "subdir", "file2.schema")
        non_schema_file = os.path.join(temp_dir, "file3.txt")
        subdir = os.path.join(temp_dir, "subdir")
        os.makedirs(subdir, exist_ok=True)

        # Write files
        open(schema_file_1, "w").close()
        open(schema_file_2, "w").close()
        open(non_schema_file, "w").close()

        # Run the function
        result = get_all_schema_paths(temp_dir)

        # Assert that only schema files are included
        assert schema_file_1 in result
        assert schema_file_2 in result
        assert non_schema_file not in result

        # Assert the correct number of schema files
        assert len(result) == 2


def create_activity_structure(base_path, activity_name, schema_files=None):
    """Helper function to create a test activity structure with schema files."""
    activity_dir = os.path.join(base_path, "activities", activity_name)
    os.makedirs(activity_dir, exist_ok=True)
    if schema_files:
        for schema_file in schema_files:
            with open(os.path.join(activity_dir, schema_file), "w") as f:
                f.write("dummy schema content")
    return activity_dir


def test_get_activity_schema_path_single_schema():
    """Test retrieving a single schema file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test activity structure
        activity_name = "test_activity"
        create_activity_structure(temp_dir, activity_name, schema_files=["activity.schema"])

        # Construct item path
        item_path = os.path.join(temp_dir, "activities", activity_name, "item.json")

        # Test function
        result = get_activity_schema_path(item_path)
        expected_path = os.path.join(temp_dir, "activities", activity_name, "activity.schema")
        assert result == expected_path


def test_get_activity_schema_path_no_schema():
    """Test raising an error when no schema files are found."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test activity structure without schemas
        activity_name = "test_activity"
        create_activity_structure(temp_dir, activity_name)

        # Construct item path
        item_path = os.path.join(temp_dir, "activities", activity_name, "item.json")

        # Test function
        with pytest.raises(ValueError, match="Wrong number of schema paths: 0"):
            get_activity_schema_path(item_path)


def test_get_activity_schema_path_multiple_schemas():
    """Test raising an error when multiple schema files are found."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test activity structure with multiple schemas
        activity_name = "test_activity"
        create_activity_structure(
            temp_dir, activity_name, schema_files=["schema1.schema", "schema2.schema"]
        )

        # Construct item path
        item_path = os.path.join(temp_dir, "activities", activity_name, "item.json")

        # Test function
        with pytest.raises(ValueError, match="Wrong number of schema paths: 2"):
            get_activity_schema_path(item_path)
