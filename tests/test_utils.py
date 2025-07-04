import json
import os
import tempfile
from pathlib import Path
import pandas as pd
import pytest

from b2aiprep.prepare.utils import (
    copy_package_resource,
    make_tsv_files,
    reformat_resources,
    remove_files_by_pattern,
    fetch_json_options_number,
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


def test_copy_package_resource():
    with tempfile.TemporaryDirectory() as temp_dir:
        package_name = "b2aiprep.prepare.resources.b2ai-data-bids-like-template"
        resource_name = "phenotype"
        destination_dir = temp_dir
        destination_name = "copied_resource"

        # Directly copy the resource without using resource_path
        copy_package_resource(package_name, resource_name, destination_dir, destination_name)

        # Create the destination path
        destination_path = os.path.join(destination_dir, destination_name)

        # Check if the resource was copied correctly
        assert os.path.exists(destination_path)


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

def test_fetch_json_options_number():   
    json_url = "https://raw.githubusercontent.com/sensein/b2ai-redcap2rs/d056c9311cb682148f1ce46bd0fe4b9e43c851a0/activities/q_generic_gad7_anxiety/items/hard_to_sit_still"
    actual = fetch_json_options_number(json_url)
    expected_answer = 4
    assert actual == expected_answer

def test_get_wav_duration():
    b2ai_wav_path = "data/test_audio2.wav"
    actual = get_wav_duration(b2ai_wav_path)
    expected  = 2.9257142857142857
    assert actual == expected

if __name__ == "__main__":
    pytest.main()
