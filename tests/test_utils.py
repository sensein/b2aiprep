import json
import os
import tempfile

import pandas as pd
import pytest

from b2aiprep.prepare.utils import (
    construct_all_tsvs_from_jsons,
    construct_tsv_from_json,
    copy_package_resource,
    make_tsv_files,
    reformat_resources,
    remove_files_by_pattern,
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


def test_construct_tsv_from_json():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a sample DataFrame
        data = {"record_id": [1, 2, 3, 4], "col1": ["A", "B", "C", "D"], "col2": [10, 20, 30, 40]}
        df = pd.DataFrame(data)

        # Create a sample JSON file
        json_data = {"record_id": {"description": ""}, "col1": {"description": ""}}
        json_path = os.path.join(temp_dir, "sample.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f)

        # Run the function
        construct_tsv_from_json(df, json_path, temp_dir)

        # Check if the TSV file is created
        tsv_path = os.path.join(temp_dir, "sample.tsv")
        assert os.path.exists(tsv_path)

        # Load TSV file and check content
        result_df = pd.read_csv(tsv_path, sep="\t")
        assert list(result_df.columns) == ["record_id", "col1"]
        assert result_df.shape == (4, 2)  # Check number of rows and columns


def test_construct_all_tsvs_from_jsons():
    with tempfile.TemporaryDirectory() as temp_dir, tempfile.TemporaryDirectory() as output_dir:
        # Create a sample DataFrame
        data = {"record_id": [1, 2, 3, 4], "col1": ["A", "B", "C", "D"], "col2": [10, 20, 30, 40]}
        df = pd.DataFrame(data)

        # Create multiple sample JSON files
        json_data_1 = {"record_id": {"description": ""}, "col1": {"description": ""}}
        json_path_1 = os.path.join(temp_dir, "sample1.json")
        with open(json_path_1, "w") as f:
            json.dump(json_data_1, f)

        json_data_2 = {"record_id": {"description": ""}, "col2": {"description": ""}}
        json_path_2 = os.path.join(temp_dir, "sample2.json")
        with open(json_path_2, "w") as f:
            json.dump(json_data_2, f)

        # Create an excluded file that should not be processed
        json_data_excluded = {"record_id": {"description": ""}, "excluded_col": {"description": ""}}
        excluded_json_path = os.path.join(temp_dir, "excluded.json")
        with open(excluded_json_path, "w") as f:
            json.dump(json_data_excluded, f)

        # Run the function
        construct_all_tsvs_from_jsons(
            df, input_dir=temp_dir, output_dir=output_dir, excluded_files=["excluded.json"]
        )

        # Check if the TSV files are created
        tsv_path_1 = os.path.join(output_dir, "sample1.tsv")
        tsv_path_2 = os.path.join(output_dir, "sample2.tsv")
        excluded_tsv_path = os.path.join(output_dir, "excluded.tsv")

        assert os.path.exists(tsv_path_1)
        assert os.path.exists(tsv_path_2)
        assert not os.path.exists(excluded_tsv_path)  # Ensure the excluded file was not processed

        # Load TSV files and check content
        result_df_1 = pd.read_csv(tsv_path_1, sep="\t")
        assert list(result_df_1.columns) == ["record_id", "col1"]
        assert result_df_1.shape == (4, 2)  # Check number of rows and columns

        result_df_2 = pd.read_csv(tsv_path_2, sep="\t")
        assert list(result_df_2.columns) == ["record_id", "col2"]
        assert result_df_2.shape == (4, 2)  # Check number of rows and columns


if __name__ == "__main__":
    pytest.main()
