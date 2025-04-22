import glob
import importlib.resources as pkg_resources
import json
import logging
import os
import shutil
import time
import wave
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

_LOGGER = logging.getLogger(__name__)


def reformat_resources(input_dir: str, output_dir: str) -> None:
    """
    Converts lists from all JSON files in the input directory to dictionaries
    where each list item becomes a key in the dictionary with a value of
    {"description": ""}, and saves them to the output directory with the same filename.

    Args:
        input_dir (str): The directory path containing input JSON files with lists.
        output_dir (str): The directory path to save the output JSON dictionary files.

    Raises:
        ValueError: If input_dir and output_dir are the same.
        FileNotFoundError: If input_dir does not exist.
        Exception: If any unexpected error occurs during file processing.

    Returns:
        None
    """
    # Check if input and output directories are the same
    if os.path.abspath(input_dir) == os.path.abspath(output_dir):
        raise ValueError("Input and output directories must be different.")

    # Check if input directory exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            input_json_path = os.path.join(input_dir, filename)
            output_json_path = os.path.join(output_dir, filename)
            print(output_json_path)

            try:
                # Load the list from the input JSON file
                with open(input_json_path, "r") as input_file:
                    keys_list: List[str] = json.load(input_file)
                result_dict: Dict[str, Dict[str, str]] = {
                    key: {"description": ""} for key in keys_list
                }

                # Save the resulting dictionary to the output JSON file
                with open(output_json_path, "w") as output_file:
                    json.dump(result_dict, output_file, indent=4)

            except Exception as e:
                print(f"Error processing file '{filename}': {e}")


def make_tsv_files(directory: str) -> None:
    """
    Creates .tsv files for each .json file present in the specified directory.
    The .tsv files will have columns corresponding to the keys of the JSON files,
    maintaining the order of the keys.

    Args:
        directory (str): The path to the directory containing .json files for which
                         .tsv files need to be created.

    Returns:
        None

    Raises:
        FileNotFoundError: If the specified directory does not exist.
        Exception: If any unexpected error occurs during file creation.
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory '{directory}' does not exist.")

    # Iterate over all files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            base_name = os.path.splitext(filename)[0]
            tsv_filename = f"{base_name}.tsv"
            json_filepath = os.path.join(directory, filename)
            tsv_filepath = os.path.join(directory, tsv_filename)

            try:
                with open(json_filepath, "r") as json_file:
                    data = json.load(json_file)

                # Extract top-level keys from the JSON file to be the columns
                keys = list(data.keys())
                df = pd.DataFrame([{key: "" for key in keys}])

                # Write the DataFrame to a .tsv file
                df.to_csv(tsv_filepath, sep="\t", index=False)

            except Exception as e:
                print(f"Error processing file '{filename}': {e}")


def copy_package_resource(
    package: str, resource: str, destination_dir: str, destination_name: str = None
) -> None:
    """
    Copy a file or directory from within a package to a specified directory,
    overwriting if it already exists.

    Args:
        package (str): The package name where the file or directory is located.
        resource (str): The resource name (file or directory path within the package).
        destination_dir (str): The directory where the file or directory should be copied.
        destination_name (str, optional): The new name for the copied file or directory.
        If not provided, the original name is used.

    Returns:
        None
    """
    if destination_name is None:
        destination_name = os.path.basename(resource)
    destination_path = os.path.join(destination_dir, destination_name)

    # Check if the destination path already exists and remove it if so
    if os.path.exists(destination_path):
        if os.path.isdir(destination_path):
            shutil.rmtree(destination_path)
        else:
            os.remove(destination_path)

    with pkg_resources.path(package, resource) as src_path:
        src_path = str(src_path)  # Convert to string to avoid issues with path-like objects
        if os.path.isdir(src_path):  # Check if the resource is a directory
            shutil.copytree(src_path, destination_path)
        else:  # Otherwise, assume it is a file
            shutil.copy(src_path, destination_path)


def remove_files_by_pattern(directory: str, pattern: str) -> None:
    """
    Remove all files in a given directory that match a specific pattern.

    Args:
        directory (str): The path to the directory.
        pattern (str): The pattern to match files (e.g., "*.txt" for all text files).

    Returns:
        None
    """
    # Construct the full path pattern
    path_pattern = os.path.join(directory, pattern)

    # Use glob to find all files that match the pattern
    files_to_remove = glob.glob(path_pattern)

    for file_path in files_to_remove:
        try:
            os.remove(file_path)
            print(f"Removed file: {file_path}")
        except OSError as e:
            print(f"Error removing file {file_path}: {e}")


def initialize_data_directory(bids_dir_path: str) -> None:
    """Initializes the data directory using the template.

    Args:
        bids_dir_path (str): The path to the BIDS directory where the data should be initialized.

    Returns:
        None
    """
    if not os.path.exists(bids_dir_path):
        os.makedirs(bids_dir_path)
        _LOGGER.info(f"Created directory: {bids_dir_path}")

    template_package = "b2aiprep.prepare.resources.b2ai-data-bids-like-template"
    copy_package_resource(template_package, "CHANGELOG.md", bids_dir_path)
    copy_package_resource(template_package, "README.md", bids_dir_path)
    copy_package_resource(template_package, "dataset_description.json", bids_dir_path)
    copy_package_resource(template_package, "participants.json", bids_dir_path)
    copy_package_resource(template_package, "phenotype", bids_dir_path)
    phenotype_path = Path(bids_dir_path).joinpath("phenotype")
    remove_files_by_pattern(phenotype_path, "<measurement_tool_name>*")


def open_json_as_dict(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


# Updated function to take file paths as input and perform the comparison
def compare_json_keys_only_differences_from_files(file1_path, file2_path):
    # Load JSON files as dictionaries
    json1 = open_json_as_dict(file1_path)
    json2 = open_json_as_dict(file2_path)

    # Get keys from both JSON objects
    keys1 = set(json1.keys())
    keys2 = set(json2.keys())

    # Find differences
    only_in_file1 = keys1 - keys2
    only_in_file2 = keys2 - keys1

    result = {"only_in_file1": list(only_in_file1), "only_in_file2": list(only_in_file2)}

    return result


def sort_json_by_another_and_save(
    json_to_sort: Dict[str, any], reference_json: Dict[str, any], output_file_path: str
) -> None:
    # Create an ordered list of keys from the reference JSON
    reference_keys = list(reference_json.keys())

    # Sort the json_to_sort according to the reference keys order
    sorted_json = {key: json_to_sort[key] for key in reference_keys if key in json_to_sort}

    # Save the sorted JSON to the output file
    with open(output_file_path, "w") as output_file:
        json.dump(sorted_json, output_file, indent=4)

    print(f"Sorted JSON saved to {output_file_path}")


def save_json(data: Dict[str, any], file_path: str) -> None:
    """Helper function to save a dictionary as a JSON file with correct character rendering."""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def update_jsons_in_directory(directory: str, reference_file: str) -> None:
    """Function to load all JSON files in a directory, replace their values
    with values from a reference JSON file based on matching keys, and remove
    those entries from the reference JSON at the end, excluding <measurement_tool_name>.json."""

    # Load reference JSON
    reference_json = open_json_as_dict(reference_file)

    # Track the keys that have been updated
    updated_keys: List[str] = []

    # Iterate over all files in the directory
    for file_name in os.listdir(directory):
        # Exclude <measurement_tool_name>.json
        if file_name.endswith(".json") and file_name != "<measurement_tool_name>.json":
            file_path = os.path.join(directory, file_name)

            # Load the current JSON file as a dict
            json_data = open_json_as_dict(file_path)

            # Replace values in the json_data with the ones from reference_json for matching keys
            updated_json = {}
            for key in json_data.keys():
                if key in reference_json:
                    updated_json[key] = reference_json[key]  # Use the value from reference JSON
                    updated_keys.append(key)  # Track the key to be removed later
                else:
                    raise KeyError(f"Key '{key}' in {file_name} not found in reference JSON.")

            # Save the updated JSON file
            save_json(updated_json, file_path)
            print(f"Updated: {file_name}")

    # Remove all updated keys from the reference JSON
    for key in updated_keys:
        reference_json.pop(key, None)

    # Save the modified reference JSON back to its original file
    save_json(reference_json, reference_file)
    print(f"Updated reference JSON saved back to {reference_file}")


def retry(func, max_retries=3, delay=0.5, exceptions=(Exception,)):
    """
    Retry a function n times with a specified delay.

    Args:
        func: The function to be called.
        max_retries: The maximum number of retries (default: 3).
        delay: The delay between retries in seconds (default: 1).
        exceptions: A tuple of exceptions to retry on (default: all exceptions).
    """

    def wrapper(*args, **kwargs):
        attempts = 0
        while attempts < max_retries:
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                attempts += 1
                if attempts < max_retries:
                    print(
                        f"Retry attempt {attempts} failed due to {e}. Retrying in {delay} seconds..."
                    )
                    time.sleep(delay)
                else:
                    raise

    return wrapper


def fetch_json_options_number(raw_url):
    """
    Function to retrieve how many options a specific reproschema item contains.

    Args:
        raw_url: The github raw url to the given reproschema item.
    """
    try:
        # fix url due to the split
        raw_url = raw_url.replace("combined", "questionnaires")
        # Make a GET request to the raw URL
        response = requests.get(raw_url, verify=True)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)

        # Parse the JSON data
        json_data = response.json()
        return len(json_data["responseOptions"]["choices"])

    except requests.exceptions.RequestException as e:
        _LOGGER.info(f"Error fetching data: {e}")
        return
    except ValueError:
        _LOGGER.info("Error parsing JSON data")


def get_wav_duration(file_path):
    """
    Function to retrieve duration of .wav audio file
    Args:
        file_path: The url to the given audio files.
    """
    with wave.open(file_path, "rb") as audio_file:
        # Get the total number of frames
        frames = audio_file.getnframes()
        # Get the frame rate (samples per second)
        rate = audio_file.getframerate()
        # Calculate duration in seconds
        duration = frames / float(rate)
    return duration
