import json
import os
from typing import Dict, List


def _transform_str_for_bids_filename(filename: str):
    """Replace spaces in a string with hyphens to match BIDS string format rules.."""
    return filename.replace(" ", "-")


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
    print("HELLO")
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
