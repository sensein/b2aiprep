import copy
import json
import os

import requests

from b2aiprep.prepare.resources.phenotype_json_descriptions import (
    PHENOTYPE_JSON_FILE_DESCRIPTIONS,
)

B2AI_REDCAP2RS_ACTIVITIES_DIR = (
    "/Users/isaacbevers/sensein/reproschema-wrapper/b2ai-redcap2rs/activities"
)

CHECKSUM = "24cbb461c2ff6556f047dd4bc1275b4b08d52eb8"


def search_string_in_json_files(directory, search_string):
    """Searches JSON files in a directory for a specific string.

    Args:
        directory (str): Path of the directory to search.
        search_string (str): String to look for in JSON content or filenames.

    Returns:
        list[str]: List of file paths containing the search string.
    """
    matching_files = []

    # Walk through all files in the directory and its subdirectories
    for root, _, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            try:
                # Open and parse the file as JSON
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                # Check if the search string is in the JSON content or file name
                if search_string in str(data) or search_string in file_name:
                    matching_files.append(file_path)
            except (json.JSONDecodeError, FileNotFoundError, IOError, UnicodeDecodeError):
                continue

    return matching_files


def is_url_resolvable(url):
    """Checks if a URL can be accessed.

    Args:
        url (str): The URL to test.

    Returns:
        bool: True if the URL is accessible, False otherwise.
    """
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_reproschema_raw_url(path, checksum=CHECKSUM, branch="main"):
    """Generates a raw GitHub URL for a given file path.

    Args:
        path (str): File path in the project.
        checksum (str): Checksum of the file (default is provided value).
        branch (str): Git branch name (default is 'main').

    Returns:
        str | bool: The raw GitHub URL if resolvable, False otherwise.
    """
    path = path.split("/b2ai-redcap2rs/", 1)[-1]
    url = f"https://raw.githubusercontent.com/sensein/b2ai-redcap2rs/{checksum}/{path}"
    if is_url_resolvable(url):
        return url
    else:
        return False


def populate_data_element(output_phenotype_dict, key, item_file_path, phenotype_file_name):
    """Populates phenotype data from a JSON item file into a dictionary.

    Args:
        output_phenotype_dict (dict): Dictionary to populate with phenotype data.
        key (str): Key for the phenotype entry in the dictionary.
        item_file_path (str): Path to the JSON file.
        phenotype_file_name (str): Name of the phenotype file.

    Returns:
        None: The dictionary is modified in place.
    """
    with open(item_file_path, "r", encoding="utf-8") as file:
        reproschema_item = json.load(file)
    if key not in output_phenotype_dict:
        output_phenotype_dict[key] = {}
    if "question" in reproschema_item:
        output_phenotype_dict[key]["question"] = reproschema_item["question"]

    output_phenotype_dict[key]["datatype"] = reproschema_item["responseOptions"]["valueType"]
    if "maxValue" in reproschema_item["responseOptions"]:
        output_phenotype_dict[key]["maxValue"] = reproschema_item["responseOptions"]["maxValue"]
    if "minValue" in reproschema_item["responseOptions"]:
        output_phenotype_dict[key]["minValue"] = reproschema_item["responseOptions"]["minValue"]
    if "valueType" in reproschema_item["responseOptions"]:
        output_phenotype_dict[key]["valueType"] = reproschema_item["responseOptions"]["valueType"]
    if "choices" in reproschema_item["responseOptions"]:
        output_phenotype_dict[key]["choices"] = reproschema_item["responseOptions"]["choices"]
    else:
        output_phenotype_dict[key]["choices"] = None
    reproschema_raw_url = get_reproschema_raw_url(item_file_path)
    if reproschema_raw_url:
        output_phenotype_dict[key]["termURL"] = reproschema_raw_url


def get_all_schema_paths(directory):
    """Finds all file paths ending with 'schema' in a directory.

    Args:
        directory (str): Root directory to search for schema files.

    Returns:
        list[str]: List of paths to schema files.
    """
    schema_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith("schema"):
                schema_paths.append(os.path.join(root, file))
    return schema_paths


def get_activity_schema_path(item_path):
    """Gets the schema path for a given activity.

    Args:
        item_path (str): Path to an item within the activities directory.

    Returns:
        str: Path to the schema file.

    Raises:
        ValueError: If no schema or multiple schema files are found.
    """
    # Determine the activity directory based on the item path
    activity_dir = os.path.join(
        item_path.split("/activities/", 1)[0],
        "activities",
        item_path.split("/activities/", 1)[-1].split("/")[0],
    )

    # Collect all schema file paths within the activity directory
    schema_paths = []
    for root, _, files in os.walk(activity_dir):
        for file in files:
            if file.endswith("schema"):
                schema_paths.append(os.path.join(root, file))

    # Return the schema path if exactly one is found, otherwise raise an error
    if len(schema_paths) == 1:
        return schema_paths[0]
    else:
        raise ValueError(f"Wrong number of schema paths: {len(schema_paths)}")


def process_phenotype_file(
    phenotype_file_name, b2ai_redcap2rs_activities_dir, file_descriptions, phenotype_dir
):
    """Processes a phenotype JSON file and updates its structure.

    Args:
        phenotype_file_name (str): Name of the phenotype JSON file.
        b2ai_redcap2rs_activities_dir (str): Directory containing activity schemas.
        file_descriptions (dict): Descriptions of phenotype files.
        phenotype_dir (str): Directory of phenotype files.

    Returns:
        dict: Updated phenotype data structure.
    """
    file_path = os.path.join(phenotype_dir, phenotype_file_name)
    with open(file_path, "r", encoding="utf-8") as file:
        phenotype_file_dict = json.load(file)

    # Unnest to make output idempotent
    if len(phenotype_file_dict) == 1:
        key = next(iter(phenotype_file_dict))
        if "data_elements" in phenotype_file_dict[key]:
            phenotype_file_dict = phenotype_file_dict[key]["data_elements"]

    activity_schema_path = ""
    output_phenotype_dict = copy.deepcopy(phenotype_file_dict)
    matching_non_item_files = {}
    multiple_item_files = []
    non_matching = []

    # Process each key in the phenotype file
    for key in phenotype_file_dict:
        print(key)
        if "___" in key:
            new_key = key.split("___")[0]
            if new_key not in phenotype_file_dict:
                key = new_key
                output_phenotype_dict[key] = {}

        file_paths = search_string_in_json_files(b2ai_redcap2rs_activities_dir, key)
        if file_paths:
            item_file_paths = [path for path in file_paths if "item" in path]
            if item_file_paths and len(item_file_paths) == 1:
                populate_data_element(
                    output_phenotype_dict, key, item_file_paths[0], phenotype_file_name
                )
                if not activity_schema_path:
                    activity_schema_path = get_activity_schema_path(item_file_paths[0])
            elif item_file_paths and len(item_file_paths) > 1:
                for path in item_file_paths:
                    if os.path.basename(path) == key:
                        if not activity_schema_path:
                            activity_schema_path = get_activity_schema_path(path)
                        populate_data_element(output_phenotype_dict, key, path, phenotype_file_name)
                multiple_item_files.append(item_file_paths)
            else:
                matching_non_item_files[key] = file_paths
        else:
            non_matching.append(key)

    # Update output dictionary with metadata
    activity_schema_name = os.path.basename(activity_schema_path)
    output_phenotype_dict = {
        "description": file_descriptions[phenotype_file_name],
        "url": get_reproschema_raw_url(activity_schema_path),
        "data_elements": output_phenotype_dict,
    }
    return {activity_schema_name: output_phenotype_dict}


def update_phenotype_jsons(
    b2ai_redcap2rs_activities_dir,
    file_descriptions=PHENOTYPE_JSON_FILE_DESCRIPTIONS,
    phenotype_dir=None,
):
    """Processes all phenotype JSON files in a directory.

    Args:
        b2ai_redcap2rs_activities_dir (str): Directory with activity schemas.
        file_descriptions (dict): Mapping of file names to descriptions.
        phenotype_dir (str, optional): Directory of phenotype files.
    """
    if not phenotype_dir:
        phenotype_dir = os.path.abspath(
            "b2aiprep/src/b2aiprep/prepare/resources/b2ai-data-bids-like-template/phenotype"
        )
    for phenotype_file_name in os.listdir(phenotype_dir):
        # Skip non-JSON or template files
        print("PHENOTYPE FILE: ", phenotype_file_name)
        if (
            not phenotype_file_name.endswith(".json")
            or phenotype_file_name == "<measurement_tool_name>.json"
        ):
            continue

        updated_dict = process_phenotype_file(
            phenotype_file_name, b2ai_redcap2rs_activities_dir, file_descriptions, phenotype_dir
        )

        # Save updated JSON
        file_path = os.path.join(phenotype_dir, phenotype_file_name)
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(updated_dict, file, ensure_ascii=False, indent=4)
