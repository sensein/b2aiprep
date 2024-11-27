import json
import os

import requests

B2AI_REDCAP2RS_ACTIVITIES_DIR = (
    "/Users/isaacbevers/sensein/reproschema-wrapper/b2ai-redcap2rs/activities"
)

CHECKSUM = "24cbb461c2ff6556f047dd4bc1275b4b08d52eb8"

FILE_DESCRIPTIONS = {
    "laryngealDystonia.json": "Measures symptoms and characteristics of laryngeal dystonia.",
    "demographics.json": "Measures participant background details.",
    "adhd.json": "Measures attention-related behaviors and symptoms.",
    "airwaystenosis.json": "Measures the severity and characteristics of airway stenosis.",
    "als.json": "Measures symptoms and progression of ALS.",
    "alzheimers.json": "Measures cognitive decline and related symptoms.",
    "benignLesion.json": "Measures features of benign lesions.",
    "bipolar.json": "Measures symptoms and behaviors related to bipolar disorder.",
    "confounders.json": "Measures variables that could impact study outcomes.",
    "customAffectScale.json": "Measures emotional states.",
    "depression.json": "Measures severity and impact of depressive symptoms.",
    "dsm5.json": "Measures criteria according to DSM-5 standards.",
    "dyspnea.json": "Measures the presence and severity of dyspnea.",
    "eligibility.json": "Measures participant eligibility for the study.",
    "enrollment.json": "Measures participant registration details.",
    "gad7.json": "Measures severity of generalized anxiety.",
    "laryngealCancer.json": "Measures characteristics of laryngeal cancer.",
    "leicester.json": "Measures specific health or psychological attributes.",
    "panas.json": "Measures positive and negative affect.",
    "parkinsons.json": "Measures symptoms and progression of Parkinson's disease.",
    "participant.json": "Measures general study-related information.",
    "phq9.json": "Measures severity of depressive symptoms.",
    "precancerousLesions.json": "Measures features of precancerous lesions.",
    "ptsd.json": "Measures trauma-related symptoms.",
    "random.json": "Measures variables for various study purposes.",
    "stroop.json": "Measures cognitive control and processing speed.",
    "vhi10.json": "Measures perceived impact of voice disorders.",
    "vocab.json": "Measures language and word knowledge.",
    "vocalFoldParalysis.json": "Measures characteristics of vocal fold paralysis.",
    "voicePerception.json": "Measures how participants perceive voice quality.",
    "voiceSeverity.json": "Measures the impact and seriousness of voice disorders.",
    "winograd.json": "Measures language comprehension and reasoning.",
}


def search_string_in_json_files(directory, search_string):
    """Searches for a string in JSON files within a directory.

    Args:
        directory (str): Path to the directory to search in.
        search_string (str): String to search for in JSON files and filenames.

    Returns:
        list: List of file paths where the search string is found.
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
            except (json.JSONDecodeError, FileNotFoundError, IOError):
                # Ignore files that are not JSON or cannot be read
                continue

    return matching_files


def is_url_resolvable(url):
    """
    Checks if the URL is resolvable.

    Parameters:
        url (str): The URL to check.

    Returns:
        bool: True if the URL is resolvable, False otherwise.
    """
    try:
        response = requests.get(url)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def get_reproschema_raw_url(path, checksum=CHECKSUM, branch="main"):
    """
    Generates a raw GitHub URL for a file in the project.

    Parameters:
        path (str): Path to the file in the project.
        checksum (str): The checksum of the file (default is a specific value).
        branch (str): Branch name (default is 'main').

    Returns:
        str: The raw GitHub URL.
    """
    path = path.split("/b2ai-redcap2rs/", 1)[-1]
    url = f"https://raw.githubusercontent.com/sensein/b2ai-redcap2rs/{checksum}/{path}"
    if is_url_resolvable(url):
        return url
    else:
        return False


# def populate_data_element(output_phenotype_dict, key, item_file_path, phenotype_file_name):
#     # print(item_file_path)
#     with open(item_file_path, 'r', encoding='utf-8') as file:
#         reproschema_item = json.load(file)
#     if key not in output_phenotype_dict:
#         output_phenotype_dict[key] = {}
#     if "question" in reproschema_item:
#         output_phenotype_dict[key]["question"] = reproschema_item["question"]

#     output_phenotype_dict[key]["datatype"] = reproschema_item["responseOptions"]["valueType"]
#     if "maxValue" in reproschema_item["responseOptions"]:
#         output_phenotype_dict[key]["maxValue"] = reproschema_item["responseOptions"]["maxValue"]
#     if "minValue" in reproschema_item["responseOptions"]:
#         output_phenotype_dict[key]["minValue"] = reproschema_item["responseOptions"]["minValue"]
#     if "valueType" in reproschema_item["responseOptions"]:
#         output_phenotype_dict[key]["valueType"] = reproschema_item["responseOptions"]["valueType"]
#     if "choices" in reproschema_item["responseOptions"]:
#         output_phenotype_dict[key]["choices"] = reproschema_item["responseOptions"]["choices"]
#     else:
#         output_phenotype_dict[key]["choices"] = None
#     reproschema_raw_url = get_reproschema_raw_url(item_file_path)
#     if reproschema_raw_url:
#         output_phenotype_dict[key]["termURL"] = reproschema_raw_url


# def get_all_schema_paths(directory):
#     schema_paths = []
#     for root, _, files in os.walk(directory):
#         for file in files:
#             if file.endswith('schema'):
#                 schema_paths.append(os.path.join(root, file))
#     return schema_paths


# def get_activity_schema_path(item_path):
#     activity_dir = os.path.join(item_path.split('/activities/', 1)[0], 'activities', item_path.split('/activities/', 1)[-1].split('/')[0])
#     schema_paths = []
#     for root, _, files in os.walk(activity_dir):
#         for file in files:
#             if file.endswith('schema'):
#                 schema_paths.append(os.path.join(root, file))
#     if len(schema_paths) == 1:
#         return schema_paths[0]
#     else:
#         # print(schema_paths)
#         raise ValueError(f"Wrong number of schema paths: {len(schema_paths)}")


# import copy


# matching_non_item_files = {}
# multiple_item_files = []
# non_matching = []

# # Specify the phenotype_dir containing the .json files
# phenotype_dir = "/Users/isaacbevers/sensein/b2ai-wrapper/b2aiprep/src/b2aiprep/prepare/resources/b2ai-data-bids-like-template/phenotype"

# def main():
#     # Loop through each file in the phenotype_dir
#     for phenotype_file_name in os.listdir(phenotype_dir):
#         # Check if the file ends with .json and is not "<measurement_tool_name>.json"
#         if phenotype_file_name.endswith(".json") and phenotype_file_name != "<measurement_tool_name>.json":
#             file_path = os.path.join(phenotype_dir, phenotype_file_name)


#             with open(file_path, 'r', encoding='utf-8') as file:
#                 phenotype_file_dict = json.load(file)

#             # unnest to make output idempotent
#             if len(phenotype_file_dict) == 1:
#                 key = next(iter(phenotype_file_dict))
#                 if "data_elements" in phenotype_file_dict[key]:
#                     phenotype_file_dict = phenotype_file_dict[key]["data_elements"]

#             activity_schema_path = ""
#             output_phenotype_dict = copy.deepcopy(phenotype_file_dict)
#             for key in phenotype_file_dict:
#                 if '___' in key:
#                     new_key = key.split('___')[0]
#                     if new_key not in phenotype_file_dict:
#                         key = new_key
#                         output_phenotype_dict[key] = {}

#                 file_paths = search_string_in_json_files(b2ai_redcap2rs_activities_dir, key)
#                 if file_paths:
#                     item_file_paths = [path for path in file_paths if "item" in path]
#                     if item_file_paths and len(item_file_paths) == 1:
#                         populate_data_element(output_phenotype_dict, key, item_file_paths[0], phenotype_file_name)
#                         if not activity_schema_path:
#                             activity_schema_path = get_activity_schema_path(item_file_paths[0])
#                     elif item_file_paths and len(item_file_paths) > 1:
#                         # select the correct one
#                         for path in item_file_paths:
#                             if os.path.basename(path) == key:
#                                 if not activity_schema_path:
#                                     activity_schema_path = get_activity_schema_path(path)
#                                 populate_data_element(output_phenotype_dict, key, path, phenotype_file_name)
#                         multiple_item_files.append(item_file_paths)
#                     else:
#                         matching_non_item_files[key] = file_paths
#                 else:
#                     non_matching.append(key)
#             print(activity_schema_path)

#             activity_schema_name = os.path.basename(activity_schema_path)
#             output_phenotype_dict = {"data_elements": output_phenotype_dict}
#             output_phenotype_dict["description"] = file_descriptions[phenotype_file_name]
#             output_phenotype_dict["url"] = get_reproschema_raw_url(activity_schema_path)
#             output_phenotype_dict = {
#                     "description": output_phenotype_dict["description"],
#                     "url": output_phenotype_dict["url"],
#                     "data_elements": output_phenotype_dict["data_elements"]
#                 }
#             output_phenotype_dict = {activity_schema_name: output_phenotype_dict}
#             # output_phenotype_dict = dict(sorted(output_phenotype_dict.items()))
#             # output_phenotype_dict = dict(sorted(output_phenotype_dict.items(), key=lambda item: len(str(item[1]))))

#             # TODO
#             # output_phenotype_dict[phenotype_file_name]["url"] =
#             #phenotype_file_dict[phenotype_file_name][data elements] =
#             # if phenotype_file_name not in output_phenotype_dict:
#             #     output_phenotype_dict = {phenotype_file_name: RS ASSESSMENT NAME}
#             # print(output_phenotype_dict)
#             with open(file_path, 'w', encoding='utf-8') as file:
#                 json.dump(output_phenotype_dict, file, ensure_ascii=False, indent=4)
# """
# {
#   [assessment_name]: {
#     "description": [description text],
#     "url": [reproschema_url],
#     "data elements": {
# """
# phenotype_dir = "/Users/isaacbevers/sensein/b2ai-wrapper/b2aiprep/src/b2aiprep/prepare/resources/b2ai-data-bids-like-template/phenotype"

# def count_items_with_only_descriptions():
#     single_entry_fields = {}
#     for phenotype_file_name in os.listdir(phenotype_dir):
#         if phenotype_file_name.endswith(".json") and phenotype_file_name != "<measurement_tool_name>.json":
#             single_entry_fields[phenotype_file_name] = []
#             file_path = os.path.join(phenotype_dir, phenotype_file_name)

#             # Open and load the JSON file
#             with open(file_path, 'r', encoding='utf-8') as file:
#                 phenotype_file_dict = json.load(file)

#             for key in phenotype_file_dict:
#                 if len(phenotype_file_dict[key]) < 2:
#                     single_entry_fields[phenotype_file_name].append(key)

#     single_entry_fields_count = 0
#     for key in single_entry_fields:
#         single_entry_fields_count += len(single_entry_fields[key])
#         single_entry_fields[key] = len(single_entry_fields[key])
#     print(single_entry_fields)
#     print(single_entry_fields_count)
