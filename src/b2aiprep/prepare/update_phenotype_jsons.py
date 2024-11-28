import json
import os

import requests

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
