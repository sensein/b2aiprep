"""Utility functions for converting participant data (usually nested dictionaries)
to FHIR format.
"""

import json
import re
import typing as t
from collections import OrderedDict
from functools import lru_cache
from importlib.resources import files
import logging
_logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def _load_stimulus_bank(bank_name: str) -> tuple:
    """Load an ordered stimulus bank (list of strings) from prepare/resources.

    Cached so the resource is read once per bank, not once per recording.
    Returns a tuple so the result is hashable/immutable.
    """
    resource = files("b2aiprep.prepare.resources").joinpath(f"{bank_name}.json")
    data = json.loads(resource.read_text())
    return tuple(data["words"])


def _resolve_prompt_ref(task_name: str, prompt_ref: dict) -> t.List[str]:
    """Resolve a task name to its specific stimulus via a stimulus bank.

    Supports both task-name forms that occur in the data:
    - numeral form (e.g. "repeat-words-24" / "repeat_words_24") -> bank[index-1]
    - word form   (e.g. "repeating-words-slice")                -> the word itself,
      validated against the bank.
    The trailing token (after the last '-' or '_') carries the index or word.
    Returns [] when the token can't be resolved (e.g. a bare task name).
    """
    words = _load_stimulus_bank(prompt_ref["bank"])
    token = re.split(r"[-_]", task_name)[-1]
    if token.isdigit():
        idx = int(token)
        if 1 <= idx <= len(words):
            return [words[idx - 1]]
        return []
    for word in words:
        if word.lower() == token.lower():
            return [word]
    return []


def extract_items(participant_json: dict, outline: list) -> t.List[dict]:
    """Iterates over questions specified in the outline and extracts them from the data JSON.

    Items are output as a list of dictionaries, following the FHIR QuestionnaireResponse format
    for a single item.

    NOTE: Due to issues in the conversion, this function is additionally doing data cleaning.
    This includes mapping binary responses to the "valueBinary" field, and parsing instances of
    the "nan" string as missing data.

    Parameters
    ----------
    participant_json : dict
        The JSON data for the participant.
    outline : list
        The list of questions to extract.

    Returns
    -------
    list
        The extracted items.
    """
    items = []
    for question in outline:
        answer_value = str(participant_json[question])
        if answer_value in ("Checked", "Unchecked"):
            answer = answer_value == "Checked"
        elif answer_value == "nan":
            answer = None
        else:
            answer = answer_value.replace("_", "-")
        item = {
            "metadata": question,
            "answer": answer,
        }
        items.append(item)
    return items


def is_invalid_response(participant: dict, outline) -> bool:
    """Determines whether a questionnaire contains only NaN/unchecked responses.
    This indicates that the questionnaire was likely unpopulated by the participant.

    Parameters
    ----------
    participant : dict
        The participant data.
    outline : list
        The questionnaire outline.

    Returns
    -------
    bool
        Whether the response is invalid.
    """
    generic_items = extract_items(participant, outline)
    answers = []
    for item in generic_items:
        if item["answer"] is None:
            continue
        
        if isinstance(item["answer"], str):
            answers.append(item["answer"].lower())
        elif isinstance(item["answer"], bool):
            if item["answer"]:
                answers.append("checked")
            else:
                answers.append("unchecked")
        else:
            answers.append(item["answer"])
            
    answers = set(answers)
    return len(answers.difference({"nan", "unchecked"})) == 0


def convert_response_to_bids_metadata( participant: dict,
    # repeat_instrument: RepeatInstrument
    questionnaire_name: str,
    mapping_name: str,
    columns: t.List[str],
    audio_task_descriptions: OrderedDict,
) -> dict:
    """Converts a participant's response to a metadata json file.

    Given a dictionary of individual data, the function:
    1. identifies the audio task, recording id, duration, and task instructions

    Parameters
    ----------
    participant : dict
        The participant data.
    questionnaire_name : str
        The name of the questionnaire.

    Returns
    -------
    dict containing the associated metadata

    Raises
    ------
    ValueError
        If the parsed JSON does not adhere to the QuestionnaireResponse data structure.
        Usually due to missing or invalid attributes.
    """
    participant_id = participant["record_id"]

    generic_items = extract_items(participant, columns)
    if is_invalid_response(participant, columns):
        generic_items = []

    # determine if it's an acoustic task or recording and grab task names
    linkid_to_find = None
    if mapping_name == "acoustictaskschema":
        linkid_to_find = "acoustic_task_id"
    elif mapping_name == "recordingschema":
        linkid_to_find = "recording_id"
    
    if not linkid_to_find:
        _logger.warning(f"File is missing acoustic_task_id and recording_id, skipping....")
        return {}
    
    metadata_file = {}
    task_name = ""
    metadata_file["instructions"] = ""
    for item in generic_items:
        metadata_field = item.get("metadata")
        metadata_value = item.get("answer")
        if "session_id" in metadata_field:
            metadata_file["session_id"] = metadata_value
        elif metadata_field in ("acoustic_task_name", "recording_name") and metadata_field is not None:
            if isinstance(metadata_value, str):
                task_name = metadata_value.replace(" ", "-").lower()
            metadata_file[metadata_field] = task_name
        elif metadata_file is not None and metadata_field is not None:
            metadata_file[metadata_field] = metadata_value
    
    if "recording_name" in metadata_file:
        metadata_file["task_name"] = metadata_file["recording_name"]
        metadata_file.pop("recording_name")
    # Resolve the task description. Prefer an exact match on the task name;
    # otherwise fall back to the longest description key that is a substring of
    # the task name. Numbered instances of a task (e.g. "picture-12",
    # "productive-Vocabulary-3") intentionally share a single key ("picture",
    # "productive-Vocabulary") through this substring fallback. Selecting the
    # longest match rather than the first makes the lookup independent of key
    # ordering, so e.g. "harvard-sentences-list-1-10" resolves to its own key
    # instead of being shadowed by the shorter "harvard-sentences-list-1-1".
    if task_name:
        task_name_lower = task_name.lower()
        best_task = None
        for task in audio_task_descriptions:
            task_lower = task.lower()
            if task_lower == task_name_lower:
                best_task = task
                break
            if task_lower in task_name_lower and (
                best_task is None or len(task) > len(best_task)
            ):
                best_task = task
        if best_task is not None:
            description = audio_task_descriptions[best_task]
            # Follow alias_of pointers so variant task names (e.g.
            # generative-naming-task-animals -> naming-animals) reuse a single
            # canonical entry without duplicating content. Guarded against cycles.
            seen_aliases = set()
            while isinstance(description, dict) and "alias_of" in description:
                target = description["alias_of"]
                if target in seen_aliases or target not in audio_task_descriptions:
                    break
                seen_aliases.add(target)
                description = audio_task_descriptions[target]
            metadata_file["instructions"] = description["instructions"]
            prompt_ref = description.get("prompt_ref")
            if prompt_ref:
                # Per-item stimulus resolved from a bank by index or word in the
                # task name (e.g. repeat-words-24 / repeating-words-slice -> "slice").
                resolved = _resolve_prompt_ref(task_name_lower, prompt_ref)
                metadata_file["prompts"] = resolved if resolved else description.get("prompts", [])
            else:
                metadata_file["prompts"] = description.get("prompts", [])


    metadata_file.update({"audio_channel_count": 1, "audio_sample_rate": "16000"})

    return metadata_file
