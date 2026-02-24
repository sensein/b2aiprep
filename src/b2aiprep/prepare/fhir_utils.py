"""Utility functions for converting participant data (usually nested dictionaries)
to FHIR format.
"""

import json
import typing as t
from importlib.resources import files

from fhir.resources import construct_fhir_element
from fhir.resources.questionnaireresponse import QuestionnaireResponse
import logging
_logger = logging.getLogger(__name__)

def is_empty_questionnaire_response(response: QuestionnaireResponse) -> bool:
    """Determines whether a questionnaire response is empty.

    Parameters
    ----------
    response : QuestionnaireResponse
        The questionnaire response.

    Returns
    -------
    bool
        Whether the response is empty.
    """
    if response.item is None:
        return True

    for item in response.item:
        # skip record_id -> it is always present
        if (item.linkId is not None) and (item.linkId == "record_id"):
            continue
        if item.answer is not None:
            return False

    return True


def create_fhir_questionnaire_response(id: str, name: str, items: list) -> dict:
    """Creates a FHIR QuestionnaireResponse object with default values for the Bridge2AI
    voice biomarker questionnaires. The items provided should already adhere to the FHIR
    format for items in a QuestionnaireResponse.

    Parameters
    ----------
    id : str
        The ID of the response.
    name : str
        The name of the questionnaire.
    items : list
        The items in the response.

    Returns
    -------
    dict
        The FHIR QuestionnaireResponse object.
    """
    fhir_response = {}
    fhir_response["resourceType"] = "QuestionnaireResponse"
    fhir_response["id"] = id
    fhir_response["questionnaire"] = f"https://kind-lab.github.io/vbai-fhir/Questionnaire-{name}"
    fhir_response["status"] = "completed"
    fhir_response["item"] = items
    return fhir_response


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
        
        if item["answer"] is not None:
            answers.append(item["answer"].lower())
        elif isinstance(item["answer"], bool):
            if item["answer"]:
                answers.append("checked")
            else:
                answers.append("unchecked")
        else:
            answers.append(list(answers))
    answers = set(answers)
    return len(answers.difference({"nan", "unchecked"})) == 0


def load_questionnaire_outline(questionnaire_name):
    b2ai_resources = (
        files("b2aiprep").joinpath("prepare").joinpath("resources").joinpath("instrument_columns")
    )
    return json.loads(b2ai_resources.joinpath(f"{questionnaire_name}.json").read_text())

def convert_response_to_bids_metadata( participant: dict,
    # repeat_instrument: RepeatInstrument
    questionnaire_name: str,
    mapping_name: str,
    columns: t.List[str],
    audio_task_descriptions: dict,
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
        return
    
    metadata_file = {}
    task_name = ""
    metadata_file["Instructions"] = ""
    for item in generic_items:
        metadata_field = item.get("metadata")
        metadata_value = item.get("answer")
        if metadata_file is not None and metadata_field is not None:
            metadata_file[metadata_field] = metadata_value
        if metadata_field in ("acoustic_task_name", "recording_name") and metadata_field is not None:
            task_name = metadata_value
    
    for task in audio_task_descriptions:
        if (task_name is not None and task.lower() in task_name.lower()):
            metadata_file["Instructions"] = audio_task_descriptions[task]["instructions"]
            metadata_file["Prompts"] = audio_task_descriptions[task]["prompts"]
    
    metadata_file.update({"audio_channel_count": 1, "audio_sameple_rate": "16000"})
    
    # fhir_id = None
    # for item in generic_items:
    #     if item.get("metadata") == linkid_to_find and item.get("answer"):
    #         fhir_id = item["answer"][0].get("valueString", "")
    #         if fhir_id:
    #             break


    return metadata_file


def convert_response_to_fhir(
    participant: dict,
    # repeat_instrument: RepeatInstrument
    questionnaire_name: str,
    mapping_name: str,
    columns: t.List[str],
) -> QuestionnaireResponse:
    """Converts a participant's response to a FHIR QuestionnaireResponse.

    Given a dictionary of individual data, the function:
    (1) identifies specific data elements using the questionnaire name
    (2) extracts the responses associated with these elements
    (3) reformats the data into a QuestionnaireResponse object and returns it

    Parameters
    ----------
    participant : dict
        The participant data.
    questionnaire_name : str
        The name of the questionnaire.

    Returns
    -------
    QuestionnaireResponse
        The FHIR QuestionnaireResponse object.

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
        return
    fhir_id = None
    for item in generic_items:
        if item.get("linkId") == linkid_to_find and item.get("answer"):
            fhir_id = item["answer"][0].get("valueString", "")
            if fhir_id:
                break

    generic_response = create_fhir_questionnaire_response(
        # create a unique identifier for this FHIR resource by combining
        # the participant ID with the questionnaire name
        id=fhir_id,
        name=mapping_name,
        items=generic_items,
    )

    # to validate that that the generated response conforms to fhir
    questionnaire_response_json = construct_fhir_element(
        "QuestionnaireResponse",
        json.dumps(generic_response),
    )

    return questionnaire_response_json
