"""Utility functions for converting participant data (usually nested dictionaries)
to FHIR format.
"""

import json
from importlib.resources import files
import typing as t

from fhir.resources import construct_fhir_element
from fhir.resources.questionnaireresponse import QuestionnaireResponse

from b2aiprep.constants import DATA_COLUMNS, QUESTIONNAIRE_MAPPING

def is_empty_response(fhir_response: dict):
    answers = set(
        [
            x["answer"][0]["valueString"].lower()
            for x in fhir_response["item"]
        ]
    )
    has_nan = "nan" in answers
    has_uncheck = "unchecked" in answers
    return (
        (len(answers) == 0)
        or (len(answers) == 2 and (has_nan and has_uncheck))
        or (len(answers) == 1 and (has_nan or has_uncheck))
    )

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
    fhir_response["questionnaire"] = (
        f"https://kind-lab.github.io/vbai-fhir/Questionnaire-{name}"
    )
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
            answer = [{"valueBoolean": answer_value == "Checked"}]
        elif answer_value == "nan":
            answer = None
        else:
            answer = [{"valueString": answer_value}]
        item = {
            "linkId": question,
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
        if "valueString" in item["answer"][0]:
            if item["answer"][0]["valueString"] is not None:
                answers.append(item["answer"][0]["valueString"].lower())
        elif "valueBoolean" in item["answer"][0]:
            if item["answer"][0]["valueBoolean"]:
                answers.append("checked")
            else:
                answers.append("unchecked")
        else:
            answers.append(list(item["answer"][0].keys())[0])
    answers = set(answers)
    return len(answers.difference({"nan", "unchecked"})) == 0

def load_questionnaire_outline(questionnaire_name):
    b2ai_resources = files("b2aiprep").joinpath("resources").joinpath('instrument_columns')
    return json.loads(b2ai_resources.joinpath(f"{questionnaire_name}.json").read_text())

def convert_response_to_fhir(participant: dict, questionnaire_name: str) -> QuestionnaireResponse:
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
    reg_questionnaire_name = questionnaire_name.replace("_", "-")
    mapping_name = QUESTIONNAIRE_MAPPING[questionnaire_name]
    outline = DATA_COLUMNS[questionnaire_name]
    generic_items = extract_items(participant, outline)
    if is_invalid_response(participant, outline):
        generic_items = []

    generic_response = create_fhir_questionnaire_response(
        id=f"{participant_id}{reg_questionnaire_name}",
        name=mapping_name,
        items=generic_items,
    )

    # to validate that that the generated response conforms to fhir
    questionnaire_response_json = construct_fhir_element(
        "QuestionnaireResponse",
        json.dumps(generic_response),
    )

    return questionnaire_response_json
