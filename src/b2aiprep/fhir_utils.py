import json
from importlib.resources import files
from pathlib import Path
import typing as t

from fhir.resources import construct_fhir_element
from fhir.resources.questionnaireresponse import QuestionnaireResponse

def _load_questionnaire_mapping() -> dict:
    b2ai_resources = files("b2aiprep")
    return json.loads(b2ai_resources.joinpath("resources", "questionnaire2schema.json").read_text())

_QUESTIONNAIRE_MAPPING = _load_questionnaire_mapping()

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
    b2ai_resources = files("b2aiprep").joinpath("resources")
    return json.loads(b2ai_resources.joinpath(f"{questionnaire_name}.json").read_text())

def convert_response_to_fhir(participant: dict, questionnaire_name: str) -> QuestionnaireResponse:
    participant_id = participant["record_id"]
    reg_questionnaire_name = questionnaire_name.replace("_", "-")
    mapping_name = _QUESTIONNAIRE_MAPPING[questionnaire_name]
    outline = load_questionnaire_outline(questionnaire_name)
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

def write_response_to_file(
        output_path: Path,
        data: QuestionnaireResponse,
        schema_name: str,
        subject_id: str,
        session_id: t.Optional[str] = None,
        task_name: t.Optional[str] = None,
        recording_name: t.Optional[str] = None,
        ):
    filename = f"sub-{subject_id}"
    if session_id is not None:
        filename += f"_ses-{session_id}"
    if task_name is not None:
        filename += f"_task-{task_name}"
    if recording_name is not None:
        filename += f"_rec-{recording_name}"
    filename += f"_{schema_name}.json"
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / filename, "w") as f:
        f.write(data.json(indent=2))
