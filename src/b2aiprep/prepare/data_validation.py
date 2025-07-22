from importlib import resources
import json
import pandas as pd
from pathlib import Path
import logging

_LOGGER = logging.getLogger(__name__)

WEIGHT_KG_TO_POUNDS_CONVERSION_FACTOR = 2.20462
HEIGHT_CM_TO_INCHES_CONVERSION_FACTOR = 0.393701


class Validator:

    def __init__(self, data_dictionary: dict, participant_id: str, field_requirements: dict) -> None:
        self.data_dictionary = data_dictionary
        self.participant_id = participant_id
        self.field_requirements = field_requirements

    def clean(self, value: str) -> str | int | float:
        """
        Cleans the provided input value by processing it based on its type.
        
        Args:
        value: The value to clean. It can be of type str, int, or float.
        
        Returns:
            A cleaned version of the input:
                - If the input is a string, returns the string with single quotes removed.
                - If the input is a number (int or float), returns an integer if the number is a whole number, 
                or a float otherwise.
        """
        if not isinstance(value, str):
            clean_num = float(value)
            return int(clean_num) if clean_num.is_integer() else clean_num
        value = value.replace("'", "")
        return value

    def validate_choices(self, field: str, value: str | int) -> bool:
        """
        Validates if the provided value is a valid choice for the given field.

        Args:
            field: The field name (string) to validate the value against.
            value: The value to validate, which would be checked against the available choices.

        Returns:
            bool: True if the value is valid for the given field, False otherwise.
        """
        result_dict = self.data_dictionary[field]
        options = set()
        if "choices" in result_dict and result_dict["choices"] is not None:
            options = set(self.get_choices(result_dict["choices"]))

        # edge case for for slider questions
        if "maxValue" in result_dict and result_dict["maxValue"] is not None:
            slide_options = set(range(int(result_dict["maxValue"] + 1)))
            slide_options = set(str(i) for i in slide_options)
            options = options.union(slide_options)

        if len(options) != 0 and str(value) not in options:
            _LOGGER.error(f"Invalid '{value}' for '{field}' (participant {self.participant_id})")
            return False
        return True

    def validate_range(self, field: str, value: int) -> bool:
        """
        Validates whether the provided value falls within the allowed range for the given field.
        Args:
            field: The field name (string) to validate the value against.
            value: The value to validate, which should be within the range defined for the field.
        
        Returns:
            bool: True if the value is within the allowed range for the given field, False otherwise.
        
        """
        if field in self.field_requirements:
            max_value = int(self.field_requirements[field]["maxValue"])
            min_value = int(self.field_requirements[field]["minValue"])

            if value not in range(min_value, max_value + 1):
                _LOGGER.error(
                    f"Field {field} is out of range for (participant {self.participant_id})"
                )
                return False
        return True

    def get_choices(self, choices: list) -> set:
        """
        Helper function to generate a set of possible answers to a given question based on
        the data dictionary.
        
        Args:
            choices: A list of possible answers for a particular question.
            
        Returns:
            set: A set containing all possible choices for a particular question.
        """
        solution_set = set()
        if isinstance(choices, list):
            options = [item["name"]["en"].strip() for item in choices]
            for option in options:
                # Edge case to remove apostrophes as csv answers don't have apostrophes present
                option = option.replace('"', "")
                solution_set.add(option)
        return solution_set


def validate_fields(field: str, participant: dict, data_dictionary: dict) -> None:
    """
    Validates the fields in the provided participant data based on field requirements.
    
    Args:
        field: The field name to validate in the participant's data.
        participant: A dictionary containing the participant's data, including the field to validate.
        data_dictionary: A dictionary that represents the ansswer set of all questions.
    
    Returns:
        None: The function does not return any value.

    """
    field_requirements = json.load(
        (Path(__file__).parent.parent.parent.parent / "data" / "field_requirements.json").open()
    )
    participant_id = participant.get("participant_id") or participant.get("record_id")
    cleaner = Validator(
        data_dictionary=data_dictionary,
        participant_id=participant_id,
        field_requirements=field_requirements,
    )
    if field is not None:
        # sanitize answers due to dataframes automatically adding trailing decimals and 
        # assertions failing due to answers containing apostrophes
        value = cleaner.clean(participant[field])
        if pd.isna(value):
            return
        if field not in data_dictionary:
            return
        cleaner.validate_choices(field, value)

        # For height and weight edge case where value can be metric or imperial
        if field == "height" and participant["unit"] != "Metric":
            value = round(value * HEIGHT_CM_TO_INCHES_CONVERSION_FACTOR)

        elif field == "weight" and participant["unit"] != "Metric":
            value = round(value * WEIGHT_KG_TO_POUNDS_CONVERSION_FACTOR)

        cleaner.validate_range(field, value)


def validate_participant_data(participant: dict) -> None:
    """
    Function validates each field with data dictionary and asserts whether value
    matches the expected answers.
    """
    data_dictionary = json.load(
        (
            resources.files("b2aiprep")
            .joinpath("prepare", "resources", "b2ai-data-bids-like-template", "participants.json")
            .open()
        )
    )
    for field in participant:
        validate_fields(field=field, participant=participant, data_dictionary=data_dictionary)


def validate_derivatives(derivatives_csv_path: str) -> None:
    """
    Function parses the data frame and validates each record.
    """
    file_path = Path(derivatives_csv_path)
    sep = "\t" if file_path.suffix.lower() == ".tsv" else ","

    df = pd.read_csv(derivatives_csv_path, sep=sep)
    participants = df.to_dict(orient="records")

    for participant in participants:
        validate_participant_data(participant)
