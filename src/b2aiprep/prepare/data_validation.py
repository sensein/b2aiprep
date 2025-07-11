from importlib import resources
import json
import pandas as pd
from pathlib import Path
import logging

WEIGHT_METRIC_TO_US_CONVERSION_FACTOR = 2.20462
HEIGHT_METRIC_TO_US_CONVERSION_FACTOR = 0.393701

class Validator():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)

    def __init__(self, data_dictionary, participant_id, field_requirements):
        self.data_dictionary = data_dictionary
        self.participant_id = participant_id
        self.field_requirements = field_requirements

    def clean(self, value):
        if not isinstance(value, str):
            clean_num = float(value)
            return int(clean_num) if clean_num.is_integer() else clean_num
        value = value.replace("'", "")
        return value

    def validate_choices(self, field, value):
        result_dict = self.data_dictionary[field]
        options = set()
        if "choices" in result_dict and result_dict["choices"] is not None:
            options = set(self.get_choices(result_dict["choices"]))

        # edge case for for slider questions
        if "maxValue" in result_dict and result_dict["maxValue"] is not None:
            slide_options = set(
                range(int(result_dict["maxValue"] + 1)))
            slide_options = set(str(i) for i in slide_options)
            options = options.union(slide_options)

        if len(options) != 0 and str(value) not in options:
            Validator.logger.error(
                f"Invalid '{value}' for '{field}' (participant {self.participant_id})"
            )
            return False

    def validate_range(self, field, value):
        if field in self.field_requirements:
            max_value = int(self.field_requirements[field]["maxValue"])
            min_value = int(self.field_requirements[field]["minValue"])

            if value not in range(min_value, max_value+1):
                Validator.logger.error(
                    f"Field {field} is out of range for (participant {self.participant_id})"
                )
                return False

    def get_choices(self, choices):
        """
        Helper function to generate a set of possible answers to a given question based on
        the data dictionary.
        """
        solution_set = set()
        if isinstance(choices, list):
            options = [item['name']['en'].strip() for item in choices]
            for option in options:
                option = option.replace("'", "")
                option = option.replace("\"", "")
                solution_set.add(option)
        return solution_set


def validate_fields(field, participant, data_dictionary):
    field_requirements = json.load(
        (Path(__file__).parent.parent.parent.parent / "data" / "field_requirements.json").open()
    )
    participant_id = participant.get("participant_id") or participant.get("record_id")
    cleaner = Validator(data_dictionary=data_dictionary,
                        participant_id=participant_id, field_requirements=field_requirements)
    if field is not None:
        value = cleaner.clean(participant[field])
        if pd.isna(value):
            return
        if field not in data_dictionary:
            return
        cleaner.validate_choices(field, value)

        # For height and weight edge case where value can be metric or imperial
        if field == "height" and participant["unit"] != "Metric":
            value = round(value * HEIGHT_METRIC_TO_US_CONVERSION_FACTOR)

        elif field == "weight" and participant["unit"] != "Metric":
            value = round(value * WEIGHT_METRIC_TO_US_CONVERSION_FACTOR)

        cleaner.validate_range(field, value)

def validate_participant_data(participant):
    """
    Function validates each field with data dictionary and asserts whether value
    matches the expected answers. 
    """
    data_dictionary = json.load((resources.files("b2aiprep").joinpath(
        "prepare", "resources", "b2ai-data-bids-like-template", "participants.json").open())
    )
    for field in participant:
        validate_fields(field=field, participant=participant, data_dictionary=data_dictionary)


def validate_derivatives(derivatives_csv_path):
    """
    Function parses the data frame and validates each record.
    """
    file_path = Path(derivatives_csv_path)
    sep = '\t' if file_path.suffix.lower() == '.tsv' else ','

    df = pd.read_csv(derivatives_csv_path, sep=sep)
    participants = df.to_dict(orient='records')

    for participant in participants:
        validate_participant_data(participant)
