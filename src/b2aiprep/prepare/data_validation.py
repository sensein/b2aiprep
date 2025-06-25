
import pandas as pd
from importlib import resources
import json
import pycountry
import iso639


def get_choices(choices):
    if isinstance(choices, list):
        return [item['name']['en'].strip() for item in choices]

def clean_number(s):
    try:
        num = float(s)
        return int(num) if num.is_integer() else num
    except (ValueError, TypeError):
        return s

def validate_participant_data(participant):
    participant_id = participant["participant_id"]
    data_dictionary = json.load((resources.files("b2aiprep").joinpath(
        "prepare", "resources", "b2ai-data-bids-like-template", "participants.json").open())
    )
    for field in participant:
        print(field)
        if field is not None:
            value = clean_number(participant[field])
            
            
            if pd.isna(value):
                continue 
            if field not in ["redcap_repeat_instrument", "redcap_repeat_instance", "participant_id"]:  # fields that are redcap specific
                if field =="sex_at_birth":
                    assert value in ["Male", "Female"]
                    continue
                assert field in data_dictionary, f"Dataset {field} does not exist."
                result_dict = data_dictionary[field]
                if "choices" not in result_dict or result_dict["choices"] is None:
                    pass
                else:
                    options = get_choices(result_dict["choices"])
                    if options is not None:
                        assert str(value) in options, f"The value {value} is not a valid answer for the {field} {options} field for participant {participant_id}"


def validate_derivatives(derivatives_csv_path):
    df = pd.read_csv(derivatives_csv_path, sep='\t')
    participants = df.to_dict(orient='records')

    for participant in participants:
        validate_participant_data(participant)
