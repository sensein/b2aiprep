from importlib import resources
import json
import pandas as pd

# max height ever recorded
MAX_HUMAN_HEIGHT_METRIC = 272
MAX_HUMAN_HEIGHT_US_UNIT = 107

# max weight in ever recorded
MAX_HUMAN_WEIGHT_METRIC = 635
MAX_HUMAN_WEIGHT_US_UNIT = 1400


def get_choices(choices):
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


def clean_number(num):
    """
    Helper function to help with comparing the answer with the possible answers
    with the expected answers from the data dictionary.
    """
    try:
        clean_num = float(num)
        return int(clean_num) if clean_num.is_integer() else clean_num
    except (ValueError, TypeError):
        return num


def validate_participant_data(participant):
    """
    Function validates each field with data dictionary and asserts whether value
    matches the expected answers. 
    """
    participant_id = participant.get("participant_id") or participant.get("record_id")
    data_dictionary = json.load((resources.files("b2aiprep").joinpath(
        "prepare", "resources", "b2ai-data-bids-like-template", "participants.json").open())
    )
    for field in participant:
        if field is not None:
            value = clean_number(participant[field])
            if isinstance(value, str):
                value = value.replace("'", "")
            if pd.isna(value):
                continue
            # fields that are redcap specific
            if field not in ["redcap_repeat_instrument", "redcap_repeat_instance", "participant_id"]:
                # added in derivatives, not included in data dictionary
                if field == "sex_at_birth":
                    assert value in ["Male", "Female"]
                    continue
                assert field in data_dictionary, f"Dataset {field} does not exist."
                result_dict = data_dictionary[field]
                if "choices" not in result_dict or result_dict["choices"] is None:
                    options = set()
                else:
                    options = set(get_choices(result_dict["choices"]))

                if "minValue" in result_dict and result_dict["minValue"] is not None:
                    assert float(value) >= float(result_dict["minValue"]), (
                        f"The value {value} does not meet the minimum "
                        f"value allowed for {field} for participant {participant_id}"
                    )
                if "maxValue" in result_dict and result_dict["maxValue"] is not None:
                    assert float(value) <= float(result_dict["maxValue"]), (
                        f"The value {value} exceeds the max value for "
                        f"{field} for participant {participant_id}"
                    )
                    slide_options = set(
                        range(int(result_dict["maxValue"] + 1)))
                    slide_options = set(str(i) for i in slide_options)
                    options = options.union(slide_options)

                if len(options) != 0:
                    assert str(value) in options, (
                        f"The value {value} is not a valid answer for the {field} "
                        f"field for participant {participant_id}"
                    )

                # fields with a history of have inconsistent values
                if field == "Height":
                    if result_dict["unit"] == "Metric":
                        assert float(value) <= MAX_HUMAN_HEIGHT_METRIC, (
                            f"The height of {value} exceeds the "
                            f"max value for {field} for participant {participant_id}"
                        )
                    else:
                        assert float(value) <= MAX_HUMAN_HEIGHT_US_UNIT, (
                            f"The value {value} exceeds the max "
                            f"value for {field} for participant {participant_id}"
                        )

                if field == "Weight":
                    if result_dict["unit"] == "Metric":
                        assert float(value) <= MAX_HUMAN_WEIGHT_METRIC, (
                            f"The Weight of {value} exceeds the "
                            f"max value for {field} for participant {participant_id}"
                        )
                    else:
                        assert float(value) <= MAX_HUMAN_WEIGHT_US_UNIT, (
                            f"The Weight of {value} exceeds the "
                            f"max value for {field} for participant {participant_id}"
                        )


def validate_derivatives(derivatives_csv_path):
    """
    Function parses the data frame and validates each record.
    """
    df = pd.read_csv(derivatives_csv_path, sep='\t')
    participants = df.to_dict(orient='records')

    for participant in participants:
        validate_participant_data(participant)
