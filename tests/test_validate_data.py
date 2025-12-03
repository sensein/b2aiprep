import pytest

from b2aiprep.prepare.data_validation import (
    validate_participant_data,
    Validator,
)

@pytest.fixture
def synthetic_record_and_dictionary(tmp_path):
    synthetic_record = {
        "record_id": "test_0001",
        "height": 170,
        "weight": 80,
        "unit": "Metric",
        "nervous_anxious": "Not at all",
    }
    data_dictionary = {
        "height": {
            "type": "number",
            "range": {"min": 50, "max": 250},
        },
        "weight": {
            "type": "number",
            "range": {"min": 20, "max": 500},
        },
        "unit": {
            "description": "Unit of measurement for the associated confounder value (for example: \"kg\", \"cm\", \"m\", \"lb\", \"s\", \"min\", or \"years\").",
            "question": {
                "en": "Unit"
            },
            "choices": [
                {
                    "name": {
                        "en": "Metric"
                    },
                    "value": 1
                },
                {
                    "name": {
                        "en": "US customary units"
                    },
                    "value": 2
                }
            ],
            "datatype": [
                "xsd:integer"
            ],
            "valueType": [
                "xsd:integer"
            ]
        },
        "nervous_anxious": {
            "type": "string",
            "choices": [
                {"name": {"en": "Not at all"}, "value": 0},
                {"name": {"en": "Several days"}, "value": 1},
                {"name": {"en": "More than half the days"}, "value": 2},
                {"name": {"en": "Nearly every day"}, "value": 3},
            ]
        }
    }
    return synthetic_record, data_dictionary

@pytest.fixture
def validator(synthetic_record_and_dictionary) -> Validator:
    _, data_dictionary = synthetic_record_and_dictionary
    validator = Validator(data_dictionary=data_dictionary)
    return validator

def test_validate_participant_data(synthetic_record_and_dictionary):
    data, dictionary = synthetic_record_and_dictionary
    validate_participant_data(data, dictionary)
    # If no exception is raised, the test passes

def test_validator_validate_choices(validator: Validator):
    field = "nervous_anxious"
    value = "Not at all"

    assert validator.validate_choices(field=field, value=value) == True


def test_validator_validate_choices_wrong_choice(validator: Validator):
    field = "nervous_anxious"
    value = "Not at all, but sometimes"

    assert validator.validate_choices(field=field, value=value) == False

def test_get_choices(validator: Validator):
    choices = [
        {"name": {"en": "Not at all"}, "value": 0},
        {"name": {"en": "Several days"}, "value": 1},
        {"name": {"en": "More than half the days"}, "value": 2},
        {"name": {"en": "Nearly every day"}, "value": 3},
    ]

    actual = validator.get_choices(choices=choices)
    expected = {"Not at all", "Several days", "More than half the days", "Nearly every day"}

    assert actual == expected


def test_validator_validate_range(validator: Validator):
    assert validator.validate_range("height", 171) == True  # assumes metric


def test_validator_validate_range_incorrect_range(validator: Validator):
    assert validator.validate_range("height", 99999) == False  # assumes metric
