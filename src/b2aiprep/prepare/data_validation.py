from importlib import resources
import json
import pandas as pd
from pathlib import Path
import logging
import typing as t

from b2aiprep.prepare.dataset import _SENSITIVE_FEATURES_REMOVED_FROM_BUNDLE
from b2aiprep.prepare.utils import normalize_task_label

_LOGGER = logging.getLogger(__name__)

WEIGHT_KG_TO_POUNDS_CONVERSION_FACTOR = 2.20462
HEIGHT_CM_TO_INCHES_CONVERSION_FACTOR = 0.393701


class Validator:

    def __init__(self, data_dictionary: dict) -> None:
        self.data_dictionary = data_dictionary
        self.data_thresholds = self._load_thresholds()
    
    def _load_thresholds(self) -> dict:
        """Loads the thresholds used for validating the data."""
        thresholds = json.load(
            (
                resources.files("b2aiprep")
                .joinpath(
                    "prepare",
                    "resources",
                    "data_validation_ranges.json"
                )
                .open()
            )
        )
        return thresholds

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
            _LOGGER.error(f"Invalid '{value}' for '{field}'")
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
        if field in self.data_thresholds:
            max_value = int(self.data_thresholds[field]["maxValue"])
            min_value = int(self.data_thresholds[field]["minValue"])

            if value not in range(min_value, max_value + 1):
                _LOGGER.error(
                    f"Field {field} is out of range"
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


def validate_field(field: str, data: dict, data_dictionary: dict) -> None:
    """
    Validates the field in the provided participant data based on field requirements.
    
    Args:
        field: The field name to validate in the participant's data.
        data: A dictionary containing the participant's data, including the field to validate.
        data_dictionary: A dictionary that represents the answer set of all questions.
    
    Returns:
        None: The function does not return any value.

    """
    cleaner = Validator(data_dictionary=data_dictionary)
    if field is not None:
        # sanitize answers due to dataframes automatically adding trailing decimals and 
        # assertions failing due to answers containing apostrophes
        value = cleaner.clean(data[field])
        if pd.isna(value):
            return
        if field not in data_dictionary:
            return
        cleaner.validate_choices(field, value)

        # For height and weight edge case where value can be metric or imperial
        if field == "height" and data["unit"] != "Metric":
            value = round(value * HEIGHT_CM_TO_INCHES_CONVERSION_FACTOR)

        elif field == "weight" and data["unit"] != "Metric":
            value = round(value * WEIGHT_KG_TO_POUNDS_CONVERSION_FACTOR)

        cleaner.validate_range(field, value)

def validate_phenotype(phenotype_data: pd.DataFrame, data_dictionary: dict) -> None:
    """
    Function to validate the phenotype data for all participants.
    """
    for _, row in phenotype_data.iterrows():
        participant_data = row.to_dict()
        validate_participant_data(data=participant_data, data_dictionary=data_dictionary)

def validate_participant_data(data: dict, data_dictionary: dict) -> None:
    """
    Function validates each field with data dictionary and asserts whether value
    matches the expected answers.
    """
    for field in data:
        validate_field(field=field, data=data, data_dictionary=data_dictionary)


def validate_sensitive_feature_removal_in_bundle(
    *,
    features_dir: Path,
    sensitive_tasks: t.Set[str],
    forbidden_features: t.Union[
        t.AbstractSet[str],
        t.Mapping[str, t.AbstractSet[str]],
    ] = _SENSITIVE_FEATURES_REMOVED_FROM_BUNDLE,
) -> t.List[str]:
    """Validate that bundled parquet feature files do not contain forbidden features for sensitive tasks.

    In the current bundling implementation, sensitive audio feature arrays are removed from the per-record
    feature `.pt` files, and the bundle generators skip those rows entirely for certain features.
    This check makes that expectation explicit.

    Returns a list of human-readable issue strings.
    """

    if not sensitive_tasks:
        return []

    if isinstance(forbidden_features, t.AbstractSet):
        forbidden_features = {"": forbidden_features}

    issues: t.List[str] = []
    sensitive_normalized = {normalize_task_label(t) for t in sensitive_tasks}
    for group, keys_to_remove in forbidden_features.items():
        for feature_name in sorted(keys_to_remove):
            if group == "":
                parquet_name = feature_name
            else:
                parquet_name = f"{group}_{feature_name}"
            
            parquet_path = features_dir / f"{parquet_name}.parquet"
            if not parquet_path.exists():
                continue

            try:
                df = pd.read_parquet(parquet_path, columns=["participant_id", "task_name", "session_id"])
            except Exception as e:
                issues.append(f"Error reading {parquet_path.name} for sensitive-feature validation: {e}")
                continue

            if "task_name" not in df.columns:
                issues.append(f"{parquet_path.name} missing required column task_name")
                continue

            present_sensitive = {
                task
                for task in set(df["task_name"].dropna().unique())
                if normalize_task_label(task) in sensitive_normalized
            }
            if present_sensitive:
                issues.append(
                    "Sensitive tasks unexpectedly present in "
                    f"{parquet_path.name} (forbidden feature: {feature_name}): {sorted(present_sensitive)}"
                )

    return issues