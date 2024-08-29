"""Tests for the resources module."""

from importlib.resources import files

from b2aiprep.prepare.constants import GENERAL_QUESTIONNAIRES


def test_verify_all_resources_are_present():
    """Verify that all resource JSON files are present in the instrument_columns folder."""

    b2ai_resources = (
        files("b2aiprep").joinpath("prepare").joinpath("resources").joinpath("instrument_columns")
    )
    for questionnaire_name in GENERAL_QUESTIONNAIRES:
        assert b2ai_resources.joinpath(f"{questionnaire_name}.json").exists()
