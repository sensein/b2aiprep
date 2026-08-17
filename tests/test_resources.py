"""Tests for the resources module."""

import json
from importlib.resources import files

from b2aiprep.prepare.constants import GENERAL_QUESTIONNAIRES


def test_verify_all_resources_are_present():
    """Verify that all resource JSON files are present in the instrument_columns folder."""

    b2ai_resources = (
        files("b2aiprep").joinpath("prepare").joinpath("resources").joinpath("instrument_columns")
    )
    for questionnaire_name in GENERAL_QUESTIONNAIRES:
        assert b2ai_resources.joinpath(f"{questionnaire_name}.json").exists()


def test_task_registry_resources_are_packaged():
    """The vendored task_registry (registry + stimulus banks) must ship and load."""
    registry_dir = files("b2aiprep.prepare.resources").joinpath("task_registry")
    registry = json.loads(registry_dir.joinpath("registry.json").read_text())
    assert registry["tasks"] and registry["alias_index"]
    for bank in (
        "harvard_sentences_bank",
        "cape_v_sentences_bank",
        "repeating_sentences_bank",
        "reading_passage_bank",
    ):
        assert json.loads(registry_dir.joinpath(f"{bank}.json").read_text())
