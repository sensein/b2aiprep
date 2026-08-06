"""Tests for audio-task description/prompt resolution in fhir_utils.

Covers the alias_of layer, the scalar prompted_text + speech_type model, the
stimulus-bank resolution (both numeral and word task-name forms), and the
questionnaire-join that surfaces per-participant stimulus (vocab/random/stroop).
"""

import collections
import json
from importlib.resources import files

import pytest

from b2aiprep.prepare.fhir_utils import (
    convert_response_to_bids_metadata,
    _classify_speech_type,
    _prompted_text_from_questionnaire,
    _resolve_prompt_ref,
)


@pytest.fixture(scope="module")
def descriptions():
    path = files("b2aiprep.prepare.resources").joinpath("audio_task_descriptions.json")
    return json.loads(path.read_text(), object_pairs_hook=collections.OrderedDict)


def _resolve(descriptions, recording_name, questionnaire_lookup=None, acoustic_task_id="AT-1"):
    """Run the recording-metadata resolver for a given recording name."""
    recording = {
        "recording_name": recording_name,
        "recording_acoustic_task_id": acoustic_task_id,
        "recording_session_id": "S1",
        "record_id": "r1",
    }
    columns = ["recording_name", "recording_acoustic_task_id", "recording_session_id"]
    return convert_response_to_bids_metadata(
        recording,
        questionnaire_name="recordings",
        mapping_name="recordingschema",
        columns=columns,
        audio_task_descriptions=descriptions,
        questionnaire_lookup=questionnaire_lookup,
    )


def test_alias_of_dereferences_to_target(descriptions):
    naming = _resolve(descriptions, "Naming-Animals")
    aliased = _resolve(descriptions, "Generative-Naming-Task-animals")
    assert aliased["instructions"] == naming["instructions"]
    assert aliased["instructions"] != ""


def test_repeating_words_both_forms_resolve_to_same_word(descriptions):
    numeral = _resolve(descriptions, "repeat-words-24")
    word = _resolve(descriptions, "Repeating-Words-slice")
    assert numeral["prompted_text"] == "slice"
    assert word["prompted_text"] == "slice"
    assert numeral["speech_type"] == "read"
    # prompted_text is a scalar string, not the legacy array
    assert isinstance(numeral["prompted_text"], str)
    assert "prompts" not in numeral


def test_resolve_prompt_ref_scalar_and_out_of_range():
    ref = {"bank": "repeating_words_bank"}
    assert _resolve_prompt_ref("repeating-words-slice", ref) == "slice"
    assert _resolve_prompt_ref("repeat-words-1", ref) == "smile"
    assert _resolve_prompt_ref("repeat-words-99", ref) == ""  # out of range
    assert _resolve_prompt_ref("repeat-words", ref) == ""  # no index/word token


def test_speech_type_classification():
    assert _classify_speech_type("harvard-sentences-list-3-2") == "read"
    assert _classify_speech_type("cape-V-sentences-(v2)-4") == "read"
    assert _classify_speech_type("story-recall") == "recall"
    assert _classify_speech_type("loudness-(v2)") == "non-lexical"
    assert _classify_speech_type("diadochokinesis-PA") == "non-lexical"
    assert _classify_speech_type("picture") == "elicited"


def test_read_task_prompted_text(descriptions):
    m = _resolve(descriptions, "Harvard-Sentences-List-1-2")
    assert m["speech_type"] == "read"
    assert m["prompted_text"] == "Glue the sheet to the dark blue background."


def test_recall_task_carries_reference(descriptions):
    m = _resolve(descriptions, "story-recall")
    assert m["speech_type"] == "recall"
    assert m["prompted_text"].startswith("There was once a boy")


def test_questionnaire_join_vocab_random_stroop(descriptions):
    tid = "AT-JOIN"
    lookup = {
        ("vocab", tid): {
            "vocabulary_item_word_1": "boil",
            "vocabulary_item_word_2": "",
            "vocabulary_item_word_3": "fuddle",
        },
        ("random", tid): {"random_item_generation_category": "animals"},
        ("stroop", tid): {
            "stroop_item_color_1": "red",
            "stroop_item_color_2": "green",
            "stroop_item_color_3": "blue",
            "stroop_item_color_4": "nan",
        },
    }
    vocab = _resolve(descriptions, "Productive-Vocabulary-3", lookup, tid)
    assert vocab["prompted_text"] == "fuddle"
    assert vocab["speech_type"] == "elicited"

    vocab_empty = _resolve(descriptions, "Productive-Vocabulary-2", lookup, tid)
    assert vocab_empty["prompted_text"] == ""

    rand = _resolve(descriptions, "Random-Item-Generation", lookup, tid)
    assert rand["prompted_text"] == ""
    assert rand["instructions"].endswith("Category: animals.")

    stroop = _resolve(descriptions, "Word-color-Stroop", lookup, tid)
    assert stroop["prompted_text"] == "red green blue"
    assert stroop["speech_type"] == "read"


def test_questionnaire_join_absent_is_noop(descriptions):
    # No lookup provided -> vocab prompted_text stays empty, no crash.
    vocab = _resolve(descriptions, "Productive-Vocabulary-3", questionnaire_lookup=None)
    assert vocab["prompted_text"] == ""
    # A lookup missing this task id -> also empty.
    vocab2 = _resolve(descriptions, "Productive-Vocabulary-3", {}, "AT-MISSING")
    assert vocab2["prompted_text"] == ""
