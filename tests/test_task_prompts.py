"""Tests for audio-task description/prompt resolution in fhir_utils.

Covers the alias_of layer, the scalar stimulus_text + speech_type model, the
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
    _stimulus_text_from_questionnaire,
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


def test_registry_first_grouped_recording_instruction(descriptions):
    # Generative Naming is a grouped task: the registry now supplies the curated,
    # per-recording instruction (verbatim from redcap), which wins over the flat
    # alias. The animals and food recordings get distinct instructions.
    animals = _resolve(descriptions, "Generative-Naming-Task-animals")
    food = _resolve(descriptions, "Generative-Naming-Task-food")
    assert animals["instructions"] != ""
    assert "animals" in animals["instructions"].lower()
    assert "food" in food["instructions"].lower()
    assert animals["instructions"] != food["instructions"]


def test_flat_alias_of_fallback():
    # A name the registry does not cover still dereferences alias_of in the flat
    # file (tier-2 fallback), independent of the packaged registry.
    from b2aiprep.prepare.fhir_utils import _flat_bids_fields

    descs = collections.OrderedDict(
        [
            ("naming-animals", {"instructions": "Name animals.", "speech_type": "elicited"}),
            ("made-up-alias-xyz", {"alias_of": "naming-animals"}),
        ]
    )
    fields = _flat_bids_fields("made-up-alias-xyz", descs, None, None)
    assert fields["instructions"] == "Name animals."


def test_registry_reading_passage_indexed(descriptions):
    m = _resolve(descriptions, "reading-passage-2")
    assert m["speech_type"] == "read"
    assert m["stimulus_text"].startswith("My most MEMORABLE moment")
    assert m["stimulus_source"] == "transcribed"


def test_registry_repeating_sentences_indexed(descriptions):
    m = _resolve(descriptions, "repeating-sentences-1")
    assert m["stimulus_text"] == "The blue spot is on the key again."
    assert m["speech_type"] == "read"


def test_registry_cape_v_version_index_fix(descriptions):
    # The adult data names v2 CAPE-V as "...-N-(v2)" (version after the number).
    # The flat longest-substring matcher collided v2 onto the v1 key; the registry
    # version-index resolves the correct version regardless of token position.
    v1 = _resolve(descriptions, "Cape-V-sentences-2")
    v2 = _resolve(descriptions, "Cape-V-sentences-2-(v2)")
    assert v1["stimulus_text"] == "How hard did he hit him?"
    assert v2["stimulus_text"] == "He helped her hurry home."


def test_flat_per_key_instruction_wins_over_coarse_registry(descriptions):
    # diadochokinesis has per-syllable instructions in the flat file. The registry
    # only carries a coarse (non-curated) task-level instruction, so the flat
    # per-key text must win -- the 'pa' recording keeps its 'pa' syllable and is
    # not overwritten with the task-level 'puhtuhkuh' demonstration text.
    m = _resolve(descriptions, "diadochokinesis-pa")
    assert "'pa'" in m["instructions"]
    assert "puhtuhkuh" not in m["instructions"]


def test_registry_respiration_v2_swapped_instructions(descriptions):
    # The recording badge names are correct; the instruction blocks were swapped
    # in redcap (eipm/bridge2ai-redcap#44). The registry carries the corrected
    # per-recording instructions.
    hardcough = _resolve(descriptions, "Respiration-and-cough-(v2)-HardCough")
    nose = _resolve(descriptions, "Respiration-and-cough-(v2)-ThreeBreathsNose")
    assert "cough HARD" in hardcough["instructions"]
    assert "through your nose" in nose["instructions"]


def test_repeating_words_both_forms_resolve_to_same_word(descriptions):
    numeral = _resolve(descriptions, "repeat-words-24")
    word = _resolve(descriptions, "Repeating-Words-slice")
    assert numeral["stimulus_text"] == "slice"
    assert word["stimulus_text"] == "slice"
    assert numeral["speech_type"] == "read"
    # stimulus_text is a scalar string, not the legacy array
    assert isinstance(numeral["stimulus_text"], str)
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


def test_read_task_stimulus_text(descriptions):
    m = _resolve(descriptions, "Harvard-Sentences-List-1-2")
    assert m["speech_type"] == "read"
    assert m["stimulus_text"] == "Glue the sheet to the dark blue background."


def test_recall_task_carries_reference(descriptions):
    m = _resolve(descriptions, "story-recall")
    assert m["speech_type"] == "recall"
    assert m["stimulus_text"].startswith("There was once a boy")


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
    assert vocab["stimulus_text"] == "fuddle"
    assert vocab["speech_type"] == "elicited"

    vocab_empty = _resolve(descriptions, "Productive-Vocabulary-2", lookup, tid)
    assert vocab_empty["stimulus_text"] == ""

    rand = _resolve(descriptions, "Random-Item-Generation", lookup, tid)
    assert rand["stimulus_text"] == ""
    assert rand["instructions"].endswith("Category: animals.")

    stroop = _resolve(descriptions, "Word-color-Stroop", lookup, tid)
    assert stroop["stimulus_text"] == "red green blue"
    assert stroop["speech_type"] == "read"


def test_questionnaire_join_absent_is_noop(descriptions):
    # No lookup provided -> vocab stimulus_text stays empty, no crash.
    vocab = _resolve(descriptions, "Productive-Vocabulary-3", questionnaire_lookup=None)
    assert vocab["stimulus_text"] == ""
    # A lookup missing this task id -> also empty.
    vocab2 = _resolve(descriptions, "Productive-Vocabulary-3", {}, "AT-MISSING")
    assert vocab2["stimulus_text"] == ""
