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


def _resolve(descriptions, recording_name, questionnaire_lookup=None, acoustic_task_id="AT-1",
             population=None):
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
        population=population,
    )


def test_population_aware_picture_description(descriptions):
    # picture-description exists in both populations; the recording's cohort ->
    # population routes it to the right task (peds child-facing vs adult).
    peds = _resolve(descriptions, "Picture Description", population="pediatric")
    adult = _resolve(descriptions, "Picture Description", population="adult")
    assert "no right or wrong answers" in peds["instructions"]  # peds child-facing
    assert peds["instructions"] != adult["instructions"]


def test_picture_description_image_assets(descriptions):
    # peds: single fixed image; adult: bare & option1 are Picture 1, option2 is
    # Picture 2 (per the doc note that bare == option1).
    peds = _resolve(descriptions, "Picture Description", population="pediatric")
    assert peds["stimulus_asset"].endswith("pediatric_10plus_picture_description.jpg")
    bare = _resolve(descriptions, "Picture description", population="adult")
    assert bare["stimulus_asset"].endswith("PictureDescriptionTaskPicture1.png")
    opt2 = _resolve(descriptions, "Picture description-option2", population="adult")
    assert opt2["stimulus_asset"].endswith("PictureDescriptionTaskPicture2.jpg")


def test_non_lexical_tasks_have_empty_stimulus(descriptions):
    # non-lexical tasks carry no lexical reference; stimulus_text must be "" and
    # must NOT leak the instruction (the flat file stores it under "prompts").
    for name in ["Respiration and cough-Breath-1", "Maximum phonation time-1",
                 "Diadochokinesis-PA", "Loudness", "Glides-Low to High"]:
        m = _resolve(descriptions, name)
        assert m["speech_type"] == "non-lexical"
        assert m["stimulus_text"] == "", (name, m["stimulus_text"])
    # read/recall still carry their reference text (not emptied)
    assert _resolve(descriptions, "Rainbow Passage")["stimulus_text"].startswith("When the sunlight")


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


def test_diadochokinesis_v1_curated_instruction(descriptions):
    # diadochokinesis v1 is now curated from the Retired doc: the 'pa' recording
    # keeps its own /PA/ syllable ("as fast as possible 10 times"), NOT the v2
    # demonstration/timer template ("puhtuhkuh"/"until the timer runs out").
    m = _resolve(descriptions, "diadochokinesis-pa")
    assert "/PA/" in m["instructions"]
    assert "as fast as possible 10 times" in m["instructions"]
    assert "puhtuhkuh" not in m["instructions"]
    assert "timer" not in m["instructions"]


def test_language_field_recorded(descriptions):
    from b2aiprep.prepare.fhir_utils import _language_from_selected

    assert _language_from_selected("English") == "en"
    assert _language_from_selected("Spanish") == "es-419"
    assert _language_from_selected("Español") == "es-419"
    assert _language_from_selected("French") == "fr-CA"
    assert _language_from_selected(None) == "en"
    assert _language_from_selected("nan") == "en"
    # A coded export (selected_language radio codes 1/2/3) must not collapse
    # Spanish/French to English.
    assert _language_from_selected("3") == "es-419"
    assert _language_from_selected("2") == "fr-CA"
    assert _language_from_selected("1") == "en"
    # selected_language_2 BCP-47 codes
    assert _language_from_selected("es-419") == "es-419"
    assert _language_from_selected("fr-CA") == "fr-CA"
    # Every sidecar carries a language; default is 'en', explicit values pass through.
    assert _resolve(descriptions, "Rainbow Passage")["language"] == "en"
    m = convert_response_to_bids_metadata(
        {"recording_name": "Rainbow Passage", "recording_acoustic_task_id": "AT",
         "recording_session_id": "S", "record_id": "r"},
        questionnaire_name="recordings", mapping_name="recordingschema",
        columns=["recording_name", "recording_acoustic_task_id", "recording_session_id"],
        audio_task_descriptions=descriptions, language="es-419",
    )
    assert m["language"] == "es-419"


def test_unrecognized_language_warns(caplog):
    import logging
    from b2aiprep.prepare.fhir_utils import _language_from_selected

    # A present-but-unmapped value (a new language, or a coded/integer export)
    # must default to 'en' AND warn -- never silently mislabel a non-English
    # session as English.
    with caplog.at_level(logging.WARNING, logger="b2aiprep.prepare.fhir_utils"):
        assert _language_from_selected("German") == "en"
        assert _language_from_selected("9") == "en"
    assert "unrecognized selected_language" in caplog.text
    # A recognized value does not warn.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="b2aiprep.prepare.fhir_utils"):
        assert _language_from_selected("Spanish") == "es-419"
    assert "unrecognized selected_language" not in caplog.text


def test_spanish_es419_stimulus(descriptions):
    # es-419 sessions get the Spanish reference for read/recall tasks that have a
    # Spanish stimulus in the redcap (Harvard, CAPE-V v2, Caterpillar, Story Recall
    # v2) -- never the English text. English stimulus_text is unchanged. (English
    # random-item *instructions* do change intentionally -- the time limit was
    # dropped per request, which makes v1/v2 identical; see the random-item test.)
    harv_en = _resolve(descriptions, "Harvard Sentences-List 4-1", population="adult")
    harv_es = _resolve(descriptions, "Harvard Sentences-List 4-1", population="adult")
    harv_es = convert_response_to_bids_metadata(
        {"recording_name": "Harvard Sentences-List 4-1", "recording_acoustic_task_id": "AT",
         "recording_session_id": "S", "record_id": "r"},
        questionnaire_name="recordings", mapping_name="recordingschema",
        columns=["recording_name", "recording_acoustic_task_id", "recording_session_id"],
        audio_task_descriptions=descriptions, population="adult", language="es-419")
    assert harv_en["stimulus_text"] != harv_es["stimulus_text"]
    assert harv_es["stimulus_text"] == "El duque salió del parque en un coche negro."
    assert harv_es["language"] == "es-419" and harv_es["stimulus_source"] == "doc-text"

    def es(name):
        return convert_response_to_bids_metadata(
            {"recording_name": name, "recording_acoustic_task_id": "AT",
             "recording_session_id": "S", "record_id": "r"},
            questionnaire_name="recordings", mapping_name="recordingschema",
            columns=["recording_name", "recording_acoustic_task_id", "recording_session_id"],
            audio_task_descriptions=descriptions, population="adult", language="es-419")

    assert es("Cape V sentences-2-(v2)")["stimulus_text"] == "Hacen más fuerza si crece la asociación."
    assert es("Caterpillar Passage")["stimulus_text"].startswith("¿Te gustan los parques de atracciones?")
    assert es("Story-Recall-(v2)")["stimulus_text"].startswith("Había un niño")
    # A task with no Spanish stimulus (Rainbow, retired) falls back to English text
    # but is still tagged es-419 so it is filterable -- never silently mislabeled.
    rainbow = es("Rainbow Passage")
    assert rainbow["language"] == "es-419"
    assert rainbow["stimulus_text"].startswith("When the sunlight")
    # A v1-labelled CAPE-V recording in a Spanish session still gets the current
    # Spanish sentence set (Spanish has one set per family); story-recall v1 too.
    assert es("Cape V sentences-1")["stimulus_text"] == "Este bus de aquí para poco en el mes de agosto."
    assert es("Story recall")["stimulus_text"].startswith("Había un niño")


def test_spanish_es419_instructions(descriptions):
    def es(name):
        return convert_response_to_bids_metadata(
            {"recording_name": name, "recording_acoustic_task_id": "AT",
             "recording_session_id": "S", "record_id": "r"},
            questionnaire_name="recordings", mapping_name="recordingschema",
            columns=["recording_name", "recording_acoustic_task_id", "recording_session_id"],
            audio_task_descriptions=descriptions, population="adult", language="es-419")

    assert es("Harvard Sentences-List 4-1")["instructions"].startswith("Por favor, lea")
    assert es("Caterpillar Passage")["instructions"].startswith("Este es un pasaje")
    assert es("Diadochokinesis (v2)-puh")["instructions"].startswith("Esta tarea nos ayuda")
    # English is unchanged
    assert _resolve(descriptions, "Harvard Sentences-List 4-1", population="adult")["instructions"].startswith("Please read")


def test_questionnaire_join_language_agnostic(descriptions):
    # The vocab/random/stroop stimulus comes from the questionnaire join, which
    # passes through whatever value was stored (Spanish for a Spanish session).
    # No es-419 sessions recorded these tasks in 07_01, but if one did the join
    # must still resolve, in the recording's language, from the stored values.
    tid = "AT-ES"
    lookup = {
        ("vocab", tid): {"vocabulary_item_word_3": "enredar"},
        ("random", tid): {"random_item_generation_category": "animales"},
        ("stroop", tid): {"stroop_item_color_1": "rojo", "stroop_item_color_2": "verde"},
    }

    def es(name):
        return convert_response_to_bids_metadata(
            {"recording_name": name, "recording_acoustic_task_id": tid,
             "recording_session_id": "S", "record_id": "r"},
            questionnaire_name="recordings", mapping_name="recordingschema",
            columns=["recording_name", "recording_acoustic_task_id", "recording_session_id"],
            audio_task_descriptions=descriptions, questionnaire_lookup=lookup,
            language="es-419", population="adult")

    v = es("Productive-Vocabulary-3")
    assert v["language"] == "es-419" and v["stimulus_text"] == "enredar"
    r = es("Random-Item-Generation")
    # a semantic category (Category_2 only) -> Spanish non-repeatable instruction,
    # with the Spanish category label.
    assert r["language"] == "es-419"
    assert r["instructions"].startswith("Diga tantos elementos de la siguiente categoría")
    assert r["instructions"].endswith("Categoría: animales.")
    s = es("Word-color-Stroop")
    assert s["language"] == "es-419" and s["stimulus_text"] == "rojo verde"


def test_random_item_instruction_by_category(descriptions):
    def R(cat, lang):
        lk = {("random", "AT"): {"random_item_generation_category": cat}}
        return convert_response_to_bids_metadata(
            {"recording_name": "Random-Item-Generation", "recording_acoustic_task_id": "AT",
             "recording_session_id": "S", "record_id": "r"},
            questionnaire_name="recordings", mapping_name="recordingschema",
            columns=["recording_name", "recording_acoustic_task_id", "recording_session_id"],
            audio_task_descriptions=descriptions, questionnaire_lookup=lk,
            population="adult", language=lang)["instructions"]

    # A semantic category is unambiguously the non-repeatable variant.
    assert R("Drinks", "en").startswith("Say as many items from the following category")
    assert "Do not repeat any item" in R("Drinks", "en")
    assert R("Drinks", "en").endswith("Category: Drinks.")
    assert R("Drinks", "es-419").startswith("Diga tantos elementos de la siguiente categoría")
    # Numbers/Letters appear in BOTH category lists -> ambiguous -> the general
    # instruction (describes both variants), never asserting repeatability, but the
    # drawn category is still surfaced.
    for cat in ("Numbers", "Letters"):
        assert R(cat, "en").startswith("You will have to speak a series of")
        assert R(cat, "en").endswith(f"Category: {cat}.")
        assert R(cat, "es-419").startswith("Deberá decir una serie de")
    # Both variants carry the "selection appears / auto-stops" procedural line.
    assert "automatically stop at the end" in R("Drinks", "en")
    # Time limit is removed for random-item (durations show it was not enforced).
    for cat in ("Drinks", "Numbers"):
        assert "Time limit" not in R(cat, "en")
        assert "Límite de tiempo" not in R(cat, "es-419")


def test_free_speech_es419_cue_v2_only(descriptions):
    def es(name):
        return convert_response_to_bids_metadata(
            {"recording_name": name, "recording_acoustic_task_id": "AT",
             "recording_session_id": "S", "record_id": "r"},
            questionnaire_name="recordings", mapping_name="recordingschema",
            columns=["recording_name", "recording_acoustic_task_id", "recording_session_id"],
            audio_task_descriptions=descriptions, population="adult", language="es-419")

    # Numbered current (v2) free-speech gets its per-recording Spanish cue.
    assert es("Free speech (v2)-1")["stimulus_text"].startswith("¿Cuál es su estación favorita")
    assert es("Free speech (v2)-3")["stimulus_text"].startswith("Cuéntenos sobre su libro")
    # The voice variant (unnumbered) and v1 (both Retired, no es-419 source) must
    # NOT be given the v2 questions -- they keep their English cue.
    voice = es("Free speech")
    assert voice["stimulus_text"].startswith("Can you explain your voice/speech problems")
    v1 = es("Free speech-1")
    assert v1["stimulus_text"].startswith("Can you")  # English v1 cue, not the v2 Spanish one
    assert "favorita" not in v1["stimulus_text"] and "estación" not in v1["stimulus_text"]


def test_static_inline_read_recall_has_doc_text_source(descriptions):
    # Static-inline read/recall tasks (rainbow/caterpillar passages, story recall)
    # carry their reference text via the flat fallback; the sidecar must still tag
    # provenance as doc-text so consumers can identify these WER/reference targets.
    for name in ["Rainbow Passage", "Caterpillar Passage", "Story-Recall"]:
        m = _resolve(descriptions, name, population="adult")
        assert m["speech_type"] in ("read", "recall"), name
        assert m["stimulus_text"], name
        assert m["stimulus_source"] == "doc-text", (name, m["stimulus_source"])
    # non-read/recall static-inline tasks must NOT be tagged doc-text (empty stim)
    mpt = _resolve(descriptions, "Maximum phonation time-1")
    assert mpt["stimulus_text"] == ""
    assert "stimulus_source" not in mpt or mpt.get("stimulus_source") is None


def test_version_select_tolerates_glued_alias():
    # _select_version must find the version token in "(v2)", bare "v2", and the
    # glued "...timev2" form _norm() yields, without mistaking family names.
    from b2aiprep.prepare.fhir_utils import _resolve_task_registry, _norm

    for name in ("maximum-phonation-time-(v2)", "maximum-phonation-time (v2)",
                 _norm("maximum-phonation-time(v2)")):
        match = _resolve_task_registry(name, population="adult")
        assert match is not None, name
        task, _ = match
        assert task["task_id"].endswith(".v2"), (name, task["task_id"])
    # the version-less name resolves to the unversioned/v1 task, not v2
    bare = _resolve_task_registry("maximum-phonation-time", population="adult")
    assert bare is not None and not bare[0]["task_id"].endswith(".v2")


def test_registry_image_stimulus_asset(descriptions):
    m = _resolve(descriptions, "noisy-sounds-3")
    # the target sound is printed on the card -> transcribed into the text bank,
    # but the image asset is still pinned (see test_sound_cards_* for the token).
    assert m["stimulus_source"] == "transcribed"
    assert m["stimulus_text"] == "oo oo oo"
    # commit-pinned raw GitHub URL, path URL-encoded (spaces -> %20)
    assert m["stimulus_asset"].startswith(
        "https://raw.githubusercontent.com/eipm/bridge2ai-redcap/"
    )
    assert "%20" in m["stimulus_asset"]
    assert m["stimulus_asset"].endswith("pediatric_noisy_sounds_3.jpg")
    # identifying-pictures uses a zero-padded index in the asset filename
    ip = _resolve(descriptions, "Identifying-Pictures-8")
    assert ip["stimulus_asset"].endswith("pediatric_identifying_pictures_08.jpg")
    # transcribed-from-image bank tasks also carry a pinned image URL
    rp = _resolve(descriptions, "reading-passage-2")
    assert rp["stimulus_source"] == "transcribed"
    assert rp["stimulus_asset"].endswith("pediatric_10plus_reading_passage_2.jpg")


def test_no_registry_or_flat_match_logs_warning(descriptions, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="b2aiprep.prepare.fhir_utils"):
        m = _resolve(descriptions, "totally-unknown-task-xyz")
    assert m["instructions"] == ""
    assert "stimulus_text" not in m  # nothing resolved
    assert any("no task match" in r.getMessage() for r in caplog.records)


def test_version_aware_resolution():
    from b2aiprep.prepare.fhir_utils import _resolve_task_registry, _registry_numbering_status

    # bare (version-less) name -> v1 task (not v2, despite alias_index ordering)
    v1 = _resolve_task_registry("maximum-phonation-time-3")
    assert v1[0]["task_id"] == "adult.maximum-phonation-time.v1"
    assert _registry_numbering_status(v1[0], "maximum-phonation-time-3") == "ok"  # v1 has 3
    # "(v2)" marker -> v2 task; index 3 exceeds its 2 recordings
    v2 = _resolve_task_registry("maximum-phonation-time-(v2)-3")
    assert v2[0]["task_id"] == "adult.maximum-phonation-time.v2"
    assert _registry_numbering_status(v2[0], "maximum-phonation-time-(v2)-3") == "out-of-range"


def test_registry_numbering_status():
    from b2aiprep.prepare.fhir_utils import _load_registry, _registry_numbering_status

    tasks = _load_registry()["tasks"]
    # bounded by recording_count (uniform), bank (banked), and nested recordings
    assert _registry_numbering_status(tasks["adult.maximum-phonation-time.v2"],
                                      "maximum-phonation-time-(v2)-2") == "ok"
    assert _registry_numbering_status(tasks["adult.maximum-phonation-time.v2"],
                                      "maximum-phonation-time-(v2)-5") == "out-of-range"
    assert _registry_numbering_status(tasks["pediatric.repeating-sentences"],
                                      "repeating-sentences-9") == "out-of-range"
    assert _registry_numbering_status(tasks["adult.harvard-sentences"],
                                      "harvard-sentences-list-3-11") == "out-of-range"


def test_out_of_range_index_warns_but_resolves(descriptions, caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="b2aiprep.prepare.fhir_utils"):
        m = _resolve(descriptions, "Maximum phonation time (v2)-5")
    assert m["instructions"] != ""  # still resolved (uniform instruction)
    assert any("outside the known numbering" in r.getMessage() for r in caplog.records)


def test_unknown_task_warns_but_best_guesses(descriptions, caplog):
    import logging

    # "rainbowpassage" (missed space) has no registry match; the registry -- the
    # authority -- flags it unknown, but a flat best-guess still populates it.
    with caplog.at_level(logging.WARNING, logger="b2aiprep.prepare.fhir_utils"):
        m = _resolve(descriptions, "RainbowPassage")
    assert m["instructions"] != ""
    assert any("no match in the task registry" in r.getMessage() for r in caplog.records)


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
    # story-recall is versioned: bare (v1) is the grandfather / "Banana Oil"
    # passage; (v2) is the boy-and-frog story. Each carries its own reference.
    v1 = _resolve(descriptions, "story-recall")
    assert v1["speech_type"] == "recall"
    assert v1["stimulus_text"].startswith("You wished to know all about my grandfather")
    v2 = _resolve(descriptions, "story-recall-(v2)")
    assert v2["stimulus_text"].startswith("There was once a boy")


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


def test_random_item_no_lookup_yields_empty_stimulus(descriptions):
    # Regression: random-item-generation must not concatenate its two
    # mutually-exclusive instruction variants into stimulus_text when no
    # questionnaire row is available. The per-participant category is joined
    # from the questionnaire; the reference text is otherwise empty.
    for name in ("Random-Item-Generation", "Random-Item-Generation-(v2)"):
        m = _resolve(descriptions, name)
        assert m["stimulus_text"] == ""
        assert m["speech_type"] == "elicited"


def test_questionnaire_join_absent_is_noop(descriptions):
    # No lookup provided -> vocab stimulus_text stays empty, no crash.
    vocab = _resolve(descriptions, "Productive-Vocabulary-3", questionnaire_lookup=None)
    assert vocab["stimulus_text"] == ""
    # A lookup missing this task id -> also empty.
    vocab2 = _resolve(descriptions, "Productive-Vocabulary-3", {}, "AT-MISSING")
    assert vocab2["stimulus_text"] == ""


def test_identifying_pictures_read_target(descriptions):
    # The target word is printed on each flashcard and is in neither the task name
    # nor the instructions; it is transcribed into an index-keyed bank and emitted
    # as a read (WER-scorable) reference, with the image asset still pinned.
    m = _resolve(descriptions, "Identifying Pictures-10", population="pediatric")
    assert m["speech_type"] == "read"
    assert m["stimulus_text"] == "WEB"
    assert m["stimulus_source"] == "transcribed"
    assert m["stimulus_asset"].endswith("pediatric_identifying_pictures_10.jpg")
    assert _resolve(descriptions, "Identifying Pictures-1", population="pediatric")["stimulus_text"] == "MY"
    assert _resolve(descriptions, "Identifying Pictures-37", population="pediatric")["stimulus_text"] == "TEETH"


def test_sound_cards_nonlexical_with_printed_token(descriptions):
    # Noisy/Silly/Long Sounds print the target sound on the card. Capture it as the
    # (verbatim) stimulus_text but keep speech_type non-lexical: it is a sound
    # repeated to the time limit, not a WER target. Image asset stays pinned.
    m = _resolve(descriptions, "Noisy Sounds-3", population="pediatric")
    assert m["speech_type"] == "non-lexical"
    assert m["stimulus_text"] == "oo oo oo"
    assert m["stimulus_source"] == "transcribed"
    assert m["stimulus_asset"].endswith("pediatric_noisy_sounds_3.jpg")
    assert _resolve(descriptions, "Silly Sounds-1", population="pediatric")["stimulus_text"].startswith("puh puh")


def test_identifying_pictures_bank_matches_recording_count():
    # Exactly one transcribed word per card the registry declares.
    from importlib.resources import files as _files
    base = _files("b2aiprep.prepare.resources.task_registry")
    reg = json.loads(base.joinpath("registry.json").read_text())
    bank = json.loads(base.joinpath("identifying_pictures_bank.json").read_text())
    assert len(bank["words"]) == reg["tasks"]["pediatric.identifying-pictures"]["recording_count"] == 37


def test_story_recall_v2_image_sequence_scalar(descriptions):
    # Story Recall v2 shows a wordless 10-panel picture story before the retelling.
    # Capture it as ONE scalar template URL (NOT a list -> the sidecar stays flat /
    # parquet-friendly): a '{n}' placeholder + ' [n=1..N]' range. Language-independent
    # (wordless); the recall reference narrative stays a scalar string.
    m = _resolve(descriptions, "Story Recall-(v2)", population="adult")
    a = m["stimulus_asset"]
    assert isinstance(a, str) and a.startswith(
        "https://raw.githubusercontent.com/eipm/bridge2ai-redcap/"
    )
    assert a.endswith("/StoryRecall_{n}.jpg")           # '{n}' kept literal, not %7B
    assert "%20" in a                                   # path spaces url-encoded
    # the count lives in its own scalar field, not baked into the URL
    assert m["stimulus_asset_n_images"] == 10
    assert m["speech_type"] == "recall"
    assert isinstance(m["stimulus_text"], str) and m["stimulus_text"].startswith("There was once a boy")
    # v1 (Grandfather Passage) is text-only -- no image sequence, no count
    v1 = _resolve(descriptions, "Story Recall-1", population="adult")
    assert v1.get("stimulus_asset") is None
    assert v1.get("stimulus_asset_n_images") is None


def test_single_image_tasks_omit_n_images(descriptions):
    # A single, directly-resolvable image carries NO stimulus_asset_n_images -- its
    # absence is the documented signal that stimulus_asset needs no {n} expansion.
    for name, pop in [("Identifying Pictures-10", "pediatric"),
                      ("Picture Description", "pediatric"),
                      ("Picture description", "adult")]:
        m = _resolve(descriptions, name, population=pop)
        assert m["stimulus_asset"] and "{n}" not in m["stimulus_asset"]
        assert "stimulus_asset_n_images" not in m


def test_metadata_bundle_tsv_null_equivalence(descriptions):
    # Mirror the metadata bundling step (commands.py create_bundled_dataset:
    # records=[sidecar dicts] -> pd.DataFrame -> to_csv(sep='\t')). A sequence task
    # carries stimulus_asset_n_images; single-image / no-image sidecars OMIT the key
    # (JSON), which must materialize as a null column cell (TSV/parquet) and round-
    # trip as NaN == absent. Also proves the '{n}' template survives the TSV.
    import io
    import pandas as pd

    recs = [
        _resolve(descriptions, "Story Recall-(v2)", population="adult"),      # sequence -> 10
        _resolve(descriptions, "Identifying Pictures-10", population="pediatric"),  # single image, no count
        _resolve(descriptions, "Rainbow Passage", population="adult"),         # no asset at all
    ]
    # the single-image / no-asset sidecars must NOT carry the key (absent in JSON)
    assert "stimulus_asset_n_images" not in recs[1]
    assert "stimulus_asset_n_images" not in recs[2]

    df = pd.DataFrame(recs)
    assert "stimulus_asset_n_images" in df.columns  # union of keys -> column exists
    # bundler casts to nullable Int64 so the count stays an integer (not 10.0) while
    # absent rows write as an empty cell.
    df["stimulus_asset_n_images"] = df["stimulus_asset_n_images"].astype("Int64")
    buf = io.StringIO()
    df.to_csv(buf, sep="\t", index=False)
    tsv = buf.getvalue()

    # on-disk cells: integer '10' (NOT '10.0') for the sequence, empty otherwise
    hdr = tsv.splitlines()[0].split("\t")
    ci, ti = hdr.index("stimulus_asset_n_images"), hdr.index("task_name")
    cells = {ln.split("\t")[ti]: ln.split("\t")[ci] for ln in tsv.splitlines()[1:]}
    assert cells["story-recall-(v2)"] == "10"
    assert cells["identifying-pictures-10"] == ""
    assert cells["rainbow-passage"] == ""

    rt = pd.read_csv(io.StringIO(tsv), sep="\t")
    by = {r["task_name"]: r for _, r in rt.iterrows()}
    assert int(by["story-recall-(v2)"]["stimulus_asset_n_images"]) == 10
    assert "{n}" in by["story-recall-(v2)"]["stimulus_asset"]
    # absent-key sidecars -> null table cells (== absent)
    assert pd.isna(by["identifying-pictures-10"]["stimulus_asset_n_images"])
    assert pd.isna(by["rainbow-passage"]["stimulus_asset_n_images"])
