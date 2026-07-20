import numpy as np
from b2aiprep.prepare.redcap import parse_survey, parse_audio
import pandas as pd


def test_parse_survey():
    record_id = "99999"
    session_path = "7896a213-caca-499b-941f-da5eb4d44566-/activity_0.jsonld"
    contents = [
        {
            "@context": "https://raw.githubusercontent.com/ReproNim/reproschema/1.0.0/contexts/reproschema",
            "@type": "reproschema:ResponseActivity",
            "@id": "uuid:016fadb8-f0af-42b6-8140-6b5b2a2cc9b8",
            "used": [
                "https://raw.githubusercontent.com/kind-lab/b2ai-peds-protocol/2.2.3/peds-protocol-questionnaires/activities/subjectparticipant_contact_information/items/age",
                "https://raw.githubusercontent.com/kind-lab/b2ai-peds-protocol/2.2.3/peds-protocol-questionnaires/activities/subjectparticipant_contact_information/subjectparticipant_contact_information_schema",
                "https://raw.githubusercontent.com/kind-lab/b2ai-peds-protocol/2.2.3/peds-protocol-questionnaires/peds-protocol/peds-protocol.json",
            ],
            "inLanguage": "en",
            "startedAtTime": "2025-01-31T16:13:28.937Z",
            "endedAtTime": "2025-01-31T16:14:09.746Z",
            "wasAssociatedWith": {
                "version": "1.0.0",
                "url": "https://www.repronim.org/reproschema-ui/",
                "@id": "https://github.com/ReproNim/reproschema-ui",
            },
            "generated": "uuid:c468295d-01d7-4514-8685-450375af3855",
        },
        {
            "@context": "https://raw.githubusercontent.com/ReproNim/reproschema/1.0.0/contexts/reproschema",
            "@type": "reproschema:Response",
            "@id": "uuid:c468295d-01d7-4514-8685-450375af3855",
            "wasAttributedTo": {"@id": "d974992b-da4b-4bfc-9135-9b0bd5736e5a"},
            "isAbout": "https://raw.githubusercontent.com/kind-lab/b2ai-peds-protocol/2.2.3/peds-protocol-questionnaires/activities/subjectparticipant_contact_information/items/age",
            "value": "25",
        },
    ]

    expected_df = pd.DataFrame(
        {
            "record_id": ["99999"],
            "redcap_repeat_instrument": ["subjectparticipant_contact_information_schema"],
            "redcap_repeat_instance": [1],
            "age": ["25"],
            "subjectparticipant_contact_information_schema_start_time": [
                "2025-01-31T16:13:28.937Z"
            ],
            "subjectparticipant_contact_information_schema_end_time": ["2025-01-31T16:14:09.746Z"],
            "subjectparticipant_contact_information_schema_duration": [809],
            "subjectparticipant_contact_information_schema_session_id": [
                "7896a213-caca-499b-941f-da5eb4d44566-"
            ],
        }
    )

    actual_df = parse_survey(contents, record_id, session_path, {})[0]

    assert expected_df.equals(actual_df)


def test_audio_csv():
    audio_files = [
        "./mock_audio/99999/14829893-3879-4723-932b-d98d6bc356a8-/long_sounds_task_1_10_plus-d6d7411f-c934-4bca-91a7-5ddad353b801.wav",
        "./mock_audio/99999/14829893-3879-4723-932b-d98d6bc356a8-/long_sounds_task_2_10_plus-69181d90-aae4-49d0-861d-4ff4a939bff0.wav",
    ]

    expected_output = [
        {
            "record_id": "99999",
            "redcap_repeat_instrument": "Acoustic Task",
            "redcap_repeat_instance": 1,
            "acoustic_task_id": "14829893-3879-4723-932b-d98d6bc356a8-",
            "acoustic_task_session_id": "14829893-3879-4723-932b-d98d6bc356a8-",
            "acoustic_task_name": "long_sounds_task",
            "acoustic_task_cohort": "Pediatric",
            "acoustic_task_status": "Completed",
            "acoustic_task_completed_at": None,
            "acoustic_task_started_at": None,
            "acoustic_task_duration": 0,
            "acoustic_task_complete": "2",
        },
        {
            "record_id": "99999",
            "redcap_repeat_instrument": "Recording",
            "redcap_repeat_instance": 1,
            "recording_id": "d6d7411f-c934-4bca-91a7-5ddad353b801",
            "recording_acoustic_task_id": "14829893-3879-4723-932b-d98d6bc356a8-",
            "recording_session_id": "14829893-3879-4723-932b-d98d6bc356a8-",
            "recording_name": "long_sounds_task-1",
            "recording_duration": 0,
            "recording_size": 0,
            "recording_profile_name": "Speech",
            "recording_profile_version": "v1.0.0",
            "recording_input_gain": np.nan,
            "recording_microphone": "USB-C to 3.5mm Headphone Jack Adapter",
            "recording_filepath": f"/mounts/b2ai-api/Data/SickKids/d6d7411f-c934-4bca-91a7-5ddad353b801.wav",
            "recording_complete": "2",
            "recording_created_at": None,
            "recording_file_share": "",
            "recording_storage_account": "",
        },
        {
            "record_id": "99999",
            "redcap_repeat_instrument": "Recording",
            "redcap_repeat_instance": 2,
            "recording_id": "69181d90-aae4-49d0-861d-4ff4a939bff0",
            "recording_acoustic_task_id": "14829893-3879-4723-932b-d98d6bc356a8-",
            "recording_session_id": "14829893-3879-4723-932b-d98d6bc356a8-",
            "recording_name": "long_sounds_task-2",
            "recording_duration": 0,
            "recording_size": 0,
            "recording_complete": "2",
            "recording_created_at": None,
            "recording_file_share": "",
            "recording_storage_account": "",
            "recording_profile_name": "Speech",
            "recording_profile_version": "v1.0.0",
            "recording_input_gain": np.nan,
            "recording_microphone": "USB-C to 3.5mm Headphone Jack Adapter",
            "recording_filepath": f"/mounts/b2ai-api/Data/SickKids/69181d90-aae4-49d0-861d-4ff4a939bff0.wav",
        },
    ]

    actual = parse_audio(audio_files, True)
    assert expected_output == actual


# --------------------------------------------------------------------------- #
# Recording-numbering regression tests. `parse_audio` must label a task's Nth
# recording -N. Two failure modes we have seen / could see:
#   (1) >=10 recordings scrambled by lexicographic sort (noisy_sounds_10 -> -2)
#   (2) the age-band suffix (_10_plus / _6_to_10) mistaken for the recording
#       number (favorite_food_10_plus -> conversation(10+)-10; days_4_to_6 -> days-4)
# --------------------------------------------------------------------------- #
_SES = "14829893-3879-4723-932b-d98d6bc356a8-"


def _uuid(n):  # deterministic 36-char id in [a-f0-9-]
    return f"{n:08d}-0000-0000-0000-000000000000"


def _path(stem, n):
    return f"./mock_audio/99999/{_SES}/{stem}-{_uuid(n)}.wav"


def _recording_names(actual):
    return [r["recording_name"] for r in actual
            if r["redcap_repeat_instrument"] == "Recording"]


def test_numbering_double_digit_not_scrambled():
    """>=10 recordings, given out of order + with an age band: labels follow the
    filename index and come out in numeric order (1,2,3,10 -- not 1,10,2,3)."""
    files = [_path(f"noisy_sounds_task_{n}_6_to_10", n) for n in [3, 1, 10, 2]]
    actual = parse_audio(files, True)
    assert _recording_names(actual) == [
        "noisy_sounds_task-1", "noisy_sounds_task-2",
        "noisy_sounds_task-3", "noisy_sounds_task-10",
    ]
    by_id = {r["recording_id"]: r["recording_name"] for r in actual
             if r["redcap_repeat_instrument"] == "Recording"}
    assert by_id[_uuid(10)] == "noisy_sounds_task-10"   # not -2


def test_numbering_index_without_age_band():
    """Index-only filenames (no age band) still follow the filename number."""
    files = [_path(f"repeat_words_{n}", n) for n in [2, 1, 10]]
    assert _recording_names(parse_audio(files, True)) == [
        "repeat_words-1", "repeat_words-2", "repeat_words-10"]


def test_age_band_not_used_as_recording_number():
    """Tasks carrying only an age band (days/months/123s -- no per-recording
    index) are numbered positionally (=1), never by the age-band number (4)."""
    files = [_path("days_4_to_6", 1), _path("months_4_to_6", 2), _path("123s_4_to_6", 3)]
    names = set(_recording_names(parse_audio(files, True)))
    assert names == {"days-1", "months-1", "123s-1"}
    assert not any(n.endswith("-4") for n in names)   # age-band number must not leak


def test_conversation_positional_not_age_band(monkeypatch):
    """conversation sub-tasks (no filename index, only an age band) number
    positionally 1..4 by sub-task -- NOT by the age band (would be -6/-10)."""
    from b2aiprep.prepare import redcap as _rc
    monkeypatch.setattr(_rc, "get_age_from_jsonld", lambda p: 8)
    subs = ["favorite_food", "favorite_show_movie_game",
            "outside_of_school", "ready_for_school"]
    files = [_path(f"{s}_6_to_10", i) for i, s in enumerate(subs, 1)]
    assert _recording_names(parse_audio(files, True)) == [
        "conversation(6to10)-1", "conversation(6to10)-2",
        "conversation(6to10)-3", "conversation(6to10)-4",
    ]


def test_conversation_age_boundary_10(monkeypatch):
    """Age exactly 10 maps to conversation(10+), not conversation(2to4)."""
    from b2aiprep.prepare import redcap as _rc
    monkeypatch.setattr(_rc, "get_age_from_jsonld", lambda p: 10)
    names = _recording_names(parse_audio([_path("favorite_food_10_plus", 1)], True))
    assert names == ["conversation(10+)-1"]
