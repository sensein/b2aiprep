import pytest
from b2aiprep.prepare.reproschema_to_redcap import parse_survey, parse_audio
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
                "https://raw.githubusercontent.com/kind-lab/b2ai-peds-protocol/2.2.3/peds-protocol-questionnaires/peds-protocol/peds-protocol.json"
            ],
            "inLanguage": "en",
            "startedAtTime": "2025-01-31T16:13:28.937Z",
            "endedAtTime": "2025-01-31T16:14:09.746Z",
            "wasAssociatedWith": {
                "version": "1.0.0",
                "url": "https://www.repronim.org/reproschema-ui/",
                "@id": "https://github.com/ReproNim/reproschema-ui"
            },
            "generated": "uuid:c468295d-01d7-4514-8685-450375af3855"
        },
        {
            "@context": "https://raw.githubusercontent.com/ReproNim/reproschema/1.0.0/contexts/reproschema",
            "@type": "reproschema:Response",
            "@id": "uuid:c468295d-01d7-4514-8685-450375af3855",
            "wasAttributedTo": {
                "@id": "d974992b-da4b-4bfc-9135-9b0bd5736e5a"
            },
            "isAbout": "https://raw.githubusercontent.com/kind-lab/b2ai-peds-protocol/2.2.3/peds-protocol-questionnaires/activities/subjectparticipant_contact_information/items/age",
            "value": "25"
        }
    ]

    expected_df = pd.DataFrame({'record_id': ['99999'],
     'redcap_repeat_instrument': ['subjectparticipant_contact_information_schema'], 
     'redcap_repeat_instance': [1], 
     'age': ['25'], 
     'subjectparticipant_contact_information_schema_start_time': ['2025-01-31T16:13:28.937Z'], 
     'subjectparticipant_contact_information_schema_end_time': ['2025-01-31T16:14:09.746Z'], 
     'subjectparticipant_contact_information_schema_duration': [809], 
     'subjectparticipant_contact_information_schema_session_id': ['7896a213-caca-499b-941f-da5eb4d44566-']}
                                )

    actual_df = parse_survey(contents, record_id, session_path)[0]

    assert expected_df.equals(actual_df)



def test_audio_csv():
    audio_files = ['./mock_audio/99999/14829893-3879-4723-932b-d98d6bc356a8-/long_sounds_task_1_10_plus-d6d7411f-c934-4bca-91a7-5ddad353b801.wav',
                   './mock_audio/99999/14829893-3879-4723-932b-d98d6bc356a8-/long_sounds_task_2_10_plus-69181d90-aae4-49d0-861d-4ff4a939bff0.wav']

    expected_output = [{"record_id": "99999",
                        "redcap_repeat_instrument": "Acoustic Task",
                        "redcap_repeat_instance": 1,
                        "acoustic_task_id": "long_sounds_task-14829893-3879-4723-932b-d98d6bc356a8-",
                        "acoustic_task_session_id": "14829893-3879-4723-932b-d98d6bc356a8-",
                        "acoustic_task_name": "long_sounds_task",
                        "acoustic_task_cohort": "Pediatrics",
                        "acoustic_task_status": "Completed", "acoustic_task_duration": 0},

                       {"record_id": "99999",
                        "redcap_repeat_instrument": "Recording",
                        "redcap_repeat_instance": 1,
                        "recording_id": "d6d7411f-c934-4bca-91a7-5ddad353b801",
                        "recording_acoustic_task_id":
                        "long_sounds_task-14829893-3879-4723-932b-d98d6bc356a8-",
                        "recording_session_id": "14829893-3879-4723-932b-d98d6bc356a8-",
                        "recording_name": "long_sounds_task-1",
                        "recording_duration": 0,
                        "recording_size": 0,
                        "recording_profile_name": "Speech",
                        "recording_profile_version": "v1.0.0",
                        "recording_input_gain": 0.0,
                        "recording_microphone": "ipad"},

                       {"record_id": "99999",
                       "redcap_repeat_instrument": "Recording",
                        "redcap_repeat_instance": 2,
                        "recording_id": "69181d90-aae4-49d0-861d-4ff4a939bff0",
                        "recording_acoustic_task_id": "long_sounds_task-14829893-3879-4723-932b-d98d6bc356a8-",
                        "recording_session_id": "14829893-3879-4723-932b-d98d6bc356a8-",
                        "recording_name": "long_sounds_task-2",
                        "recording_duration": 0,
                        "recording_size": 0,
                        "recording_profile_name": "Speech",
                        "recording_profile_version": "v1.0.0",
                        "recording_input_gain": 0.0,
                        "recording_microphone": "ipad"}]

    actual = parse_audio(audio_files, True)
    assert expected_output == actual
