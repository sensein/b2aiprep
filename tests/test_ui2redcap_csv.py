import pytest
from b2aiprep.prepare.reproschema_to_redcap import parse_survey, parse_audio
import pandas as pd


def test_parse_survey():
    record_id = "99999"
    session_path = "f09f394a-7805-4b38-9a08-540ea5e4fd02-/activity_0.jsonld"
    contents = [
        {"@context": "https://raw.githubusercontent.com/ReproNim/reproschema/1.0.0/contexts/reproschema",
         "@type": "reproschema:ResponseActivity", "@id": "uuid:343d79cb-806e-4ee4-96f4-c2d8f06c7bc5",
         "used": ["https://raw.githubusercontent.com/kind-lab/b2ai-peds-protocol/2.2.3/peds-protocol-questionnaires/activities/subjectparticipant_basic_information/items/record_id",
                  "https://raw.githubusercontent.com/kind-lab/b2ai-peds-protocol/2.2.3/peds-protocol-questionnaires/activities/subjectparticipant_basic_information/subjectparticipant_basic_information_schema",
                  "https://raw.githubusercontent.com/kind-lab/b2ai-peds-protocol/2.2.3/peds-protocol-questionnaires/peds-protocol/peds-protocol.json"],
         "inLanguage": "en",
         "startedAtTime": "2025-01-24T15:16:07.241Z",
         "endedAtTime": "2025-01-24T15:16:14.926Z",
         "wasAssociatedWith": {"version": "1.0.0",
                               "url": "https://www.repronim.org/reproschema-ui/",
                               "@id": "https://github.com/ReproNim/reproschema-ui"},
         "generated": "uuid:8bd00a7a-4efa-454c-9eb7-37b6b1fca3f6"}]

    expected_df = pd.DataFrame(
        {'record_id': ['99999'],
         'redcap_repeat_instrument': ['subjectparticipant_basic_information_schema'],
         'redcap_repeat_instance': [1],
         'subjectparticipant_basic_information_schema_start_time': ['2025-01-24T15:16:07.241Z'],
         'subjectparticipant_basic_information_schema_end_time': ['2025-01-24T15:16:14.926Z'],
         'subjectparticipant_basic_information_schema_duration': [685],
         'subjectparticipant_basic_information_schema_sessionId': ['f09f394a-7805-4b38-9a08-540ea5e4fd02-']})

    actual_df = parse_survey(contents, record_id, session_path)[0]

    assert expected_df.equals(actual_df)


def test_audio_csv():
    audio_files = ['mock_audio/99999/14829893-3879-4723-932b-d98d6bc356a8-/long_sounds_task_1_10_plus-d6d7411f-c934-4bca-91a7-5ddad353b801.wav',
                   'mock_audio/99999/14829893-3879-4723-932b-d98d6bc356a8-/long_sounds_task_2_10_plus-69181d90-aae4-49d0-861d-4ff4a939bff0.wav']

    expected_output = [
        {'record_id': '99999', 'redcap_repeat_instrument': 'Recording', 'redcap_repeat_instance': 1, 'recording_id': 'd6d7411f-c934-4bca-91a7-5ddad353b801', 'recording_acoustic_task_id': 'long_sounds_task-1', 'recording_session_id': '14829893-3879-4723-932b-d98d6bc356a8-',
            'recording_name': 'd6d7411f-c934-4bca-91a7-5ddad353b801.wav', 'recording_duration': 2.432, 'recording_size': 466988, 'recording_profile_name': 'Speech', 'recording_profile_version': 'v1.0.0', 'recording_input_gain': 0.0, 'recording_microphone': 'ipad'},
        {'record_id': '99999', 'redcap_repeat_instrument': 'Recording', 'redcap_repeat_instance': 2, 'recording_id': '69181d90-aae4-49d0-861d-4ff4a939bff0', 'recording_acoustic_task_id': 'long_sounds_task-2', 'recording_session_id': '14829893-3879-4723-932b-d98d6bc356a8-', 
            'recording_name': '69181d90-aae4-49d0-861d-4ff4a939bff0.wav', 'recording_duration': 2.3893333333333335, 'recording_size': 458796, 'recording_profile_name': 'Speech', 'recording_profile_version': 'v1.0.0', 'recording_input_gain': 0.0, 'recording_microphone': 'ipad'}]
    
    actual = parse_audio(audio_files)
    
    assert expected_output == actual
