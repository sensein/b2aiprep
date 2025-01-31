from pathlib import Path
import re
import pandas as pd
from datetime import datetime
from b2aiprep.prepare.utils import fetch_json_options_number, get_wav_duration


def parse_survey(survey_data, record_id, session_path):
    """
    Function that generates a list of data frames in order to generate a redcap csv
    Args: 
        survey_data is the raw json generated from reproschema ui
        record_id is the id tat identifies the participant
        session_path is the path containing the session id
    """
    session_id = session_path.split("/")[0]
    questionnaire_name = survey_data[0]["used"][1].split("/")[-1]
    questions_answers = dict()
    questions_answers["record_id"] = [record_id]
    questions_answers["redcap_repeat_instrument"] = [questionnaire_name]
    questions_answers["redcap_repeat_instance"] = [1]
    start_time = survey_data[0]["startedAtTime"]
    end_time = survey_data[0]["endedAtTime"]
    for i in range(len(survey_data)):
        if survey_data[i]["@type"] == "reproschema:Response":
            question = survey_data[i]["isAbout"].split("/")[-1]
            answer = survey_data[i]["value"]
            if not isinstance(answer, list):
                questions_answers[question] = [str(answer).capitalize()]

            else:
                num = fetch_json_options_number(survey_data[i]["isAbout"])

                for options in range(num):
                    if options in answer:
                        questions_answers[f"""{question}___{options}"""] = [
                            "Checked"]

                    else:
                        questions_answers[f"""{question}___{options}"""] = [
                            "Unchecked"]

        else:
            end_time = survey_data[i]["endedAtTime"]
    # Adding metadata values for redcap
    questions_answers[f"{questionnaire_name}_start_time"] = [start_time]
    questions_answers[f"{questionnaire_name}_end_time"] = [end_time]

    # Convert the time strings to datetime objects with UTC format
    time_format = "%Y-%m-%dT%H:%M:%S.%fZ"
    start = datetime.strptime(start_time, time_format)
    end = datetime.strptime(end_time, time_format)
    duration = end - start
    # convert to milliseconds
    duration = duration.microseconds // 1000

    questions_answers[f"{questionnaire_name}_duration"] = [duration]

    questions_answers[f"{questionnaire_name}_session_id"] = [session_id]

    df = pd.DataFrame(questions_answers)
    return [df]

def parse_audio(audio_list, dummy_audio_files=False):
    """
    Function that generates a list of Json's to be converted into a redcap csv based on audio files.
    Args: 
        audio_list is a list of paths to each audio files
        dummy_audio_files is an optional variable for testing 
    """
    protocol_order = {
        "ready_for_school": [],
        "favorite_show_movie": [],
        "favorite_food": [],
        "outside_of_school": [],
        "abcs": [],
        "123s": [],
        "naming_animals": [],
        "role_naming": [],
        "naming_food": [],
        "noisy_sounds": [],
        "long_sounds": [],
        "silly_sounds": [],
        "picture": [],
        "sentence": [],
        "picture_description": [],
        "passage": [],
        "other": []
    }

    for name in audio_list:
        found = False
        for order in protocol_order:
            if order in name:
                protocol_order[order].append(name)
                found = True
                break
        if not found:
            protocol_order["other"].append(name)

    for questions in protocol_order:
        protocol_order[questions] = sorted(protocol_order[questions])

    flattened_list = [
        value for key in protocol_order for value in protocol_order[key]]

    audio_output_list = []
    count = 1
    acoustic_task_count = 1
    acoustic_count = 1
    acoustic_prev = None
    acoustic_tasks = set()
    #acoustic_task_dict = dict()
    for file_path in (flattened_list):

        record_id = Path(file_path).parent.parent.name
        session = Path(file_path).parent.name
        if dummy_audio_files:
            duration = 0
            file_size = 0
        else:
            duration = get_wav_duration(file_path)
            file_size = Path(file_path).stat().st_size
        file_name = file_path.split("/")[-1]
        recording_id = re.search(r'([a-f0-9\-]{36})\.', file_name).group(1)
        acoustic_task = re.search(r"^(.*?)(_\d+)", file_name).group(1)
        if acoustic_task not in acoustic_tasks:
            
            acoustic_tasks.add(acoustic_task)
            acoustic_task_dict = {
                "record_id" : record_id,
                "redcap_repeat_instrument": "Acoustic Task",
                "redcap_repeat_instance": acoustic_task_count ,
                "acoustic_task_id": f"{acoustic_task}-{session}",
                "acoustic_task_session_id": session,
                "acoustic_task_name": acoustic_task,
                "acoustic_task_cohort": "Pediatrics",
                "acoustic_task_status": "Completed",
                "acoustic_task_duration": duration
            }
            audio_output_list.append(acoustic_task_dict)
            acoustic_task_count += 1
        else:
            for index in audio_output_list:
                if "acoustic_task_id" in index:
                    if acoustic_task == index["acoustic_task_name"]:
                        index["acoustic_task_duration"] += duration

        if acoustic_prev != acoustic_task:
            acoustic_count = 1
        file_dict = {
            "record_id": record_id,
            "redcap_repeat_instrument": "Recording",
            "redcap_repeat_instance": count,
            "recording_id": recording_id,
            "recording_acoustic_task_id": f"{acoustic_task}-{session}",
            "recording_session_id": session,
            "recording_name": f"{acoustic_task}-{acoustic_count}",
            "recording_duration": duration,
            "recording_size": file_size,
            "recording_profile_name": "Speech",
            "recording_profile_version": "v1.0.0",
            "recording_input_gain": 0.0,
            "recording_microphone": "ipad"
        }

        acoustic_prev = acoustic_task
        acoustic_count += 1
        audio_output_list.append(file_dict)
        count += 1

    return audio_output_list
