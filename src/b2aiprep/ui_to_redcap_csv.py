import os
from pydub.utils import mediainfo
import argparse
from collections import OrderedDict
from pathlib import Path
import re
import pandas as pd
import requests
import json
from datetime import datetime


def parse_survey(survey_data, record_id, session_path):
    session_id = session_path.split("/")[0]
    questionnaire_name = survey_data[0]["used"][1].split("/")[-1]
    questions_answers = dict()
    questions_answers["record_id"] = [record_id]
    questions_answers["redcap_repeat_instrument"] = [questionnaire_name]
    questions_answers["redcap_repeat_instance"] = [1]
    start_time = survey_data[0]["startedAtTime"]
    end_time = survey_data[0]["endedAtTime"]
    for i in range(len(survey_data)):
        if i % 2 == 1:  # odd index contains the answer
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

    duration = calculate_duration(start_time, end_time)
    questions_answers[f"{questionnaire_name}_duration"] = [duration]

    questions_answers[f"{questionnaire_name}_sessionId"] = [session_id]

    df = pd.DataFrame(questions_answers)
    return [df]


def load_json_files(directory):
    json_data = []

    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                json_data += data

    return dict.fromkeys(json_data)


def calculate_duration(start_time, end_time):
    # Convert the time strings to datetime objects with UTC format
    time_format = "%Y-%m-%dT%H:%M:%S.%fZ"
    start = datetime.strptime(start_time, time_format)
    end = datetime.strptime(end_time, time_format)

    # Calculate the difference between the two times
    duration = end - start

    # Get the duration in milliseconds
    milliseconds = duration.microseconds // 1000

    return milliseconds


def fetch_json_options_number(raw_url):
    try:
        # fix url due to the split
        raw_url = raw_url.replace("combined", "questionnaires")
        # Make a GET request to the raw URL
        response = requests.get(raw_url, verify=True)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)

        # Parse the JSON data
        json_data = response.json()
        return len(json_data["responseOptions"]["choices"])

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return
    except ValueError:
        print("Error parsing JSON data")


def parse_audio(audio_list):
    # peds specific tasks
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

    flattened_list = [value for key in protocol_order for value in protocol_order[key]]

    audio_output_list = []
    count = 1
    acoustic_count = 1
    acoustic_prev = None
    for file_path in (flattened_list):

        record_id = Path(file_path).parent.parent.name
        session = Path(file_path).parent.name
        info = mediainfo(file_path)
        duration = float(info['duration'])
        file_name = file_path.split("/")[-1]
        file_size = os.path.getsize(file_path)
        recording_id = re.search(r'([a-f0-9\-]{36})\.', file_name).group(1)
        acoustic_task = re.search(r"^(.*?)(_\d+)", file_name).group(1)
        if acoustic_prev != acoustic_task:
            acoustic_count = 1
        file_dict = {
            "record_id": record_id,
            "redcap_repeat_instrument": "Recording",
            "redcap_repeat_instance": count,
            "recording_id": recording_id,
            "recording_acoustic_task_id" : f"{acoustic_task}-{acoustic_count}",
            "recording_session_id": session,
            "recording_name": f"{recording_id}.wav",
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # string param of path to folder containing reproschema files
    parser.add_argument("audio_dir",
                        type=str,
                        help="path to folder containing audio files")
    parser.add_argument("output_dir",
                        type=str,
                        help="path to where to store audio redcap csv")

    parser.add_argument("survey_file",
                        type=str,
                        help="path to folder containing survey files")
    parser.add_argument("redcap_csv",
                        type=str,
                        help="path to where to store redcap csv")

    args = parser.parse_args()

    folder = Path(args.survey_file)
    sub_folders = sorted([os.path.join(folder, f)
                         for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))])
    if not os.path.isdir(folder):
        raise FileNotFoundError(
            f"{folder} does not exist. "
        )

    merged_questionnaire_data = []
    # load each file recursively within the folder into its own key
    for subject in sub_folders:
        content = OrderedDict()
        for file in Path(subject).glob("**/*"):
            if file.is_file():
                filename = str(file.relative_to(subject))
                with open(f"{subject}/{filename}") as f:
                    content[filename] = json.loads(f.read())

        record_id = args.survey_file.split("/")[-1]

        for questionnaire in content.keys():
            try:
                merged_questionnaire_data += (parse_survey(content[questionnaire], (subject.split(
                    "/")[-1]).split()[0], questionnaire))
            except Exception:
                continue

    survey_df = pd.concat(merged_questionnaire_data, ignore_index=True)
    os.makedirs(args.redcap_csv, exist_ok=True)
    filename = "survey_redcap.csv"
    file_path = os.path.join(args.redcap_csv, filename)
    survey_df.to_csv(file_path, index=False)

    audio_folders = Path(args.audio_dir)
    audio_sub_folders = sorted([os.path.join(audio_folders, f) for f in os.listdir(
        audio_folders) if os.path.isdir(os.path.join(audio_folders, f))])

    if not os.path.isdir(audio_folders):
        raise FileNotFoundError(
            f"{audio_folders} does not exist."
        )
    merged_csv = []
    for subject in audio_sub_folders:
        audio_list = []
        for file in Path(subject).glob("**/*"):
            if file.is_file() and str(file).endswith(".wav"):
                audio_list.append(str(file))

        merged_csv += parse_audio(audio_list)

    csv_path = os.path.join(args.output_dir, "audio-redcap.csv")
    audio_df = pd.DataFrame(merged_csv)
    audio_df.to_csv(csv_path, index=False)

    merged_df = [survey_df, audio_df]
    output_df = pd.concat(merged_df , ignore_index=True)
    merged_csv_path = os.path.join(args.output_dir, "merged-redcap.csv")
    output_df.to_csv(merged_csv_path, index=False)
