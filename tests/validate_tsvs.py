import os
import json
import pandas as pd
import math
import argparse

def extract_data_elements(name, questionnaire):
    columns = {}
    name = name.replace(".json", "")
    for questionnaire_name in questionnaire.keys():
        questions = list(
            questionnaire[questionnaire_name]["data_elements"].keys())
        for question in questions:
            if ("minValue" in questionnaire[questionnaire_name]["data_elements"][question]
                    and "maxValue" in questionnaire[questionnaire_name]["data_elements"][question]):
                numerical_l = []
                for i in range(101):
                    option = {"name": {"en": i}, "value": i}
                    numerical_l.append(option)
                columns[name] = numerical_l
                break
            else:
                if "choices" in questionnaire[questionnaire_name]["data_elements"][question]:
                    choices = questionnaire[questionnaire_name]["data_elements"][question]["choices"]
                else:
                    choices = None

                q = {question: choices}
                if name not in columns:
                    columns[name] = [q]
                else:
                    columns[name].append(q)

    return columns


def get_json_data_from_folder(folder_path):

    file_json_data = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.endswith('.json') and os.path.isfile(file_path):
            try:
                with open(file_path, 'r') as f:
                    file_json_data[filename] = json.load(f)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return file_json_data


def read_tsv_files_from_folder_to_dict(folder_path):
    dataframes_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.tsv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path, sep='\t')

            file_key = os.path.splitext(filename)[0]

            dataframes_dict[file_key] = df

    return dataframes_dict


def validate(questionnaire, dfs, answer_key):
    questionnaire_data = dfs
    column_names = list(questionnaire_data.columns)
    for column in column_names:
        for index in questionnaire_data[column]:
            try:
                # ignore check and unchecked cells due to it being redcap specific
                if index in ("Unchecked", "Checked"):
                    pass
                elif isinstance(index, float) and math.isnan(index):
                    pass
                else:
                    choices = get_choices(
                        answer_key[questionnaire], column)
                    if choices is not None:
                        if isinstance(index, float) and index.is_integer():
                            index = str(int(index))
                        assert index in choices
            except Exception:
                with open('error_log.txt', 'a') as file:
                    file.write(
                        f"{questionnaire} value: {index} in columns {column} does not match the json phenotype \n")
                    file.write(f"Columns: {column}\n")
                    file.write(f"Value: {index}\n")
                    file.write(f"Questionnaire: {questionnaire}\n")
                    file.write(f"Choices: {choices}\n")
                    file.write("\n")


def get_choices(questionnaire, column):
    for i in range(len(questionnaire)):
        if column in questionnaire[i]:
            if isinstance(questionnaire[i][column], list):
                return [item['name']['en'].strip() for item in questionnaire[i][column]]
            else:
                return None
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inputs for phenotype jsons and tsv data")

    # Add two arguments
    parser.add_argument("phenotype_path", type=str, help="input path to phenotype json")
    parser.add_argument("tsv_path", type=str, help="path to tsv data")

    # Parse the arguments
    args = parser.parse_args()

    folder_path_jsons = args.phenotype_path 
    folder_path_tsvs = args.tsv_path 
    data = get_json_data_from_folder(folder_path_jsons)
    columns = dict()
    for questionnaire in data:
        columns.update(extract_data_elements(
            questionnaire, data[questionnaire]))

    dfs = (read_tsv_files_from_folder_to_dict(folder_path_tsvs))
    for questionnaire in dfs:
        validate(questionnaire, dfs[questionnaire], columns)
