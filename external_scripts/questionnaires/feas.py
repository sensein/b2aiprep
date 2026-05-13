import json
import pandas as pd
import argparse


def create_questionnaire_redcap(redcap_csv, consent_csv, output_path, uuid_map_path=None):
    df = pd.read_csv(redcap_csv, dtype="str")
    dates = pd.read_csv(consent_csv, dtype="str")

    # Load uuid remap if provided: {original_id: uuid}
    uuid_map = {}
    if uuid_map_path:
        with open(uuid_map_path, "r") as f:
            uuid_map = json.load(f)

    # consent_csv still uses original record_ids, so match on participant_study_id
    record_ids = df["participant_study_id"].unique()

    for original_id in record_ids:
        match = dates.loc[dates["record_id"] == original_id, "consent_date"]
        if match.empty:
            print(f"No consent date found for record_id {original_id}, skipping.")
            continue
        consent_date = match.iloc[0]

        age_match = df.loc[df["participant_study_id"] == original_id, "age"]
        if age_match.empty:
            print(f"No age found for record_id {original_id}, skipping.")
            continue
        age = int(age_match.iloc[0])

        eligible_studies = None
        if age >= 2 and age < 4:
            eligible_studies = "age_2_4"
        elif age >= 4 and age < 6:
            eligible_studies = "age_4_6"
        elif age >= 6 and age < 10:
            eligible_studies = "age_6_10"
        elif age >= 10:
            eligible_studies = "age_10_plus"

        # Use uuid if available, otherwise fall back to original id
        new_record_id = uuid_map.get(original_id, original_id)

        feas = pd.DataFrame(
            [
                {
                    "record_id": new_record_id,
                    #"participant_study_id": original_id,
                    "selected_language": "1",
                    "consent_method": "paper",
                    "enrolled": "1",
                    "consent_date": consent_date,
                    "is_feasibility_participant": "no",
                    "enrollment_institution": "sickkids",
                    "is_control_participant": "no",
                    f"eligible_studies___{eligible_studies}": "1",
                }
            ]
        )
        df = pd.concat([df, feas], ignore_index=True)

    columns_list = [
        "session_status",
        "session_is_control_participant",
        "session_duration",
        "session_site",
        "acoustic_task_id",
        "acoustic_task_session_id",
        "acoustic_task_name",
        "acoustic_task_cohort",
        "acoustic_task_status",
        "acoustic_task_duration",
        "recording_id",
        "recording_acoustic_task_id",
        "recording_session_id",
        "recording_name",
        "recording_duration",
        "recording_size",
        "recording_profile_name",
        "recording_profile_version",
        "recording_microphone",
        "participant_study_id"
    ]

    df.drop(columns=[c for c in columns_list if c in df.columns], axis=1, inplace=True)

    df.to_csv(f"{output_path}/questionnaire-may13.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser example")
    parser.add_argument("redcap_csv", type=str)
    parser.add_argument("consent_csv", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--uuid_map_path", type=str, default=None, help="Path to JSON file mapping original record_id to uuid")
    args = parser.parse_args()
    create_questionnaire_redcap(args.redcap_csv, args.consent_csv, args.output_path, args.uuid_map_path)