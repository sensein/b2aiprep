import pandas as pd
import argparse

def create_questionnaire_redcap(redcap_csv, consent_csv, output_path):
    # Paths to your CSV files
  

    df = pd.read_csv(redcap_csv, dtype="str")
    dates = pd.read_csv(consent_csv)

    record_ids = df["record_id"].unique()

    for id in record_ids:
        # consent was captured externally so we need to add it to the csv
        consent_date = dates.loc[dates["record_id"] == id, "consent_date"].iloc[0]
        age = df.loc[df["record_id"] == id, "age"].iloc[0]
        age = int(age)
        eligible_studies = None
        if age >= 2 and age < 4:
            eligible_studies = "age_2_4"
        elif age >= 4 and age < 6:
            eligible_studies = "age_4_6"
        elif age >= 6 and age < 10:
            eligible_studies = "age_6_10"
        elif age >= 10:
            eligible_studies = "age_10_plus"

        feas = pd.DataFrame(
            [
                {
                    "record_id": id,
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
    ]


    df.drop(columns=columns_list, axis=1, inplace=True)


    df.to_csv(f"{output_path}/questionnaire-may13.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser example")
    parser.add_argument("redcap_csv", type=str)
    parser.add_argument("consent_csv", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    create_questionnaire_redcap(args.redcap_csv, args.consent_csv, args.output_path)