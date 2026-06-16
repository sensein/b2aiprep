import json
import pandas as pd
import argparse
import uuid

def mark_repeat_instruments_complete(df):
    """For any row with a value in redcap_repeat_instrument, set {value}_complete = "2",
    creating the column if it doesn't already exist."""
    if "redcap_repeat_instrument" not in df.columns:
        return df
 
    # Unique, non-empty instrument values
    instruments = df["redcap_repeat_instrument"].dropna()
    instruments = instruments[instruments.str.strip() != ""].unique()
 
    # Pre-create the *_complete columns to avoid fragmenting the frame
    for inst in instruments:
        col = f"{inst}_complete"
        if col not in df.columns:
            df[col] = pd.NA
 
    # Mark each row's matching column
    for idx, value in df["redcap_repeat_instrument"].items():
        if pd.notna(value) and str(value).strip() != "":
            df.at[idx, f"{value}_complete"] = "2"
 
    return df


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

        age_match = df.loc[df["participant_study_id"] == original_id, "age"].dropna()
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
                    "subjectparticipant_basic_information_complete": "2",
                    "subjectparticipant_contact_information_complete": "2",
                    "redcap_data_access_group": "sickkids",
                    "is_remote_data_collection_enabled": "no",
                    "exclude_participant": "no",
                    "inclusion_date": consent_date, # inclusion and consent are the same date
                    "data_dissemination_complete": "2",
                    "subjectparticipant_eligible_studies_complete": "2",
                    # "pediatric_q_generic_demographics_complete": "2",
                    # "pediatric_q_generic_vhi_10_complete": "2",
                    # "pediatric_q_generic_voice_outcome_survey_complete": "2",
                    # "pediatric_q_generic_voice_related_qol_survey_complete": "2",
                    # "pediatric_q_generic_phqa_complete": "2",
                    # "pediatric_q_generic_medical_conditions_complete": "2"
                    
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
    
    if "session_id" in df.columns:
        df["session_id"] = df["session_id"].apply(
            lambda x: str(uuid.UUID(x)) if pd.notna(x) else x
        )
    df = mark_repeat_instruments_complete(df)
 
    df.drop(columns=[c for c in columns_list if c in df.columns], axis=1, inplace=True)

    df.to_csv(f"{output_path}/questionnaire-final-jun15.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser example")
    parser.add_argument("redcap_csv", type=str)
    parser.add_argument("consent_csv", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("--uuid_map_path", type=str, default=None, help="Path to JSON file mapping original record_id to uuid")
    args = parser.parse_args()
    create_questionnaire_redcap(args.redcap_csv, args.consent_csv, args.output_path, args.uuid_map_path)