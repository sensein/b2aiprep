import uuid
import pandas as pd
import argparse
def create_recording_redcap(recording_csv, output_csv):
    # recording_csv = (
    #     #"/home/evan/Documents/publish_deid/b2aiprep/external_scripts/recordings_redcap/recordings.csv"
    #     "/home/evan/Documents/publish_deid/b2aiprep/external_scripts/may12-import/redcap-recording.csv"
    # )
    df = pd.read_csv(recording_csv, dtype="str")

    l = []
    for col in df.columns:
        if df[col].isnull().all():
            l.append(col)

    l += [
        "peds_mc_neck_mass___thyroglossal_duct_cyst",
        "peds_mc_neck_mass___branchial_cleft_cyst",
        "peds_mc_neck_mass___dermoid_cyst",
        "peds_mc_neck_mass___enlarged_lymph_node",
        "participant_study_id"
        
    ]
    df.drop(columns=l, axis=1, inplace=True)

    # Convert recording_id from 32 char hex to standard UUID format
    df["recording_id"] = df["recording_id"].apply(
        lambda x: str(uuid.UUID(x)) if pd.notna(x) else x
    )

    df.to_csv(f"{output_csv}/recording-may13.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("recording_csv", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    
    create_recording_redcap(args.recording_csv, args.output_path)
    