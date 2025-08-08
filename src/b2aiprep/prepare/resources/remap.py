import pandas as pd
import json
import os
import sys
def update_participant_ids(tsv_dir: str, mapping_file: str) -> None:
    """
    Updates the 'participant_id' column in all .tsv files in a directory,
    replacing original IDs with new ones from a mapping file.

    Parameters:
        tsv_dir (str): Path to the directory containing .tsv files (recursively).
        mapping_file (str): Path to the JSON file containing the ID mapping (original_id -> new_id).
    """
    with open(f"{mapping_file}", "r") as f:
        id_map = json.load(f)

    for root, dirs, files in os.walk(tsv_dir):
        for filename in files:
            if filename.endswith(".tsv"):
                file_path = os.path.join(root, filename)
                df = pd.read_csv(file_path, sep="\t")
                
                if "participant_id" in df.columns:
                    df["participant_id"] = df["participant_id"].map(id_map).fillna(df["participant_id"])
                    df.to_csv(file_path, sep="\t", index=False)
                    print(f"Updated {file_path}")
                else:
                    print(f"Skipped {file_path}: 'participant_id' column not found.")

if __name__ == "__main__":
    update_participant_ids(sys.argv[1], sys.argv[2])