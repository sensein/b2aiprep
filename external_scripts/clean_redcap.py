import argparse
import os
import re
import pandas as pd
from io import StringIO


def find_replace_in_csv(filepath):
    """Read a CSV file as text, perform find-and-replace, and save to a new file."""
    with open(filepath, "r", encoding="utf-8") as f:
        csv_text = f.read()

    # inconsistencies with certain record_ids and fixing mistake put having inputted 021sm and 022ls as the
    # record ids for 022sm and 024ls respectively.
    modifications = {
        " js,": "js,",
        " ls,": "ls,",
        " sm,": "sm,",
        "-sm,": "sm,",
        "-ls,": "ls,",
        "-js": "js,",
        " sm ,": "sm,",
        "019js ": "019js",
    }

    for old, new in modifications.items():
        csv_text = csv_text.replace(old, new)

    return csv_text


def fix_specific_record_ids(csv_str, new_path):
    """
    Prepend a '0' to specific record_ids if they match a given list.
    """
    column = "record_id"
    csv_io = StringIO(csv_str)

    df = pd.read_csv(csv_io, dtype=str)

    if column not in df.columns:
        raise ValueError(f"The CSV must contain a '{column}' column.")

    # remove specific entry as it was a user input error
    record_to_remove = "066 sm medical condition part 2"
    df = df[df["record_id"] != record_to_remove]

    # these ids are missing the leading 0 causing the survey and recordings to not match
    ids_to_fix = ["12sm", "15sm", "16sm", "17sm", "19sm", "23sm"]

    df.loc[df[column].isin(ids_to_fix), column] = "0" + df.loc[
        df[column].isin(ids_to_fix), column
    ].astype(str)

    df.to_csv(f"{new_path}/redcap_csv.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and replace text in a CSV file.")
    parser.add_argument("filepath", help="Path to the CSV file")
    parser.add_argument("outputpath", help="Path of output")
    args = parser.parse_args()

    modifed_csv_str = find_replace_in_csv(args.filepath)

    fix_specific_record_ids(modifed_csv_str, args.outputpath)
