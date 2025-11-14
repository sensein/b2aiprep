import pandas as pd
import argparse
import json
import re


def to_camel_case(text):
    if not isinstance(text, str):
        return text
    text = text.strip()
    parts = re.split(r"[\s_\-]+", text)
    if not parts:
        return ""
    return parts[0].lower() + "".join(word.capitalize() for word in parts[1:])


def convert_date_columns(df):
    for col in df.columns:
        if col.lower().endswith("_date"):
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                original_col = df[col].copy()
                try:
                    df.loc[:, col] = pd.to_datetime(df[col], errors="coerce")
                except Exception as e:
                    # Restore original column if conversion fails
                    df.loc[:, col] = original_col
    return df


def transform_cell(value, transform_path):

    if (isinstance(value, float) and value.is_integer()) or (
        isinstance(value, str) and value.isdigit()
    ):
        return str(int(value))

    if not isinstance(value, str):
        return value

    with open(transform_path, "r") as f:
        config = json.load(f)

    if value in config:
        return config[value]

    output = (
        lambda v: (
            "1" if v in ["checked", "v1.0.0"] else "0" if v == "unchecked" else to_camel_case(value)
        )
    )(value.lower())

    if output in ["completed", "consented", "canada"]:
        return "2"

    if output == "none,NotAProblem":
        return "none"

    return output


def modify_neckmass(csv_path):
    df = pd.read_csv(csv_path)

    df.rename(
        columns={
            "thyroglossal_duct_cyst": "peds_mc_neck_mass___thyroglossal_duct_cyst",
            "branchial_cleft_cyst": "peds_mc_neck_mass___branchial_cleft_cyst",
            "dermoid_cyst": "peds_mc_neck_mass___dermoid_cyst",
            "enlarged_lymph_node": "peds_mc_neck_mass___enlarged_lymph_node",
        },
        inplace=True,
    )

    for col in [
        "peds_mc_neck_mass___thyroglossal_duct_cyst",
        "peds_mc_neck_mass___branchial_cleft_cyst",
        "peds_mc_neck_mass___dermoid_cyst",
        "peds_mc_neck_mass___enlarged_lymph_node",
    ]:
        df[col] = df[col].astype("string").fillna("").str.strip().str.lower()
        df[col] = df[col].apply(lambda x: "1" if x == "Yes" or x == "yes" else "0")

    df.to_csv(csv_path, index=False)


def remove_fields(csv_path, fields_to_remove):
    df = pd.read_csv(csv_path)
    df = df.dropna(axis=1, how="all")

    # Load list of columns to remove from JSON
    with open(fields_to_remove, "r") as f:
        columns_to_remove = json.load(f)

    df.drop(columns=[col for col in columns_to_remove if col in df.columns], inplace=True)
    df.to_csv(csv_path, index=False)


def remap_columns(csv_path, column_remap):
    df = pd.read_csv(csv_path)

    with open(column_remap, "r") as f:
        column_mapping = json.load(f)

    df.rename(columns=column_mapping, inplace=True)
    df.to_csv(csv_path, index=False)


def combine_instruments(csv_path, insrument_mapping_path):
    df = pd.read_csv(csv_path)
    df["_original_row"] = df.index  # Add this to preserve order

    with open(insrument_mapping_path, "r") as f:
        instrument_mapping = json.load(f)

    # Separate Participant and Contact rows
    participant_df = df[df["redcap_repeat_instrument"] == "Participant"].copy()
    contact_df = df[
        df["redcap_repeat_instrument"] == "subjectparticipant_contact_information_schema"
    ].copy()

    # Drop redcap_repeat_instrument for merge
    participant_df_nodup = participant_df.drop(columns=["redcap_repeat_instrument"])
    contact_df_nodup = contact_df.drop(columns=["redcap_repeat_instrument"])

    # Rename contact columns to avoid conflicts
    contact_df_nodup = contact_df_nodup.add_suffix("_contact")
    contact_df_nodup = contact_df_nodup.rename(columns={"record_id_contact": "record_id"})

    # Merge by record_id
    merged_df = pd.merge(participant_df_nodup, contact_df_nodup, on="record_id", how="inner")

    # Build merged rows manually
    merged_rows = []
    for _, row in merged_df.iterrows():
        merged_row = {"record_id": row["record_id"], "redcap_repeat_instrument": ""}

        for col in participant_df_nodup.columns:
            if col != "record_id":
                merged_row[col] = row[col]

        for col in contact_df_nodup.columns:
            if col == "record_id":
                continue
            original_col = col.replace("_contact", "")
            if original_col not in merged_row:
                merged_row[original_col] = row[col]
            elif pd.isna(merged_row[original_col]):
                merged_row[original_col] = row[col]

        orig_row_pos = df[
            (df["record_id"] == row["record_id"])
            & (
                df["redcap_repeat_instrument"].isin(
                    ["Participant", "subjectparticipant_contact_information_schema"]
                )
            )
        ]["_original_row"].min()
        merged_row["_original_row"] = orig_row_pos
        merged_rows.append(merged_row)

    merged_clean_df = pd.DataFrame(merged_rows)

    rows_to_remove = df["redcap_repeat_instrument"].isin(
        ["Participant", "subjectparticipant_contact_information_schema"]
    )
    remaining_rows = df[~rows_to_remove]

    final_df = pd.concat([merged_clean_df, remaining_rows], ignore_index=True)

    # Apply instrument mapping
    final_df["redcap_repeat_instrument"] = final_df["redcap_repeat_instrument"].replace(
        instrument_mapping
    )

    # Remove unwanted instrument
    final_df = final_df[final_df["redcap_repeat_instrument"] != "peds_medical_history_schema"]

    final_df = final_df[final_df["redcap_repeat_instrument"] != "conclusion_schema"]

    final_df = (
        final_df.sort_values("_original_row").drop(columns=["_original_row"]).reset_index(drop=True)
    )

    final_df.to_csv(csv_path, index=False)


def clean_up(df):
    df.loc[df["redcap_repeat_instrument"].fillna("") == "", "redcap_repeat_instance"] = ""
    df.loc[df["peds_edu_level"] == "preferNotToAnswer", "peds_edu_level"] = "12"
    df.loc[df["peds_edu_level"] == "other", "peds_edu_level"] = "11"
    csv_string = df.to_csv(index=False)
    csv_string = csv_string.replace("preferNotToAnswer", "noAnswer")
    csv_string = csv_string.replace("problemIs“asBadAsItCanBe”", "asBadAsItCanBe")
    csv_string = csv_string.replace(".0", "")

    return csv_string


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List .wav files and save to CSV.")
    parser.add_argument("csv_path", help="Path to the directory to csv files")
    parser.add_argument("remove_fields_path", help="Path to json to remove un-needed fields")
    parser.add_argument("column_remap_path", help="Path to json remap column fields")
    parser.add_argument("instrument_mapping_path", help="insrument mapping path")
    parser.add_argument("transformer_path", help="instrument mapping path")
    parser.add_argument("exclude_columns", help="instrument mapping path")

    args = parser.parse_args()
    modify_neckmass(args.csv_path)
    remove_fields(args.csv_path, args.remove_fields_path)
    remap_columns(args.csv_path, args.column_remap_path)
    combine_instruments(args.csv_path, args.instrument_mapping_path)

    with open(args.exclude_columns, "r") as f:
        exclude_columns = json.load(f)
        exclude_columns.append("redcap_repeat_instrument")
    df = pd.read_csv(args.csv_path, dtype=str)
    df = convert_date_columns(df)
    for col in df.columns:
        if col not in exclude_columns:
            df[col] = df[col].apply(lambda x: transform_cell(x, args.transformer_path))

    csv_output = clean_up(df)
    with open(args.csv_path, "w", newline="") as f:
        f.write(csv_output)
