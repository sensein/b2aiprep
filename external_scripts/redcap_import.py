import pandas as pd
import argparse
import json
import re

def to_camel_case(text):
    if not isinstance(text, str):
        return text
    text = text.strip()
    parts = re.split(r'[\s_\-]+', text)
    if not parts:
        return ''
    return parts[0].lower() + ''.join(word.capitalize() for word in parts[1:])
    

def convert_date_columns(df):
    for col in df.columns:
        if col.lower().endswith('_date'):
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                original_col = df[col].copy()
                try:
                    df.loc[:, col] = pd.to_datetime(df[col], errors='coerce')
                except Exception as e:
                    # Restore original column if conversion fails
                    df.loc[:, col] = original_col
    return df

def transform_cell(value, transform_path):
    
    if isinstance(value, float) and value.is_integer():
        return int(value)
    
    if not isinstance(value, str):
        return value
    
    
    with open(transform_path, 'r') as f:
        config = json.load(f)
    
    if value in config:
        return config[value]
    
    lowered = value.lower()
    if lowered == 'checked' or lowered == "v1.0.0":
        output = 1
    elif lowered == 'unchecked':
        output = 0
    else:
        output =  to_camel_case(value)
        
    if output == "completed" or output == "consented" or output == "canada":
        output = 2
        return output

    if output == "none,NotAProblem":
        output = "none"
        return output
    

    
    return output


def modify_neckmass(csv_path):
        # Step 1: Load the CSV
    df = pd.read_csv(csv_path)  # Replace with your actual file path

    # Step 2: Rename columns
    df.rename(columns={
        'thyroglossal_duct_cyst': 'peds_mc_neck_mass___thyroglossal_duct_cyst',
        'branchial_cleft_cyst': 'peds_mc_neck_mass___branchial_cleft_cyst',
        'dermoid_cyst': 'peds_mc_neck_mass___dermoid_cyst',
        'enlarged_lymph_node': 'peds_mc_neck_mass___enlarged_lymph_node'
    }, inplace=True)

    # Step 3: Normalize values in m_a, m_b, m_c
    for col in ['peds_mc_neck_mass___thyroglossal_duct_cyst', 'peds_mc_neck_mass___branchial_cleft_cyst', 'peds_mc_neck_mass___dermoid_cyst', 'peds_mc_neck_mass___enlarged_lymph_node']:
        df[col] = df[col].fillna('').str.strip().str.lower()
        df[col] = df[col].apply(lambda x: '1' if x == 'Yes' or x == "yes" else '0')

    # Step 4: Save to CSV
    df.to_csv(csv_path, index=False)


def remove_fields(csv_path, fields_to_remove):    
    # Load CSV
    df = pd.read_csv(csv_path)
    df = df.dropna(axis=1, how='all')
    # Load list of columns to remove from JSON
    with open(fields_to_remove, 'r') as f:
        columns_to_remove = json.load(f)

    # Drop columns that exist in both the list and the DataFrame
    df.drop(columns=[col for col in columns_to_remove if col in df.columns], inplace=True)

    # Optional: Save to new CSV
    df.to_csv(csv_path, index=False)


def remap_columns(csv_path, column_remap):
    # Load CSV file

    df = pd.read_csv(csv_path)

    # Load JSON dict
    #json_path = '/home/evan/Documents/Peds2redcap-remap/all.json'  # Replace with your actual JSON file path
    with open(column_remap, 'r') as f:
        column_mapping = json.load(f)

    df.rename(columns=column_mapping, inplace=True)

    df.to_csv(csv_path, index=False)


def combine_instruments(csv_path, insrument_mapping_path):
    # Load input CSV and keep original row order using a helper column
    df = pd.read_csv(csv_path)
    df['_original_row'] = df.index  # Add this to preserve order

    # Load instrument mapping
    with open(insrument_mapping_path, "r") as f:
        instrument_mapping = json.load(f)

    # Step 1: Separate Participant and Contact rows
    participant_df = df[df['redcap_repeat_instrument'] == 'Participant'].copy()
    contact_df = df[df['redcap_repeat_instrument'] == 'subjectparticipant_contact_information_schema'].copy()

    # Drop redcap_repeat_instrument for merge
    participant_df_nodup = participant_df.drop(columns=['redcap_repeat_instrument'])
    contact_df_nodup = contact_df.drop(columns=['redcap_repeat_instrument'])

    # Rename contact columns to avoid conflicts
    contact_df_nodup = contact_df_nodup.add_suffix('_contact')
    contact_df_nodup = contact_df_nodup.rename(columns={'record_id_contact': 'record_id'})

    # Merge by record_id
    merged_df = pd.merge(participant_df_nodup, contact_df_nodup, on='record_id', how='inner')

    # Build merged rows manually
    merged_rows = []
    for _, row in merged_df.iterrows():
        merged_row = {'record_id': row['record_id'], 'redcap_repeat_instrument': ''}
        
        # From Participant
        for col in participant_df_nodup.columns:
            if col != 'record_id':
                merged_row[col] = row[col]
        
        # From Contact (only if not duplicate)
        for col in contact_df_nodup.columns:
            if col == 'record_id':
                continue
            original_col = col.replace('_contact', '')
            if original_col not in merged_row:
                merged_row[original_col] = row[col]
            elif pd.isna(merged_row[original_col]):
                merged_row[original_col] = row[col]
        
        # Use the lower of the two original rows’ position for sorting
        orig_row_pos = df[(df['record_id'] == row['record_id']) &
                        (df['redcap_repeat_instrument'].isin(['Participant', 'subjectparticipant_contact_information_schema']))]['_original_row'].min()
        merged_row['_original_row'] = orig_row_pos
        merged_rows.append(merged_row)

    merged_clean_df = pd.DataFrame(merged_rows)

    # Step 2: Remove original Participant + Contact rows from original df
    rows_to_remove = df['redcap_repeat_instrument'].isin(['Participant', 'subjectparticipant_contact_information_schema'])
    remaining_rows = df[~rows_to_remove]

    # Step 3: Combine merged + untouched rows
    final_df = pd.concat([merged_clean_df, remaining_rows], ignore_index=True)

    # Step 4: Apply instrument mapping
    final_df['redcap_repeat_instrument'] = final_df['redcap_repeat_instrument'].replace(instrument_mapping)

    # Step 5: Remove unwanted instrument
    final_df = final_df[final_df['redcap_repeat_instrument'] != 'peds_medical_history_schema']

    final_df  = final_df [final_df ['redcap_repeat_instrument'] != 'conclusion_schema']

    # Step 6: Sort by original row index to maintain input order
    final_df = final_df.sort_values('_original_row').drop(columns=['_original_row']).reset_index(drop=True)

    # Step 7: Save result
    final_df.to_csv(csv_path, index=False)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="List .wav files and save to CSV.")
    parser.add_argument("csv_path", help="Path to the directory to csv files")
    parser.add_argument("remove_fields_path", help="Path to json to remove un-needed fields")
    parser.add_argument("column_remap_path", help="Path to json remap column fields")
    parser.add_argument("instrument_mapping_path", help="insrument mapping path")
    parser.add_argument("transformer_path", help="instrument mapping path")
    parser.add_argument("exclude_columns", help="instrument mapping path")
    
    
    args = parser.parse_args()
    # /home/evan/Documents/redcap_trial_run/redcapcsv.csv
    modify_neckmass(args.csv_path)
    remove_fields(args.csv_path, args.remove_fields_path)
    remap_columns(args.csv_path, args.column_remap_path)
    combine_instruments(args.csv_path, args.instrument_mapping_path)
    
    with open(args.exclude_columns, 'r') as f:
        exclude_columns = json.load(f)
        exclude_columns.append('redcap_repeat_instrument')
    df = pd.read_csv(args.csv_path, dtype=str)
    df = convert_date_columns(df)
    for col in df.columns:
        if col not in exclude_columns:
            # Using lambda to pass the extra argument
            df[col] = df[col].apply(lambda x: transform_cell(x, args.transformer_path))

            
    df.to_csv(args.csv_path, index=False)

    