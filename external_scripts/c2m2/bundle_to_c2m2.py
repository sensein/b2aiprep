

from c2m2_mappings import * 
from constants import *

from pathlib import Path
import pandas as pd
import argparse
import os
import hashlib


def calculate_sha256_file_digest(file_path):
    """
    Calculates the SHA256 hash of a file using hashlib.file_digest().

    Args:
        file_path (str or Path): The path to the file.

    Returns:
        str: The hexadecimal SHA256 hash.
    """
    # Open the file in binary mode ('rb')
    with open(file_path, 'rb') as f:
        # Calculate the digest
        hash_object = hashlib.file_digest(f, 'sha256')

    return hash_object.hexdigest()


def get_participant_races(participant_demo_df):
    race_cols = [f'race___{i}' for i in range(1, 9)]
    pid = participant_demo_df['participant_id'].iloc[0]
    
    # Get union of all race selections across rows
    races_selected = [
        col for col in race_cols
        if (participant_demo_df[col].notna()).any()
    ]
    
    if 'race___8' in races_selected:
        print(f"Participant {pid} flagged — 'Prefer not to answer' selected.")
        print(f"  All races ever checked: {races_selected}")
        return []
    
    return races_selected


def get_all_files(current_dir):
    current_dir_objects = os.listdir(current_dir)
    all_files = []
    for current_dir_object in current_dir_objects:
        current_dir_object_path = current_dir / current_dir_object
        if os.path.isdir(current_dir_path):
            all_files += get_all_files(current_dir_object_path)
        else:
            all_files.append(current_dir_object_path)
    return all_files


def load_or_init_map(c2m2_id_map: Path) -> pd.DataFrame:
    if c2m2_id_map.exists():
        return pd.read_csv(c2m2_id_map)
    return pd.DataFrame(columns=['participant_id', 'session_id', 'anon_id'])

def save_map(id_map: pd.DataFrame, c2m2_id_map: Path):
    id_map.to_csv(c2m2_id_map, index=False)


def get_biosamples_from_features(bundle_path, c2m2_id_map):
    features_path = bundle_path / "features"

    tsv_features = features_path.rglob('*.tsv')
    parquet_features = features_path.rglob('*.parquet')
    id_map = load_or_init_map(c2m2_id_map)

    for parquet in parquet_features:
        df = pd.read_parquet(parquet, columns=['participant_id', 'session_id', 'task_name'])
        # Find combos not yet in the map
        new_combos = (
            df[['participant_id', 'session_id']]
            .drop_duplicates()
            .merge(id_map[['participant_id', 'session_id']], on=['participant_id', 'session_id'], how='left', indicator=True)
            .query('_merge == "left_only"')
            .drop(columns='_merge')
        )
        # Assign new IDs continuing from where we left off
        if not new_combos.empty:
            next_id = len(id_map) + 1
            new_combos['anon_id'] = [f"{i:06d}" for i in range(next_id, next_id + len(new_combos))]
            id_map = pd.concat([id_map, new_combos], ignore_index=True)
        
        # Merge anon_id onto df and create biosample name
        df = df.merge(id_map[['participant_id', 'session_id', 'anon_id']], on=['participant_id', 'session_id'])
        df['biosample_name'] = df['anon_id'] + '_' + df['task_name']

    save_map(id_map, c2m2_id_map)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle_path', help='Path to the bundle dataset')
    parser.add_argument('c2m2_path', help='Path to the C2M2 output')
    parser.add_argument('--c2m2_id_remap', type=str, help='File for remapping IDs in c2m2')
    parser.add_argument('--physionet_version', type=str,default=None, help='Physionet Version')
    args = parser.parse_args()

    bundle_path = Path(args.bundle_path)
    c2m2_path = Path(args.c2m2_path)

    all_files = get_all_files(bundle_path)

    file_metadata = []
    for f in all_files:
        f_size = f.stat().st_size
        if args.physionet_version:
            rel_path = f.relative_to(bundle_path)
            access_url = f"https://physionet.org/files/b2ai-voice/{args.physionet_version}/{str(rel_path)}?download"
        else:
            access_url = None
        f_sha256 = calculate_sha256_file_digest(f)

        file_metadata.append({
            'id_namespace': ID_NAMESPACE,
            'local_id': f.name,
            'project_id_namespace': PROJECT_ID_NAMESPACE,
            'project_local_id': PROJECT_LOCAL_ID,
            'persistent_id': None,
            'creation_time': None,
            'size_in_bites': f_size,
            'uncompressed_size_in_bytes': None,
            'sha256': f_sha256,
            'md5': None,
            'filename': f.name,
            'file_format': file_format_map[f.suffix],
            'compression_format': None,
            'data_type': data_type_map[f.suffix],
            'assay_type': None,
            'analysis_type': None,
            'mime_type': mime_type_map[f.suffix],
            'bundle_collection_id_namespace': None,
            'bundle_collection_local_id': None,
            'dbgap_study_id': None,
            'access_url': access_url
        })
    files_df = pd.DataFrame(file_metadata)
    files_df.to_csv(c2m2_path / "file.tsv", sep='\t',index=False)

    biosamples = get_biosamples_from_features(bundle_path)


def remove():
  bundle_path = Path(args.bundle_path)
  c2m2_path = Path(args.c2m2_path)

  participant_path = bundle_path / "phenotype" / "enrollment" / "participant.tsv"
  participant_df = pd.read_csv(participant_path, sep='\t', dtype=str)
  participants = list(participant_df['participant_id'])
  
  demographics_path = bundle_path / "phenotype" / "demographics" / "demographics.tsv"
  demographics = pd.read_csv(demographics_path, sep='\t', dtype=str)

  subject_races = []
  subjects = []

  for participant in participants:
    participant_demo = demographics[demographics['participant_id']==participant]
    sex_responses = list(participant_demo['sex_at_birth'].dropna().unique())
    if len(sex_responses) == 0:
      print(f'{participant}: Sex is missing but is optional')
      sex = None #sex_map['']
    elif len(sex_responses) == 1:
      sex = sex_map[sex_responses[0]]
    else:
      print(f'{participant}: More than one unique response for sex at birth, default to latest')
      sex = sex_map[sex_responses[-1]]

    ethnicity_responses = list(participant_demo['ethnicity'].dropna().unique())
    if len(ethnicity_responses) == 0:
      print(f"{participant}: Missing ethnicity but is optional")
      ethnicity = None #ethnicity_map['Not Hispanic or Latino']
    elif len(ethnicity_responses) == 1:
      ethnicity = ethnicity_map[ethnicity_responses[0]]
    else:
      print(f'{participant}: More than one unique response for ethnicity, default to latest')
      ethnicity = ethnicity_map[ethnicity_responses[-1]]

    age_responses = list(participant_demo['age'].dropna().unique())
    if len(age_responses) == 0:
      print(f"{participant}: Missing age")
      age = None
    elif len(age_responses) == 1:
      if age_responses[0] == '90 and above':
        age = None
      else:
        age = age_responses[0]
    else:
      print(f'{participant}: More than one unique response for age, default to latest')
      age = age_responses[-1]

    subjects.append({
      'id_namespace': ID_NAMESPACE,
      'local_id': participant,
      'project_id_namespace': PROJECT_ID_NAMESPACE,
      'project_local_id': PROJECT_LOCAL_ID,
      'persistent_id': '',
      'creation_time': '',
      'granularity': SUBJECT_GRANULARITY,
      'sex': sex,
      'ethnicity': ethnicity,
      'age_at_enrollment': age
    })
    
    participant_races = get_participant_races(participant_demo)
    for race in participant_races:
      subject_races.append({
        'subject_id_namespace': SUBJECT_ID_NAMESPACE,
        'subject_local_id': participant,
        'race': race_map[race]
      })
      
  subject_df = pd.DataFrame(subjects)
  subject_df.to_csv(c2m2_path / "subject.tsv", sep='\t', index=False)


  race_df = pd.DataFrame(subject_races)
  race_df.to_csv(c2m2_path / "subject_race.tsv", sep='\t',index=False)


if __name__=="__main__":
  main()
