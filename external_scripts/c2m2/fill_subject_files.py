

from c2m2_mappings import * 
from constants import *

from pathlib import Path
import pandas as pd
import argparse

SUBJECT_FILES = ['subject.tsv','subject_race.tsv']


def get_participant_races(participant_demo_df):
    race_cols = [f'race___{i}' for i in range(1, 9)]
    pid = participant_demo_df['participant_id'].iloc[0]
    
    # Get union of all race selections across rows
    races_selected = [
        col for col in race_cols
        if (participant_demo_df[col] == 1.0).any()
    ]
    
    if 'race___8' in races_selected:
        print(f"Participant {pid} flagged — 'Prefer not to answer' selected.")
        print(f"  All races ever checked: {races_selected}")
        return []
    
    return races_selected

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('bundle_path', help='Path to the bundle dataset')
  parser.add_argument('c2m2_path', help='Path to the C2M2 output')
  args = parser.parse_args()

  bundle_path = args.bundle_path
  c2m2_path = args.c2m2_path

  participant_path = bundle_path / "phenotype" / "enrollment" / "participant.tsv"
  participant_df = pd.read_csv(participant_path, sep='\t', dtype=str)
  participants = list(participant_df['participant_id'])
  
  demographics_path = bundle_path / "phenotype" / "demographics" / "demographics.tsv"
  demographics_df = pd.read_csv(demographics_path, sep='\t', dtype=str)

  subject_races = []
  subjects = []

  for participant in participants:
    participant_demo = demographics[demographics['participant_id']==participant_id]
    sex_responses = list(participant_demo['sex_at_birth'].dropna().unique())
    if len(sex_responses) == 0:
      print('Sex is missing but is optional')
      sex = None #sex_map['']
    elif len(sex_responses) == 1:
      sex = sex_map[sex_responses[0]]
    else:
      print('More than one unique response for sex at birth, default to latest')
      sex = sex_map[sex_responses[-1]]

    ethnicity_responses = list(participant_demo['ethnicity'].dropna().unique())
    if len(ethnicity_responses) == 0:
      print("Missing ethnicity but is optional")
      ethnicity = None #ethnicity_map['Not Hispanic or Latino']
    elif len(ethnicity_responses) == 1:
      ethnicity = ethnicity_map[ethnicity_responses[0]]
    else:
      print('More than one unique response for ethnicity, default to latest')
      ethnicity = ethnicity_map[ethnicity_responses[-1]]

    age_responses = list(participant_demo['age'].dropna().unique())
    if len(age_responses) == 0:
      print("Missing age")
      age = None
    elif len(age_responses) == 1:
      if age_responses[0] == '90 and above':
        age = None
      else:
        age = age_responses[0]
    else:
      print('More than one unique response for age, default to latest')
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
