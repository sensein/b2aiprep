from c2m2_mappings import *
from constants import *

from pathlib import Path
import pandas as pd
import argparse


def get_adult_participant_races(participant_demo_df):
    """Return list of race___N column names checked by this adult participant."""
    race_cols = [f'race___{i}' for i in range(1, 9)]
    pid = participant_demo_df['participant_id'].iloc[0]

    races_selected = [
        col for col in race_cols
        if (participant_demo_df[col].notna()).any()
    ]

    if 'race___8' in races_selected:
        print(f"Participant {pid} flagged — 'Prefer not to answer' selected.")
        print(f"  All races ever checked: {races_selected}")
        return []

    return races_selected


def get_peds_participant_races(participant_demo_df):
    """Return list of individual race strings for this peds participant."""
    races_selected = participant_demo_df['peds_race'].dropna().unique().tolist()

    races = set()
    for race_answer in races_selected:
        for race in race_answer.split(', '):
            races.add(race)
    races = list(races)

    if 'Prefer not to answer' in races:
        pid = participant_demo_df['participant_id'].unique().item()
        print(f"Participant {pid} flagged — 'Prefer not to answer' selected.")
        print(f"  All races ever checked: {races}")
        return []

    return races


def main():
    parser = argparse.ArgumentParser(
        description='Fill subject.tsv and subject_race.tsv from a B2AI voice bundle.'
    )
    parser.add_argument('bundle_path', help='Path to the bundle dataset')
    parser.add_argument('c2m2_path', help='Path to the C2M2 output directory')
    parser.add_argument('--peds', action='store_true', help='Use pediatric logic')
    args = parser.parse_args()

    bundle_path = Path(args.bundle_path)
    c2m2_path = Path(args.c2m2_path)

    if args.peds:
        demographics_path = bundle_path / "phenotype" / "pediatric" / "pediatric_demographics.tsv"
        demographics = pd.read_csv(demographics_path, sep='\t', dtype=str)
        participants = list(demographics['participant_id'].unique())
        project_local_id = PEDS_PROJECT_LOCAL_ID
        sex_col = 'peds_gender_identity'
        used_sex_map = peds_sex_map
        get_races = get_peds_participant_races
        used_race_map = peds_race_map
    else:
        participant_path = bundle_path / "phenotype" / "enrollment" / "participant.tsv"
        participant_df = pd.read_csv(participant_path, sep='\t', dtype=str)
        participants = list(participant_df['participant_id'])
        demographics_path = bundle_path / "phenotype" / "demographics" / "demographics.tsv"
        demographics = pd.read_csv(demographics_path, sep='\t', dtype=str)
        project_local_id = PROJECT_LOCAL_ID
        sex_col = 'sex_at_birth'
        used_sex_map = sex_map
        get_races = get_adult_participant_races
        used_race_map = race_map

    subject_races = []
    subjects = []

    for participant in participants:
        participant_demo = demographics[demographics['participant_id'] == participant]

        sex_responses = list(participant_demo[sex_col].dropna().unique())
        if len(sex_responses) == 0:
            print(f'{participant}: Sex is missing but is optional')
            sex = None
        elif len(sex_responses) == 1:
            sex = used_sex_map[sex_responses[0]]
        else:
            print(f'{participant}: More than one unique response for sex, default to latest')
            sex = used_sex_map[sex_responses[-1]]

        if args.peds:
            ethnicity = None
        else:
            ethnicity_responses = list(participant_demo['ethnicity'].dropna().unique())
            if len(ethnicity_responses) == 0:
                print(f"{participant}: Missing ethnicity but is optional")
                ethnicity = None
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
            'project_local_id': project_local_id,
            'persistent_id': '',
            'creation_time': '',
            'granularity': SUBJECT_GRANULARITY,
            'sex': sex,
            'ethnicity': ethnicity,
            'age_at_enrollment': age,
        })

        participant_races = get_races(participant_demo)
        for race in participant_races:
            subject_races.append({
                'subject_id_namespace': SUBJECT_ID_NAMESPACE,
                'subject_local_id': participant,
                'race': used_race_map[race],
            })

    subject_df = pd.read_csv(c2m2_path / "subject.tsv", sep='\t')
    subject_df = pd.concat([subject_df, pd.DataFrame(subjects)], ignore_index=True)
    subject_df.to_csv(c2m2_path / "subject.tsv", sep='\t', index=False)

    race_df = pd.read_csv(c2m2_path / "subject_race.tsv", sep='\t')
    race_df = pd.concat([race_df, pd.DataFrame(subject_races)], ignore_index=True)
    race_df.to_csv(c2m2_path / "subject_race.tsv", sep='\t', index=False)


if __name__ == "__main__":
    main()
