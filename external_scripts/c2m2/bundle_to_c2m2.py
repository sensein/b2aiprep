from c2m2_mappings import *
from constants import *

from pathlib import Path
import pandas as pd
import argparse
import os
import re
import hashlib


def lookup_task_obi(task_name: str):
    """
    Look up the OBI term for a task name.

    Trailing -<digits> groups are always stripped before the lookup, so
    'maximum-phonation-time-3' and 'harvard-sentences-5-1' resolve to their
    stems. The lookup is then an exact key match against task_to_OBI —
    no substring or prefix matching — so 'caterpillar-passage' will never
    accidentally match an entry keyed on 'passage'.
    """
    stem = re.sub(r'(-\d+)+$', '', task_name)
    return task_to_OBI.get(stem)


def calculate_sha256_file_digest(file_path):
    """Calculates the SHA256 hash of a file."""
    with open(file_path, 'rb') as f:
        hash_object = hashlib.file_digest(f, 'sha256')
    return hash_object.hexdigest()


def get_all_files(current_dir):
    """Recursively collects all file paths under current_dir."""
    all_files = []
    for current_dir_object in os.listdir(current_dir):
        current_dir_object_path = current_dir / current_dir_object
        if os.path.isdir(current_dir_object_path):
            all_files += get_all_files(current_dir_object_path)
        else:
            all_files.append(current_dir_object_path)
    return all_files


def load_or_init_map(c2m2_id_map: Path) -> pd.DataFrame:
    if c2m2_id_map.exists():
        return pd.read_csv(c2m2_id_map, dtype=str)
    return pd.DataFrame(columns=['participant_id', 'session_id', 'anon_id'])


def save_map(id_map: pd.DataFrame, c2m2_id_map: Path):
    id_map.to_csv(c2m2_id_map, index=False)


def get_biosamples_from_features(bundle_path: Path, c2m2_id_map: Path, project_local_id: str):
    """
    Reads feature parquet files to build the anonymized biosample list.

    Returns:
        tuple: (biosample_df, id_map) where:
            - biosample_df: one row per unique (anon_id, task_name), with C2M2 biosample fields
            - id_map: the full participant_id/session_id -> anon_id mapping DataFrame
    """
    features_path = bundle_path / "features"
    parquet_features = list(features_path.rglob('*.parquet'))

    if not parquet_features:
        print("Warning: No parquet files found in features directory")
        return pd.DataFrame(), pd.DataFrame(columns=['participant_id', 'session_id', 'anon_id'])

    id_map = load_or_init_map(c2m2_id_map)

    # Use the first parquet to get participant_id/session_id/task_name
    df = pd.read_parquet(parquet_features[0], columns=['participant_id', 'session_id', 'task_name'])
    df = df[['participant_id', 'session_id', 'task_name']].drop_duplicates()

    # Find (participant_id, session_id) pairs not yet assigned an anon_id
    unique_pairs = df[['participant_id', 'session_id']].drop_duplicates()
    new_combos = (
        unique_pairs
        .merge(id_map[['participant_id', 'session_id']], on=['participant_id', 'session_id'],
               how='left', indicator=True)
        .query('_merge == "left_only"')
        .drop(columns='_merge')
    )

    if not new_combos.empty:
        next_id = len(id_map) + 1
        new_combos['anon_id'] = [f"{i:06d}" for i in range(next_id, next_id + len(new_combos))]
        id_map = pd.concat([id_map, new_combos], ignore_index=True)

    save_map(id_map, c2m2_id_map)

    # Map task_name -> OBI term; warn and skip unmapped tasks
    df = df.merge(id_map[['participant_id', 'session_id', 'anon_id']],
                  on=['participant_id', 'session_id'])
    df['OBI_term'] = df['task_name'].apply(lookup_task_obi)
    unmapped = df[df['OBI_term'].isna()]['task_name'].unique()
    if len(unmapped) > 0:
        print(f"Warning: {len(unmapped)} task name(s) not in task_to_OBI, skipping: {sorted(unmapped)}")
    df = df.dropna(subset=['OBI_term'])

    # One biosample per unique (anon_id, task_name). The full task_name (including
    # any numeric suffixes) is used in local_id so that each recording is its own
    # biosample and all repetitions are accounted for. The OBI term is derived from
    # the stripped stem and used only for sample_prep_method.
    biosample_df = df[['anon_id', 'task_name', 'OBI_term']].drop_duplicates().copy()
    biosample_df['local_id'] = biosample_df['anon_id'] + '_' + biosample_df['task_name']
    biosample_df['id_namespace'] = ID_NAMESPACE
    biosample_df['project_id_namespace'] = PROJECT_ID_NAMESPACE
    biosample_df['project_local_id'] = project_local_id
    biosample_df['persistent_id'] = None
    biosample_df['creation_time'] = None
    biosample_df['sample_prep_method'] = biosample_df['OBI_term']
    biosample_df['anatomy'] = 'UBERON:0001737'
    biosample_df['biofluid'] = None

    return biosample_df, id_map


def _add_copd_asthma_rows(diag_df, id_map, biosample_df, rows):
    """Handle COPD/asthma: diagnosis_ca_copd_asthma encodes both conditions."""
    for _, row in diag_df.iterrows():
        pid = str(row['participant_id'])
        value = str(row.get('diagnosis_ca_copd_asthma', '')).strip().lower()

        doids = []
        if 'asthma' in value:
            doids.append('DOID:2841')
        if 'copd' in value:
            doids.append('DOID:3083')

        if not doids:
            continue

        participant_anon_ids = id_map[id_map['participant_id'] == pid]['anon_id']
        participant_biosamples = biosample_df[biosample_df['anon_id'].isin(participant_anon_ids)]
        for _, bs in participant_biosamples.iterrows():
            for doid in doids:
                rows.append({
                    'biosample_id_namespace': ID_NAMESPACE,
                    'biosample_local_id': bs['local_id'],
                    'association_type': 'cfde_disease_association_type:1',
                    'disease': doid,
                })


def generate_biosample_diseases_adult(bundle_path: Path, id_map: pd.DataFrame,
                                      biosample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reads phenotype/diagnosis/*.tsv files and creates biosample_disease rows.
    Each biosample for a participant with a positive diagnosis is linked to the
    corresponding DOID term(s) via cfde_disease_association_type:1.
    """
    diagnosis_path = bundle_path / "phenotype" / "diagnosis"
    if not diagnosis_path.exists():
        print("Warning: No phenotype/diagnosis directory found")
        return pd.DataFrame(columns=['biosample_id_namespace', 'biosample_local_id',
                                     'association_type', 'disease'])

    rows = []
    for diag_file in sorted(diagnosis_path.glob('*.tsv')):
        stem = diag_file.stem
        if stem == 'control':
            continue

        if stem not in adult_diagnosis_to_DOID:
            print(f"Warning: {stem} not in adult_diagnosis_to_DOID mapping, skipping")
            continue

        diag_df = pd.read_csv(diag_file, sep='\t', dtype=str)

        # Special case: copd_and_asthma uses a combined column instead of gold_standard_diagnosis
        if stem == 'copd_and_asthma':
            _add_copd_asthma_rows(diag_df, id_map, biosample_df, rows)
            continue

        doid_terms = [d for d in adult_diagnosis_to_DOID[stem] if d]
        if not doid_terms:
            print(f"Info: No DOID terms configured for {stem}, skipping")
            continue

        # General case: find the *_gold_standard_diagnosis column
        gsd_cols = [c for c in diag_df.columns if 'gold_standard_diagnosis' in c]
        if not gsd_cols:
            print(f"Warning: No gold_standard_diagnosis column in {stem}.tsv, skipping")
            continue

        gsd_col = gsd_cols[0]
        positive_participants = diag_df[
            diag_df[gsd_col].str.lower() == 'yes'
        ]['participant_id'].unique()

        for pid in positive_participants:
            pid = str(pid)
            participant_anon_ids = id_map[id_map['participant_id'] == pid]['anon_id']
            participant_biosamples = biosample_df[
                biosample_df['anon_id'].isin(participant_anon_ids)
            ]
            for _, bs in participant_biosamples.iterrows():
                for doid in doid_terms:
                    rows.append({
                        'biosample_id_namespace': ID_NAMESPACE,
                        'biosample_local_id': bs['local_id'],
                        'association_type': 'cfde_disease_association_type:1',
                        'disease': doid,
                    })

    return pd.DataFrame(rows, columns=['biosample_id_namespace', 'biosample_local_id',
                                       'association_type', 'disease'])


def generate_biosample_diseases_peds(bundle_path: Path, id_map: pd.DataFrame,
                                     biosample_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reads phenotype/pediatric/pediatric_medical_conditions.tsv and creates
    biosample_disease rows for conditions defined in peds_condition_to_DOID.
    """
    mc_path = bundle_path / "phenotype" / "pediatric" / "pediatric_medical_conditions.tsv"
    if not mc_path.exists():
        print("Warning: pediatric_medical_conditions.tsv not found")
        return pd.DataFrame(columns=['biosample_id_namespace', 'biosample_local_id',
                                     'association_type', 'disease'])

    mc_df = pd.read_csv(mc_path, sep='\t', dtype=str)
    rows = []

    for condition_col, doid_terms in peds_condition_to_DOID.items():
        if condition_col not in mc_df.columns:
            print(f"Warning: column {condition_col} not in pediatric_medical_conditions.tsv, skipping")
            continue
        doid_terms = [d for d in doid_terms if d]
        if not doid_terms:
            continue

        positive_mask = mc_df[condition_col].str.lower().isin(['yes', '1', 'true'])
        positive_participants = mc_df[positive_mask]['participant_id'].unique()

        for pid in positive_participants:
            pid = str(pid)
            participant_anon_ids = id_map[id_map['participant_id'] == pid]['anon_id']
            participant_biosamples = biosample_df[
                biosample_df['anon_id'].isin(participant_anon_ids)
            ]
            for _, bs in participant_biosamples.iterrows():
                for doid in doid_terms:
                    rows.append({
                        'biosample_id_namespace': ID_NAMESPACE,
                        'biosample_local_id': bs['local_id'],
                        'association_type': 'cfde_disease_association_type:1',
                        'disease': doid,
                    })

    return pd.DataFrame(rows, columns=['biosample_id_namespace', 'biosample_local_id',
                                       'association_type', 'disease'])


def _recording_name_to_task_name(name: str) -> str:
    """Transform recording.tsv recording_name to a features-style task_name slug.

    Steps: lowercase, spaces → hyphens, remove parentheses around version strings
    e.g. 'Cape V sentences (v2)-1' → 'cape-v-sentences-v2-1'
         'Diadochokinesis-KA'       → 'diadochokinesis-ka'
    """
    s = name.lower().replace(' ', '-')
    s = re.sub(r'-\(([^)]+)\)', r'-\1', s)
    return s


def generate_file_describes_biosample(bundle_path: Path, all_files_df: pd.DataFrame,
                                       biosample_df: pd.DataFrame,
                                       id_map: pd.DataFrame) -> pd.DataFrame:
    """
    Generate file_describes_biosample rows.

    Only task-related files are linked to biosamples:
      - features/*.parquet and features/*.tsv: have task_name column
      - metadata/metadata.parquet: has task_name column
      - phenotype/task/recording.tsv: recording_name -> exact task_name slug
      - phenotype/task/acoustic_task.tsv: acoustic_task_name prefix per (participant, session)
      - phenotype/task/stroop.tsv: exact 'word-color-stroop'
      - phenotype/task/harvard_sentences.tsv: prefix 'harvard-sentences'
      - phenotype/task/random_item_generation.tsv: prefix 'random-item-generation'
    All other files are skipped (no biosample link).
    """
    valid_biosample_ids = set(biosample_df['local_id'])

    # Group biosample local_ids by anon_id for efficient prefix matching
    biosample_by_anon = {}
    for row in biosample_df[['anon_id', 'local_id']].itertuples(index=False):
        biosample_by_anon.setdefault(row.anon_id, []).append(row.local_id)

    id_map_cols = id_map[['participant_id', 'session_id', 'anon_id']]

    rows = set()  # (file_local_id, biosample_local_id)

    def add_exact(anon_id, task_name, fid):
        bid = f"{anon_id}_{task_name}"
        if bid in valid_biosample_ids:
            rows.add((fid, bid))

    def add_prefix(anon_id, prefix, fid):
        full_prefix = f"{anon_id}_{prefix}"
        for bid in biosample_by_anon.get(anon_id, []):
            if bid.startswith(full_prefix):
                rows.add((fid, bid))

    def process_task_name_file(df, fid):
        """Handle any file that has participant_id, session_id, task_name columns."""
        df = df[['participant_id', 'session_id', 'task_name']].drop_duplicates()
        df = df.merge(id_map_cols, on=['participant_id', 'session_id'], how='inner')
        df['bid'] = df['anon_id'] + '_' + df['task_name']
        for bid in df.loc[df['bid'].isin(valid_biosample_ids), 'bid'].unique():
            rows.add((fid, bid))

    for file_local_id in all_files_df['local_id']:
        # local_id has an 'adult/' or 'peds/' prefix; strip it to get the
        # bundle-relative path used for reading the actual file
        rel = Path(*Path(file_local_id).parts[1:])
        parts = rel.parts
        suffix = rel.suffix.lower()
        abs_path = bundle_path / rel

        # --- features/ files (parquet and tsv) ---
        if parts[0] == 'features' and suffix in ('.parquet', '.tsv'):
            try:
                df = (pd.read_parquet(abs_path, columns=['participant_id', 'session_id', 'task_name'])
                      if suffix == '.parquet'
                      else pd.read_csv(abs_path, sep='\t', dtype=str,
                                       usecols=['participant_id', 'session_id', 'task_name']))
                process_task_name_file(df, file_local_id)
            except Exception as e:
                print(f"Warning: skipping {file_local_id}: {e}")
            continue

        # --- metadata/ parquet ---
        if parts[0] == 'metadata' and suffix == '.parquet':
            try:
                df = pd.read_parquet(abs_path,
                                     columns=['participant_id', 'session_id', 'task_name'])
                process_task_name_file(df, file_local_id)
            except Exception as e:
                print(f"Warning: skipping {file_local_id}: {e}")
            continue

        # --- phenotype/task/*.tsv ---
        if suffix == '.tsv' and len(parts) >= 2 and parts[-2] == 'task':
            fname = rel.stem
            try:
                if fname == 'recording':
                    df = pd.read_csv(abs_path, sep='\t', dtype=str,
                                     usecols=['participant_id', 'recording_session_id',
                                              'recording_name'])
                    df = df.merge(id_map_cols,
                                  left_on=['participant_id', 'recording_session_id'],
                                  right_on=['participant_id', 'session_id'], how='inner')
                    df['task_name'] = df['recording_name'].apply(_recording_name_to_task_name)
                    df['bid'] = df['anon_id'] + '_' + df['task_name']
                    for bid in df.loc[df['bid'].isin(valid_biosample_ids), 'bid'].unique():
                        rows.add((file_local_id, bid))

                elif fname == 'acoustic_task':
                    df = pd.read_csv(abs_path, sep='\t', dtype=str,
                                     usecols=['participant_id', 'acoustic_task_session_id',
                                              'acoustic_task_name'])
                    df = df.merge(id_map_cols,
                                  left_on=['participant_id', 'acoustic_task_session_id'],
                                  right_on=['participant_id', 'session_id'], how='inner')
                    for row in df[['anon_id', 'acoustic_task_name']].itertuples(index=False):
                        add_prefix(row.anon_id, row.acoustic_task_name, file_local_id)

                elif fname == 'stroop':
                    df = pd.read_csv(abs_path, sep='\t', dtype=str,
                                     usecols=['participant_id', 'stroop_session_id'])
                    df = df.merge(id_map_cols,
                                  left_on=['participant_id', 'stroop_session_id'],
                                  right_on=['participant_id', 'session_id'], how='inner')
                    for anon_id in df['anon_id'].unique():
                        add_exact(anon_id, 'word-color-stroop', file_local_id)

                elif fname == 'harvard_sentences':
                    df = pd.read_csv(abs_path, sep='\t', dtype=str,
                                     usecols=['participant_id', 'harvard_sentences_session_id'])
                    df = df.merge(id_map_cols,
                                  left_on=['participant_id', 'harvard_sentences_session_id'],
                                  right_on=['participant_id', 'session_id'], how='inner')
                    for anon_id in df['anon_id'].unique():
                        add_prefix(anon_id, 'harvard-sentences', file_local_id)

                elif fname == 'random_item_generation':
                    df = pd.read_csv(abs_path, sep='\t', dtype=str,
                                     usecols=['participant_id', 'random_session_id'])
                    df = df.merge(id_map_cols,
                                  left_on=['participant_id', 'random_session_id'],
                                  right_on=['participant_id', 'session_id'], how='inner')
                    for anon_id in df['anon_id'].unique():
                        add_prefix(anon_id, 'random-item-generation', file_local_id)

                # session.tsv, winograd.tsv, *.json → no biosample link; skip

            except Exception as e:
                print(f"Warning: skipping {file_local_id}: {e}")

    if not rows:
        return pd.DataFrame(columns=['file_id_namespace', 'file_local_id',
                                     'biosample_id_namespace', 'biosample_local_id'])

    result = pd.DataFrame(list(rows), columns=['file_local_id', 'biosample_local_id'])
    result.insert(0, 'file_id_namespace', ID_NAMESPACE)
    result.insert(2, 'biosample_id_namespace', ID_NAMESPACE)
    return result[['file_id_namespace', 'file_local_id',
                   'biosample_id_namespace', 'biosample_local_id']]


def main():
    parser = argparse.ArgumentParser(
        description='Convert a B2AI voice bundle dataset to C2M2 metadata TSV files.'
    )
    parser.add_argument('bundle_path', help='Path to the bundle dataset')
    parser.add_argument('c2m2_path', help='Path to the C2M2 output directory')
    parser.add_argument('--c2m2_id_map', required=True, type=str,
                        help='Path to persistent participant->anon_id CSV map file')
    parser.add_argument('--peds', action='store_true', help='Use pediatric logic')
    parser.add_argument('--physionet_version', type=str, default=None,
                        help='PhysioNet version string for access URLs (e.g. 2.0.1)')
    args = parser.parse_args()

    bundle_path = Path(args.bundle_path)
    c2m2_path = Path(args.c2m2_path)
    c2m2_id_map = Path(args.c2m2_id_map)
    project_local_id = PEDS_PROJECT_LOCAL_ID if args.peds else PROJECT_LOCAL_ID
    local_id_prefix = 'peds' if args.peds else 'adult'

    # --- file.tsv ---
    all_files = get_all_files(bundle_path)
    file_metadata = []
    for f in all_files:
        rel_path = f.relative_to(bundle_path)
        if args.physionet_version:
            access_url = (f"https://physionet.org/files/b2ai-voice/"
                          f"{args.physionet_version}/{str(rel_path)}?download")
        else:
            access_url = None
        suffix = f.suffix.lower()
        file_metadata.append({
            'id_namespace': ID_NAMESPACE,
            'local_id': f"{local_id_prefix}/{str(rel_path)}",
            'project_id_namespace': PROJECT_ID_NAMESPACE,
            'project_local_id': project_local_id,
            'persistent_id': None,
            'creation_time': None,
            'size_in_bytes': f.stat().st_size,
            'uncompressed_size_in_bytes': None,
            'sha256': calculate_sha256_file_digest(f),
            'md5': None,
            'filename': f.name,
            'file_format': file_format_map.get(suffix, ''),
            'compression_format': None,
            'data_type': data_type_map.get(suffix, ''),
            'assay_type': None,
            'analysis_type': None,
            'mime_type': mime_type_map.get(suffix, ''),
            'bundle_collection_id_namespace': None,
            'bundle_collection_local_id': None,
            'dbgap_study_id': None,
            'access_url': access_url,
        })
    new_files_df = pd.DataFrame(file_metadata)
    files_df = pd.concat(
        [pd.read_csv(c2m2_path / "file.tsv", sep='\t'), new_files_df], ignore_index=True
    )
    files_df.to_csv(c2m2_path / "file.tsv", sep='\t', index=False)
    print(f"Appended {len(new_files_df)} rows to file.tsv ({len(files_df)} total)")

    # --- biosample.tsv ---
    biosample_df, id_map = get_biosamples_from_features(bundle_path, c2m2_id_map, project_local_id)
    biosample_cols = ['id_namespace', 'local_id', 'project_id_namespace', 'project_local_id',
                      'persistent_id', 'creation_time', 'sample_prep_method', 'anatomy', 'biofluid']
    new_biosample_df = biosample_df[biosample_cols]
    merged_biosample_df = pd.concat(
        [pd.read_csv(c2m2_path / "biosample.tsv", sep='\t'), new_biosample_df], ignore_index=True
    )
    merged_biosample_df.to_csv(c2m2_path / "biosample.tsv", sep='\t', index=False)
    print(f"Appended {len(new_biosample_df)} rows to biosample.tsv ({len(merged_biosample_df)} total)")

    # --- biosample_disease.tsv ---
    if args.peds:
        disease_df = generate_biosample_diseases_peds(bundle_path, id_map, biosample_df)
    else:
        disease_df = generate_biosample_diseases_adult(bundle_path, id_map, biosample_df)
    merged_disease_df = pd.concat(
        [pd.read_csv(c2m2_path / "biosample_disease.tsv", sep='\t'), disease_df], ignore_index=True
    )
    merged_disease_df.to_csv(c2m2_path / "biosample_disease.tsv", sep='\t', index=False)
    print(f"Appended {len(disease_df)} rows to biosample_disease.tsv ({len(merged_disease_df)} total)")

    # --- file_describes_biosample.tsv ---
    fdb_df = generate_file_describes_biosample(
        bundle_path, new_files_df, biosample_df, id_map
    )
    merged_fdb_df = pd.concat(
        [pd.read_csv(c2m2_path / "file_describes_biosample.tsv", sep='\t'), fdb_df],
        ignore_index=True
    )
    merged_fdb_df.to_csv(c2m2_path / "file_describes_biosample.tsv", sep='\t', index=False)
    print(f"Appended {len(fdb_df)} rows to file_describes_biosample.tsv ({len(merged_fdb_df)} total)")


if __name__ == "__main__":
    main()
