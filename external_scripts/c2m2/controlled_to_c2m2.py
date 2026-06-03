"""controlled_to_c2m2.py — add controlled-access files to a C2M2 package.

The controlled bundle has BIDS-style raw audio recordings
    sub-<participant_id>/ses-<session_id>/audio/sub-<participant_id>_ses-<session_id>_task-<task>.{wav,json}
plus dataset-level files (README, dataset_description.json) and the phenotype tree.

Two kinds of file rows are written, both under the *_controlled project:
  - Recordings (task files): identified by the sub/ses/task name. They link to an
    existing biosample via file_describes_biosample, and their public identifiers
    are anonymous -- local_id/filename are built from the anon biosample id, never
    the raw sub-<id> name. Biosamples and the participant->anon_id map come from the
    registered run (bundle_to_c2m2.py); run this AFTER it against the SAME
    --c2m2_id_map. A (participant, session) the registered run never saw gets a new
    anon_id minted and a biosample created (divergence), so files never dangle.
  - Non-recording files (README, dataset_description, phenotype TSV/JSON): described
    by a plain file row with their real relative path (these names carry no subject
    identity) and NO biosample link.

Disease-link firewall: only recordings link to biosamples, and recording file nodes
carry no subject id, so there is no public path from a subject to a disease-bearing
biosample. The raw BIDS name and the Synapse id live only in the optional, PRIVATE
--crosswalk_out CSV, never in the package.

File checksums: C2M2 requires at least one of sha256/md5 per file. md5/size come from
the Sage manifest. In directory mode size comes from os.stat and md5 only with --hash.
"""

from c2m2_mappings import *
from constants import *
from bundle_to_c2m2 import lookup_task_obi, generate_file_describes_biosample

from pathlib import Path
import pandas as pd
import argparse
import hashlib
import re

# sub-<id>_ses-<id>_task-<task>.<ext>  (task may contain hyphens; ext anchors it)
FILENAME_RE = re.compile(r'^sub-(?P<sub>[^_]+)_ses-(?P<ses>[^_]+)_task-(?P<task>.+)\.(?P<ext>wav|json)$')

# Manifest column auto-detection (first present wins); override via CLI flags.
NAME_COL_CANDIDATES = ['name', 'filename', 'dataFileName']
RELPATH_COL_CANDIDATES = ['path', 'relpath', 'key']
MD5_COL_CANDIDATES = ['dataFileMD5Hex', 'dataFileMd5Hex', 'md5', 'contentMd5']
SIZE_COL_CANDIDATES = ['dataFileSizeBytes', 'contentSize', 'dataFileSize', 'size']
SYNID_COL_CANDIDATES = ['id', 'synId', 'synapseId']

FILE_COLS = ['id_namespace', 'local_id', 'project_id_namespace', 'project_local_id',
             'persistent_id', 'creation_time', 'size_in_bytes', 'uncompressed_size_in_bytes',
             'sha256', 'md5', 'filename', 'file_format', 'compression_format', 'data_type',
             'assay_type', 'analysis_type', 'mime_type', 'bundle_collection_id_namespace',
             'bundle_collection_local_id', 'dbgap_study_id', 'access_url']

FDB_COLS = ['file_id_namespace', 'file_local_id', 'biosample_id_namespace', 'biosample_local_id']


def parse_recording(name):
    """Parse a BIDS audio basename into (participant_id, session_id, task, ext).

    participant_id/session_id are returned exactly as they appear after the
    sub-/ses- prefixes (zero-padding preserved), matching how the registered
    feature files — and the id_map — store them. Returns None for non-recordings.
    """
    m = FILENAME_RE.match(Path(name).name)
    if not m:
        return None
    return m.group('sub'), m.group('ses'), m.group('task'), '.' + m.group('ext')


def coerce_size(size):
    try:
        return int(size) if size is not None and str(size) not in ('', 'nan') else None
    except (TypeError, ValueError):
        return None


def make_file_row(project, local_id, filename, ext, md5, size):
    return {
        'id_namespace': ID_NAMESPACE,
        'local_id': local_id,
        'project_id_namespace': PROJECT_ID_NAMESPACE,
        'project_local_id': project,
        'persistent_id': None,
        'creation_time': None,
        'size_in_bytes': coerce_size(size),
        'uncompressed_size_in_bytes': None,
        'sha256': None,
        'md5': md5 or None,
        'filename': filename,
        'file_format': file_format_map.get(ext, ''),
        'compression_format': None,
        'data_type': data_type_map.get(ext, ''),
        'assay_type': None,
        'analysis_type': None,
        'mime_type': mime_type_map.get(ext, ''),
        'bundle_collection_id_namespace': None,
        'bundle_collection_local_id': None,
        'dbgap_study_id': None,
        'access_url': None,
    }


def pick_column(df, override, candidates, what):
    if override:
        if override not in df.columns:
            raise SystemExit(f"--{what} '{override}' not in manifest columns: {list(df.columns)}")
        return override
    for c in candidates:
        if c in df.columns:
            return c
    return None


def records_from_manifest(args):
    """Yield {name, relpath, md5, size, synid} from a Sage manifest CSV."""
    df = pd.read_csv(args.manifest, dtype=str)
    name_col = pick_column(df, args.name_col, NAME_COL_CANDIDATES, 'name_col')
    if name_col is None:
        raise SystemExit(f"No name column found in manifest. Columns: {list(df.columns)}")
    relpath_col = pick_column(df, args.path_col, RELPATH_COL_CANDIDATES, 'path_col')
    md5_col = pick_column(df, args.md5_col, MD5_COL_CANDIDATES, 'md5_col')
    size_col = pick_column(df, args.size_col, SIZE_COL_CANDIDATES, 'size_col')
    synid_col = pick_column(df, args.synid_col, SYNID_COL_CANDIDATES, 'synid_col')
    print(f"Manifest columns -> name:{name_col} path:{relpath_col} md5:{md5_col} "
          f"size:{size_col} synid:{synid_col}")
    if md5_col is None:
        print("Warning: no md5 column found in manifest; files will lack a checksum "
              "and fail C2M2 validation.")
    for _, row in df.iterrows():
        yield {
            'name': row[name_col],
            'relpath': row[relpath_col] if relpath_col and pd.notna(row[relpath_col]) else None,
            'md5': row[md5_col] if md5_col else None,
            'size': row[size_col] if size_col else None,
            'synid': row[synid_col] if synid_col else None,
        }


def records_from_directory(bundle_path, do_hash):
    """Yield {name, relpath, md5, size, synid} for every file under the bundle."""
    for f in sorted(p for p in bundle_path.rglob('*') if p.is_file()):
        md5 = None
        if do_hash:
            h = hashlib.md5()
            with open(f, 'rb') as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b''):
                    h.update(chunk)
            md5 = h.hexdigest()
        yield {'name': f.name, 'relpath': str(f.relative_to(bundle_path)),
               'md5': md5, 'size': f.stat().st_size, 'synid': None}


def main():
    parser = argparse.ArgumentParser(
        description='Add controlled-access files to a C2M2 package.')
    parser.add_argument('bundle_path', help='Path to the controlled BIDS bundle (used in directory mode)')
    parser.add_argument('c2m2_path', help='Path to the C2M2 output directory')
    parser.add_argument('--c2m2_id_map', required=True, type=str,
                        help='Persistent participant->anon_id CSV from the registered run')
    parser.add_argument('--peds', action='store_true', help='Use pediatric controlled project')
    parser.add_argument('--manifest', type=str, default=None,
                        help='Sage manifest CSV (provides md5/size/synId/path). If omitted, walks bundle_path.')
    parser.add_argument('--hash', action='store_true',
                        help='Directory mode only: compute md5 by reading each file (slow).')
    parser.add_argument('--crosswalk_out', type=str, default=None,
                        help='Optional PRIVATE CSV mapping synId/raw name -> anon local_id. Never package it.')
    parser.add_argument('--name_col', default=None, help='Override manifest basename column')
    parser.add_argument('--path_col', default=None, help='Override manifest relative-path column')
    parser.add_argument('--md5_col', default=None, help='Override manifest md5 column')
    parser.add_argument('--size_col', default=None, help='Override manifest size column')
    parser.add_argument('--synid_col', default=None, help='Override manifest Synapse-id column')
    args = parser.parse_args()

    bundle_path = Path(args.bundle_path)
    c2m2_path = Path(args.c2m2_path)
    file_project = (PEDS_CONTROLLED_PROJECT_LOCAL_ID if args.peds
                    else ADULT_CONTROLLED_PROJECT_LOCAL_ID)
    # Divergent biosamples (in controlled but not registered) belong to the
    # tier-neutral cohort project, like all biosamples.
    biosample_project = PEDS_PROJECT_LOCAL_ID if args.peds else PROJECT_LOCAL_ID

    # (participant_id, session_id) -> anon_id, shared with the registered run.
    id_map = pd.read_csv(args.c2m2_id_map, dtype=str)
    anon_lookup = {(r.participant_id, r.session_id): r.anon_id
                   for r in id_map.itertuples(index=False)}
    next_anon = len(id_map) + 1
    new_idmap_rows = []

    biosample_df = pd.read_csv(c2m2_path / 'biosample.tsv', sep='\t', dtype=str)
    valid_biosamples = set(biosample_df['local_id'])
    new_biosample_rows = []
    biosample_cols = ['id_namespace', 'local_id', 'project_id_namespace', 'project_local_id',
                      'persistent_id', 'creation_time', 'sample_prep_method', 'anatomy', 'biofluid']

    if args.manifest:
        records = records_from_manifest(args)
    else:
        print("No --manifest given; walking the bundle directory (size via stat; md5 only with --hash).")
        records = records_from_directory(bundle_path, args.hash)

    file_rows, fdb_rows, crosswalk_rows = [], [], []
    nonrec_local_ids = []          # non-recording files, for phenotype->biosample linking
    seen_local_ids = set()
    n_total = n_rec = n_nonrec = n_minted = n_new_biosample = n_no_obi = n_dup = n_no_checksum = 0

    for rec in records:
        n_total += 1
        parsed = parse_recording(rec['name'])

        # --- non-recording file: real path, no biosample link ---
        if parsed is None:
            rel = rec.get('relpath') or rec['name']
            local_id = f"{file_project}/{rel}"
            if local_id in seen_local_ids:
                n_dup += 1
                continue
            seen_local_ids.add(local_id)
            ext = Path(rec['name']).suffix.lower()
            if not (rec.get('md5')):
                n_no_checksum += 1
            file_rows.append(make_file_row(file_project, local_id, Path(rec['name']).name,
                                            ext, rec.get('md5'), rec.get('size')))
            nonrec_local_ids.append(local_id)
            n_nonrec += 1
            if args.crosswalk_out:
                crosswalk_rows.append({'synapse_id': rec.get('synid'), 'raw_name': rec['name'],
                                       'relpath': rel, 'file_local_id': local_id,
                                       'participant_id': '', 'session_id': '', 'task_name': '',
                                       'anon_id': '', 'biosample_local_id': ''})
            continue

        # --- recording: anon naming + biosample link ---
        pid, ses, task, ext = parsed
        anon = anon_lookup.get((pid, ses))
        if anon is None:
            anon = f"{next_anon:06d}"
            next_anon += 1
            anon_lookup[(pid, ses)] = anon
            new_idmap_rows.append({'participant_id': pid, 'session_id': ses, 'anon_id': anon})
            n_minted += 1
        bid = f"{anon}_{task}"
        local_id = f"{file_project}/{anon}_{task}{ext}"
        if local_id in seen_local_ids:
            n_dup += 1
            continue
        seen_local_ids.add(local_id)

        if bid not in valid_biosamples:
            obi = lookup_task_obi(task)
            if obi is None:
                n_no_obi += 1
            else:
                new_biosample_rows.append({
                    'id_namespace': ID_NAMESPACE, 'local_id': bid,
                    'project_id_namespace': PROJECT_ID_NAMESPACE,
                    'project_local_id': biosample_project,
                    'persistent_id': None, 'creation_time': None,
                    'sample_prep_method': obi, 'anatomy': 'UBERON:0001737', 'biofluid': None,
                })
                valid_biosamples.add(bid)
                n_new_biosample += 1

        if not (rec.get('md5')):
            n_no_checksum += 1
        file_rows.append(make_file_row(file_project, local_id, f"{anon}_{task}{ext}",
                                        ext, rec.get('md5'), rec.get('size')))
        n_rec += 1

        if bid in valid_biosamples:
            fdb_rows.append({
                'file_id_namespace': ID_NAMESPACE, 'file_local_id': local_id,
                'biosample_id_namespace': ID_NAMESPACE, 'biosample_local_id': bid,
            })

        if args.crosswalk_out:
            crosswalk_rows.append({'synapse_id': rec.get('synid'), 'raw_name': rec['name'],
                                   'relpath': rec.get('relpath') or '', 'file_local_id': local_id,
                                   'participant_id': pid, 'session_id': ses, 'task_name': task,
                                   'anon_id': anon, 'biosample_local_id': bid})

    # --- extend the shared id_map and biosample.tsv for any divergent combos ---
    if new_idmap_rows:
        id_map = pd.concat([id_map, pd.DataFrame(new_idmap_rows)], ignore_index=True)
        id_map.to_csv(args.c2m2_id_map, index=False)
    if new_biosample_rows:
        merged_bio = pd.concat(
            [pd.read_csv(c2m2_path / 'biosample.tsv', sep='\t'),
             pd.DataFrame(new_biosample_rows, columns=biosample_cols)], ignore_index=True)
        merged_bio.to_csv(c2m2_path / 'biosample.tsv', sep='\t', index=False)

    new_files_df = pd.DataFrame(file_rows, columns=FILE_COLS)
    files_df = pd.concat([pd.read_csv(c2m2_path / 'file.tsv', sep='\t'), new_files_df],
                         ignore_index=True)
    files_df.to_csv(c2m2_path / 'file.tsv', sep='\t', index=False)

    # Phenotype/task files describe biosamples too — link them exactly as the
    # registered run does, by reusing generate_file_describes_biosample on the
    # controlled bundle's non-recording files (reads phenotype/task/*.tsv locally).
    # biosample_df needs anon_id + local_id; reconstruct anon_id from the local_id.
    bf = biosample_df
    if new_biosample_rows:
        bf = pd.concat([bf, pd.DataFrame(new_biosample_rows, columns=biosample_cols)],
                       ignore_index=True)
    bf = bf[['local_id']].copy()
    bf['anon_id'] = bf['local_id'].str.split('_', n=1).str[0]
    pheno_fdb = generate_file_describes_biosample(
        bundle_path, pd.DataFrame({'local_id': nonrec_local_ids}), bf, id_map)
    n_pheno = len(pheno_fdb)

    new_fdb_df = pd.concat([pd.DataFrame(fdb_rows, columns=FDB_COLS), pheno_fdb],
                           ignore_index=True)
    fdb_df = pd.concat([pd.read_csv(c2m2_path / 'file_describes_biosample.tsv', sep='\t'),
                        new_fdb_df], ignore_index=True)
    fdb_df.to_csv(c2m2_path / 'file_describes_biosample.tsv', sep='\t', index=False)

    if args.crosswalk_out and crosswalk_rows:
        pd.DataFrame(crosswalk_rows).to_csv(args.crosswalk_out, index=False)
        print(f"Wrote PRIVATE crosswalk ({len(crosswalk_rows)} rows) to {args.crosswalk_out} "
              f"-- do NOT include this in the package.")

    print(f"\nControlled ingestion summary ({file_project}):")
    print(f"  records seen:                   {n_total}")
    print(f"  recording file rows:            {n_rec}")
    print(f"  non-recording file rows:        {n_nonrec}")
    print(f"  file_describes_biosample (recordings): {len(fdb_rows)}")
    print(f"  file_describes_biosample (phenotype):  {n_pheno}")
    print(f"  anon_ids minted (divergent):    {n_minted}")
    print(f"  biosamples created (divergent): {n_new_biosample}")
    print(f"  duplicate local_ids skipped:    {n_dup}")
    print(f"  recordings unlinked (task w/o OBI term): {n_no_obi}")
    if n_no_checksum:
        print(f"  WARNING: {n_no_checksum} file(s) have no md5/sha256 -> will fail C2M2 validation.")


if __name__ == '__main__':
    main()
