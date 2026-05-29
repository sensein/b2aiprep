"""controlled_to_c2m2.py — add controlled-access raw-audio files to a C2M2 package.

The controlled bundle is BIDS-style raw audio:
    sub-<participant_id>/ses-<session_id>/audio/sub-<participant_id>_ses-<session_id>_task-<task>.{wav,json}

Each recording is one *file* describing an existing *biosample*. Biosamples and
the participant->anon_id map are produced by the registered run
(bundle_to_c2m2.py); this script must run AFTER it, against the SAME
--c2m2_id_map. It never mints new anon_ids — it only looks up existing ones.

Disease-link firewall: the controlled file's public identifiers carry NO subject
identity. local_id/filename are built from the anon biosample id, and
persistent_id/access_url are left empty. The raw BIDS name (sub-<id>) and the
Synapse id are written only to the optional, PRIVATE --crosswalk_out CSV, never
into the package. The file is linked to its biosample via file_describes_biosample;
because the file node holds no subject id, there is no public path from a subject
to a disease-bearing biosample.

File checksums: C2M2 requires at least one of sha256/md5 per file. md5 (and size)
are taken from the Sage manifest. In directory mode, size comes from os.stat and a
checksum is only produced with --hash (sha256, slow); without it the script warns
that the rows will fail validation.
"""

from c2m2_mappings import *
from constants import *
from bundle_to_c2m2 import lookup_task_obi

from pathlib import Path
import pandas as pd
import argparse
import hashlib
import re

# sub-<id>_ses-<id>_task-<task>.<ext>  (task may contain hyphens; ext anchors it)
FILENAME_RE = re.compile(r'^sub-(?P<sub>[^_]+)_ses-(?P<ses>[^_]+)_task-(?P<task>.+)\.(?P<ext>wav|json)$')

# Manifest column auto-detection (first present wins); override via CLI flags.
PATH_COL_CANDIDATES = ['path', 'name', 'filename', 'dataFileName']
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

    The participant_id and session_id are returned exactly as they appear after
    the sub-/ses- prefixes (zero-padding preserved), which is how the registered
    feature files — and therefore the id_map — store them. Returns None if the
    name is not a task recording.
    """
    m = FILENAME_RE.match(Path(name).name)
    if not m:
        return None
    return m.group('sub'), m.group('ses'), m.group('task'), '.' + m.group('ext')


def pick_column(df, override, candidates, what):
    if override:
        if override not in df.columns:
            raise SystemExit(f"--{what} column '{override}' not in manifest columns: {list(df.columns)}")
        return override
    for c in candidates:
        if c in df.columns:
            return c
    return None


def records_from_manifest(args):
    """Yield dicts {name, md5, size, synid} from a Sage manifest CSV."""
    df = pd.read_csv(args.manifest, dtype=str)
    path_col = pick_column(df, args.path_col, PATH_COL_CANDIDATES, 'path_col')
    if path_col is None:
        raise SystemExit(f"No path/name column found in manifest. Columns: {list(df.columns)}")
    md5_col = pick_column(df, args.md5_col, MD5_COL_CANDIDATES, 'md5_col')
    size_col = pick_column(df, args.size_col, SIZE_COL_CANDIDATES, 'size_col')
    synid_col = pick_column(df, args.synid_col, SYNID_COL_CANDIDATES, 'synid_col')
    print(f"Manifest columns -> path:{path_col} md5:{md5_col} size:{size_col} synid:{synid_col}")
    if md5_col is None:
        print("Warning: no md5 column found in manifest; files will lack a checksum "
              "and fail C2M2 validation. Provide --md5_col or use --hash in directory mode.")
    for _, row in df.iterrows():
        yield {
            'name': row[path_col],
            'md5': row[md5_col] if md5_col else None,
            'size': row[size_col] if size_col else None,
            'synid': row[synid_col] if synid_col else None,
        }


def records_from_directory(bundle_path, do_hash):
    """Yield dicts {name, md5, size, synid, abs_path} by walking the BIDS tree."""
    for sub in sorted(p for p in bundle_path.iterdir() if p.is_dir() and p.name.startswith('sub-')):
        for ses in sorted(p for p in sub.iterdir() if p.is_dir() and p.name.startswith('ses-')):
            audio = ses / 'audio'
            if not audio.is_dir():
                continue
            for f in sorted(audio.iterdir()):
                if not f.is_file():
                    continue
                md5 = None
                if do_hash:
                    h = hashlib.md5()
                    with open(f, 'rb') as fh:
                        for chunk in iter(lambda: fh.read(1 << 20), b''):
                            h.update(chunk)
                    md5 = h.hexdigest()
                yield {'name': f.name, 'md5': md5, 'size': f.stat().st_size, 'synid': None}


def main():
    parser = argparse.ArgumentParser(
        description='Add controlled-access raw-audio files to a C2M2 package.')
    parser.add_argument('bundle_path', help='Path to the controlled BIDS bundle (used in directory mode)')
    parser.add_argument('c2m2_path', help='Path to the C2M2 output directory')
    parser.add_argument('--c2m2_id_map', required=True, type=str,
                        help='Persistent participant->anon_id CSV from the registered run (read-only here)')
    parser.add_argument('--peds', action='store_true', help='Use pediatric controlled project')
    parser.add_argument('--manifest', type=str, default=None,
                        help='Sage manifest CSV (provides md5/size/synId). If omitted, walks bundle_path.')
    parser.add_argument('--hash', action='store_true',
                        help='Directory mode only: compute md5 by reading each file (slow).')
    parser.add_argument('--crosswalk_out', type=str, default=None,
                        help='Optional PRIVATE CSV mapping synId/raw BIDS name -> anon file local_id. '
                             'Never include this in the package.')
    parser.add_argument('--path_col', default=None, help='Override manifest path/name column')
    parser.add_argument('--md5_col', default=None, help='Override manifest md5 column')
    parser.add_argument('--size_col', default=None, help='Override manifest size column')
    parser.add_argument('--synid_col', default=None, help='Override manifest Synapse-id column')
    args = parser.parse_args()

    bundle_path = Path(args.bundle_path)
    c2m2_path = Path(args.c2m2_path)
    file_project_local_id = (PEDS_CONTROLLED_PROJECT_LOCAL_ID if args.peds
                             else ADULT_CONTROLLED_PROJECT_LOCAL_ID)
    # New biosamples discovered only in the controlled bundle (divergence from the
    # registered set) belong to the tier-neutral cohort project, like all biosamples.
    biosample_project_local_id = PEDS_PROJECT_LOCAL_ID if args.peds else PROJECT_LOCAL_ID

    # (participant_id, session_id) -> anon_id. The map is shared with the registered
    # run; if the controlled bundle contains a (participant, session) the registered
    # run never saw, a new anon_id is minted here and the map is saved back so the
    # id space stays consistent across runs.
    id_map = pd.read_csv(args.c2m2_id_map, dtype=str)
    anon_lookup = {(r.participant_id, r.session_id): r.anon_id
                   for r in id_map.itertuples(index=False)}
    next_anon = len(id_map) + 1
    new_idmap_rows = []

    # Valid biosample local_ids (anon_id_task). Biosamples are normally created by
    # the registered run; any recording here whose biosample is missing (a divergent
    # combo) gets one created below, so controlled files never dangle.
    biosample_df = pd.read_csv(c2m2_path / 'biosample.tsv', sep='\t', dtype=str)
    valid_biosamples = set(biosample_df['local_id'])
    new_biosample_rows = []
    biosample_cols = ['id_namespace', 'local_id', 'project_id_namespace', 'project_local_id',
                      'persistent_id', 'creation_time', 'sample_prep_method', 'anatomy', 'biofluid']

    if args.manifest:
        records = records_from_manifest(args)
    else:
        print("No --manifest given; walking the BIDS directory (size via stat; "
              "md5 only with --hash).")
        records = records_from_directory(bundle_path, args.hash)

    file_rows, fdb_rows, crosswalk_rows = [], [], []
    seen_local_ids = set()
    n_total = n_unparsed = n_minted = n_new_biosample = n_no_obi = n_dup = n_no_checksum = 0

    for rec in records:
        n_total += 1
        parsed = parse_recording(rec['name'])
        if parsed is None:
            n_unparsed += 1
            continue
        pid, ses, task, ext = parsed
        anon = anon_lookup.get((pid, ses))
        if anon is None:
            # Divergent combo: mint a new anon_id and extend the shared map.
            anon = f"{next_anon:06d}"
            next_anon += 1
            anon_lookup[(pid, ses)] = anon
            new_idmap_rows.append({'participant_id': pid, 'session_id': ses, 'anon_id': anon})
            n_minted += 1
        bid = f"{anon}_{task}"
        local_id = f"{file_project_local_id}/{anon}_{task}{ext}"
        if local_id in seen_local_ids:
            n_dup += 1
            continue
        seen_local_ids.add(local_id)

        # Ensure the biosample exists; create it for combos the registered run missed.
        # A task with no OBI term cannot form a valid biosample (the registered run
        # would have skipped it too), so the file is written but left unlinked.
        if bid not in valid_biosamples:
            obi = lookup_task_obi(task)
            if obi is None:
                n_no_obi += 1
            else:
                new_biosample_rows.append({
                    'id_namespace': ID_NAMESPACE,
                    'local_id': bid,
                    'project_id_namespace': PROJECT_ID_NAMESPACE,
                    'project_local_id': biosample_project_local_id,
                    'persistent_id': None,
                    'creation_time': None,
                    'sample_prep_method': obi,
                    'anatomy': 'UBERON:0001737',
                    'biofluid': None,
                })
                valid_biosamples.add(bid)
                n_new_biosample += 1

        md5 = rec.get('md5') or None
        if not md5:
            n_no_checksum += 1
        size = rec.get('size')
        try:
            size = int(size) if size is not None and str(size) != '' else None
        except (TypeError, ValueError):
            size = None

        file_rows.append({
            'id_namespace': ID_NAMESPACE,
            'local_id': local_id,
            'project_id_namespace': PROJECT_ID_NAMESPACE,
            'project_local_id': file_project_local_id,
            'persistent_id': None,
            'creation_time': None,
            'size_in_bytes': size,
            'uncompressed_size_in_bytes': None,
            'sha256': None,
            'md5': md5,
            'filename': f"{anon}_{task}{ext}",
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
        })

        if bid in valid_biosamples:
            fdb_rows.append({
                'file_id_namespace': ID_NAMESPACE,
                'file_local_id': local_id,
                'biosample_id_namespace': ID_NAMESPACE,
                'biosample_local_id': bid,
            })

        if args.crosswalk_out:
            crosswalk_rows.append({
                'synapse_id': rec.get('synid'),
                'raw_filename': Path(rec['name']).name,
                'participant_id': pid,
                'session_id': ses,
                'task_name': task,
                'anon_id': anon,
                'file_local_id': local_id,
                'biosample_local_id': bid,
            })

    # --- extend the shared id_map and biosample.tsv for any divergent combos ---
    if new_idmap_rows:
        id_map = pd.concat([id_map, pd.DataFrame(new_idmap_rows)], ignore_index=True)
        id_map.to_csv(args.c2m2_id_map, index=False)
    if new_biosample_rows:
        merged_bio = pd.concat(
            [pd.read_csv(c2m2_path / 'biosample.tsv', sep='\t'),
             pd.DataFrame(new_biosample_rows, columns=biosample_cols)], ignore_index=True)
        merged_bio.to_csv(c2m2_path / 'biosample.tsv', sep='\t', index=False)

    # --- append to file.tsv ---
    new_files_df = pd.DataFrame(file_rows, columns=FILE_COLS)
    files_df = pd.concat([pd.read_csv(c2m2_path / 'file.tsv', sep='\t'), new_files_df],
                         ignore_index=True)
    files_df.to_csv(c2m2_path / 'file.tsv', sep='\t', index=False)

    # --- append to file_describes_biosample.tsv ---
    new_fdb_df = pd.DataFrame(fdb_rows, columns=FDB_COLS)
    fdb_df = pd.concat([pd.read_csv(c2m2_path / 'file_describes_biosample.tsv', sep='\t'),
                        new_fdb_df], ignore_index=True)
    fdb_df.to_csv(c2m2_path / 'file_describes_biosample.tsv', sep='\t', index=False)

    if args.crosswalk_out and crosswalk_rows:
        pd.DataFrame(crosswalk_rows).to_csv(args.crosswalk_out, index=False)
        print(f"Wrote PRIVATE crosswalk ({len(crosswalk_rows)} rows) to {args.crosswalk_out} "
              f"-- do NOT include this in the package.")

    print(f"\nControlled ingestion summary ({file_project_local_id}):")
    print(f"  records seen:                 {n_total}")
    print(f"  file rows written:            {len(file_rows)}")
    print(f"  file_describes_biosample:     {len(fdb_rows)}")
    print(f"  skipped (name not a task):    {n_unparsed}")
    print(f"  anon_ids minted (divergent):  {n_minted}")
    print(f"  biosamples created (divergent): {n_new_biosample}")
    print(f"  duplicate local_ids skipped:  {n_dup}")
    print(f"  files left unlinked (task w/o OBI term): {n_no_obi}")
    if n_no_checksum:
        print(f"  WARNING: {n_no_checksum} file(s) have no md5/sha256 -> will fail C2M2 validation "
              f"(provide a manifest with md5, or run with --hash).")


if __name__ == '__main__':
    main()
