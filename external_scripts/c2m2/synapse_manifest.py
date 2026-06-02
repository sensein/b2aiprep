"""synapse_manifest.py — build per-cohort controlled-file manifests from Synapse.

Queries the citation-versioned fileviews READ-ONLY (get + getColumns + tableQuery;
never store/delete) and writes synapse_manifest_{adult,peds}.csv with columns:
    id, name, path, dataFileMD5Hex, dataFileSizeBytes
which controlled_to_c2m2.py auto-detects. `path` is the file's relative path within
the dataset, resolved only for NON-recording files (README/dataset_description/
phenotype) since recordings are renamed anonymously and ignore it.

Auth: reads SAGE_PAT (or SYNAPSE_PAT) from the environment, else ~/.synapseConfig.
Run in a shell where the token is available, e.g.:
    ! /home/wilke18/.conda/envs/b2aiprep_test/bin/python external_scripts/c2m2/synapse_manifest.py

Only --create-temp-views creates (and deletes) a temporary view; use it solely if a
citation view is unavailable.
"""
import os
import re
import argparse
import time
import synapseclient
from synapseclient import EntityViewSchema, EntityViewType

# Existing citation-versioned fileviews -- READ ONLY, never modified.
CITATION_VIEWS = {'adult': 'syn74341293', 'peds': 'syn74341287'}
# Scope folders, used only with --create-temp-views.
COHORT_FOLDERS = {'adult': 'syn72493850', 'peds': 'syn72493849'}

FILENAME_RE = re.compile(r'^sub-[^_]+_ses-[^_]+_task-.+\.(wav|json)$')
MD5_CANDIDATES = ['dataFileMD5Hex', 'md5', 'contentMd5']
SIZE_CANDIDATES = ['dataFileSizeBytes', 'contentSize', 'size']


def login():
    syn = synapseclient.Synapse()
    tok = os.environ.get('SAGE_PAT') or os.environ.get('SYNAPSE_PAT')
    syn.login(authToken=tok) if tok else syn.login()
    print('LOGIN OK as', syn.getUserProfile().get('userName'))
    return syn


def find_project(syn, sid):
    e = syn.get(sid, downloadFile=False)
    while 'Project' not in e.concreteType:
        e = syn.get(e['parentId'], downloadFile=False)
    return e.id


def pick(cols, candidates):
    return next((c for c in candidates if c in cols), None)


def query_with_retry(syn, view_id, select, tries=6, wait=10):
    for i in range(tries):
        try:
            df = syn.tableQuery(f"SELECT {select} FROM {view_id}").asDataFrame()
            if len(df) > 0:
                return df
            print(f"  view empty (attempt {i+1}/{tries}); waiting {wait}s...")
        except Exception as ex:
            print(f"  query attempt {i+1}/{tries} failed: {str(ex)[:120]}; waiting {wait}s...")
        time.sleep(wait)
    raise SystemExit(f"View {view_id} returned no rows after {tries} attempts")


def resolve_relpaths(syn, df, name_col, parent_col, stop_ids):
    """Return a list of relative paths; only non-recording rows are resolved."""
    cache = {}

    def folder(fid):
        if fid not in cache:
            f = syn.get(fid, downloadFile=False)
            cache[fid] = (f.name, f.get('parentId'))
        return cache[fid]

    paths = []
    for name, parent in zip(df[name_col].astype(str), df[parent_col].astype(str)):
        if FILENAME_RE.match(name):           # recording: path unused
            paths.append('')
            continue
        parts, cur, guard = [], parent, 0
        while cur and cur not in stop_ids and guard < 50:
            nm, par = folder(cur)
            parts.append(nm)
            cur = par
            guard += 1
        paths.append('/'.join(reversed(parts) + [name]) if parts else name)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir', default=os.getcwd(), help='Where to write the manifest CSVs')
    parser.add_argument('--create-temp-views', action='store_true',
                        help='Create+delete a temp view per cohort instead of using citation views.')
    parser.add_argument('--adult_view', default=None, help='Override adult view synId')
    parser.add_argument('--peds_view', default=None, help='Override peds view synId')
    args = parser.parse_args()
    overrides = {'adult': args.adult_view, 'peds': args.peds_view}

    syn = login()
    for cohort in ('adult', 'peds'):
        print(f"\n=== {cohort} ===")
        created = False
        if overrides[cohort]:
            view_id = overrides[cohort]
            print(f"  using override view {view_id}")
        elif args.create_temp_views:
            folder = COHORT_FOLDERS[cohort]
            project = find_project(syn, folder)
            view = syn.store(EntityViewSchema(
                name=f"_tmp_c2m2_manifest_{cohort}", parent=project, scopes=[folder],
                includeEntityTypes=[EntityViewType.FILE],
                addDefaultViewColumns=True, addAnnotationColumns=False))
            view_id, created = view.id, True
            print(f"  project {project}; created temp view {view_id}")
        else:
            view_id = CITATION_VIEWS[cohort]
            print(f"  using citation view {view_id} (read-only)")

        try:
            view = syn.get(view_id, downloadFile=False)
            if 'View' not in view.concreteType:
                raise SystemExit(f"{view_id} is not a fileview ({view.concreteType})")
            cols = [c['name'] for c in syn.getColumns(view_id)]
            md5_col = pick(cols, MD5_CANDIDATES)
            size_col = pick(cols, SIZE_CANDIDATES)
            has_parent = 'parentId' in cols
            print(f"  {view.name!r} | md5={md5_col} size={size_col} parentId={'yes' if has_parent else 'NO'}")
            select = ', '.join(['id', 'name'] + (['parentId'] if has_parent else [])
                               + [c for c in (md5_col, size_col) if c])
            df = query_with_retry(syn, view_id, select)

            stop_ids = set(getattr(view, 'scopeIds', []) or []) | {COHORT_FOLDERS[cohort]}
            if has_parent:
                df['path'] = resolve_relpaths(syn, df, 'name', 'parentId', stop_ids)
            else:
                print("  WARNING: no parentId column; non-recording files will use basename as path.")
                df['path'] = df['name']

            out_cols = ['id', 'name', 'path'] + [c for c in (md5_col, size_col) if c]
            out = os.path.join(args.out_dir, f"synapse_manifest_{cohort}.csv")
            df[out_cols].to_csv(out, index=False)
            n_rec = df['name'].astype(str).str.match(FILENAME_RE).sum()
            n_md5 = df[md5_col].notna().sum() if md5_col else 0
            print(f"  wrote {out}: {len(df)} files ({n_rec} recordings, "
                  f"{len(df) - n_rec} other), {n_md5} with md5")
        finally:
            if created:
                syn.delete(view_id)
                print(f"  deleted temp view {view_id}")


if __name__ == '__main__':
    main()
