import os
import re
import time
import logging
import argparse

from synapseclient.models import Project, Folder
from synapseclient import EntityViewSchema, EntityViewType

from sage_config import (
    get_dataset_config, validate_sync_folder, subject_segment, file_md5,
    login, configure_logging,
)

logger = logging.getLogger(__name__)

# Column-name candidates for a file's MD5 in an entity view.
_MD5_CANDIDATES = ["dataFileMD5Hex", "md5", "contentMd5"]
# Citation/DOI fileviews. This script must NEVER create against, query, or delete
# these -- the --use_view path only ever uses its own temporary view.
_DOI_VIEWS = {"syn74341293", "syn74341287"}


def _has_md5(value):
    """True only for a usable (non-empty string) md5; guards against None/NaN."""
    return isinstance(value, str) and len(value) > 0


def walk_synapse_folder(syn, folder_id, path=""):
    """Recursively walk a Synapse folder via per-file ``syn.get``.

    This is the slow O(files)-network-calls fallback used when ``--use_view`` is
    not set; prefer ``build_from_view`` for whole-dataset checks. Returns a tuple
    ``(files, folders)``: ``files`` maps relative paths to entity info, ``folders``
    is a list of folder info dicts.
    """
    synapse_files = {}
    synapse_folders = []

    try:
        children = syn.getChildren(folder_id)

        for child in children:
            child_name = child['name']
            child_id = child['id']
            child_type = child['type']

            rel_path = os.path.join(path, child_name) if path else child_name

            if child_type == 'org.sagebionetworks.repo.model.Folder':
                logger.info("Scanning folder: %s", rel_path)
                synapse_folders.append({'path': rel_path, 'id': child_id})
                syn_files, syn_folders = walk_synapse_folder(syn, child_id, rel_path)
                synapse_files.update(syn_files)
                synapse_folders += syn_folders

            elif child_type == 'org.sagebionetworks.repo.model.FileEntity':
                file_entity = syn.get(child_id, downloadFile=False)
                synapse_files[rel_path] = {
                    'id': child_id,
                    'name': child_name,
                    'md5': file_entity.get('md5', None),
                    'path': rel_path,
                }
    except Exception as e:
        # A partial walk under-reports Sage files (safe direction: fewer deletion
        # candidates), but flag it loudly so a truncated scan isn't mistaken for clean.
        logger.error("Error processing folder %s: %s", path, str(e))

    return synapse_files, synapse_folders


def find_child_folder_id(syn, parent_id, name):
    """Return the Synapse ID of the child folder named `name`, or None."""
    for child in syn.getChildren(parent_id):
        if (child['name'] == name
                and child['type'] == 'org.sagebionetworks.repo.model.Folder'):
            return child['id']
    return None


def _default_workers():
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        return os.cpu_count() or 4


def _enumerate_local(path):
    """Recursively list (file_paths, folder_paths) under path (no hashing)."""
    files, folders = [], []
    for entity in os.listdir(path):
        full = os.path.join(path, entity)
        if os.path.isdir(full):
            folders.append(full)
            f, d = _enumerate_local(full)
            files += f
            folders += d
        else:
            files.append(full)
    return files, folders


def walk_local_folder(path, get_md5=False, workers=1):
    """List local files/folders; md5 the files in parallel when get_md5 is set.

    Local hashing of many small files over networked storage is I/O-latency bound,
    so hashing files concurrently (workers>1) overlaps that latency and is much
    faster than sequential hashing.
    """
    file_paths, folder_paths = _enumerate_local(path)
    if get_md5 and file_paths:
        if workers and workers > 1:
            from multiprocessing import Pool
            with Pool(workers) as pool:
                md5s = pool.map(file_md5, file_paths, chunksize=8)
        else:
            md5s = [file_md5(p) for p in file_paths]
    else:
        md5s = [None] * len(file_paths)
    local_files = {p: {'md5': m, 'path': p} for p, m in zip(file_paths, md5s)}
    local_folders = [{'path': p} for p in folder_paths]
    return local_files, local_folders


def delete_from_sage(syn, entity_id, label, dry_run=True):
    """Move a Synapse entity to the project Trash (recoverable until purged).

    Returns True if a deletion was performed. Content is not fully removed until
    the Trash is purged, which is done manually (e.g. for PII).
    """
    if dry_run:
        return False
    try:
        syn.delete(entity_id)
        logger.info("\tDELETED %s (id %s) -> moved to Trash "
                    "(purge the Trash to remove content, e.g. for PII)", label, entity_id)
        return True
    except Exception as e:
        logger.error("\tFAILED to delete %s (id %s): %s", label, entity_id, e)
        return False


def run_folder_comparisons(syn, sage_folders, local_folders, dry_run=True):
    for folder in sage_folders:
        sage_path = folder['path']
        if os.path.exists(sage_path) and not os.path.isdir(sage_path):
            logger.warning("Sage ID %s (%s) found locally but is not a directory",
                           folder['id'], sage_path)

    sage_folder_id = {f['path']: f['id'] for f in sage_folders}
    sage_folder_set = set(sage_folder_id.keys())
    local_folder_set = set([f['path'] for f in local_folders])
    sage_only = sorted(sage_folder_set - local_folder_set)
    if sage_only:
        header = ("The following folders are on Sage but not local (deletion candidates; "
                  "re-run with --sync to delete):") if dry_run else \
                 ("Deleting folders that are on Sage but not local (and their contents):")
        logger.info(header)
        for folder in sage_only:
            logger.info("\t%s (id %s)", folder, sage_folder_id[folder])
            # A folder deletion cascades to its contents. Some of those entities may
            # already have been trashed (files are handled first), so tolerate errors.
            delete_from_sage(syn, sage_folder_id[folder], folder, dry_run=dry_run)
    if local_folder_set - sage_folder_set:
        logger.info("The following folders were found locally and not on Sage "
                    "(re-run the upload manifest flow to create them on Sage):")
        for folder in (local_folder_set - sage_folder_set):
            logger.info("\t%s", folder)


def run_file_comparisons(syn, sage_files, local_files, md5=False, dry_run=True):
    for f in sage_files:
        sage_path = f
        if not os.path.exists(sage_path):
            logger.info("Sage ID %s (%s) not found locally", sage_files[f]['id'], sage_path)
        elif not os.path.isfile(sage_path):
            logger.warning("Sage ID %s (%s) found locally but is not a file",
                           sage_files[f]['id'], sage_path)
        elif sage_path in local_files and md5:
            smd5, lmd5 = sage_files[f]['md5'], local_files[f]['md5']
            if not _has_md5(smd5):
                logger.warning("Sage ID %s (%s) has no server md5 to compare against",
                               sage_files[f]['id'], sage_path)
            elif smd5 != lmd5:
                logger.info("Sage ID %s (%s) md5 (%s) does not match local md5 hash (%s)",
                            sage_files[f]['id'], sage_path, smd5, lmd5)

    sage_file_set = set(sage_files.keys())
    local_file_set = set(local_files.keys())
    sage_only = sorted(sage_file_set - local_file_set)
    if sage_only:
        header = ("The following files are on Sage but not local (deletion candidates; "
                  "re-run with --sync to delete):") if dry_run else \
                 ("Deleting files that are on Sage but not local:")
        logger.info(header)
        for path in sage_only:
            logger.info("\t%s (id %s)", path, sage_files[path]['id'])
            delete_from_sage(syn, sage_files[path]['id'], path, dry_run=dry_run)
    if local_file_set - sage_file_set:
        logger.info("The following files were found locally and not on Sage "
                    "(re-run the upload manifest flow to add them to Sage):")
        for path in (local_file_set - sage_file_set):
            logger.info("\t%s", path)


def build_from_view(syn, config, bids_folder):
    """Fast alternative to walk_synapse_folder using a TEMPORARY file+folder view.

    Creates an EntityView (FILE+FOLDER) scoped to the cohort folder, queries it
    once for all files and folders, reconstructs each path in-memory from the
    folder rows (no per-file/per-folder syn.get), then deletes the temp view.
    Returns (files, folders) keyed by the local-equivalent absolute path so the
    existing comparison functions can be reused. Never touches the DOI views.
    """
    root = config["folder"]
    view = syn.store(EntityViewSchema(
        name=f"_tmp_verify_{root}", parent=config["project"], scopes=[root],
        includeEntityTypes=[EntityViewType.FILE, EntityViewType.FOLDER],
        addDefaultViewColumns=True, addAnnotationColumns=False))
    vid = view.id
    assert vid not in _DOI_VIEWS, f"refusing to operate on citation view {vid}"
    logger.info("Created temp view %s over %s", vid, root)
    try:
        cols = [c["name"] for c in syn.getColumns(vid)]
        md5col = next((c for c in _MD5_CANDIDATES if c in cols), None)
        select = "id,name,parentId,type" + (f",{md5col}" if md5col else "")
        df = None
        for attempt in range(40):  # a freshly created view needs time to index
            try:
                df = syn.tableQuery(f"SELECT {select} FROM {vid}").asDataFrame()
                if len(df) > 0:
                    break
            except Exception as e:
                logger.info("  view query attempt %d failed: %s", attempt + 1, str(e)[:90])
            time.sleep(15)
        if df is None or len(df) == 0:
            raise SystemExit(f"temp view {vid} returned no rows")

        folders_df = df[df["type"].astype(str).str.lower() == "folder"]
        files_df = df[df["type"].astype(str).str.lower() == "file"]
        logger.info("View rows: %d (%d files, %d folders)", len(df), len(files_df), len(folders_df))
        if len(folders_df) == 0 and len(files_df) > 0:
            raise SystemExit("view has no folder rows; cannot reconstruct paths")
        fmap = {r.id: (str(r.name), r.parentId) for r in folders_df.itertuples(index=False)}

        def relpath(name, parent):
            """Relative path from `root` down to `name`, or None if it can't be
            fully resolved (a folder missing from the view, NaN parentId, a cycle,
            or excessive depth). Returning None -- rather than a truncated path --
            prevents a mis-keyed entity from later being treated as a deletion
            candidate under --sync."""
            parts, cur, guard = [], parent, 0
            while cur != root:
                if guard >= 200 or cur is None or cur not in fmap:
                    return None
                nm, par = fmap[cur]
                parts.append(nm)
                cur = par
                guard += 1
            return os.path.join(*(list(reversed(parts)) + [name]))

        files, folders, unresolved = {}, [], []
        for r in files_df.itertuples(index=False):
            rp = relpath(str(r.name), r.parentId)
            if rp is None:
                unresolved.append((r.id, str(r.name)))
                continue
            key = os.path.join(bids_folder, rp)
            files[key] = {"id": r.id, "name": str(r.name),
                          "md5": getattr(r, md5col) if md5col else None, "path": key}
        for r in folders_df.itertuples(index=False):
            rp = relpath(str(r.name), r.parentId)
            if rp is None:
                unresolved.append((r.id, str(r.name)))
                continue
            folders.append({"path": os.path.join(bids_folder, rp), "id": r.id})

        if unresolved:
            raise SystemExit(
                f"Could not reconstruct paths for {len(unresolved)} entity/entities from the "
                f"view (incomplete folder set / NaN parent?); refusing to proceed so --sync "
                f"cannot delete valid files. Examples: {unresolved[:5]}"
            )
        return files, folders
    finally:
        # We only ever delete the temp view we just created; _DOI_VIEWS can never
        # be `vid`, so this guard is purely defensive.
        if vid not in _DOI_VIEWS:
            syn.delete(vid)
            logger.info("Deleted temp view %s", vid)


def main():
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Compare a local BIDS folder against its Synapse destination folder."
    )
    parser.add_argument('--bids_folder', default='./', type=str,
                        help='Local BIDS folder to compare against Synapse.')
    parser.add_argument('--adult', default=False, action='store_true',
                        help='Target the adult dataset. Omit for the pediatric dataset.')
    parser.add_argument('--get_md5', default=False, action='store_true',
                        help='Compute local MD5 hashes and compare them to the Synapse files.')
    parser.add_argument('--sync', default=False, action='store_true',
                        help='Apply changes on Synapse. Without it the script only reports differences (dry run).')
    parser.add_argument('--subject', default=None, type=str,
                        help='Restrict the comparison to one subject (e.g. sub-010729) for a quick spot check.')
    parser.add_argument('--use_view', default=False, action='store_true',
                        help='Fast md5 check via a temporary file+folder view instead of a per-file '
                             'walk (one query + in-memory path reconstruction). Implies an md5 '
                             'comparison and never touches the DOI/citation views.')
    parser.add_argument('--workers', default=None, type=int,
                        help='Parallel processes for local md5 hashing. Defaults to allocated CPUs.')

    args = parser.parse_args()
    workers = args.workers or _default_workers()

    config = get_dataset_config(args.adult)

    syn = login()
    project = Project(id=config["project"]).get()
    logger.info("Looking at project: %s, id: %s", project.name, project.id)

    parent_id = config["folder"]
    root_data_folder = Folder(id=parent_id).get()
    validate_sync_folder(root_data_folder, config)

    # The view path is only meaningful as an md5 comparison, so it implies --get_md5.
    md5 = args.get_md5 or args.use_view
    dry_run = not args.sync

    if args.use_view:
        synapse_files, synapse_folders = build_from_view(syn, config, args.bids_folder)
        local_files, local_folders = walk_local_folder(args.bids_folder, md5, workers)
        if args.subject:
            # Same segment matcher as the upload filter so the two never diverge.
            pat = re.compile(subject_segment(args.subject))
            synapse_files = {k: v for k, v in synapse_files.items() if pat.search(k)}
            local_files = {k: v for k, v in local_files.items() if pat.search(k)}
            synapse_folders = [d for d in synapse_folders if pat.search(d['path'])]
            local_folders = [d for d in local_folders if pat.search(d['path'])]
            if not synapse_files and not local_files:
                logger.warning("subject=%s matched no files on Sage or locally -- check the id",
                               args.subject)
    elif args.subject:
        # Scope both walks to the subject's subtree. Keys on both sides are
        # prefixed with "<bids_folder>/<subject>" so they line up.
        local_root = os.path.join(args.bids_folder, args.subject)
        if not os.path.isdir(local_root):
            raise SystemExit(f"Subject folder not found locally: {local_root}")
        subject_folder_id = find_child_folder_id(syn, parent_id, args.subject)
        if subject_folder_id is None:
            logger.warning("Subject %s not found on Sage under %s; "
                           "all local files will report as missing on Sage.", args.subject, parent_id)
            synapse_files, synapse_folders = {}, []
        else:
            synapse_files, synapse_folders = walk_synapse_folder(syn, subject_folder_id, local_root)
        local_files, local_folders = walk_local_folder(local_root, md5, workers)
    else:
        synapse_files, synapse_folders = walk_synapse_folder(syn, parent_id, args.bids_folder)
        local_files, local_folders = walk_local_folder(args.bids_folder, md5, workers)

    # Delete files before folders so cascading folder deletes don't trip over
    # children that were already trashed.
    run_file_comparisons(syn, synapse_files, local_files, md5=md5, dry_run=dry_run)
    run_folder_comparisons(syn, synapse_folders, local_folders, dry_run=dry_run)


if __name__=='__main__':
    main()
