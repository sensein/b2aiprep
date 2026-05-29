import os
import argparse
from dotenv import load_dotenv

import synapseclient
import synapseutils
from synapseclient.models import Project, Folder, File
import hashlib

from sage_config import get_dataset_config, validate_sync_folder


def get_file_md5(filepath):
    """Calculate MD5 hash of a local file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def walk_synapse_folder(syn, folder_id, path=""):
    """
    Recursively walk through a Synapse folder structure.
    Returns a tuple ``(files, folders)`` where ``files`` is a dict mapping
    relative paths to Synapse entity info and ``folders`` is a list of folder
    info dicts.
    """
    synapse_files = {}
    synapse_folders = []

    try:
        # Get all children of this folder
        children = syn.getChildren(folder_id)

        for child in children:
            child_name = child['name']
            child_id = child['id']
            child_type = child['type']

            # Build relative path
            rel_path = os.path.join(path, child_name) if path else child_name

            if child_type == 'org.sagebionetworks.repo.model.Folder':
                print(f"Scanning folder: {rel_path}")
                synapse_folders.append({'path':rel_path,'id':child_id})
                # Recursively process subfolder
                syn_files, syn_folders = walk_synapse_folder(syn, child_id, rel_path)
                synapse_files.update(syn_files)
                synapse_folders += syn_folders

            elif child_type == 'org.sagebionetworks.repo.model.FileEntity':
                # Get file metadata without downloading
                file_entity = syn.get(child_id, downloadFile=False)

                synapse_files[rel_path] = {
                    'id': child_id,
                    'name': child_name,
                    'md5': file_entity.get('md5', None),
                    'size': file_entity.get('dataFileHandleId', {}).get('contentSize', 0) if isinstance(file_entity.get('dataFileHandleId'), dict) else 0,
                    'modifiedOn': file_entity.get('modifiedOn', None),
                    'path': rel_path
                }

                #print(f"Found file: {rel_path}")

    except Exception as e:
        print(f"Error processing folder {path}: {str(e)}")

    return synapse_files, synapse_folders


def find_child_folder_id(syn, parent_id, name):
    """Return the Synapse ID of the child folder named `name`, or None."""
    for child in syn.getChildren(parent_id):
        if (child['name'] == name
                and child['type'] == 'org.sagebionetworks.repo.model.Folder'):
            return child['id']
    return None


def walk_local_folder(path, get_md5=False):
    local_files = {}
    local_folders = []

    for entity in os.listdir(path):
        rel_path = os.path.join(path,entity)
        if os.path.isdir(rel_path):
            local_folders.append({'path':rel_path})
            temp_files, temp_folders = walk_local_folder(rel_path, get_md5=get_md5)
            local_files.update(temp_files)
            local_folders += temp_folders
        else:
            local_files[rel_path] = {
                'md5': get_file_md5(rel_path) if get_md5 else None,
                'path': rel_path
            }
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
        print(f"\tDELETED {label} (id {entity_id}) -> moved to Trash "
              "(purge the Trash to remove content, e.g. for PII)")
        return True
    except Exception as e:
        print(f"\tFAILED to delete {label} (id {entity_id}): {e}")
        return False


def run_folder_comparisons(syn, sage_folders, local_folders, dry_run=True):
    for folder in sage_folders:
        sage_path = folder['path']
        if os.path.exists(sage_path) and not os.path.isdir(sage_path):
            print(f"Sage ID {folder['id']} ({sage_path}) found locally but is not a directory")

    sage_folder_id = {f['path']: f['id'] for f in sage_folders}
    sage_folder_set = set(sage_folder_id.keys())
    local_folder_set = set([f['path'] for f in local_folders])
    sage_only = sorted(sage_folder_set - local_folder_set)
    if sage_only:
        header = ("The following folders are on Sage but not local (deletion candidates; "
                  "re-run with --sync to delete):") if dry_run else \
                 ("Deleting folders that are on Sage but not local (and their contents):")
        print(header)
        for folder in sage_only:
            print(f"\t{folder} (id {sage_folder_id[folder]})")
            # A folder deletion cascades to its contents. Some of those entities may
            # already have been trashed (files are handled first), so tolerate errors.
            delete_from_sage(syn, sage_folder_id[folder], folder, dry_run=dry_run)
    if local_folder_set - sage_folder_set:
        print("The following folders were found locally and not on Sage")
        print("(re-run the upload manifest flow to create them on Sage):")
        for folder in (local_folder_set - sage_folder_set):
            print(f"\t{folder}")


def run_file_comparisons(syn, sage_files, local_files, md5=False, dry_run=True):
    for f in sage_files:
        sage_path = f
        if not os.path.exists(sage_path):
            print(f"Sage ID {sage_files[f]['id']} ({sage_path}) not found locally")
        elif not os.path.isfile(sage_path):
            print(f"Sage ID {sage_files[f]['id']} ({sage_path}) found locally but is not a file")
        elif sage_path in local_files and md5:
            if sage_files[f]['md5'] != local_files[f]['md5']:
                print(f"Sage ID {sage_files[f]['id']} ({sage_path}) md5 ({sage_files[f]['md5']}) does not match local md5 hash ({local_files[f]['md5']})")
                # Content changed locally; re-run the upload manifest flow to push a
                # new version. This script does not re-upload file content.

    sage_file_set = set(sage_files.keys())
    local_file_set = set(local_files.keys())
    sage_only = sorted(sage_file_set - local_file_set)
    if sage_only:
        header = ("The following files are on Sage but not local (deletion candidates; "
                  "re-run with --sync to delete):") if dry_run else \
                 ("Deleting files that are on Sage but not local:")
        print(header)
        for path in sage_only:
            print(f"\t{path} (id {sage_files[path]['id']})")
            delete_from_sage(syn, sage_files[path]['id'], path, dry_run=dry_run)
    if local_file_set - sage_file_set:
        print("The following files were found locally and not on Sage")
        print("(re-run the upload manifest flow to add them to Sage):")
        for path in (local_file_set - sage_file_set):
            print(f"\t{path}")


def main():
    load_dotenv()

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

    args = parser.parse_args()

    sage_pat = os.getenv("SAGE_PAT")
    config = get_dataset_config(args.adult)

    syn = synapseclient.login(authToken=sage_pat)
    project = Project(id=config["project"]).get()
    print(f"Looking at project: {project.name}, id: {project.id}")

    parent_id = config["folder"]
    root_data_folder = Folder(id=parent_id).get()
    validate_sync_folder(root_data_folder, config)

    if args.subject:
        # Scope both walks to the subject's subtree. Keys on both sides are
        # prefixed with "<bids_folder>/<subject>" so they line up.
        local_root = os.path.join(args.bids_folder, args.subject)
        if not os.path.isdir(local_root):
            raise SystemExit(f"Subject folder not found locally: {local_root}")
        subject_folder_id = find_child_folder_id(syn, parent_id, args.subject)
        if subject_folder_id is None:
            print(f"Subject {args.subject} not found on Sage under {parent_id}; "
                  "all local files will report as missing on Sage.")
            synapse_files, synapse_folders = {}, []
        else:
            synapse_files, synapse_folders = walk_synapse_folder(syn, subject_folder_id, local_root)
        local_files, local_folders = walk_local_folder(local_root, args.get_md5)
    else:
        synapse_files, synapse_folders = walk_synapse_folder(syn, parent_id, args.bids_folder)
        local_files, local_folders = walk_local_folder(args.bids_folder, args.get_md5)

    dry_run = not args.sync
    # Delete files before folders so cascading folder deletes don't trip over
    # children that were already trashed.
    run_file_comparisons(syn, synapse_files, local_files, md5=args.get_md5, dry_run=dry_run)
    run_folder_comparisons(syn, synapse_folders, local_folders, dry_run=dry_run)


if __name__=='__main__':
    main()
