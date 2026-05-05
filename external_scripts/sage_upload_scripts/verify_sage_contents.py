import os
import argparse
from dotenv import load_dotenv

import synapseclient
import synapseutils
from synapseclient.models import Project, Folder, File
import hashlib

def get_file_md5(filepath):
    """Calculate MD5 hash of a local file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()



PEDS_PROJECT = "syn72418607"
ADULT_PROJECT = "syn72370534"
OVERALL_PROJECT = "syn72370534"
PEDS_FOLDER = "syn72493849"
ADULT_FOLDER = "syn72493850"


def walk_synapse_folder(syn, folder_id, path=""):
    """
    Recursively walk through a Synapse folder structure.
    Returns dict mapping relative paths to Synapse entity info.
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


def run_folder_comparisons(sage_folders, local_folders, dry_run=True):
    for folder in sage_folders:
        sage_path = folder['path']
        if os.path.exists(sage_path) and not os.path.isdir(sage_path):
            print(f"Sage ID {folder['id']} ({sage_path}) found locally but is not a directory")
            if not dry_run:
                # not even sure what or why this might happen so not sure what to do
                pass

    sage_folder_set = set([f['path'] for f in sage_folders])
    local_folder_set = set([f['path'] for f in local_folders])
    if (sage_folder_set - local_folder_set) != set():
        print("The following folders were found on Sage and not locally")
        for folder in (sage_folder_set - local_folder_set):
            print(f"\t{folder}")
            if not dry_run:
                # Add code to remove folders from Sage
                pass
    if (local_folder_set - sage_folder_set) != set():
        print("The following folders were found locally and not on Sage")
        for folder in (local_folder_set - sage_folder_set):
            print(f"\t{folder}")
            if not dry_run:
                # Add code to add folder to Sage
                pass


def run_file_comparisons(sage_files, local_files, md5=False, dry_run=True):
    for f in sage_files:
        sage_path = f
        if not os.path.exists(sage_path):
            print(f"Sage ID {sage_files[f]['id']} ({sage_path}) not found locally")
        elif not os.path.isfile(sage_path):
            print(f"Sage ID {sage_files[f]['id']} ({sage_path}) found locally but is not a file")
        elif sage_path in local_files and md5:
            if sage_files[f]['md5'] != local_files[f]['md5']:
                print(f"Sage ID {sage_files[f]['id']} ({sage_path}) md5 ({sage_files[f]['md5']}) does not match local md5 hash ({local_files[f]['md5']})")
                if not dry_run:
                    # Add logic for updating the files on sage
                    pass

    sage_file_set = set(sage_files.keys())
    local_file_set = set(local_files.keys())
    if (sage_file_set - local_file_set) != set():
        print("The following files were found on Sage and not locally")
        for folder in (sage_file_set - local_file_set):
            print(f"\t{folder}")
            if not dry_run:
                # Add logic for removing files from Sage
                pass
    if (local_file_set - sage_file_set) != set():
        print("The following files were found locally and not on Sage")
        for folder in (local_file_set - sage_file_set):
            print(f"\t{folder}")
            if not dry_run:
                # Add logic for adding file to Sage
                pass

    
def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="A simple script for ...")
    parser.add_argument('--bids_folder', default='./', type=str, help='Example help information')
    parser.add_argument('--adult', default=False, action='store_true', help='is adult dataset')
    parser.add_argument('--get_md5', default=False, action='store_true')
    parser.add_argument('--sync', default=False, action='store_true')

    args = parser.parse_args()
    
    sage_pat = os.getenv("SAGE_PAT")
    sage_project_id = OVERALL_PROJECT#ADULT_PROJECT if args.adult else PEDS_PROJECT

    syn = synapseclient.login(authToken=sage_pat)
    project = Project(id=sage_project_id).get()
    print(f"I just got my project: {project.name}, id: {project.id}")
    
    parent_id = ADULT_FOLDER if args.adult else PEDS_FOLDER

    my_project = Folder(id=parent_id).get()
    synapse_files, synapse_folders = walk_synapse_folder(syn,my_project,args.bids_folder)
    local_files, local_folders = walk_local_folder(args.bids_folder, args.get_md5)
    
    run_folder_comparisons(synapse_folders, local_folders, dry_run=~args.sync)
    run_file_comparisons(synapse_files, local_files, md5=args.get_md5, dry_run=~args.sync)

    #run_comparison(dir_mapping, args.bids_folder)


if __name__=='__main__':
    main()
