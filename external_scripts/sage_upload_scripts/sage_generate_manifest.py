import os
import argparse
from dotenv import load_dotenv

import synapseclient
import synapseutils
from synapseclient.models import Project

PEDS_PROJECT = "syn72418607"
ADULT_PROJECT = "syn72370534"
OVERALL_PROJECT = "syn72370534"
PEDS_FOLDER = "syn72493849"
ADULT_FOLDER = "syn72493850"

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="A simple script for ...")
    parser.add_argument('--bids_folder', default='./', type=str, help='Example help information')
    parser.add_argument('--manifest_file', default='temp_manifest_file.tsv',type=str)
    parser.add_argument('--adult', default=False, action='store_true', help='is adult dataset')

    args = parser.parse_args()
    print(f"Received args: {args}")
    
    sage_pat = os.getenv("SAGE_PAT")
    sage_project_id = OVERALL_PROJECT#ADULT_PROJECT if args.adult else PEDS_PROJECT

    syn = synapseclient.login(authToken=sage_pat)
    project = Project(id=sage_project_id).get()
    print(f"I just got my project: {project.name}, id: {project.id}")
    
    parent_id = ADULT_FOLDER if args.adult else PEDS_FOLDER
    print(f"Syncing data with parent folder: {parent_id}")

    synapseutils.generate_sync_manifest(
        syn=syn,
        directory_path=args.bids_folder,
        parent_id=parent_id,
        manifest_path=args.manifest_file,
    )


if __name__=='__main__':
    main()
