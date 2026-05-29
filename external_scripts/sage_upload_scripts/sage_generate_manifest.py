import os
import argparse
from dotenv import load_dotenv

import synapseclient
import synapseutils
from synapseclient.models import Project, Folder

from sage_config import get_dataset_config, validate_sync_folder


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate a Synapse sync manifest from a local BIDS folder."
    )
    parser.add_argument('--bids_folder', default='./', type=str,
                        help='Path to the local BIDS folder to upload.')
    parser.add_argument('--manifest_file', default='temp_manifest_file.tsv', type=str,
                        help='Path to write the generated manifest TSV.')
    parser.add_argument('--adult', default=False, action='store_true',
                        help='Target the adult dataset. Omit for the pediatric dataset.')

    args = parser.parse_args()
    print(f"Received args: {args}")

    sage_pat = os.getenv("SAGE_PAT")
    config = get_dataset_config(args.adult)

    syn = synapseclient.login(authToken=sage_pat)
    project = Project(id=config["project"]).get()
    print(f"Looking at project: {project.name}, id: {project.id}")

    parent_id = config["folder"]
    print(f"Syncing data with parent folder: {parent_id}")
    root_data_folder = Folder(id=parent_id).get()

    # Sanity check that we resolved the expected destination folder before upload.
    # The folder is nested inside the project (not necessarily at its top level),
    # so we validate by name rather than asserting it is a direct child.
    validate_sync_folder(root_data_folder, config)

    synapseutils.generate_sync_manifest(
        syn=syn,
        directory_path=args.bids_folder,
        parent_id=parent_id,
        manifest_path=args.manifest_file,
    )


if __name__=='__main__':
    main()
