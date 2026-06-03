import argparse
import logging

import synapseutils
import pandas as pd
from synapseclient.models import Project, Folder

from sage_config import get_dataset_config, validate_sync_folder, login, configure_logging

logger = logging.getLogger(__name__)


def main():
    configure_logging()

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
    logger.info("Received args: %s", args)

    config = get_dataset_config(args.adult)

    syn = login()
    project = Project(id=config["project"]).get()
    logger.info("Looking at project: %s, id: %s", project.name, project.id)

    parent_id = config["folder"]
    logger.info("Syncing data with parent folder: %s", parent_id)
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

    # Guard: a wrong/empty --bids_folder silently yields a header-only manifest,
    # which later "uploads" nothing. Fail loudly instead.
    manifest = pd.read_csv(args.manifest_file, sep='\t')
    logger.info("Wrote %s with %d file rows", args.manifest_file, len(manifest))
    if len(manifest) == 0:
        raise SystemExit(
            f"Manifest {args.manifest_file} has 0 file rows -- is --bids_folder "
            f"({args.bids_folder!r}) correct and non-empty?"
        )


if __name__=='__main__':
    main()
