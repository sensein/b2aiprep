import os
import re
import argparse
from dotenv import load_dotenv

import synapseclient
import synapseutils
import tempfile
import pandas as pd

# Matches a BIDS subject directory segment anywhere in a path (any separator,
# leading or mid-path), e.g. "sub-010729/..." or ".../sub-010729/...".
_SUBJECT_DIR = r"(?:^|[\\/])sub-[^\\/]+(?:[\\/]|$)"


def _subject_segment(subject):
    """Regex matching `subject` as a path segment regardless of separators/position."""
    return rf"(?:^|[\\/]){re.escape(subject)}(?:[\\/]|$)"


def filter_manifest(data, subject=None, toplevel=False):
    """Return the subset of manifest rows to upload.

    --subject restricts to one subject's subtree (per-subject parallel uploads,
    which avoid concurrent writes to the same Synapse folder). --toplevel selects
    the dataset-level files that live outside any sub-* directory.
    """
    if subject:
        return data[data['path'].str.contains(_subject_segment(subject), regex=True)]
    if toplevel:
        return data[~data['path'].str.contains(_SUBJECT_DIR, regex=True)]
    return data


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Upload a Synapse sync manifest, optionally filtered for per-subject parallel uploads."
    )
    parser.add_argument('--manifest_file', default='temp_manifest_file.tsv', type=str,
                        help='Path to the manifest TSV produced by sage_generate_manifest.py.')
    parser.add_argument('--subject', default=None, type=str,
                        help='Upload only rows under this subject directory (e.g. sub-010729).')
    parser.add_argument('--toplevel', default=False, action='store_true',
                        help='Upload only dataset-level rows (those not under any sub-* directory).')
    parser.add_argument('--start', default=None, type=int,
                        help='First row (inclusive) to upload after filtering. Defaults to the start.')
    parser.add_argument('--end', default=None, type=int,
                        help='Last row (exclusive) to upload after filtering. Defaults to the end.')
    parser.add_argument('--dry_run', default=False, action='store_true',
                        help='Validate the (filtered) manifest without uploading or touching the cache.')

    args = parser.parse_args()

    sage_pat = os.getenv("SAGE_PAT")

    syn = synapseclient.login(authToken=sage_pat)
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.tsv', delete=True) as temp:
        data = pd.read_csv(args.manifest_file, sep='\t')
        total = len(data)
        data = filter_manifest(data, subject=args.subject, toplevel=args.toplevel)
        if args.subject and len(data) == 0 and total > 0:
            print(f"WARNING: subject={args.subject} matched 0 of {total} manifest rows. "
                  f"Sample manifest path: {pd.read_csv(args.manifest_file, sep='\t')['path'].iloc[0]}")
        start = args.start if args.start and args.start >= 0 else 0
        end = args.end if args.end else len(data)
        df = data[start:end]
        print(f"Uploading {len(df)} matched rows of {total} in manifest "
              f"(subject={args.subject}, toplevel={args.toplevel}, dry_run={args.dry_run})")
        df.to_csv(temp.name, sep='\t', index=False)
        synapseutils.syncToSynapse(
            syn=syn, manifestFile=temp.name, sendMessages=False, dryRun=args.dry_run
        )


if __name__=='__main__':
    main()
