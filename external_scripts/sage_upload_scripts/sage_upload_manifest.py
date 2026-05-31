import os
import re
import time
import hashlib
import argparse
from dotenv import load_dotenv

import synapseclient
import synapseutils
import tempfile
import pandas as pd

# Matches a BIDS subject directory segment anywhere in a path (any separator,
# leading or mid-path), e.g. "sub-010729/..." or ".../sub-010729/...".
_SUBJECT_DIR = r"(?:^|[\\/])sub-[^\\/]+(?:[\\/]|$)"

# Retry defaults for transient upload errors (e.g. 412 update conflicts).
_DEFAULT_MAX_RETRIES = 4
_DEFAULT_RETRY_BASE_DELAY_S = 5
_MAX_RETRY_DELAY_S = 60


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


def _local_md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _server_md5(syn, parent_id, name):
    """Current Synapse md5 for child `name` under `parent_id`, or None if absent."""
    try:
        sid = syn.findEntityId(name, parent=parent_id)
    except Exception:
        sid = None
    if not sid:
        return None
    try:
        return syn.get(sid, downloadFile=False).get('md5', None)
    except Exception:
        return None


def _unsynced(syn, df):
    """Return the rows whose local file does NOT yet match the server md5.

    This is how we decide whether a (possibly 412'd) upload actually landed: a
    file already on Synapse with a matching md5 is considered done and dropped.
    """
    mask = []
    for path, parent in zip(df['path'], df['parent']):
        if not os.path.isfile(path):
            mask.append(False)  # nothing to upload locally
            continue
        mask.append(_server_md5(syn, parent, os.path.basename(path)) != _local_md5(path))
    return df[mask]


def _sync_once(syn, df, dry_run):
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.tsv', delete=True) as temp:
        df.to_csv(temp.name, sep='\t', index=False)
        synapseutils.syncToSynapse(
            syn=syn, manifestFile=temp.name, sendMessages=False, dryRun=dry_run
        )


def upload_with_retry(syn, df, max_retries, base_delay):
    """Upload df, retrying transient failures with exponential backoff.

    After any failure we md5-check each file against the server and retry only
    the ones that still don't match -- so a 412 that already landed is not
    re-uploaded, and genuinely failed files are retried.
    """
    remaining = df
    for attempt in range(max_retries + 1):
        try:
            _sync_once(syn, remaining, dry_run=False)
            return  # clean success, nothing raised
        except Exception as e:
            print(f"Upload attempt {attempt} raised {type(e).__name__}: {e}")
            remaining = _unsynced(syn, remaining)
            if len(remaining) == 0:
                print("All files verified on Synapse with matching md5 -- "
                      "treating as success despite the error.")
                return
            print(f"{len(remaining)} file(s) still mismatch the server after attempt {attempt}.")
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), _MAX_RETRY_DELAY_S)
                print(f"Retrying {len(remaining)} file(s) in {delay}s ...")
                time.sleep(delay)
            else:
                raise RuntimeError(
                    f"Upload failed after {max_retries} retries; "
                    f"{len(remaining)} file(s) still mismatch the server."
                )


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
    parser.add_argument('--max_retries', default=_DEFAULT_MAX_RETRIES, type=int,
                        help='Retries for transient upload failures (md5-verified). 0 disables retrying.')
    parser.add_argument('--retry_base_delay', default=_DEFAULT_RETRY_BASE_DELAY_S, type=float,
                        help='Base seconds for exponential backoff between retries.')

    args = parser.parse_args()

    sage_pat = os.getenv("SAGE_PAT")

    syn = synapseclient.login(authToken=sage_pat)
    data = pd.read_csv(args.manifest_file, sep='\t')
    total = len(data)
    data = filter_manifest(data, subject=args.subject, toplevel=args.toplevel)
    if args.subject and len(data) == 0 and total > 0:
        print(f"WARNING: subject={args.subject} matched 0 of {total} manifest rows. "
              f"Sample manifest path: {data['path'].iloc[0] if len(data) else pd.read_csv(args.manifest_file, sep='\t')['path'].iloc[0]}")
    start = args.start if args.start and args.start >= 0 else 0
    end = args.end if args.end else len(data)
    df = data[start:end]
    print(f"Uploading {len(df)} matched rows of {total} in manifest "
          f"(subject={args.subject}, toplevel={args.toplevel}, dry_run={args.dry_run})")

    if args.dry_run:
        _sync_once(syn, df, dry_run=True)
    else:
        upload_with_retry(syn, df, args.max_retries, args.retry_base_delay)


if __name__=='__main__':
    main()
