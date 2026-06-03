import os
import time
import logging
import argparse
import tempfile

import synapseutils
import pandas as pd

from sage_config import (
    SUBJECT_DIR_RE, subject_segment, file_md5, login, configure_logging,
)

logger = logging.getLogger(__name__)

# Retry defaults for transient upload errors (e.g. 412 update conflicts).
_DEFAULT_MAX_RETRIES = 4
_DEFAULT_RETRY_BASE_DELAY_S = 5
_MAX_RETRY_DELAY_S = 60


def resolve_range(start, end, n):
    """Resolve --start/--end into a [start, end) row slice over n rows.

    Uses `is not None` so a deliberate 0 is honored, and rejects invalid ranges
    instead of silently coercing them. Returns (start, end); raises ValueError.
    """
    start = start if start is not None else 0
    end = end if end is not None else n
    if start < 0 or end < 0 or start > end:
        raise ValueError(f"Invalid row range: start={start}, end={end} (have {n} rows)")
    return start, end


def filter_manifest(data, subject=None, toplevel=False):
    """Return the subset of manifest rows to upload.

    --subject restricts to one subject's subtree (per-subject parallel uploads,
    which avoid concurrent writes to the same Synapse folder). --toplevel selects
    the dataset-level files that live outside any sub-* directory.
    """
    if subject:
        return data[data['path'].str.contains(subject_segment(subject), regex=True)]
    if toplevel:
        return data[~data['path'].str.contains(SUBJECT_DIR_RE, regex=True)]
    return data


def _server_md5(syn, parent_id, name, tries=3):
    """Current Synapse md5 for child `name` under `parent_id`, or None if absent.

    Retries transient lookup errors so a flaky get does not make a correctly
    uploaded file look unsynced (which would otherwise exhaust retries and raise).
    A genuine None (entity absent) is returned only after the retries.
    """
    sid = None
    for _ in range(tries):
        try:
            sid = syn.findEntityId(name, parent=parent_id)
            break
        except Exception:
            time.sleep(1)
    if not sid:
        return None
    for _ in range(tries):
        try:
            return syn.get(sid, downloadFile=False).get('md5', None)
        except Exception:
            time.sleep(1)
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
        mask.append(_server_md5(syn, parent, os.path.basename(path)) != file_md5(path))
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
            logger.warning("Upload attempt %d raised %s: %s", attempt, type(e).__name__, e)
            remaining = _unsynced(syn, remaining)
            if len(remaining) == 0:
                logger.info("All files verified on Synapse with matching md5 -- "
                            "treating as success despite the error.")
                return
            logger.info("%d file(s) still mismatch the server after attempt %d.",
                        len(remaining), attempt)
            if attempt < max_retries:
                delay = min(base_delay * (2 ** attempt), _MAX_RETRY_DELAY_S)
                logger.info("Retrying %d file(s) in %ss ...", len(remaining), delay)
                time.sleep(delay)
            else:
                raise RuntimeError(
                    f"Upload failed after {max_retries} retries; "
                    f"{len(remaining)} file(s) still mismatch the server."
                )


def main():
    configure_logging()

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

    syn = login()
    full = pd.read_csv(args.manifest_file, sep='\t')
    total = len(full)
    data = filter_manifest(full, subject=args.subject, toplevel=args.toplevel)
    if args.subject and len(data) == 0 and total > 0:
        logger.warning("subject=%s matched 0 of %d manifest rows. Sample manifest path: %s",
                       args.subject, total, full['path'].iloc[0])

    try:
        start, end = resolve_range(args.start, args.end, len(data))
    except ValueError as e:
        raise SystemExit(str(e))
    df = data[start:end]
    logger.info("Uploading %d matched rows of %d in manifest (subject=%s, toplevel=%s, dry_run=%s)",
                len(df), total, args.subject, args.toplevel, args.dry_run)

    if args.dry_run:
        _sync_once(syn, df, dry_run=True)
    else:
        upload_with_retry(syn, df, args.max_retries, args.retry_base_delay)


if __name__=='__main__':
    main()
