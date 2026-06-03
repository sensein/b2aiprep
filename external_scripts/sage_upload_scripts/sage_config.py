"""Shared Synapse config and helpers for the Sage upload scripts.

Each dataset is uploaded into a single destination folder that lives inside its
own Synapse project. ``folder`` is the sync target -- everything is uploaded
*inside* it -- and ``folder_name`` is the expected name of that folder, used to
sanity-check that we resolved the right destination before touching any data.

Note: the adult and pediatric data live in separate Synapse projects.
"""
import os
import re
import hashlib
import logging

from dotenv import load_dotenv
import synapseclient

ADULT = {
    "project": "syn72370534",
    "folder": "syn72493850",
    "folder_name": "Dataset - Adult Data",
}

PEDS = {
    "project": "syn73617068",
    "folder": "syn72493849",
    "folder_name": "Dataset - Pediatric Data",
}

# Matches a BIDS subject directory segment anywhere in a path (any separator,
# leading or mid-path), e.g. "sub-010729/..." or ".../sub-010729/...".
SUBJECT_DIR_RE = r"(?:^|[\\/])sub-[^\\/]+(?:[\\/]|$)"


def get_dataset_config(adult):
    """Return the project/folder config for the adult or pediatric dataset."""
    return ADULT if adult else PEDS


def validate_sync_folder(folder, config):
    """Raise if the resolved Synapse folder is not the expected sync target."""
    if folder.name != config["folder_name"]:
        raise ValueError(
            f"Expected sync folder named {config['folder_name']!r} "
            f"(id {config['folder']}) but resolved {folder.name!r}. "
            "Refusing to proceed against the wrong folder."
        )


def subject_segment(subject):
    """Regex matching `subject` as a full path segment, regardless of separators
    or position. Shared by the upload filter and the verify subject scope so the
    two never diverge."""
    return rf"(?:^|[\\/]){re.escape(subject)}(?:[\\/]|$)"


def file_md5(path, chunk_size=1 << 20):
    """MD5 hex digest of a local file (1 MiB reads)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def login():
    """Log in to Synapse using the SAGE_PAT environment variable (loaded from .env)."""
    load_dotenv()
    return synapseclient.login(authToken=os.getenv("SAGE_PAT"))


def configure_logging(level=logging.INFO):
    """Configure logging once for the Sage scripts, quieting noisy HTTP loggers."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("synapseclient", "urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
