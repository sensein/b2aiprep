"""Opt-in integration test against a THROWAWAY Synapse project (no real data).

This exercises the real round trip the unit tests can't: generate manifest ->
upload -> build a temp file+folder view -> reconstruct paths -> compare to local,
plus the "missing locally" and "md5 mismatch" detections.

It is skipped unless you point it at a disposable project you own:

    export SAGE_PAT=...                 # token with modify rights on the test project
    export SAGE_TEST_PROJECT=syn#####   # a throwaway project, NOT a real dataset
    pytest test_sage_integration.py -q

It creates a temp folder under that project, uploads a few synthetic files, runs
the checks, and deletes the temp folder afterward. It refuses to run against the
known real dataset projects.
"""
import os
import sys
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sage_config as cfg
from verify_sage_contents import build_from_view, walk_local_folder

TEST_PROJECT = os.getenv("SAGE_TEST_PROJECT")
_REAL_PROJECTS = {cfg.ADULT["project"], cfg.PEDS["project"]}

pytestmark = pytest.mark.skipif(
    not TEST_PROJECT, reason="set SAGE_TEST_PROJECT to a throwaway Synapse project to run"
)


@pytest.fixture(scope="module")
def synapse():
    assert TEST_PROJECT not in _REAL_PROJECTS, "refusing to run against a real dataset project"
    yield cfg.login()


@pytest.fixture
def staged(tmp_path, synapse):
    """Create a small synthetic BIDS-ish tree locally and upload it into a fresh
    temp folder under the test project. Yields (config, bids_folder, syn); cleans up."""
    import synapseclient
    import synapseutils

    # synthetic local tree (random-but-deterministic bytes; no real data)
    files = {
        "sub-01/ses-a/audio/sub-01_ses-a_task-x.json": b'{"k": 1}',
        "sub-01/ses-a/audio/sub-01_ses-a_task-x.wav": b"RIFF....fakeaudio...." * 64,
        "sub-02/ses-a/audio/sub-02_ses-a_task-y.json": b'{"k": 2}',
        "dataset_description.json": b'{"Name": "test"}',
    }
    bids = tmp_path / "bids"
    for rel, content in files.items():
        p = bids / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)

    syn = synapse
    fname = f"_tmp_sage_test_{uuid.uuid4().hex[:8]}"
    folder = syn.store(synapseclient.Folder(name=fname, parent=TEST_PROJECT))
    manifest = str(tmp_path / "manifest.tsv")
    synapseutils.generate_sync_manifest(
        syn=syn, directory_path=str(bids), parent_id=folder.id, manifest_path=manifest)
    synapseutils.syncToSynapse(syn=syn, manifestFile=manifest, sendMessages=False)

    config = {"project": TEST_PROJECT, "folder": folder.id, "folder_name": fname}
    try:
        yield config, str(bids), syn
    finally:
        syn.delete(folder.id)


def _relkeys(d, bids):
    return {os.path.relpath(k, bids) for k in d}


def test_view_roundtrip_matches_local(staged):
    config, bids, syn = staged
    sage_files, _ = build_from_view(syn, config, bids)
    local_files, _ = walk_local_folder(bids, get_md5=True)
    # every local file shows up on Sage under the same relative path...
    assert _relkeys(sage_files, bids) == _relkeys(local_files, bids)
    # ...with matching md5 (proves path reconstruction + md5 column line up)
    for k in sage_files:
        assert sage_files[k]["md5"] == local_files[k]["md5"]


def test_detects_missing_and_mismatch(staged):
    config, bids, syn = staged
    sage_files, _ = build_from_view(syn, config, bids)
    # delete one local file -> it should be "on Sage, not local"
    victim = next(k for k in sage_files if k.endswith("task-y.json"))
    os.remove(victim)
    # change another -> md5 should differ
    changed = next(k for k in sage_files if k.endswith("task-x.json"))
    with open(changed, "wb") as f:
        f.write(b'{"k": 999}')

    local_files, _ = walk_local_folder(bids, get_md5=True)
    assert victim not in local_files                       # missing locally
    assert sage_files[changed]["md5"] != local_files[changed]["md5"]  # mismatch
