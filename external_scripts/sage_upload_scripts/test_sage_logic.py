"""Pure-logic tests for the Sage upload/verify scripts -- no Synapse required.

Run from this directory:
    pytest test_sage_logic.py -q
or:
    python -m pytest external_scripts/sage_upload_scripts/test_sage_logic.py -q
"""
import os
import sys
import hashlib

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sage_config as cfg
from sage_upload_manifest import filter_manifest, resolve_range
from verify_sage_contents import reconstruct_relpath, _has_md5


# ---- filter_manifest / subject_segment (fixes #6/#7: segment-accurate matching) ----

PATHS = pd.DataFrame({"path": [
    "/data/bids/sub-01/ses-a/audio/sub-01_ses-a_task-x.wav",   # sub-01
    "/data/bids/sub-01/sessions.tsv",                          # sub-01
    "/data/bids/sub-010/ses-b/audio/sub-010_ses-b_task-y.wav", # sub-010 (prefix collision)
    "sub-01/leading/seg.wav",                                  # sub-01 as leading segment
    "/data/bids/dataset_description.json",                     # top-level
    "/data/bids/phenotype/diagnosis/anxiety.json",             # top-level (no sub-)
]})


def test_subject_filter_is_segment_exact_not_prefix():
    got = set(filter_manifest(PATHS, subject="sub-01")["path"])
    assert got == {
        "/data/bids/sub-01/ses-a/audio/sub-01_ses-a_task-x.wav",
        "/data/bids/sub-01/sessions.tsv",
        "sub-01/leading/seg.wav",
    }
    # the prefix sibling must NOT be swept in
    assert "/data/bids/sub-010/ses-b/audio/sub-010_ses-b_task-y.wav" not in got


def test_subject_filter_matches_sibling_exactly():
    got = set(filter_manifest(PATHS, subject="sub-010")["path"])
    assert got == {"/data/bids/sub-010/ses-b/audio/sub-010_ses-b_task-y.wav"}


def test_toplevel_filter_excludes_all_subjects():
    got = set(filter_manifest(PATHS, toplevel=True)["path"])
    assert got == {
        "/data/bids/dataset_description.json",
        "/data/bids/phenotype/diagnosis/anxiety.json",
    }


def test_unfiltered_returns_all():
    assert len(filter_manifest(PATHS)) == len(PATHS)


# ---- resolve_range (fix #5: honor 0, reject invalid) ----

def test_resolve_range_defaults():
    assert resolve_range(None, None, 10) == (0, 10)


def test_resolve_range_honors_explicit_zero_end():
    # a deliberate --end 0 means "upload nothing", not "everything"
    assert resolve_range(None, 0, 10) == (0, 0)
    assert resolve_range(0, 0, 10) == (0, 0)


def test_resolve_range_normal():
    assert resolve_range(2, 5, 10) == (2, 5)


@pytest.mark.parametrize("start,end", [(-1, 5), (0, -1), (5, 2)])
def test_resolve_range_rejects_invalid(start, end):
    with pytest.raises(ValueError):
        resolve_range(start, end, 10)


# ---- _has_md5 (fix #4: NaN/None server md5 is not a "mismatch") ----

@pytest.mark.parametrize("value,expected", [
    ("d41d8cd98f00b204e9800998ecf8427e", True),
    ("", False),
    (None, False),
    (float("nan"), False),
    (12345, False),
])
def test_has_md5(value, expected):
    assert _has_md5(value) is expected


# ---- file_md5 (shared helper) ----

def test_file_md5_matches_hashlib(tmp_path):
    p = tmp_path / "f.bin"
    data = b"hello sage" * 100000
    p.write_bytes(data)
    assert cfg.file_md5(str(p)) == hashlib.md5(data).hexdigest()


# ---- reconstruct_relpath (fix #1: resolve fully or return None, never truncate) ----

# folder map: id -> (name, parentId); root is "R"
FMAP = {
    "A": ("sub-01", "R"),
    "B": ("ses-a", "A"),
    "C": ("audio", "B"),
}


def test_reconstruct_nested():
    assert reconstruct_relpath("f.wav", "C", FMAP, "R") == os.path.join("sub-01", "ses-a", "audio", "f.wav")


def test_reconstruct_top_level_parent_is_root():
    assert reconstruct_relpath("dataset_description.json", "R", FMAP, "R") == "dataset_description.json"


def test_reconstruct_missing_folder_returns_none():
    # parent chain references a folder not in the view -> unresolved, not truncated
    assert reconstruct_relpath("f.wav", "MISSING", FMAP, "R") is None


def test_reconstruct_nan_parent_returns_none():
    assert reconstruct_relpath("f.wav", float("nan"), FMAP, "R") is None


def test_reconstruct_none_parent_returns_none():
    assert reconstruct_relpath("f.wav", None, FMAP, "R") is None


def test_reconstruct_cycle_returns_none():
    cyc = {"X": ("x", "Y"), "Y": ("y", "X")}
    assert reconstruct_relpath("f.wav", "X", cyc, "R") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-q"]))
