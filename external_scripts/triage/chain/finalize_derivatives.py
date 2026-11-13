"""Write the top-level files of a ``derivatives/senselab-triage`` copy: the tables, their dictionary, the viewer, and
``dataset_description.json``.

    python finalize_derivatives.py --vectors VECTORS_DIR --dst DST --bids-root BIDS_ROOT --commit SHA \
        --profile PROFILE_DIR [--description TEXT]

Refuses (exit 2) unless every decision row carries one commit, the one given, and one configuration hash; the hash
and the profile's name and version go into ``GeneratedBy``. Run roots (``sub-*``) are copied by publish.sh, not here.
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from importlib.resources import files
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
import yaml

TABLES = (
    "recording_vectors.parquet",
    "recording_vectors.summary.json",
    "triage_decisions.parquet",
    "triage_decisions.tsv",
    "triage_evidence.parquet",
    "triage.tsv",
    "triage.json",
)
DICTIONARY_KEY = b"senselab.recording_vectors.dictionary"


def main(argv: list[str] | None = None) -> int:
    """Copy the tables and write the description; 2 where the decisions do not hold one commit and configuration."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument(
        "--vectors", type=Path, required=True, help="The run's recording_vectors folder"
    )
    parser.add_argument(
        "--dst", type=Path, required=True, help="The derivatives/senselab-triage folder"
    )
    parser.add_argument("--bids-root", type=Path, required=True, help="The BIDS tree triage read")
    parser.add_argument("--commit", required=True, help="The senselab revision the run used")
    parser.add_argument(
        "--profile", type=Path, required=True, help="The dataset profile the run used"
    )
    parser.add_argument("--description", default="", help="What this run was, for GeneratedBy")
    args = parser.parse_args(argv)

    missing = [name for name in TABLES if not (args.vectors / name).is_file()]
    if missing:
        print(f"ERROR: {args.vectors} lacks {missing}; nothing written", file=sys.stderr)
        return 2
    decisions = pq.read_table(
        args.vectors / "triage_decisions.parquet", columns=["commit", "config_hash"]
    )
    commits = pc.unique(decisions["commit"]).to_pylist()
    hashes = pc.unique(decisions["config_hash"]).to_pylist()
    if commits != [args.commit] or len(hashes) != 1:
        print(
            f"ERROR: decisions carry commits {commits} and configurations {hashes}; expected one of each",
            file=sys.stderr,
        )
        return 2
    manifest = yaml.safe_load((args.profile / "profile.yaml").read_text())

    args.dst.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        shutil.copy2(args.vectors / name, args.dst / name)
    metadata = pq.read_metadata(args.dst / "recording_vectors.parquet").metadata or {}
    if DICTIONARY_KEY in metadata:
        dictionary = json.loads(metadata[DICTIONARY_KEY])
        (args.dst / "recording_vectors.dictionary.json").write_text(
            json.dumps(dictionary, indent=1) + "\n"
        )
    viewer = files("senselab.audio.workflows.triage") / "viewer" / "recording_vectors_viewer.html"
    if viewer.is_file():
        (args.dst / "recording_vectors_viewer.html").write_bytes(viewer.read_bytes())

    description = {
        "Name": f"senselab triage of {args.bids_root.name}",
        "BIDSVersion": "1.9.0",
        "DatasetType": "derivative",
        "GeneratedBy": [
            {
                "Name": "senselab triage",
                "CodeURL": "https://github.com/sensein/senselab",
                "CodeCommit": args.commit,
                "ConfigHash": hashes[0],
                "Profile": f"{manifest.get('name')} {manifest.get('version')}",
                "Date": datetime.date.today().isoformat(),
                "Description": args.description
                or "The b2aiprep triage chain (external_scripts/triage/chain), every stage in its README's order.",
            }
        ],
        "SourceDatasets": [{"URL": f"file://{args.bids_root}", "Version": args.bids_root.name}],
        "Notes": "Contains unredacted streams, stores and provenance; not itself a release.",
    }
    (args.dst / "dataset_description.json").write_text(json.dumps(description, indent=2) + "\n")
    print(f"wrote {len(TABLES)} tables and dataset_description.json to {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
