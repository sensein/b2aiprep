"""Crosswalk of released session labels between two deidentified releases.

Input: the internal session maps that ``deidentify-bids-dataset --session-id-map`` writes for each
release (they hold original session IDs and are never released). Output: a TSV of released
participant ID, old session label and new session label for every session released in both, which
is safe to publish alongside the new release.

For a release made before session maps existed whose labels were the truncated session IDs (v3.1),
pass ``--old-uuid-labels`` instead of ``--old``: its labels are recomputed from the original IDs in
the new map, so only sessions released in the new release can be matched.

    python scripts/session_label_crosswalk.py --old v4.0_map.json --new v4.1_map.json -o crosswalk.tsv
    python scripts/session_label_crosswalk.py --old-uuid-labels --new v4.0_map.json -o crosswalk.tsv
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from b2aiprep.prepare.dataset import BIDSDataset, SessionLabels


def load_map(path: Path) -> pd.DataFrame:
    return pd.DataFrame(json.loads(Path(path).read_text())["sessions"], dtype=str)


def uuid_labels(new: pd.DataFrame) -> pd.DataFrame:
    """The v3.1 labels of the sessions in *new*, per participant as deidentify computes them."""
    rows = []
    for pid, group in new.groupby("participant_id"):
        labels, _ = BIDSDataset._session_labels(group[["session_id"]], SessionLabels.UUID)
        rows += [
            {"participant_id": pid, "released_participant_id": r.released_participant_id,
             "session_id": r.session_id, "released_session_label": labels[r.session_id]}
            for r in group.itertuples()
        ]
    return pd.DataFrame(rows, dtype=str)


def crosswalk(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    merged = old.merge(new, on=["participant_id", "session_id"], suffixes=("_old", "_new"))
    out = pd.DataFrame({
        "participant_id": merged["released_participant_id_new"],
        "old_participant_id": merged["released_participant_id_old"],
        "old_session_label": merged["released_session_label_old"],
        "new_session_label": merged["released_session_label_new"],
    })
    if (out["participant_id"] == out["old_participant_id"]).all():
        out = out.drop(columns="old_participant_id")
    return out.sort_values(["participant_id", "new_session_label"]).reset_index(drop=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    old = parser.add_mutually_exclusive_group(required=True)
    old.add_argument("--old", type=Path, help="Session map of the earlier release.")
    old.add_argument("--old-uuid-labels", action="store_true",
                     help="The earlier release used truncated-session-ID labels (v3.1).")
    parser.add_argument("--new", type=Path, required=True, help="Session map of the new release.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Crosswalk TSV to write.")
    args = parser.parse_args(argv)

    new = load_map(args.new)
    table = crosswalk(uuid_labels(new) if args.old_uuid_labels else load_map(args.old), new)
    table.to_csv(args.output, sep="\t", index=False)
    changed = int((table["old_session_label"] != table["new_session_label"]).sum())
    print(f"{len(table)} sessions in both releases; {changed} with a new label -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
