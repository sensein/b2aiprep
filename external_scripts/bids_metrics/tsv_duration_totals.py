#!/usr/bin/env python3
"""Write total extend_time and recording hours for a TSV to a file.

extend_time     -> sum of task_duration_s / 3600
recording hours -> sum of recording_duration_s / 3600

Usage: python tsv_duration_totals.py <path.tsv> [-o OUTPUT]

If -o is not given, writes <input_name>_duration_totals.tsv
next to the input file.
"""
import argparse
import csv
import os
import sys

FIELDS = {
    "extend_time": "task_duration_s",
    "recording_hours": "recording_duration_s",
}


def main(path, out_path=None):
    if out_path is None:
        out_path = os.path.splitext(path)[0] + "_duration_totals.tsv"

    totals = {name: 0.0 for name in FIELDS}
    skipped = {name: 0 for name in FIELDS}

    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        missing = [col for col in FIELDS.values() if col not in reader.fieldnames]
        if missing:
            sys.exit(f"Missing column(s) in {path}: {', '.join(missing)}")

        for row in reader:
            for name, col in FIELDS.items():
                try:
                    totals[name] += float(row[col])
                except (TypeError, ValueError):
                    skipped[name] += 1

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["metric", "seconds", "hours", "source_column", "rows_skipped"])
        for name, col in FIELDS.items():
            writer.writerow([name, f"{totals[name]:.2f}", f"{totals[name] / 3600:.2f}", col, skipped[name]])

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Total extend_time and recording hours for a TSV.")
    parser.add_argument("tsv", help="input TSV path")
    parser.add_argument("-o", "--output", help="output TSV path (default: <input_name>_duration_totals.tsv next to the input)")
    args = parser.parse_args()
    main(args.tsv, args.output)
