#!/usr/bin/env python
"""Recursively find parquet files under a path and write a TSV of row counts.

Row counts come from the parquet footer metadata, so no data is loaded.

Usage:
    python parquet_row_counts.py <path> [-o output.tsv]
"""
import argparse
import csv
import sys
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", type=Path, help="Directory to search (or a single parquet file)")
    parser.add_argument("-o", "--output", type=Path, default=Path("parquet_row_counts.tsv"),
                        help="Output TSV path (default: ./parquet_row_counts.tsv)")
    args = parser.parse_args()

    root = args.path.resolve()
    files = [root] if root.is_file() else sorted(root.rglob("*.parquet"))

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["path", "name", "n"])
        for p in files:
            rel = p.name if p == root else str(p.relative_to(root))
            try:
                n = pq.ParquetFile(p).metadata.num_rows
            except Exception as e:
                print(f"WARNING: could not read {p}: {e}", file=sys.stderr)
                n = "NA"
            writer.writerow([rel, p.name, n])

    print(f"Wrote {len(files)} rows to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
