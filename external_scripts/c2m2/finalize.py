"""
finalize.py — Copy static C2M2 files into the output directory.

Run this before the adult and peds runs to copy the files that don't change
between runs (id_namespace.tsv, project.tsv).
"""

import shutil
import argparse
from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static_files"
STATIC_FILES = ['id_namespace.tsv', 'project.tsv', 'project_in_project.tsv', 'dcc.tsv']


def main():
    parser = argparse.ArgumentParser(
        description='Copy static C2M2 files into the output directory.'
    )
    parser.add_argument('c2m2_path', help='Path to the C2M2 output directory')
    args = parser.parse_args()

    c2m2_path = Path(args.c2m2_path)
    c2m2_path.mkdir(parents=True, exist_ok=True)

    for fname in STATIC_FILES:
        src = STATIC_DIR / fname
        dst = c2m2_path / fname
        shutil.copy(src, dst)
        print(f"Copied {fname}")


if __name__ == '__main__':
    main()
