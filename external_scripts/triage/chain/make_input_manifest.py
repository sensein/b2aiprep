"""Write the recordings of a BIDS tree as the triage chain's input: one ``{"source": path}`` per line.

    python make_input_manifest.py BIDS_ROOT --out INPUT.jsonl [--stems STEMS.txt]

Every ``sub-*/ses-*/audio/*.wav`` or ``*.flac`` is listed, sorted by path. ``--stems`` keeps only the recordings whose
file stem is listed (one per line), and stops if a listed stem is not in the tree. Two recordings with one stem stop
the run: the triage runs and rows are named by stem.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AUDIO_SUFFIXES = (".wav", ".flac")


def recordings(bids_root: Path) -> list[Path]:
    """Every recording under the tree's ``sub-*/ses-*/audio/`` folders, sorted by path."""
    found = [
        path
        for path in bids_root.glob("sub-*/ses-*/audio/*")
        if path.suffix.lower() in AUDIO_SUFFIXES
        and path.is_file()
        and not path.name.startswith(".")
    ]
    return sorted(found)


def main(argv: list[str] | None = None) -> int:
    """Write the input manifest; 2 where the tree or the stem list does not hold what is asked."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", maxsplit=1)[0])
    parser.add_argument("bids_root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--stems", type=Path, default=None, help="Keep only these file stems, one per line"
    )
    args = parser.parse_args(argv)

    paths = recordings(args.bids_root)
    by_stem: dict[str, Path] = {}
    for path in paths:
        if path.stem in by_stem:
            print(
                f"ERROR: two recordings have the stem {path.stem}: {by_stem[path.stem]} and {path}",
                file=sys.stderr,
            )
            return 2
        by_stem[path.stem] = path
    if args.stems is not None:
        wanted = [line.strip() for line in args.stems.read_text().splitlines() if line.strip()]
        missing = [stem for stem in wanted if stem not in by_stem]
        if missing:
            print(
                f"ERROR: {len(missing)} listed stems are not in {args.bids_root}, e.g. {missing[:3]}",
                file=sys.stderr,
            )
            return 2
        paths = sorted(by_stem[stem] for stem in set(wanted))
    if not paths:
        print(f"ERROR: no recordings under {args.bids_root}/sub-*/ses-*/audio/", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + ".partial")
    partial.write_text("".join(json.dumps({"source": str(path)}) + "\n" for path in paths))
    partial.replace(args.out)
    print(f"{len(paths)} recordings -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
