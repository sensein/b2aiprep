#!/usr/bin/env python3
"""Generate field_metadata.csv from the externally-sourced REDCap data dictionary.

The REDCap data dictionary (bridge2ai_voice_redcap_project_data_dictionary.csv,
maintained in the bridge2ai-redcap repo) is the authoritative source of truth for
field names *and* their PHI-relevant properties: field type, text-validation type,
and the "Identifier?" flag.

This script distills that dictionary into a compact manifest shipped inside
b2aiprep (src/b2aiprep/prepare/resources/field_metadata.csv). The manifest is used
at deidentification time to decide, by rule, which columns are free-text / dates /
identifiers and must be removed (see b2aiprep.prepare.dataset.BIDSDataset).

Re-run this whenever the external data dictionary is updated so that the
deidentification rules stay in sync with the source schema.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT / "src" / "b2aiprep" / "prepare" / "resources" / "field_metadata.csv"
)

# Columns copied verbatim from the REDCap data dictionary header.
_VARIABLE = "Variable / Field Name"
_FIELD_TYPE = "Field Type"
_VALIDATION = "Text Validation Type OR Show Slider Number"
_IDENTIFIER = "Identifier?"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Distill the REDCap data dictionary into resources/field_metadata.csv "
            "used by deidentification."
        )
    )
    parser.add_argument(
        "data_dictionary",
        type=Path,
        help=(
            "Path to bridge2ai_voice_redcap_project_data_dictionary.csv "
            "(from the bridge2ai-redcap repo)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination path for field_metadata.csv.",
    )
    return parser.parse_args()


def build_rows(dictionary_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    # utf-8-sig strips the BOM that REDCap exports include.
    with dictionary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            variable = (row.get(_VARIABLE) or "").strip()
            if not variable or variable in seen:
                continue
            seen.add(variable)
            rows.append(
                {
                    "variable": variable,
                    "field_type": (row.get(_FIELD_TYPE) or "").strip().lower(),
                    "validation": (row.get(_VALIDATION) or "").strip().lower(),
                    "identifier": (row.get(_IDENTIFIER) or "").strip().lower(),
                }
            )
    rows.sort(key=lambda r: r["variable"])
    return rows


def main() -> None:
    args = parse_args()
    rows = build_rows(args.data_dictionary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["variable", "field_type", "validation", "identifier"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} field definitions to {args.output}")


if __name__ == "__main__":
    main()
