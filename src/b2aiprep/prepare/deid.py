"""Rule-based PHI detection for deidentification."""

from __future__ import annotations

import csv
import logging
import typing as t
from functools import lru_cache
from importlib.resources import files

_LOGGER = logging.getLogger(__name__)

_MANIFEST_RESOURCE = "field_metadata.csv"

# Field types that encode a bounded set of coded responses.
CATEGORICAL_FIELD_TYPES = frozenset(
    {"radio", "checkbox", "dropdown", "yesno", "slider"}
)

# Text-validation types that make a text field a structured number.
NUMERIC_VALIDATIONS = frozenset({"number", "integer"})

# Text-validation types that indicate an absolute date/time.
DATE_VALIDATIONS = frozenset(
    {
        "date_mdy",
        "date_ymd",
        "date_dmy",
        "datetime_mdy",
        "datetime_ymd",
        "datetime_dmy",
        "datetime_seconds_mdy",
        "datetime_seconds_ymd",
        "datetime_seconds_dmy",
        "time",
    }
)

# Text-validation types that are direct contact identifiers.
CONTACT_VALIDATIONS = frozenset({"email", "phone", "signature"})

# Absolute-timestamp suffixes are always dropped regardless of the manifest.
_TIMESTAMP_SUFFIXES = ("_started_at", "_completed_at")

# Columns that are structural keys or explicitly-retained non-PHI values and must
# never be removed by this backstop.
_KEEP_EXACT = frozenset({"participant_id", "record_id", "age"})
_KEEP_SUFFIXES = ("_session_id", "_duration")


@lru_cache(maxsize=1)
def load_field_metadata() -> t.Dict[str, t.Dict[str, str]]:
    """Load the distilled REDCap field metadata manifest.

    Returns:
        Mapping of ``variable`` -> ``{"field_type", "validation", "identifier"}``.
    """
    resource = files("b2aiprep.prepare.resources").joinpath(_MANIFEST_RESOURCE)
    metadata: t.Dict[str, t.Dict[str, str]] = {}
    with resource.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            variable = (row.get("variable") or "").strip()
            if not variable:
                continue
            metadata[variable] = {
                "field_type": (row.get("field_type") or "").strip().lower(),
                "validation": (row.get("validation") or "").strip().lower(),
                "identifier": (row.get("identifier") or "").strip().lower(),
            }
    return metadata


def phi_reason(
    column: str, metadata: t.Optional[t.Dict[str, t.Dict[str, str]]] = None
) -> t.Optional[str]:
    """Return a reason string if ``column`` is PHI and should be removed, else None.

      1. Structural keys / retained values (ids, durations, age) are always kept.
      2. Categorical fields (radio/checkbox/...) are always kept.
      3. Columns with suffixes that imply a timestamp are always dropped.
      4. Columns absent from the manifest are kept, except for timestamp suffixes.
      5. Using the RedCap CSV, we drop columns with:
        a) the identifier flag
        b) date/time validation
        c) contact validation
        d) free-text field types (notes, text with non-numeric validation)
    """
    if metadata is None:
        metadata = load_field_metadata()

    if column in _KEEP_EXACT or column.endswith(_KEEP_SUFFIXES):
        return None

    meta = metadata.get(column)
    field_type = meta["field_type"] if meta else ""

    if field_type in CATEGORICAL_FIELD_TYPES:
        return None

    if column.endswith(_TIMESTAMP_SUFFIXES):
        return "timestamp"

    if meta is None:
        return None

    if meta["identifier"] == "y":
        return "identifier"

    validation = meta["validation"]
    if validation in DATE_VALIDATIONS:
        return "date"
    if validation in CONTACT_VALIDATIONS:
        return "contact"

    if field_type == "notes":
        return "free-text"
    if field_type == "text" and validation not in NUMERIC_VALIDATIONS:
        return "free-text"

    return None


def classify_phi_columns(columns: t.Iterable[str]) -> t.Dict[str, str]:
    """Return {column: reason} for every column classified as PHI."""
    metadata = load_field_metadata()
    return {
        column: reason
        for column in columns
        if (reason := phi_reason(column, metadata)) is not None
    }
