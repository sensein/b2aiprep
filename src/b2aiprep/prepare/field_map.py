"""Which cohort's rows of the field map (bids_field_organization.csv) a run reads.

The field map has one row per (cohort, table, column): a REDCap field that both the adult and the
pediatric export carry has an adult row and a pediatric row, so the two can differ in table,
description, disposition or access tier. Every command that reads the field map takes the cohort
explicitly (``--cohort``) and selects it here before reading; there is no default.
"""

import typing as t

import pandas as pd

COHORTS: t.Tuple[str, ...] = ("adult", "pediatric")

_selected: t.Optional[str] = None


def select_cohort(cohort: str) -> None:
    """Read the field map's *cohort* rows from now on (``adult`` or ``pediatric``)."""
    global _selected
    if cohort not in COHORTS:
        raise ValueError(f"cohort must be one of {COHORTS}; got {cohort!r}")
    _selected = cohort


def selected_cohort() -> str:
    """The cohort chosen with ``select_cohort``; raises if none was chosen."""
    if _selected is None:
        raise RuntimeError(
            "No cohort selected for the field map: pass --cohort (adult or pediatric), or call "
            "b2aiprep.prepare.field_map.select_cohort() before reading it."
        )
    return _selected


def rows_for_cohort(field_map: pd.DataFrame, cohort: t.Optional[str] = None) -> pd.DataFrame:
    """The rows of *field_map* for *cohort* (the selected cohort when not given).

    Raises when *cohort*, or any value of the ``cohort`` column, is not ``adult``/``pediatric``: an
    unknown cohort would match no row, and a blank or misspelled row would silently drop out of
    every build.
    """
    if cohort is None:
        cohort = selected_cohort()
    if cohort not in COHORTS:
        raise ValueError(f"cohort must be one of {COHORTS}; got {cohort!r}")
    if "cohort" not in field_map.columns:
        raise ValueError("bids_field_organization.csv has no 'cohort' column.")
    values = field_map["cohort"].fillna("").astype(str).str.strip()
    invalid = sorted(set(values) - set(COHORTS))
    if invalid:
        raise ValueError(f"bids_field_organization.csv: cohort must be one of {COHORTS}; got {invalid}")
    return field_map.loc[values == cohort]
