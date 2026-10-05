"""Fixes from the 2026-10-04 review of feat/date-shift (demographics rows, typed 'None' answers,
removed recordings, checkbox fold vs verdicts, bundle validation, access tiers, dates and zones)."""
import json

import pandas as pd
import pytest

from b2aiprep.prepare.dataset import BIDSDataset


def test_filtering_phenotype_tables_keeps_typed_na_like_answers(tmp_path):
    """Re-reading a table to drop participants must not turn 'None' / 'N/A' answers into blanks."""
    phenotype = tmp_path / "phenotype" / "confounders"
    phenotype.mkdir(parents=True)
    fp = phenotype / "confounders.tsv"
    fp.write_text("participant_id\tph_walking\tnote\n"
                  "p1\tNone\tN/A\n"
                  "p2\tMild\tNA\n"
                  "p3\tNone\tnull\n")
    BIDSDataset._filter_phenotype_to_participants(str(tmp_path / "phenotype"), {"p1", "p2"})
    out = BIDSDataset._read_tsv_as_written(fp)
    assert out.to_dict("list") == {"participant_id": ["p1", "p2"], "ph_walking": ["None", "Mild"],
                                   "note": ["N/A", "NA"]}
