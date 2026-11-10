"""Cohort selection for the field map (b2aiprep.prepare.field_map)."""

import pandas as pd
import pytest

from b2aiprep.prepare import field_map
from b2aiprep.prepare.dataset import BIDSDataset
from b2aiprep.prepare.redcap import _dropped_source_columns


def test_reading_the_field_map_without_a_cohort_raises(monkeypatch):
    monkeypatch.setattr(field_map, "_selected", None)
    with pytest.raises(RuntimeError, match="No cohort selected"):
        BIDSDataset._load_reorganization_file(exclude_dropped=False)
    with pytest.raises(RuntimeError, match="No cohort selected"):
        _dropped_source_columns()


def test_select_cohort_refuses_anything_but_adult_or_pediatric():
    with pytest.raises(ValueError, match="cohort must be one of"):
        field_map.select_cohort("peds")


@pytest.mark.parametrize("bad", ["", "peds"])
def test_rows_for_cohort_refuses_an_unknown_cohort_argument(bad):
    rows = pd.DataFrame({"cohort": ["adult"], "column_name": ["a"]})
    with pytest.raises(ValueError, match="cohort must be one of"):
        field_map.rows_for_cohort(rows, bad)


@pytest.mark.parametrize("bad", ["", "Adult", "child"])
def test_rows_for_cohort_refuses_a_blank_or_misspelled_cohort(bad):
    rows = pd.DataFrame({"cohort": ["adult", bad], "column_name": ["a", "b"]})
    with pytest.raises(ValueError, match="cohort must be one of"):
        field_map.rows_for_cohort(rows, "adult")


def test_each_cohort_reads_only_its_rows(monkeypatch):
    for cohort in field_map.COHORTS:
        monkeypatch.setattr(field_map, "_selected", cohort)
        rows = BIDSDataset._field_map()
        assert set(rows["cohort"]) == {cohort}
    # Tables only one cohort has.
    monkeypatch.setattr(field_map, "_selected", "adult")
    assert "pediatric_demographics" not in set(BIDSDataset._field_map()["schema_name"])
    monkeypatch.setattr(field_map, "_selected", "pediatric")
    assert "demographics" not in set(BIDSDataset._field_map()["schema_name"])


def test_deidentify_refuses_a_tree_built_for_the_other_cohort_and_writes_nothing(tmp_path):
    bids = tmp_path / "bids"
    (bids / "phenotype").mkdir(parents=True)
    (bids / "dataset_description.json").write_text('{"Name": "x"}')
    BIDSDataset._record_cohort(bids, "pediatric")
    with pytest.raises(ValueError, match="built for cohort 'pediatric'"):
        BIDSDataset(bids).deidentify(outdir=tmp_path / "out", deidentify_config_dir=tmp_path, cohort="adult")
    assert not (tmp_path / "out").exists()


def test_deidentify_checks_the_tables_before_creating_the_output(tmp_path):
    """A pediatric table under the adult field map stops the run before anything is written."""
    bids = tmp_path / "bids"
    (bids / "phenotype" / "pediatric").mkdir(parents=True)
    table = bids / "phenotype" / "pediatric" / "pediatric_demographics"
    pd.DataFrame({"participant_id": ["p1"]}).to_csv(table.with_suffix(".tsv"), sep="\t", index=False)
    table.with_suffix(".json").write_text('{"pediatric_demographics": {"data_elements": {}}}')
    with pytest.raises(ValueError, match="not in the field map"):
        BIDSDataset(bids).deidentify(outdir=tmp_path / "out", deidentify_config_dir=tmp_path, cohort="adult")
    assert not (tmp_path / "out").exists()
