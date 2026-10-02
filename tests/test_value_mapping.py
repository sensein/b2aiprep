"""Deidentify value mapping (value_mapping.json) and reading phenotype tables as written."""

import json

import pandas as pd
import pytest

from b2aiprep.prepare.dataset import BIDSDataset

SPEC = {"participant": {"enrollment_institution": {
    "output_column": "site", "values": {"MIT": "site_A", "USF": "site_B", "WCM": "site_C"}}}}


def _participants(sites):
    return pd.DataFrame({"participant_id": [f"p{i}" for i in range(len(sites))],
                         "enrollment_institution": sites, "age": ["40"] * len(sites)})


def test_site_is_written_next_to_its_source_with_its_labels_as_choices():
    phenotype = {"enrollment_institution": {"description": "Enrollment Institution"}}
    df, phenotype = BIDSDataset._apply_value_mappings(
        _participants(["MIT", "USF", None]), phenotype, "participant", SPEC)
    assert list(df.columns) == ["participant_id", "enrollment_institution", "site", "age"]
    assert df["site"].tolist()[:2] == ["site_A", "site_B"] and pd.isna(df["site"].iloc[2])
    assert df["enrollment_institution"].tolist()[:2] == ["MIT", "USF"]  # left for the disposition step
    element = phenotype["site"]
    assert [c["value"] for c in element["choices"]] == ["site_A", "site_B"]
    assert "arbitrary label" in element["description"] and "termURL" not in element


def test_a_value_with_no_mapping_stops_the_run():
    with pytest.raises(ValueError, match="no mapping"):
        BIDSDataset._apply_value_mappings(_participants(["MIT", "Elsewhere"]), {}, "participant", SPEC)


def test_rare_labels_become_other_when_a_minimum_is_set():
    spec = json.loads(json.dumps(SPEC))
    spec["participant"]["enrollment_institution"].update({"min_participants": 2, "other": "other"})
    df, phenotype = BIDSDataset._apply_value_mappings(
        _participants(["MIT", "MIT", "USF"]), {}, "participant", spec)
    assert df["site"].tolist() == ["site_A", "site_A", "other"]
    assert [c["value"] for c in phenotype["site"]["choices"]] == ["other", "site_A"]


def test_other_tables_and_missing_config_are_untouched(tmp_path):
    df = _participants(["MIT"])
    out, _ = BIDSDataset._apply_value_mappings(df, {}, "demographics", SPEC)
    assert out.equals(df)
    assert BIDSDataset._load_value_mappings(tmp_path) == {}


def test_phenotype_tables_are_read_as_written(tmp_path):
    """Integers with blanks stay integers ("10", not "10.0"); typed "N/A" and "None" are kept."""
    base = tmp_path / "recording"
    pd.DataFrame({"participant_id": ["005009", "005010", "005011"], "recording_local_hour": ["10", "", "7"],
                  "comment": ["N/A", "None", ""]}).to_csv(base.with_suffix(".tsv"), sep="\t", index=False)
    base.with_suffix(".json").write_text(json.dumps({"recording": {"description": "", "data_elements": {}}}))
    df, *_ = BIDSDataset.load_phenotype_file(base)
    assert df["participant_id"].tolist() == ["005009", "005010", "005011"]
    assert df["recording_local_hour"].tolist()[0::2] == ["10", "7"] and pd.isna(df["recording_local_hour"].iloc[1])
    assert df["comment"].tolist()[:2] == ["N/A", "None"] and pd.isna(df["comment"].iloc[2])
