"""RedCap supplements: fields RedCap does not hold yet, added from a CSV shaped like an export."""

import pandas as pd
import pytest

from b2aiprep.prepare.redcap import RedCapDataset


def _dataset():
    df = pd.DataFrame([
        {"record_id": "a", "redcap_repeat_instrument": "Participant", "enrollment_institution": "MIT"},
        {"record_id": "a", "redcap_repeat_instrument": "Session", "session_id": "s1"},
        {"record_id": "b", "redcap_repeat_instrument": "Participant", "enrollment_institution": "USF"},
    ], dtype=object)
    return RedCapDataset(df=df, source_type="redcap")


def test_supplement_fills_the_participant_row_only(tmp_path, caplog):
    path = tmp_path / "remote.csv"
    pd.DataFrame({"record_id": ["a", "b", "zzz"], "some_data_collected_remotely": ["Yes", "No", "Yes"]}).to_csv(path, index=False)
    dataset = _dataset()
    before = list(dataset.df.index)
    with caplog.at_level("WARNING"):
        dataset.add_supplement(path)
    df = dataset.df
    assert list(df.index) == before
    assert df["some_data_collected_remotely"].tolist()[0] == "Yes" and df["some_data_collected_remotely"].tolist()[2] == "No"
    assert pd.isna(df["some_data_collected_remotely"].iloc[1])  # the Session row is untouched
    assert "1 record_id(s) not in the RedCap export" in caplog.text and "zzz" in caplog.text


def test_supplement_refuses_a_column_redcap_already_has(tmp_path):
    path = tmp_path / "clash.csv"
    pd.DataFrame({"record_id": ["a"], "enrollment_institution": ["WCM"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="already has"):
        _dataset().add_supplement(path)


def test_supplement_refuses_duplicate_rows(tmp_path):
    path = tmp_path / "dup.csv"
    pd.DataFrame({"record_id": ["a", "a"], "x": ["Yes", "No"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="more than one row"):
        _dataset().add_supplement(path)


def test_building_without_a_published_supplement_column_warns(tmp_path, caplog):
    """redcap2bids without --supplement must not drop some_data_collected_remotely silently."""
    from b2aiprep.prepare.dataset import BIDSDataset

    df = pd.DataFrame({"record_id": ["r1"], "redcap_repeat_instrument": ["Acoustic Task"],
                       "acoustic_task_name": ["A"]})
    with caplog.at_level("WARNING"):
        BIDSDataset._construct_phenotype_from_reproschema(df, output_dir=str(tmp_path))
    assert any("some_data_collected_remotely" in r.getMessage() and "--supplement" in r.getMessage()
               for r in caplog.records if r.levelname == "WARNING")
