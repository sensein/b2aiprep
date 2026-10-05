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


def _write_table(folder, name, df):
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / f"{name}.tsv", sep="\t", index=False)
    (folder / f"{name}.json").write_text(json.dumps(
        {name: {"description": name, "data_elements": {c: {"description": c} for c in df.columns}}}))


def _tree_with_two_recordings(tmp_path, removal):
    """p1 has recordings r1 (task t1) and r2 (task t2) in session s1; *removal* is the config file."""
    bids, config = tmp_path / "bids", tmp_path / "config"
    config.mkdir(parents=True)
    (config / "participants_to_include.json").write_text(json.dumps(["p1"]))
    (config / "id_remapping.json").write_text(json.dumps({"p1": "900001"}))
    (config / "deidentify_settings.json").write_text(json.dumps({"access_tier": "registered"}))
    (config / "audio_tasks_to_include.json").write_text(json.dumps(["rainbow-passage", "maximum-phonation-time-1"]))
    (config / "audio_filestems_to_remove.json").write_text(json.dumps(removal.get("filestems", [])))
    if "recording_ids" in removal:
        (config / "audio_recording_ids_to_remove.json").write_text(json.dumps(removal["recording_ids"]))
    audio = bids / "sub-p1" / "ses-s1" / "audio"
    audio.mkdir(parents=True)
    for rid, task in (("r1", "rainbow-passage"), ("r2", "maximum-phonation-time-1")):
        stem = f"sub-p1_ses-s1_task-{task}"
        (audio / f"{stem}.wav").write_bytes(b"RIFF" + b"\x00" * 8192)
        (audio / f"{stem}_recording-metadata.json").write_text(
            json.dumps({"record_id": "p1", "session_id": "s1", "recording_id": rid}))
    pd.DataFrame({"record_id": ["p1"], "session_id": ["s1"], "session_index": ["1"]}).to_csv(
        bids / "sub-p1" / "sessions.tsv", sep="\t", index=False)
    task = bids / "phenotype" / "task"
    _write_table(task, "recording", pd.DataFrame({
        "participant_id": ["p1", "p1"], "recording_id": ["r1", "R2"],
        "recording_acoustic_task_id": ["t1", "t2"], "recording_name": ["a", "b"]}))
    _write_table(task, "acoustic_task", pd.DataFrame({
        "participant_id": ["p1", "p1"], "acoustic_task_id": ["t1", "t2"],
        "acoustic_task_name": ["Rainbow Passage", "Maximum phonation time-1"]}))
    (bids / "dataset_description.json").write_text(json.dumps({"Name": "test"}))
    return bids, config


@pytest.mark.parametrize("removal", [
    {"recording_ids": ["r2"]},
    {"filestems": ["sub-p1_ses-s1_task-maximum-phonation-time-1"]},
], ids=["by-recording-id", "by-filestem"])
def test_removed_recordings_leave_recording_and_acoustic_task_tables(tmp_path, removal):
    bids, config = _tree_with_two_recordings(tmp_path, removal)
    out = tmp_path / "out"
    BIDSDataset(bids).deidentify(outdir=out, deidentify_config_dir=config)
    rec = BIDSDataset._read_tsv_as_written(out / "phenotype" / "task" / "recording.tsv")
    at = BIDSDataset._read_tsv_as_written(out / "phenotype" / "task" / "acoustic_task.tsv")
    assert list(rec["recording_id"]) == ["r1"]
    assert list(at["acoustic_task_id"]) == ["t1"]
    assert len(list(out.rglob("*.wav"))) == 1


def _tree(tmp_path, participants, tables, config_files=()):
    """A tree where each participant has one rainbow-passage recording in session s1.

    *tables* maps "group/name" to a DataFrame; *config_files* are extra (name, object) pairs.
    """
    bids, config = tmp_path / "bids", tmp_path / "config"
    config.mkdir(parents=True)
    (config / "participants_to_include.json").write_text(json.dumps(participants))
    (config / "id_remapping.json").write_text(json.dumps({p: f"90000{i}" for i, p in enumerate(participants)}))
    (config / "deidentify_settings.json").write_text(json.dumps({"access_tier": "registered"}))
    (config / "audio_tasks_to_include.json").write_text(json.dumps(["rainbow-passage"]))
    (config / "audio_filestems_to_remove.json").write_text(json.dumps([]))
    for name, obj in config_files:
        (config / name).write_text(json.dumps(obj))
    for p in participants:
        audio = bids / f"sub-{p}" / "ses-s1" / "audio"
        audio.mkdir(parents=True)
        stem = f"sub-{p}_ses-s1_task-rainbow-passage"
        (audio / f"{stem}.wav").write_bytes(b"RIFF" + b"\x00" * 8192)
        (audio / f"{stem}_recording-metadata.json").write_text(json.dumps({"record_id": p, "session_id": "s1"}))
        pd.DataFrame({"record_id": [p], "session_id": ["s1"], "session_index": ["1"]}).to_csv(
            bids / f"sub-{p}" / "sessions.tsv", sep="\t", index=False)
    (bids / "phenotype").mkdir(exist_ok=True)
    for path, df in tables.items():
        group, name = path.split("/")
        _write_table(bids / "phenotype" / group, name, df)
    (bids / "dataset_description.json").write_text(json.dumps({"Name": "test"}))
    return bids, config


def test_rare_checkbox_label_is_not_published_without_a_review_verdict(tmp_path):
    """The fold runs before the verdicts: a label folded into other_voice_activity for a participant
    with no verdict is blanked with the rest of that cell, not published unreviewed."""
    confounders = pd.DataFrame({
        "participant_id": ["p1", "p2"],
        "voice_activity_v2___attorney": ["1", ""],
        "voice_activity_v2___other": ["", "1"],
        "other_voice_activity": ["", "Podcaster"],
    })
    bids, config = _tree(tmp_path, ["p1", "p2"], {"confounders/confounders": confounders}, [
        ("column_value_reviews.json",
         {"verdicts": [{"participant_id": "p2", "column_name": "other_voice_activity", "verdict": "safe"}]}),
        ("deidentify_settings.json", {"access_tier": "registered", "small_checkbox_options": {"confounders": {
            "voice_activity_v2": {"other": "voice_activity_v2___other", "specify": "other_voice_activity",
                                  "min_participants": 10}}}}),
    ])
    out = tmp_path / "out"
    BIDSDataset(bids).deidentify(outdir=out, deidentify_config_dir=config)
    df = BIDSDataset._read_tsv_as_written(out / "phenotype" / "confounders" / "confounders.tsv")
    by_id = df.set_index("participant_id")
    assert "voice_activity_v2___attorney" not in df.columns
    assert by_id.loc["900000", "voice_activity_v2___other"] == "1"
    assert pd.isna(by_id.loc["900000", "other_voice_activity"])  # no verdict: no label
    assert by_id.loc["900001", "other_voice_activity"] == "Podcaster"


def _bundle_with_confounders(tmp_path, columns, verdicts=None):
    from click.testing import CliRunner
    from b2aiprep.commands import validate_bundled_dataset
    from test_review_fixes_20260930 import _bundle, _config

    _bundle(tmp_path / "bundle", ["session_status"])
    folder = tmp_path / "bundle" / "phenotype" / "confounders"
    folder.mkdir(parents=True)
    pd.DataFrame({"participant_id": ["005009"], **{c: ["x"] for c in columns}}).to_csv(
        folder / "confounders.tsv", sep="\t", index=False)
    config = _config(tmp_path / "cfg")
    if verdicts is not None:
        (config / "column_value_reviews.json").write_text(json.dumps({"verdicts": verdicts}))
    return CliRunner().invoke(validate_bundled_dataset, [str(tmp_path / "bundle"), str(config)])


def test_bundle_validation_fails_on_an_unreviewed_review_column(tmp_path):
    result = _bundle_with_confounders(tmp_path, ["other_voice_activity"])
    assert result.exit_code != 0 and "other_voice_activity" in result.output


def test_bundle_validation_passes_a_review_column_with_verdicts(tmp_path):
    result = _bundle_with_confounders(tmp_path, ["other_voice_activity"], [
        {"participant_id": "p1", "column_name": "other_voice_activity", "verdict": "safe"}])
    assert result.exit_code == 0, result.output


def test_bundle_validation_fails_on_a_record_id_column(tmp_path):
    result = _bundle_with_confounders(tmp_path, ["record_id"])
    assert result.exit_code != 0 and "record_id" in result.output


def _three_session_tree(tmp_path, verdicts):
    """p1: audio in s1 and s3; s2 has only a review-column answer (other_voice_activity)."""
    confounders = pd.DataFrame({"participant_id": ["p1"], "confounders_session_id": ["s2"],
                                "other_voice_activity": ["Podcaster"]})
    extra = [("column_value_reviews.json", {"verdicts": verdicts})] if verdicts else []
    bids, config = _tree(tmp_path, ["p1"], {"confounders/confounders": confounders}, extra)
    audio = bids / "sub-p1" / "ses-s3" / "audio"
    audio.mkdir(parents=True)
    stem = "sub-p1_ses-s3_task-rainbow-passage"
    (audio / f"{stem}.wav").write_bytes(b"RIFF" + b"\x00" * 8192)
    (audio / f"{stem}_recording-metadata.json").write_text(json.dumps({"record_id": "p1", "session_id": "s3"}))
    pd.DataFrame({"record_id": ["p1"] * 3, "session_id": ["s1", "s2", "s3"],
                  "session_index": ["1", "2", "3"]}).to_csv(bids / "sub-p1" / "sessions.tsv", sep="\t", index=False)
    return bids, config


def test_session_labels_do_not_depend_on_which_columns_a_run_publishes(tmp_path):
    """A session holding only review (or controlled-only) data must not shift later labels."""
    labels = {}
    for name, verdicts in (("unreviewed", None),
                           ("reviewed", [{"participant_id": "p1", "column_name": "other_voice_activity",
                                          "verdict": "safe"}])):
        bids, config = _three_session_tree(tmp_path / name, verdicts)
        out = tmp_path / name / "out"
        BIDSDataset(bids).deidentify(outdir=out, deidentify_config_dir=config)
        (sessions,) = list((out / "sub-900000").glob("*sessions.tsv"))
        labels[name] = list(BIDSDataset._read_tsv_as_written(sessions)["session_id"])
    assert labels["unreviewed"] == ["01", "03"]  # s2 withheld here, but s3 keeps its label
    assert labels["reviewed"] == ["01", "02", "03"]


def test_field_map_refuses_an_access_tier_typo(monkeypatch, tmp_path):
    from b2aiprep.prepare import dataset as dataset_module

    real = dataset_module.files("b2aiprep.prepare.resources").joinpath("bids_field_organization.csv")
    df = pd.read_csv(real, dtype={"access_tier": str})
    assert BIDSDataset._load_reorganization_file(exclude_dropped=False) is not None  # the shipped map is valid
    df.loc[df.index[0], "access_tier"] = "controled"
    (tmp_path / "bids_field_organization.csv").write_text(df.to_csv(index=False))
    monkeypatch.setattr(dataset_module, "files", lambda _: tmp_path)
    with pytest.raises(ValueError, match="access_tier must be blank or 'controlled'.*controled"):
        BIDSDataset._load_reorganization_file(exclude_dropped=False)


def test_bundle_validation_compares_sessions_per_participant(tmp_path):
    """Ordinal labels repeat across participants: (P, '02') must fail even if Q has a session 02."""
    from click.testing import CliRunner
    from b2aiprep.commands import validate_bundled_dataset
    from test_review_fixes_20260930 import _bundle, _config

    _bundle(tmp_path / "bundle", ["session_status"])
    task = tmp_path / "bundle" / "phenotype" / "task"
    pd.DataFrame({"participant_id": ["005009", "005010"], "session_id": ["01", "02"],
                  "session_status": ["x", "x"]}).to_csv(task / "session.tsv", sep="\t", index=False)
    features = tmp_path / "bundle" / "features"
    pd.DataFrame({"participant_id": ["005009"], "task_name": ["test"], "session_id": ["02"]}).to_parquet(
        features / "torchaudio_mfcc.parquet")
    config = _config(tmp_path / "cfg")
    (config / "id_remapping.json").write_text(json.dumps({"p1": "005009", "p2": "005010"}))
    (config / "participants_to_include.json").write_text(json.dumps(["p1", "p2"]))
    result = CliRunner().invoke(validate_bundled_dataset, [str(tmp_path / "bundle"), str(config)])
    assert result.exit_code != 0 and "torchaudio_mfcc.parquet not found in session.tsv" in result.output


def test_relabel_changes_only_the_listed_labels_in_cells_and_choices():
    df = pd.DataFrame({"participant_id": ["1", "2", "3"], "ph_walking": ["None", "Mild", pd.NA]})
    choices = [{"name": {"en": "None"}, "value": "none"}, {"name": {"en": "Mild"}, "value": "mild"}]
    elements = {"ph_walking": {"choices": choices}}
    out, el = BIDSDataset._apply_relabels(df, elements, "confounders",
                                         {"confounders": {"ph_walking": {"None": "No difficulty"}}})
    assert out["ph_walking"].tolist()[:2] == ["No difficulty", "Mild"] and pd.isna(out["ph_walking"].iloc[2])
    assert el["ph_walking"]["choices"] == [{"name": {"en": "No difficulty"}, "value": "none"},
                                           {"name": {"en": "Mild"}, "value": "mild"}]


def test_relabel_refuses_a_label_the_column_does_not_have():
    df = pd.DataFrame({"participant_id": ["1"], "ph_walking": ["None"]})
    elements = {"ph_walking": {"choices": [{"name": {"en": "None"}, "value": "none"}]}}
    with pytest.raises(ValueError, match="no choice labelled"):
        BIDSDataset._apply_relabels(df, elements, "confounders", {"confounders": {"ph_walking": {"Nome": "x"}}})


def test_bundle_validation_fails_on_cells_readers_take_for_missing(tmp_path):
    from click.testing import CliRunner
    from b2aiprep.commands import validate_bundled_dataset
    from test_review_fixes_20260930 import _bundle, _config

    _bundle(tmp_path / "bundle", ["session_status"])
    folder = tmp_path / "bundle" / "phenotype" / "confounders"
    folder.mkdir(parents=True)
    pd.DataFrame({"participant_id": ["005009"], "ph_walking": ["None"]}).to_csv(
        folder / "confounders.tsv", sep="\t", index=False)
    (folder / "confounders.json").write_text(json.dumps({"confounders": {"data_elements": {
        "ph_walking": {"choices": [{"name": {"en": "None"}, "value": "none"}]}}}}))
    result = CliRunner().invoke(validate_bundled_dataset, [str(tmp_path / "bundle"), str(_config(tmp_path / "cfg"))])
    assert result.exit_code != 0 and "ph_walking (1)" in result.output


def test_bundle_validation_leaves_typed_na_in_free_text(tmp_path):
    result = _bundle_with_confounders(tmp_path, ["other_voice_activity"], [
        {"participant_id": "p1", "column_name": "other_voice_activity", "verdict": "safe"}])
    assert result.exit_code == 0, result.output
    folder = tmp_path / "bundle" / "phenotype" / "confounders"
    pd.DataFrame({"participant_id": ["005009"], "other_voice_activity": ["N/A"]}).to_csv(
        folder / "confounders.tsv", sep="\t", index=False)
    from click.testing import CliRunner
    from b2aiprep.commands import validate_bundled_dataset
    result = CliRunner().invoke(validate_bundled_dataset, [str(tmp_path / "bundle"), str(tmp_path / "cfg")])
    assert result.exit_code == 0, result.output


def test_relabelled_answers_and_their_choices_reach_the_released_json(tmp_path):
    choices = [{"name": {"en": "None"}, "value": "none"}, {"name": {"en": "Mild"}, "value": "mild"},
               {"name": {"en": "Extreme or cannot do"}, "value": "extreme"}]
    confounders = pd.DataFrame({"participant_id": ["p1", "p2"], "ph_walking": ["None", "Mild"]})
    bids, config = _tree(tmp_path, ["p1", "p2"], {"confounders/confounders": confounders})
    sidecar = bids / "phenotype" / "confounders" / "confounders.json"
    meta = json.loads(sidecar.read_text())
    meta["confounders"]["data_elements"]["ph_walking"]["choices"] = choices
    sidecar.write_text(json.dumps(meta))
    (config / "deidentify_settings.json").write_text(json.dumps({
        "access_tier": "registered",
        "relabel": {"confounders": {"ph_walking": {"None": "No difficulty"}}}}))
    out = tmp_path / "out"
    BIDSDataset(bids).deidentify(outdir=out, deidentify_config_dir=config)
    released = out / "phenotype" / "confounders"
    tsv = BIDSDataset._read_tsv_as_written(released / "confounders.tsv")
    assert sorted(tsv["ph_walking"]) == ["Mild", "No difficulty"]
    written = json.loads((released / "confounders.json").read_text())
    element = written["confounders"]["data_elements"]["ph_walking"]
    assert element["choices"] == [{"name": {"en": "No difficulty"}, "value": "none"},
                                  {"name": {"en": "Mild"}, "value": "mild"},
                                  {"name": {"en": "Extreme or cannot do"}, "value": "extreme"}]


def test_a_row_whose_only_released_value_is_derived_is_kept():
    """Only REDCap-generated columns are bookkeeping; a derived column such as state_province is data."""
    df = pd.DataFrame({"participant_id": ["900000", "900001"], "demographics_session_id": ["01", "01"],
                       "state_province": ["Ontario", pd.NA], "demographics_duration": ["120", "95"]})
    out = BIDSDataset._drop_rows_emptied_by_deidentify(df, "demographics")
    assert list(out["participant_id"]) == ["900000"]


def test_exclude_participants_removes_a_participant_by_one_answer(tmp_path):
    demographics = pd.DataFrame({"participant_id": ["p1", "p2"],
                                 "peds_gender_identity": ["Female gender identity", "Other"]})
    bids, config = _tree(tmp_path, ["p1", "p2"], {"pediatric/pediatric_demographics": demographics})
    (config / "deidentify_settings.json").write_text(json.dumps({
        "access_tier": "registered",
        "exclude_participants": [{"table": "pediatric_demographics", "column": "peds_gender_identity",
                                  "values": ["Other"]}]}))
    out = tmp_path / "out"
    BIDSDataset(bids).deidentify(outdir=out, deidentify_config_dir=config)
    assert sorted(p.name for p in out.glob("sub-*")) == ["sub-900000"]
    released = BIDSDataset._read_tsv_as_written(out / "phenotype" / "pediatric" / "pediatric_demographics.tsv")
    assert list(released["participant_id"]) == ["900000"]


def test_exclude_participants_refuses_a_column_that_is_not_there(tmp_path):
    demographics = pd.DataFrame({"participant_id": ["p1"], "peds_gender_identity": ["Other"]})
    bids, config = _tree(tmp_path, ["p1"], {"pediatric/pediatric_demographics": demographics})
    (config / "deidentify_settings.json").write_text(json.dumps({
        "access_tier": "registered",
        "exclude_participants": [{"table": "pediatric_demographics", "column": "peds_gender_identiy", "values": ["Other"]}]}))
    with pytest.raises(ValueError, match="no column 'peds_gender_identiy'"):
        BIDSDataset(bids).deidentify(outdir=tmp_path / "out", deidentify_config_dir=config)


def test_ingest_task_table_filter_keeps_typed_na_like_answers(tmp_path):
    """The recording / acoustic_task filters at ingest re-read and rewrite the tables."""
    task = tmp_path / "task"
    task.mkdir()
    (task / "recording.tsv").write_text("recording_id\trecording_acoustic_task_id\trecording_microphone\n"
                                        "r1\tt1\tNone\nr2\tt2\tN/A\n")
    (task / "acoustic_task.tsv").write_text("acoustic_task_id\tacoustic_task_notes\nt1\tNA\nt2\tnull\n")
    BIDSDataset._filter_task_tables_to_recordings(str(tmp_path), {"r1"})
    rec = BIDSDataset._read_tsv_as_written(task / "recording.tsv")
    at = BIDSDataset._read_tsv_as_written(task / "acoustic_task.tsv")
    assert rec.to_dict("list") == {"recording_id": ["r1"], "recording_acoustic_task_id": ["t1"],
                                   "recording_microphone": ["None"]}
    assert at.to_dict("list") == {"acoustic_task_id": ["t1"], "acoustic_task_notes": ["NA"]}


def test_exclude_participants_refuses_a_value_that_is_not_an_answer_choice(tmp_path):
    demographics = pd.DataFrame({"participant_id": ["p1"], "peds_gender_identity": ["Other"]})
    bids, config = _tree(tmp_path, ["p1"], {"pediatric/pediatric_demographics": demographics})
    sidecar = bids / "phenotype" / "pediatric" / "pediatric_demographics.json"
    meta = json.loads(sidecar.read_text())
    meta["pediatric_demographics"]["data_elements"]["peds_gender_identity"]["choices"] = [
        {"name": {"en": "Female gender identity"}, "value": "female"}, {"name": {"en": "Other"}, "value": "other"}]
    sidecar.write_text(json.dumps(meta))
    (config / "deidentify_settings.json").write_text(json.dumps({
        "access_tier": "registered",
        "exclude_participants": [{"table": "pediatric_demographics", "column": "peds_gender_identity",
                                  "values": ["other"]}]}))
    with pytest.raises(ValueError, match=r"no choice labelled \['other'\]"):
        BIDSDataset(bids).deidentify(outdir=tmp_path / "out", deidentify_config_dir=config)


def test_session_labels_agree_between_tiers_when_a_session_has_only_controlled_data(tmp_path, monkeypatch):
    """A registered build numbers labels over what the controlled tier could release too."""
    field_map = BIDSDataset._load_reorganization_file(exclude_dropped=False).copy()
    field_map["access_tier"] = field_map["access_tier"].astype(object)
    field_map.loc[(field_map["schema_name"] == "confounders") & (field_map["column_name"] == "ph_walking"),
                  "access_tier"] = "controlled"
    monkeypatch.setattr(BIDSDataset, "_load_reorganization_file",
                        staticmethod(lambda exclude_dropped=True: field_map if not exclude_dropped
                                     else field_map.loc[field_map["disposition"] != "drop"]))
    monkeypatch.setattr(BIDSDataset, "_cached_field_map_df", None)
    labels = {}
    for tier in ("registered", "controlled"):
        confounders = pd.DataFrame({"participant_id": ["p1"], "confounders_session_id": ["s2"],
                                    "ph_walking": ["Mild"]})
        bids, config = _tree(tmp_path / tier, ["p1"], {"confounders/confounders": confounders})
        audio = bids / "sub-p1" / "ses-s3" / "audio"
        audio.mkdir(parents=True)
        stem = "sub-p1_ses-s3_task-rainbow-passage"
        (audio / f"{stem}.wav").write_bytes(b"RIFF" + b"\x00" * 8192)
        (audio / f"{stem}_recording-metadata.json").write_text(json.dumps({"record_id": "p1", "session_id": "s3"}))
        pd.DataFrame({"record_id": ["p1"] * 3, "session_id": ["s1", "s2", "s3"],
                      "session_index": ["1", "2", "3"]}).to_csv(bids / "sub-p1" / "sessions.tsv", sep="\t", index=False)
        (config / "deidentify_settings.json").write_text(json.dumps({"access_tier": tier}))
        out = tmp_path / tier / "out"
        BIDSDataset(bids).deidentify(outdir=out, deidentify_config_dir=config)
        (sessions,) = list((out / "sub-900000").glob("*sessions.tsv"))
        labels[tier] = list(BIDSDataset._read_tsv_as_written(sessions)["session_id"])
    assert labels["registered"] == ["01", "03"]  # s2 is controlled-only: withheld, but s3 keeps 03
    assert labels["controlled"] == ["01", "02", "03"]
