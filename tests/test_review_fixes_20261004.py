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


def test_sidecar_never_keeps_an_unreleased_session_id(tmp_path):
    bids, config = _tree(tmp_path, ["p1"], {})
    sidecar = bids / "sub-p1" / "ses-s1" / "audio" / "sub-p1_ses-s1_task-rainbow-passage_recording-metadata.json"
    sidecar.write_text(json.dumps({"record_id": "p1", "session_id": "s2-unreleased"}))
    out = tmp_path / "out"
    BIDSDataset(bids).deidentify(outdir=out, deidentify_config_dir=config)
    (written,) = [p for p in out.rglob("*.json") if "task-rainbow-passage" in p.name]
    text = written.read_text()
    assert "s2-unreleased" not in text
    session_dir = written.parent.parent.name  # ses-<label>
    assert json.loads(text)["session_id"] == session_dir[len("ses-"):]


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
