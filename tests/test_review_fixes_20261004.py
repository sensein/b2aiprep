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
