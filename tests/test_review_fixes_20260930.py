"""Fixes from the 2026-09-29 review of feat/date-shift (drop columns, pseudonyms, bundle columns,
quality-metric exclusions, derived-field edge cases)."""
import json

import pandas as pd
import pytest
from click.testing import CliRunner

from b2aiprep.commands import validate_bundled_dataset
from b2aiprep.prepare.dataset import BIDSDataset, DispositionLevel


# the adult configs' small_checkbox_options (4.0-release/configs)
VOICE_ACTIVITY_FOLD = {"confounders": {
    "voice_activity_v2": {"other": "voice_activity_v2___other", "specify": "other_voice_activity",
                          "keep": ["voice_activity_v2___none"], "min_participants": 10},
    "voice_activity": {"other": "voice_activity___7", "min_participants": 10}}}


def _fm(rows):
    return pd.DataFrame(rows, columns=["schema_name", "column_name", "disposition", "date_shift"])


def test_drop_columns_are_removed_at_every_level():
    fm = _fm([("t", "a_release", "release", "NO"), ("t", "b_internal", "internal", "NO"),
              ("t", "c_drop", "drop", "NO"), ("t", "d_drop_date", "drop", "YES"),
              ("t", "e_internal_date", "internal", "YES")])
    assert BIDSDataset._names_to_drop_at_level(fm, DispositionLevel.INTERNAL) == {"c_drop", "d_drop_date"}
    assert BIDSDataset._names_to_drop_at_level(fm, DispositionLevel.REVIEW, keep_date_shifted=True) == {
        "b_internal", "c_drop", "d_drop_date"}
    assert BIDSDataset._names_to_drop_at_level(fm, DispositionLevel.RELEASE) == {
        "b_internal", "c_drop", "d_drop_date", "e_internal_date"}


def _tree(tmp_path, remap):
    bids, config = tmp_path / "bids", tmp_path / "config"
    config.mkdir(parents=True)
    (config / "participants_to_include.json").write_text(json.dumps(["p1"]))
    (config / "id_remapping.json").write_text(json.dumps(remap))
    (config / "deidentify_settings.json").write_text(json.dumps({"access_tier": "registered"}))
    (config / "audio_filestems_to_remove.json").write_text(json.dumps([]))
    (config / "audio_tasks_to_include.json").write_text(json.dumps(["rainbow-passage"]))
    audio = bids / "sub-p1" / "ses-s1" / "audio"
    audio.mkdir(parents=True)
    stem = "sub-p1_ses-s1_task-rainbow-passage"
    (audio / f"{stem}.wav").write_bytes(b"RIFF" + b"\x00" * 8192)
    (audio / f"{stem}_recording-metadata.json").write_text(json.dumps({"record_id": "p1", "session_id": "s1"}))
    pd.DataFrame({"record_id": ["p1"], "session_id": ["s1"], "session_index": ["1"]}).to_csv(
        bids / "sub-p1" / "sessions.tsv", sep="\t", index=False)
    (bids / "phenotype").mkdir()
    (bids / "dataset_description.json").write_text(json.dumps({"Name": "test"}))
    return bids, config


def test_deidentify_refuses_an_allowlisted_participant_without_a_pseudonym(tmp_path):
    bids, config = _tree(tmp_path, {})
    with pytest.raises(ValueError, match="1 allowlisted participant.*no pseudonym.*p1"):
        BIDSDataset(bids).deidentify(outdir=tmp_path / "out", deidentify_config_dir=config)
    assert not (tmp_path / "out" / "sub-p1").exists()


def test_quality_metrics_exclusion_matches_the_recording_exactly(tmp_path):
    (tmp_path / "bids").mkdir()
    pd.DataFrame({
        "participant_id": ["p1", "p1", "p1"], "session_id": ["s1"] * 3,
        "task_name": ["identifying-pictures-2", "identifying-pictures-20", "identifying-pictures-21"],
        "snr": ["1", "2", "3"],
    }).to_csv(tmp_path / "bids" / "audio_quality_metrics.tsv", sep="\t", index=False)
    (tmp_path / "out").mkdir()
    BIDSDataset._deidentify_quality_metrics(
        tmp_path / "bids", tmp_path / "out", [], ["sub-p1_ses-s1_task-identifying-pictures-2"],
        [], {"p1": "900001"}, {})
    out = pd.read_csv(tmp_path / "out" / "audio_quality_metrics.tsv", sep="\t", dtype=str)
    assert sorted(out.task_name) == ["identifying-pictures-20", "identifying-pictures-21"]


def _bundle(root, session_columns):
    features, task = root / "features", root / "phenotype" / "task"
    features.mkdir(parents=True)
    task.mkdir(parents=True)
    df = pd.DataFrame({"participant_id": ["005009"], "task_name": ["test"], "session_id": ["01"]})
    df.to_parquet(features / "torchaudio_spectrogram.parquet")
    df.to_parquet(features / "torchaudio_mfcc.parquet")
    (features / "static_features.tsv").write_text("participant_id\tsession_id\n005009\t01\n")
    (features / "static_features.json").write_text("{}")
    pd.DataFrame({"participant_id": ["005009"], "session_id": ["01"],
                  **{c: ["x"] for c in session_columns}}).to_csv(task / "session.tsv", sep="\t", index=False)


def _config(root):
    root.mkdir()
    (root / "participants_to_include.json").write_text(json.dumps(["p1"]))
    (root / "id_remapping.json").write_text(json.dumps({"p1": "005009"}))
    (root / "deidentify_settings.json").write_text(json.dumps({"access_tier": "registered"}))
    (root / "audio_tasks_to_include.json").write_text(json.dumps(["test"]))
    return root


@pytest.mark.parametrize("column", ["session_started_at", "session_complete", "session_site", "not_a_field"])
def test_bundle_validation_fails_on_columns_a_release_must_not_have(tmp_path, column):
    _bundle(tmp_path / "bundle", [column])
    result = CliRunner().invoke(validate_bundled_dataset, [str(tmp_path / "bundle"), str(_config(tmp_path / "cfg"))])
    assert result.exit_code != 0 and column in result.output


def test_bundle_validation_passes_released_columns(tmp_path):
    _bundle(tmp_path / "bundle", ["session_status", "session_index", "session_local_hour"])
    result = CliRunner().invoke(validate_bundled_dataset, [str(tmp_path / "bundle"), str(_config(tmp_path / "cfg"))])
    assert result.exit_code == 0, result.output


def _na(values):
    return [v if isinstance(v, str) else None for v in values]


def test_date_only_start_has_no_local_hour():
    out = BIDSDataset._add_local_hours(pd.DataFrame({"session_started_at": ["2100-01-05", "2100-01-05T08:00:00-05:00"]}))
    assert _na(out.session_local_hour) == [None, "8"]


def test_days_since_surgery_uses_the_first_numeric_age(caplog):
    mc = "Q - Pediatric - Generic Medical Conditions"
    df = pd.DataFrame([
        {"record_id": "c1", "redcap_repeat_instrument": "Participant", "age": "5"},
        {"record_id": "c1", "redcap_repeat_instrument": "Participant", "age": "9"},
        {"record_id": "c1", "redcap_repeat_instrument": "Session", "session_id": "S1",
         "session_started_at": "2100-01-05T10:00:00-05:00"},
        {"record_id": "c1", "redcap_repeat_instrument": mc, "peds_mc_session_id": "S1",
         "peds_mc_tonsillectomy_date": "2093-01-05"},  # 2,557 days: over 5+1 years, under 9+1
    ], dtype=object)
    with caplog.at_level("WARNING"):
        BIDSDataset._add_days_since_surgery(df)
    assert "before the participant was born" in caplog.text


def test_session_hour_check_sees_a_later_dated_row_of_the_session(caplog):
    df = pd.DataFrame([
        {"record_id": "p1", "redcap_repeat_instrument": "Session", "session_id": "S1", "session_local_hour": None},
        {"record_id": "p1", "redcap_repeat_instrument": "Session", "session_id": "S1", "session_local_hour": "22"},
    ], dtype=object)
    with caplog.at_level("WARNING"):
        BIDSDataset._check_session_hours(df)
    assert "1 in-clinic session(s) of 1" in caplog.text and "p1 S1 (22h)" in caplog.text


def test_rare_checkbox_options_fold_into_other_counted_over_released_participants():
    pids = [f"p{i}" for i in range(12)]
    df = pd.DataFrame({
        "participant_id": pids,
        "voice_activity_v2___teacher": [1] * 10 + [None, None],          # 10 participants: kept
        "voice_activity_v2___attorney": [None] * 10 + [1, 1],            # 2: folded
        "voice_activity_v2___other": [None] * 11 + [1],
        "voice_activity_v2___none": [None] * 12,
        "other_voice_activity": [None] * 11 + ["Podcaster"],
    }, dtype=object)
    elements = {c: {"description": c} for c in df.columns}
    elements["voice_activity_v2___attorney"]["choices"] = [{"name": {"en": "Attorney"}, "value": "attorney"}]
    out, el = BIDSDataset._fold_small_checkbox_options(df, elements, "confounders", VOICE_ACTIVITY_FOLD)
    assert "voice_activity_v2___attorney" not in out.columns and "voice_activity_v2___attorney" not in el
    assert "voice_activity_v2___teacher" in out.columns
    assert list(out["voice_activity_v2___other"])[-2:] == [1, 1]
    assert list(out["other_voice_activity"])[-2:] == ["Attorney", "Podcaster; Attorney"]


def test_checkbox_fold_works_on_a_table_read_as_text():
    """Deidentify reads tables as text, so the fold must write "1", not the integer 1."""
    df = pd.DataFrame({
        "participant_id": [f"p{i}" for i in range(3)],
        "voice_activity_v2___attorney": ["1", "", ""],
        "voice_activity_v2___other": ["", "", "1"],
        "other_voice_activity": ["", "", "Podcaster"],
    }, dtype=str).replace("", pd.NA)
    out, _ = BIDSDataset._fold_small_checkbox_options(df, {}, "confounders", VOICE_ACTIVITY_FOLD)
    assert out["voice_activity_v2___other"].tolist()[0::2] == ["1", "1"]


def test_small_checkbox_rules_can_come_from_the_deidentify_settings(tmp_path):
    rules = {"t": {"q": {"other": "q___other", "min_participants": 5}}}
    (tmp_path / "deidentify_settings.json").write_text(json.dumps({"access_tier": "registered", "small_checkbox_options": rules}))
    assert BIDSDataset._load_deidentify_settings(tmp_path)["small_checkbox_options"] == rules
    (tmp_path / "deidentify_settings.json").write_text(json.dumps({"access_tier": "registered"}))
    assert BIDSDataset._load_deidentify_settings(tmp_path)["small_checkbox_options"] == {}


