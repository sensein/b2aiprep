"""Tests for per-participant date shifting at ingest."""

import datetime
import json

import pandas as pd
import pytest

from b2aiprep.prepare.dataset import BIDSDataset
from b2aiprep.prepare.date_shift import (
    SITE_TIMEZONES,
    offset_weeks,
    parse_date,
    parse_utc_timestamp,
    shift_dates,
)
from b2aiprep.prepare.redcap import RedCapDataset

ANCHOR = datetime.date(2100, 1, 1)
UTC = datetime.timezone.utc


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("2024-11-04T18:34:41Z", datetime.datetime(2024, 11, 4, 18, 34, 41, tzinfo=UTC)),
        ("2024-11-04T18:34:41.250Z", datetime.datetime(2024, 11, 4, 18, 34, 41, 250000, tzinfo=UTC)),
        ("2024-11-04-T18:34:41Z", datetime.datetime(2024, 11, 4, 18, 34, 41, tzinfo=UTC)),
        ("20240512T14:22:01.123Z", datetime.datetime(2024, 5, 12, 14, 22, 1, 123000, tzinfo=UTC)),
        ("2024051214:22:01.123", datetime.datetime(2024, 5, 12, 14, 22, 1, 123000, tzinfo=UTC)),
        ("04/08/2026 19:36:53", datetime.datetime(2026, 4, 8, 19, 36, 53, tzinfo=UTC)),
        ("4/08/2026 7:36:53 PM", datetime.datetime(2026, 4, 8, 19, 36, 53, tzinfo=UTC)),
    ],
)
def test_parse_utc_timestamp_formats(raw, expected):
    assert parse_utc_timestamp(raw) == expected


@pytest.mark.parametrize("raw", ["", "[not completed]", "2024-11-04", "Yes", None, float("nan")])
def test_parse_utc_timestamp_rejects_non_timestamps(raw):
    assert parse_utc_timestamp(raw) is None


def test_parse_date():
    assert parse_date("2019-02-28") == datetime.date(2019, 2, 28)
    assert parse_date("2019-02-30") is None
    assert parse_date("2019-02-28T00:00:00Z") is None


@pytest.mark.parametrize("days_before", range(0, 800, 37))
def test_offset_lands_within_three_days_on_same_weekday(days_before):
    first = ANCHOR - datetime.timedelta(days=days_before + 30000)
    shifted = first + datetime.timedelta(weeks=offset_weeks(ANCHOR, first))
    assert abs((shifted - ANCHOR).days) <= 3
    assert shifted.weekday() == first.weekday()


def test_mt_sinai_is_toronto():
    assert SITE_TIMEZONES["Mt. Sinai"] == "America/Toronto"


def _frame(rows):
    columns = ["record_id", "redcap_repeat_instrument", "enrollment_institution", "session_started_at", "phq_9_started_at", "surgery_date"]
    df = pd.DataFrame([dict(zip(columns, r)) for r in rows], columns=columns, dtype=object)
    is_session = df["redcap_repeat_instrument"] == "Session"
    df["session_id"] = [f"{rid}-s{i}" if s else None for i, (rid, s) in enumerate(zip(df["record_id"], is_session))]
    return df


def test_shift_keeps_local_time_and_real_offset():
    # 2024-07-01 16:00Z is 12:00 EDT (-04:00); 2024-12-02 17:00Z is 12:00 EST (-05:00).
    df = _frame([
        ("a", "Session", "MIT", "2024-07-01T16:00:00Z", None, None),
        ("a", "Q", None, None, "2024-12-02T17:00:00Z", None),
    ])
    out, report = shift_dates(df, ["session_started_at", "phq_9_started_at"], ANCHOR)
    first = datetime.datetime.fromisoformat(out.loc[0, "session_started_at"])
    later = datetime.datetime.fromisoformat(out.loc[1, "phq_9_started_at"])
    assert (first.hour, first.utcoffset()) == (12, datetime.timedelta(hours=-4))
    assert (later.hour, later.utcoffset()) == (12, datetime.timedelta(hours=-5))
    assert abs((first.date() - ANCHOR).days) <= 3
    assert first.weekday() == datetime.date(2024, 7, 1).weekday()
    # Elapsed time is exact across the DST change.
    real = datetime.datetime(2024, 12, 2, 17, tzinfo=UTC) - datetime.datetime(2024, 7, 1, 16, tzinfo=UTC)
    assert later - first == real
    assert report["participants_shifted"] == 1


def test_date_only_values_shift_by_the_same_weeks():
    df = _frame([
        ("a", "Session", "SickKids", "2024-07-01T16:00:00Z", None, None),
        ("a", None, None, None, None, "2019-02-28"),
    ])
    out, _ = shift_dates(df, ["session_started_at", "surgery_date"], ANCHOR)
    shifted_session = datetime.datetime.fromisoformat(out.loc[0, "session_started_at"]).date()
    shifted_surgery = datetime.date.fromisoformat(out.loc[1, "surgery_date"])
    assert (shifted_session - shifted_surgery) == (datetime.date(2024, 7, 1) - datetime.date(2019, 2, 28))


def test_participants_without_offset_are_blanked():
    df = _frame([
        ("no_site", "Session", None, "2024-07-01T16:00:00Z", None, None),
        ("unknown_site", "Session", "Elsewhere", "2024-07-01T16:00:00Z", None, None),
        ("no_session", "Q", None, None, "2024-07-01T16:00:00Z", None),
    ])
    out, report = shift_dates(df, ["session_started_at", "phq_9_started_at"], ANCHOR)
    assert out[["session_started_at", "phq_9_started_at"]].isna().all().all()
    assert set(report["participants_without_offset"]) == {"no_site", "unknown_site", "no_session"}


def test_unparseable_and_empty_markers_are_blanked():
    df = _frame([
        ("a", "Session", "MIT", "2024-07-01T16:00:00Z", None, None),
        ("a", "Q", None, None, "[not completed]", "sometime in 2019"),
    ])
    out, report = shift_dates(df, ["phq_9_started_at", "surgery_date"], ANCHOR)
    assert pd.isna(out.loc[1, "phq_9_started_at"]) and pd.isna(out.loc[1, "surgery_date"])
    assert report["columns"]["surgery_date"] == {"blanked_unparseable": 1}


def test_no_anchor_blanks_every_date():
    df = _frame([("a", "Session", "MIT", "2024-07-01T16:00:00Z", "2024-07-01T16:10:00Z", "2019-02-28")])
    out, report = shift_dates(df, ["session_started_at", "phq_9_started_at", "surgery_date"], None)
    assert out[["session_started_at", "phq_9_started_at", "surgery_date"]].isna().all().all()
    assert not report["anchor_given"]


def test_no_real_value_survives():
    real = ["2024-07-01T16:00:00Z", "2024-07-01T16:10:00Z", "2019-02-28"]
    df = _frame([("a", "Session", "WCM", real[0], real[1], real[2])])
    out, _ = shift_dates(df, ["session_started_at", "phq_9_started_at", "surgery_date"], ANCHOR)
    text = out.to_csv()
    assert not any(value[:10] in text for value in real)


def _ingest_frame():
    return pd.DataFrame(
        [
            {"record_id": "a", "redcap_repeat_instrument": "Session", "session_id": "s1",
             "enrollment_institution": "MIT", "session_site": "MIT", "session_started_at": "2024-07-01T16:00:00Z",
             "session_is_control_participant": "No"},
            {"record_id": "a", "redcap_repeat_instrument": "Participant", "city": "Cambridge"},
        ],
        dtype=object,
    )


def test_ingest_shifts_dates_and_removes_drop_columns(tmp_path):
    dataset = RedCapDataset(df=_ingest_frame(), source_type="redcap")
    log = tmp_path / "logs" / "date_shift.json"
    out = BIDSDataset._apply_field_map_at_ingest(dataset, ANCHOR, log, tmp_path / "bids")
    assert "city" not in out.df.columns
    assert "session_is_control_participant" not in out.df.columns
    # internal columns the pipeline reads are kept
    assert {"redcap_repeat_instrument", "enrollment_institution"} <= set(out.df.columns)
    assert "session_site" not in out.df.columns
    shifted = datetime.datetime.fromisoformat(out.df.loc[0, "session_started_at"]).date()
    assert abs((shifted - ANCHOR).days) <= 3
    report = json.loads(log.read_text())
    assert report["anchor"] == "2100-01-01" and report["participants_shifted"] == 1
    # the input dataset is not modified
    assert dataset.df.loc[0, "session_started_at"] == "2024-07-01T16:00:00Z"


def test_ingest_refuses_a_log_inside_the_bids_tree(tmp_path):
    dataset = RedCapDataset(df=_ingest_frame(), source_type="redcap")
    with pytest.raises(ValueError, match="outside the BIDS output"):
        BIDSDataset._apply_field_map_at_ingest(dataset, ANCHOR, tmp_path / "bids" / "log.json", tmp_path / "bids")


def test_instrument_columns_skip_dropped_columns():
    df = _ingest_frame().drop(columns=["session_is_control_participant"])
    dataset = RedCapDataset(df=df, source_type="redcap")
    from b2aiprep.prepare.constants import RepeatInstrument

    sessions = dataset.get_df_of_repeat_instrument(RepeatInstrument.SESSION.value)
    assert "session_is_control_participant" not in sessions.columns
    assert "session_started_at" in sessions.columns


def _remote_frame(zipcode=None, state=None, via="Participant", site="MIT"):
    """One participant: a session at 2024-07-01 20:00Z with one self-administered recording."""
    return pd.DataFrame(
        [
            {"record_id": "r", "redcap_repeat_instrument": "Session", "session_id": "s1",
             "enrollment_institution": site, "session_started_at": "2024-07-01T20:00:00Z"},
            {"record_id": "r", "redcap_repeat_instrument": "Recording", "recording_session_id": "s1",
             "recording_via": via, "recording_created_at": "2024-07-01T20:05:00Z"},
            {"record_id": "r", "redcap_repeat_instrument": "Q - Generic - Demographics",
             "zipcode": zipcode, "state_province": state},
        ],
        dtype=object,
    )


def _local_hour(df, col, row):
    return datetime.datetime.fromisoformat(df.loc[row, col]).hour


@pytest.mark.parametrize(
    "zipcode, state, expected_hour, source",
    [
        ("90210", None, 13, "self_administered_postal_code"),   # Los Angeles, PDT
        ("80202-1234", None, 14, "self_administered_postal_code"),  # Denver, MDT
        ("M5G 1X5", None, 16, "self_administered_postal_code"),  # Toronto, EDT
        (None, "IL", 15, "self_administered_region"),            # Chicago, CDT
        (None, "FL", 16, "self_administered_site_fallback"),     # FL spans two zones -> site
        (None, None, 16, "self_administered_site_fallback"),
    ],
)
def test_self_administered_sessions_use_the_participants_location(zipcode, state, expected_hour, source):
    out, report = shift_dates(_remote_frame(zipcode, state), ["session_started_at", "recording_created_at"], ANCHOR)
    assert _local_hour(out, "session_started_at", 0) == expected_hour
    assert _local_hour(out, "recording_created_at", 1) == expected_hour
    assert report["session_timezone_sources"] == {source: 1}


def test_in_person_sessions_use_the_site_even_if_the_participant_lives_elsewhere():
    out, report = shift_dates(
        _remote_frame("90210", via="Data Collector"), ["session_started_at", "recording_created_at"], ANCHOR
    )
    assert _local_hour(out, "session_started_at", 0) == 16  # MIT, EDT
    assert report["session_timezone_sources"] == {"site": 1}


def test_split_state_zip_codes_resolve_to_their_own_zone():
    from b2aiprep.prepare.date_shift import postal_code_timezone, region_timezone

    assert postal_code_timezone("37203") == "America/Chicago"   # Nashville
    assert postal_code_timezone("37902") == "America/New_York"  # Knoxville
    assert postal_code_timezone("32501") == "America/Chicago"   # Pensacola
    assert postal_code_timezone("P9N 1A1") == "America/Winnipeg"  # Kenora
    assert postal_code_timezone("2139") == "America/New_York"   # leading zero lost
    assert region_timezone("TN") is None and region_timezone("ON") is None


def test_site_comes_from_enrollment_institution_not_session_site():
    df = _frame([
        ("a", "Session", "VUMC", "2024-07-01T16:00:00Z", None, None),
        ("a", "Session", None, "2024-08-01T16:00:00Z", None, None),
    ])
    df["session_site"] = ["MIT", None]  # ignored
    out, report = shift_dates(df, ["session_started_at"], ANCHOR)
    assert _local_hour(out, "session_started_at", 0) == 11  # Chicago
    assert _local_hour(out, "session_started_at", 1) == 11  # institution is per participant
    assert report["session_timezone_sources"] == {"site": 2}


def test_shifted_timestamps_keep_milliseconds():
    df = _frame([("a", "Session", "MIT", "2024-07-01T16:00:00.900Z", "2024-07-01T16:00:01.100Z", None)])
    out, _ = shift_dates(df, ["session_started_at", "phq_9_started_at"], ANCHOR)
    start = datetime.datetime.fromisoformat(out.loc[0, "session_started_at"])
    later = datetime.datetime.fromisoformat(out.loc[0, "phq_9_started_at"])
    assert later - start == datetime.timedelta(milliseconds=200)


def test_disposition_is_matched_within_the_table():
    """self_reported_* is released in eligibility and internal in enrollment."""
    field_map = pd.DataFrame(
        [
            {"schema_name": "eligibility", "column_name": "self_reported_asthma", "disposition": "release"},
            {"schema_name": "enrollment", "column_name": "self_reported_asthma", "disposition": "internal"},
        ]
    )
    df = pd.DataFrame({"participant_id": ["p"], "self_reported_asthma": ["Yes"]})
    kept, dropped = BIDSDataset._drop_columns_by_disposition(df, field_map, schema_name="eligibility")
    assert "self_reported_asthma" in kept.columns and not dropped
    kept, dropped = BIDSDataset._drop_columns_by_disposition(df, field_map, schema_name="enrollment")
    assert dropped == ["self_reported_asthma"]


def test_internal_only_rows_survive_ingest_and_are_dropped_after_deidentify(monkeypatch):
    df = pd.DataFrame(
        {
            "participant_id": ["has_release", "internal_only"],
            "is_prolific": ["Yes", None],
            "self_reported_asthma": [None, "Yes"],  # internal in enrollment
            "enrollment_form_complete": ["Complete", "Complete"],
        }
    )
    # Ingest: the internal column counts as data, so both rows are kept.
    kept = BIDSDataset._drop_rows_without_substantive_data(
        df, "participant_id", {"enrollment_form_complete"}, set(), schema_name="enrollment"
    )
    assert list(kept.participant_id) == ["has_release", "internal_only"]
    # Deidentify strips the internal columns, then re-checks.
    deidentified = kept.drop(columns=["self_reported_asthma", "enrollment_form_complete"])
    field_map = pd.DataFrame(
        [{"schema_name": "enrollment", "column_name": "is_prolific", "disposition": "release",
          "source": "redcap", "is_redcap_calculation": "NO", "group": "enrollment"}]
    )
    monkeypatch.setattr(BIDSDataset, "_cached_field_map_df", field_map)
    out = BIDSDataset._drop_rows_emptied_by_deidentify(deidentified, "enrollment")
    assert list(out.participant_id) == ["has_release"]


def _sidecar_tree(root, pid="p1", ses="S1", recordings=(("rec-A", "noisy-sounds-1"), ("rec-B", "noisy-sounds-2"))):
    audio = root / f"sub-{pid}" / f"ses-{ses}" / "audio"
    audio.mkdir(parents=True)
    for rec_id, task in recordings:
        stem = f"sub-{pid}_ses-{ses}_task-{task}"
        (audio / f"{stem}_recording-metadata.json").write_text(
            json.dumps({"record_id": pid, "recording_id": rec_id.upper(), "session_id": ses})
        )
    pd.DataFrame({"record_id": [pid], "session_id": [ses], "session_index": ["1"]}).to_csv(
        root / f"sub-{pid}" / "sessions.tsv", sep="\t", index=False
    )
    return root / f"sub-{pid}"


def test_recording_ids_resolve_to_filestems_and_remove_audio_and_features(tmp_path):
    """Removal by recording_id uses the filestem mechanism, so features go too, even for a
    recording whose task is not included (its audio loop skips it before any ID check)."""
    import torch

    pdir = _sidecar_tree(tmp_path / "in", recordings=(("rec-A", "free-speech-1"), ("rec-B", "noisy-sounds-2")))
    audio = pdir / "ses-S1" / "audio"
    for task in ("free-speech-1", "noisy-sounds-2"):
        torch.save({"opensmile": {"x": 1}}, audio / f"sub-p1_ses-S1_task-{task}_features.pt")
    stems = BIDSDataset._filestems_for_recording_ids(tmp_path / "in", ["p1"], {"rec-a"})
    assert stems == ["sub-p1_ses-S1_task-free-speech-1"]
    out = tmp_path / "out"
    BIDSDataset._deidentify_participant_files(
        pdir, out, {"p1": "900001"}, stems, ["noisy-sounds-*"],
        skip_audio=True, skip_audio_features=False,
    )
    names = sorted(p.name for p in out.rglob("*") if p.is_file())
    assert not any("free-speech" in n for n in names), names
    assert any(n.endswith("noisy-sounds-2.json") for n in names)
    assert any("noisy-sounds-2_features" in n for n in names)

def test_filestem_list_from_another_registration_warns(tmp_path, caplog):
    import logging

    with caplog.at_level(logging.INFO):
        BIDSDataset._report_exclusion_coverage(
            tmp_path, ["sub-001js_ses-x_task-passage-10"], set(), {"a-uuid-participant"}
        )
    assert any(r.levelno == logging.WARNING and "None match" in r.getMessage() for r in caplog.records)


def test_task_matcher_exact_glob_and_regex():
    from b2aiprep.prepare.utils import TaskMatcher

    m = TaskMatcher(["Noisy-Sounds-1", "identifying-pictures-*", "picture-description",
                     "Repeating Words *", "re:role-naming-tasks-sounds-(days|months)", "conversation-*"])
    assert "noisy-sounds-1" in m and "Noisy Sounds 1" in m
    assert "Identifying-Pictures-35" in m and "identifying-pictures" not in m
    assert "picture-description" in m and "picture-description-2" not in m and "picture-28" not in m
    assert "repeating-words-bad" in m
    assert "Role-Naming-Tasks-Sounds-Days" in m and "role-naming-tasks-sounds-numbers" not in m
    assert "Conversation-(6-plus)-favorite-food" in m
    assert not TaskMatcher([]) and TaskMatcher(["re:x"])


def test_deidentify_includes_tasks_by_pattern(tmp_path):
    pdir = _sidecar_tree(tmp_path / "in", recordings=(("rec-A", "identifying-pictures-35"), ("rec-B", "reading-passage-3")))
    out = tmp_path / "out"
    BIDSDataset._deidentify_participant_files(
        pdir, out, {"p1": "900001"}, [], ["identifying-pictures-*"],
        skip_audio=True, skip_audio_features=True,
    )
    written = [p.name for p in out.rglob("*.json")]
    assert len(written) == 1 and "identifying-pictures-35" in written[0]


def test_anchor_that_leaves_real_dates_is_refused():
    df = _frame([("a", "Session", "MIT", "2025-06-02T16:00:00Z", None, "2019-02-28")])
    with pytest.raises(ValueError, match="would not be shifted"):
        shift_dates(df, ["session_started_at", "surgery_date"], datetime.date(2025, 6, 1))


def test_anchor_close_to_real_dates_warns(caplog):
    import logging

    df = _frame([("a", "Session", "MIT", "2024-07-01T16:00:00Z", None, None)])
    with caplog.at_level(logging.WARNING):
        shift_dates(df, ["session_started_at"], datetime.date(2027, 1, 1))
    assert any("years from the nearest real first session" in r.getMessage() for r in caplog.records)


def test_numeric_and_unhyphenated_zip_codes_resolve():
    from b2aiprep.prepare.date_shift import postal_code_timezone

    assert postal_code_timezone(2139.0) == "America/New_York"   # numeric column, leading zero lost
    assert postal_code_timezone(90210) == "America/Los_Angeles"
    assert postal_code_timezone("902101234") == "America/Los_Angeles"


def test_unknown_table_refuses_instead_of_keeping_everything():
    field_map = pd.DataFrame([{"schema_name": "demographics", "column_name": "zipcode", "disposition": "internal"}])
    df = pd.DataFrame({"participant_id": ["p"], "zipcode": ["02139"]})
    with pytest.raises(ValueError, match="no rows in the field map"):
        BIDSDataset._drop_columns_by_disposition(df, field_map, schema_name="renamed_table")


def test_deidentified_sessions_tsv_keeps_only_written_sessions(tmp_path):
    pdir = _sidecar_tree(tmp_path / "in", recordings=(("rec-A", "noisy-sounds-1"),))
    (pdir / "ses-S2" / "audio").mkdir(parents=True)
    (pdir / "ses-S2" / "audio" / "sub-p1_ses-S2_task-free-speech-1_recording-metadata.json").write_text(
        json.dumps({"record_id": "p1", "recording_id": "REC-C", "session_id": "S2"})
    )
    pd.DataFrame({"record_id": ["p1", "p1"], "session_id": ["S1", "S2"], "session_index": ["1", "2"],
                  "session_status": ["Completed"] * 2}).to_csv(
        pdir / "sessions.tsv", sep="\t", index=False
    )
    out = tmp_path / "out"
    BIDSDataset._deidentify_participant_files(
        pdir, out, {"p1": "900001"}, [], ["noisy-sounds-*"],
        skip_audio=True, skip_audio_features=True,
    )
    ses = pd.read_csv(out / "sub-900001" / "sub-900001_sessions.tsv", sep="\t", dtype=str)
    assert list(ses["session_id"]) == ["01"]


def test_participant_with_only_feature_output_is_kept(tmp_path):
    """Recordings whose task audio is not released still publish stripped features."""
    import torch

    pdir = _sidecar_tree(tmp_path / "in", recordings=(("rec-A", "free-speech-1"),))
    torch.save({"opensmile": {"x": 1}}, pdir / "ses-S1" / "audio" / "sub-p1_ses-S1_task-free-speech-1_features.pt")
    out = tmp_path / "out"
    labels, _, _ = BIDSDataset._deidentify_participant_files(
        pdir, out, {"p1": "900001"}, [], ["noisy-sounds-*"],
        skip_audio=False, skip_audio_features=False,
    )
    assert labels == {"S1": "01"}
    assert list(out.rglob("*free-speech-1_features.pt"))
