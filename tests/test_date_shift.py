"""Ingest: per-participant date shifting, time zones, and the fields derived from dates
(session index, timing, local hours, days since, state/province)."""

import datetime
import logging

import pandas as pd
import pytest

from b2aiprep.prepare.date_shift import (
    offset_weeks,
    parse_date,
    parse_utc_timestamp,
    postal_code_region,
    postal_code_timezone,
    region_timezone,
    shift_dates,
)


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


@pytest.mark.parametrize("raw, expected", [
    ("2019-02-28", datetime.date(2019, 2, 28)), ("2019-02-30", None), ("2019-02-28T00:00:00Z", None)])
def test_parse_date(raw, expected):
    assert parse_date(raw) == expected


@pytest.mark.parametrize("days_before", range(0, 800, 37))
def test_offset_lands_within_three_days_on_same_weekday(days_before):
    first = ANCHOR - datetime.timedelta(days=days_before + 30000)  # first session ~82 years before the anchor
    shifted = first + datetime.timedelta(weeks=offset_weeks(ANCHOR, first))
    assert abs((shifted - ANCHOR).days) <= 3
    assert shifted.weekday() == first.weekday()


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
    no_zone = "no session with a parseable start time and a time zone"
    assert report["participants_without_offset"] == {"no_site": no_zone, "unknown_site": no_zone,
                                                     "no_session": "no Session row"}


def test_unparseable_and_empty_markers_are_blanked():
    df = _frame([
        ("a", "Session", "MIT", "2024-07-01T16:00:00Z", None, None),
        ("a", "Q", None, None, "[not completed]", "sometime in 2019"),
    ])
    out, report = shift_dates(df, ["phq_9_started_at", "surgery_date"], ANCHOR)
    assert pd.isna(out.loc[1, "phq_9_started_at"]) and pd.isna(out.loc[1, "surgery_date"])
    assert report["columns"]["surgery_date"] == {"blanked_unparseable": 1}
    assert report["columns"]["phq_9_started_at"] == {}  # an empty marker is blanked without being counted


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
    "zipcode, state, via, expected_hour, source",
    [
        ("90210", None, "Participant", 13, "self_administered_postal_code"),   # Los Angeles, PDT
        ("80202-1234", None, "Participant", 14, "self_administered_postal_code"),  # Denver, MDT
        ("M5G 1X5", None, "Participant", 16, "self_administered_postal_code"),  # Toronto, EDT
        (None, "IL", "Participant", 15, "self_administered_region"),            # Chicago, CDT
        (None, "FL", "Participant", 16, "self_administered_site_fallback"),     # FL spans two zones -> site
        (None, None, "Participant", 16, "self_administered_site_fallback"),
        ("90210", None, "Data Collector", 16, "site"),  # in person: the site, wherever the participant lives
    ],
)
def test_session_time_zone_comes_from_the_participant_or_the_site(zipcode, state, via, expected_hour, source, caplog):
    with caplog.at_level(logging.WARNING):
        out, report = shift_dates(_remote_frame(zipcode, state, via), ["session_started_at", "recording_created_at"],
                                  ANCHOR)
    assert _local_hour(out, "session_started_at", 0) == expected_hour
    assert _local_hour(out, "recording_created_at", 1) == expected_hour
    assert report["session_timezone_sources"] == {source: 1}
    if source == "self_administered_site_fallback":
        assert "used the site's time zone" in caplog.text and "r/s1" in caplog.text


@pytest.mark.parametrize("postal_code, zone", [
    ("37203", "America/Chicago"), ("37902", "America/New_York"),   # Nashville / Knoxville: one state, two zones
    ("32501", "America/Chicago"),                                   # Pensacola
    ("P9N 1A1", "America/Winnipeg"),                                # Kenora
    ("2139", "America/New_York"), (2139.0, "America/New_York"),     # leading zero lost (text / numeric column)
    (90210, "America/Los_Angeles"), ("902101234", "America/Los_Angeles"),
])
def test_postal_code_time_zone(postal_code, zone):
    assert postal_code_timezone(postal_code) == zone


@pytest.mark.parametrize("region, zone", [
    ("TN", None), ("ON", None),  # split states and provinces have no single zone
    ("MA", "America/New_York"), ("Massachusetts", "America/New_York"),
])
def test_region_time_zone(region, zone):
    assert region_timezone(region) == zone


@pytest.mark.parametrize("postal_code, region", [
    ("02139", "MA"), (2139.0, "MA"), ("90210-1234", "CA"), ("M5V 3L9", "ON"), ("h2x", "QC"),
    ("12", None), ("not a zip", None),
])
def test_postal_code_region(postal_code, region):
    assert postal_code_region(postal_code) == region


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


def test_anchor_that_leaves_real_dates_is_refused():
    df = _frame([("a", "Session", "MIT", "2025-06-02T16:00:00Z", None, "2019-02-28")])
    with pytest.raises(ValueError, match="would not be shifted"):
        shift_dates(df, ["session_started_at", "surgery_date"], datetime.date(2025, 6, 1))


def test_anchor_close_to_real_dates_warns(caplog):
    df = _frame([("a", "Session", "MIT", "2024-07-01T16:00:00Z", None, None)])
    with caplog.at_level(logging.WARNING):
        shift_dates(df, ["session_started_at"], datetime.date(2027, 1, 1))
    assert any("years from the nearest real first session" in r.getMessage() for r in caplog.records)


















def test_sessions_self_administered_only_in_dropped_rows_use_the_home_time_zone():
    # The only "Participant" row was the microphone check, dropped before ingest (Data Collector here).
    df = _remote_frame(zipcode="90210", via="Data Collector")
    as_site, _ = shift_dates(df, ["session_started_at"], ANCHOR)
    as_home, _ = shift_dates(df, ["session_started_at"], ANCHOR, also_self_administered={"s1"})
    # 2024-07-01 20:00Z: 16:00 at MIT (EDT), 13:00 in Beverly Hills (PDT)
    assert as_site.loc[0, "session_started_at"].endswith("T16:00:00-04:00")
    assert as_home.loc[0, "session_started_at"].endswith("T13:00:00-07:00")








