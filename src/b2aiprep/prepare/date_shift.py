"""Per-participant date shifting at ingest.

Every date the field map marks ``date_shift=YES`` is moved by a whole number of weeks chosen per
participant, so that the participant's earliest session lands within three days of an anchor date
supplied at run time. The anchor is never stored in code: anyone holding the code and a shifted
table must not be able to recover real dates.

Shifted timestamps keep the participant's local wall-clock time and the UTC offset that was in
force at the *real* moment, e.g. ``2100-01-05T13:34:41-05:00``. Local time of day and elapsed
intervals are therefore exact, including across daylight-saving changes. The offset reveals the
season of the real date, which is acceptable because every shifted column is
``disposition=internal`` and stripped at deidentification.

Each timestamp is localized in the time zone where its session happened. A session a data
collector ran happened at the site. A self-administered session (any row of it with
``<prefix>_via == "Participant"``) happened wherever the participant was, taken from their
postal code, else their state or province when that region has a single zone, else the site
(with a warning). The postal and region tables are built by
``scripts/build_postal_code_timezones.py`` (sources and licences there).

Values that cannot be parsed, and every date of a participant with no session start time or no
resolvable time zone, are blanked. A real date never passes through unshifted.
"""

import csv
import datetime
import functools
import logging
import re
import typing as t
from collections import Counter
from importlib.resources import files
from zoneinfo import ZoneInfo

import pandas as pd

_LOGGER = logging.getLogger(__name__)

SITE_TIMEZONES = {
    "MIT": "America/New_York",
    "USF": "America/New_York",
    "WCM": "America/New_York",
    "VUMC": "America/Chicago",
    "Mt. Sinai": "America/Toronto",
    "SickKids": "America/Toronto",
}

SESSION_INSTRUMENT = "Session"
SELF_ADMINISTERED = "Participant"

# Participant location columns; adult and pediatric exports name them differently.
POSTAL_CODE_COLUMNS = ("zipcode", "peds_zipcode")
REGION_COLUMNS = ("state_province", "peds_state_province")

_US_ZIP = re.compile(r"^(\d{5})(-\d{4})?$")
_CA_POSTAL = re.compile(r"^([A-Z]\d[A-Z])\s*(\d[A-Z]\d)?$")

# Values RedCap writes into timestamp columns that mean "no value".
_EMPTY_MARKERS = {"", "[not completed]"}

_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Every timestamp format seen in the 2026-09-04 adult and pediatric exports. All are UTC:
# the ones without a trailing Z were checked against the same session's Z-suffixed acoustic
# task times.
_UTC_FORMATS = (
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%d-T%H:%M:%SZ",  # stray hyphen, 2 values in ef_*
    "%Y%m%dT%H:%M:%S.%fZ",  # pediatric records converted from ReproSchema
    "%Y%m%d%H:%M:%S.%f",  # pediatric session times converted from ReproSchema
    "%m/%d/%Y %H:%M:%S",  # 38 adult session_started_at values
    "%m/%d/%Y %I:%M:%S %p",
)


def parse_utc_timestamp(value: t.Any) -> t.Optional[datetime.datetime]:
    """Parse a RedCap timestamp into an aware UTC datetime, or None if it is not one."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    for fmt in _UTC_FORMATS:
        try:
            return datetime.datetime.strptime(text, fmt).replace(tzinfo=datetime.timezone.utc)
        except ValueError:
            continue
    return None


def parse_date(value: t.Any) -> t.Optional[datetime.date]:
    """Parse a ``YYYY-MM-DD`` date, or None if it is not one."""
    if not isinstance(value, str) or not _DATE_ONLY.match(value.strip()):
        return None
    try:
        return datetime.date.fromisoformat(value.strip())
    except ValueError:
        return None


def offset_weeks(anchor: datetime.date, first_local_date: datetime.date) -> int:
    """Whole weeks that move *first_local_date* to within three days of *anchor*.

    Rounding to the nearest week keeps the day of the week.
    """
    return round((anchor - first_local_date).days / 7)


def shift_timestamp(real_utc: datetime.datetime, tz: ZoneInfo, weeks: int) -> str:
    """Shift by *weeks* in local wall-clock time, keeping the real moment's UTC offset."""
    local = real_utc.astimezone(tz)
    shifted = local.replace(tzinfo=None) + datetime.timedelta(weeks=weeks)
    return shifted.replace(tzinfo=datetime.timezone(local.utcoffset())).isoformat(timespec="seconds")


def _column(df: pd.DataFrame, name: str) -> pd.Series:
    """*name* from *df*, or an all-missing column when the export does not have it."""
    return df[name] if name in df.columns else pd.Series(None, index=df.index, dtype=object)


def _is_empty(value: t.Any) -> bool:
    return value is None or (not isinstance(value, str) and pd.isna(value)) or (
        isinstance(value, str) and value.strip() in _EMPTY_MARKERS
    )


@functools.lru_cache(maxsize=1)
def _postal_timezones() -> t.Dict[t.Tuple[str, str], str]:
    path = files("b2aiprep").joinpath("prepare", "resources", "postal_code_timezones.csv")
    with path.open("r", encoding="utf-8") as fp:
        return {(r["country"], r["postal_code"]): r["timezone"] for r in csv.DictReader(fp)}


@functools.lru_cache(maxsize=1)
def _region_timezones() -> t.Dict[str, str]:
    path = files("b2aiprep").joinpath("prepare", "resources", "region_timezones.csv")
    with path.open("r", encoding="utf-8") as fp:
        return {r["region"]: r["timezone"] for r in csv.DictReader(fp)}


def postal_code_timezone(value: t.Any) -> t.Optional[str]:
    """Zone for a US ZIP (5 digits, optional +4) or Canadian postal code / FSA."""
    if not isinstance(value, str):
        return None
    text = value.strip().upper()
    if text.isdigit() and 3 <= len(text) < 5:
        text = text.zfill(5)  # leading zeros lost to a numeric read
    us = _US_ZIP.match(text)
    if us:
        return _postal_timezones().get(("US", us.group(1)))
    ca = _CA_POSTAL.match(text)
    if ca:
        return _postal_timezones().get(("CA", ca.group(1)))
    return None


def region_timezone(value: t.Any) -> t.Optional[str]:
    """Zone for a state or province code, only when the whole region is one zone."""
    if not isinstance(value, str):
        return None
    return _region_timezones().get(value.strip().upper())


def home_timezones(df: pd.DataFrame) -> t.Dict[str, t.Tuple[str, str]]:
    """Participant -> (zone, source) from their reported postal code, else state/province."""
    out: t.Dict[str, t.Tuple[str, str]] = {}
    for columns, resolve, source in (
        (POSTAL_CODE_COLUMNS, postal_code_timezone, "postal_code"),
        (REGION_COLUMNS, region_timezone, "region"),
    ):
        for col in columns:
            if col not in df.columns:
                continue
            for record_id, value in df.loc[df[col].notna(), ["record_id", col]].itertuples(index=False):
                if record_id in out:
                    continue
                zone = resolve(value)
                if zone:
                    out[record_id] = (zone, source)
    return out


def _session_links(df: pd.DataFrame) -> t.Tuple[pd.Series, t.Set[str]]:
    """Session id of every row, and the sessions with any self-administered row.

    A row names its session in ``session_id`` (Session rows) or ``<prefix>_session_id``, and who
    administered it in the matching ``<prefix>_via``.
    """
    row_session = pd.Series(None, index=df.index, dtype=object)
    self_administered: t.Set[str] = set()
    for col in df.columns:
        if not col.endswith("session_id"):
            continue
        present = df[col].notna() & row_session.isna()
        if col == "session_id":
            present &= df["redcap_repeat_instrument"] == SESSION_INSTRUMENT
        row_session[present] = df.loc[present, col]
        via = col[: -len("session_id")] + "via"
        if via in df.columns:
            self_administered |= set(df.loc[df[via] == SELF_ADMINISTERED, col].dropna())
    return row_session, self_administered


def session_timezones(
    df: pd.DataFrame,
) -> t.Tuple[t.Dict[str, ZoneInfo], Counter, t.Dict[str, str]]:
    """Session -> zone where it happened, counts by source, and sessions left without a zone."""
    sessions = df.loc[df["redcap_repeat_instrument"] == SESSION_INSTRUMENT]
    _, self_administered = _session_links(df)
    homes = home_timezones(df)
    zones: t.Dict[str, ZoneInfo] = {}
    sources: Counter = Counter()
    unresolved: t.Dict[str, str] = {}
    for record_id, session_id, site in zip(
        sessions["record_id"], _column(sessions, "session_id"), _column(sessions, "session_site")
    ):
        site_zone = SITE_TIMEZONES.get(site) if isinstance(site, str) else None
        if session_id in self_administered:
            home = homes.get(record_id)
            if home:
                zones[session_id] = ZoneInfo(home[0])
                sources[f"self_administered_{home[1]}"] += 1
            elif site_zone:
                zones[session_id] = ZoneInfo(site_zone)
                sources["self_administered_site_fallback"] += 1
            else:
                unresolved[session_id] = "self-administered, no location and no known site"
        elif site_zone:
            zones[session_id] = ZoneInfo(site_zone)
            sources["site"] += 1
        else:
            unresolved[session_id] = f"no known site ({site!r})"
    if sources["self_administered_site_fallback"]:
        _LOGGER.warning(
            "%d self-administered session(s) have no usable participant location; used the "
            "site's time zone, which may not be where the participant was.",
            sources["self_administered_site_fallback"],
        )
    return zones, sources, unresolved


def participant_offsets(
    df: pd.DataFrame, anchor: datetime.date, zones: t.Dict[str, ZoneInfo]
) -> t.Tuple[t.Dict[str, int], t.Dict[str, str]]:
    """Offset in weeks per participant, and the reason for each one left out.

    Anchored on the participant's earliest session start, in that session's local time.
    """
    sessions = df.loc[df["redcap_repeat_instrument"] == SESSION_INSTRUMENT]
    offsets: t.Dict[str, int] = {}
    skipped: t.Dict[str, str] = {}
    for record_id, rows in sessions.groupby("record_id"):
        starts = []
        for session_id, value in zip(_column(rows, "session_id"), _column(rows, "session_started_at")):
            ts = parse_utc_timestamp(value)
            if ts is not None and session_id in zones:
                starts.append((ts, zones[session_id]))
        if not starts:
            skipped[record_id] = "no session with a parseable start time and a time zone"
            continue
        first, zone = min(starts, key=lambda pair: pair[0])
        offsets[record_id] = offset_weeks(anchor, first.astimezone(zone).date())
    return offsets, skipped


def shift_dates(
    df: pd.DataFrame, date_columns: t.Iterable[str], anchor: t.Optional[datetime.date]
) -> t.Tuple[pd.DataFrame, dict]:
    """Return a copy of *df* with every *date_columns* value shifted or blanked, and a report.

    With no *anchor*, every value is blanked: the caller asked for no dates.
    """
    df = df.copy()
    columns = [c for c in date_columns if c in df.columns]
    zone_sources: Counter = Counter()
    unresolved: t.Dict[str, str] = {}
    if anchor is None:
        offsets, skipped, zones = {}, {}, {}
    else:
        zones, zone_sources, unresolved = session_timezones(df)
        offsets, skipped = participant_offsets(df, anchor, zones)

    # A row is localized in its own session's zone; rows tied to no session (the participant's
    # base row) use the zone of the participant's first session.
    row_session, _ = _session_links(df)
    sessions = df.loc[df["redcap_repeat_instrument"] == SESSION_INSTRUMENT]
    first_zone: t.Dict[str, t.Tuple[datetime.datetime, ZoneInfo]] = {}
    for record_id, session_id, value in zip(
        sessions["record_id"], _column(sessions, "session_id"), _column(sessions, "session_started_at")
    ):
        ts = parse_utc_timestamp(value)
        if ts is None or session_id not in zones:
            continue
        best = first_zone.get(record_id)
        if best is None or ts < best[0]:
            first_zone[record_id] = (ts, zones[session_id])
    row_zone = [
        zones[sid] if isinstance(sid, str) and sid in zones
        else (first_zone[rid][1] if rid in first_zone else None)
        for rid, sid in zip(df["record_id"], row_session)
    ]

    all_ids = set(df["record_id"].dropna())
    no_offset = all_ids - set(offsets)
    stats: t.Dict[str, Counter] = {}
    for col in columns:
        counts = Counter()
        out = []
        for record_id, zone, value in zip(df["record_id"], row_zone, df[col]):
            if _is_empty(value):
                out.append(None)
                continue
            weeks = offsets.get(record_id)
            if weeks is None:
                counts["blanked_no_offset"] += 1
                out.append(None)
                continue
            ts = parse_utc_timestamp(value)
            if ts is not None:
                if zone is None:
                    counts["blanked_no_timezone"] += 1
                    out.append(None)
                    continue
                out.append(shift_timestamp(ts, zone, weeks))
                counts["shifted_timestamp"] += 1
                continue
            day = parse_date(value)
            if day is not None:
                out.append((day + datetime.timedelta(weeks=weeks)).isoformat())
                counts["shifted_date"] += 1
                continue
            counts["blanked_unparseable"] += 1
            out.append(None)
        df[col] = pd.Series(out, index=df.index, dtype=object)
        stats[col] = counts

    unparseable = {c: n["blanked_unparseable"] for c, n in stats.items() if n["blanked_unparseable"]}
    if unparseable:
        _LOGGER.warning("Blanked unparseable date values (column: count): %s", unparseable)
    if anchor is None and columns:
        _LOGGER.warning("No date-shift anchor given; blanked all %d date columns.", len(columns))
    elif no_offset:
        _LOGGER.warning(
            "%d participant(s) have no date offset; their dates were blanked. Reasons: %s",
            len(no_offset),
            dict(Counter(skipped.get(r, "no Session row") for r in no_offset)),
        )

    report = {
        "anchor_given": anchor is not None,
        "participants": len(all_ids),
        "participants_shifted": len(set(offsets) & all_ids),
        "participants_without_offset": {
            r: skipped.get(r, "no Session row") for r in sorted(no_offset)
        },
        "session_timezone_sources": dict(zone_sources),
        "sessions_without_timezone": unresolved,
        "columns": {c: dict(n) for c, n in stats.items()},
    }
    return df, report
