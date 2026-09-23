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

Values that cannot be parsed, and every date of a participant with no session start time or no
known site, are blanked. A real date never passes through unshifted.
"""

import datetime
import logging
import re
import typing as t
from collections import Counter
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


def _is_empty(value: t.Any) -> bool:
    return value is None or (not isinstance(value, str) and pd.isna(value)) or (
        isinstance(value, str) and value.strip() in _EMPTY_MARKERS
    )


def participant_offsets(
    df: pd.DataFrame, anchor: datetime.date
) -> t.Tuple[t.Dict[str, t.Tuple[int, ZoneInfo]], t.Dict[str, str]]:
    """Offset in weeks and time zone per participant, and the reason for each one left out.

    The offset is anchored on the participant's earliest session start, in the site's local
    time. A participant whose sessions name no known site, or none of whose sessions has a
    parseable start time, gets no offset.
    """
    sessions = df.loc[df["redcap_repeat_instrument"] == SESSION_INSTRUMENT]
    offsets: t.Dict[str, t.Tuple[int, ZoneInfo]] = {}
    skipped: t.Dict[str, str] = {}
    for record_id, rows in sessions.groupby("record_id"):
        sites = {s for s in rows.get("session_site", pd.Series(dtype=object)).dropna()}
        if len(sites) != 1:
            skipped[record_id] = f"{len(sites)} session sites"
            continue
        site = sites.pop()
        if site not in SITE_TIMEZONES:
            skipped[record_id] = f"unknown site {site!r}"
            continue
        tz = ZoneInfo(SITE_TIMEZONES[site])
        starts = [parse_utc_timestamp(v) for v in rows["session_started_at"]]
        starts = [s for s in starts if s is not None]
        if not starts:
            skipped[record_id] = "no parseable session_started_at"
            continue
        first_local = min(starts).astimezone(tz).date()
        offsets[record_id] = (offset_weeks(anchor, first_local), tz)
    return offsets, skipped


def shift_dates(
    df: pd.DataFrame, date_columns: t.Iterable[str], anchor: t.Optional[datetime.date]
) -> t.Tuple[pd.DataFrame, dict]:
    """Return a copy of *df* with every *date_columns* value shifted or blanked, and a report.

    With no *anchor*, every value is blanked: the caller asked for no dates.
    """
    df = df.copy()
    columns = [c for c in date_columns if c in df.columns]
    if anchor is None:
        offsets, skipped = {}, {}
    else:
        offsets, skipped = participant_offsets(df, anchor)

    all_ids = set(df["record_id"].dropna())
    no_offset = all_ids - set(offsets)
    stats: t.Dict[str, Counter] = {}
    for col in columns:
        counts = Counter()
        out = []
        for record_id, value in zip(df["record_id"], df[col]):
            if _is_empty(value):
                out.append(None)
                continue
            entry = offsets.get(record_id)
            if entry is None:
                counts["blanked_no_offset"] += 1
                out.append(None)
                continue
            weeks, tz = entry
            ts = parse_utc_timestamp(value)
            if ts is not None:
                out.append(shift_timestamp(ts, tz, weeks))
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
        "columns": {c: dict(n) for c, n in stats.items()},
    }
    return df, report
