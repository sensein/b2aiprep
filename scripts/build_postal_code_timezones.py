"""Build resources/postal_code_timezones.csv and region_timezones.csv.

``postal_code_timezones.csv``: US ZIP code or Canadian FSA -> IANA time zone.
``region_timezones.csv``: US state / Canadian province -> IANA time zone, only for regions whose
every postal code falls in a single zone (a fallback when no postal code was reported).

Used by date shifting to find where a self-administered session actually happened. Run by
hand when the sources should be refreshed; the output is committed, so the pipeline has no
network or ``timezonefinder`` dependency.

Sources (both assign the zone by point-in-polygon on a postal-code centroid, with
``timezonefinder``):

* US ZIP codes: zip2info (MIT; coordinates from Census ZCTA centroids, public domain, and
  GeoNames, CC BY 4.0), pinned to ``ZIP2INFO_COMMIT``.
* Canadian forward sortation areas (first three characters of a postal code): GeoNames
  ``CA.zip`` (CC BY 4.0, https://www.geonames.org), resolved here.

Usage (``timezonefinder`` is needed only for this script)::

    python scripts/build_postal_code_timezones.py
"""

import ast
import csv
import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

from timezonefinder import TimezoneFinder

ZIP2INFO_COMMIT = "5876e295eb"
ZIP2INFO_DATA = (
    f"https://raw.githubusercontent.com/Three-Ships/zip2info/{ZIP2INFO_COMMIT}/src/zip2info/_data.py"
)
GEONAMES_CA = "https://download.geonames.org/export/zip/CA.zip"
GEONAMES_US = "https://download.geonames.org/export/zip/US.zip"
RESOURCES = Path(__file__).resolve().parents[1] / "src/b2aiprep/prepare/resources"
OUTPUT = RESOURCES / "postal_code_timezones.csv"
REGION_OUTPUT = RESOURCES / "region_timezones.csv"


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    print(f"{url}: {len(payload)} bytes, sha256 {hashlib.sha256(payload).hexdigest()}", file=sys.stderr)
    return payload


def us_zip_timezones() -> dict:
    """ZIP (5 digits) -> zone, from zip2info's generated module (parsed, never imported)."""
    tree = ast.parse(_fetch(ZIP2INFO_DATA).decode())
    values = {
        node.target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    zones, info = values["TIMEZONES"], values["ZIP_INFO"]
    return {f"{zip_code:05d}": zones[tz_index] for zip_code, (tz_index, _, _) in info.items()}


def _geonames_rows(url: str, member: str):
    with zipfile.ZipFile(io.BytesIO(_fetch(url))) as archive:
        text = archive.read(member).decode("utf-8")
    return list(csv.reader(io.StringIO(text), delimiter="\t"))


def canada_fsa_timezones() -> tuple:
    """FSA -> zone from GeoNames FSA centroids, and FSA -> province code."""
    finder = TimezoneFinder()
    zones, provinces = {}, {}
    for row in _geonames_rows(GEONAMES_CA, "CA.txt"):
        fsa, province, lat, lon = row[1].strip().upper(), row[4].strip(), row[9], row[10]
        if not (lat and lon):
            continue
        zone = finder.timezone_at(lat=float(lat), lng=float(lon))
        if zone and zone.startswith("America/"):
            zones[fsa] = zone
            provinces[fsa] = province
    return zones, provinces


def us_zip_states() -> dict:
    """ZIP -> two-letter state code, from GeoNames."""
    return {row[1].strip(): row[4].strip() for row in _geonames_rows(GEONAMES_US, "US.txt") if row[4].strip()}


def single_zone_regions(zones: dict, regions: dict) -> dict:
    """Region -> zone for regions where every postal code with a known zone shares one zone."""
    seen = {}
    for code, region in regions.items():
        if code in zones:
            seen.setdefault(region, set()).add(zones[code])
    return {region: next(iter(found)) for region, found in seen.items() if len(found) == 1}


def main() -> None:
    us_zones = us_zip_timezones()
    ca_zones, ca_provinces = canada_fsa_timezones()
    rows = [("US", code, zone) for code, zone in sorted(us_zones.items())]
    rows += [("CA", code, zone) for code, zone in sorted(ca_zones.items())]
    with open(OUTPUT, "w", newline="") as fp:
        writer = csv.writer(fp, lineterminator="\n")
        writer.writerow(("country", "postal_code", "timezone"))
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUTPUT}", file=sys.stderr)

    regions = [("US", r, z) for r, z in sorted(single_zone_regions(us_zones, us_zip_states()).items())]
    regions += [("CA", r, z) for r, z in sorted(single_zone_regions(ca_zones, ca_provinces).items())]
    with open(REGION_OUTPUT, "w", newline="") as fp:
        writer = csv.writer(fp, lineterminator="\n")
        writer.writerow(("country", "region", "timezone"))
        writer.writerows(regions)
    print(f"wrote {len(regions)} single-zone regions to {REGION_OUTPUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
