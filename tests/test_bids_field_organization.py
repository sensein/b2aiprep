"""Guards for resources/bids_field_organization.csv.

This CSV is the sole source of truth for which RedCap/ReproSchema fields reach
``phenotype/``: ``BIDSDataset._construct_phenotype_from_reproschema`` iterates the
CSV, not the reproschema, so a field with no row here is silently absent from the
release, and its ``disposition`` decides what a release publishes. Every bump of the vendored
``src/b2aiprep/redcap2rs`` snapshot must therefore be accompanied by rows here.
"""

import csv
import json
from importlib.resources import files

import pandas as pd
import pytest

from b2aiprep.prepare import dataset as dataset_module
from b2aiprep.prepare.dataset import BIDSDataset, derived_field_specs

VALID_COLUMN_TYPES = {"VARIABLE", "CHECKBOX_OPTION"}
VALID_YES_NO = {"YES", "NO"}

# Shipped-field-map dispositions that deidentify, settings and CLI tests rely on.
TEST_FIELD_DISPOSITIONS = {
    ("confounders", "ph_walking"): ("release", ""),
    ("confounders", "other_voice_activity"): ("review", ""),
    ("confounders", "voice_activity_v2___attorney"): ("release", ""),
    ("confounders", "ever_alcohol_rehab"): ("release", "controlled"),
    ("pediatric_demographics", "peds_gender_identity"): ("release", ""),
    ("demographics", "state_province"): ("release", ""),
}


def _resource(*parts):
    resource = files("b2aiprep")
    for part in parts:
        resource = resource.joinpath(part)
    return resource


@pytest.fixture(scope="module")
def reorg_rows():
    path = _resource("prepare", "resources", "bids_field_organization.csv")
    with path.open("r", encoding="utf-8") as fp:
        return list(csv.DictReader(fp))


@pytest.fixture(scope="module")
def reachable_elements():
    """Data elements reachable from the protocol ``ui.order``.

    Mirrors ``BIDSDataset._load_reproschema``: activities not listed in the
    protocol order are never loaded, so their items cannot be resolved even
    though the directory may still exist in the vendored snapshot.
    """
    redcap2rs = _resource("redcap2rs")
    schema_file = redcap2rs.joinpath("b2ai-redcap2rs", "b2ai-redcap2rs_schema")
    if not schema_file.is_file():
        schema_file = redcap2rs.joinpath("b2ai-redcap2rs_schema")
    protocol = json.loads(schema_file.read_text(encoding="utf-8"))

    elements = {}
    for rel_path in protocol["ui"]["order"]:
        activity = [part for part in rel_path.split("/") if part != ".."][1]
        activity_file = redcap2rs.joinpath("activities", activity, f"{activity}_schema")
        if not activity_file.is_file():
            pytest.fail(f"protocol order references a missing activity: {activity}")
        schema = json.loads(activity_file.read_text(encoding="utf-8"))
        for item in schema.get("ui", {}).get("addProperties", []):
            elements[item["variableName"]] = schema["id"]
    return elements


def test_every_reachable_element_has_a_row(reorg_rows, reachable_elements):
    """No data element may be dropped from the release by omission.

    Rows may be either the element itself or, for RedCap checkbox fields, the
    ``field___option`` expansions. Add a ``disposition=drop`` row to exclude a field
    deliberately -- an absent row is indistinguishable from an oversight.
    """
    covered = set()
    for row in reorg_rows:
        source = row["column_name_source"].strip()
        covered.add(source)
        covered.add(source.rsplit("___", 1)[0])

    missing = sorted(set(reachable_elements) - covered)
    assert not missing, (
        f"{len(missing)} data element(s) in the vendored reproschema have no row in "
        f"bids_field_organization.csv and would be silently excluded from phenotype/: "
        f"{missing}"
    )


# RedCap synthesizes these per instrument; they are never data-dictionary variables, so
# reproschema (which is generated from the dictionary) cannot describe them. They are the only
# columns allowed to reach the phenotype writer without a reproschema element.
REDCAP_GENERATED_SUFFIXES = ("_complete", "_timestamp")
REDCAP_STRUCTURAL_COLUMNS = {
    "redcap_repeat_instrument",
    "redcap_repeat_instance",
    "redcap_survey_identifier",
}


def _is_redcap_generated(column_name_source):
    return (
        column_name_source in REDCAP_STRUCTURAL_COLUMNS
        or column_name_source.endswith(REDCAP_GENERATED_SUFFIXES)
    )


def test_every_active_row_resolves_or_is_a_redcap_generated_column(reorg_rows, reachable_elements):
    """Active rows resolve to a reproschema element, or are RedCap-generated columns.

    ``_construct_phenotype_from_reproschema`` used to raise KeyError on an unresolved row;
    it now falls back to ``_synthetic_data_element`` and describes the column from this CSV
    alone. That fallback exists for the columns RedCap invents per instrument
    (``<form>_complete``, ``<form>_timestamp``) and its structural columns -- never for a
    typo or a row left behind by a renamed field, which would otherwise be published with a
    generic description and no termURL instead of failing loudly. Retire stale rows with
    ``disposition=drop``.
    """
    unresolved = sorted(
        row["column_name_source"]
        for row in reorg_rows
        if row["disposition"] != "drop"
        and row.get("source", "").strip().lower() not in ("pipeline", "supplement")
        and row["column_name_source"].rsplit("___", 1)[0] not in reachable_elements
        and not _is_redcap_generated(row["column_name_source"])
    )
    assert not unresolved, (
        f"{len(unresolved)} active row(s) reference elements absent from the vendored "
        f"reproschema protocol order and are not RedCap-generated columns: {unresolved}"
    )


def test_redcap_generated_rows_are_described_by_the_csv(reorg_rows):
    """Their description here is the only one that will ever exist.

    No upstream defines these columns, so ``_synthetic_data_element`` copies this CSV's
    description verbatim into the published data dictionary. An empty one would ship an
    entry saying nothing.
    """
    undescribed = sorted(
        row["column_name_source"]
        for row in reorg_rows
        if row["disposition"] != "drop"
        and _is_redcap_generated(row["column_name_source"])
        and not row["description"].strip()
    )
    assert not undescribed, f"RedCap-generated rows with no description: {undescribed}"


def test_active_rows_are_well_formed(reorg_rows):
    """Invariants relied on by the phenotype writer."""
    problems = []
    for index, row in enumerate(reorg_rows, start=2):  # 1-based, header is line 1
        if row["delete"].strip().upper() not in VALID_YES_NO:
            problems.append(f"line {index}: delete={row['delete']!r} not in {VALID_YES_NO}")
        if row["column_type"].strip() not in VALID_COLUMN_TYPES:
            problems.append(
                f"line {index}: column_type={row['column_type']!r} not in {VALID_COLUMN_TYPES}"
            )
        if not row["description"].strip():
            problems.append(f"line {index}: empty description for {row['column_name_source']}")
        if row["disposition"] == "drop":
            continue
        # Rows with no schema_name are never written to phenotype/ (e.g. the structural
        # redcap_repeat_instrument); that is only allowed for internal rows that are not dates.
        if (
            row["disposition"] == "internal"
            and row["date_shift"].strip().upper() != "YES"
            and not row["schema_name"].strip()
        ):
            continue
        # group becomes a phenotype/ subdirectory and schema_name a filename
        if not row["schema_name"].strip():
            problems.append(f"line {index}: active row missing schema_name")
        if not row["group"].strip():
            problems.append(f"line {index}: active row missing group")
    assert not problems, "\n".join(problems)


def test_checkbox_options_declare_a_base_row(reorg_rows):
    """A ``field___option`` row needs a row for ``field`` too.

    ``_construct_phenotype_from_reproschema`` reads the base element's metadata to
    build the per-option data element, and skips the base row once the expansions
    are found in the export.
    """
    sources = {row["column_name_source"].strip() for row in reorg_rows}
    orphans = sorted(
        source
        for source in sources
        if "___" in source and source.rsplit("___", 1)[0] not in sources
    )
    assert not orphans, f"checkbox option rows with no base row: {orphans}"


def test_output_column_names_are_unique_per_schema(reorg_rows):
    """Duplicate output names collapse columns; the writer raises on this."""
    seen = {}
    collisions = []
    for row in reorg_rows:
        if row["disposition"] == "drop":
            continue
        key = (row["schema_name"].strip(), row["column_name"].strip())
        if key in seen:
            collisions.append(f"{key} from {seen[key]} and {row['column_name_source']}")
        seen[key] = row["column_name_source"]
    assert not collisions, "duplicate (schema_name, column_name): " + "; ".join(collisions)


def test_group_is_consistent_within_a_schema(reorg_rows):
    """``group`` is written per-schema, so mixed values are order-dependent.

    ``_construct_phenotype_from_reproschema`` assigns ``payload["group"]`` on every
    column of a schema, so the last row processed silently wins.
    """
    groups = {}
    for row in reorg_rows:
        if row["disposition"] == "drop" or not row["schema_name"].strip():
            continue
        groups.setdefault(row["schema_name"].strip(), set()).add(row["group"].strip())
    inconsistent = {name: sorted(v) for name, v in groups.items() if len(v) > 1}
    assert not inconsistent, f"schemas with more than one group: {inconsistent}"


def test_schema_name_source_matches_the_reproschema_activity_id():
    """``schema_name_source`` should name the activity the element actually comes from.

    It is documentation rather than control flow: ``_construct_phenotype_from_reproschema``
    resolves the source schema from the element itself, and only uses this column for a
    repeat-instrument row filter that cannot fire for non-repeating forms (RedCap leaves
    ``redcap_repeat_instrument`` blank for those, and ``RedCapDataset`` fills the blanks
    with ``"Participant"``). Most of the CSV predates the ``d_*``/``q_*`` activity-id
    renaming, so this only pins rows added for redcap 4.9.1 against a further drift.
    """
    path = _resource("prepare", "resources", "bids_field_organization.csv")
    with path.open("r", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp))

    expected = {
        "d_neuro_ataxia": "d_neuro_ataxia_schema",
        "d_neuro_essential_tremor": "d_neuro_essential_tremor_schema",
        "prolific_information": "prolific_information_schema",
    }
    for activity, source in expected.items():
        activity_file = _resource("redcap2rs", "activities", activity, f"{activity}_schema")
        assert activity_file.is_file(), f"missing vendored activity {activity}"
        assert json.loads(activity_file.read_text(encoding="utf-8"))["id"] == source
        assert any(row["schema_name_source"] == source for row in rows), (
            f"no rows carry schema_name_source={source}"
        )


VALID_DISPOSITIONS = {"drop", "internal", "release", "review"}


def test_disposition_values_are_valid(reorg_rows):
    """Every row must carry one of the four defined dispositions.

    A typo or an empty cell would silently miscategorise a field -- e.g. a misspelled
    'relase' would be treated as unknown by any filter, and the field's handling would
    depend on which branch the code takes for unrecognised values.
    """
    problems = []
    for index, row in enumerate(reorg_rows, start=2):
        d = row.get("disposition", "").strip()
        if d not in VALID_DISPOSITIONS:
            problems.append(
                f"line {index}: disposition={d!r} for {row['column_name_source']} "
                f"(expected one of {sorted(VALID_DISPOSITIONS)})"
            )
    assert not problems, "\n".join(problems)


# RedCap app instruments record participant timing in these columns. The vendored ReproSchema
# types them as xsd:string, so the suffix is the only marker the snapshot carries.
APP_TIMESTAMP_SUFFIXES = ("_started_at", "_completed_at", "_created_at")
DATE_VALUE_TYPES = ("xsd:date", "xsd:datetime")


def _vendored_date_items():
    """Variable names of every vendored ReproSchema item typed as a date or datetime."""
    names = set()
    for activity in _resource("redcap2rs", "activities").iterdir():
        items = activity.joinpath("items")
        if not items.is_dir():
            continue
        for item in items.iterdir():
            try:
                schema = json.loads(item.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            options = schema.get("responseOptions")
            value_type = options.get("valueType", "") if isinstance(options, dict) else ""
            if isinstance(value_type, list):
                value_type = " ".join(value_type)
            if any(t in value_type.lower() for t in DATE_VALUE_TYPES):
                names.add(item.name)
    return names


def test_date_shift_values_are_valid(reorg_rows):
    problems = [
        (row["column_name_source"], row["date_shift"])
        for row in reorg_rows
        if row["date_shift"].strip().upper() not in VALID_YES_NO
    ]
    assert not problems, f"date_shift must be YES or NO: {problems[:10]}"


def test_date_shift_fields_are_internal(reorg_rows):
    """Shifted dates are kept for internal derivations only and never published."""
    mismatches = [
        (row["column_name_source"], row["disposition"])
        for row in reorg_rows
        if row["date_shift"].strip().upper() == "YES" and row["disposition"] != "internal"
    ]
    assert not mismatches, f"date_shift=YES but disposition!=internal: {mismatches}"


def test_dates_and_app_timestamps_are_shifted_or_dropped(reorg_rows):
    """A date the data dictionary declares, or an app timestamp column, must be shifted at
    ingest or never ingested. Anything else carries a real calendar date into the BIDS tree."""
    date_items = _vendored_date_items()
    assert date_items, "no xsd:date items found; the vendored snapshot layout changed"
    unshifted = sorted(
        (row["column_name_source"], row["disposition"])
        for row in reorg_rows
        if (row["column_name_source"] in date_items or row["column_name_source"].endswith(APP_TIMESTAMP_SUFFIXES))
        and row["disposition"] != "drop"
        and row["date_shift"].strip().upper() != "YES"
    )
    assert not unshifted, f"dates or timestamps neither shifted nor dropped: {unshifted}"


def test_redcap_timestamps_are_dropped(reorg_rows):
    """``<form>_timestamp`` is set when staff complete a form and carries no time zone."""
    kept = sorted(
        row["column_name_source"]
        for row in reorg_rows
        if row["column_name_source"].endswith("_timestamp") and row["disposition"] != "drop"
    )
    assert not kept, f"RedCap _timestamp columns must be drop: {kept}"


def test_sidecar_recording_keys_match_recording_table(reorg_rows):
    """A recording_* key has the same disposition in a sidecar as in recording.tsv."""
    by_table = {}
    for row in reorg_rows:
        by_table.setdefault(row["schema_name"], {})[row["column_name"]] = row["disposition"]
    sidecar, recording = by_table["audio_sidecar"], by_table["recording"]
    shared = sorted(set(sidecar) & set(recording))
    assert shared, "no recording_* keys shared between audio_sidecar and recording"
    assert {k: sidecar[k] for k in shared} == {k: recording[k] for k in shared}


def test_access_tier_values_are_valid(reorg_rows):
    bad = [(r["column_name_source"], r["access_tier"]) for r in reorg_rows
           if r["access_tier"].strip().lower() not in ("", "controlled")]
    assert not bad, f"access_tier must be blank or 'controlled': {bad}"


def test_loading_refuses_an_access_tier_typo(monkeypatch, tmp_path):
    """Anything but 'controlled' would mean every tier, so a typo would publish the field to both."""
    df = pd.read_csv(_resource("prepare", "resources", "bids_field_organization.csv"), dtype={"access_tier": str})
    df.loc[df.index[0], "access_tier"] = "controled"
    (tmp_path / "bids_field_organization.csv").write_text(df.to_csv(index=False))
    monkeypatch.setattr(dataset_module, "files", lambda _: tmp_path)
    with pytest.raises(ValueError, match="access_tier must be blank or 'controlled'.*controled"):
        BIDSDataset._load_reorganization_file(exclude_dropped=False)


def test_dispositions_other_tests_rely_on(reorg_rows):
    """If this fails, a field-map change also changes what those tests exercise; pick other fields there."""
    rows = {(r["schema_name"], r["column_name"]): (r["disposition"], r["access_tier"]) for r in reorg_rows}
    assert {k: rows.get(k) for k in TEST_FIELD_DISPOSITIONS} == TEST_FIELD_DISPOSITIONS


def test_every_computed_column_has_a_derived_field_spec():
    """Each computed phenotype column (source pipeline or supplement) has datatype/choices in
    derived_fields.json, and back."""
    field_map = pd.read_csv(_resource("prepare", "resources", "bids_field_organization.csv"), dtype=str)
    computed = set(field_map.loc[(field_map["source"].isin(["pipeline", "supplement"]))
                                 & (field_map["schema_name"] != "audio_sidecar"), "column_name"])
    specs = derived_field_specs()
    assert computed == set(specs)
    for name, spec in specs.items():
        assert spec["datatype"] in ("xsd:integer", "xsd:string"), name
        assert set(spec) <= {"datatype", "choices", "minValue", "maxValue", "unit", "derivedFrom"}, name
