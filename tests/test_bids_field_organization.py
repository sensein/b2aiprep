"""Guards for resources/bids_field_organization.csv.

This CSV is the sole source of truth for which RedCap/ReproSchema fields reach
``phenotype/``: ``BIDSDataset._construct_phenotype_from_reproschema`` iterates the
CSV, not the reproschema, so a field with no row here is silently absent from the
release and a field with ``delete=NO`` is published. Every bump of the vendored
``src/b2aiprep/redcap2rs`` snapshot must therefore be accompanied by rows here.
"""

import csv
import json
from importlib.resources import files

import pytest

VALID_COLUMN_TYPES = {"VARIABLE", "CHECKBOX_OPTION"}
VALID_DELETE = {"YES", "NO"}


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
    ``field___option`` expansions. Add a ``delete=YES`` row to exclude a field
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
    ``delete=YES``.
    """
    unresolved = sorted(
        row["column_name_source"]
        for row in reorg_rows
        if row["delete"].strip().upper() != "YES"
        and row.get("source", "").strip().lower() != "pipeline"
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
        if row["delete"].strip().upper() != "YES"
        and _is_redcap_generated(row["column_name_source"])
        and not row["description"].strip()
    )
    assert not undescribed, f"RedCap-generated rows with no description: {undescribed}"


def test_active_rows_are_well_formed(reorg_rows):
    """Invariants relied on by the phenotype writer."""
    problems = []
    for index, row in enumerate(reorg_rows, start=2):  # 1-based, header is line 1
        if row["delete"].strip().upper() not in VALID_DELETE:
            problems.append(f"line {index}: delete={row['delete']!r} not in {VALID_DELETE}")
        if row["column_type"].strip() not in VALID_COLUMN_TYPES:
            problems.append(
                f"line {index}: column_type={row['column_type']!r} not in {VALID_COLUMN_TYPES}"
            )
        if not row["description"].strip():
            problems.append(f"line {index}: empty description for {row['column_name_source']}")
        if row["delete"].strip().upper() == "YES":
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
        if row["delete"].strip().upper() == "YES":
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
        if row["delete"].strip().upper() == "YES":
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


def test_drop_fields_have_delete_yes(reorg_rows):
    """disposition=drop must agree with delete=YES until delete is retired."""
    mismatches = [
        (row["column_name_source"], row["delete"], row["disposition"])
        for row in reorg_rows
        if row["disposition"] == "drop" and row["delete"].strip().upper() != "YES"
    ]
    assert not mismatches, f"drop disposition but delete!=YES: {mismatches[:10]}"


def test_internal_fields_have_delete_yes(reorg_rows):
    """disposition=internal fields are consumed at ingest but never published,
    so delete=YES must hold until the pipeline reads disposition natively."""
    mismatches = [
        (row["column_name_source"], row["delete"], row["disposition"])
        for row in reorg_rows
        if row["disposition"] == "internal" and row["delete"].strip().upper() != "YES"
    ]
    assert not mismatches, f"internal disposition but delete!=YES: {mismatches[:10]}"
