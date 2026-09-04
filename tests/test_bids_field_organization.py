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


def test_every_active_row_resolves_to_a_reachable_element(reorg_rows, reachable_elements):
    """Active rows must resolve, or ``element_to_schema[column]`` raises KeyError.

    ``_construct_phenotype_from_reproschema`` only reaches that lookup when the
    column is present in the RedCap export, so a stale row is a latent crash
    rather than a guaranteed one. Retire stale rows with ``delete=YES``.
    """
    unresolved = sorted(
        row["column_name_source"]
        for row in reorg_rows
        if row["delete"].strip().upper() != "YES"
        and row["column_name_source"].rsplit("___", 1)[0] not in reachable_elements
    )
    assert not unresolved, (
        f"{len(unresolved)} active row(s) reference elements absent from the vendored "
        f"reproschema protocol order: {unresolved}"
    )


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


@pytest.fixture(scope="module")
def reproschema_choices(reachable_elements):
    """``{element: [choice values]}`` for every reachable element that declares choices."""
    choices = {}
    for element, activity_id in reachable_elements.items():
        activity = activity_id[: -len("_schema")] if activity_id.endswith("_schema") else activity_id
        item_file = _resource("redcap2rs", "activities", activity, "items", element)
        if not item_file.is_file():
            continue
        item = json.loads(item_file.read_text(encoding="utf-8"))
        values = [c.get("value") for c in item.get("responseOptions", {}).get("choices", []) or []]
        if values:
            choices[element] = values
    return choices


def test_checkbox_option_codes_resolve_to_a_reproschema_choice(reorg_rows, reproschema_choices):
    """Every ``field___code`` row must name a choice of its base element.

    RedCap option codes are opaque strings; the redcap→reproschema conversion stores codes such
    as ``2029_7`` as the integer ``20297``, so a code is accepted when it equals a choice value
    directly or, for digit-and-underscore codes, with the underscores removed — the same rule
    ``BIDSDataset._checkbox_choice_matches`` applies when building the phenotype dictionary.
    A row whose code matches nothing can never be populated: the export column is silently
    dropped from the release.

    Limitation: this CSV was generated from the vendored reproschema, so a code the *conversion*
    mangled (``asian_race___20297``) agrees with the reproschema and passes here. Only
    :func:`test_checkbox_option_codes_match_the_redcap_dictionary` catches that class, which is
    how 45 detailed-race columns went missing without any test noticing.
    """
    from b2aiprep.prepare.dataset import BIDSDataset

    unresolved = []
    for row in reorg_rows:
        source = row["column_name_source"].strip()
        if "___" not in source:
            continue
        base, code = source.split("___", 1)
        values = reproschema_choices.get(base)
        if values is None:
            continue  # base without declared choices is covered by other guards
        if not any(BIDSDataset._checkbox_choice_matches(v, code) for v in values):
            unresolved.append(source)
    assert not unresolved, (
        f"{len(unresolved)} checkbox option row(s) match no reproschema choice and would be "
        f"silently dropped from phenotype/: {unresolved}"
    )


@pytest.fixture(scope="module")
def redcap_dictionary_rows():
    """Rows of the REDCap project data dictionary.

    The dictionary (``eipm/bridge2ai-redcap`` → ``data/bridge2ai_voice_redcap_project_data_dictionary.csv``)
    is vendored next to the reproschema under ``src/b2aiprep/redcap2rs/`` and refreshed by the same
    workflow that bumps the reproschema; ``B2AI_REDCAP_DICTIONARY`` overrides the path.
    """
    import os

    override = os.environ.get("B2AI_REDCAP_DICTIONARY")
    if override:
        with open(override, newline="", encoding="utf-8-sig") as fp:
            return list(csv.DictReader(fp))
    vendored = _resource("redcap2rs", "bridge2ai_voice_redcap_project_data_dictionary.csv")
    assert vendored.is_file(), "vendored REDCap data dictionary is missing from src/b2aiprep/redcap2rs/"
    with vendored.open("r", newline="", encoding="utf-8-sig") as fp:
        return list(csv.DictReader(fp))


def test_checkbox_option_codes_match_the_redcap_dictionary(reorg_rows, redcap_dictionary_rows):
    """Against the REDCap data dictionary itself, option codes must match exactly.

    Unlike the reproschema, the dictionary keeps the literal codes (``2029_7``), so no
    normalisation is allowed here; this is the guard that catches conversion-mangled codes.
    """
    dictionary = redcap_dictionary_rows
    codes_by_field = {}
    for field in dictionary:
        if field["Field Type"] != "checkbox":
            continue
        codes = [part.split(",", 1)[0].strip() for part in field["Choices, Calculations, OR Slider Labels"].split("|")]
        codes_by_field[field["Variable / Field Name"]] = {c for c in codes if c}

    mismatched = sorted(
        row["column_name_source"]
        for row in reorg_rows
        if "___" in row["column_name_source"]
        and row["column_name_source"].split("___", 1)[0] in codes_by_field
        and row["column_name_source"].split("___", 1)[1] not in codes_by_field[row["column_name_source"].split("___", 1)[0]]
    )
    assert not mismatched, f"checkbox option rows whose code is not a dictionary choice code: {mismatched}"
