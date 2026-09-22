"""Data dictionary entries for columns that ReproSchema does not define.

ReproSchema is generated from the RedCap data dictionary, so it only describes columns that
someone authored as a field. RedCap additionally emits one `<form>_complete` per instrument and
one `<form>_timestamp` per survey-enabled instrument, plus its own structural columns; none of
these are dictionary variables. Before the fallback existed, mapping such a column in
bids_field_organization.csv crashed redcap2bids with a KeyError partway through the build.
"""

from b2aiprep.prepare.dataset import BIDSDataset


def test_csv_description_becomes_the_data_element():
    element = BIDSDataset._synthetic_data_element(
        "acoustic_task_complete",
        {"description": "RedCap form-completion status for the acoustic_task instrument."},
    )
    assert element["description"] == (
        "RedCap form-completion status for the acoustic_task instrument."
    )
    assert element["valueType"] == ["xsd:string"]


def test_no_ontology_reference_is_invented():
    """The entry must not claim provenance it does not have.

    A termURL points at a specific commit of b2ai-redcap2rs, which is meaningful only for a
    column that ReproSchema actually defines. Emitting one here would put an unsourced ontology
    reference into a published data dictionary, so the absence of these keys is what lets a
    consumer tell a CSV-described column from an authored one.
    """
    element = BIDSDataset._synthetic_data_element("enrollment_form_timestamp", {"description": "x"})
    for key in ("termURL", "choices", "question", "datatype"):
        assert key not in element, f"{key} must not be invented for a CSV-only column"
    # same shape as the synthetic participant_id element the writer already emits
    assert set(element) == {"description", "valueType"}


def test_checkbox_options_keep_their_integer_type():
    """A ___ option still becomes 0/1 when the phenotype data is cleaned."""
    cleaned = BIDSDataset._synthetic_data_element(
        "participant_data_collection_origins___reproschema", {"description": "x"},
        column_choice="reproschema", clean_phenotype_data=True,
    )
    assert cleaned["valueType"] == ["xsd:integer"]
    raw = BIDSDataset._synthetic_data_element(
        "participant_data_collection_origins___reproschema", {"description": "x"},
        column_choice="reproschema", clean_phenotype_data=False,
    )
    assert raw["valueType"] == ["xsd:string"]


def test_a_missing_description_is_stated_not_left_blank():
    """An empty description would otherwise reach a published dictionary as ""."""
    for row in ({}, {"description": ""}, {"description": "   "}, {"description": None}):
        element = BIDSDataset._synthetic_data_element("some_form_complete", row)
        assert "some_form_complete" in element["description"]
        assert "no ReproSchema definition" in element["description"]


def test_form_status_does_not_keep_an_otherwise_empty_row():
    """RedCap emits `<form>_complete` for every record, filled in or not.

    Counting it as content would put every participant in every table: measured on the v4 adult
    export, `diagnosis/amyotrophic_lateral_sclerosis.tsv` went from 6 rows to 2005 before this
    rule existed.
    """
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(
        {
            "participant_id": ["has-als", "no-als"],
            "diagnosis_als_onset": ["bulbar", np.nan],
            "d_neuro_amyotrophic_lateral_sclerosis_complete": ["Complete", "Incomplete"],
        }
    )
    kept = BIDSDataset._drop_rows_without_substantive_data(
        df, "participant_id", {"d_neuro_amyotrophic_lateral_sclerosis_complete"}
    )
    assert kept["participant_id"].tolist() == ["has-als"]

    # without the exclusion the empty row survives - this is the regression being guarded
    assert len(BIDSDataset._drop_rows_without_substantive_data(df, "participant_id", set())) == 2


def test_a_table_of_only_bookkeeping_columns_is_left_alone():
    """Nothing to test against, so keep the rows rather than silently emptying the table."""
    import pandas as pd

    df = pd.DataFrame({"participant_id": ["a", "b"], "some_form_complete": ["Incomplete"] * 2})
    kept = BIDSDataset._drop_rows_without_substantive_data(
        df, "participant_id", {"some_form_complete"}
    )
    assert len(kept) == 2
