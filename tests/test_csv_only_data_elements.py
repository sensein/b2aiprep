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


def test_a_table_of_only_bookkeeping_columns_is_emptied():
    """No substantive columns means no research value — table is emptied."""
    import pandas as pd

    df = pd.DataFrame({"participant_id": ["a", "b"], "some_form_complete": ["Incomplete"] * 2})
    result = BIDSDataset._drop_rows_without_substantive_data(
        df, "participant_id", {"some_form_complete"}
    )
    assert len(result) == 0
    assert list(result.columns) == list(df.columns)


# ---------------------------------------------------------------------------
# Calculated-column exclusion and diagnosis completeness check
# ---------------------------------------------------------------------------

def test_calculated_fields_do_not_keep_incomplete_diagnosis_rows():
    """A diagnosis form with only auto-calculated values and _complete=Incomplete is phantom."""
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(
        {
            "participant_id": ["real-als", "phantom"],
            "diagnosis_als_onset": ["bulbar", np.nan],
            "diagnosis_als_gsd_calculation": [1, 0],
            "d_neuro_als_complete": ["Complete", "Incomplete"],
        }
    )
    kept = BIDSDataset._drop_rows_without_substantive_data(
        df,
        "participant_id",
        csv_only_columns={"d_neuro_als_complete"},
        calculated_columns={"diagnosis_als_gsd_calculation"},
        schema_group="diagnosis",
    )
    assert kept["participant_id"].tolist() == ["real-als"]


def test_calculated_fields_excluded_from_substantive_check_non_diagnosis():
    """For non-diagnosis forms, calculated fields are excluded from the emptiness test."""
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(
        {
            "participant_id": ["has-score", "calc-only"],
            "vhi_item_1": ["Sometimes", np.nan],
            "vhi_10_calc_score": [12, 0],
            "vhi10_complete": ["Complete", "Incomplete"],
        }
    )
    kept = BIDSDataset._drop_rows_without_substantive_data(
        df,
        "participant_id",
        csv_only_columns={"vhi10_complete"},
        calculated_columns={"vhi_10_calc_score"},
        schema_group="questionnaire",
    )
    assert kept["participant_id"].tolist() == ["has-score"]


def test_diagnosis_incomplete_with_real_data_is_dropped():
    """Diagnosis forms require _complete=Complete; incomplete rows with real data are dropped."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "participant_id": ["verified", "unverified"],
            "diagnosis_mtd_degree": [50.0, 30.0],
            "d_voice_mtd_complete": ["Complete", "Incomplete"],
        }
    )
    kept = BIDSDataset._drop_rows_without_substantive_data(
        df,
        "participant_id",
        csv_only_columns={"d_voice_mtd_complete"},
        schema_group="diagnosis",
    )
    assert kept["participant_id"].tolist() == ["verified"]


def test_diagnosis_unverified_is_also_dropped():
    """Unverified diagnosis forms are not clinician-verified and must be dropped."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "participant_id": ["complete", "unverified"],
            "diagnosis_pd_subtype": ["IPD", "PSP"],
            "d_neuro_pd_complete": ["Complete", "Unverified"],
        }
    )
    kept = BIDSDataset._drop_rows_without_substantive_data(
        df,
        "participant_id",
        csv_only_columns={"d_neuro_pd_complete"},
        schema_group="diagnosis",
    )
    assert kept["participant_id"].tolist() == ["complete"]


def test_non_diagnosis_keeps_incomplete_rows_with_data():
    """Questionnaires/enrollment keep Incomplete rows that have real data."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "participant_id": ["complete", "incomplete-but-data"],
            "phq9_score": [12, 8],
            "phq9_complete": ["Complete", "Incomplete"],
        }
    )
    kept = BIDSDataset._drop_rows_without_substantive_data(
        df,
        "participant_id",
        csv_only_columns={"phq9_complete"},
        schema_group="questionnaire",
    )
    assert len(kept) == 2


def test_complete_form_with_no_data_is_dropped_non_diagnosis():
    """A Complete form with all-null substantive columns is still dropped (data anomaly)."""
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(
        {
            "participant_id": ["has-data", "empty-complete"],
            "confounders_smoking": ["Yes", np.nan],
            "confounders_complete": ["Complete", "Complete"],
        }
    )
    kept = BIDSDataset._drop_rows_without_substantive_data(
        df,
        "participant_id",
        csv_only_columns={"confounders_complete"},
        schema_group="confounders",
    )
    assert kept["participant_id"].tolist() == ["has-data"]


def test_diagnosis_no_complete_col_falls_back_to_substantive_check():
    """If no _complete column exists, diagnosis forms fall back to the data check."""
    import numpy as np
    import pandas as pd

    df = pd.DataFrame(
        {
            "participant_id": ["has-data", "empty"],
            "diagnosis_field": ["Yes", np.nan],
        }
    )
    kept = BIDSDataset._drop_rows_without_substantive_data(
        df,
        "participant_id",
        csv_only_columns=set(),
        schema_group="diagnosis",
    )
    assert kept["participant_id"].tolist() == ["has-data"]

