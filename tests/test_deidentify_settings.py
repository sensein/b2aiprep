"""deidentify_settings.json: loading, and the release rules it holds (value_mappings, relabel,
exclude_participants, small_checkbox_options). Nothing is applied unless the config lists it."""

import json

import pandas as pd
import pytest

from b2aiprep.prepare.dataset import AccessTier, BIDSDataset

# the adult configs' small_checkbox_options (4.0-release/configs)
VOICE_ACTIVITY_FOLD = {"confounders": {
    "voice_activity_v2": {"other": "voice_activity_v2___other", "specify": "other_voice_activity",
                          "keep": ["voice_activity_v2___none"], "min_participants": 10},
    "voice_activity": {"other": "voice_activity___7", "min_participants": 10}}}

SITES = {"participant": {"enrollment_institution": {
    "output_column": "site", "values": {"MIT": "site_A", "USF": "site_B", "WCM": "site_C"}}}}


# --- loading -------------------------------------------------------------------------------

@pytest.mark.parametrize("contents, error, match", [
    (None, FileNotFoundError, "access tier"),
    ({"access_tier": "public"}, ValueError, "access_tier must be one of"),
    ({"access_tier": "controlled", "typo": 1}, ValueError, "unknown setting"),
], ids=["missing-file", "unknown-tier", "unknown-key"])
def test_settings_file_errors(tmp_path, contents, error, match):
    if contents is not None:
        (tmp_path / "deidentify_settings.json").write_text(json.dumps(contents))
    with pytest.raises(error, match=match):
        BIDSDataset._load_deidentify_settings(tmp_path)


@pytest.mark.parametrize("key, value, empty", [
    ("value_mappings", SITES, {}),
    ("relabel", {"confounders": {"ph_walking": {"None": "No difficulty"}}}, {}),
    ("exclude_participants", [{"table": "t", "column": "c", "values": ["x"]}], []),
    ("small_checkbox_options", VOICE_ACTIVITY_FOLD, {}),
])
def test_optional_settings_pass_through_or_default_to_nothing(tmp_path, key, value, empty):
    path = tmp_path / "deidentify_settings.json"
    path.write_text(json.dumps({"access_tier": "controlled"}))
    settings = BIDSDataset._load_deidentify_settings(tmp_path)
    assert settings["access_tier"] is AccessTier.CONTROLLED and settings[key] == empty
    path.write_text(json.dumps({"access_tier": "registered", key: value}))
    assert BIDSDataset._load_deidentify_settings(tmp_path)[key] == value


# --- value_mappings ------------------------------------------------------------------------

def _participants(sites):
    return pd.DataFrame({"participant_id": [f"p{i}" for i in range(len(sites))],
                         "enrollment_institution": sites, "age": ["40"] * len(sites)})


def test_site_is_written_next_to_its_source_with_its_labels_as_choices():
    phenotype = {"enrollment_institution": {"description": "Enrollment Institution"}}
    df, phenotype = BIDSDataset._apply_value_mappings(
        _participants(["MIT", "USF", None]), phenotype, "participant", SITES)
    assert list(df.columns) == ["participant_id", "enrollment_institution", "site", "age"]
    assert df["site"].tolist()[:2] == ["site_A", "site_B"] and pd.isna(df["site"].iloc[2])
    assert df["enrollment_institution"].tolist()[:2] == ["MIT", "USF"]  # left for the disposition step
    element = phenotype["site"]
    assert [c["value"] for c in element["choices"]] == ["site_A", "site_B"]
    assert "arbitrary label" in element["description"] and "termURL" not in element


def test_a_value_with_no_mapping_stops_the_run():
    with pytest.raises(ValueError, match="no mapping"):
        BIDSDataset._apply_value_mappings(_participants(["MIT", "Elsewhere"]), {}, "participant", SITES)


def test_rare_labels_become_other_when_a_minimum_is_set():
    spec = json.loads(json.dumps(SITES))
    spec["participant"]["enrollment_institution"].update({"min_participants": 2, "other": "other"})
    df, phenotype = BIDSDataset._apply_value_mappings(_participants(["MIT", "MIT", "USF"]), {}, "participant", spec)
    assert df["site"].tolist() == ["site_A", "site_A", "other"]
    assert [c["value"] for c in phenotype["site"]["choices"]] == ["other", "site_A"]


def test_tables_without_a_mapping_are_untouched():
    df = _participants(["MIT"])
    out, _ = BIDSDataset._apply_value_mappings(df, {}, "demographics", SITES)
    assert out.equals(df)


# --- small_checkbox_options ----------------------------------------------------------------

def _voice_activity(dtype):
    """12 participants: Teacher ticked by p0-p9 (10), Attorney by p10-p11 (2); p11 also typed 'Podcaster'."""
    tick = "1" if dtype == "text" else 1
    blank = "" if dtype == "text" else None
    df = pd.DataFrame({
        "participant_id": [f"p{i}" for i in range(12)],
        "voice_activity_v2___teacher": [tick] * 10 + [blank, blank],
        "voice_activity_v2___attorney": [blank] * 10 + [tick, tick],
        "voice_activity_v2___other": [blank] * 11 + [tick],
        "voice_activity_v2___none": [blank] * 12,
        "other_voice_activity": [blank] * 11 + ["Podcaster"],
    }, dtype=str if dtype == "text" else object)
    if dtype == "text":
        df = df.replace("", pd.NA)
    elements = {c: {"description": c} for c in df.columns}
    elements["voice_activity_v2___attorney"]["choices"] = [{"name": {"en": "Attorney"}, "value": "attorney"}]
    return df, elements, tick


@pytest.mark.parametrize("dtype", ["numeric", "text"])  # deidentify reads tables as text
def test_rare_checkbox_options_fold_into_other_with_their_label(dtype):
    df, elements, tick = _voice_activity(dtype)
    out, el = BIDSDataset._fold_small_checkbox_options(df, elements, "confounders", VOICE_ACTIVITY_FOLD)
    assert "voice_activity_v2___attorney" not in out.columns and "voice_activity_v2___attorney" not in el
    assert "voice_activity_v2___teacher" in out.columns
    assert list(out["voice_activity_v2___other"])[-2:] == [tick, tick]
    assert list(out["other_voice_activity"])[-2:] == ["Attorney", "Podcaster; Attorney"]


def test_rare_checkbox_options_are_counted_over_released_participants():
    df, elements, _ = _voice_activity("text")
    out, _ = BIDSDataset._fold_small_checkbox_options(
        df, elements, "confounders", VOICE_ACTIVITY_FOLD, released_participants={f"p{i}" for i in range(1, 12)})
    assert "voice_activity_v2___teacher" not in out.columns  # 9 released participants: folded


def test_deprecated_voice_activity_folds_rare_options_into_its_other_option():
    """The first version of the question has no 'please specify' box: rare options only tick Other."""
    df = pd.DataFrame({"participant_id": [f"p{i}" for i in range(12)],
                       "voice_activity___4": ["1"] * 10 + ["", ""],    # Teacher, 10 participants: kept
                       "voice_activity___6": [""] * 10 + ["1", "1"],   # Cheerleading, 2: folded
                       "voice_activity___7": [""] * 12}, dtype=str).replace("", pd.NA)
    out, _ = BIDSDataset._fold_small_checkbox_options(df, {}, "confounders", VOICE_ACTIVITY_FOLD)
    assert "voice_activity___6" not in out.columns and "voice_activity___4" in out.columns
    assert out["voice_activity___7"].tolist()[-2:] == ["1", "1"]


# --- relabel -------------------------------------------------------------------------------

def test_relabel_changes_only_the_listed_labels_in_cells_and_choices():
    df = pd.DataFrame({"participant_id": ["1", "2", "3"], "ph_walking": ["None", "Mild", pd.NA]})
    choices = [{"name": {"en": "None"}, "value": "none"}, {"name": {"en": "Mild"}, "value": "mild"}]
    out, el = BIDSDataset._apply_relabels(df, {"ph_walking": {"choices": choices}}, "confounders",
                                         {"confounders": {"ph_walking": {"None": "No difficulty"}}})
    assert out["ph_walking"].tolist()[:2] == ["No difficulty", "Mild"] and pd.isna(out["ph_walking"].iloc[2])
    assert el["ph_walking"]["choices"] == [{"name": {"en": "No difficulty"}, "value": "none"},
                                           {"name": {"en": "Mild"}, "value": "mild"}]


def test_relabel_refuses_a_label_the_column_does_not_have():
    df = pd.DataFrame({"participant_id": ["1"], "ph_walking": ["None"]})
    elements = {"ph_walking": {"choices": [{"name": {"en": "None"}, "value": "none"}]}}
    with pytest.raises(ValueError, match="no choice labelled"):
        BIDSDataset._apply_relabels(df, elements, "confounders", {"confounders": {"ph_walking": {"Nome": "x"}}})


def test_relabelled_answers_and_their_choices_reach_the_released_json(tmp_path, make_deid_tree):
    """ph_walking is a released field (shipped field map)."""
    bids, config = make_deid_tree(
        {"p1": [("s1", "rainbow-passage")], "p2": [("s1", "rainbow-passage")]},
        tables={"confounders/confounders": pd.DataFrame({"participant_id": ["p1", "p2"],
                                                         "ph_walking": ["None", "Mild"]})},
        choices={"confounders/confounders": {"ph_walking": ["None", "Mild", "Extreme"]}},
        settings={"relabel": {"confounders": {"ph_walking": {"None": "No difficulty"}}}})
    BIDSDataset(bids).deidentify(outdir=tmp_path / "out", deidentify_config_dir=config)
    released = tmp_path / "out" / "phenotype" / "confounders"
    assert sorted(BIDSDataset._read_tsv_as_written(released / "confounders.tsv")["ph_walking"]) == [
        "Mild", "No difficulty"]
    element = json.loads((released / "confounders.json").read_text())["confounders"]["data_elements"]["ph_walking"]
    assert element["choices"] == [{"name": {"en": "No difficulty"}, "value": "none"},
                                  {"name": {"en": "Mild"}, "value": "mild"},
                                  {"name": {"en": "Extreme"}, "value": "extreme"}]


# --- exclude_participants ------------------------------------------------------------------

GENDER_RULE = {"table": "pediatric_demographics", "column": "peds_gender_identity", "values": ["Other"]}


def _pediatric_tree(make_deid_tree, rule):
    """peds_gender_identity is a released field (shipped field map)."""
    return make_deid_tree(
        {"p1": [("s1", "rainbow-passage")], "p2": [("s1", "rainbow-passage")]},
        tables={"pediatric/pediatric_demographics": pd.DataFrame({
            "participant_id": ["p1", "p2"], "peds_gender_identity": ["Female gender identity", "Other"]})},
        choices={"pediatric/pediatric_demographics": {"peds_gender_identity": ["Female gender identity", "Other"]}},
        settings={"exclude_participants": [rule]})


def test_exclude_participants_removes_a_participant_by_one_answer(tmp_path, make_deid_tree):
    bids, config = _pediatric_tree(make_deid_tree, GENDER_RULE)
    BIDSDataset(bids).deidentify(outdir=tmp_path / "out", deidentify_config_dir=config)
    assert sorted(p.name for p in (tmp_path / "out").glob("sub-*")) == ["sub-900000"]
    released = BIDSDataset._read_tsv_as_written(
        tmp_path / "out" / "phenotype" / "pediatric" / "pediatric_demographics.tsv")
    assert list(released["participant_id"]) == ["900000"]


@pytest.mark.parametrize("typo, match", [
    ({"column": "peds_gender_identiy"}, "no column 'peds_gender_identiy'"),
    ({"values": ["other"]}, r"no choice labelled \['other'\]"),
], ids=["column", "value"])
def test_exclude_participants_refuses_a_rule_that_would_match_nothing(tmp_path, make_deid_tree, typo, match):
    bids, config = _pediatric_tree(make_deid_tree, {**GENDER_RULE, **typo})
    with pytest.raises(ValueError, match=match):
        BIDSDataset(bids).deidentify(outdir=tmp_path / "out", deidentify_config_dir=config)
