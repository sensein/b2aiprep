"""RedCap sliders are described as the integer they record, not as their end labels."""
import json

from b2aiprep.prepare.update import _build_data_elements


def _activity(tmp_path, item):
    items = tmp_path / "act" / "items"
    items.mkdir(parents=True)
    (items / item["id"]).write_text(json.dumps(item))
    activity = {"id": "act", "ui": {"order": [f"items/{item['id']}"]}}
    return _build_data_elements(activity, tmp_path / "act" / "act_schema", tmp_path, "sha")


def test_slider_labels_are_not_choices(tmp_path):
    # A CAPE-V slider as b2ai-redcap2rs defines it: anchor labels in choices, range in min/max.
    element = _activity(tmp_path, {
        "id": "diagnosis_degree_s",
        "ui": {"inputType": "slider"},
        "responseOptions": {
            "choices": [{"name": {"en": v}, "value": v} for v in ("MI", "MO", "SE")],
            "minValue": 0, "maxValue": 100, "valueType": ["xsd:string"],
        },
    })["diagnosis_degree_s"]
    assert element["inputType"] == "slider"
    assert element["valueType"] == ["xsd:integer"] and element["choices"] is None
    assert element["sliderLabels"] == ["MI", "MO", "SE"]
    assert (element["minValue"], element["maxValue"]) == (0, 100)


def test_radio_choices_untouched(tmp_path):
    element = _activity(tmp_path, {
        "id": "q",
        "ui": {"inputType": "radio"},
        "responseOptions": {"choices": [{"name": {"en": "Yes"}, "value": 1}], "valueType": ["xsd:integer"]},
    })["q"]
    assert element["choices"] == [{"name": {"en": "Yes"}, "value": 1}] and "sliderLabels" not in element
