"""The payload column of each bundled feature file is named after the feature, not the file.

`create_bundled_dataset` writes one parquet per entry in its `features_to_extract` list,
naming the file `{feature_class}_{feature_name}` (or just `{feature_name}` when there is no
class) and the payload column `{feature_name}` -- see `bundle_data.feature_extraction_generator`,
which does `output[feature_name] = data`.

The shipped data dictionary for each file must therefore declare a payload field whose name is
exactly that `feature_name`. Two dictionaries drifted from this (`ppgs.json` declared `ppg`,
`torchaudio_spectrogram.json` declared `spectrograms`), so every published copy of those two
files documented a column that did not exist. This test pins the rule.
"""

import json
import re
from pathlib import Path

import pytest

# Read the tree this test lives in, not the installed package: the point is to check the
# dictionaries shipped alongside this checkout's code.
_SRC = Path(__file__).resolve().parents[1] / "src" / "b2aiprep"

# Columns every feature parquet carries regardless of the feature itself.
_INDEX_FIELDS = {"participant_id", "session_id", "task_name", "n_frames"}

_FEATURE_ENTRY = re.compile(
    r"\{'feature_class': (None|'[a-z_]+'), 'feature_name': '([a-z_]+)'\}"
)


def _features_to_extract():
    """Read the (feature_class, feature_name) pairs the bundler actually iterates."""
    source = (_SRC / "commands.py").read_text()
    pairs = _FEATURE_ENTRY.findall(source)
    assert pairs, "could not locate features_to_extract in commands.py"
    return [(None if c == "None" else c.strip("'"), n) for c, n in pairs]


def _schema_fields(alias):
    path = _SRC / "prepare" / "resources" / "feature_schemas" / f"{alias}.json"
    return json.loads(path.read_text())["fields"]


@pytest.mark.parametrize("feature_class,feature_name", _features_to_extract())
def test_payload_field_is_named_after_the_feature(feature_class, feature_name):
    alias = f"{feature_class}_{feature_name}" if feature_class else feature_name
    payload = [f["name"] for f in _schema_fields(alias) if f["name"] not in _INDEX_FIELDS]
    assert payload == [feature_name], (
        f"{alias}.json declares payload field(s) {payload}; the bundled column is "
        f"'{feature_name}'. The dictionary must name the column the file actually has."
    )


@pytest.mark.parametrize("feature_class,feature_name", _features_to_extract())
def test_index_fields_present(feature_class, feature_name):
    alias = f"{feature_class}_{feature_name}" if feature_class else feature_name
    names = {f["name"] for f in _schema_fields(alias)}
    missing = _INDEX_FIELDS - names
    assert not missing, f"{alias}.json is missing index field(s) {sorted(missing)}"
