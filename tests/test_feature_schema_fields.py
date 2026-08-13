"""Each feature data dictionary must name the column its parquet actually has.

The payload column is the file name minus its backend prefix -- `ppgs.parquet` holds
`ppgs`, `torchaudio_spectrogram.parquet` holds `spectrogram`, `sparc_ema.parquet` holds
`ema` -- because `create_bundled_dataset` writes `output[feature_name] = data` and names
the file `{feature_class}_{feature_name}`.

Two dictionaries had drifted from that in opposite directions (`ppgs.json` declared
`ppg`, `torchaudio_spectrogram.json` declared `spectrograms`), so every published copy of
those files documented a column no reader can find. Validating a dictionary against the
data file itself needs a built release; this only pins the naming rule, which is the part
that can regress in a pull request.
"""

import json
from pathlib import Path

_SCHEMAS = (
    Path(__file__).resolve().parents[1]
    / "src" / "b2aiprep" / "prepare" / "resources" / "feature_schemas"
)
_INDEX_FIELDS = {"participant_id", "session_id", "task_name", "n_frames"}


def test_payload_field_matches_file_name():
    paths = sorted(_SCHEMAS.glob("*.json"))
    assert paths, f"no feature dictionaries found in {_SCHEMAS}"
    for path in paths:
        fields = json.loads(path.read_text())["fields"]
        payload = [f["name"] for f in fields if f["name"] not in _INDEX_FIELDS]
        assert len(payload) == 1 and path.stem.endswith(payload[0]), (
            f"{path.name} declares payload field(s) {payload}; the column is the file "
            f"name minus its backend prefix, so it must end with the declared name"
        )
