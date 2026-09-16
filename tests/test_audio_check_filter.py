"""The microphone check is removed once, at ingest, from a single definition.

Audio check is collection apparatus, not research data: it is absent from the task registry,
stripped again at deidentify, and has never shipped in a release. Before this filter existed it
was materialised in full and removed only at deidentify -- 7,970 of 70,520 adult recordings
(11%) and 1,383 of 99,724 pediatric ones were copied, sidecar'd, listed in recording.tsv, and
had features and quality metrics computed for them, all to be deleted at the last stage.
"""

import pandas as pd
import pytest

from b2aiprep.prepare.utils import AUDIO_CHECK_LABEL, is_audio_check


@pytest.mark.parametrize(
    "name",
    ["Audio Check", "Audio Check-1", "Audio Check (v2)-5", "audio-check-v2-3",
     "audio-check-(v2)", "AUDIO CHECK", "audio_check"],
)
def test_every_collected_spelling_is_recognised(name):
    """All of these appear in the v4 exports; they must not need separate handling."""
    assert is_audio_check(name)


@pytest.mark.parametrize(
    "name",
    ["Glides", "free-speech-1", "Rainbow Passage", "Cape V sentences", "Word-color Stroop"],
)
def test_real_tasks_are_untouched(name):
    assert not is_audio_check(name)


@pytest.mark.parametrize("empty", [None, "", "   ", "nan", "NaN", "None", float("nan")])
def test_missing_names_are_not_audio_checks(empty):
    """Callers pass raw RedCap cells, which are freely NaN; that is not an audio check."""
    assert not is_audio_check(empty)


def test_rows_are_dropped_for_both_instruments():
    from b2aiprep.prepare.dataset import BIDSDataset

    df = pd.DataFrame({
        "redcap_repeat_instrument": ["Recording", "Recording", "Acoustic Task",
                                     "Acoustic Task", "Session"],
        "recording_name": ["Audio Check (v2)-3", "Glides", None, None, None],
        "acoustic_task_name": [None, None, "Audio Check", "Rainbow Passage", None],
    })
    out = BIDSDataset._drop_audio_check_rows(df)
    assert out["recording_name"].dropna().tolist() == ["Glides"]
    assert out["acoustic_task_name"].dropna().tolist() == ["Rainbow Passage"]
    # the session row carries neither column and must survive
    assert (out["redcap_repeat_instrument"] == "Session").sum() == 1


def test_filtering_is_idempotent_and_safe_on_unrelated_frames():
    from b2aiprep.prepare.dataset import BIDSDataset

    df = pd.DataFrame({"redcap_repeat_instrument": ["Recording"], "recording_name": ["Glides"]})
    assert len(BIDSDataset._drop_audio_check_rows(df)) == 1
    # a frame with no instrument column at all (e.g. an already-built phenotype table)
    assert len(BIDSDataset._drop_audio_check_rows(pd.DataFrame({"x": [1, 2]}))) == 2


def test_the_registry_miss_warning_is_not_suppressed():
    """Audio check has no registry entry, and that warning must stay loud.

    The warning exists to surface registry gaps. It stopped being noisy because audio check no
    longer reaches the sidecar builder -- not because anyone special-cased it. A build run with
    drop_audio_check=False still needs to hear about it.
    """
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "src/b2aiprep/prepare/fhir_utils.py"
    text = source.read_text(encoding="utf-8")
    start = text.index("no task match for recording task_name")
    context = text[max(0, start - 400):start]
    assert "is_audio_check" not in context, (
        "the registry-miss warning must not special-case audio check; "
        "it is silent only because those rows are filtered upstream"
    )
    assert AUDIO_CHECK_LABEL == "audio-check"
