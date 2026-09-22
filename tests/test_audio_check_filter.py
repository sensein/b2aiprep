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


def test_participants_without_audio_are_excluded_from_tree():
    """No sub-*/ directory is created for participants with no locatable source file."""
    from b2aiprep.prepare.dataset import BIDSDataset
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = os.path.join(tmpdir, "bids")
        os.makedirs(outdir)
        # Simulate: had_audio returns False when audio_files_by_recording has nothing
        participant = {
            "record_id": "test-no-audio",
            "sessions": [{
                "session_id": "S1",
                "session_status": "Completed",
                "session_is_control_participant": "No",
                "session_duration": "100",
                "session_site": "MIT",
                "acoustic_tasks": [{
                    "acoustic_task_id": "T1",
                    "acoustic_task_session_id": "S1",
                    "acoustic_task_name": "Rainbow Passage",
                    "acoustic_task_cohort": "adult",
                    "acoustic_task_status": "Completed",
                    "recordings": [{
                        "recording_id": "NONEXISTENT-UUID",
                        "recording_name": "Rainbow Passage",
                        "recording_duration": "5.0",
                    }]
                }]
            }],
        }
        from pathlib import Path
        from collections import OrderedDict
        had_audio = BIDSDataset._output_participant_data_to_metadata_file(
            participant, Path(outdir),
            audio_files_by_recording={},  # no audio at all
            audio_descriptor_dict=OrderedDict(),
            questionnaire_lookup={},
        )
        assert had_audio[0] is False, "should return False when no recordings have source audio"
        assert not os.path.exists(os.path.join(outdir, "sub-test-no-audio")), (
            "no sub-*/ directory should exist for a participant with no audio"
        )
