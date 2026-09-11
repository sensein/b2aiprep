"""Task-entity normalization at ingest.

redcap2bids names every audio file and sidecar with ``task-<entity>`` derived from the RedCap
recording name. The entity is normalized once, at ingest (``canonical_task_entity``), so the
internal and published trees agree and no parentheses, spaces, or case variants reach a file
name; ``deidentify`` then only verifies. Curated variants that normalization alone cannot merge
(word order) live in ``resources/task_registry/recording_name_aliases.json``.
"""

import json
from importlib.resources import files
from pathlib import Path

import pandas as pd
import pytest

from b2aiprep.prepare.dataset import BIDSDataset
from b2aiprep.prepare.fhir_utils import _resolve_task_registry
from b2aiprep.prepare.utils import (
    canonical_task_entity,
    load_recording_name_aliases,
    normalize_task_label,
    sanitize_task_entity_in_bids_stem,
)


def _row(instrument, **values):
    """A RedCap export row for an instrument: every column present, overrides applied.

    ``convert_response_to_bids_metadata`` indexes every column of the instrument, so a
    partial dict raises KeyError before reaching the code under test.
    """
    cols = json.loads(
        files("b2aiprep.prepare.resources")
        .joinpath("instrument_columns", f"{instrument}.json")
        .read_text()
    )
    base = {c: None for c in cols}
    base["record_id"] = "p1"
    base.update(values)
    return base


def _session(*acoustic_tasks):
    """A session dict carrying every sessions.json column (sessions.tsv is written from them)."""
    return {
        "session_id": "S1",
        "session_status": "Completed",
        "session_is_control_participant": "No",
        "session_duration": 1,
        "session_site": "test",
        "acoustic_tasks": list(acoustic_tasks),
    }


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Audio Check (v2)-1", "audio-check-v2-1"),                       # parentheses
        ("Diadochokinesis (v2)-puhtuhkuh", "diadochokinesis-v2-puhtuhkuh"),
        ("Conversation (6 plus)-favorite_food", "conversation-6-plus-favorite-food"),  # underscores
        ("Prolonged Vowel", "prolonged-vowel"),                           # case
        ("Prolonged vowel", "prolonged-vowel"),
        ("Harvard Sentences-List 38-1", "harvard-sentences-list-38-1"),
        ("Cape V sentences (v2)-1", "cape-v-sentences-v2-1"),             # canonical order
        ("Cape V sentences-1 (v2)", "cape-v-sentences-v2-1"),             # curated alias
        ("  Loudness (v2) ", "loudness-v2"),                              # surrounding whitespace
    ],
)
def test_canonical_task_entity(raw, expected):
    assert canonical_task_entity(raw) == expected


def test_canonical_task_entity_is_idempotent_and_deidentify_safe():
    for raw in ["Audio Check (v2)-1", "Cape V sentences-1 (v2)", "Respiration and cough-FiveBreaths-4"]:
        entity = canonical_task_entity(raw)
        assert canonical_task_entity(entity) == entity
        stem = f"sub-abc_ses-DEF-123_task-{entity}"
        # deidentify sanitizes the task entity of every stem; an ingest-normalized stem is a no-op
        assert sanitize_task_entity_in_bids_stem(stem) == stem


def test_entity_has_no_forbidden_characters():
    for raw in ["Audio Check (v2)-1", "Conversation (6 plus)-favorite_food", "Cape V sentences-1 (v2)"]:
        entity = canonical_task_entity(raw)
        assert entity == entity.lower()
        assert not any(ch in entity for ch in "() _"), entity
        assert "--" not in entity and not entity.startswith("-") and not entity.endswith("-")


def test_alias_table_is_well_formed_and_resolves():
    aliases = load_recording_name_aliases()
    assert aliases, "expected curated aliases (at least the CAPE-V ordering variants)"
    for variant, canonical in aliases.items():
        assert normalize_task_label(variant) == variant, variant
        assert normalize_task_label(canonical) == canonical, canonical
        assert canonical not in aliases, f"alias chain: {variant} -> {canonical} -> {aliases[canonical]}"
        # the canonical label must be a task the registry knows
        resolved = _resolve_task_registry(canonical)
        assert resolved is not None, f"{canonical} resolves to no registry task"


def test_alias_file_documents_itself():
    resource = Path(__file__).resolve().parents[1] / "src/b2aiprep/prepare/resources/task_registry/recording_name_aliases.json"
    data = json.loads(resource.read_text(encoding="utf-8"))
    assert "_comment" in data


def test_sidecar_filename_uses_normalized_task_entity(tmp_path):
    BIDSDataset._write_pydantic_model_to_bids_file(
        tmp_path,
        {"id": "x"},
        schema_name="recording",
        subject_id="p1",
        session_id="S 1",
        task_name="Audio Check (v2)",
        recording_name="Audio Check (v2)-1",
    )
    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == ["sub-p1_ses-S-1_task-audio-check-v2-1_recording-metadata.json"]

    BIDSDataset._write_pydantic_model_to_bids_file(
        tmp_path, {"id": "y"}, schema_name="acoustic_task", subject_id="p1",
        session_id="S1", task_name="Cape V sentences-1 (v2)",
    )
    assert (tmp_path / "sub-p1_ses-S1_task-cape-v-sentences-v2-1_acoustic-task-metadata.json").exists()


def test_audio_and_sidecar_share_the_entity(tmp_path):
    """The wav and its _recording-metadata.json must have the same stem so deidentify can pair them."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    wav = src_dir / "11111111-2222-3333-4444-555555555555.wav"
    wav.write_bytes(b"RIFF")  # never decoded: sanitize_audio_format=False copies bytes
    # convert_response_to_bids_metadata indexes every instrument column, so give the task and
    # recording rows the full column set (as a RedCap export row would) and override a few.
    def row(instrument, **values):
        cols = json.loads(files("b2aiprep.prepare.resources").joinpath("instrument_columns", f"{instrument}.json").read_text())
        base = {c: None for c in cols}
        base["record_id"] = "p1"
        base.update(values)
        return base

    participant = {
        "record_id": "p1",
        "sessions": [
            {
                # every sessions.json column: the function writes sub-*/sessions.tsv from them
                "session_id": "S1",
                "session_status": "Completed",
                "session_is_control_participant": "No",
                "session_duration": 1,
                "session_site": "test",
                "acoustic_tasks": [
                    row(
                        "acoustic_tasks",
                        acoustic_task_id="t1",
                        acoustic_task_name="Audio Check (v2)",
                        acoustic_task_session_id="S1",
                        recordings=[
                            row(
                                "recordings",
                                recording_id="11111111-2222-3333-4444-555555555555",
                                recording_name="Audio Check (v2)-1",
                                recording_acoustic_task_id="t1",
                                recording_session_id="S1",
                            )
                        ],
                    )
                ],
            }
        ],
    }
    out = tmp_path / "bids"
    out.mkdir()
    BIDSDataset._output_participant_data_to_metadata_file(
        participant, out, audio_files_by_recording={wav.stem: wav}, max_audio_workers=1,
        sanitize_audio_format=False, audio_descriptor_dict={},
    )
    audio_dir = out / "sub-p1" / "ses-S1" / "audio"
    names = sorted(p.name for p in audio_dir.iterdir())
    assert "sub-p1_ses-S1_task-audio-check-v2-1.wav" in names
    assert "sub-p1_ses-S1_task-audio-check-v2-1_recording-metadata.json" in names
    assert not any("(" in n or ")" in n or n != n.lower().replace("sub-p1_ses-s1", "sub-p1_ses-S1") for n in names if n.endswith(".wav"))


def test_high_to_low_resolves_to_the_glides_task():
    """The bare "High to Low" is the Glides pair's second half.

    43 adult recordings carry it; every one of those sessions also holds
    "Glides-Low to High" and none holds "Glides-High to Low". Before the alias it
    resolved to no task at all, so those recordings got no registry instructions,
    prompt, or speech_type.
    """
    assert canonical_task_entity("High to Low") == "glides-high-to-low"
    bare = _resolve_task_registry(canonical_task_entity("High to Low"), population="adult")
    prefixed = _resolve_task_registry(
        canonical_task_entity("Glides-High to Low"), population="adult"
    )
    assert bare is not None and prefixed is not None
    assert bare[0]["task_id"] == prefixed[0]["task_id"] == "adult.glides"


def test_registry_resolution_is_unchanged_for_non_aliased_names():
    """Routing resolution through the alias table must not perturb anything else."""
    for name in (
        "Prolonged vowel",
        "Cape V sentences (v2)-1",
        "Harvard Sentences-List 1-10",
        "Glides-Low to High",
    ):
        aliased = _resolve_task_registry(canonical_task_entity(name), population="adult")
        plain = _resolve_task_registry(name.replace(" ", "-").lower(), population="adult")
        assert (aliased is None) == (plain is None), name
        if aliased is not None:
            assert aliased[0]["task_id"] == plain[0]["task_id"], name


def test_exclusion_matches_across_the_naming_change():
    """A removal entry written against a pre-normalization tree still matches.

    The list is keyed on file stems, so an entry naming "task-Animal-fluency" has to
    match the "task-animal-fluency" that redcap2bids now writes -- otherwise a file
    marked for removal is silently published.
    """
    new_style = Path("sub-x/ses-y/audio/sub-x_ses-y_task-animal-fluency.wav")
    kept = BIDSDataset._apply_exclusion_list_to_filepaths(
        [new_style],
        exclusion_list=["sub-x_ses-y_task-Animal-fluency"],
        exclusion_type="filename",
    )
    assert kept == []
    # a different participant with the same task is untouched: only the task entity
    # is normalized, never the subject or session
    other = Path("sub-z/ses-y/audio/sub-z_ses-y_task-animal-fluency.wav")
    assert BIDSDataset._apply_exclusion_list_to_filepaths(
        [other],
        exclusion_list=["sub-x_ses-y_task-Animal-fluency"],
        exclusion_type="filename",
    ) == [other]


def test_unmatched_exclusion_entries_are_reported(caplog):
    """An exclusion list that matches nothing must say so, not report "removed 0"."""
    with caplog.at_level("WARNING"):
        BIDSDataset._apply_exclusion_list_to_filepaths(
            [Path("sub-x/ses-y/audio/sub-x_ses-y_task-animal-fluency.wav")],
            exclusion_list=["sub-gone_ses-gone_task-passage-10"],
            exclusion_type="filename",
        )
    assert "matched no file" in caplog.text


def test_recording_without_a_name_is_skipped_not_named_nan(tmp_path, caplog):
    """A missing recording_name must not yield a "task-nan" audio file.

    The wav name came from str(recording_name) while the sidecar name came from a
    pd.notna() check, so a NaN name produced task-nan.wav beside a sidecar named after
    the acoustic task -- and deidentify then dropped the recording for a missing sidecar.
    """
    participant = {
        "record_id": "p1",
        "selected_language": "English",
        "sessions": [
            _session(
                _row(
                    "acoustic_tasks",
                    acoustic_task_id="t1",
                    acoustic_task_name="Audio Check (v2)",
                    acoustic_task_session_id="S1",
                    recordings=[
                        _row(
                            "recordings",
                            recording_id="r1",
                            recording_name=float("nan"),
                            recording_acoustic_task_id="t1",
                            recording_session_id="S1",
                        )
                    ],
                )
            )
        ],
    }
    with caplog.at_level("WARNING"):
        BIDSDataset._output_participant_data_to_metadata_file(participant, tmp_path)
    assert "missing recording_name" in caplog.text
    assert not list(tmp_path.rglob("*task-nan*"))
    assert not list(tmp_path.rglob("*task-none*"))


def test_acoustic_task_collision_is_reported(tmp_path, caplog):
    """Two acoustic tasks whose names differ only in case share one sidecar name.

    Their recordings keep distinct names and both tasks stay in acoustic_task.tsv, so
    only the redundant sidecar is lost -- but it must not be lost silently.
    """
    participant = {
        "record_id": "p1",
        "selected_language": "English",
        "sessions": [
            _session(
                _row(
                    "acoustic_tasks",
                    acoustic_task_id="t1",
                    acoustic_task_name="Free speech",
                    acoustic_task_session_id="S1",
                    recordings=[],
                ),
                _row(
                    "acoustic_tasks",
                    acoustic_task_id="t2",
                    acoustic_task_name="Free Speech",
                    acoustic_task_session_id="S1",
                    recordings=[],
                ),
            )
        ],
    }
    with caplog.at_level("WARNING"):
        BIDSDataset._output_participant_data_to_metadata_file(participant, tmp_path)
    assert "acoustic_task_name collision" in caplog.text


def test_missing_alias_resource_raises(monkeypatch):
    """A missing packaged alias file must fail, not silently rename every file."""
    import b2aiprep.prepare.utils as utils

    load_recording_name_aliases.cache_clear()
    monkeypatch.setattr(
        utils, "_RECORDING_NAME_ALIASES", ("prepare", "resources", "task_registry", "absent.json")
    )
    try:
        with pytest.raises(FileNotFoundError):
            load_recording_name_aliases()
    finally:
        load_recording_name_aliases.cache_clear()


def test_collision_bookkeeping_survives_a_missing_id():
    """A missing id must still mark the entity as seen.

    Regression guard for the review finding on #333: the checks used
    `seen.get(entity) is not None`, so an entity first recorded with a None/NaN id
    read as "never seen" and the next record mapping to the same entity was
    overwritten without a warning -- silently, which is what these guards exist to
    prevent. Equality is only trusted when both ids are present, so NaN != NaN
    cannot fabricate a collision either.
    """
    from b2aiprep.prepare.dataset import _note_entity_collision

    # first record carries no id at all
    seen = {}
    assert _note_entity_collision(seen, "glides-high-to-low", None) == (False, None)
    assert "glides-high-to-low" in seen, "a missing id must still mark the entity as seen"
    collided, prior = _note_entity_collision(seen, "glides-high-to-low", "REC-2")
    assert collided, "a second record on the same entity is a collision even if the first had no id"
    assert prior is None

    # NaN is treated the same way, and never collides with itself
    nan = float("nan")
    seen = {}
    assert _note_entity_collision(seen, "free-speech", nan) == (False, None)
    collided, _ = _note_entity_collision(seen, "free-speech", nan)
    assert collided, "two distinct records with unusable ids are still a collision"

    # the ordinary cases are unchanged
    seen = {}
    assert _note_entity_collision(seen, "audio-check", "A") == (False, None)
    assert _note_entity_collision(seen, "audio-check", "A") == (False, "A"), "same id twice is not a collision"
    collided, prior = _note_entity_collision(seen, "audio-check", "B")
    assert (collided, prior) == (True, "A")
