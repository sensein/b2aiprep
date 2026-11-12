"""Tests of ``triage_hints``: the release's file-name reader for triage (moved from senselab, its plan step 7)."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

# No skip on an older senselab: these modules exist only with senselab's triage refactor, so a run against an
# older senselab must fail here, not pass by skipping every test.
import senselab.audio.workflows.triage.task_catalogue  # noqa: F401

from senselab.audio.workflows.triage.identity import FROM_STEM
from senselab.audio.workflows.triage.routing_analysis.families import UNKNOWN_TASK, task_family
from senselab.audio.workflows.triage.run import entity_subdir

from b2aiprep.prepare.triage_hints import b2ai_task, identity_from_name, task_id_of

ADULT_PROFILE = (
    Path(__file__).resolve().parents[1] / "src/b2aiprep/prepare/resources/triage_profile/adult"
)
"""The adult triage profile, whose task list the hint builder reads."""


@pytest.fixture(autouse=True)
def _adult_profile() -> Iterator[None]:
    """Every test reads the adult triage profile, and leaves none active."""
    from senselab.audio.workflows.triage.dataset_profile import (
        activate_profile,
        deactivate_profile,
        load_profile,
    )

    deactivate_profile()
    activate_profile(load_profile(ADULT_PROFILE))
    yield
    deactivate_profile()


STEM = "sub-a_ses-b_task-respiration-and-cough-fivebreaths-2"


class TestFromTheStem:
    """With no hint, the identity is exactly what the stem parsers read."""

    def test_a_bids_stem_names_participant_session_and_task(self) -> None:
        """Every field comes from the stem, and the family drops the trailing index."""
        identity = identity_from_name(STEM)
        assert (identity.participant, identity.session) == ("sub-a", "ses-b")
        assert identity.task_id == "respiration-and-cough-fivebreaths-2"
        assert task_family(identity.task_id) == "respiration-and-cough-fivebreaths"
        assert set(identity.sources.values()) == {FROM_STEM}

    def test_each_entity_is_read_on_its_own(self) -> None:
        """``sub-`` and ``ses-`` are read wherever they sit; a lone participant keeps its folder and group."""
        cases = {
            STEM: ("sub-a", "ses-b"),
            "sub-a_acq-x_ses-b_task-loudness": ("sub-a", "ses-b"),
            "sub-a_task-rainbow-passage": ("sub-a", None),
            "ses-b_task-x": (None, "ses-b"),
            "sub-_ses-_task-x": (None, None),
            "plain-run-id": (None, None),
            "": (None, None),
        }
        for stem, expected in cases.items():
            identity = identity_from_name(stem)
            assert (identity.participant, identity.session) == expected
            assert identity.task_id == task_id_of(stem)
            assert task_family(identity.task_id) == task_family(task_id_of(stem))
        assert identity_from_name("sub-a_task-x").session_key is None

    def test_a_stem_with_no_task_entity_declares_none(self) -> None:
        """The task id is the parsers' own placeholder."""
        assert identity_from_name("plain-run-id").task_id == UNKNOWN_TASK


def test_task_id_reads_the_bids_entity() -> None:
    """The task id comes from the stem's trailing ``task-`` entity, lowercased."""
    assert task_id_of("sub-A_ses-B_task-Prolonged-Vowel") == "prolonged-vowel"
    assert task_id_of("sub-A_ses-B") == "unknown"


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("sub-01_ses-02_task-vowel", ("sub-01", "ses-02")),
        ("sub-01_task-vowel", ("sub-01",)),
        ("recording", ()),
        ("nothing_bids_here", ()),
        ("sub-SUBJ01_ses-SES01_task-diadochokinesis-ka", ("sub-SUBJ01", "ses-SES01")),
        ("ses-02_task-vowel", ("ses-02",)),
        ("sub-_ses-_task-vowel", ()),
        ("sub-01_ses-_task-vowel", ("sub-01",)),
    ],
)
def test_the_name_readers_entities_place_the_run(stem: str, expected: tuple[str, ...]) -> None:
    """The reader's participant and session place a run: a label may contain hyphens, an empty one is none."""
    assert entity_subdir(identity_from_name(stem)) == (Path(*expected) if expected else Path("."))


class TestTheTaskAHintDeclares:
    """``b2ai_task``: the task, variant, take and route a B2AI task id declares."""

    @pytest.mark.parametrize(
        ("take", "route"), [(1, "nose"), (2, "mouth"), (3, "nose"), (4, "mouth")]
    )
    def test_a_five_breaths_take_names_its_route(self, take: int, route: str) -> None:
        """A task whose route is declared per recording takes its route from the take."""
        task, variant, declared_take = b2ai_task(f"respiration-and-cough-fivebreaths-{take}")
        assert task == "respiration-and-cough-fivebreaths"
        assert variant == {"index": str(take), "route": route}
        assert declared_take == take

    def test_a_take_outside_the_protocol_names_no_route(self) -> None:
        """A fifth take is still a take, but the protocol names no route for it."""
        task, variant, take = b2ai_task("respiration-and-cough-fivebreaths-5")
        assert (task, variant, take) == ("respiration-and-cough-fivebreaths", {"index": "5"}, 5)

    def test_a_task_without_a_per_recording_route_has_an_index_but_no_take(self) -> None:
        """Another task's trailing number is kept as its index; it is not a take and names no route."""
        task, variant, take = b2ai_task("free-speech-2")
        assert task == "free-speech"
        assert variant == {"index": "2"}
        assert take is None

    def test_a_task_with_no_index_has_no_variant(self) -> None:
        """A task id with no trailing number declares the task alone."""
        assert b2ai_task("rainbow-passage") == ("rainbow-passage", {}, None)

    def test_an_unknown_task_declares_nothing(self) -> None:
        """The parsers' placeholder declares no task, variant or take."""
        assert b2ai_task(UNKNOWN_TASK) == (None, {}, None)


def test_the_installed_package_carries_a_profile_that_passes_the_start_check() -> None:
    """The profile is read through the package's resources and passes senselab's start check as packaged."""
    from importlib.resources import files

    from senselab.audio.workflows.triage.config import load_triage_config
    from senselab.audio.workflows.triage.dataset_profile import (
        deactivate_profile,
        missing_dataset_values,
    )

    packaged = Path(str(files("b2aiprep") / "prepare/resources/triage_profile/adult"))
    assert (packaged / "profile.yaml").is_file()
    deactivate_profile()
    config = load_triage_config(profile=packaged)
    assert missing_dataset_values(config.values) == []
