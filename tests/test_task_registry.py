"""Integrity tests for the vendored task registry and its resolution.

These validate the generated ``resources/task_registry/registry.json`` + stimulus
banks as data (every recording resolvable, aliases resolve, banks indexable,
provenance well-formed), independent of any redcap checkout. Resolver-output
behavior (parity, curated fixes, version fix) lives in ``test_task_prompts.py``.
"""

import json
from importlib.resources import files

import pytest

from b2aiprep.prepare.fhir_utils import (
    _bank,
    _load_registry,
    _norm,
    _resolve_prompt_ref,
    _resolve_task_registry,
)

VALID_SPEECH_TYPES = {"read", "recall", "elicited", "non-lexical"}
VALID_INSTR_SOURCES = {"curated", "harvested", ""}


@pytest.fixture(scope="module")
def registry():
    return _load_registry()


def test_registry_loads_and_has_expected_shape(registry):
    assert registry["schema_version"]
    assert registry["source"]["repo"] == "bridge2ai-redcap"
    assert len(registry["tasks"]) >= 40
    assert registry["alias_index"]


def test_every_task_resolvable_by_each_alias(registry):
    for task_id, task in registry["tasks"].items():
        for alias in task["aliases"]:
            match = _resolve_task_registry(alias)
            assert match is not None, f"{alias} did not resolve"
            resolved_task, _ = match
            # An alias may be shared across versions (alias_index keeps the first);
            # resolving it must at least land on the same family.
            assert resolved_task["family"] == task["family"]


def test_alias_index_targets_exist(registry):
    for alias, task_id in registry["alias_index"].items():
        assert task_id in registry["tasks"]


def test_every_nested_recording_resolves_to_itself(registry):
    seen = 0
    for task_id, task in registry["tasks"].items():
        for rec in task["recordings"]:
            match = _resolve_task_registry(rec["recording_id"])
            assert match is not None, f"{rec['recording_id']} did not resolve"
            resolved_task, resolved_rec = match
            assert resolved_rec is not None, rec["recording_id"]
            assert resolved_rec["recording_id"] == rec["recording_id"]
            seen += 1
    assert seen >= 30  # the grouped/curated recordings


def test_speech_type_and_instructions_source_valid(registry):
    for task_id, task in registry["tasks"].items():
        assert task["speech_type"] in VALID_SPEECH_TYPES, task_id
        assert task.get("instructions_source", "") in VALID_INSTR_SOURCES, task_id
        for rec in task["recordings"]:
            assert rec.get("instructions_source", "") in VALID_INSTR_SOURCES


def test_every_task_can_produce_instructions_or_stimulus(registry):
    """No task is a dead end: it has a task-level instruction, at least one
    recording with an instruction, or a prompt_ref that yields stimulus."""
    for task_id, task in registry["tasks"].items():
        has_task_instr = bool(task["instructions"])
        has_rec_instr = any(r["instructions"] for r in task["recordings"])
        has_prompt = bool((task.get("prompt_ref") or {}).get("type"))
        assert has_task_instr or has_rec_instr or has_prompt, task_id


def test_stimulus_bank_refs_load_and_index(registry):
    """Every stimulus-bank prompt_ref points to a bank that loads and is indexable
    by its declared select strategy."""
    for task_id, task in registry["tasks"].items():
        pr = task.get("prompt_ref") or {}
        if pr.get("type") != "stimulus-bank":
            continue
        bank = _bank(pr["bank"])
        select = pr["select"]
        if select == "list-index":
            assert bank["lists"]  # harvard: {list: [sentences]}
            first = next(iter(bank["lists"]))
            assert _resolve_prompt_ref(f"harvard-sentences-list-{first}-1", pr)
        elif select == "version-index":
            assert bank["lists"]["v1"] and bank["lists"]["v2"]
        elif select == "index":
            items = bank.get("sentences", bank.get("words"))
            assert items and isinstance(items, list)


def test_norm_matches_paren_forms():
    # the real data uses parens; registry recording_ids are paren-free slugs.
    assert _norm("Conversation-(6-plus)-favorite-food") == "conversation-6-plus-favorite-food"
    assert _norm("Cape-V-sentences-4-(v2)") == "cape-v-sentences-4-v2"


def test_banks_have_expected_counts():
    reg_dir = files("b2aiprep.prepare.resources").joinpath("task_registry")
    harvard = json.loads(reg_dir.joinpath("harvard_sentences_bank.json").read_text())
    assert len(harvard["lists"]) == 72
    assert all(len(v) == 10 for v in harvard["lists"].values())
    capev = json.loads(reg_dir.joinpath("cape_v_sentences_bank.json").read_text())
    assert len(capev["lists"]["v1"]) == 6 and len(capev["lists"]["v2"]) == 6
    reading = json.loads(reg_dir.joinpath("reading_passage_bank.json").read_text())
    assert len(reading["sentences"]) == 11
    repeating = json.loads(reg_dir.joinpath("repeating_sentences_bank.json").read_text())
    assert len(repeating["sentences"]) == 6
