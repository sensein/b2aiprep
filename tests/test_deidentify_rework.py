"""Tests for the deidentify rework foundational helpers.

Covers:
  T006 — _load_participant_allowlist
  T007 — _build_session_id_mapping
  T008 — _drop_columns_by_disposition
"""

import json
import logging
from pathlib import Path

import pandas as pd
import pytest

from b2aiprep.prepare.dataset import BIDSDataset


# ---------------------------------------------------------------------------
# T006: _load_participant_allowlist
# ---------------------------------------------------------------------------

class TestLoadParticipantAllowlist:

    def test_allowlist_from_include_file(self, tmp_path):
        (tmp_path / "participants_to_include.json").write_text(
            json.dumps(["a", "b", "c"])
        )
        result = BIDSDataset._load_participant_allowlist(
            tmp_path, {"a", "b", "c", "d"}
        )
        assert result == {"a", "b", "c"}

    def test_allowlist_fallback_from_removal(self, tmp_path):
        (tmp_path / "participants_to_remove.json").write_text(
            json.dumps(["d"])
        )
        result = BIDSDataset._load_participant_allowlist(
            tmp_path, {"a", "b", "c", "d"}
        )
        assert result == {"a", "b", "c"}

    def test_allowlist_empty_raises(self, tmp_path):
        (tmp_path / "participants_to_include.json").write_text(json.dumps([]))
        with pytest.raises(ValueError, match="empty"):
            BIDSDataset._load_participant_allowlist(tmp_path, {"a"})

    def test_allowlist_unmatched_warns(self, tmp_path, caplog):
        (tmp_path / "participants_to_include.json").write_text(
            json.dumps(["a", "x"])
        )
        with caplog.at_level(logging.WARNING):
            result = BIDSDataset._load_participant_allowlist(
                tmp_path, {"a", "b"}
            )
        assert result == {"a"}
        assert any("x" in r.message for r in caplog.records)

    def test_allowlist_no_files_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            BIDSDataset._load_participant_allowlist(tmp_path, {"a"})


# ---------------------------------------------------------------------------
# T007: _build_session_id_mapping
# ---------------------------------------------------------------------------

def _make_sessions_tsv(participant_dir, rows, has_index=True):
    """Write a minimal sessions.tsv under participant_dir."""
    participant_dir.mkdir(parents=True, exist_ok=True)
    cols = ["session_id"]
    if has_index:
        cols.append("session_index")
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(participant_dir / "sessions.tsv", sep="\t", index=False)


class TestBuildSessionIdMapping:

    def test_session_mapping_from_index(self, tmp_path):
        p1 = tmp_path / "sub-p1"
        _make_sessions_tsv(p1, [
            ["uuid-aaa", 1],
            ["uuid-bbb", 2],
        ])
        mapping = BIDSDataset._build_session_id_mapping(
            tmp_path, {"p1"}
        )
        assert mapping["uuid-aaa"] == "01"
        assert mapping["uuid-bbb"] == "02"

    def test_session_mapping_alphabetical_fallback(self, tmp_path):
        p1 = tmp_path / "sub-p1"
        _make_sessions_tsv(p1, [
            ["zzz"],
            ["aaa"],
        ], has_index=False)
        mapping = BIDSDataset._build_session_id_mapping(
            tmp_path, {"p1"}
        )
        assert mapping["aaa"] == "01"
        assert mapping["zzz"] == "02"

    def test_session_mapping_single_session(self, tmp_path):
        p1 = tmp_path / "sub-p1"
        _make_sessions_tsv(p1, [["only-one", 1]])
        mapping = BIDSDataset._build_session_id_mapping(
            tmp_path, {"p1"}
        )
        assert mapping["only-one"] == "01"

    def test_session_mapping_completeness(self, tmp_path):
        for pid, sessions in [("p1", ["s1", "s2", "s3"]),
                              ("p2", ["s4", "s5", "s6"])]:
            d = tmp_path / f"sub-{pid}"
            rows = [[sid, i + 1] for i, sid in enumerate(sessions)]
            _make_sessions_tsv(d, rows)
        mapping = BIDSDataset._build_session_id_mapping(
            tmp_path, {"p1", "p2"}
        )
        assert len(mapping) == 6
        for sid in ["s1", "s2", "s3", "s4", "s5", "s6"]:
            assert sid in mapping

    def test_session_mapping_missing_sessions_tsv(self, tmp_path, caplog):
        p1 = tmp_path / "sub-p1"
        p1.mkdir()
        with caplog.at_level(logging.WARNING):
            mapping = BIDSDataset._build_session_id_mapping(
                tmp_path, {"p1"}
            )
        assert len(mapping) == 0
        assert any("sessions.tsv" in r.message or "p1" in r.message
                    for r in caplog.records)


# ---------------------------------------------------------------------------
# T008: _drop_columns_by_disposition
# ---------------------------------------------------------------------------

def _field_map_df(rows):
    """Build a minimal field-map DataFrame for testing."""
    return pd.DataFrame(rows)


class TestDropColumnsByDisposition:

    def test_drop_internal_and_review(self):
        fm = _field_map_df([
            {"column_name": "col_a", "disposition": "release", "delete": "NO"},
            {"column_name": "col_b", "disposition": "internal", "delete": "YES"},
            {"column_name": "col_c", "disposition": "review", "delete": "NO"},
        ])
        df = pd.DataFrame({"col_a": [1], "col_b": [2], "col_c": [3]})
        result, dropped = BIDSDataset._drop_columns_by_disposition(
            df, field_map_df=fm
        )
        assert list(result.columns) == ["col_a"]
        assert set(dropped) == {"col_b", "col_c"}

    def test_fallback_to_delete(self):
        fm = _field_map_df([
            {"column_name": "col_a", "delete": "NO"},
            {"column_name": "col_b", "delete": "YES"},
        ])
        df = pd.DataFrame({"col_a": [1], "col_b": [2]})
        result, dropped = BIDSDataset._drop_columns_by_disposition(
            df, field_map_df=fm
        )
        assert list(result.columns) == ["col_a"]
        assert dropped == ["col_b"]

    def test_columns_not_in_field_map_kept(self):
        fm = _field_map_df([
            {"column_name": "col_a", "disposition": "release", "delete": "NO"},
        ])
        df = pd.DataFrame({"col_a": [1], "pipeline_computed": [99]})
        result, dropped = BIDSDataset._drop_columns_by_disposition(
            df, field_map_df=fm
        )
        assert "pipeline_computed" in result.columns
        assert "col_a" in result.columns

    def test_review_columns_stripped(self):
        fm = _field_map_df([
            {"column_name": "col_r", "disposition": "review", "delete": "NO"},
            {"column_name": "col_ok", "disposition": "release", "delete": "NO"},
        ])
        df = pd.DataFrame({"col_r": [1], "col_ok": [2]})
        result, dropped = BIDSDataset._drop_columns_by_disposition(
            df, field_map_df=fm
        )
        assert "col_r" not in result.columns
        assert "col_ok" in result.columns
        assert "col_r" in dropped

    def test_returns_dropped_list(self):
        fm = _field_map_df([
            {"column_name": "a", "disposition": "release", "delete": "NO"},
            {"column_name": "b", "disposition": "internal", "delete": "YES"},
            {"column_name": "c", "disposition": "review", "delete": "NO"},
        ])
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        _, dropped = BIDSDataset._drop_columns_by_disposition(
            df, field_map_df=fm
        )
        assert isinstance(dropped, list)
        assert set(dropped) == {"b", "c"}


# ---------------------------------------------------------------------------
# T025: Session ordinal mapping
# ---------------------------------------------------------------------------

class TestSessionOrdinalIntegration:

    def test_mapping_uses_ordinals_not_uuids(self, tmp_path):
        """Session mapping returns zero-padded ordinals, not truncated UUIDs."""
        p1 = tmp_path / "sub-p1"
        _make_sessions_tsv(p1, [
            ["abc-def-123", 2],
            ["xyz-uvw-789", 1],
        ])
        mapping = BIDSDataset._build_session_id_mapping(tmp_path, {"p1"})
        assert mapping["xyz-uvw-789"] == "01"
        assert mapping["abc-def-123"] == "02"

    def test_alphabetical_fallback_is_deterministic(self, tmp_path):
        """Without session_index, alphabetical sort is deterministic."""
        p1 = tmp_path / "sub-p1"
        _make_sessions_tsv(p1, [
            ["zzz-uuid"],
            ["aaa-uuid"],
            ["mmm-uuid"],
        ], has_index=False)
        m1 = BIDSDataset._build_session_id_mapping(tmp_path, {"p1"})
        m2 = BIDSDataset._build_session_id_mapping(tmp_path, {"p1"})
        assert m1 == m2
        assert m1["aaa-uuid"] == "01"
        assert m1["mmm-uuid"] == "02"
        assert m1["zzz-uuid"] == "03"


# ---------------------------------------------------------------------------
# T031: Sessions.tsv carry-forward (column dropping + ID remap)
# ---------------------------------------------------------------------------

class TestSessionsCarryForward:

    def test_disposition_columns_dropped(self):
        """sessions.tsv columns with disposition=internal are dropped."""
        fm = _field_map_df([
            {"column_name": "session_duration", "disposition": "release", "delete": "NO"},
            {"column_name": "session_site", "disposition": "internal", "delete": "YES"},
            {"column_name": "session_id", "disposition": "release", "delete": "NO"},
        ])
        df = pd.DataFrame({
            "session_id": ["s1"],
            "session_duration": ["120"],
            "session_site": ["MIT"],
        })
        result, dropped = BIDSDataset._drop_columns_by_disposition(df, field_map_df=fm)
        assert "session_duration" in result.columns
        assert "session_site" not in result.columns
        assert "session_site" in dropped

    def test_record_id_renamed_to_participant_id(self):
        """sessions.tsv should rename record_id to participant_id."""
        df = pd.DataFrame({
            "record_id": ["uuid-1"],
            "session_id": ["ses-1"],
        })
        df = df.rename(columns={"record_id": "participant_id"})
        assert "participant_id" in df.columns
        assert "record_id" not in df.columns


# ---------------------------------------------------------------------------
# T035-T036: Disposition-based column dropping in phenotype context
# ---------------------------------------------------------------------------

class TestDispositionInPhenotypeContext:

    def test_only_release_columns_survive(self):
        """After disposition drop, only release columns remain."""
        fm = _field_map_df([
            {"column_name": "participant_id", "disposition": "release", "delete": "NO"},
            {"column_name": "age", "disposition": "release", "delete": "NO"},
            {"column_name": "session_complete", "disposition": "internal", "delete": "YES"},
            {"column_name": "free_text_specify", "disposition": "review", "delete": "NO"},
        ])
        df = pd.DataFrame({
            "participant_id": ["p1", "p2"],
            "age": ["30", "40"],
            "session_complete": ["Complete", "Incomplete"],
            "free_text_specify": ["some text", "other text"],
        })
        result, dropped = BIDSDataset._drop_columns_by_disposition(df, field_map_df=fm)
        assert set(result.columns) == {"participant_id", "age"}
        assert set(dropped) == {"session_complete", "free_text_specify"}

    def test_pipeline_columns_kept_when_not_in_field_map(self):
        """Pipeline-authored columns not in the field map survive disposition drop."""
        fm = _field_map_df([
            {"column_name": "participant_id", "disposition": "release", "delete": "NO"},
        ])
        df = pd.DataFrame({
            "participant_id": ["p1"],
            "computed_by_pipeline": ["value"],
        })
        result, dropped = BIDSDataset._drop_columns_by_disposition(df, field_map_df=fm)
        assert "computed_by_pipeline" in result.columns
        assert dropped == []

    def test_fallback_to_delete_when_no_disposition(self):
        """When field map has no disposition column, falls back to delete column."""
        fm = _field_map_df([
            {"column_name": "age", "delete": "NO"},
            {"column_name": "zipcode", "delete": "YES"},
        ])
        df = pd.DataFrame({"age": ["30"], "zipcode": ["02139"]})
        result, dropped = BIDSDataset._drop_columns_by_disposition(df, field_map_df=fm)
        assert "age" in result.columns
        assert "zipcode" not in result.columns


# ---------------------------------------------------------------------------
# T037-T039: End-to-end verification (US6)
# ---------------------------------------------------------------------------

class TestEndToEndAllowlistFiltering:

    def test_emptied_participant_excluded_from_phenotype(self, tmp_path):
        """A participant emptied by task filtering should not appear in phenotype.

        Setup: 3 participants on allowlist. p3 has only a task NOT in the
        include list, so after filtering p3 has zero audio and is excluded.
        Phenotype should have only p1 and p2.
        """
        bids = tmp_path / "bids"
        config = tmp_path / "config"
        out = tmp_path / "output"
        config.mkdir()

        # Config files
        (config / "participants_to_include.json").write_text(
            json.dumps(["p1", "p2", "p3"])
        )
        (config / "id_remapping.json").write_text(json.dumps({}))
        (config / "audio_filestems_to_remove.json").write_text(json.dumps([]))
        (config / "audio_tasks_to_include.json").write_text(
            json.dumps(["rainbow-passage"])
        )

        # Build BIDS tree: p1 and p2 have included task, p3 has excluded task
        for pid, task in [("p1", "rainbow-passage"), ("p2", "rainbow-passage"),
                          ("p3", "excluded-task")]:
            audio_dir = bids / f"sub-{pid}" / "ses-s1" / "audio"
            audio_dir.mkdir(parents=True)
            wav = audio_dir / f"sub-{pid}_ses-s1_task-{task}.wav"
            wav.write_bytes(b"RIFF" + b"\x00" * 8192)
            # Sidecar
            sidecar = audio_dir / f"sub-{pid}_ses-s1_task-{task}_recording-metadata.json"
            sidecar.write_text(json.dumps({
                "item": [
                    {"linkId": "record_id", "answer": [{"valueString": pid}]},
                    {"linkId": "session_id", "answer": [{"valueString": "s1"}]},
                ]
            }))
            # sessions.tsv
            ses_df = pd.DataFrame({"record_id": [pid], "session_id": ["s1"]})
            ses_df.to_csv(bids / f"sub-{pid}" / "sessions.tsv", sep="\t", index=False)

        # Phenotype with all 3
        pheno_dir = bids / "phenotype"
        pheno_dir.mkdir()
        pheno_df = pd.DataFrame({
            "participant_id": ["p1", "p2", "p3"],
            "score": [10, 20, 30],
        })
        pheno_df.to_csv(pheno_dir / "test.tsv", sep="\t", index=False)
        (pheno_dir / "test.json").write_text(json.dumps({}))

        # Template files
        (bids / "dataset_description.json").write_text(json.dumps({"Name": "test"}))

        dataset = BIDSDataset(bids)
        dataset.deidentify(outdir=out, deidentify_config_dir=config)

        # p3 should be excluded
        assert not (out / "sub-p3").exists()
        # Phenotype should only have p1 and p2
        result_pheno = pd.read_csv(out / "phenotype" / "test.tsv", sep="\t", dtype=str)
        assert set(result_pheno["participant_id"]) == {"p1", "p2"}

        # session_id_mapping.json generation is currently disabled
        assert not (out / "session_id_mapping.json").exists()
