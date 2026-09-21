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

    def test_session_mapping_truncated_uuid_fallback(self, tmp_path):
        """Without session_index, falls back to truncated UUID (reduce_id_length)."""
        p1 = tmp_path / "sub-p1"
        _make_sessions_tsv(p1, [
            ["abcd1234-5678-9abc-def0-111111111111"],
            ["ffffffff-eeee-dddd-cccc-bbbbbbbbbbbb"],
        ], has_index=False)
        mapping = BIDSDataset._build_session_id_mapping(
            tmp_path, {"p1"}
        )
        assert mapping["abcd1234-5678-9abc-def0-111111111111"] == "abcd1234"
        assert mapping["ffffffff-eeee-dddd-cccc-bbbbbbbbbbbb"] == "ffffffff"

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

    def test_truncated_uuid_fallback_is_deterministic(self, tmp_path):
        """Without session_index, truncated UUID mapping is deterministic."""
        p1 = tmp_path / "sub-p1"
        _make_sessions_tsv(p1, [
            ["zzz00000-1111-2222-3333-444444444444"],
            ["aaa00000-5555-6666-7777-888888888888"],
            ["mmm00000-9999-aaaa-bbbb-cccccccccccc"],
        ], has_index=False)
        m1 = BIDSDataset._build_session_id_mapping(tmp_path, {"p1"})
        m2 = BIDSDataset._build_session_id_mapping(tmp_path, {"p1"})
        assert m1 == m2
        assert m1["aaa00000-5555-6666-7777-888888888888"] == "aaa00000"
        assert m1["mmm00000-9999-aaaa-bbbb-cccccccccccc"] == "mmm00000"
        assert m1["zzz00000-1111-2222-3333-444444444444"] == "zzz00000"


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


# ---------------------------------------------------------------------------
# Column value review manifest tests
# ---------------------------------------------------------------------------

class TestLoadColumnValueReviews:

    def test_loads_verdicts(self, tmp_path):
        manifest = {
            "verdicts": [
                {"participant_id": "p1", "column_name": "col_a", "verdict": "safe"},
                {"participant_id": "p2", "column_name": "col_a", "verdict": "redact"},
            ]
        }
        (tmp_path / "column_value_reviews.json").write_text(json.dumps(manifest))
        result = BIDSDataset._load_column_value_reviews(tmp_path)
        assert result[("p1", "col_a")] == "safe"
        assert result[("p2", "col_a")] == "redact"

    def test_missing_file_returns_empty(self, tmp_path):
        result = BIDSDataset._load_column_value_reviews(tmp_path)
        assert result == {}

    def test_duplicate_last_wins(self, tmp_path, caplog):
        manifest = {
            "verdicts": [
                {"participant_id": "p1", "column_name": "col_a", "verdict": "safe"},
                {"participant_id": "p1", "column_name": "col_a", "verdict": "redact"},
            ]
        }
        (tmp_path / "column_value_reviews.json").write_text(json.dumps(manifest))
        with caplog.at_level(logging.WARNING):
            result = BIDSDataset._load_column_value_reviews(tmp_path)
        assert result[("p1", "col_a")] == "redact"
        assert any("duplicate" in r.message.lower() for r in caplog.records)


class TestApplyColumnValueReviews:

    def test_safe_passes_through(self):
        df = pd.DataFrame({
            "participant_id": ["p1", "p2"],
            "score": [10, 20],
            "free_text": ["safe value", "also safe"],
        })
        verdicts = {
            ("p1", "free_text"): "safe",
            ("p2", "free_text"): "safe",
        }
        result, dropped = BIDSDataset._apply_column_value_reviews(
            df, {"free_text"}, verdicts
        )
        assert result.loc[result["participant_id"] == "p1", "free_text"].values[0] == "safe value"
        assert dropped == []

    def test_redact_replaces(self):
        df = pd.DataFrame({
            "participant_id": ["p1", "p2"],
            "free_text": ["Dr. Smith", "safe"],
        })
        verdicts = {
            ("p1", "free_text"): "redact",
            ("p2", "free_text"): "safe",
        }
        result, _ = BIDSDataset._apply_column_value_reviews(
            df, {"free_text"}, verdicts
        )
        assert result.loc[result["participant_id"] == "p1", "free_text"].values[0] == "[REDACTED]"
        assert result.loc[result["participant_id"] == "p2", "free_text"].values[0] == "safe"

    def test_drop_nulls_value(self):
        df = pd.DataFrame({
            "participant_id": ["p1"],
            "free_text": ["PII content"],
        })
        verdicts = {("p1", "free_text"): "drop"}
        result, _ = BIDSDataset._apply_column_value_reviews(
            df, {"free_text"}, verdicts
        )
        assert pd.isna(result.loc[0, "free_text"])

    def test_no_verdict_nulls_value(self):
        df = pd.DataFrame({
            "participant_id": ["p1", "p2"],
            "free_text": ["reviewed", "unreviewed"],
        })
        verdicts = {("p1", "free_text"): "safe"}  # p2 has no verdict
        result, _ = BIDSDataset._apply_column_value_reviews(
            df, {"free_text"}, verdicts
        )
        assert result.loc[result["participant_id"] == "p1", "free_text"].values[0] == "reviewed"
        assert pd.isna(result.loc[result["participant_id"] == "p2", "free_text"].values[0])

    def test_no_verdicts_for_column_drops_it(self):
        df = pd.DataFrame({
            "participant_id": ["p1"],
            "reviewed_col": ["has verdict"],
            "unreviewed_col": ["no verdicts at all"],
        })
        verdicts = {("p1", "reviewed_col"): "safe"}
        result, dropped = BIDSDataset._apply_column_value_reviews(
            df, {"reviewed_col", "unreviewed_col"}, verdicts
        )
        assert "reviewed_col" in result.columns
        assert "unreviewed_col" not in result.columns
        assert "unreviewed_col" in dropped

    def test_empty_verdicts_drops_all_review_columns(self):
        df = pd.DataFrame({
            "participant_id": ["p1"],
            "review_col": ["some value"],
        })
        result, dropped = BIDSDataset._apply_column_value_reviews(
            df, {"review_col"}, {}
        )
        assert "review_col" not in result.columns
        assert "review_col" in dropped
