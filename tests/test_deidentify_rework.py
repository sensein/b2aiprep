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

from b2aiprep.prepare.dataset import BIDSDataset, DispositionLevel, SessionLabels


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

    def test_error_when_no_disposition(self):
        fm = _field_map_df([
            {"column_name": "col_a", "delete": "NO"},
            {"column_name": "col_b", "delete": "YES"},
        ])
        df = pd.DataFrame({"col_a": [1], "col_b": [2]})
        with pytest.raises(ValueError, match="disposition"):
            BIDSDataset._drop_columns_by_disposition(df, field_map_df=fm)

    def test_columns_not_in_field_map_removed(self, caplog):
        """A column with no field-map row has no disposition, so it is never published."""
        fm = _field_map_df([
            {"column_name": "col_a", "disposition": "release", "delete": "NO"},
        ])
        df = pd.DataFrame({"participant_id": ["p1"], "col_a": [1], "pipeline_computed": [99]})
        with caplog.at_level(logging.WARNING):
            result, dropped = BIDSDataset._drop_columns_by_disposition(df, field_map_df=fm)
        assert list(result.columns) == ["participant_id", "col_a"]
        assert dropped == ["pipeline_computed"]
        assert any("pipeline_computed" in r.getMessage() for r in caplog.records)
        result, _ = BIDSDataset._drop_columns_by_disposition(
            df, field_map_df=fm, level=DispositionLevel.INTERNAL)
        assert "pipeline_computed" not in result.columns

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

    def test_pipeline_columns_need_a_field_map_row(self):
        """A pipeline-computed column is published only once the field map describes it."""
        fm = _field_map_df([
            {"column_name": "participant_id", "disposition": "release", "delete": "NO"},
        ])
        df = pd.DataFrame({
            "participant_id": ["p1"],
            "computed_by_pipeline": ["value"],
        })
        result, dropped = BIDSDataset._drop_columns_by_disposition(df, field_map_df=fm)
        assert "computed_by_pipeline" not in result.columns
        assert dropped == ["computed_by_pipeline"]

    def test_error_when_no_disposition_in_phenotype_context(self):
        """When field map has no disposition column, raises ValueError."""
        fm = _field_map_df([
            {"column_name": "age", "delete": "NO"},
            {"column_name": "zipcode", "delete": "YES"},
        ])
        df = pd.DataFrame({"age": ["30"], "zipcode": ["02139"]})
        with pytest.raises(ValueError, match="disposition"):
            BIDSDataset._drop_columns_by_disposition(df, field_map_df=fm)


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
            ses_df = pd.DataFrame({"record_id": [pid], "session_id": ["s1"], "session_index": ["1"]})
            ses_df.to_csv(bids / f"sub-{pid}" / "sessions.tsv", sep="\t", index=False)

        # Phenotype with all 3
        pheno_dir = bids / "phenotype"
        pheno_dir.mkdir()
        pheno_df = pd.DataFrame({
            "participant_id": ["p1", "p2", "p3"],
            "acid_reflux": ["Yes", "No", "Yes"],
        })
        pheno_df.to_csv(pheno_dir / "confounders.tsv", sep="\t", index=False)
        (pheno_dir / "confounders.json").write_text(json.dumps({}))

        # Template files
        (bids / "dataset_description.json").write_text(json.dumps({"Name": "test"}))

        dataset = BIDSDataset(bids)
        dataset.deidentify(outdir=out, deidentify_config_dir=config)

        # p3 should be excluded
        assert not (out / "sub-p3").exists()
        # Phenotype should only have p1 and p2
        result_pheno = pd.read_csv(out / "phenotype" / "confounders.tsv", sep="\t", dtype=str)
        assert set(result_pheno["participant_id"]) == {"p1", "p2"}

        # session_id_mapping.json generation is currently disabled
        assert not (out / "session_id_mapping.json").exists()


class TestSessionIndexAtIngest:

    def test_numbers_by_start_time_undated_last_duplicates_share(self):
        df = pd.DataFrame({
            "record_id": ["p1", "p1", "p1", "p1", "p2", "p1"],
            "redcap_repeat_instrument": ["Session"] * 5 + ["Acoustic Task"],
            "session_id": ["B", "A", "C", "B", "Z", None],
            "session_started_at": ["2025-03-01T10:00:00Z", "2025-01-01T10:00:00Z", None,
                                   "2025-03-01T10:00:00Z", "2024-01-01T00:00:00Z", None],
        })
        out = BIDSDataset._add_session_index(df)
        got = dict(zip(zip(out.record_id, out.session_id), out.session_index))
        assert got[("p1", "A")] == "1" and got[("p1", "B")] == "2" and got[("p1", "C")] == "3"
        assert got[("p2", "Z")] == "1"
        assert out.loc[5, "session_index"] is pd.NA
        assert list(out.loc[out.session_id == "B", "session_index"]) == ["2", "2"]


class TestSessionLabels:

    def _sessions(self):
        return pd.DataFrame({
            "session_id": ["AAAA1111-0000-0000-0000-000000000001", "BBBB2222-0000-0000-0000-000000000002",
                           "CCCC3333-0000-0000-0000-000000000003"],
            "session_index": ["1", "2", "3"],
        })

    def test_ordinal_numbers_released_without_gaps(self):
        ses = self._sessions()
        released = {ses.session_id[0], ses.session_id[2]}
        labels, order = BIDSDataset._session_labels(ses, SessionLabels.ORDINAL, released)
        assert labels == {ses.session_id[0]: "01", ses.session_id[2]: "02"}
        assert order == {ses.session_id[0]: 1, ses.session_id[2]: 2}

    def test_index_keeps_numbers_with_gaps(self):
        ses = self._sessions()
        released = {ses.session_id[0], ses.session_id[2]}
        labels, order = BIDSDataset._session_labels(ses, SessionLabels.INDEX, released)
        assert labels == {ses.session_id[0]: "01", ses.session_id[2]: "03"}
        assert order == {ses.session_id[0]: 1, ses.session_id[2]: 3}

    def test_uuid_lower_case_and_long_on_collision(self):
        ses = pd.DataFrame({"session_id": ["ABCD1234-0000-0000-0000-00000000000A",
                                           "ABCD1234-1111-0000-0000-00000000000B",
                                           "FFFF0000-0000-0000-0000-00000000000C"]})
        labels, _ = BIDSDataset._session_labels(ses, SessionLabels.UUID)
        assert labels == {ses.session_id[0]: "abcd123400000000", ses.session_id[1]: "abcd123411110000",
                          ses.session_id[2]: "ffff0000"}

    def test_ordinal_without_session_index_refuses(self):
        with pytest.raises(ValueError, match="session_index"):
            BIDSDataset._session_labels(pd.DataFrame({"session_id": ["x"]}), SessionLabels.ORDINAL)


class TestReleasedSessions:
    """A session is released with a file to publish or questionnaire rows; empty ones are not,
    and no original session ID reaches the output."""

    A, B, C = ("AAAA1111-0000-0000-0000-000000000001", "BBBB2222-0000-0000-0000-000000000002",
               "CCCC3333-0000-0000-0000-000000000003")

    def _deidentify(self, tmp_path, **kwargs):
        bids, config, out = tmp_path / "bids", tmp_path / "config", tmp_path / "out"
        config.mkdir(parents=True)
        (config / "participants_to_include.json").write_text(json.dumps(["p1"]))
        (config / "id_remapping.json").write_text(json.dumps({"p1": "900001"}))
        (config / "audio_filestems_to_remove.json").write_text(json.dumps([]))
        (config / "audio_tasks_to_include.json").write_text(json.dumps(["rainbow-passage"]))
        audio_dir = bids / "sub-p1" / f"ses-{self.A}" / "audio"
        audio_dir.mkdir(parents=True)
        stem = f"sub-p1_ses-{self.A}_task-rainbow-passage"
        (audio_dir / f"{stem}.wav").write_bytes(b"RIFF" + b"\x00" * 8192)
        (audio_dir / f"{stem}_recording-metadata.json").write_text(
            json.dumps({"record_id": "p1", "session_id": self.A}))
        pd.DataFrame({"record_id": ["p1"] * 3, "session_id": [self.A, self.B, self.C],
                      "session_index": ["1", "2", "3"], "session_status": ["Completed"] * 3}).to_csv(
            bids / "sub-p1" / "sessions.tsv", sep="\t", index=False)
        pheno = bids / "phenotype"
        pheno.mkdir()
        pd.DataFrame({"participant_id": ["p1"] * 3, "session_id": [self.A, self.B, self.C],
                      "session_index": ["1", "2", "3"], "session_status": ["Completed"] * 3}).to_csv(
            pheno / "session.tsv", sep="\t", index=False)
        (pheno / "session.json").write_text(json.dumps({}))
        pd.DataFrame({"participant_id": ["p1"], "confounders_session_id": [self.C], "acid_reflux": ["Yes"]}).to_csv(
            pheno / "confounders.tsv", sep="\t", index=False)
        (pheno / "confounders.json").write_text(json.dumps({}))
        (bids / "dataset_description.json").write_text(json.dumps({"Name": "test"}))
        BIDSDataset(bids).deidentify(outdir=out, deidentify_config_dir=config, **kwargs)
        return out

    def _no_original_ids(self, out):
        for f in out.rglob("*"):
            if f.is_file() and f.suffix in (".tsv", ".json"):
                text = f.read_text()
                assert not any(x in text or x.lower() in text for x in (self.A, self.B, self.C)), f

    def test_ordinal(self, tmp_path):
        out = self._deidentify(tmp_path, session_id_map=tmp_path / "internal" / "map.json")
        assert [d.name for d in (out / "sub-900001").glob("ses-*")] == ["ses-01"]
        ses = pd.read_csv(out / "sub-900001" / "sub-900001_sessions.tsv", sep="\t", dtype=str)
        assert list(ses.session_id) == ["01", "02"] and list(ses.session_index) == ["1", "2"]
        session = pd.read_csv(out / "phenotype" / "session.tsv", sep="\t", dtype=str)
        assert sorted(session.session_id) == ["01", "02"] and sorted(session.session_index) == ["1", "2"]
        conf = pd.read_csv(out / "phenotype" / "confounders.tsv", sep="\t", dtype=str)
        assert list(conf.confounders_session_id) == ["02"]
        self._no_original_ids(out)
        record = json.loads((tmp_path / "internal" / "map.json").read_text())
        assert record["session_labels"] == "ordinal"
        assert [(r["session_id"], r["released_session_label"]) for r in record["sessions"]] == [
            (self.A, "01"), (self.C, "02")]

    def test_index_leaves_gap(self, tmp_path):
        out = self._deidentify(tmp_path, session_labels=SessionLabels.INDEX)
        ses = pd.read_csv(out / "sub-900001" / "sub-900001_sessions.tsv", sep="\t", dtype=str)
        assert list(ses.session_id) == ["01", "03"] and list(ses.session_index) == ["1", "3"]
        conf = pd.read_csv(out / "phenotype" / "confounders.tsv", sep="\t", dtype=str)
        assert list(conf.confounders_session_id) == ["03"]
        self._no_original_ids(out)

    def test_uuid(self, tmp_path):
        out = self._deidentify(tmp_path, session_labels=SessionLabels.UUID)
        assert [d.name for d in (out / "sub-900001").glob("ses-*")] == ["ses-aaaa1111"]
        conf = pd.read_csv(out / "phenotype" / "confounders.tsv", sep="\t", dtype=str)
        assert list(conf.confounders_session_id) == ["cccc3333"]

    def test_crosswalk_between_label_schemes(self, tmp_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "session_label_crosswalk", Path(__file__).parents[1] / "scripts" / "session_label_crosswalk.py")
        xw = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(xw)
        self._deidentify(tmp_path / "a", session_labels=SessionLabels.INDEX, session_id_map=tmp_path / "index.json")
        self._deidentify(tmp_path / "b", session_id_map=tmp_path / "ordinal.json")
        table = xw.crosswalk(xw.load_map(tmp_path / "index.json"), xw.load_map(tmp_path / "ordinal.json"))
        assert table.to_dict("records") == [
            {"participant_id": "900001", "old_session_label": "01", "new_session_label": "01"},
            {"participant_id": "900001", "old_session_label": "03", "new_session_label": "02"},
        ]
        v31 = xw.crosswalk(xw.uuid_labels(xw.load_map(tmp_path / "ordinal.json")), xw.load_map(tmp_path / "ordinal.json"))
        assert list(v31.old_session_label) == ["aaaa1111", "cccc3333"]
        assert xw.main(["--old-uuid-labels", "--new", str(tmp_path / "ordinal.json"), "-o", str(tmp_path / "x.tsv")]) == 0
        assert not any(x in (tmp_path / "x.tsv").read_text() for x in (self.A, self.C))

    def test_session_id_map_inside_output_refused(self, tmp_path):
        with pytest.raises(ValueError, match="outside the output"):
            self._deidentify(tmp_path, session_id_map=tmp_path / "out" / "map.json")


class TestFormBookkeepingIsNotData:
    """A questionnaire row holding only its form's session/via/origin/duration has no answers."""

    def test_form_metadata_columns(self):
        cols = ["participant_id", "demographics_session_id", "demographics_via", "demographics_origin",
                "demographics_duration", "demographics_started_at", "marital_status", "employ_status"]
        assert BIDSDataset._form_metadata_columns(cols) == {
            "demographics_session_id", "demographics_via", "demographics_origin",
            "demographics_duration", "demographics_started_at"}

    def test_metadata_only_row_dropped(self):
        df = pd.DataFrame({
            "participant_id": ["p1", "p2"],
            "confounders_session_id": ["s1", "s2"],
            "confounders_via": ["Participant", "Participant"],
            "confounders_duration": ["30", "12"],
            "acid_reflux": ["Yes", None],
        })
        out = BIDSDataset._drop_rows_emptied_by_deidentify(df, "confounders")
        assert list(out.participant_id) == ["p1"]

    def test_metadata_only_row_releases_no_session(self, tmp_path):
        pheno = tmp_path / "phenotype"
        pheno.mkdir()
        pd.DataFrame({
            "participant_id": ["p1", "p1"],
            "confounders_session_id": ["S1", "S2"],
            "confounders_via": ["Participant", "Participant"],
            "acid_reflux": ["Yes", None],
        }).to_csv(pheno / "confounders.tsv", sep="\t", index=False)
        (pheno / "confounders.json").write_text(json.dumps({}))
        assert BIDSDataset._sessions_with_questionnaire_data(pheno) == {"S1"}


class TestParticipantFailureStopsRun:

    def test_one_failing_participant_raises(self, tmp_path):
        """A participant that errors must not vanish from the release like one with no audio."""
        bids, config, out = tmp_path / "bids", tmp_path / "config", tmp_path / "out"
        config.mkdir()
        (config / "participants_to_include.json").write_text(json.dumps(["p1", "p2"]))
        (config / "id_remapping.json").write_text(json.dumps({}))
        (config / "audio_filestems_to_remove.json").write_text(json.dumps([]))
        (config / "audio_tasks_to_include.json").write_text(json.dumps(["rainbow-passage"]))
        for pid in ("p1", "p2"):
            audio = bids / f"sub-{pid}" / "ses-s1" / "audio"
            audio.mkdir(parents=True)
            stem = f"sub-{pid}_ses-s1_task-rainbow-passage"
            (audio / f"{stem}.wav").write_bytes(b"RIFF" + b"\x00" * 8192)
            (audio / f"{stem}_recording-metadata.json").write_text(json.dumps({"record_id": pid, "session_id": "s1"}))
            ses = {"record_id": [pid], "session_id": ["s1"]}
            if pid == "p1":
                ses["session_index"] = ["1"]  # p2 lacks it, so ordinal labelling fails for p2 only
            pd.DataFrame(ses).to_csv(bids / f"sub-{pid}" / "sessions.tsv", sep="\t", index=False)
        (bids / "phenotype").mkdir()
        (bids / "dataset_description.json").write_text(json.dumps({"Name": "test"}))
        with pytest.raises(RuntimeError, match=r"failed for 1 participant.*p2"):
            BIDSDataset(bids).deidentify(outdir=out, deidentify_config_dir=config)


class TestLabelsSharedAcrossTiers:
    """A features-only session is released where features are, and withheld where they are not;
    either way the other sessions keep the same labels."""

    def _tree(self, root):
        import torch
        pdir = root / "sub-p1"
        for ses, task in (("S1", "free-speech-1"), ("S2", "rainbow-passage")):
            audio = pdir / f"ses-{ses}" / "audio"
            audio.mkdir(parents=True)
            stem = f"sub-p1_ses-{ses}_task-{task}"
            (audio / f"{stem}.wav").write_bytes(b"RIFF" + b"\x00" * 8192)
            (audio / f"{stem}_recording-metadata.json").write_text(json.dumps({"record_id": "p1", "session_id": ses}))
            torch.save({"opensmile": {"x": 1}}, audio / f"{stem}_features.pt")
        pd.DataFrame({"record_id": ["p1", "p1"], "session_id": ["S1", "S2"], "session_index": ["1", "2"]}).to_csv(
            pdir / "sessions.tsv", sep="\t", index=False)
        return pdir

    def test_registered_and_controlled_share_labels(self, tmp_path):
        pdir = self._tree(tmp_path / "in")
        registered, _, _ = BIDSDataset._deidentify_participant_files(
            pdir, tmp_path / "reg", {"p1": "900001"}, [], ["rainbow-passage"])
        controlled, order, _ = BIDSDataset._deidentify_participant_files(
            pdir, tmp_path / "con", {"p1": "900001"}, [], ["rainbow-passage"], skip_audio_features=True)
        assert registered == {"S1": "01", "S2": "02"}
        assert controlled == {"S2": "02"} and order == {"S2": 2}
        ses = pd.read_csv(tmp_path / "con" / "sub-900001" / "sub-900001_sessions.tsv", sep="\t", dtype=str)
        assert list(ses.session_id) == ["02"] and list(ses.session_index) == ["2"]


class TestSidecarDispositions:
    """Audio sidecar keys follow the audio_sidecar table's dispositions, as sessions.tsv follows
    the session table's; a key the table does not list is removed and logged."""

    def _deidentify(self, tmp_path, **kwargs):
        bids, config, out = tmp_path / "bids", tmp_path / "config", tmp_path / "out"
        config.mkdir(parents=True)
        (config / "participants_to_include.json").write_text(json.dumps(["p1"]))
        (config / "id_remapping.json").write_text(json.dumps({}))
        (config / "audio_filestems_to_remove.json").write_text(json.dumps([]))
        (config / "audio_tasks_to_include.json").write_text(json.dumps(["rainbow-passage"]))
        audio_dir = bids / "sub-p1" / "ses-s1" / "audio"
        audio_dir.mkdir(parents=True)
        stem = "sub-p1_ses-s1_task-rainbow-passage"
        (audio_dir / f"{stem}.wav").write_bytes(b"RIFF" + b"\x00" * 8192)
        (audio_dir / f"{stem}_recording-metadata.json").write_text(json.dumps({
            "record_id": "p1",
            "session_id": "s1",
            "recording_duration": 1.5,
            "recording_microphone": "Built-in",
            "task_name": "rainbow-passage",
            "not_a_sidecar_key": "x",
        }))
        pd.DataFrame({"record_id": ["p1"], "session_id": ["s1"], "session_index": ["1"]}).to_csv(
            bids / "sub-p1" / "sessions.tsv", sep="\t", index=False)
        (bids / "phenotype").mkdir()
        (bids / "dataset_description.json").write_text(json.dumps({"Name": "test"}))
        BIDSDataset(bids).deidentify(outdir=out, deidentify_config_dir=config, **kwargs)
        (sidecar,) = (out / "sub-p1").rglob("*.json")
        return json.loads(sidecar.read_text())

    def test_known_keys_kept_unknown_removed_and_logged(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            meta = self._deidentify(tmp_path)
        assert meta == {"participant_id": "p1", "session_id": "01", "recording_duration": 1.5,
                        "recording_microphone": "Built-in", "task_name": "rainbow-passage"}
        assert any("not_a_sidecar_key" in r.getMessage() for r in caplog.records)

    def test_internal_key_removed_at_release_kept_at_internal(self, tmp_path, monkeypatch):
        fm = BIDSDataset._load_reorganization_file(exclude_dropped=False)
        fm.loc[(fm.schema_name == "audio_sidecar") & (fm.column_name == "recording_microphone"),
               "disposition"] = "internal"
        monkeypatch.setattr(BIDSDataset, "_cached_field_map_df", fm)
        assert "recording_microphone" not in self._deidentify(tmp_path / "a")
        assert "recording_microphone" in self._deidentify(
            tmp_path / "b", disposition_level=DispositionLevel.INTERNAL)


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


    def test_source_column_name_normalized(self, tmp_path):
        """Verdicts using source_column_name are normalized to output names."""
        manifest = {
            "verdicts": [
                {"participant_id": "p1", "source_column_name": "old_src_name", "verdict": "safe"},
                {"participant_id": "p2", "column_name": "new_out_name", "verdict": "redact"},
            ]
        }
        (tmp_path / "column_value_reviews.json").write_text(json.dumps(manifest))
        field_map = pd.DataFrame({
            "column_name_source": ["old_src_name", "other_col"],
            "column_name": ["new_out_name", "other_col"],
        })
        result = BIDSDataset._load_column_value_reviews(tmp_path, field_map_df=field_map)
        assert ("p1", "new_out_name") in result
        assert result[("p1", "new_out_name")] == "safe"
        assert result[("p2", "new_out_name")] == "redact"

    def test_column_name_not_in_field_map_kept_as_is(self, tmp_path):
        """Verdicts with column names not in the field map are kept unchanged."""
        manifest = {
            "verdicts": [
                {"participant_id": "p1", "column_name": "unknown_col", "verdict": "safe"},
            ]
        }
        (tmp_path / "column_value_reviews.json").write_text(json.dumps(manifest))
        field_map = pd.DataFrame({
            "column_name_source": ["other"],
            "column_name": ["other"],
        })
        result = BIDSDataset._load_column_value_reviews(tmp_path, field_map_df=field_map)
        assert ("p1", "unknown_col") in result


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
