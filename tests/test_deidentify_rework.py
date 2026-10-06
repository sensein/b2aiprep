"""Deidentify: the participant allowlist, column dispositions and access tiers, released sessions
and their labels, sidecar keys, removal lists, and column value reviews."""

import importlib.util
import json
import logging
from pathlib import Path

import pandas as pd
import pytest
import torch

from b2aiprep.prepare.dataset import AccessTier, BIDSDataset, DispositionLevel, SessionLabels

# three session IDs of one participant, in session_index order
A, B, C = ("AAAA1111-0000-0000-0000-000000000001", "BBBB2222-0000-0000-0000-000000000002",
           "CCCC3333-0000-0000-0000-000000000003")


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

    def test_names_to_drop_at_each_level(self):
        """drop is removed at every level; shifted dates only with keep_date_shifted."""
        fm = _field_map_df([
            {"column_name": c, "disposition": d, "date_shift": ds, "access_tier": ""}
            for c, d, ds in [("a_release", "release", "NO"), ("b_internal", "internal", "NO"),
                             ("c_drop", "drop", "NO"), ("d_drop_date", "drop", "YES"),
                             ("e_internal_date", "internal", "YES")]])
        drop = BIDSDataset._names_to_drop_at_level
        assert drop(fm, DispositionLevel.INTERNAL) == {"c_drop", "d_drop_date"}
        assert drop(fm, DispositionLevel.REVIEW, keep_date_shifted=True) == {"b_internal", "c_drop", "d_drop_date"}
        assert drop(fm, DispositionLevel.RELEASE) == {"b_internal", "c_drop", "d_drop_date", "e_internal_date"}

    def test_controlled_only_columns_are_kept_only_in_the_controlled_tier(self):
        fm = _field_map_df([{"column_name": c, "disposition": d, "date_shift": "NO", "access_tier": t}
                            for c, d, t in [("a", "release", ""), ("b", "release", "controlled"),
                                            ("c", "internal", "")]])
        drop = BIDSDataset._names_to_drop_at_level
        assert drop(fm, DispositionLevel.RELEASE) == {"b", "c"}  # default tier is the narrower one
        assert drop(fm, DispositionLevel.RELEASE, access_tier=AccessTier.CONTROLLED) == {"c"}
        assert drop(fm, DispositionLevel.INTERNAL, access_tier=AccessTier.REGISTERED) == set()  # QA keeps all

    def test_disposition_is_matched_within_the_table(self):
        """self_reported_* is released in eligibility and internal in enrollment."""
        fm = _field_map_df([
            {"schema_name": "eligibility", "column_name": "self_reported_asthma", "disposition": "release"},
            {"schema_name": "enrollment", "column_name": "self_reported_asthma", "disposition": "internal"}])
        df = pd.DataFrame({"participant_id": ["p"], "self_reported_asthma": ["Yes"]})
        kept, dropped = BIDSDataset._drop_columns_by_disposition(df, fm, schema_name="eligibility")
        assert "self_reported_asthma" in kept.columns and not dropped
        _, dropped = BIDSDataset._drop_columns_by_disposition(df, fm, schema_name="enrollment")
        assert dropped == ["self_reported_asthma"]

    def test_unknown_table_refuses_instead_of_keeping_everything(self):
        fm = _field_map_df([{"schema_name": "demographics", "column_name": "zipcode", "disposition": "internal"}])
        df = pd.DataFrame({"participant_id": ["p"], "zipcode": ["02139"]})
        with pytest.raises(ValueError, match="no rows in the field map"):
            BIDSDataset._drop_columns_by_disposition(df, fm, schema_name="renamed_table")


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
        # identity pseudonyms: deidentify refuses an allowlisted participant without one
        (config / "id_remapping.json").write_text(json.dumps({p: p for p in ("p1", "p2", "p3")}))
        (config / "deidentify_settings.json").write_text(json.dumps({"access_tier": "registered"}))
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

    def test_allowlisted_participant_without_a_pseudonym_is_refused(self, tmp_path, make_deid_tree):
        bids, config = make_deid_tree({"p1": [("s1", "rainbow-passage")]}, pseudonyms={})
        with pytest.raises(ValueError, match="1 allowlisted participant.*no pseudonym.*p1"):
            BIDSDataset(bids).deidentify(outdir=tmp_path / "out", deidentify_config_dir=config)
        assert list((tmp_path / "out").iterdir()) == []


class TestSessionLabels:

    def _sessions(self):
        return pd.DataFrame({"session_id": [A, B, C], "session_index": ["1", "2", "3"]})

    @pytest.mark.parametrize("mode, labels, order", [
        (SessionLabels.ORDINAL, {A: "01", C: "02"}, {A: 1, C: 2}),   # numbered without gaps
        (SessionLabels.INDEX, {A: "01", C: "03"}, {A: 1, C: 3}),     # session_index, gaps kept
    ])
    def test_numbered_labels(self, mode, labels, order):
        assert BIDSDataset._session_labels(self._sessions(), mode, {A, C}) == (labels, order)

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


def _released_sessions_tree(make_deid_tree, root):
    """p1: audio in A, nothing in B, a questionnaire row in C."""
    sessions = pd.DataFrame({"record_id": ["p1"] * 3, "session_id": [A, B, C],
                             "session_index": ["1", "2", "3"], "session_status": ["Completed"] * 3})
    return make_deid_tree(
        {"p1": [(A, "rainbow-passage")]}, root=root, sessions={"p1": sessions}, pseudonyms={"p1": "900001"},
        tables={"session": sessions.rename(columns={"record_id": "participant_id"}),
                "confounders": pd.DataFrame({"participant_id": ["p1"], "confounders_session_id": [C],
                                             "acid_reflux": ["Yes"]})})


class TestReleasedSessions:
    """A session is released with a file to publish or questionnaire rows; empty ones are not,
    and no original session ID reaches the output."""

    @pytest.mark.parametrize("mode, audio_dir, session_labels, questionnaire_label", [
        (SessionLabels.ORDINAL, "ses-01", ["01", "02"], "02"),
        (SessionLabels.INDEX, "ses-01", ["01", "03"], "03"),
        (SessionLabels.UUID, "ses-aaaa1111", ["aaaa1111", "cccc3333"], "cccc3333"),
    ])
    def test_labels_and_no_original_ids(self, tmp_path, make_deid_tree, mode, audio_dir, session_labels,
                                        questionnaire_label):
        bids, config = _released_sessions_tree(make_deid_tree, tmp_path)
        out, session_map = tmp_path / "out", tmp_path / "internal" / "map.json"
        BIDSDataset(bids).deidentify(outdir=out, deidentify_config_dir=config, session_labels=mode,
                                     session_id_map=session_map)
        assert [d.name for d in (out / "sub-900001").glob("ses-*")] == [audio_dir]
        ses = pd.read_csv(out / "sub-900001" / "sub-900001_sessions.tsv", sep="\t", dtype=str)
        assert list(ses.session_id) == session_labels
        session = pd.read_csv(out / "phenotype" / "session.tsv", sep="\t", dtype=str)
        assert sorted(session.session_id) == session_labels
        conf = pd.read_csv(out / "phenotype" / "confounders.tsv", sep="\t", dtype=str)
        assert list(conf.confounders_session_id) == [questionnaire_label]
        for f in out.rglob("*"):
            if f.is_file() and f.suffix in (".tsv", ".json"):
                text = f.read_text()
                assert not any(x in text or x.lower() in text for x in (A, B, C)), f
        record = json.loads(session_map.read_text())
        assert record["session_labels"] == mode.value
        assert [(r["session_id"], r["released_session_label"]) for r in record["sessions"]] == [
            (A, session_labels[0]), (C, session_labels[1])]

    def test_session_id_map_inside_output_refused(self, tmp_path):
        """Checked before anything is read or written."""
        (tmp_path / "bids").mkdir()
        with pytest.raises(ValueError, match="outside the output"):
            BIDSDataset(tmp_path / "bids").deidentify(outdir=tmp_path / "out", deidentify_config_dir=tmp_path,
                                                      session_id_map=tmp_path / "out" / "map.json")
        assert not (tmp_path / "out").exists()


@pytest.fixture(scope="module")
def crosswalk():
    spec = importlib.util.spec_from_file_location(
        "session_label_crosswalk", Path(__file__).parents[1] / "scripts" / "session_label_crosswalk.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSessionLabelCrosswalk:
    """scripts/session_label_crosswalk.py, on the session maps deidentify writes."""

    def _maps(self, tmp_path, make_deid_tree):
        for name, mode in (("index", SessionLabels.INDEX), ("ordinal", SessionLabels.ORDINAL)):
            bids, config = _released_sessions_tree(make_deid_tree, tmp_path / name)
            BIDSDataset(bids).deidentify(outdir=tmp_path / name / "out", deidentify_config_dir=config,
                                         session_labels=mode, session_id_map=tmp_path / f"{name}.json")
        return tmp_path / "index.json", tmp_path / "ordinal.json"

    def test_crosswalk_between_label_schemes(self, tmp_path, make_deid_tree, crosswalk):
        index, ordinal = self._maps(tmp_path, make_deid_tree)
        table = crosswalk.crosswalk(crosswalk.load_map(index), crosswalk.load_map(ordinal))
        assert table.to_dict("records") == [
            {"participant_id": "900001", "old_session_label": "01", "new_session_label": "01"},
            {"participant_id": "900001", "old_session_label": "03", "new_session_label": "02"},
        ]
        v31 = crosswalk.crosswalk(crosswalk.uuid_labels(crosswalk.load_map(ordinal)), crosswalk.load_map(ordinal))
        assert list(v31.old_session_label) == ["aaaa1111", "cccc3333"]

    def test_main_writes_no_original_ids(self, tmp_path, make_deid_tree, crosswalk):
        _, ordinal = self._maps(tmp_path, make_deid_tree)
        assert crosswalk.main(["--old-uuid-labels", "--new", str(ordinal), "-o", str(tmp_path / "x.tsv")]) == 0
        assert not any(x in (tmp_path / "x.tsv").read_text() for x in (A, C))


class TestFormBookkeepingIsNotData:
    """Only REDCap-generated and form bookkeeping columns are not data; a row holding nothing else is
    dropped after deidentify, while a pipeline-derived value (e.g. state_province) is data."""

    def test_form_metadata_columns(self):
        cols = ["participant_id", "demographics_session_id", "demographics_via", "demographics_origin",
                "demographics_duration", "demographics_started_at", "marital_status", "employ_status"]
        assert BIDSDataset._form_metadata_columns(cols) == {
            "demographics_session_id", "demographics_via", "demographics_origin",
            "demographics_duration", "demographics_started_at"}

    @pytest.mark.parametrize("schema, df, kept", [
        ("confounders", pd.DataFrame({"participant_id": ["p1", "p2"], "confounders_session_id": ["s1", "s2"],
                                      "confounders_via": ["Participant"] * 2, "confounders_duration": ["30", "12"],
                                      "acid_reflux": ["Yes", None]}), ["p1"]),
        ("demographics", pd.DataFrame({"participant_id": ["p1", "p2"], "demographics_session_id": ["01", "01"],
                                       "state_province": ["Ontario", None],
                                       "demographics_duration": ["120", "95"]}), ["p1"]),
    ], ids=["answer-kept-metadata-only-dropped", "derived-value-kept"])
    def test_rows_with_only_bookkeeping_are_dropped(self, schema, df, kept):
        assert list(BIDSDataset._drop_rows_emptied_by_deidentify(df, schema).participant_id) == kept

    def test_metadata_only_row_releases_no_session(self, tmp_path):
        pheno = tmp_path / "phenotype"
        pheno.mkdir()
        pd.DataFrame({
            "participant_id": ["p1", "p1"],
            "confounders_session_id": ["S1", "S2"],
            "acid_reflux": ["Yes", None],
        }).to_csv(pheno / "confounders.tsv", sep="\t", index=False)
        (pheno / "confounders.json").write_text(json.dumps({}))
        assert BIDSDataset._sessions_with_questionnaire_data(pheno) == {"S1"}


class TestParticipantFailureStopsRun:

    def test_one_failing_participant_raises(self, tmp_path, make_deid_tree):
        """A participant that errors must not vanish from the release like one with no audio."""
        bids, config = make_deid_tree(
            {"p1": [("s1", "rainbow-passage")], "p2": [("s1", "rainbow-passage")]},
            # no session_index for p2, so ordinal labelling fails for p2 only
            sessions={"p2": pd.DataFrame({"record_id": ["p2"], "session_id": ["s1"]})})
        with pytest.raises(RuntimeError, match=r"failed for 1 participant.*p2"):
            BIDSDataset(bids).deidentify(outdir=tmp_path / "out", deidentify_config_dir=config)


class TestLabelsSharedAcrossTiers:
    """Labels are numbered over every session some tier could release, so a session one build
    withholds leaves a gap instead of shifting the later labels."""

    def test_features_only_session_withheld_without_features(self, tmp_path, make_deid_tree):
        """S1 has only features to publish (its task's audio is not released)."""
        bids, _ = make_deid_tree({"p1": [("S1", "free-speech-1"), ("S2", "rainbow-passage")]}, features=True)
        pdir = bids / "sub-p1"
        with_features, _, _, _ = BIDSDataset._deidentify_participant_files(
            pdir, tmp_path / "a", {"p1": "900001"}, [], ["rainbow-passage"])
        without, order, _, _ = BIDSDataset._deidentify_participant_files(
            pdir, tmp_path / "b", {"p1": "900001"}, [], ["rainbow-passage"], skip_audio_features=True)
        assert with_features == {"S1": "01", "S2": "02"}
        assert without == {"S2": "02"} and order == {"S2": 2}
        ses = pd.read_csv(tmp_path / "b" / "sub-900001" / "sub-900001_sessions.tsv", sep="\t", dtype=str)
        assert list(ses.session_id) == ["02"] and list(ses.session_index) == ["2"]

    def test_participant_with_only_feature_output_is_kept(self, tmp_path, make_deid_tree):
        """Recordings whose task audio is not released still publish stripped features."""
        bids, _ = make_deid_tree({"p1": [("S1", "free-speech-1")]}, features=True)
        labels, _, _, _ = BIDSDataset._deidentify_participant_files(
            bids / "sub-p1", tmp_path / "out", {"p1": "900001"}, [], ["noisy-sounds-*"])
        assert labels == {"S1": "01"}
        assert list((tmp_path / "out").rglob("*free-speech-1_features.pt"))

    # other_voice_activity is a review column and ever_alcohol_rehab controlled-only (shipped field map)
    @pytest.mark.parametrize("column, value, narrow, wide", [
        ("other_voice_activity", "Podcaster", {},
         {"column_value_reviews.json": {"verdicts": [
             {"participant_id": "p1", "column_name": "other_voice_activity", "verdict": "safe"}]}}),
        ("ever_alcohol_rehab", "No", {}, {"deidentify_settings.json": {"access_tier": "controlled"}}),
    ], ids=["review-column", "controlled-only-column"])
    def test_session_with_only_withheld_data_keeps_later_labels(self, tmp_path, make_deid_tree, column, value,
                                                                narrow, wide):
        """s2 holds only one answer, which the narrow build withholds and the wide one publishes."""
        labels = {}
        for name, config_files in (("narrow", narrow), ("wide", wide)):
            bids, config = make_deid_tree(
                {"p1": [("s1", "rainbow-passage"), ("s3", "rainbow-passage")]}, root=tmp_path / name,
                sessions={"p1": ["s1", "s2", "s3"]}, config_files=config_files,
                tables={"confounders/confounders": pd.DataFrame(
                    {"participant_id": ["p1"], "confounders_session_id": ["s2"], column: [value]})})
            out = tmp_path / name / "out"
            BIDSDataset(bids).deidentify(outdir=out, deidentify_config_dir=config)
            (sessions,) = list((out / "sub-900000").glob("*sessions.tsv"))
            labels[name] = list(BIDSDataset._read_tsv_as_written(sessions)["session_id"])
        assert labels == {"narrow": ["01", "03"], "wide": ["01", "02", "03"]}


class TestSidecarDispositions:
    """Audio sidecar keys follow the audio_sidecar table's dispositions, as sessions.tsv follows
    the session table's; a key the table does not list is removed and logged."""

    def _deidentify(self, root, make_deid_tree, **kwargs):
        bids, config = make_deid_tree({"p1": [("s1", "rainbow-passage", {
            "recording_duration": 1.5, "recording_microphone": "Built-in", "task_name": "rainbow-passage",
            "not_a_sidecar_key": "x"})]}, root=root, pseudonyms={"p1": "p1"})
        BIDSDataset(bids).deidentify(outdir=root / "out", deidentify_config_dir=config, **kwargs)
        (sidecar,) = (root / "out" / "sub-p1").rglob("*.json")
        return json.loads(sidecar.read_text())

    def test_known_keys_kept_unknown_removed_and_logged(self, tmp_path, make_deid_tree, caplog):
        with caplog.at_level(logging.WARNING):
            meta = self._deidentify(tmp_path, make_deid_tree)
        assert meta == {"participant_id": "p1", "session_id": "01", "recording_duration": 1.5,
                        "recording_microphone": "Built-in", "task_name": "rainbow-passage"}
        assert any("not_a_sidecar_key" in r.getMessage() for r in caplog.records)

    def test_internal_key_removed_at_release_kept_at_internal(self, tmp_path, make_deid_tree, monkeypatch):
        # deidentify reads the field map through this cache; no public way to substitute one
        fm = BIDSDataset._load_reorganization_file(exclude_dropped=False)
        fm.loc[(fm.schema_name == "audio_sidecar") & (fm.column_name == "recording_microphone"),
               "disposition"] = "internal"
        monkeypatch.setattr(BIDSDataset, "_cached_field_map_df", fm)
        assert "recording_microphone" not in self._deidentify(tmp_path / "a", make_deid_tree)
        assert "recording_microphone" in self._deidentify(
            tmp_path / "b", make_deid_tree, disposition_level=DispositionLevel.INTERNAL)


def _sidecar_tree(root, pid="p1", ses="S1", recordings=(("rec-A", "noisy-sounds-1"), ("rec-B", "noisy-sounds-2"))):
    """Sidecars only (no audio): for runs with skip_audio=True."""
    audio = root / f"sub-{pid}" / f"ses-{ses}" / "audio"
    audio.mkdir(parents=True)
    for rec_id, task in recordings:
        stem = f"sub-{pid}_ses-{ses}_task-{task}"
        (audio / f"{stem}_recording-metadata.json").write_text(
            json.dumps({"record_id": pid, "recording_id": rec_id.upper(), "session_id": ses}))
    pd.DataFrame({"record_id": [pid], "session_id": [ses], "session_index": ["1"]}).to_csv(
        root / f"sub-{pid}" / "sessions.tsv", sep="\t", index=False)
    return root / f"sub-{pid}"


class TestRemovedRecordings:
    """audio_recording_ids_to_remove.json and audio_filestems_to_remove.json remove a recording's
    audio, sidecar, features, quality-metric row and phenotype rows; audio_tasks_to_include selects tasks."""

    def test_exclusion_key_ignores_case_and_suffixes(self):
        key = BIDSDataset._exclusion_key
        assert key("sub-P1_ses-ABCD-1234_task-Rainbow-Passage.wav") == key(
            "sub-p1_ses-abcd-1234_task-rainbow-passage_recording-metadata.json")
        assert key("sub-p1_ses-s1_task-noisy-sounds-2_features.pt") == "sub-p1_ses-s1_task-noisy-sounds-2"

    def test_recording_ids_resolve_to_filestems_and_remove_audio_and_features(self, tmp_path):
        """Removal by recording_id uses the filestem mechanism, so features go too, even for a
        recording whose task is not included (its audio loop skips it before any ID check)."""
        pdir = _sidecar_tree(tmp_path / "in", recordings=(("rec-A", "free-speech-1"), ("rec-B", "noisy-sounds-2")))
        audio = pdir / "ses-S1" / "audio"
        for task in ("free-speech-1", "noisy-sounds-2"):
            torch.save({"opensmile": {"x": 1}}, audio / f"sub-p1_ses-S1_task-{task}_features.pt")
        stems = BIDSDataset._filestems_for_recording_ids(tmp_path / "in", ["p1"], {"rec-a"})
        assert stems == ["sub-p1_ses-S1_task-free-speech-1"]
        out = tmp_path / "out"
        BIDSDataset._deidentify_participant_files(
            pdir, out, {"p1": "900001"}, stems, ["noisy-sounds-*"], skip_audio=True, skip_audio_features=False)
        names = sorted(p.name for p in out.rglob("*") if p.is_file())
        assert not any("free-speech" in n for n in names), names
        assert any(n.endswith("noisy-sounds-2.json") for n in names)
        assert any("noisy-sounds-2_features" in n for n in names)

    def test_filestem_differing_in_case_still_removes(self, tmp_path):
        """A configured stem with a lower-case session ID removes the recording in an upper-case tree."""
        pdir = _sidecar_tree(tmp_path / "in", ses="ABC1")
        BIDSDataset._deidentify_participant_files(
            pdir, tmp_path / "out", {"p1": "900001"}, ["sub-p1_ses-abc1_task-Noisy-Sounds-1"], ["noisy-sounds-*"],
            skip_audio=True, skip_audio_features=True)
        names = [p.name for p in (tmp_path / "out").rglob("*.json")]
        assert len(names) == 1 and "noisy-sounds-2" in names[0]

    def test_tasks_are_included_by_pattern(self, tmp_path):
        pdir = _sidecar_tree(tmp_path / "in", recordings=(("rec-A", "identifying-pictures-35"),
                                                          ("rec-B", "reading-passage-3")))
        BIDSDataset._deidentify_participant_files(
            pdir, tmp_path / "out", {"p1": "900001"}, [], ["identifying-pictures-*"],
            skip_audio=True, skip_audio_features=True)
        written = [p.name for p in (tmp_path / "out").rglob("*.json")]
        assert len(written) == 1 and "identifying-pictures-35" in written[0]

    @pytest.mark.parametrize("stems, matched, participants, recording_ids, expected", [
        (["sub-001js_ses-x_task-passage-10"], set(), {"a-uuid-participant"}, set(),
         ["0 matched", "1 are for participants outside this run"]),
        (["sub-p1_ses-s1_task-noisy-sounds-1", "sub-p1_ses-s1_task-noisy-sounds-9"],
         {"sub-p1_ses-s1_task-noisy-sounds-1"}, {"p1"}, {"rid-x"},
         ["1 matched", "1 for participants in this run matched nothing", "None was found"]),
    ], ids=["participants-outside-the-run", "entries-matching-nothing"])
    def test_unmatched_removal_entries_warn(self, caplog, stems, matched, participants, recording_ids, expected):
        keys = {s: {BIDSDataset._exclusion_key(s)} for s in stems}
        with caplog.at_level(logging.INFO):
            BIDSDataset._report_exclusion_coverage(
                stems, keys, {BIDSDataset._exclusion_key(m) for m in matched}, participants, recording_ids, [])
        warnings = " ".join(r.getMessage() for r in caplog.records if r.levelno == logging.WARNING)
        assert all(text in warnings for text in expected), warnings

    def test_quality_metrics_exclusion_matches_the_recording_exactly(self, tmp_path):
        """identifying-pictures-2 must not also remove identifying-pictures-20 and -21."""
        (tmp_path / "bids").mkdir()
        pd.DataFrame({
            "participant_id": ["p1", "p1", "p1"], "session_id": ["s1"] * 3,
            "task_name": ["identifying-pictures-2", "identifying-pictures-20", "identifying-pictures-21"],
            "snr": ["1", "2", "3"],
        }).to_csv(tmp_path / "bids" / "audio_quality_metrics.tsv", sep="\t", index=False)
        (tmp_path / "out").mkdir()
        BIDSDataset._deidentify_quality_metrics(
            tmp_path / "bids", tmp_path / "out", [], ["sub-p1_ses-s1_task-identifying-pictures-2"],
            [], {"p1": "900001"}, {})
        out = pd.read_csv(tmp_path / "out" / "audio_quality_metrics.tsv", sep="\t", dtype=str)
        assert sorted(out.task_name) == ["identifying-pictures-20", "identifying-pictures-21"]

    @pytest.mark.parametrize("removal", [
        {"audio_recording_ids_to_remove.json": ["r2"]},
        {"audio_filestems_to_remove.json": ["sub-p1_ses-s1_task-maximum-phonation-time-1"]},
    ], ids=["by-recording-id", "by-filestem"])
    def test_removed_recordings_leave_recording_and_acoustic_task_tables(self, tmp_path, make_deid_tree, removal):
        """recording.tsv says 'R2': the recording ID is matched without regard to case."""
        bids, config = make_deid_tree(
            {"p1": [("s1", "rainbow-passage", {"recording_id": "r1"}),
                    ("s1", "maximum-phonation-time-1", {"recording_id": "r2"})]},
            config_files=removal,
            tables={"task/recording": pd.DataFrame({
                        "participant_id": ["p1", "p1"], "recording_id": ["r1", "R2"],
                        "recording_acoustic_task_id": ["t1", "t2"], "recording_name": ["a", "b"]}),
                    "task/acoustic_task": pd.DataFrame({
                        "participant_id": ["p1", "p1"], "acoustic_task_id": ["t1", "t2"],
                        "acoustic_task_name": ["Rainbow Passage", "Maximum phonation time-1"]})})
        out = tmp_path / "out"
        BIDSDataset(bids).deidentify(outdir=out, deidentify_config_dir=config)
        assert list(BIDSDataset._read_tsv_as_written(out / "phenotype/task/recording.tsv")["recording_id"]) == ["r1"]
        assert list(BIDSDataset._read_tsv_as_written(
            out / "phenotype/task/acoustic_task.tsv")["acoustic_task_id"]) == ["t1"]
        assert len(list(out.rglob("*.wav"))) == 1


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

    def test_rare_checkbox_label_needs_a_verdict_like_the_rest_of_the_cell(self, tmp_path, make_deid_tree):
        """The fold runs before the verdicts: a label folded into other_voice_activity (a review
        column) for a participant with no verdict is blanked with the cell."""
        bids, config = make_deid_tree(
            {"p1": [("s1", "rainbow-passage")], "p2": [("s1", "rainbow-passage")]},
            tables={"confounders/confounders": pd.DataFrame({
                "participant_id": ["p1", "p2"], "voice_activity_v2___attorney": ["1", ""],
                "voice_activity_v2___other": ["", "1"], "other_voice_activity": ["", "Podcaster"]})},
            settings={"small_checkbox_options": {"confounders": {"voice_activity_v2": {
                "other": "voice_activity_v2___other", "specify": "other_voice_activity", "min_participants": 10}}}},
            config_files={"column_value_reviews.json": {"verdicts": [
                {"participant_id": "p2", "column_name": "other_voice_activity", "verdict": "safe"}]}})
        BIDSDataset(bids).deidentify(outdir=tmp_path / "out", deidentify_config_dir=config)
        df = BIDSDataset._read_tsv_as_written(tmp_path / "out" / "phenotype" / "confounders" / "confounders.tsv")
        by_id = df.set_index("participant_id")
        assert "voice_activity_v2___attorney" not in df.columns
        assert by_id.loc["900000", "voice_activity_v2___other"] == "1"
        assert pd.isna(by_id.loc["900000", "other_voice_activity"])  # no verdict: no label
        assert by_id.loc["900001", "other_voice_activity"] == "Podcaster"
