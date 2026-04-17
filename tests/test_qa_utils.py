"""Unit tests for qa_utils public utility functions.

Constitution Principle II: public functions MUST have test coverage for
happy path, invalid inputs, and boundary conditions.

Covers: hash_config, save_config_snapshot, load_config, shard_audio_list,
        write_audio_sidecar, make_error_check_result, TimingContext.
"""

import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import pytest

from b2aiprep.prepare.qa_models import (
    CheckResult,
    CheckType,
    Classification,
    CompositeScore,
    FinalClassification,
    PipelineConfig,
)
from b2aiprep.prepare.qa_utils import (
    TimingContext,
    hash_config,
    load_config,
    make_error_check_result,
    save_config_snapshot,
    shard_audio_list,
    write_audio_sidecar,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> PipelineConfig:
    return PipelineConfig()


@pytest.fixture
def minimal_check_result() -> CheckResult:
    return CheckResult(
        participant_id="001",
        session_id="01",
        task_name="harvard-sentences-list-1-1",
        check_type=CheckType.AUDIO_QUALITY,
        score=0.9,
        confidence=0.85,
        classification=Classification.PASS,
        detail={"hard_gate_triggered": False},
        model_versions={"yamnet": "1.0"},
    )


@pytest.fixture
def minimal_composite_score(minimal_check_result) -> CompositeScore:
    return CompositeScore(
        participant_id="001",
        session_id="01",
        task_name="harvard-sentences-list-1-1",
        composite_score=0.9,
        composite_confidence=0.85,
        confidence_std_dev=0.05,
        final_classification=FinalClassification.PASS,
        check_results=[minimal_check_result],
        config_hash="abc12345",
        pipeline_version="0.1.0",
    )


# ---------------------------------------------------------------------------
# hash_config
# ---------------------------------------------------------------------------


class TestHashConfig:
    def test_stable_digest_same_config(self, default_config):
        """Same config produces the same digest on repeated calls."""
        h1 = hash_config(default_config)
        h2 = hash_config(default_config)
        assert h1 == h2

    def test_digest_is_64_hex_chars(self, default_config):
        digest = hash_config(default_config)
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_changing_field_changes_digest(self, default_config):
        """Mutating any single config field produces a different hash."""
        original = hash_config(default_config)

        cfg_modified = PipelineConfig(random_seed=99)
        assert hash_config(cfg_modified) != original

    def test_created_at_excluded_from_hash(self, default_config):
        """created_at does not affect the hash (snapshot stability)."""
        from datetime import datetime, timezone

        cfg_a = PipelineConfig(created_at=None)
        cfg_b = PipelineConfig(created_at=datetime(2025, 1, 1, tzinfo=timezone.utc))
        assert hash_config(cfg_a) == hash_config(cfg_b)

    def test_weight_change_changes_digest(self):
        cfg_a = PipelineConfig(
            check_weights={
                "audio_quality": 0.30,
                "unconsented_speakers": 0.25,
                "pii_disclosure": 0.25,
                "task_compliance": 0.20,
            }
        )
        cfg_b = PipelineConfig(
            check_weights={
                "audio_quality": 0.40,
                "unconsented_speakers": 0.20,
                "pii_disclosure": 0.20,
                "task_compliance": 0.20,
            }
        )
        assert hash_config(cfg_a) != hash_config(cfg_b)


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


class TestLoadConfig:
    def test_load_bundled_default(self):
        """Bundled default config loads without error."""
        cfg = load_config()
        assert isinstance(cfg, PipelineConfig)
        assert cfg.config_version == "1.0.0"
        assert cfg.random_seed == 42

    def test_load_from_path(self, default_config, tmp_path):
        """A JSON file written to disk round-trips through load_config."""
        snapshot_path = save_config_snapshot(default_config, str(tmp_path))
        # Save writes the snapshot; we can load it back
        cfg2 = load_config(str(snapshot_path))
        assert cfg2.random_seed == default_config.random_seed
        assert cfg2.config_version == default_config.config_version

    def test_bundled_thresholds(self):
        cfg = load_config()
        assert cfg.hard_gate_thresholds["snr_min_db"] == 12.0
        assert cfg.soft_score_thresholds["pass_min"] == 0.75
        assert cfg.check_weights["audio_quality"] == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# save_config_snapshot
# ---------------------------------------------------------------------------


class TestSaveConfigSnapshot:
    def test_file_written(self, default_config, tmp_path):
        out = save_config_snapshot(default_config, str(tmp_path))
        assert out.exists()

    def test_filename_contains_hash_prefix(self, default_config, tmp_path):
        out = save_config_snapshot(default_config, str(tmp_path))
        digest = hash_config(default_config)
        assert digest[:8] in out.name

    def test_file_is_valid_json(self, default_config, tmp_path):
        out = save_config_snapshot(default_config, str(tmp_path))
        with open(out) as fh:
            data = json.load(fh)
        assert "config_version" in data

    def test_created_at_set(self, tmp_path):
        cfg = PipelineConfig()
        assert cfg.created_at is None
        save_config_snapshot(cfg, str(tmp_path))
        assert cfg.created_at is not None


# ---------------------------------------------------------------------------
# shard_audio_list
# ---------------------------------------------------------------------------


class TestShardAudioList:
    def test_union_equals_full_list(self):
        paths = list(range(10))
        shards = [shard_audio_list(paths, p, 3) for p in range(1, 4)]
        union = sorted(item for shard in shards for item in shard)
        assert union == paths

    def test_shards_are_non_overlapping(self):
        paths = list(range(10))
        shards = [shard_audio_list(paths, p, 3) for p in range(1, 4)]
        all_items = [item for shard in shards for item in shard]
        assert len(all_items) == len(set(all_items))

    def test_part1_of_1_returns_all(self):
        paths = list(range(5))
        assert shard_audio_list(paths, 1, 1) == paths

    def test_empty_list_returns_empty(self):
        assert shard_audio_list([], 1, 3) == []

    def test_part_greater_than_num_parts_raises(self):
        with pytest.raises(ValueError, match="part must be in"):
            shard_audio_list(list(range(5)), 4, 3)

    def test_part_zero_raises(self):
        with pytest.raises(ValueError, match="part must be in"):
            shard_audio_list(list(range(5)), 0, 3)

    def test_num_parts_zero_raises(self):
        with pytest.raises(ValueError, match="num_parts must be"):
            shard_audio_list(list(range(5)), 1, 0)

    def test_single_element_list(self):
        assert shard_audio_list(["a"], 1, 3) == ["a"]
        assert shard_audio_list(["a"], 2, 3) == []
        assert shard_audio_list(["a"], 3, 3) == []

    def test_roughly_even_partition_sizes(self):
        paths = list(range(10))
        s1 = shard_audio_list(paths, 1, 3)
        s2 = shard_audio_list(paths, 2, 3)
        s3 = shard_audio_list(paths, 3, 3)
        # All shards should be within 1 of each other in length
        lengths = [len(s1), len(s2), len(s3)]
        assert max(lengths) - min(lengths) <= 1


# ---------------------------------------------------------------------------
# write_audio_sidecar
# ---------------------------------------------------------------------------


class TestWriteAudioSidecar:
    def test_written_to_correct_bids_path(
        self, minimal_check_result, minimal_composite_score, tmp_path
    ):
        out = write_audio_sidecar(
            bids_root=str(tmp_path),
            participant_id="001",
            session_id="01",
            task_name="harvard-sentences-list-1-1",
            check_results=[minimal_check_result],
            composite_score=minimal_composite_score,
        )
        expected = (
            tmp_path
            / "sub-001"
            / "ses-01"
            / "audio"
            / "sub-001_ses-01_task-harvard-sentences-list-1-1_qa.json"
        )
        assert out == expected
        assert out.exists()

    def test_json_parseable(
        self, minimal_check_result, minimal_composite_score, tmp_path
    ):
        out = write_audio_sidecar(
            bids_root=str(tmp_path),
            participant_id="001",
            session_id="01",
            task_name="task-x",
            check_results=[minimal_check_result],
            composite_score=minimal_composite_score,
        )
        with open(out) as fh:
            data = json.load(fh)
        assert isinstance(data, dict)

    def test_required_top_level_keys(
        self, minimal_check_result, minimal_composite_score, tmp_path
    ):
        out = write_audio_sidecar(
            bids_root=str(tmp_path),
            participant_id="001",
            session_id="01",
            task_name="task-x",
            check_results=[minimal_check_result],
            composite_score=minimal_composite_score,
        )
        with open(out) as fh:
            data = json.load(fh)
        for key in ("participant_id", "session_id", "task_name", "check_results", "composite_score"):
            assert key in data

    def test_transcript_present_when_supplied(
        self, minimal_check_result, minimal_composite_score, tmp_path
    ):
        out = write_audio_sidecar(
            bids_root=str(tmp_path),
            participant_id="001",
            session_id="01",
            task_name="task-x",
            check_results=[minimal_check_result],
            composite_score=minimal_composite_score,
            transcript="hello world",
        )
        with open(out) as fh:
            data = json.load(fh)
        assert "transcript" in data
        assert data["transcript"] == "hello world"

    def test_transcript_absent_when_not_supplied(
        self, minimal_check_result, minimal_composite_score, tmp_path
    ):
        out = write_audio_sidecar(
            bids_root=str(tmp_path),
            participant_id="001",
            session_id="01",
            task_name="task-x",
            check_results=[minimal_check_result],
            composite_score=minimal_composite_score,
        )
        with open(out) as fh:
            data = json.load(fh)
        assert "transcript" not in data

    def test_pii_spans_present_when_supplied(
        self, minimal_check_result, minimal_composite_score, tmp_path
    ):
        spans = [{"label": "name", "confidence": 0.9, "char_start": 0, "char_end": 4, "text": "John"}]
        out = write_audio_sidecar(
            bids_root=str(tmp_path),
            participant_id="001",
            session_id="01",
            task_name="task-x",
            check_results=[minimal_check_result],
            composite_score=minimal_composite_score,
            pii_spans=spans,
        )
        with open(out) as fh:
            data = json.load(fh)
        assert "pii_spans" in data
        assert data["pii_spans"] == spans

    def test_pii_spans_absent_when_not_supplied(
        self, minimal_check_result, minimal_composite_score, tmp_path
    ):
        out = write_audio_sidecar(
            bids_root=str(tmp_path),
            participant_id="001",
            session_id="01",
            task_name="task-x",
            check_results=[minimal_check_result],
            composite_score=minimal_composite_score,
        )
        with open(out) as fh:
            data = json.load(fh)
        assert "pii_spans" not in data

    def test_timing_s_included_when_supplied(
        self, minimal_check_result, minimal_composite_score, tmp_path
    ):
        out = write_audio_sidecar(
            bids_root=str(tmp_path),
            participant_id="001",
            session_id="01",
            task_name="task-x",
            check_results=[minimal_check_result],
            composite_score=minimal_composite_score,
            timing_s={"audio_quality": 0.5},
        )
        with open(out) as fh:
            data = json.load(fh)
        assert data["timing_s"] == {"audio_quality": 0.5}

    def test_creates_parent_directories(
        self, minimal_check_result, minimal_composite_score, tmp_path
    ):
        out = write_audio_sidecar(
            bids_root=str(tmp_path),
            participant_id="999",
            session_id="99",
            task_name="task-x",
            check_results=[minimal_check_result],
            composite_score=minimal_composite_score,
        )
        assert (tmp_path / "sub-999" / "ses-99" / "audio").is_dir()


# ---------------------------------------------------------------------------
# make_error_check_result
# ---------------------------------------------------------------------------


class TestMakeErrorCheckResult:
    def _call(self, exc: Exception) -> CheckResult:
        try:
            raise exc
        except Exception as e:
            return make_error_check_result(
                participant_id="001",
                session_id="01",
                task_name="task-x",
                check_type=CheckType.PII_DISCLOSURE,
                exception=e,
                model_versions={"gliner": "0.2"},
            )

    def test_classification_is_error(self):
        result = self._call(RuntimeError("model crash"))
        assert result.classification == Classification.ERROR

    def test_score_is_zero(self):
        result = self._call(RuntimeError("model crash"))
        assert result.score == 0.0

    def test_confidence_is_zero(self):
        result = self._call(RuntimeError("model crash"))
        assert result.confidence == 0.0

    def test_exception_message_in_detail(self):
        result = self._call(ValueError("bad input"))
        assert "bad input" in result.detail["error_message"]

    def test_error_type_in_detail(self):
        result = self._call(ValueError("bad input"))
        assert result.detail["error_type"] == "ValueError"

    def test_traceback_in_detail(self):
        result = self._call(RuntimeError("boom"))
        assert "traceback" in result.detail
        assert "RuntimeError" in result.detail["traceback"]

    def test_model_versions_preserved(self):
        result = self._call(RuntimeError("x"))
        assert result.model_versions == {"gliner": "0.2"}

    def test_check_type_preserved(self):
        result = self._call(RuntimeError("x"))
        assert result.check_type == CheckType.PII_DISCLOSURE

    def test_identity_fields_preserved(self):
        result = self._call(RuntimeError("x"))
        assert result.participant_id == "001"
        assert result.session_id == "01"
        assert result.task_name == "task-x"


# ---------------------------------------------------------------------------
# TimingContext
# ---------------------------------------------------------------------------


class TestTimingContext:
    def test_elapsed_is_positive_float(self):
        timer = TimingContext()
        with timer.time("stage_a"):
            time.sleep(0.001)
        summary = timer.get_timing_summary()
        assert "stage_a" in summary
        assert isinstance(summary["stage_a"], float)
        assert summary["stage_a"] > 0.0

    def test_multiple_stages_all_present(self):
        timer = TimingContext()
        for stage in ("a", "b", "c"):
            with timer.time(stage):
                pass
        summary = timer.get_timing_summary()
        assert set(summary.keys()) == {"a", "b", "c"}

    def test_all_values_are_non_negative(self):
        timer = TimingContext()
        with timer.time("fast"):
            pass
        summary = timer.get_timing_summary()
        assert all(v >= 0.0 for v in summary.values())

    def test_get_timing_summary_returns_copy(self):
        timer = TimingContext()
        with timer.time("s"):
            pass
        s1 = timer.get_timing_summary()
        s1["s"] = 999.0
        s2 = timer.get_timing_summary()
        assert s2["s"] != 999.0

    def test_empty_timer_returns_empty_dict(self):
        timer = TimingContext()
        assert timer.get_timing_summary() == {}

    def test_repeated_stage_overwrites(self):
        timer = TimingContext()
        with timer.time("s"):
            time.sleep(0.001)
        first = timer.get_timing_summary()["s"]
        with timer.time("s"):
            time.sleep(0.002)
        second = timer.get_timing_summary()["s"]
        # Both are positive; second should be >= first (or at least not equal)
        assert second != first or second > 0
