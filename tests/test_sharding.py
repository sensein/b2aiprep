"""T036 — SLURM sharding unit tests for shard_audio_list."""

import pytest
from b2aiprep.prepare.qa_utils import shard_audio_list


def _paths(n):
    return [f"audio_{i:04d}.wav" for i in range(n)]


class TestShardAudioList:
    def test_two_shards_non_overlapping_complete(self):
        paths = _paths(10)
        shard1 = shard_audio_list(paths, part=1, num_parts=2)
        shard2 = shard_audio_list(paths, part=2, num_parts=2)
        assert set(shard1) | set(shard2) == set(paths)
        assert set(shard1) & set(shard2) == set()

    def test_single_shard_returns_all(self):
        paths = _paths(7)
        assert shard_audio_list(paths, part=1, num_parts=1) == paths

    def test_empty_list_returns_empty(self):
        assert shard_audio_list([], part=1, num_parts=3) == []

    def test_more_parts_than_items(self):
        paths = _paths(3)
        all_items = []
        for p in range(1, 6):
            shard = shard_audio_list(paths, part=p, num_parts=5)
            all_items.extend(shard)
        assert sorted(all_items) == sorted(paths)

    def test_parts_cover_full_list(self):
        paths = _paths(20)
        collected = []
        for p in range(1, 5):
            collected.extend(shard_audio_list(paths, part=p, num_parts=4))
        assert sorted(collected) == sorted(paths)

    def test_invalid_part_raises(self):
        with pytest.raises((ValueError, IndexError, AssertionError)):
            shard_audio_list(_paths(5), part=0, num_parts=2)

    def test_part_exceeds_num_parts_raises(self):
        with pytest.raises((ValueError, IndexError, AssertionError)):
            shard_audio_list(_paths(5), part=3, num_parts=2)
