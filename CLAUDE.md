# b2aiprep — Claude Code context

## Active feature branch: 185-audio-quality-pipeline

This branch adds the Audio Quality Assurance Pipeline. Key modules added/extended:

### New QA modules (`src/b2aiprep/prepare/`)
| File | Purpose |
|------|---------|
| `qa_models.py` | Dataclasses: `AudioRecord`, `CheckResult`, `CompositeScore`, `ReviewDecision`, `QualityReport`, `PipelineConfig` |
| `qa_utils.py` | Utilities: `load_config`, `hash_config`, `save_config_snapshot`, `write_audio_sidecar`, `TimingContext`, `shard_audio_list`, `make_error_check_result` |
| `quality_control.py` | Extended with `check_audio_quality()` — SNR/clipping/silence hard gates + AST environment classifier |
| `unconsented_speakers.py` | `check_unconsented_speakers()` — diarization + Evans model + language ID |
| `pii_detection.py` | `check_pii_disclosure()` — GLiNER + Presidio fallback, transcript confidence proxy |
| `task_compliance.py` | `check_task_compliance()` dispatcher → Tier A (WER), Tier B (signal), Tier C (Phi-4 LLM) |
| `qa_report.py` | `compute_composite_score()` — weighted combination, hard/soft gates, final classification |

### New CLI command
`b2aiprep-cli qa-run BIDS_DIR OUTPUT_DIR [OPTIONS]`

Key options: `--pipeline-config`, `--part`, `--num-parts`, `--skip-pii`, `--skip-task-compliance`, `--task-filter`, `--use-existing-qc`

### Config resource
`src/b2aiprep/prepare/resources/qa_pipeline_config.json` — default thresholds/weights/model versions. Pass via `--pipeline-config` to override.

### Output files (written to OUTPUT_DIR)
- `qa_check_results.tsv` — per-check scores and classifications for every audio
- `qa_composite_scores.tsv` — composite classification per audio (pass/fail/needs_review)
- `needs_review_queue.tsv` — audios routed to human review
- `qa_pipeline_config_<hash8>.json` — exact config snapshot

Per-audio JSON sidecars (with transcript + PII spans) are written co-located with source audio in BIDS_DIR.

### Tests
`tests/test_qa_utils.py`, `tests/test_quality_control.py`, `tests/test_unconsented_speakers.py`, `tests/test_pii_detection.py`, `tests/test_task_compliance.py`, `tests/test_qa_run.py`, `tests/test_sharding.py`, `tests/test_ground_truth.py`

### Deferred (US2/US3)
`qa-review` and `qa-report` CLI commands are deferred pending further specification. Tasks T024–T033 in `specs/001-audio-quality-pipeline/tasks.md`.

## General notes

- **No CHANGELOG.md edits** — changelog is auto-generated via CI
- **No class definitions in `commands.py`** — all imports at top level
- **Config flag is `--pipeline-config`** (not `--config`) — avoids collision with `yapecs`
- SLURM sharding: `--part` is 1-based, consistent with `generate-audio-features` pattern

## Active Technologies
- Python 3.10+ + torch, speechbrain, pyannote (already in env); no new model downloads (186-speaker-profile-detection)
- Files (`speaker_profile.json` per participant, written to `PROFILES_DIR/sub-*/`) (186-speaker-profile-detection)
- Python 3.10+ + torch, speechbrain, pyannote, senselab (already in env); (186-speaker-profile-detection)
- JSON files per participant (`PROFILES_DIR/sub-*/speaker_profile.json`) (186-speaker-profile-detection)

## Recent Changes
- 186-speaker-profile-detection: Added Python 3.10+ + torch, speechbrain, pyannote (already in env); no new model downloads
