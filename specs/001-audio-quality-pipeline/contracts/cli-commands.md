# CLI Contracts: Audio Quality Assurance Pipeline

**Date**: 2026-04-09
**Plan**: [plan.md](../plan.md)

All commands are added to `b2aiprep-cli` via `commands.py` + `cli.py` using the existing
Click pattern. Existing commands are unchanged.

---

## Command 1: `qa-run`

Run the full automated quality assurance pipeline over a BIDS dataset.

```
b2aiprep-cli qa-run BIDS_DIR OUTPUT_DIR [OPTIONS]
```

**Arguments**:
- `BIDS_DIR` — Path to the root of the BIDS dataset. Must contain per-audio `_features.pt`
  files from a prior `generate-audio-features` run. Technical quality metrics
  (`audio_quality_metrics.tsv`) are computed by `qa-run` itself as Stage 1 via the existing
  `quality_control_wrapper`; they do **not** need to exist in advance.
- `OUTPUT_DIR` — Directory where QA outputs are written.

**Options**:
```
--config PATH          Path to a PipelineConfig JSON file. If omitted, default config is used
                       and written to OUTPUT_DIR.
--part INT             1-based index of this shard in a SLURM array job. Must be used with
                       --num-parts. When provided, only the corresponding subset of audios is
                       processed (consistent with generate-audio-features sharding pattern).
--num-parts INT        Total number of shards. Used with --part to divide the audio list across
                       SLURM array tasks.
--batch-size INT       Batch size for parallelisable checks. [default: 8]
--num-cores INT        Number of parallel workers. [default: 4]
--skip-pii             Skip PII detection checks (e.g., for non-free-speech-only subsets).
--skip-task-compliance Skip task compliance checks.
--task-filter TEXT     Comma-separated list of task names to process (default: all).
--use-existing-qc      Skip Stage 1 technical quality control and use an already-present
                       audio_quality_metrics.tsv in BIDS_DIR. Fails if the file is absent.
--log-level TEXT       Logging level: DEBUG, INFO, WARNING, ERROR. [default: INFO]
```

**Outputs written to OUTPUT_DIR**:
- `audio_quality_metrics.tsv` — Technical QC metrics (written by Stage 1 unless
  `--use-existing-qc` is set, in which case it is read from BIDS_DIR)
- `qa_check_results.tsv` — CheckResult per audio × check type
- `qa_composite_scores.tsv` — CompositeScore per audio
- `needs_review_queue.tsv` — Audios classified as needs_review
- `qa_pipeline_config_{hash[:8]}.json` — Config snapshot

**Exit codes**:
- `0` — Pipeline completed (some audios may be flagged for review; this is not an error)
- `1` — Fatal error (input not found, dependency unavailable)

**Pre-condition**: `generate-audio-features` must have been run so `_features.pt` files exist.
`run-quality-control-on-audios` does **not** need to be run separately; `qa-run` invokes
`quality_control_wrapper` internally as its first stage.

---

## Command 2: `qa-review`

Interactive CLI for human review of flagged audios.

```
b2aiprep-cli qa-review OUTPUT_DIR [OPTIONS]
```

**Arguments**:
- `OUTPUT_DIR` — Directory containing `needs_review_queue.tsv` (from `qa-run`).

**Options**:
```
--reviewer-id TEXT     Reviewer identifier stored in decisions file. [required]
--audio-root PATH      Root path of the BIDS dataset (for audio playback). If omitted,
                       playback is skipped and only scores are displayed.
--limit INT            Maximum number of audios to review in this session (default: all).
--reopen               Include audios already reviewed (allows changing decisions).
```

**Interactive session per audio**:
```
[001/047] sub-003 | ses-02 | harvard-sentences-list-4-7
  Composite score:  0.58  (needs_review)
  Audio quality:    0.91  (pass)
  Unconsented spk:  0.72  (needs_review)  — num_speakers=2, primary_ratio=0.83
  PII disclosure:   0.88  (pass)
  Task compliance:  0.61  (needs_review)  — WER=0.28

  [p]lay audio  [a]ccept  [r]eject  [s]kip  [n]ote  [q]uit
```

**Outputs written to OUTPUT_DIR** (appended or created):
- `human_review_decisions.tsv` — ReviewDecision per reviewed audio

**Exit codes**:
- `0` — Session completed or quit by user
- `1` — `needs_review_queue.tsv` not found

---

## Command 3: `qa-report`

Generate the release quality report over a completed batch.

```
b2aiprep-cli qa-report OUTPUT_DIR [OPTIONS]
```

**Arguments**:
- `OUTPUT_DIR` — Directory containing `qa_composite_scores.tsv` and (optionally)
  `human_review_decisions.tsv`.

**Options**:
```
--confidence-target FLOAT  Target confidence level for the claim statement (0–1).
                            [default: 0.95]
--output-format TEXT        `markdown`, `json`, or `both`. [default: both]
--fail-on-below-target      Exit with code 2 if achieved confidence < target.
```

**Outputs written to OUTPUT_DIR**:
- `qa_release_report.md` (if format=markdown or both)
- `qa_release_report.json` (if format=json or both)

**Sample report excerpt**:
```markdown
# Release Quality Report — 2026-04-09

**Pipeline config**: qa_pipeline_config_a3f7c891.json
**Audios processed**: 4,812

| Check | Pass Rate |
|-------|-----------|
| Audio quality | 94.2% |
| Unconsented speakers | 98.7% |
| PII disclosure | 99.1% |
| Task compliance | 87.3% |

**Overall**: At **93.4% confidence**, **91.8%** of released audios pass all quality checks.
```

**Exit codes**:
- `0` — Report generated
- `1` — Required input files not found
- `2` — Achieved confidence below `--confidence-target` (only if `--fail-on-below-target` set)
