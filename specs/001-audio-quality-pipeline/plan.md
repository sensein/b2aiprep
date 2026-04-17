# Implementation Plan: Audio Quality Assurance Pipeline

**Branch**: `185-audio-quality-pipeline` | **Date**: 2026-04-09 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/001-audio-quality-pipeline/spec.md`

## Summary

Build a multi-check, confidence-scored audio quality assurance pipeline on top of the existing
b2aiprep processing stack. The pipeline combines four check domains — technical audio quality,
unconsented speaker detection, PII disclosure detection, and task compliance — into a
configurable composite score with per-task-type weighting. Audios that fall in uncertain
territory are routed to a CLI-based human review queue. A release report summarises batch
quality at a stated confidence level.

Most infrastructure already exists or is on an unmerged branch (`pii_detection`). The primary
new work is: (1) integrating existing checks into a unified scoring/classification layer,
(2) adding task-compliance verification (WER + phoneme checks), (3) porting PII detection
from the branch, (4) building the composite score, human review CLI, and release report,
(5) adding acoustic scene classification (YAMNet or equivalent) to Stage 1 technical quality,
(6) adding per-speaker language-ID to the unconsented-speaker check, (7) per-stage timing
metrics for bottleneck analysis, and (8) graceful model-failure handling (error classification
+ automatic routing to human review without halting the pipeline).

## Technical Context

**Language/Version**: Python 3.10–3.13
**Primary Dependencies**: senselab~=1.3.0 (audio QC, diarization, transcription, embeddings),
  click (CLI), torch/torchaudio, parselmouth (Praat phoneme/pitch), presidio-analyzer,
  nvidia/gliner-pii, microsoft/Phi-4-mini-instruct (optional, for LLM compliance checks),
  torchaudio (YAMNet acoustic scene classifier via torchaudio.pipelines),
  langdetect or equivalent (per-speaker language identification),
  sounddevice (optional, for audio playback in qa-review; gracefully degraded on HPC nodes),
  pytest (testing)
**Storage**: BIDS-structured TSV/JSON files in BIDS root; `.pt` feature files per audio
**Testing**: pytest with Click CliRunner; synthetic WAV fixtures via `create_dummy_wav_file()`
**Target Platform**: Linux server (HPC/cluster typical)
**Project Type**: library/cli
**Performance Goals**: Batch-parallelisable; avoid re-reading `.pt` features already computed
**Constraints**: All outputs MUST be deterministic (seed pinning, model version locking);
  config recorded in output for audit/reproducibility; existing CLI interface unchanged
**Scale/Scope**: Full Bridge2AI Voice dataset — > 10,000 recordings across 788 distinct task
  IDs; node-level distribution via SLURM array jobs with sharding CLI args; intra-node
  parallelism via multiprocessing

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Reproducibility | ✅ PASS | Model versions pinned in `PipelineConfig`; random seeds locked; config hash recorded in all outputs |
| II. Test Coverage | ✅ PASS | New modules (task_compliance, pii_detection, unconsented_speakers, qa_report) each get dedicated test files with synthetic fixtures |
| III. Backward Compatibility | ✅ PASS | All new CLI commands are additive; no existing commands removed or changed |
| IV. Data Integrity & Standards | ✅ PASS | Outputs written to BIDS root in TSV/JSON; schema versioned; config versioned |
| V. Performance-Aware | ✅ PASS | Batch processing + parallelism inherited from existing `quality_control_wrapper()`; `.pt` feature reuse avoids redundant computation; SLURM array job sharding via `--part`/`--num-parts` CLI args for > 10,000 audio batches |

No violations. Phase 0 cleared.

## Project Structure

### Documentation (this feature)

```text
specs/001-audio-quality-pipeline/
├── plan.md              ← this file
├── research.md          ← Phase 0 output
├── data-model.md        ← Phase 1 output
├── quickstart.md        ← Phase 1 output
├── contracts/
│   └── cli-commands.md  ← Phase 1 output
└── tasks.md             ← Phase 2 output (/speckit.tasks)
```

### Source Code

```text
src/b2aiprep/
├── prepare/
│   ├── qa_models.py               (new — QA data model dataclasses only: CheckResult,
│   │                               CompositeScore, ReviewDecision, QualityReport, PipelineConfig)
│   ├── qa_utils.py                (new — pipeline utilities: config loading/hashing, JSON
│   │                               sidecar writer, timing context, SLURM sharding, error handler)
│   ├── quality_control.py         (existing — extend with classification thresholds +
│   │                               acoustic scene classifier / YAMNet integration)
│   ├── task_compliance.py         (new — WER, phoneme, LLM compliance checks)
│   ├── pii_detection.py           (new — port + refactor from pii_detection branch)
│   ├── unconsented_speakers.py    (new — unified diarization + Evan's model + embeddings
│   │                               + per-speaker language-ID)
│   └── qa_report.py               (new — composite score, human review, release report,
│                                   timing metrics, error/model-failure handling)
├── commands.py                    (extend — 3 new CLI commands)
└── prepare/resources/
    ├── qa_pipeline_config.json    (new — default thresholds, weights, model versions)
    └── qa_pipeline_schema.json    (new — output schema documentation)

tests/
├── test_task_compliance.py        (new)
├── test_pii_detection.py          (new)
├── test_unconsented_speakers.py   (new)
└── test_qa_report.py              (new)
```

**Structure Decision**: Single project layout extending the existing `src/b2aiprep/prepare/`
module hierarchy. New modules follow the existing pattern (one logical concern per file,
wrapped by `commands.py` CLI functions).

## Complexity Tracking

No constitution violations requiring justification.
