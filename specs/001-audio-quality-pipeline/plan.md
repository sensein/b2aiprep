# Implementation Plan: Audio Quality Assurance Pipeline

**Branch**: `185-audio-quality-pipeline` | **Date**: 2026-04-17 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/001-audio-quality-pipeline/spec.md`

**Scope**: This plan covers **User Story 1 only** (Automated Batch Quality Screening).
User Story 2 (Human Review) and User Story 3 (Release Quality Report) are deferred —
see `## Deferred Stories` in spec.md and Phases 4–5 in tasks.md.

## Summary

Implement the automated batch audio quality screening pipeline for the Bridge2AI Voice
dataset release. The pipeline evaluates each audio recording across four quality checks
(technical quality, unconsented speakers, PII disclosure, task compliance), combines
per-check scores into a weighted composite score, classifies each audio as pass / fail /
needs-review, and writes per-audio JSON sidecars and batch TSV outputs. Runs via the
existing `b2aiprep-cli` interface with SLURM sharding for HPC deployment.

## Technical Context

**Language/Version**: Python 3.10+
**Primary Dependencies**: click, torch, torchaudio, transformers (HuggingFace),
  pyannote.audio, openai-whisper, soundfile, numpy, pandas, parselmouth, senselab
**Storage**: Files — per-audio JSON sidecars (BIDS co-located) + batch TSV outputs
**Testing**: pytest with Click CliRunner for CLI integration tests
**Target Platform**: Linux (HPC SLURM cluster, Rocky Linux 8)
**Project Type**: CLI tool extending existing `b2aiprep-cli` command set
**Performance Goals**: >10,000 audios per batch via SLURM array job sharding;
  no per-audio wall-clock target in v1 (timing metrics collected for future budgeting)
**Constraints**: Fully deterministic outputs — random seed pinned in PipelineConfig,
  model versions locked; per-stage timing recorded in JSON sidecar (FR-013)
**Scale/Scope**: Full Bridge2AI Voice dataset batch (estimated >10k audio files)

## Constitution Check

*Constitution file is unpopulated (template placeholder) — no project-specific gates to
evaluate. Standard quality practices apply: tests written before implementation, public
functions have test coverage, no hardcoded thresholds (all in PipelineConfig).*

**Pre-design gates**: PASS (no violations)
**Post-design gates**: PASS (no violations)

## Project Structure

### Documentation (this feature)

```text
specs/001-audio-quality-pipeline/
├── plan.md              ← this file
├── research.md          ← Phase 0 output (complete)
├── data-model.md        ← Phase 1 output (complete)
├── quickstart.md        ← Phase 1 output (complete)
├── contracts/
│   └── cli-commands.md  ← Phase 1 output (complete)
└── tasks.md             ← task breakdown (complete; Phase 4/5 deferred)
```

### Source Code

```text
src/b2aiprep/
├── prepare/
│   ├── quality_control.py          # extended (T014, T015) [x done]
│   ├── unconsented_speakers.py     # new (T016) [x done]
│   ├── pii_detection.py            # new (T017) [x done]
│   ├── task_compliance.py          # new (T018, T019, T020) [x done]
│   ├── qa_report.py                # new — composite score only for US1 (T021) [x done]
│   ├── qa_models.py                # new — dataclasses (T003) [x done]
│   ├── qa_utils.py                 # new — utilities (T004–T008) [x done]
│   └── resources/
│       ├── qa_pipeline_config.json # new (T001) [x done]
│       └── qa_pipeline_schema.json # new (T002) [x done]
├── commands.py                     # extended — qa-run command (T022) [x done]
└── cli.py                          # extended — qa-run registered (T023) [x done]

tests/
├── test_qa_utils.py                # new (T008b) [x done]
├── test_quality_control.py         # extended (T009) [x done]
├── test_unconsented_speakers.py    # new (T010) [x done]
├── test_pii_detection.py           # new (T011) [x done]
├── test_task_compliance.py         # new (T012) [x done]
└── test_qa_run.py                  # new (T013) [x done]
```

## Deferred Modules (US2 / US3)

These will be added in a future iteration once US2 and US3 are fully specified:

```text
src/b2aiprep/
├── prepare/
│   └── qa_report.py               # will be extended with compute_quality_report,
│                                  # write_quality_report_*, format_review_card,
│                                  # record_decision, load_decided_keys
├── commands.py                    # will be extended with qa-review, qa-report
└── cli.py                         # will register qa-review, qa-report
```

## Complexity Tracking

No constitution violations requiring justification.
