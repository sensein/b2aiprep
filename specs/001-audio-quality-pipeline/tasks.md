# Tasks: Audio Quality Assurance Pipeline

**Input**: Design documents from `specs/001-audio-quality-pipeline/`
**Prerequisites**: plan.md ✅ spec.md ✅ research.md ✅ data-model.md ✅ contracts/cli-commands.md ✅ quickstart.md ✅

**Organization**: Tasks are grouped by user story to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files or independent functions, no unresolved dependencies)
- **[Story]**: Which user story this task belongs to
- All paths are relative to the repository root

---

## Phase 1: Setup (Shared Resources)

**Purpose**: Create configuration and schema resource files that the pipeline bundles and ships with.

- [x] T001 Create default PipelineConfig JSON resource at `src/b2aiprep/prepare/resources/qa_pipeline_config.json` with all required fields: `hard_gate_thresholds` (snr_min=12 dB, clipping_max=0.05, silence_max=0.50), `soft_score_thresholds` (pass_min=0.75, fail_max=0.40, check_min=0.50), `check_weights` (audio_quality=0.30, unconsented_speakers=0.25, pii_disclosure=0.25, task_compliance=0.20), `task_compliance_params` (diadochokinesis.ddk_rate_expected_hz=[5.0,7.0], ddk_rate_flag_outside_hz=[2.0,9.0]; phonation.min_duration_s=3.0; conversational.min_active_speech_duration_s=3.0), `environment_noise_threshold=0.60`, `environment_noise_classes=["Speech","Crowd","Music","Vehicle"]`, `confidence_disagreement_penalty=0.50`, `min_transcript_confidence=0.70`, `random_seed=42`, `human_review_timeout_days=30`, `model_versions={"evans_model": "TODO: HuggingFace model path to be added when model is published"}`, `config_version="1.0.0"`
- [x] T002 [P] Create output schema JSON resource at `src/b2aiprep/prepare/resources/qa_pipeline_schema.json` documenting all output fields for CheckResult, CompositeScore, ReviewDecision, and QualityReport entities as defined in `specs/001-audio-quality-pipeline/data-model.md`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared data model, config infrastructure, and cross-cutting utilities that every user story depends on.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [x] T003 Implement QA data model dataclasses (dataclasses only — no I/O logic) in `src/b2aiprep/prepare/qa_models.py`: `CheckResult` (participant_id, session_id, task_name, check_type, score, confidence, classification enum [pass/fail/needs_review/error], detail dict, model_versions dict), `CompositeScore` (participant_id, session_id, task_name, composite_score, composite_confidence, confidence_std_dev, final_classification, check_results, config_hash, pipeline_version), `ReviewDecision` (participant_id, session_id, task_name, decision, reviewer_id, reviewed_at, notes), `QualityReport` (all fields from data-model.md), `PipelineConfig` (all fields from data-model.md including task_compliance_params, environment thresholds, confidence_disagreement_penalty)
- [x] T004 [P] Implement PipelineConfig loading and SHA-256 hashing in `src/b2aiprep/prepare/qa_utils.py`: `load_config(path=None) -> PipelineConfig` loads from path or bundled default; `hash_config(config: PipelineConfig) -> str` returns SHA-256 hex digest; `save_config_snapshot(config, output_dir)` writes `qa_pipeline_config_{hash[:8]}.json`
- [x] T005 [P] Implement per-audio JSON sidecar writer in `src/b2aiprep/prepare/qa_utils.py`: `write_audio_sidecar(bids_root, participant_id, session_id, task_name, check_results, composite_score, transcript=None, pii_spans=None)` writes `sub-{id}/ses-{id}/voice/sub-{id}_ses-{id}_task-{name}_qa.json` containing full transcript and PII spans with text (sensitive; release-gated separately from TSV outputs)
- [x] T006 [P] Implement per-stage wall-clock timing context manager in `src/b2aiprep/prepare/qa_utils.py`: `TimingContext` records start/end time per stage label; `get_timing_summary() -> dict` returns `{stage_name: duration_s}` for inclusion in per-audio JSON sidecar (FR-013)
- [x] T007 [P] Implement SLURM sharding utility in `src/b2aiprep/prepare/qa_utils.py`: `shard_audio_list(audio_paths: list, part: int, num_parts: int) -> list` returns the non-overlapping subset for this SLURM array task index (1-based part, consistent with existing generate-audio-features pattern)
- [x] T008 Implement model-failure error handler in `src/b2aiprep/prepare/qa_utils.py`: `make_error_check_result(participant_id, session_id, task_name, check_type, exception, model_versions) -> CheckResult` returns a CheckResult with classification=`error`, score=0.0, confidence=0.0, and logs the full exception with traceback (FR-014); pipeline continues to next audio
- [x] T008b [P] Write unit tests for all public utility functions in `tests/test_qa_utils.py` (constitution Principle II — public functions MUST have test coverage for happy path, invalid inputs, and boundary conditions): `hash_config` — assert known PipelineConfig produces a stable SHA-256 digest and that changing any single field changes the digest; `shard_audio_list` — assert correct partition sizes, non-overlapping subsets whose union equals the full list, edge cases: part=1 of num_parts=1 returns all, empty list returns empty, part > num_parts raises; `write_audio_sidecar` — assert file written to correct BIDS path (`bids_root/sub-{id}/ses-{id}/voice/`), JSON parseable, contains expected top-level keys, transcript and pii_spans fields present when supplied and absent when not; `make_error_check_result` — assert returned CheckResult has classification=`error`, score=0.0, confidence=0.0, and exception message captured in detail dict; `TimingContext` — assert elapsed time is a positive float for a timed block, `get_timing_summary()` returns all registered stage labels with float values

**Checkpoint**: Foundation complete — all user story phases may now begin.

---

## Phase 3: User Story 1 — Automated Batch Quality Screening (Priority: P1) 🎯 MVP

**Goal**: Pipeline evaluates every audio across all four quality checks, produces per-check confidence scores and classifications, emits per-audio JSON sidecars and batch TSVs, routes borderline/error audios to a human review queue.

**Independent Test**: Run pipeline on a synthetic batch of 5 WAV files with known ground truth (one clean, one clipped, one multi-speaker, one PII-containing, one off-task). Assert correct classification in `qa_composite_scores.tsv` and that `needs_review_queue.tsv` contains only the borderline cases.

### Tests for User Story 1

- [x] T009 [P] [US1] Write synthetic WAV fixture factory (`create_dummy_wav_file` extension) producing clean, clipped, silent, and noisy WAVs; extend existing `tests/test_quality_control.py` with: audio technical quality check unit tests asserting hard gate triggering and soft threshold assignment; and FR-011 invalid-input coverage — assert that a corrupt/unreadable file (zero-byte WAV, truncated WAV, non-WAV binary) is skipped and logged without halting the pipeline, and is absent from all output TSVs (constitution Principle II — invalid input path MUST be tested)
- [x] T010 [P] [US1] Write unconsented-speaker detection unit tests: single-speaker WAV passes, two-speaker WAV flags with `extra_speaker_count=1`, assert `detected_languages` list populated, assert Evan's model flag forces `needs_review` in `tests/test_unconsented_speakers.py`
- [x] T011 [P] [US1] Write PII detection unit tests: known PII transcript flagged with correct entity labels and char offsets; low-transcript-confidence recording forces `needs_review`; full transcript stored in sidecar but not in TSV in `tests/test_pii_detection.py`
- [x] T012 [P] [US1] Write task compliance unit tests: Tier A WER=0.0 passes, WER=0.5 needs_review; Tier B phonation duration below minimum fails; Tier C LLM=false fails; assert compliance_tier assigned correctly per task category in `tests/test_task_compliance.py`
- [x] T013 [US1] Write end-to-end integration test for `qa-run` using Click CliRunner on synthetic batch: assert `qa_check_results.tsv`, `qa_composite_scores.tsv`, `needs_review_queue.tsv`, and `qa_pipeline_config_*.json` all written; assert determinism by running twice and comparing outputs in `tests/test_qa_run.py`

### Implementation for User Story 1

- [x] T014 [US1] Extend `quality_control_wrapper` in `src/b2aiprep/prepare/quality_control.py` to translate existing raw metric values (SNR, clipping, silence, AM depth) into CheckResult scores and classifications using configurable hard-gate and soft-score thresholds from PipelineConfig; add `hard_gate_triggered` field to output
- [x] T015 [US1] Integrate YAMNet (via `torchaudio.pipelines`) acoustic scene classifier into `quality_control_wrapper` in `src/b2aiprep/prepare/quality_control.py`; add `environment_top_labels` (top-3 label+confidence) and `environment_noise_flag` (true if top-1 label in configured noise classes with confidence ≥ threshold) to audio_quality detail block; append per-audio timing entry via TimingContext; pin model version in PipelineConfig `model_versions` (YAMNet is a feedforward classifier with no stochastic sampling — version pinning is sufficient for reproducibility)
- [x] T016 [US1] Implement `src/b2aiprep/prepare/unconsented_speakers.py`: `check_unconsented_speakers(audio_record, config) -> CheckResult` — read diarization from `.pt` file, compute `primary_speaker_ratio` and `extra_speaker_count`, load Evan's model via `transformers.AutoModel.from_pretrained(config.model_versions["evans_model"])` (binary flag; TODO placeholder path in default config until model is published on HuggingFace), compute cosine similarity of speaker embeddings across diarized segments, run language-ID per segment to populate `detected_languages`; forced `needs_review` if `evans_model_flag==1`; append timing entry; language-ID model is a classifier — seed any internal randomness (e.g., `langdetect.DetectorFactory.seed = config.random_seed`) and pin model version in PipelineConfig `model_versions`
- [x] T017 [US1] Implement `src/b2aiprep/prepare/pii_detection.py`: port `pii_detection_gliner()` and Presidio fallback from `pii_detection` branch; add `transcript_confidence_proxy()` (Whisper log-prob average or tiny-vs-large-turbo WER divergence); `check_pii_disclosure(audio_record, config) -> CheckResult` returns CheckResult with entity labels + char offsets in TSV detail and full transcript + PII spans (label, confidence, character offsets, span text) written to sidecar via `write_audio_sidecar`; forced `needs_review` if transcript confidence below threshold; append timing entry; note: `task_correctness_phi4()` (LLM compliance helper) is ported separately in T020 into `task_compliance.py`, not here
- [x] T018 [US1] Implement Tier A task compliance in `src/b2aiprep/prepare/task_compliance.py`: `check_tier_a(audio_record, config) -> CheckResult` computes character-normalised edit distance between Whisper transcript (from `.pt` file) and known prompt text from `audio_task_descriptions.json`; compliance_confidence = 1 − normalised_edit_distance (clamped 0–1); WER < 0.10 → high, 0.10–0.30 → moderate, > 0.30 → low; covers harvard-sentences, cape-V-sentences, passage, rainbow, caterpillar-passage, repeat-words, sentence task categories; implement `get_compliance_tier(task_category: str) -> str` dispatcher that raises `ValueError` for unrecognised task_category values — all 788 task IDs must map to a known tier; include a test that enumerates all known task categories from `audio_task_descriptions.json` and asserts each resolves without error
- [x] T019 [US1] Implement Tier B task compliance in `src/b2aiprep/prepare/task_compliance.py`: sustained phonation (duration via silence-trimming ≥ `phonation.min_duration_s` from config + F1/F2 Gaussian likelihood via parselmouth); DDK periodicity (frame-energy autocorrelation rate within configurable `ddk_rate_flag_outside_hz` bounds) + phoneme accuracy (read pre-computed PPGs from `_features.pt` file and compare the dominant phoneme class per burst to the target phoneme(s) for that DDK variant — PPGs are already computed by `generate-audio-features` and stored in `.pt` files, avoiding a new G2P dependency); pitch glide (F0 slope sign consistency ≥ 70%); breathing/cough (< 10% voiced frames); `check_tier_b(audio_record, config) -> CheckResult` dispatches by task_category; append timing entry
- [x] T020 [US1] Implement Tier C task compliance in `src/b2aiprep/prepare/task_compliance.py`: port `task_correctness_phi4()` from the `pii_detection` branch directly into `task_compliance.py` (not pii_detection.py — this function is a task-compliance helper, not a PII detection function); `check_tier_c(audio_record, config) -> CheckResult` calls `task_correctness_phi4()` for LLM boolean + duration gate (active_speech_duration ≥ `conversational.min_active_speech_duration_s`); confidence=0.9 if LLM=true AND duration_met, 0.5 if LLM=true but duration short or transcript confidence low, 0.3 if LLM=false; covers naming, story, picture, conversational, cognitive, recitation, loudness categories; append timing entry; use `temperature=0, do_sample=False` for greedy decode — appropriate for this binary classification task (output space is constrained; quality impact is minimal; borderline cases are handled by the human review path rather than relying on LLM sampling); if post-v1 empirical evaluation shows systematic edge-case misses, upgrade path is fixed-seed constrained decoding, not free sampling
- [x] T021 [US1] Implement `compute_composite_score(check_results: list[CheckResult], config: PipelineConfig) -> CompositeScore` in `src/b2aiprep/prepare/qa_report.py`: weighted mean of per-check scores using per-task-category weights; composite_confidence = weighted_mean(confidences) × (1 − λ × std_dev(confidences)) where λ = config.confidence_disagreement_penalty; Stage 1 hard gates → FAIL; Stage 2 forced review gates (evans_model_flag==1, transcript_confidence below threshold, any check==error) → NEEDS_REVIEW; Stage 3 soft classification (composite ≥ 0.75 AND all checks ≥ 0.50 → PASS; composite < 0.40 → FAIL; else NEEDS_REVIEW)
- [x] T022 [US1] Implement `qa-run` CLI command in `src/b2aiprep/commands.py`: orchestrate Stage 1 (quality_control_wrapper, skipped if `--use-existing-qc`), Stage 2 (unconsented_speakers), Stage 3 (pii_detection), Stage 4 (task_compliance), Stage 5 (compute_composite_score) per audio; apply SLURM sharding via `shard_audio_list` when `--part`/`--num-parts` provided; two distinct error paths must be implemented — (a) FR-011: if audio file is unreadable/corrupt before any check runs, log the error, skip the audio entirely, exclude from all output TSVs and sidecars, do not route to human review; (b) FR-014: if a check model fails during processing, call `make_error_check_result` and route that audio to `needs_review_queue.tsv` without halting the pipeline; write `qa_check_results.tsv`, `qa_composite_scores.tsv`, `needs_review_queue.tsv`, and `qa_pipeline_config_{hash[:8]}.json` to OUTPUT_DIR; write per-audio JSON sidecars to BIDS_DIR (co-located with source audio via `write_audio_sidecar(bids_root=BIDS_DIR, ...)` from `qa_utils.py`) not to OUTPUT_DIR; collect `TimingContext` results from each check and pass to `write_audio_sidecar`; pin random seed from config
- [x] T023 [US1] Register `qa-run` command in `src/b2aiprep/cli.py` following existing `cli.add_command` pattern; verify `b2aiprep-cli qa-run --help` shows all options including `--part`, `--num-parts`, `--use-existing-qc`, `--skip-pii`, `--skip-task-compliance`, `--task-filter`, `--config`, `--batch-size`, `--num-cores`, `--log-level`

**Checkpoint**: User Story 1 fully functional — pipeline runs end-to-end on synthetic batch, all output files present, classifications correct.

---

## Phase 4: User Story 2 — Human Review of Flagged Audios (Priority: P2)

**Goal**: Reviewer works through `needs_review_queue.tsv` audio by audio, sees per-check confidence breakdown, optionally plays audio, records accept/reject, with session resumability.

**Independent Test**: Pre-populate `needs_review_queue.tsv` with 3 known entries; run `qa-review` via CliRunner submitting [a], [r], [s] decisions in sequence; assert `human_review_decisions.tsv` contains exactly the two non-skipped decisions with correct reviewer_id and timestamp; assert skipped item remains in queue on re-run.

### Tests for User Story 2

- [ ] T024 [P] [US2] Write `qa-review` CLI session tests using Click CliRunner with pre-populated `needs_review_queue.tsv`; assert accept/reject decisions written to `human_review_decisions.tsv`; assert skip leaves item in queue; assert `--reopen` exposes already-decided items; assert session exits cleanly on [q] in `tests/test_qa_review.py`

### Implementation for User Story 2

- [ ] T025 [US2] Implement review session display in `src/b2aiprep/prepare/qa_report.py`: `format_review_card(composite_score: CompositeScore) -> str` renders participant_id, task_name, composite score, and per-check score/classification/key-detail-fields in the tabular format shown in `specs/001-audio-quality-pipeline/contracts/cli-commands.md`; attempt `sounddevice.play` for audio playback when `--audio-root` provided; catch `sounddevice` import/playback errors and degrade gracefully (print path only) for HPC nodes without audio output
- [ ] T026 [US2] Implement ReviewDecision persistence in `src/b2aiprep/prepare/qa_report.py`: `record_decision(output_dir, decision: ReviewDecision)` appends one TSV row to `human_review_decisions.tsv` (creating with header if absent); `load_decided_keys(output_dir) -> set` returns set of (participant_id, session_id, task_name) tuples already decided, used to skip already-reviewed items unless `--reopen` is set
- [ ] T027 [US2] Implement `qa-review` CLI command in `src/b2aiprep/commands.py`: read `needs_review_queue.tsv`, filter already-decided unless `--reopen`, iterate presenting review cards, handle [a]ccept/[r]eject/[s]kip/[n]ote/[q]uit keystrokes via `click.prompt`, call `record_decision` after each non-skip response; respect `--limit` for session length; print session summary (N accepted, N rejected, N skipped, N remaining) on exit
- [ ] T028 [US2] Register `qa-review` command in `src/b2aiprep/cli.py`; verify `b2aiprep-cli qa-review --help` shows `--reviewer-id` (required), `--audio-root`, `--limit`, `--reopen`

**Checkpoint**: User Stories 1 and 2 functional — full automated + human review flow works end-to-end.

---

## Phase 5: User Story 3 — Release Quality Report (Priority: P3)

**Goal**: Single command reads all automated and human-review results and produces a Markdown and JSON release report with per-check pass rates, composite score distribution, human override counts, and a defensible top-level confidence claim.

**Independent Test**: Pre-populate `qa_composite_scores.tsv` (mix of pass/fail/needs_review) and `human_review_decisions.tsv` (some accept, some reject); run `qa-report`; assert report contains correct per-check pass rates, correct `released_count`, and a `claim_statement` string; assert `--fail-on-below-target` exits with code 2 when achieved confidence < target.

### Tests for User Story 3

- [ ] T029a [P] [US3] Write unit tests for `compute_quality_report` and report serialisation functions in `tests/test_quality_report.py`: use synthetic `qa_composite_scores.tsv` and `human_review_decisions.tsv` data; assert correct `released_count`, `excluded_count`, `per_check_pass_rates`, `claim_confidence`, and `claim_statement` values; assert `write_quality_report_json` and `write_quality_report_markdown` produce correctly formatted output
- [ ] T029b [P] [US3] Write CLI integration tests for `qa-report` command using Click CliRunner in `tests/test_qa_report_cli.py`: assert Markdown and JSON output files are written to OUTPUT_DIR; assert exit code 2 when `--fail-on-below-target` set and achieved confidence below target; assert exit code 1 when required input files are missing

### Implementation for User Story 3

- [ ] T030 [US3] Implement `compute_quality_report(output_dir, confidence_target) -> QualityReport` in `src/b2aiprep/prepare/qa_report.py`: read `qa_composite_scores.tsv` and `human_review_decisions.tsv`; compute all QualityReport aggregate fields (auto_pass, auto_fail, human_accepted, human_rejected, pending_review, released_count, excluded_count, per_check_pass_rates, composite_score_percentiles [p10/p25/p50/p75/p90]); compute `claim_confidence = mean(composite_confidence) × (1 − human_override_rate × 0.5)` and `claim_statement` string; compute `released_pass_rate = (PASS + human_accepted) / total`; emit a WARNING-level log message if `needs_review_total / total_audios` exceeds the configured `sc_004_review_fraction_warn` threshold (from PipelineConfig, default 0.15) so operators are alerted when reviewer burden is unexpectedly high (SC-004)
- [ ] T031 [US3] Implement report serialisation in `src/b2aiprep/prepare/qa_report.py`: `write_quality_report_json(report, output_dir)` writes `qa_release_report.json`; `write_quality_report_markdown(report, output_dir)` writes `qa_release_report.md` in the format shown in `specs/001-audio-quality-pipeline/contracts/cli-commands.md` (per-check pass rate table, overall confidence claim, counts breakdown)
- [ ] T032 [US3] Implement `qa-report` CLI command in `src/b2aiprep/commands.py`: call `compute_quality_report`, dispatch to `write_quality_report_json` and/or `write_quality_report_markdown` per `--output-format` (markdown/json/both); exit code 2 if `--fail-on-below-target` and `claim_confidence < confidence_target`; exit code 1 if required input files missing
- [ ] T033 [US3] Register `qa-report` command in `src/b2aiprep/cli.py`; verify `b2aiprep-cli qa-report --help` shows `--confidence-target`, `--output-format`, `--fail-on-below-target`

**Checkpoint**: All three user stories functional — full pipeline flow from automated QA through human review to release report.

---

## Phase 6: Polish & Cross-Cutting Concerns

- [ ] T034 [P] Validate determinism (SC-002): run `b2aiprep-cli qa-run` twice on the same synthetic batch with the same config; assert `qa_composite_scores.tsv` outputs are byte-for-byte identical; confirm random seed pinning works across Whisper, GLiNER, Phi-4, and pyannote in `tests/test_determinism.py`
- [ ] T035 [P] Validate SC-006 ground-truth examples: assert clean single-speaker on-task audio → PASS; multi-speaker audio → FAIL or NEEDS_REVIEW; PII-containing audio → FAIL or NEEDS_REVIEW; off-task audio → FAIL or NEEDS_REVIEW using synthetic fixtures in `tests/test_ground_truth.py`
- [ ] T036 [P] Validate SLURM sharding: assert `shard_audio_list(paths, part=1, num_parts=2)` and `shard_audio_list(paths, part=2, num_parts=2)` produce non-overlapping subsets whose union equals full list; assert running qa-run on both shards separately produces same composite scores as running on the full list in `tests/test_sharding.py`
- [ ] T037 Run the full three-command quickstart flow from `specs/001-audio-quality-pipeline/quickstart.md` on a synthetic dataset; confirm all expected output files present, correctly formatted, and report passes; fix any command-line option discrepancies found between implementation and `contracts/cli-commands.md`
- [ ] T038 [P] Update `CLAUDE.md` to reflect the new active modules (`quality_control.py` extended, `unconsented_speakers.py`, `pii_detection.py`, `task_compliance.py`, `qa_report.py`, `qa_models.py`) and new CLI commands (`qa-run`, `qa-review`, `qa-report`)

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: No dependencies — start immediately
- **Phase 2 (Foundational)**: Depends on Phase 1 — **blocks all user stories**
- **Phase 3 (US1 — P1)**: Depends on Phase 2 — highest priority, implement first
- **Phase 4 (US2 — P2)**: Depends on Phase 2 + Phase 3 outputs (needs `needs_review_queue.tsv`)
- **Phase 5 (US3 — P3)**: Depends on Phase 2 + Phase 3 outputs (needs `qa_composite_scores.tsv`)
- **Phase 6 (Polish)**: Depends on all story phases complete

### User Story Dependencies

- **US1 (P1)**: Depends only on Foundational — no other story dependencies
- **US2 (P2)**: Depends on Foundational + US1 (reads US1 output files) — can start once US1 checkpointed
- **US3 (P3)**: Depends on Foundational + US1 (reads US1 output files) — can start in parallel with US2

### Within Each User Story

- Tests (T009–T013, T024, T029a, T029b) should be written first and verified failing before implementation
- qa_models.py types before check implementations
- Check implementations before composite scoring
- Composite scoring before CLI command wiring
- CLI wiring before registration in cli.py

### Parallel Opportunities Per Phase

**Phase 1**: T001, T002 — fully parallel  
**Phase 2**: T003 first; then T004, T005, T006, T007, T008 in parallel  
**Phase 3**: T009–T013 fully parallel (all different test files); then T014→T015→T016, T017, T018→T019→T020 (T016/T017/T018 parallel after T003–T008); T021 after all checks; T022 after T021; T023 after T022  
**Phase 4**: T024 parallel with implementation; T025→T026→T027→T028 sequential  
**Phase 5**: T029a, T029b parallel with implementation; T030→T031→T032→T033 sequential  
**Phase 6**: T034, T035, T036, T038 fully parallel; T037 after all story phases

---

## Parallel Example: User Story 1 Tests

```bash
# Launch all US1 test stubs in parallel (write and verify failing):
Task: "Write synthetic WAV fixture factory ... in tests/test_quality_control.py"       # T009
Task: "Write unconsented-speaker detection tests ... in tests/test_unconsented_speakers.py"  # T010
Task: "Write PII detection tests ... in tests/test_pii_detection.py"                   # T011
Task: "Write task compliance tests ... in tests/test_task_compliance.py"               # T012
```

## Parallel Example: User Story 1 Check Implementations

```bash
# After T014/T015 (quality_control.py), these three can run in parallel:
Task: "Implement unconsented_speakers.py ..."   # T016
Task: "Implement pii_detection.py ..."          # T017
Task: "Implement Tier A task compliance ..."    # T018
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 (Setup) + Phase 2 (Foundational)
2. Complete Phase 3 (US1): write tests → implement checks → wire CLI
3. **STOP and VALIDATE**: run synthetic batch, confirm all output files, check determinism
4. Demo `b2aiprep-cli qa-run ... && cat qa_composite_scores.tsv`

### Incremental Delivery

1. Setup + Foundational → shared infrastructure ready
2. US1 → automated batch pipeline ships (MVP)
3. US2 → human review loop ships (adds ethical compliance coverage for borderline cases)
4. US3 → release report ships (enables defensible quality claims to external stakeholders)
5. Polish → determinism, sharding, ground-truth validation confirmed

### Parallel Team Strategy

Once Phase 2 (Foundational) is complete:
- **Developer A**: US1 check implementations (T014–T020 sequential per module)
- **Developer B**: US1 test stubs (T009–T013) + US2 test stub (T024) in parallel
- After US1 checkpoint: Developer A → US2 implementation; Developer B → US3 implementation

---

## Notes

- `[P]` tasks operate on different files or clearly independent functions — verify no same-file conflicts before parallelising
- Each user story has an independent test that can be used as a checkpoint before proceeding
- Sensitive sidecar files (full transcript + PII spans) are written to BIDS subject directories, not to the shared TSV outputs at BIDS root — preserve this separation throughout implementation
- DDK rate thresholds, phonation duration minimums, and environment noise classes are all in `PipelineConfig` — never hardcode them in check implementations
- Model failures must always route through `make_error_check_result` (T008); never let a single check exception halt the full pipeline run
