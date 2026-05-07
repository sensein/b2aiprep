# Tasks: Speaker Profile-Based Unconsented Speaker Detection

**Input**: Design documents from `specs/002-speaker-profile-detection/`
**Branch**: `186-speaker-profile-detection`

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to
- Exact file paths in all descriptions

---

## Phase 1: Setup

**Purpose**: Verify environment has all required components before writing code.

- [x] T001 Verify `speaker_embedding` (192-dim) and `sparc["spk_emb"]` (64-dim) are accessible in at least one `_features.pt` file from the active dataset; print shapes and confirm no `None` values — no code change, just a sanity-check script run in the existing conda env

**Checkpoint**: Confirmed both embedding fields are present in `.pt` files

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared data model changes required by both US1 and US2.

**⚠️ CRITICAL**: Complete before any US1/US2 implementation begins.

- [x] T002 Update `src/b2aiprep/prepare/qa_models.py`: (a) add `participant_age_years: Optional[float] = None` field to the `AudioRecord` dataclass; (b) add a `speaker_profile` sub-dict to `PipelineConfig` with fields: `min_profile_recordings=3`, `min_active_speech_s=3.0`, `low_confidence_speech_fraction=0.15`, `outlier_rejection_std_multiplier=1.5`, `contamination_quality_threshold=0.30`, `ecapa_cosine_threshold=0.25`, `sparc_cosine_threshold=0.20`, `excluded_task_prefixes=["Diadochokinesis","Prolonged-vowel","Maximum-phonation-time","Respiration-and-cough","Glides","Loudness","long-sounds","silly-sounds","repeat-words"]`

- [x] T003 [P] Update `src/b2aiprep/prepare/resources/qa_pipeline_config.json`: add `"speaker_profile"` top-level key with all nine fields from T002 and their defaults

**Checkpoint**: `PipelineConfig` can be loaded with `speaker_profile` sub-dict; `AudioRecord` accepts `participant_age_years`

---

## Phase 3: User Story 1 — Speaker Profile Construction (Priority: P1) 🎯 MVP

**Goal**: Build per-participant dual-centroid (ECAPA-TDNN + SPARC) speaker profiles from existing `_features.pt` files via a new `build-speaker-profiles` CLI command.

**Independent Test**: Given a BIDS dataset with 10 participants each having ≥5 speech-based recordings, run `b2aiprep-cli build-speaker-profiles BIDS_DIR PROFILES_DIR` and verify each participant has a `sub-{pid}/speaker_profile.json` containing non-null `ecapa_embedding_centroid` (len=192), `sparc_embedding_centroid` (len=64), and `profile_status ∈ {ready, insufficient_data, contaminated}`.

### Implementation for User Story 1

- [x] T004 [P] [US1] Create `src/b2aiprep/prepare/speaker_profiles.py` — define `SpeakerProfile` dataclass matching `data-model.md` (all fields including `ecapa_embedding_centroid`, `sparc_embedding_centroid`, `ecapa_profile_quality_score`, `sparc_profile_quality_score`, `profile_status`, `age_group`, `included_recordings`, `excluded_recordings`, `created_at`, `pipeline_config_hash`); add `load_speaker_profile(profiles_dir, participant_id) -> Optional[SpeakerProfile]` function that reads `sub-{pid}/speaker_profile.json` and deserialises to the dataclass

- [x] T005 [P] [US1] Write test file `tests/test_speaker_profiles.py` — create pytest fixtures: (a) a minimal fake `.pt` dict with `speaker_embedding` (torch.Tensor shape [192]), `sparc={"spk_emb": np.zeros(64)}`, `diarization` list with one segment covering 12 s, `is_speech_task=True`; (b) a tmp BIDS directory with 5 participants × 6 recordings; write test stubs for: task prefix exclusion, speech duration gating, dual-centroid computation, outlier rejection, `insufficient_data` status, `contaminated` status (quality < 0.30), JSON round-trip of `SpeakerProfile`

- [x] T006 [US1] Implement the profile builder function `build_speaker_profiles(bids_dir, profiles_dir, config)` in `src/b2aiprep/prepare/speaker_profiles.py` — full algorithm: (1) scan BIDS dir for `_features.pt` files grouped by participant; (2) for each recording load `speaker_embedding`, `sparc["spk_emb"]`, `diarization`, task name; (3) gate by `excluded_task_prefixes` using case-insensitive `task_name.lower().startswith(prefix.lower())`; (4) compute `active_speech_s` from diarization; (5) compute `active_speech_fraction = active_speech_s / total_duration_s`; gate: skip (log reason `low_speech_fraction`) if `active_speech_fraction < low_confidence_speech_fraction` (default 0.15); (6) skip if `active_speech_s < 1 s`; down-weight (× 0.3) if 1–3 s; (7) compute weight `w_i = min(active_speech_s / 10.0, 1.0)`; (8) L2-normalise and accumulate weighted ECAPA and SPARC embedding arrays; (9) outlier-reject each set independently: pairwise cosine, drop if mean_pairwise_sim < overall_mean − 1.5 × std; (10) compute weighted centroid for each embedding type from survivors; (11) compute `profile_quality_score` per type; (12) set `profile_status`; (13) write `profiles_dir/sub-{pid}/speaker_profile.json` — depends on T002, T004

- [x] T007 [US1] Add `build-speaker-profiles` CLI command to `src/b2aiprep/commands.py` — click command with `BIDS_DIR` and `PROFILES_DIR` positional args and options: `--pipeline-config`, `--task-exclude` (comma-separated, overrides config default), `--min-active-speech`, `--min-recordings`, `--age-col`, `--part`, `--num-parts`, `--log-level`; calls `build_speaker_profiles()`; registers in the CLI group — depends on T006; follow the no-class-definition-in-commands.py rule (all imports at top level)

- [x] T008 [US1] Fill in and pass tests in `tests/test_speaker_profiles.py` — verify all stubs from T005 pass after T006 is implemented; ensure task prefix gating uses prefix match not substring match; test that a participant with 2 usable recordings gets `insufficient_data`; test that a participant whose ECAPA quality score < 0.30 gets `contaminated`

**Checkpoint**: `build-speaker-profiles` runs end-to-end on the peds dataset and produces JSON profiles; `test_speaker_profiles.py` passes

---

## Phase 4: User Story 2 — Per-Recording Verification (Priority: P2)

**Goal**: Update `check_unconsented_speakers()` to use dual-embedding profile comparison (ECAPA-TDNN + SPARC, OR logic) while always running diarization signals regardless of recording duration.

**Independent Test**: Given a participant with a pre-built profile, run `b2aiprep-cli qa-run BIDS_DIR OUTPUT_DIR --profiles-dir PROFILES_DIR` and verify `qa_check_results.tsv` contains columns `ecapa_cosine_similarity`, `sparc_cosine_similarity`, `or_flag` for every `check_type=unconsented_speakers` row; verify a < 1 s recording has `confidence=0.10` and `needs_review` but still has `num_speakers_diarized` populated.

### Implementation for User Story 2

- [x] T009 [P] [US2] Write test file `tests/test_unconsented_speakers_profile.py` — fixtures: a mock `SpeakerProfile` with known ECAPA and SPARC centroids; mock `AudioRecord` with `features_path` pointing to a tmp `.pt` with controlled cosine distances; test stubs for: no profile → needs_review, active_speech_s < 1 s → confidence=0.10 + diarization still runs, active_speech_fraction < 0.15 → confidence=0.30, ECAPA below threshold only → or_flag=True, SPARC below threshold only → or_flag=True, both above threshold → or_flag=False, profile status=insufficient_data → needs_review

- [x] T010 [US2] Refactor `src/b2aiprep/prepare/unconsented_speakers.py` — update `check_unconsented_speakers(audio_record, config, profiles_dir=None)` to: (1) always compute diarization signals (num_speakers, primary_ratio, extra_count) regardless of duration; (2) if `profiles_dir` is None fall back to existing diarization-only logic; (3) load `SpeakerProfile` via `load_speaker_profile()`; missing profile → `needs_review` with `profile_status="missing"`; (4) load `speaker_embedding` and `sparc["spk_emb"]` from `.pt` file; (5) compute `active_speech_s` and `active_speech_fraction`; (6) if `active_speech_s < 1.0` → `confidence=0.10`, `or_flag=True`, `needs_review`; (7) if `active_speech_fraction < low_confidence_speech_fraction` → cap confidence at 0.30; (8) L2-normalise and compute `ecapa_cosine = dot(ecapa_emb, ecapa_centroid)`; (9) compute `sparc_cosine = dot(sparc_emb, sparc_centroid)`; (10) set `or_flag = ecapa_cosine < ecapa_threshold OR sparc_cosine < sparc_threshold`; (11) blend with diarization score for final classification; (12) return `CheckResult` with extended `detail` dict (all `EmbeddingVerificationResult` fields from data-model.md) — depends on T002, T004

- [x] T011 [US2] Add `--profiles-dir` option to the `qa-run` command in `src/b2aiprep/commands.py` — add `@click.option("--profiles-dir", ...)` and pass `profiles_dir` to `check_unconsented_speakers()` at the call site on line ~1695 — depends on T010

- [x] T012 [US2] Fill in and pass tests in `tests/test_unconsented_speakers_profile.py` — verify all stubs from T009 pass after T010 is implemented; confirm diarization signals populated even for < 1 s recording; confirm OR logic is correct for each combination

**Checkpoint**: `qa-run --profiles-dir` produces extended output; `test_unconsented_speakers_profile.py` passes

---

## Phase 5: User Story 3 — Embedding Reliability Research (Priority: P3)

**Goal**: Generate operating characteristic curves (FNR vs review-queue fraction) for ECAPA-TDNN, SPARC, and OR-combined scoring using synthetic participant mixtures; produce a structured report to inform threshold selection.

**Independent Test**: Run `b2aiprep-cli embedding-reliability-report BIDS_DIR PROFILES_DIR --output-format both` and verify `embedding_reliability_report.json` exists containing `ecapa_operating_curve`, `sparc_operating_curve`, `or_operating_curve` arrays each with ≥ 5 threshold points, and `embedding_reliability_report.md` with a human-readable summary.

### Implementation for User Story 3

- [x] T013 [P] [US3] Write test file `tests/test_embedding_reliability.py` — stubs for: synthetic mixture generation (verify output audio has correct duration ratio), operating characteristic computation (given known-cosine scores for positive/negative pairs, verify FNR and FPR at a threshold), report JSON schema validation

- [x] T014 [US3] Create `src/b2aiprep/prepare/embedding_reliability.py` — implement `generate_synthetic_mixtures(bids_dir, profiles_dir, intruder_ratios, intruder_snr_db_values, output_dir)`: (1) for each participant load profile + find clean recordings; (2) randomly select intruder participant; (3) load audio via torchaudio from paths stored in `.pt`; (4) trim intruder to `duration × ratio`; (5) scale to achieve target SNR; (6) overlay at end of target; (7) save to `output_dir/{target}_{intruder}_{ratio}.wav`; (8) return list of `SyntheticMixture` dicts with `label`, `mixed_audio_path`, `target_participant_id`, `intruder_participant_id`, `intruder_duration_ratio`

- [x] T015 [US3] Implement embedding extraction for synthetic mixtures in `src/b2aiprep/prepare/embedding_reliability.py` — function `extract_embeddings_for_mixtures(mixture_list)` that uses `senselab` ECAPA and SPARC extractors to compute embeddings for each mixture audio; returns dict mapping `mixed_audio_path → {ecapa_emb, sparc_emb}` — requires GPU or CPU inference (accept longer runtime for this research command)

- [x] T016 [US3] Implement operating characteristic computation in `src/b2aiprep/prepare/embedding_reliability.py` — function `compute_operating_curves(mixture_list, profiles_dir, emb_dict, speech_fraction_bins)`: score each mixture against target participant profile using both embeddings; compute per-speech-fraction-bin stats (`mean_cosine_same`, `mean_cosine_diff`, `accuracy`, `fpr`); sweep threshold from 0.0 to 1.0 in steps of 0.01; compute FNR and FPR per threshold for ECAPA, SPARC, and OR-combined; return `EmbeddingReliabilityReport` dict — depends on T014, T015

- [x] T017 [US3] Implement report writer in `src/b2aiprep/prepare/embedding_reliability.py` — function `write_reliability_report(report_dict, output_dir, output_format)` that writes `embedding_reliability_report.json` and/or `.md`; Markdown report must include: recommended thresholds achieving ≤5% FNR, review-queue fraction at those thresholds, per-bin accuracy table, knee-point fraction, adult vs. child subgroup breakdown

- [x] T018 [US3] Add `embedding-reliability-report` CLI command to `src/b2aiprep/commands.py` — click command with `BIDS_DIR` and `PROFILES_DIR` positional args and options: `--output-dir`, `--speech-fraction-bins`, `--output-format`, `--intruder-ratios`, `--intruder-snr-db`, `--keep-mixtures` (flag), `--pipeline-config`; orchestrates T014→T015→T016→T017; cleans up `synthetic_mixtures/` unless `--keep-mixtures` — depends on T017

- [x] T019 [US3] Fill in and pass tests in `tests/test_embedding_reliability.py` — verify stubs from T013 pass; test that a mixture with `intruder_duration_ratio=0.4` has the correct audio length; test report JSON contains required keys; test that recommended threshold achieves ≤5% FNR on synthetic positives

**Checkpoint**: `embedding-reliability-report` runs and produces a report; `test_embedding_reliability.py` passes

---

## Phase 5 Addendum: User Story 3 — Real-Data Validation & Rendering (Clarification 2026-05-07)

**Goal**: Implement the optional peds real-data validation section described in FR-009/SC-002 (added via spec clarification). When `--exclusion-list` is supplied, score uncertain positives from the peds release exclusion list against their speaker profiles and report per-signal recall (ECAPA, SPARC, OR-combined, diarization, Evans). Also label the adult→peds scenario as primary in the intruder-type breakdown rendering.

**Independent Test**: Run `embedding-reliability-report BIDS_DIR PROFILES_DIR --exclusion-list EXCL.json` and verify `embedding_reliability_report.json` contains a non-null `real_data_validation` key with fields `num_uncertain_positives`, `num_with_diarization_multispeaker`, `diarization_fraction`, `per_signal_recall` (sub-keys: `ecapa`, `sparc`, `or_combined`, `diarization`, `evans`), and `caveat`. Run without `--exclusion-list` and verify `real_data_validation` is `null` and the report is otherwise complete.

### Implementation for Phase 5 Addendum

- [x] T023 [US3] Implement `compute_real_data_validation(exclusion_list_path, bids_dir, profiles_dir, ecapa_threshold, sparc_threshold, evans_predictions_csv=None, evans_train_annotations_csv=None) -> dict` in `src/b2aiprep/prepare/embedding_reliability.py`: (1) load exclusion list JSON → set of file stems (uncertain positives); (2) scan `bids_dir` for all `_features.pt` files; (3) for each `.pt` whose stem is in the exclusion set, load `speaker_embedding`, `sparc["spk_emb"]`, `diarization`; (4) load participant's `SpeakerProfile` via `load_speaker_profile()`; skip if profile missing or not ready; (5) L2-normalise and compute `ecapa_cosine` vs. profile centroid, `sparc_cosine` vs. profile centroid; apply thresholds → `ecapa_flag`, `sparc_flag`, `or_flag`; (6) parse diarization → `diarization_flag = (num_speakers > 1)`; (7) optionally join Evans predictions by stem using `_load_evans_predictions()` ported from `external_scripts/compare_speaker_detection.py`; (8) compute `per_signal_recall = {signal: n_flagged / max(num_uncertain_positives, 1)}`; (9) return `{num_uncertain_positives, num_with_diarization_multispeaker, diarization_fraction, per_signal_recall: {ecapa, sparc, or_combined, diarization, evans}, caveat}` where `per_signal_recall["evans"]` is `null` (not omitted) when `--evans-predictions` is not supplied, and `caveat` is a fixed string noting that exclusion-list membership does not confirm unconsented-speaker presence — depends on T004, T016

- [x] T024 [US3] Add `--exclusion-list`, `--evans-predictions`, `--evans-train-annotations` optional CLI parameters to `embedding-reliability-report` command in `src/b2aiprep/commands.py`: when `--exclusion-list` is provided, call `compute_real_data_validation()` with the resolved thresholds from the operating-curves report, then merge the result into the report dict as `real_data_validation` before passing to `write_reliability_report()`; when not provided, set `report["real_data_validation"] = None`; all three params default to `None` — depends on T023

- [x] T025 [P] [US3] Update `_render_markdown_report()` in `src/b2aiprep/prepare/embedding_reliability.py`: (a) in the Intruder-Type Breakdown table, label the `adult` row as `"adult→peds (primary)"` when an `adult` key exists in `intruder_type_breakdown`; (b) after the Intruder-Type Breakdown section, add a "Real-Data Validation (Peds Only)" Markdown section when `report["real_data_validation"]` is not null — render: num_uncertain_positives, num_with_diarization_multispeaker, diarization_fraction, a per-signal recall table (`ecapa`, `sparc`, `or_combined`, `diarization`, `evans`), and the caveat text; (c) always render in the report header a one-line note: "Adult cohort: no real-data validation section; threshold calibration is synthetic-only." — this note appears regardless of whether `real_data_validation` is present, since the peds section is always peds-only

- [x] T026 [P] Update `external_scripts/embedding_reliability_report.sh`: add `EXCLUSION_LIST`, `EVANS_PREDICTIONS`, `EVANS_TRAIN_ANNOTATIONS` variables (empty by default) with inline comments documenting the three-scenario approach (adult→peds primary via `--intruder-bids-dir`, peds→peds same-cohort default, adult→adult separate run against adult BIDS); pass `--exclusion-list`, `--evans-predictions`, `--evans-train-annotations` flags conditionally when their variables are non-empty

**Checkpoint**: `embedding-reliability-report --exclusion-list` produces `real_data_validation` section in JSON and Markdown; running without the flag still produces a valid complete report; adult→peds row labelled as primary in Intruder-Type Breakdown

---

## Phase 6: Polish & Cross-Cutting Concerns

- [x] T020 [P] Run full test suite: `pytest tests/test_speaker_profiles.py tests/test_unconsented_speakers_profile.py tests/test_embedding_reliability.py -v` and fix any remaining failures

- [ ] T021 [P] End-to-end integration validation: follow `specs/002-speaker-profile-detection/quickstart.md` steps 1–3 on the peds BIDS dataset; verify profile JSONs contain both centroids; verify `qa_check_results.tsv` contains `ecapa_cosine_similarity` and `sparc_cosine_similarity` columns; spot-check 5 participant profiles for expected task exclusions; verify `embedding-reliability-report` with and without `--exclusion-list` produces valid JSON with correct top-level keys

- [x] T022 Update `specs/002-speaker-profile-detection/spec.md` status from `Draft` to `Complete (US1 implemented; US2 implemented; US3 implemented)` after all phases pass

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies
- **Foundational (Phase 2)**: Depends on Phase 1 — **BLOCKS US1 and US2**
- **US1 (Phase 3)**: Depends on Phase 2
- **US2 (Phase 4)**: Depends on Phase 2 + T004 (SpeakerProfile dataclass)
- **US3 (Phase 5)**: Depends on US1 (profiles must exist); T004 and T010 provide shared helpers
- **US3 Addendum (Phase 5 Addendum)**: Depends on Phase 5 (T016 operating curves) + T004 (profile loader)
- **Polish (Phase 6)**: Depends on all desired stories complete

### Within Each Story

- T004 and T005 [US1] are parallel (dataclass + test stubs, different concerns)
- T006 depends on T002 (PipelineConfig) and T004 (dataclass)
- T007 (CLI) depends on T006 (builder)
- T008 (fill tests) depends on T006

- T009 (test stubs) and T010 (implementation) can be started in parallel after T002+T004
- T011 (CLI wiring) depends on T010
- T012 (fill tests) depends on T010

- T013 (test stubs) [US3] can start any time
- T014 → T015 → T016 → T017 → T018 are sequential within US3

- T023 (real-data validation function) depends on T004 (profile loader) and T016 (to access threshold values)
- T024 (CLI wiring) depends on T023
- T025 (markdown rendering) is parallel to T024 — different code paths in the same function
- T026 (shell script update) is parallel to T023/T024/T025 — different file

---

## Parallel Example: User Story 1

```bash
# Parallel start after T002:
Task T004: "Create SpeakerProfile dataclass + load_speaker_profile() in speaker_profiles.py"
Task T005: "Write test stubs in tests/test_speaker_profiles.py"

# Sequential after T004:
Task T006: "Implement build_speaker_profiles() builder algorithm"
Task T007: "Add build-speaker-profiles CLI to commands.py"
Task T008: "Fill in and pass tests in test_speaker_profiles.py"
```

---

## Implementation Strategy

### MVP (User Story 1 Only)

1. Complete Phase 1: Setup (T001)
2. Complete Phase 2: Foundational (T002, T003)
3. Complete Phase 3: US1 (T004–T008)
4. **STOP and VALIDATE**: `build-speaker-profiles` runs on peds dataset; profiles produced
5. Commit and demo

### Incremental Delivery

1. Setup + Foundational → T001–T003
2. US1 → profiles exist → MVP
3. US2 → qa-run uses profiles → dual-embedding detection active
4. US3 → operating characteristics → threshold calibration

### Note on US3 Runtime

US3 (`embedding-reliability-report`) requires extracting ECAPA-TDNN and SPARC
embeddings from synthetic mixture audio files — this requires model inference and
will be slow on CPU. Run on a SLURM GPU node or allocate sufficient time.
