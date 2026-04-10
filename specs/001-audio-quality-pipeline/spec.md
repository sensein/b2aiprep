# Feature Specification: Audio Quality Assurance Pipeline

**Feature Branch**: `185-audio-quality-pipeline`
**Created**: 2026-04-09
**Status**: Draft
**Input**: User description: "Quality check/assurance pipeline for Bridge2AI Voice dataset release"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Automated Batch Quality Screening (Priority: P1)

A dataset release manager runs the quality pipeline against a batch of audio recordings. The
pipeline automatically evaluates each audio against all four quality checks (technical quality,
unconsented speakers, PII disclosure, task compliance), produces per-check confidence scores,
and classifies each audio as **pass**, **fail**, or **needs human review**.

**Why this priority**: This is the core pipeline that gates dataset release. Without it, no
quality claims can be made. Every other story depends on the outputs produced here.

**Independent Test**: Can be tested end-to-end by running the pipeline on a small set of
synthetic audio files with known ground truth (clean, noisy, multi-speaker, PII-containing)
and verifying that classifications and confidence scores are produced correctly and
reproducibly.

**Acceptance Scenarios**:

1. **Given** a directory of audio files and a task-metadata manifest, **When** the pipeline
   runs, **Then** every audio receives a per-check score, a composite quality score, and a
   classification (pass / fail / needs-review).
2. **Given** the same inputs on two separate runs, **When** the pipeline executes, **Then**
   outputs are bit-for-bit identical (determinism requirement).
3. **Given** an audio with SNR below threshold, **When** evaluated for technical quality,
   **Then** it is classified as fail (or needs-review if confidence is borderline).
4. **Given** a recording containing a second, unconsented voice, **When** evaluated for
   speaker presence, **Then** it is classified as fail (or needs-review if confidence is
   borderline).
5. **Given** a recording where the participant verbally discloses a PII item, **When**
   evaluated for PII, **Then** it is classified as fail (or needs-review).
6. **Given** a recording where the participant clearly did not perform the assigned task,
   **When** evaluated for task compliance, **Then** it is classified as fail (or
   needs-review).

---

### User Story 2 - Human Review of Flagged Audios (Priority: P2)

A human reviewer receives the list of audios classified as **needs-review** from the automated
pipeline. They can listen to each audio, see the model's confidence breakdown per check, and
record a manual accept or reject decision. The accepted/rejected decisions are persisted and
feed back into the final quality report.

**Why this priority**: The automated pipeline cannot achieve 100% confidence on all audios.
A human-in-the-loop step is essential for ethical compliance (PII, unconsented speakers) and
for borderline cases where automated judgment is uncertain.

**Independent Test**: Can be tested by feeding a curated set of known borderline audios into
the review interface, submitting accept/reject decisions, and verifying that decisions are
persisted correctly and override automated classifications in the final report.

**Acceptance Scenarios**:

1. **Given** a list of needs-review audios, **When** the reviewer opens the review interface,
   **Then** each audio is presented with its per-check confidence scores and an audio playback
   option.
2. **Given** an audio in review, **When** the reviewer records a decision (accept / reject),
   **Then** the decision is persisted alongside the automated scores.
3. **Given** a reviewer's accept decision, **When** the final report is generated, **Then** the
   audio is counted as passing, with the human override noted.

---

### User Story 3 - Release Quality Report (Priority: P3)

After automated screening and human review are complete, the pipeline generates a
release-quality report. The report summarizes pass rates per check, the composite quality
score distribution, confidence levels, and makes a top-level claim: "At confidence level X%,
Y% of released audios pass all quality checks."

**Why this priority**: Enables the team to make defensible, quantifiable claims about dataset
quality to external researchers and ethics boards.

**Independent Test**: Can be tested by supplying a completed set of per-audio quality results
(mix of passes, fails, human overrides) and verifying that the report correctly computes
aggregate statistics and confidence claims.

**Acceptance Scenarios**:

1. **Given** all per-audio results (automated + human), **When** the report is generated,
   **Then** it includes per-check pass rates, composite score distribution, and a top-level
   confidence claim.
2. **Given** a configured confidence threshold (e.g., 95%), **When** the report is generated,
   **Then** the report explicitly states whether the released batch meets that threshold.
3. **Given** the same result set on two runs, **When** the report is generated, **Then**
   outputs are identical (reproducibility).

---

### Edge Cases

- What happens when an audio file is corrupt or unreadable?
- How does the system handle silence-only recordings?
- What happens when task-metadata for an audio is missing?
- How are multi-session participants handled if one session is flagged?
- What if the PII detection model is unavailable or returns an error?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The pipeline MUST compute a technical audio quality score for each recording,
  encompassing at minimum: signal-to-noise ratio estimate, clipping detection, silence ratio,
  and sample rate / channel format compliance.
- **FR-002**: The pipeline MUST detect the presence of additional speakers beyond the primary
  participant in each recording and flag recordings with detected non-participant voices.
- **FR-003**: The pipeline MUST transcribe audio and detect the disclosure of personally
  identifiable information, covering at minimum: names, dates of birth, geographic identifiers,
  phone numbers, and health-related identifiers.
- **FR-004**: The pipeline MUST assess whether the participant performed the assigned task
  correctly, using task-type metadata (from `audio_task_descriptions.json`) to apply
  compliance criteria grouped by task category. The 788 distinct task IDs map to the
  following compliance groups:
  All task compliance checks produce a **confidence score (0–1)** indicating how correctly
  the task was performed, rather than a binary pass/fail. These scores feed into the
  composite quality score and flagging thresholds.
  - **Sentence/passage reading** (harvard-sentences ×720, cape-V-sentences ×12, sentence,
    passage, rainbow, caterpillar-passage, repeat-words): transcript of the recording is
    compared against the known prompt text; compliance confidence is proportional to
    transcript similarity (e.g., word error rate or character-level similarity) to the
    expected prompt.
  - **Sustained phonation** (prolonged-Vowel, maximum-phonation-time, long-sounds):
    sustained phonation detected for a task-appropriate minimum duration AND the dominant
    phoneme in the recording is verified to match (or be phonetically close to) the target
    phoneme; compliance confidence combines duration adequacy with phoneme accuracy.
  - **Pitch glides** (high-to-low, glides-high-to-low, glides-low-to-high):
    pitch variation across expected range detected.
  - **Diadochokinesis** (diadochokinesis ×10 variants): the repeated phoneme sequence is
    verified against the target phoneme(s) for that variant (e.g., /pʌ/, /tʌ/, /kʌ/,
    /pʌtʌkʌ/); compliance confidence combines repetition-rate regularity with phoneme
    accuracy.
  - **Fluency / naming** (naming-animals, naming-food, animal-Fluency,
    productive-Vocabulary, random-item-generation): multiple discrete responses detected
    across the recording duration.
  - **Breathing / cough** (respiration-and-cough ×9, breath-sounds):
    non-speech respiratory event detected; absence of continuous speech.
  - **Story / picture description** (story-recall, cinderella-story, picture,
    picture-description, pictures-and-doors): extended connected speech detected.
  - **Sequential recitation** (123s, abcs, days, months): speech detected; duration
    consistent with reciting the full sequence.
  - **Conversational / personal** (favorite-food, favorite-show-movie-game,
    outside-of-school, role-naming, ready-for-school, choose-book): speech detected;
    minimum duration met.
  - **Cognitive** (word-color-stroop): speech detected; minimum duration met.
  - **Loudness** (loudness): speech detected; amplitude variation present.
- **FR-005**: The pipeline MUST produce a composite quality score for each audio that combines
  per-check scores with configurable per-task-type weights.
- **FR-006**: Audios whose composite score or any individual check confidence falls below
  configurable thresholds MUST be routed to a human review queue rather than auto-classified.
- **FR-007**: The human review interface MUST present each flagged audio with its per-check
  confidence breakdown, allow audio playback, and record a binary accept/reject decision.
- **FR-008**: The pipeline MUST produce a release report summarizing: per-check pass rates,
  composite score distribution, count of human overrides, and an overall confidence-level
  claim about the released batch.
- **FR-009**: All pipeline outputs MUST be deterministic given the same audio inputs, model
  versions, and configuration (random seeds pinned; model versions locked).
- **FR-010**: Quality check configuration (thresholds, weights, model versions) MUST be
  recorded in the output artifacts so results can be reproduced and audited.
- **FR-011**: The pipeline MUST handle corrupt or unreadable audio files gracefully, logging
  the error and excluding the file from release rather than halting the pipeline.
- **FR-012**: The pipeline MUST be runnable via the existing `b2aiprep-cli` command interface.

### Key Entities

- **AudioRecord**: A single audio file + its task-type metadata + participant ID.
- **CheckResult**: Output of one quality check for one audio — score (0–1), confidence (0–1),
  classification (pass / fail / needs-review), and check-specific detail fields.
- **CompositeScore**: Weighted combination of per-check scores for one audio, with overall
  confidence and final classification.
- **ReviewDecision**: A human reviewer's accept/reject decision for one audio, including
  reviewer ID and timestamp.
- **QualityReport**: Aggregate summary over a batch: per-check statistics, composite
  distribution, human override counts, top-level confidence claim.
- **PipelineConfig**: Versioned configuration artifact — check thresholds, composite weights,
  model versions, and task-specific compliance parameters.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The pipeline processes a full dataset batch without manual intervention for
  audios that clearly pass or clearly fail (human review queue is bounded to genuinely
  ambiguous cases).
- **SC-002**: Running the pipeline twice on the same inputs produces identical classifications
  and scores on 100% of audios.
- **SC-003**: The release report makes a defensible confidence claim: the team can assert with
  a stated confidence level (e.g., ≥ 95%) that released audios pass all quality checks.
- **SC-004**: Human review is required for no more than a target fraction of audios (threshold
  configurable), keeping reviewer burden tractable for large batches.
- **SC-005**: All quality check computations complete within a time budget that does not
  significantly exceed the time already required for existing pipeline processing steps.
- **SC-006**: The pipeline correctly identifies known ground-truth examples: clean single-speaker
  on-task audios pass; multi-speaker, PII-containing, or off-task audios are flagged.

## Assumptions

- The B2AI Voice dataset BIDS output produced by the existing `b2aiprep` pipeline is the input
  to the quality pipeline (not raw RedCap files).
- Task-type labels are available in the BIDS metadata for each audio file.
- A participant's consent record is available and can be used to establish that only the
  enrolled participant's voice is consented.
- PII detection covers spoken PII only (not metadata fields — those are handled separately).
- The human review interface is a CLI tool consistent with the project's existing CLI style;
  a web UI is out of scope for v1.
- Composite score weighting is configurable per task type but will ship with sensible defaults.
- Model weights and versions are pinned at pipeline configuration time; updates to models
  constitute a new pipeline configuration version.
