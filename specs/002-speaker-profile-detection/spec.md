# Feature Specification: Speaker Profile-Based Unconsented Speaker Detection

**Feature Branch**: `186-speaker-profile-detection`
**Created**: 2026-04-24
**Status**: Draft
**Input**: User description: "We want to improve the detection of unconsented speakers by using speaker embeddings to determine how likely it is that someone else was talking during a recording. We have a lot of recordings per speaker, so it might be possible to create some form of a speaker profile that can then be compared against each individual recording. However, not every recording is speech based and so we need to research how well speaker embeddings do when non-speech is present vs. portions of silence."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Speaker Profile Construction (Priority: P1)

A dataset curator wants a stable, per-participant voice profile built automatically
from all available recordings before quality checks run. The profile captures the
expected acoustic characteristics of that participant's voice so that individual
recordings can later be compared against it. Profiles are built only from recordings
that contain sufficient active speech; silence-dominated and non-speech recordings
are excluded from profile construction but are still evaluated in US2.

**Why this priority**: Without a participant profile there is nothing to compare
individual recordings against. This is the foundational prerequisite for every
downstream verification step.

**Independent Test**: Given a BIDS dataset with 10 participants each having ≥5
speech-based recordings, run profile construction and confirm that each participant
produces a profile file containing a summary embedding and quality metadata. Verify
that participants with fewer than the minimum usable recordings are flagged as
"insufficient data" rather than silently dropped.

**Acceptance Scenarios**:

1. **Given** a participant with 15 recordings of which 12 contain ≥10 s of active
   speech, **When** profile construction runs, **Then** the profile is built from
   those 12 speech-rich recordings and the profile quality metadata reports
   `num_recordings_used=12` and total active speech duration.

2. **Given** a participant with only 2 usable speech recordings (below the minimum),
   **When** profile construction runs, **Then** the participant is flagged as
   `profile_status=insufficient_data` and no profile embedding is stored; downstream
   verification for this participant defaults to `needs_review`.

3. **Given** a participant whose recordings include DDK, sustained phonation, and
   breathing tasks (non-speech or near-silence), **When** profile construction runs,
   **Then** those recordings are excluded from the profile but a log entry records
   which recordings were excluded and why.

---

### User Story 2 - Per-Recording Speaker Verification (Priority: P2)

A quality assurance pipeline operator wants every audio recording automatically
compared against the participant's speaker profile and assigned a score indicating
how likely it is that only the enrolled participant is speaking. Recordings with
low active speech (silence-dominated or non-speech tasks) receive a low-confidence
score rather than a false failure or false pass.

**Why this priority**: This is the core detection logic that replaces the current
placeholder unconsented-speaker check and directly improves the ethical compliance
screening of the dataset.

**Independent Test**: Given a synthetic set of recordings — (a) clean single-speaker
from the enrolled participant, (b) recording with a second speaker interjecting,
(c) silence-only recording, (d) DDK (non-speech) recording — run per-recording
verification against a known profile and confirm: (a) scores high similarity
(pass), (b) scores low similarity (fail or needs_review), (c) and (d) are flagged
low-confidence (needs_review) without failing due to embedding mismatch.

**Acceptance Scenarios**:

1. **Given** a recording from the enrolled participant with no other speakers present,
   **When** verified against that participant's profile, **Then** the similarity
   score is above the pass threshold and the recording is classified as pass.

2. **Given** a recording in which a second adult voice speaks for >20% of the
   duration, **When** verified against the enrolled participant's profile, **Then**
   the similarity score falls below the review threshold and the recording is
   classified as fail or needs_review.

3. **Given** a recording that is ≥85% silence or non-speech (e.g., a breathing
   task), **When** verified against the participant's profile, **Then** the system
   returns a low-confidence result (confidence ≤0.30) and classifies it as
   needs_review, never as fail due to embedding mismatch alone.

4. **Given** a participant whose profile has `profile_status=insufficient_data`,
   **When** any of their recordings is verified, **Then** every recording receives
   `needs_review` with a detail note of `no_profile_available`.

---

### User Story 3 - Embedding Reliability Research on Non-Speech Audio (Priority: P3)

A researcher wants an empirical characterisation of how speaker embedding reliability
degrades as active speech fraction decreases, so that principled thresholds can be
set for when embeddings are informative vs. unreliable. This story produces a
report and recommended thresholds, not a real-time screening output.

**Why this priority**: Without empirical evidence the silence/non-speech thresholds
in US2 are arbitrary. This story validates or revises those thresholds using real
data from the dataset.

**Independent Test**: Given recordings with known ground-truth speaker identity and
a range of active-speech fractions (0–100%), compute per-recording embedding
similarity and confirm the report identifies a knee-point below which accuracy drops
sharply.

**Acceptance Scenarios**:

1. **Given** a sample of recordings spanning the full range of speech fractions,
   **When** the research analysis runs, **Then** a report is produced showing
   embedding accuracy at multiple speech-fraction bins (0–15%, 15–30%, 30–50%,
   50–100%).

2. **Given** the research report, **When** its recommended speech-fraction threshold
   is applied as the US2 low-confidence cutoff, **Then** the false positive rate on
   silence-only recordings drops to <2% while preserving ≥90% detection accuracy
   on speech-rich recordings.

---

### Edge Cases

- Participant has exactly the minimum number of usable recordings (marginal profile
  quality — report confidence accordingly in metadata).
- All recordings for a participant are non-speech or silence (profile cannot be
  built — flag as insufficient_data for all recordings).
- Profile is built from recordings that themselves contained unconsented speakers
  (contaminates the profile — flag profile as potentially contaminated when source
  recordings were previously flagged by the prior QA check).
- Child voices: speaker embeddings may underperform for child participants relative
  to adults; the research report must quantify any accuracy gap.
- Very short recordings (<3 s total duration) with marginal speech activity.
- First-time participant with no prior recordings (cold-start — all recordings
  default to needs_review until a profile can be built from sufficient data).

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST construct a per-participant speaker profile from all
  recordings that meet a minimum active-speech duration threshold (default: ≥10 s
  active speech per recording, configurable).

- **FR-002**: Profile construction MUST exclude recordings with active speech
  fraction below a configurable threshold (default: <15%) and log which recordings
  were excluded and the reason.

- **FR-003**: A participant MUST be marked `profile_status=insufficient_data` when
  fewer than a configurable minimum number of usable recordings exist (default: 3).
  All recordings for such participants MUST be routed to needs_review without
  performing an embedding comparison.

- **FR-004**: Each recording MUST be assigned a speaker similarity score (0–1) and
  a confidence value (0–1) reflecting both the match to the participant profile and
  the reliability of the embedding for that recording.

- **FR-005**: Recordings with active speech fraction below the low-confidence
  threshold MUST receive confidence ≤0.30 regardless of the similarity score
  computed, and MUST be classified as needs_review rather than fail.

- **FR-006**: The per-recording result MUST be expressed as a CheckResult compatible
  with the existing QA pipeline output format, replacing the current placeholder
  unconsented-speaker check.

- **FR-007**: Speaker profiles MUST be built by a dedicated `build-speaker-profiles`
  CLI command that runs as a preprocessing step before `qa-run`. Profiles are
  written as reusable output artifacts that `qa-run` reads during the unconsented-
  speaker check. Running `qa-run` without a pre-built profile for a participant
  routes all that participant's recordings to needs_review (same behaviour as
  insufficient_data).

- **FR-008**: Profile metadata (number of recordings used, total active speech
  duration, profile quality score, included and excluded recording identifiers with
  exclusion reasons) MUST be written to a per-participant output file for
  auditability.

- **FR-009**: The research analysis (US3) MUST produce a structured report
  quantifying embedding similarity accuracy as a function of active speech fraction,
  at configurable bin boundaries, with separate breakdowns for child and adult
  participants.

- **FR-010**: Profile construction MUST only use recordings that have already passed
  the basic audio quality hard gates (Stage 1 of the existing QA pipeline); hard-
  gate-failed recordings are excluded from profile input automatically.

### Key Entities

- **SpeakerProfile**: participant identifier, profile embedding (centroid
  representation), number of recordings used in construction, total active speech
  duration used, profile quality score (0–1), profile status (ready /
  insufficient_data / contaminated), list of included recording identifiers, list
  of excluded recording identifiers with exclusion reason.

- **EmbeddingVerificationResult**: participant identifier, session identifier, task
  name, speaker similarity score (0–1), active speech fraction of the recording,
  confidence (0–1), classification (pass / needs_review / fail), detail including
  profile status used and exclusion reason if applicable.

- **EmbeddingReliabilityReport**: speech fraction bin boundaries, per-bin mean
  similarity accuracy, per-bin false positive rate, recommended low-confidence
  threshold, child vs. adult accuracy breakdown.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The enrolled participant's own voice scores above the pass threshold
  in ≥95% of speech-rich recordings (active speech fraction ≥50%) when verified
  against their profile.

- **SC-002**: Recordings confirmed to contain a second speaker score below the
  needs_review threshold in ≥85% of cases.

- **SC-003**: Recordings with active speech fraction <15% are never auto-classified
  as fail due to embedding mismatch alone; they receive needs_review with
  confidence ≤0.30.

- **SC-004**: Adding or removing any single recording from a participant's profile
  changes that participant's per-recording similarity scores by <0.05 on average
  (profile stability under leave-one-out perturbation).

- **SC-005**: The research report (US3) identifies a speech-fraction threshold below
  which embedding accuracy drops by ≥15 percentage points relative to the 50–100%
  speech-fraction bin.

- **SC-006**: End-to-end processing time for profile construction plus per-recording
  verification is no more than 2× the wall-clock time of the current placeholder
  unconsented-speaker check for the same dataset.

## Clarifications

### Session 2026-04-24

- Q: Should evaluations be done per file or use cross-file participant information?
  → A: Evaluations are per file (e.g., does this file have unconsented speakers),
  but information from the same participant (such as speaker embeddings across
  files) can inform those per-file judgements.
- Q: Should profile construction be a separate preprocessing step or automatic
  within qa-run?
  → A: Separate `build-speaker-profiles` CLI command (Option A). Profiles are
  reusable artifacts; qa-run reads them. Missing profile → needs_review for all
  recordings of that participant.

## Assumptions

- Each participant has multiple recordings available at the time the pipeline runs;
  the system is not expected to operate in true streaming/online mode.
- Active speech fraction can be derived from voice activity detection outputs
  already present in the `.pt` feature files produced by `generate-audio-features`.
- Speaker embeddings can be extracted from `.pt` feature files or computed directly
  from audio without requiring a new mandatory preprocessing step.
- Child-voice recordings are present in the pediatric dataset and must be handled;
  no separate child-specific embedding model is assumed for v1, but US3 must
  quantify any accuracy gap.
- The minimum-usable-recordings threshold (default: 3) and the low-confidence
  speech-fraction threshold (default: <15%) are configurable via PipelineConfig.
- Profile construction uses only recordings that have passed Stage 1 audio quality
  hard gates; this is enforced automatically and requires no additional input from
  the user.
