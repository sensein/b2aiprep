# Phase 0 Research: Audio Quality Assurance Pipeline

**Date**: 2026-04-09
**Plan**: [plan.md](plan.md)

---

## 1. Audio Technical Quality (AudioQC)

**Decision**: Extend the existing `quality_control_wrapper()` / Senselab `check_quality()`
rather than adding a new library.

**Finding**: "AudioQC" is not a standalone pip-installable package. The term is used
informally in the team for the quality metrics computed via `senselab.audio.tasks.quality_control`.
The existing implementation in `src/b2aiprep/prepare/quality_control.py` already produces all
five target metrics (clipping, silence, peak SNR, spectral-gating SNR, AM depth) and writes
them to `audio_quality_metrics.tsv`.

**Gap**: The current implementation computes raw metric values but does not translate them into
a confidence score or pass/fail/needs-review classification. Rahul's benchmarking showed a ~14%
misclassification rate on one task set; this must be addressed via configurable per-task-type
thresholds rather than global thresholds.

**SNR thresholds from literature**:
- Voice biomarker research minimum: ≥15 dB (NIDCD guidance); ≥20 dB recommended
- PRAAT pitch tracking reliability degrades below 12 dB SNR
- Hard gate (auto-fail): spectral-gating SNR < 12 dB or clipping > 5% or silence > 50%
- Soft gate (needs-review): SNR 12–18 dB or clipping 1–5% or silence 30–50%
- Pass zone: SNR ≥ 18 dB, clipping < 1%, silence < 30%

**Alternatives considered**:
- [BIDS audioqc plugin](https://github.com/rordenlab/audioqc): outdated, MATLAB-based, not suitable
- Rolling our own librosa/scipy metrics: more work, already solved by senselab

---

## 2. Unconsented Speaker Detection

**Decision**: Layer three signals in order of cost/reliability:
  1. Existing pyannote diarization (via `diarize_audios()`) — primary signal
  2. Evan's model — secondary gate for pediatric speech specifically
  3. Speaker embeddings cross-validation — optional confidence booster

**Finding**: Diarization is already computed in `prepare.py` and stored in `.pt` files.
`quality_control.py` already reads `diarization` from `.pt` files and flags
`no_speakers_found`, `no_primary_speaker_found`, `many_speakers_found`. This is the foundation.

**Gap**: The current diarization check is a hard-flag with no confidence score. It does not:
(a) integrate Evan's model output, (b) use speaker embeddings to verify diarized segments
belong to the same identity, or (c) produce a calibrated confidence.

**Evan's Model**:
- Produces a binary output (0/1) indicating potential non-child/multiple-speaker presence
- When output = 1 in a pediatric session, audio MUST be queued for human review
- Does not replace diarization; provides an independent signal

**Speaker Embeddings cross-validation** (enhancement, not blocking):
- Use existing `extract_speaker_embeddings_from_audios()` from senselab
- For diarized segments: compute cosine similarity of N-1 segment embeddings to verify
  all segments belong to one speaker
- High embedding variance across segments strengthens the unconsented-speaker flag

**pyannote vs alternatives**:
- pyannote (via senselab): already integrated, good accuracy on adult speech
- NeMo (NVIDIA): batch-oriented, higher complexity to integrate, no clear accuracy gain
- USC child/adult diarization: relevant for pediatric recordings but requires separate
  integration effort — defer to future work; Evan's model covers the critical pediatric path

---

## 3. PII Detection

**Decision**: Port GLiNER-PII (primary) + Presidio (secondary/fallback) from the
`pii_detection` branch into a new `src/b2aiprep/prepare/pii_detection.py` module.

**Finding**: Full implementations exist on the `pii_detection` branch:
- `pii_detection_gliner()` using `nvidia/gliner-pii` — detects 18 HIPAA-defined entity
  categories; confidence threshold 0.5 (configurable). Already outputs structured JSON.
- `pii_detection_phi4()` using `microsoft/Phi-4-mini-instruct` — LLM-based detection for
  complex/contextual PII; slower and less deterministic but catches narrative PII that
  pattern matchers miss.
- Presidio imported but not yet fully wired in.

**Transcription trust problem**: All three PII models operate on transcripts, not raw audio.
If Whisper produces an inaccurate transcript (low-quality recording, strong accent, disorder),
PII may go undetected. Mitigation:
- Compute a transcript confidence proxy: word-level Whisper log-probabilities or WER between
  `whisper-tiny` and `whisper-large-turbo` outputs — high divergence signals low trust.
- When transcript confidence is below threshold, route audio to human review regardless of
  PII model output (conservative for ethical compliance).

**PII categories in scope** (per HIPAA safe harbour + common clinical patterns):
name, date_of_birth, geographic_identifier, phone_number, fax_number, email, SSN,
medical_record_number, health_plan_number, account_number, license_number,
device_identifier, URL, IP_address, biometric_identifier, unique_identifier

**Framingham Heart Study LLM approach**: Uses an LLM prompt similar to Phi-4 approach
already on the branch. No additional integration needed; Phi-4 covers this pattern.

**GLiNER vs Presidio comparison**:
- GLiNER-PII: end-to-end NER model, fewer false positives on medical text, fast
- Presidio: regex + NLP hybrid, more configurable, richer redaction operators
- Decision: GLiNER as primary (accuracy); Presidio as fallback when GLiNER confidence < 0.5

---

## 4. Task Compliance Verification

**Decision**: Three-tier approach by task category:
  - Tier A (prompted text tasks): WER / character similarity vs. known prompt
  - Tier B (phoneme-specific tasks): signal-based phoneme/pattern verification
  - Tier C (open/conversational tasks): LLM (Phi-4) boolean + duration gate

**Finding**: A Phi-4-based `task_correctness_phi4()` function exists on the `pii_detection`
branch. It takes instructions + transcript and returns a boolean via LLM prompt. This is
suitable for Tier C but too coarse for Tiers A and B.

### Tier A — Prompted reading (harvard-sentences, cape-V-sentences, passage, rainbow, caterpillar-passage, repeat-words, sentence)

- Whisper transcript is already available in `.pt` files.
- Compute character-normalised edit distance (or WER) between transcript and the known
  prompt string from `audio_task_descriptions.json`.
- Compliance confidence = 1 − normalised_edit_distance (clamped 0–1).
- Known prompt text is available for all 788 task IDs.
- WER < 10% → high confidence; WER 10–30% → moderate; WER > 30% → low confidence.

### Tier B — Phoneme/signal verification

**Sustained phonation** (prolonged-Vowel, maximum-phonation-time, long-sounds):
- Duration check: recording active speech duration (from silence-trimming output) ≥ task
  minimum (e.g., 3 s for prolonged-Vowel).
- Phoneme check: Use Praat formant extraction (already available via parselmouth) to
  compute mean F1/F2 across voiced frames; compare to expected vowel formant targets.
  Confidence = Gaussian likelihood of observed F1/F2 given target vowel distribution.
- Combined confidence = min(duration_conf, phoneme_conf).

**Diadochokinesis** (10 variants: PA, TA, KA, Pataka, buttercup, v1/v2 variants):
- Periodicity check: compute frame-energy autocorrelation to detect repetition rate.
  Expected DDK rate: 5–7 Hz for healthy adults; flag if rate outside 2–9 Hz.
- Phoneme check: Use Praat or CMU Pronouncing Dictionary mapping to extract dominant
  consonant from each burst; compare to target phoneme(s).
- Combined confidence = geometric mean of rate_conf and phoneme_conf.

**Pitch glides** (high-to-low, glides-high-to-low, glides-low-to-high):
- Extract F0 contour using Praat (already in codebase).
- Verify monotone direction of F0 change (slope sign consistency ≥ 70% of voiced frames).
- Confidence = proportion of frames with consistent direction × (1 − F0 tracking gaps).

**Breathing / cough** (respiration-and-cough, breath-sounds):
- Verify absence of prolonged voiced speech (< 10% voiced frames).
- Cough tasks: detect transient high-energy bursts (amplitude spike + rapid decay pattern).
- HeAR embeddings (Google Health Foundation Model): not yet integrated; deferred to a
  subsequent sprint as an enhancement for this task group.

### Tier C — Open/conversational/free speech

- Tasks: naming-animals, naming-food, story-recall, cinderella-story, picture, favorite-*,
  role-naming, abcs, 123s, days, months, word-color-stroop, etc.
- LLM (Phi-4): instructions + transcript → boolean compliance + short rationale.
- Duration gate: minimum active-speech duration (configurable per task, default 3 s).
- Confidence = 0.9 if LLM=true AND duration_met, 0.3 if LLM=false, 0.5 if LLM=true
  but duration not met or transcript confidence low.

---

## 5. Composite Quality Score

**Decision**: Multi-stage gating with configurable per-task-type weighted soft score.

### Architecture

```
Stage 1 — Hard gates (auto-FAIL, no score needed):
  clipping > 5%  OR  silence > 50%  OR  SNR < 12 dB

Stage 2 — Forced review gates (auto-NEEDS-REVIEW):
  Evan's model == 1  (pediatric unconsented speaker flag)
  transcript_confidence < threshold (PII result untrustworthy)

Stage 3 — Soft composite score [0.0–1.0]:
  composite = Σ (weight_i × check_confidence_i) / Σ weight_i
  where weights are per-task-type from PipelineConfig

  Default weights (adjustable in config):
    technical_audio_quality:  0.30
    unconsented_speakers:     0.25
    pii_disclosure:           0.25
    task_compliance:          0.20

Stage 4 — Classification:
  composite ≥ 0.75  AND  all checks ≥ 0.5  → PASS
  composite < 0.40  OR  any hard check failed → FAIL
  otherwise → NEEDS_REVIEW

Stage 5 — Confidence estimate:
  confidence = 1 − std_dev(per-check confidences)
  (high variance across checks → lower confidence in composite)
```

**Rationale for weighted sum over learned ensemble**: Interpretable for external audit;
no labelled training data yet available; weights can be calibrated later when ground truth
labels accumulate. Transition to learned ensemble is a natural future step.

---

## 6. Human Review Interface

**Decision**: CLI-based with audio playback via `sounddevice` (or print-path for HPC where
audio playback may be unavailable), presenting per-check breakdown and recording decision.

**Finding**: No human review interface exists in the codebase. Needs to be built fresh.

**Design**:
- Command: `b2aiprep-cli qa-review [output_dir]`
- Reads `needs_review_queue.tsv` from output_dir
- For each audio: display participant_id, task_name, composite_score, per-check scores
- Attempt playback (graceful degradation if unavailable on HPC)
- Prompt: [a]ccept / [r]eject / [s]kip / [q]uit
- Decisions written to `human_review_decisions.tsv`
- Session-resumable: skipped items remain in queue

---

## 7. Release Report

**Decision**: Single command generating a Markdown + JSON report over a completed batch.

**Outputs**:
- Per-check pass rates
- Composite score distribution (histogram bins)
- Human override counts + override rate
- Top-level confidence claim: "At confidence level X%, Y% of released audios pass all checks"
- Audios excluded (FAIL + rejected) vs. included (PASS + accepted)

**Confidence claim formula**:
```
claim_confidence = mean(composite_confidence) × (1 - human_override_rate × 0.5)
released_pass_rate = (PASS + human_accepted) / (PASS + FAIL + NEEDS_REVIEW)
```

---

## 8. Determinism & Reproducibility Strategy

- All model versions pinned in `PipelineConfig` (stored as JSON with each run's output).
- `PipelineConfig` SHA-256 hash stored in output file headers.
- Whisper: `torch.manual_seed(42)` + fixed `beam_size` + `temperature=0.0` for greedy decode.
- GLiNER-PII: deterministic inference (no sampling).
- Phi-4: `temperature=0`, `do_sample=False` to force greedy decode.
- pyannote diarization: set random seed via `torch.manual_seed` before invocation.
- All thresholds and weights stored in `PipelineConfig` so results can be reproduced
  exactly with the same config file.

---

## 9. All NEEDS CLARIFICATION Items Resolved

All open questions from the spec have been answered via research and codebase survey.
No blockers for Phase 1 design.
