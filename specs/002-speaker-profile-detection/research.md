# Research: Speaker Profile-Based Unconsented Speaker Detection

**Date**: 2026-04-24
**Plan**: [plan.md](plan.md)

---

## Existing Assets in the Pipeline

The `_features.pt` files already contain:

| Field | Value |
|-------|-------|
| `speaker_embedding` | 192-dim ECAPA-TDNN vector (`speechbrain/spkrec-ecapa-voxceleb`) |
| `diarization` | List of `SPEAKER_N: [start – end]` Segment objects from pyannote |
| `is_speech_task` | Boolean; `False` for picture, long-sounds, and other non-speech tasks |
| `transcription` | ScriptLine from Whisper |
| `duration` | Recording duration (seconds) |

No new embedding extraction step is needed for the initial profile-building phase; the existing embeddings are reused.

---

## Decision 1: ECAPA-TDNN limitations on child speech

**Decision**: ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) is adequate for adult
participants but is known to degrade severely on child speech.

**Rationale**:
- Trained exclusively on VoxCeleb1/2 (adult celebrity interviews)
- On VoxCeleb1-O (adult): ~0.87% EER
- On child corpora (CSLU/MyST): ~20–35% EER — a 25–40× relative degradation
- Age gradient: worst for grades K–1 (ages 5–7), approaches adult performance by
  age 14–15
- 2024 TSD paper confirmed: fine-tuned ECAPA-TDNN on child data reached 2.9% EER
  (vs. 10.8% baseline), still 3–4× worse than adult

**Alternatives considered**:
- WavLM-base-plus-sv (`microsoft/wavlm-base-plus-sv`): improved cross-domain
  generalization for adults; no published benchmark specifically on children
- CAM++ (3D-Speaker toolkit): 18% lower EER than ECAPA-TDNN on VoxCeleb, similar
  child-speech gap

**Recommended path for pediatric data**: ChildAugment fine-tuned ECAPA-TDNN (see
Decision 3 below). Apply existing `speechbrain/spkrec-ecapa-voxceleb` for adult
participants; use a child-adapted variant for participants under ~14 years.

---

## Decision 2: Speaker profile aggregation method

**Decision**: Quality-weighted centroid with outlier rejection, reusing existing
ECAPA-TDNN embeddings from `.pt` files.

**Rationale**:
Simple mean pooling of all enrollment embeddings is vulnerable to contamination by:
- Non-speech or near-silence recordings (embeddings near random, low quality)
- Recordings where a second speaker was present (contaminates centroid)
- Very short speech segments (< 3 s active speech)

The chosen approach:
1. **Task-based gating**: exclude recordings whose task type is known to produce
   unreliable embeddings (DDK, breathing, sustained phonation < 3 s, silence tasks)
   and any recording with < 3 s of active speech (from diarization timestamps)
2. **Quality weighting**: weight each embedding by
   `w_i = min(active_speech_s / 10.0, 1.0) × snr_weight` where `snr_weight` maps
   estimated SNR to [0.5, 1.0]; both values derivable from existing features
3. **Outlier rejection**: compute N×N pairwise cosine similarity, reject embeddings
   with mean pairwise similarity < (overall_mean − 1.5 × std); protects against
   unconsented-speaker contamination in enrollment data

**Alternatives considered**:
- PLDA backend: Advantageous for domain mismatch and short utterances, but requires
  a held-out cohort of sufficient size for covariance estimation; deferred to a
  future iteration once a sufficient B2AI cohort is available
- Pure cosine similarity with fixed threshold: Current placeholder implementation;
  does not account for enrollment quality variation

**Scoring at verification time**: cosine similarity between test embedding and
weighted centroid, normalized by AS-norm if a cohort of held-out speakers is
available (optional; improves calibration).

---

## Decision 3: Child speech adaptation

**Decision**: ChildAugment fine-tuning of `speechbrain/spkrec-ecapa-voxceleb` for
the pediatric dataset sub-population.

**Rationale**:
- ChildAugment (JASA 2024, arXiv:2402.15214) uses LPC-based formant warping to
  synthesise child-like speech from adult VoxCeleb training data — no child speech
  required
- Published improvement: ~11–12% relative EER reduction on boys and girls vs. vanilla
  ECAPA-TDNN
- Code available at github.com/vpspeech/ChildAugment; compatible with SpeechBrain
  training recipes
- Drop-in replacement for the existing 192-dim embedding model

**Implementation**: The fine-tuned model is used only when participant age metadata
(from BIDS `participants.tsv`) indicates a child. If age is unavailable, fall back
to the standard ECAPA-TDNN model. The `AudioRecord` dataclass should be extended
with an optional `participant_age_years` field.

**Alternatives considered**:
- G-IFT adapter (arXiv:2508.07836): Most data-efficient when some B2AI child
  recordings are available as labelled training data; recommended for a future
  iteration if child recordings with confirmed ground-truth speaker identity become
  available
- Age-Agnostic Speaker Verification (WOCCI 2025, arXiv:2508.01637): Universal
  architecture with age disentanglement; no pretrained checkpoint released yet
- WavLM-base-plus-sv: Modest improvement on adults; not validated on children

---

## Decision 4: Non-speech recording handling

**Decision**: Gate on active speech fraction computed from existing diarization
timestamps; never compute a cosine similarity score for low-speech recordings.

**Rationale**:
- Published guidance: embeddings from < 1 s active speech are unreliable; < 3 s
  is the practical low-confidence floor
- ECAPA-TDNN internal attention partially suppresses non-speech frames but is
  insufficient for recordings that are > 85% silence or non-speech
- The `diarization` field in `.pt` files provides speaker-segment boundaries from
  which active speech duration is directly computable (sum of segment lengths for
  the primary speaker)
- `is_speech_task` flag is already present but is task-level metadata, not a
  speech-fraction measure; it should be used as an additional prior (if
  `is_speech_task=False`, apply a lower confidence ceiling)

**Threshold rationale**:
- `< 15% active speech fraction` → confidence = 0.30, classification = needs_review
  (independent of similarity score)
- `< 3 s active speech (absolute)` → down-weight in profile construction but still
  score; flag as low-weight enrollment candidate
- US3 (research component) will empirically validate or revise these thresholds

---

## Decision 5: Beyond clustering — recommended verification framework

**Decision**: Two-stage pipeline: (1) quality-weighted centroid enrollment,
(2) AS-norm calibrated cosine scoring. PLDA deferred.

**Rationale**:
- For large-margin-trained embeddings (like ECAPA-TDNN with AAM-Softmax), cosine
  scoring is competitive with PLDA in matched conditions (Interspeech 2022,
  arXiv:2204.03965)
- PLDA becomes beneficial under domain mismatch and with short utterances; can be
  added when a sufficient B2AI cohort (≥ 200 speakers) is available for training
- AS-norm with a held-out cohort of 100–500 speakers provides 4–10% relative EER
  improvement with minimal engineering overhead
- Trainable AS-norm (TAS-norm, arXiv:2504.04512) yields a further 4.11% relative
  EER reduction over standard AS-norm

---

## Decision 6: Minimum speech duration thresholds (from literature)

| Active speech duration | Embedding reliability | Recommended treatment |
|-----------------------|----------------------|----------------------|
| < 1 s | Very low | Exclude from enrollment; score with confidence 0.10 |
| 1–3 s | Low | Down-weight in enrollment (w × 0.3); score with confidence 0.30 |
| 3–10 s | Moderate | Include in enrollment at full weight; confidence 0.70 |
| > 10 s | High | Full enrollment weight; confidence 0.90 |

Source: Interspeech research on short-segment ECAPA-TDNN; quality measure
frameworks (ScienceDirect doi:10.1016/j.dsp.2018.07.012).

---

## Key References

- ChildAugment: arXiv:2402.15214 / github.com/vpspeech/ChildAugment
- G-IFT (child low-resource fine-tuning): arXiv:2508.07836
- FT-Boosted SV (Interspeech 2025 child SV): arXiv:2505.20222
- ECAPA-TDNN on children (TSD 2024): doi:10.1007/978-3-031-70566-3_6
- Cosine vs PLDA for large-margin embeddings: arXiv:2204.03965
- TAS-norm: arXiv:2504.04512
- AS-norm PyTorch: github.com/nidwbin/AS-Norm
- Outlier-rejection enrollment: SCITEPRESS 2025 doi:10.5220/0013256800003890
- Short utterance quality: doi:10.1016/j.dsp.2018.07.012
- VAD-gated embeddings for diarization: arXiv:2405.09142
- SVeritas robustness benchmark: arXiv:2509.17091
- CAM++: arXiv:2303.00332
- WavLM-base-plus-sv: huggingface.co/microsoft/wavlm-base-plus-sv
