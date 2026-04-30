# Research: Speaker Profile-Based Unconsented Speaker Detection

**Date**: 2026-04-30 (updated from 2026-04-24)
**Plan**: [plan.md](plan.md)

---

## Existing Assets in the Pipeline

The `_features.pt` files already contain:

| Field | Value |
|-------|-------|
| `speaker_embedding` | 192-dim ECAPA-TDNN vector (`speechbrain/spkrec-ecapa-voxceleb`) |
| `sparc["spk_emb"]` | 64-dim SPARC speaker encoding (articulatory-feature-based) |
| `diarization` | List of `SPEAKER_N: [start – end]` Segment objects from pyannote |
| `is_speech_task` | Boolean; `False` for picture, long-sounds, and other non-speech tasks |
| `transcription` | ScriptLine from Whisper |
| `duration` | Recording duration (seconds) |

Both `speaker_embedding` and `sparc["spk_emb"]` are available in every `.pt` file
without running new inference.

---

## Decision 1: ECAPA-TDNN on child speech — acceptable for adult-intruder detection

**Decision**: Use adult ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`) for **all**
participants, including pediatric. No child-adapted model is used.

**Rationale**:
- ECAPA-TDNN degrades on child speaker *verification* (20–35% EER on CSLU/MyST vs.
  0.87% on adults — 25–40× relative degradation)
- However, the goal for **pediatric recordings is adult-intruder detection**, not
  child identity verification
- Adult and child voices occupy largely non-overlapping regions of the ECAPA-TDNN
  embedding space; an adult intruder in a child recording produces a large cosine
  distance from the child's profile even in the adult model
- The SPARC `spk_emb` (Decision 3 below) provides a complementary articulatory-
  feature-based embedding that may further separate adult/child voices
- Using the same model for all participants eliminates age-conditioned branching,
  reduces operational complexity, and avoids requiring reliable age metadata

**Implications for child profiles**:
- Child profiles will have higher intra-participant variance (noisier centroid)
- Outlier rejection (Decision 2) mitigates this: low-coherence centroids are flagged
  `contaminated` and routed to `needs_review`
- US3 research must quantify adult-intruder detection rate on synthetic pediatric
  mixtures specifically

**Alternatives considered and rejected**:
- ChildAugment fine-tuned ECAPA-TDNN (arXiv:2402.15214): Improves child *identity*
  verification but does not address adult-intruder detection. Rejected because it
  optimises the wrong objective.
- G-IFT adapter (arXiv:2508.07836): Same issue; also requires labelled child data
- Age-Agnostic Speaker Verification (WOCCI 2025, arXiv:2508.01637): No pretrained
  checkpoint released

---

## Decision 2: Speaker profile aggregation — dual-centroid, quality-weighted

**Decision**: Build **two independent quality-weighted, outlier-rejected centroids**
per participant: one from ECAPA-TDNN embeddings (192-dim), one from SPARC `spk_emb`
(64-dim). Both use the same gating and weighting logic.

**Rationale**:
Simple mean pooling is vulnerable to:
- Non-speech / near-silence recordings (embeddings near random)
- Recordings with a second speaker (contaminates centroid)
- Very short speech segments (< 3 s active speech)

The chosen approach:
1. **Task-based gating**: exclude recordings whose task name matches any configured
   exclusion prefix (see Decision 8); exclude recordings with < 1 s active speech
2. **Quality weighting**: `w_i = min(active_speech_s / 10.0, 1.0) × snr_weight`
   where `snr_weight` maps estimated SNR to [0.5, 1.0]; both derivable from existing
   features
3. **Outlier rejection**: compute N×N pairwise cosine similarity; reject embeddings
   with mean pairwise similarity < (overall_mean − 1.5 × std); applied independently
   to ECAPA-TDNN and SPARC embedding sets
4. **L2-normalise** before averaging; recompute centroid on surviving set
5. **Profile quality score** = mean pairwise cosine similarity of the final survivor
   set (computed separately for each embedding type)

**Alternatives considered**:
- PLDA backend: Useful for domain mismatch; deferred until ≥ 200-speaker cohort
- Single embedding only: Rejected; two independent signals provide higher recall
  with OR logic (Decision 4)

---

## Decision 3: SPARC speaker embedding as complementary signal

**Decision**: Use the SPARC `spk_emb` (64-dim) as a second independent speaker
representation alongside ECAPA-TDNN, building a separate centroid profile for each.

**Rationale**:
- SPARC (Streaming Phoneme-Articulatory Representation Codec) encodes speech via
  articulatory features: EMA trajectories, pitch, loudness, periodicity, and a
  speaker encoding. The `spk_emb` captures speaker identity via a fundamentally
  different pathway than the discriminative large-margin training of ECAPA-TDNN.
- Both are already in every `.pt` file; no new model inference is required.
- Articulatory-feature-based speaker representations may generalise better across
  recording conditions and age groups, providing complementary error patterns.
- OR-logic combination (flag if either score falls below threshold) maximises recall
  without requiring a learned fusion model.

**Alternatives considered**:
- WavLM-base-plus-sv: Better cross-domain for adults; not available in `.pt` files
  (would require re-running feature extraction)
- CAM++: 18% EER lower than ECAPA-TDNN but also not in `.pt` files
- Learned score fusion (e.g., logistic regression on [ecapa_cos, sparc_cos]): Deferred
  until labelled training data from B2AI is available

---

## Decision 4: Verification framework — dual-embedding OR logic

**Decision**: For each recording, compute cosine similarity against the ECAPA-TDNN
centroid and against the SPARC centroid independently. Flag `needs_review` if
**either** score falls below its respective threshold (OR logic).

**Rationale**:
- For datasets where unconsented speakers must be reliably caught, false negatives
  are more costly than false positives (excess review workload)
- OR logic maximises sensitivity; threshold calibration (Decision 7 and Decision 9)
  allows tuning the operating point to achieve ≤ 5% FN rate while quantifying
  reviewer workload
- Each embedding threshold is independently configurable to account for their
  different score distributions (ECAPA-TDNN cosine ≈ 0.8–1.0 for same-speaker;
  SPARC cosine may differ)
- AS-norm calibration (optional) can improve per-embedding threshold stability;
  deferred until a held-out cohort of 100–500 speakers is available

**Config keys** (all in `PipelineConfig.speaker_profile`):

| Key | Default | Source |
|-----|---------|--------|
| `ecapa_cosine_threshold` | 0.25 | provisional — run `embedding-reliability-report` to calibrate |
| `sparc_cosine_threshold` | 0.20 | provisional — run `embedding-reliability-report` to calibrate |

**Provisional status**: Both thresholds were chosen to lean conservative (lower
cosine = wider net) pending empirical calibration. Use `embedding-reliability-report`
on representative B2AI data and update these values in `qa_pipeline_config.json`
(see Decision 9 for the calibration workflow).

**Alternatives considered**:
- AND logic: Lower recall; would miss cases where only one embedding detects the
  intruder
- Score fusion: Requires labelled training data; deferred

---

## Decision 5: Non-speech recording handling

**Decision**: Gate on active speech fraction computed from existing diarization
timestamps; apply confidence ceiling for low-speech recordings to both embeddings.

**Rationale**:
- ECAPA-TDNN and SPARC both degrade on low-speech recordings
- `< 15% active speech fraction` → confidence capped at 0.30, classify as
  needs_review regardless of cosine scores (applied to both embeddings)
- `< 1 s active speech (absolute)` → skip cosine comparison during verification;
  return needs_review with confidence 0.10. Also exclude from enrollment entirely.
- `1–3 s` → down-weight in enrollment (w × 0.3)
- `is_speech_task=False` → apply lower confidence ceiling as additional prior

**Config keys** (all in `PipelineConfig.speaker_profile`):

| Key | Default | Source |
|-----|---------|--------|
| `low_confidence_speech_fraction` | 0.15 | provisional — see tuning note |
| `confidence_low_speech_fraction_cap` | 0.30 | provisional — see tuning note |
| `short_recording_skip_s` | 1.0 | literature-anchored (see Decision 6) |
| `confidence_very_short_recording` | 0.10 | provisional — see tuning note |
| `short_enrollment_weight_multiplier` | 0.3 | provisional — see tuning note |
| `weight_normalization_s` | 10.0 | literature-anchored (see Decision 6) |

**Provisional values**: `low_confidence_speech_fraction`, `confidence_low_speech_fraction_cap`,
`confidence_very_short_recording`, and `short_enrollment_weight_multiplier` are
chosen heuristically; they should be calibrated against representative B2AI data
using `embedding-reliability-report` (see Decision 9).

---

## Decision 6: Minimum speech duration thresholds (from literature)

| Active speech duration | Embedding reliability | Recommended treatment |
|-----------------------|----------------------|----------------------|
| < 1 s | Very low | Exclude from enrollment; skip cosine check, confidence 0.10 |
| 1–3 s | Low | Down-weight in enrollment (w × 0.3); score with confidence cap 0.30 |
| 3–10 s | Moderate | Include in enrollment at full weight; confidence 0.70 |
| > 10 s | High | Full enrollment weight; confidence 0.90 |

Source: Interspeech research on short-segment ECAPA-TDNN; quality measure
frameworks (ScienceDirect doi:10.1016/j.dsp.2018.07.012).

**Literature-anchored values**: The 1 s absolute floor (`short_recording_skip_s`)
and 10 s normalisation point (`weight_normalization_s`) are consistent with
published short-utterance reliability findings. The 1–3 s region weight multiplier
(0.3) and fraction-based confidence cap (0.30 at < 15% speech) are heuristic
extrapolations; run `embedding-reliability-report` on representative B2AI data to
validate and update these provisional values.

---

## Decision 7: Ground truth evaluation via synthetic mixtures

**Decision**: Compute operating characteristics (FN rate vs. review-queue fraction)
using synthetic mixtures of real enrolled participants from the B2AI dataset.

**Rationale**:
- The B2AI dataset has no manual annotations confirming presence/absence of
  unconsented speakers
- Diarization output is unsuitable as ground truth (it is one of the signals the
  check itself uses)
- Synthetic mixtures: take known-clean solo recordings from participant A, mix in
  audio from a different enrolled participant B at controlled duration ratio
  (e.g., intruder speaks 10%, 20%, 40% of the recording) and SNR
- Mixture label = positive (unconsented speaker present); unmixed = negative
- Allows full ROC / operating characteristic curves per embedding and for OR
  combination
- Uses real B2AI acoustic characteristics; no external speaker data required

**Implementation**: Mixing is an offline preprocessing step run before evaluation,
not part of the real-time QA pipeline. `qa-run --eval-mode` or the
`embedding-reliability-report` command handles it.

---

## Decision 8: Task name exclusion patterns

**Decision**: Use **case-insensitive prefix matching** (`task_name.lower().startswith(pattern.lower())`)
to exclude task types from enrollment. NOT substring matching (to avoid false matches
when one task name is a prefix-substring of another).

**Correct exclusion prefixes** (from actual B2AI dataset task names):

| Dataset | Exclude (prefix) | Keep (examples) |
|---------|-----------------|-----------------|
| Adult | `Diadochokinesis`, `Prolonged-vowel`, `Maximum-phonation-time`, `Respiration-and-cough`, `Glides`, `Loudness` | `Cape-V-sentences`, `Caterpillar-Passage`, `Free-speech`, `Rainbow-Passage`, `Open-response-questions`, `Picture-description`, `Story-recall`, `Animal-fluency` |
| Peds | `long-sounds`, `silly-sounds`, `repeat-words` | `passage`, `picture`, `favorite`, `naming`, `outside-of-school`, `ready-for-school`, `sentence` |

**Rationale**:
- Excluded tasks are either motor-speech (DDK, sustained phonation, glides),
  respiratory (breathing, cough), non-speech sounds, or very short isolated words —
  all produce unreliable speaker embeddings
- Prefix matching prevents e.g., a pattern `passage` from matching `Caterpillar-Passage`
- Patterns are configurable via `PipelineConfig.speaker_profile.excluded_task_prefixes`

---

## Decision 9: Embedding reliability report — evaluation hyper-parameters

**Decision**: Two hyper-parameters govern how `embedding-reliability-report`
selects the recommended threshold from operating-characteristic curves:

| Parameter | Config key | Default | Meaning |
|-----------|-----------|---------|---------|
| FNR ceiling | `--max-fnr-target` (CLI) | 0.05 | Maximum acceptable false-negative rate; the lowest threshold that stays at or below this FNR is the primary candidate |
| Knee-point sensitivity | `knee_drop_pp` (internal) | 0.15 | Minimum fractional drop in review-queue fraction (as a proportion of the maximum) required to declare a knee point in the FNR-vs-queue curve |

Both values are exposed as named module-level constants in
`src/b2aiprep/prepare/embedding_reliability.py`:

```python
_DEFAULT_MAX_FNR_TARGET  = 0.05   # 5 % FNR ceiling
_DEFAULT_KNEE_DROP_PP    = 0.15   # 15 % relative queue-fraction drop for knee
```

**Rationale for 0.05 FNR target**: In data-privacy QA contexts, missing an
unconsented speaker is substantially worse than unnecessary human review. A 5%
FNR cap means at most 1 in 20 recordings containing an intruder pass unreviewed.
This is an initial operational target with no empirical calibration on B2AI data
yet; it should be revisited once the team establishes acceptable review capacity.

**Rationale for 0.15 knee sensitivity**: A 15% relative queue-fraction drop
provides a meaningful inflection point without being overly sensitive to noise in
sparse operating-characteristic curves. Chosen empirically; adjust down if the
report consistently reports no knee, or up if it finds knee points too aggressively.

**Provisional status**: Both defaults are un-validated on B2AI. To calibrate:
1. Run `embedding-reliability-report` on a representative cohort (≥ 50 participants
   with ≥ 3 recordings each)
2. Inspect the generated Markdown report; the recommended ECAPA-TDNN and SPARC
   thresholds appear under `recommended_ecapa_threshold` / `recommended_sparc_threshold`
3. If the FNR at the recommended threshold substantially exceeds the team's
   target, lower `--max-fnr-target`; if the review queue is too large, raise it
4. If no knee is detected, lower `knee_drop_pp`; if the knee is found too early,
   raise it
5. Update `ecapa_cosine_threshold` and `sparc_cosine_threshold` in
   `qa_pipeline_config.json` (and in `speaker_profile` section of
   `PipelineConfig`) with the empirically calibrated values

---

## Key References

- SPARC codec: senselab `SparcFeatureExtractor`; `spk_emb` is 64-dim, already in `.pt` files
- Cosine vs PLDA for large-margin embeddings: arXiv:2204.03965
- TAS-norm: arXiv:2504.04512
- AS-norm PyTorch: github.com/nidwbin/AS-Norm
- Outlier-rejection enrollment: SCITEPRESS 2025 doi:10.5220/0013256800003890
- Short utterance quality: doi:10.1016/j.dsp.2018.07.012
- VAD-gated embeddings for diarization: arXiv:2405.09142
- ECAPA-TDNN on children (TSD 2024): doi:10.1007/978-3-031-70566-3_6
- CAM++: arXiv:2303.00332
- WavLM-base-plus-sv: huggingface.co/microsoft/wavlm-base-plus-sv
