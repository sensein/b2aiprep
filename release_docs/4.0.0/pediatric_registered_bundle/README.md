# Bridge2AI Voice - Pediatric

This dataset contains information collected from pediatric participants (aged 2-17) as a part of the Bridge2AI voice as a biomarker of health project. It contains features derived from the voice recordings and the participants' questionnaire data, but not the recordings themselves.

`data_explorer.html`, at the top level, summarizes every phenotype field: its description, how often it is blank, and the distribution of its values. Open it in any web browser.

## phenotype

The phenotype directory stores participant-specific data that is not directly related to specific audio tasks but is relevant for understanding participants' characteristics. Primarily it contains responses to questionnaires by the participant or a parent or guardian.

The folder contains the following files:

```
phenotype
├── enrollment
│   ├── eligibility.json
│   ├── eligibility.tsv
│   ├── enrollment_form.json
│   ├── enrollment_form.tsv
│   ├── participant.json
│   └── participant.tsv
├── pediatric
│   ├── pediatric_demographics.json
│   ├── pediatric_demographics.tsv
│   ├── pediatric_medical_conditions.json
│   └── pediatric_medical_conditions.tsv
├── pediatric_questionnaire
│   ├── pediatric_hqa.json
│   ├── pediatric_hqa.tsv
│   ├── pediatric_vhi10.json
│   ├── pediatric_vhi10.tsv
│   ├── pediatric_voice_outcome_survey.json
│   ├── pediatric_voice_outcome_survey.tsv
│   ├── pediatric_voice_related_qol_survey.json
│   └── pediatric_voice_related_qol_survey.tsv
└── task
    ├── acoustic_task.json
    ├── acoustic_task.tsv
    ├── recording.json
    ├── recording.tsv
    ├── session.json
    └── session.tsv
```

Each `.tsv` file is one table, keyed by `participant_id` and, where the data belongs to a session, a session column (`session_id`, or a column ending in `_session_id` such as `peds_demographics_session_id`). Sessions are numbered within each participant (`01`, `02`, ...), so a session is identified by the participant and session columns together.

Each `.json` file describes the columns of the matching `.tsv` file: `description` says what the column holds, `question` is the question as worded on the form, and `choices` lists the response options.

Checkbox columns (`<question>___<option>`, one per option of a select-all-that-apply question) are `1` when that option was selected and blank otherwise. The option is named in the column's `description` and `choices`.

`task/recording.tsv` has one row per recording, with the task, instructions, stimulus and audio format. It joins to the feature files on participant, session (`recording_session_id`) and `task_name`.

`enrollment/participant.tsv` lists every participant in this release, one row each. The other tables only hold rows for these participants, and only when the participant has data on that form. A participant can therefore have no row in a table: `pediatric/pediatric_medical_conditions.tsv` has no row for 3 participants and `pediatric_questionnaire/pediatric_vhi10.tsv` none for 1, because nothing was recorded on those forms for them. `task/acoustic_task.tsv` and `task/recording.tsv` have one row per task and per recording. The other tables with a session column have one row per session in which the form was filled in. Every participant in this version has one session, so they have at most one row per participant; later versions may have more.

`enrollment/eligibility.tsv` comes from the study-eligibility form: whether the participant was enrolled as a control, and the answers to "Do you have any of these conditions, diagnosed by a clinician?". `enrollment/enrollment_form.tsv` comes from the enrollment survey: language, accessibility needs and how long the survey took.

The enrollment, eligibility, session and recording forms are shared with the adult study, so these tables have the same columns in both datasets. Some columns have no values in this version:

- Recorded only by the study app. All data in this version were collected before the app was used, so these columns are empty:
  - `enrollment/eligibility.tsv`: `self_reported_autism`, `self_reported_speech_delay`. Autism is recorded in `peds_mc_psych_disorders___asd`, and voice disorder or speech delay in `peds_mc_v_dis`, both in `pediatric/pediatric_medical_conditions.tsv`.
  - `enrollment/enrollment_form.tsv`: `ef_duration`, `ef_any_cognitive_challeges`, `ef_any_hearing_related_req`, `ef_any_physical_challenges`, `ef_any_vision_related_req`.
  - `task/session.tsv`: `session_assigned_tasks`.
  - `task/recording.tsv`: `recording_input_gain`, `recording_microphone`, `recording_profile_name`, `recording_profile_version`, `recording_device_info`, `recording_device_type`, `recording_platform`, `recording_platform_version`, `recording_screen_resolution`, `recording_window_resolution`.
- Recorded only for adult participants: `ef_fluent_languages___1` to `ef_fluent_languages___5` in `enrollment/enrollment_form.tsv`, and `recording_browser` in `task/recording.tsv`.
- `stimulus_asset_n_images` in `task/recording.tsv` applies only to stimuli made of several images, and no pediatric task uses one.

## features

The features folder contains derivations from the raw audio. The files contained in the features folder are:

```
features
├── audio_quality_metrics.json
├── audio_quality_metrics.tsv
├── ppgs.json
├── ppgs.parquet
├── sparc_ema.json
├── sparc_ema.parquet
├── sparc_loudness.json
├── sparc_loudness.parquet
├── sparc_periodicity.json
├── sparc_periodicity.parquet
├── sparc_pitch.json
├── sparc_pitch.parquet
├── static_features.json
├── static_features.tsv
├── torchaudio_mel_spectrogram.json
├── torchaudio_mel_spectrogram.parquet
├── torchaudio_mfcc.json
├── torchaudio_mfcc.parquet
├── torchaudio_pitch.json
├── torchaudio_pitch.parquet
├── torchaudio_spectrogram.json
└── torchaudio_spectrogram.parquet
```

Parquet files can be loaded in using any library that supports Apache Parquet. Each row holds `participant_id`, `session_id`, `task_name`, `n_frames` and the feature tensor. The JSON file with the same name describes the content and the processing done to create the feature: tensor shape, units, frame rate, extraction parameters and software version. The spectrogram, Mel spectrogram and MFCC keep every second frame (20 ms between frames).

The static_features.tsv file contains single scalar values derived from the source audio records in tab delimited format, and an automatic transcript of each recording (`transcription`).

The audio_quality_metrics.tsv file contains audio quality measures for each recording (clipping, silence, signal-to-noise ratio and amplitude modulation depth), each with a pass/fail check.

## Usage

If using Python, the parquet dataset can be loaded in with any library that supports parquet. For example, the HuggingFace Datasets library can be used to load in the spectrograms:

```python
from datasets import Dataset
ds = Dataset.from_parquet("features/torchaudio_spectrogram.parquet")
```

The spectrogram is stored in decibels. The Mel spectrogram is stored as power and can be plotted in decibels after conversion:

```python
from datasets import Dataset
import numpy as np
import librosa
import matplotlib.pyplot as plt

ds = Dataset.from_parquet("features/torchaudio_mel_spectrogram.parquet")
mel_db = librosa.power_to_db(np.asarray(ds[0]["mel_spectrogram"]))
plt.figure(figsize=(10, 4))
plt.imshow(mel_db, aspect="auto", origin="lower")
plt.title("Mel spectrogram")
plt.xlabel("Frame (20 ms)")
plt.ylabel("Mel bin")
plt.colorbar(label="dB")
plt.show()
```

The phenotype file can be loaded with any statistical analysis tool. Participant and session identifiers contain leading zeros, so read them as text. For example, the pandas library in Python can read the data:

```python
import pandas as pd
df = pd.read_csv(
    "phenotype/pediatric/pediatric_demographics.tsv",
    sep="\t",
    dtype={"participant_id": str, "peds_demographics_session_id": str},
)
```
