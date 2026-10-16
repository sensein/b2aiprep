# Bridge2AI-Voice: An ethically-sourced, diverse voice dataset linked to health information

Version 4.0.0

This dataset contains information collected from adult participants as a part of the Bridge2AI Voice as a Biomarker of Health project. It contains features derived from 48,226 voice recordings (189.7 hours of audio) from 1,370 participants at five sites in the United States and Canada, together with the participants' demographic, clinical and questionnaire data. It does not contain the recordings themselves.

`data_explorer.html`, at the top level, summarizes every phenotype field: its description, how often it is blank, and the distribution of its values. Open it in any web browser.

## phenotype

The phenotype directory stores participant-specific data that is not directly related to specific audio tasks but is relevant for understanding participants' characteristics: enrollment and demographic information, confounders, clinician-entered diagnosis forms, validated questionnaires, and per-session and per-recording information.

The folder contains the following files:

```
phenotype
├── confounders
│   ├── confounders.json
│   └── confounders.tsv
├── demographics
│   ├── demographics.json
│   └── demographics.tsv
├── diagnosis
│   ├── airway_stenosis.json
│   ├── airway_stenosis.tsv
│   ├── amyotrophic_lateral_sclerosis.json
│   ├── amyotrophic_lateral_sclerosis.tsv
│   ├── anxiety.json
│   ├── anxiety.tsv
│   ├── ataxia.json
│   ├── ataxia.tsv
│   ├── benign_lesions.json
│   ├── benign_lesions.tsv
│   ├── bipolar_disorder.json
│   ├── bipolar_disorder.tsv
│   ├── cognitive_impairment.json
│   ├── cognitive_impairment.tsv
│   ├── control.json
│   ├── control.tsv
│   ├── copd_and_asthma.json
│   ├── copd_and_asthma.tsv
│   ├── depression.json
│   ├── depression.tsv
│   ├── essential_tremor.json
│   ├── essential_tremor.tsv
│   ├── glottic_insufficiency.json
│   ├── glottic_insufficiency.tsv
│   ├── huntingtons_disease.json
│   ├── huntingtons_disease.tsv
│   ├── laryngeal_cancer.json
│   ├── laryngeal_cancer.tsv
│   ├── laryngeal_dystonia.json
│   ├── laryngeal_dystonia.tsv
│   ├── laryngitis.json
│   ├── laryngitis.tsv
│   ├── muscle_tension_dysphonia.json
│   ├── muscle_tension_dysphonia.tsv
│   ├── parkinsons_disease.json
│   ├── parkinsons_disease.tsv
│   ├── precancerous_lesions.json
│   ├── precancerous_lesions.tsv
│   ├── unexplained_chronic_cough.json
│   ├── unexplained_chronic_cough.tsv
│   ├── unilateral_vocal_fold_paralysis.json
│   └── unilateral_vocal_fold_paralysis.tsv
├── enrollment
│   ├── eligibility.json
│   ├── eligibility.tsv
│   ├── enrollment_form.json
│   ├── enrollment_form.tsv
│   ├── participant.json
│   └── participant.tsv
├── questionnaire
│   ├── adhd_adult.json
│   ├── adhd_adult.tsv
│   ├── custom_affect_scale.json
│   ├── custom_affect_scale.tsv
│   ├── dsm5_adult.json
│   ├── dsm5_adult.tsv
│   ├── dyspnea_index.json
│   ├── dyspnea_index.tsv
│   ├── gad7_anxiety.json
│   ├── gad7_anxiety.tsv
│   ├── leicester_cough_questionnaire.json
│   ├── leicester_cough_questionnaire.tsv
│   ├── panas.json
│   ├── panas.tsv
│   ├── phq9.json
│   ├── phq9.tsv
│   ├── psychiatric_history.json
│   ├── psychiatric_history.tsv
│   ├── ptsd_adult.json
│   ├── ptsd_adult.tsv
│   ├── vhi10.json
│   ├── vhi10.tsv
│   ├── voice_perception.json
│   └── voice_perception.tsv
└── task
    ├── acoustic_task.json
    ├── acoustic_task.tsv
    ├── harvard_sentences.json
    ├── harvard_sentences.tsv
    ├── productive_vocabulary.json
    ├── productive_vocabulary.tsv
    ├── random_item_generation.json
    ├── random_item_generation.tsv
    ├── recording.json
    ├── recording.tsv
    ├── session.json
    ├── session.tsv
    ├── stroop.json
    ├── stroop.tsv
    ├── winograd.json
    └── winograd.tsv
```

Each `.tsv` file is one table, keyed by `participant_id` and, where the data belongs to a session, a session column (`session_id`, or a column ending in `_session_id` such as `demographics_session_id`). Sessions are numbered within each participant (`01`, `02`, ...), so a session is identified by the participant and session columns together. Diagnosis and enrollment tables have no session column. A table only contains a row for a participant if at least one of its columns is not missing, and a participant can have more than one row when data were collected in more than one session.

Each `.json` file describes the matching `.tsv` file: the table's `description`, then for each column `description` (what the column holds), `question` (the question as worded on the form), `choices` (the response options) and the value type.

Checkbox columns (`<question>___<option>`, one per option of a select-all-that-apply question) are `1` when that option was selected and blank otherwise. The option is named in the column's `description` and `choices`.

Free-text answers were released only after review, with identifying details replaced by `[REDACTED]`.

`task/recording.tsv` has one row per recording, with the task, language, instructions, stimulus and audio format. It joins to the feature files on participant, session (`recording_session_id`) and `task_name`, and to `audio_quality_annotations.tsv` and `consensus_transcripts.parquet` on `recording_id`.

## features

The features folder contains derivations from the raw audio. The files contained in the features folder are:

```
features
├── audio_quality_annotations.json
├── audio_quality_annotations.tsv
├── audio_quality_metrics.json
├── audio_quality_metrics.tsv
├── consensus_transcripts.json
├── consensus_transcripts.parquet
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

The feature parquet files can be loaded in using any library that supports Apache Parquet. Each row holds `participant_id`, `session_id`, `task_name`, `n_frames` and the feature tensor for one recording. The JSON file with the same name describes the content and the processing done to create the feature: tensor shape, units, frame rate, extraction parameters and software version. The spectrogram, Mel spectrogram and MFCC keep every second frame (20 ms between frames).

The dataset covers 48,226 recordings, totalling 189.7 hours, of which an estimated 161.6 hours fall within the task extent (see audio_quality_annotations.tsv below). Pitch, loudness, periodicity, the static features and the audio-quality files cover all of them. Content features (spectrograms, Mel spectrograms, MFCCs, PPGs, EMA and transcripts) are not provided at this time for free-speech and open-response recordings, which need further review; the other recordings total 164.2 hours (138.7 hours within the task extent). Additionally, some files, whether due to length or other issues, could not generate certain features and so are not included in those feature files. As a result, the number of rows differs between files.

The static_features.tsv file contains single scalar values derived from the source audio records in tab delimited format, and an automatic transcript of each recording (`transcription`, empty for free-speech and open-response recordings).

The audio_quality_metrics.tsv file contains audio quality measures for each recording (clipping, silence, signal-to-noise ratio and amplitude modulation depth). Each `*_check` column is `True` when the recording was flagged by that check. The file has no `recording_id`; it joins to the other tables on `participant_id`, `session_id` and `task_name`.

The audio_quality_annotations.tsv file contains the measurements of an automated review of each recording: the estimated task extent (`task_start_s` to `task_end_s`), level, noise, signal-to-noise and speech-quality estimates, clipping and dropouts, speech-recognition agreement and speaker-consistency scores. The task extent is the part of the recording the review estimated to hold the requested task. It is provided as an aid for analysis, for example to trim silence and other sound outside the task, and may leave out part of the task where the alignment is wrong.

The consensus_transcripts.parquet file contains a consensus of two speech recognizers' transcripts of each recording with content features, with one row per word: each word's text, start and end time, whether it is a non-speech tag such as `[COUGH]`, how the two recognizers aligned, and each recognizer's own word. The rows of each recording are consecutive and in consensus order (`word_index`); in about 2% of recordings this order does not always follow the word start times, so sort by `start_s` where time order matters. Recordings in which no words were recognized have no rows.

## Usage

If using Python, the parquet dataset can be loaded in with any library that supports parquet. For example, the HuggingFace Datasets library can be used to load in the spectrograms (this writes a cached copy of the file, about 12 GB for the spectrogram):

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
    "phenotype/demographics/demographics.tsv",
    sep="\t",
    dtype={"participant_id": str, "demographics_session_id": str},
)
```

To keep only the transcript words within each recording's task:

```python
import pandas as pd
annotations = pd.read_csv(
    "features/audio_quality_annotations.tsv", sep="\t", dtype={"participant_id": str, "session_id": str}
)
words = pd.read_parquet("features/consensus_transcripts.parquet")
words = words.merge(annotations[["recording_id", "task_start_s", "task_end_s"]], on="recording_id")
in_task = words[(words["start_s"] >= words["task_start_s"]) & (words["end_s"] <= words["task_end_s"])]
```
