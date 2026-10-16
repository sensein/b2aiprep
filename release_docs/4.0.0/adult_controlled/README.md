# Bridge2AI-Voice: An ethically-sourced, diverse voice dataset linked to health information - Raw Audio Adult Dataset

Version 4.0.0

This dataset contains voice recordings collected from adult participants as a part of the Bridge2AI Voice as a Biomarker of Health project, together with the participants' demographic, clinical and questionnaire data. It contains 43,007 recordings (156.8 hours of audio) from 1,304 participants at five sites in the United States and Canada.

This is the controlled-access audio release. Access and use are governed by the Bridge2AI-Voice Data Transfer and Use Agreement (see `LICENSE`). A feature-only release, with derived audio features but not the recordings, is available on PhysioNet under a separate agreement. Participant identifiers are the same in both releases and across versions of this dataset.

To cite this dataset, use the DOI of version 4.0.0 in the "B2AI-Voice Adult Dataset (Versioned)" table on Synapse (https://www.synapse.org/Synapse:syn74341293); each released version has its own DOI there.

`data_explorer.html`, at the top level, summarizes every phenotype field: its description, how often it is blank, and the distribution of its values. Open it in any web browser.

## Directory structure

The layout is BIDS-like. The dataset is in the process of conforming to BEP 047, the proposed Brain Imaging Data Structure (BIDS) extension for audio and video recordings (https://github.com/bids-standard/bids-specification/pull/2231). In the meantime, files and data structures are organized in a BIDS-like way: they follow BIDS naming conventions (`sub-`, `ses-`, `task-`, per-file JSON sidecars, tab-separated tables with JSON data dictionaries) but do not yet conform to the standard.

```
.
├── README.md
├── CHANGES
├── LICENSE
├── dataset_description.json
├── data_explorer.html
├── audio_quality_metrics.json
├── audio_quality_metrics.tsv
├── audio_quality_annotations.json
├── audio_quality_annotations.tsv
├── consensus_transcripts.json
├── consensus_transcripts.parquet
├── phenotype/
│   └── <group>/<table>.tsv and <table>.json
└── sub-<participant_id>/
    ├── sub-<participant_id>_sessions.tsv
    └── ses-<session_id>/
        └── audio/
            ├── sub-<participant_id>_ses-<session_id>_task-<task_name>.flac
            └── sub-<participant_id>_ses-<session_id>_task-<task_name>.json
```

## Identifiers

- `participant_id` is a six-digit pseudonym. Folder names add the prefix `sub-` (`sub-<participant_id>`).
- `session_id` is the session number within the participant (`01`, `02`, ...), so a session is identified by the participant and session together. Folder names add the prefix (`ses-<session_id>`).
- `task_name` names the task, for example `rainbow-passage`, `harvard-sentences-list-31-5` or `glides-high-to-low`. A trailing number distinguishes the items or repeats of a task.
- `recording_id` identifies one recording. It links the audio sidecar, `phenotype/task/recording.tsv`, `audio_quality_annotations.tsv` and `consensus_transcripts.parquet`.

Identifiers contain leading zeros, so read them as text.

## Audio

Each recording is a FLAC file (lossless compression), 16-bit PCM, 16 kHz, mono. The audio was converted to mono and resampled to 16 kHz with senselab. Where resampling pushed the waveform past full scale, the recording was scaled down to its original peak instead of being clipped. Recordings of the free-speech tasks (`free-speech`, `free-speech-1` to `free-speech-3`, `free-speech-v2-1` to `free-speech-v2-3`) and of the open-response questions (`open-response-questions`) need further review and are not provided at this time.

Next to each FLAC file is a JSON sidecar with the same name. It holds:

| Field | Meaning |
|---|---|
| `participant_id`, `session_id`, `task_name` | identifiers as above |
| `recording_id`, `recording_acoustic_task_id` | identifiers of the recording and of the task administration it belongs to |
| `language` | language of the task as presented (`en` or `es-419`) |
| `instructions` | instructions shown to the participant |
| `speech_type` | kind of speech the task asks for (for example `read`, `elicited`, `non-lexical`) |
| `stimulus_text`, `stimulus_source`, `stimulus_asset`, `stimulus_asset_n_images` | the stimulus, where the task has one: its text, where it comes from, and a link to the image shown |
| `recording_duration` | length reported by the collection app, in seconds. It can differ from the file; `recording_duration_s` in `audio_quality_annotations.tsv` is the measured length |
| `recording_size` | size of the original recording as reported by the collection platform, before conversion. The unit depends on how the recording was made: megabytes for recordings made in the app, bytes for web recordings (`recording_microphone` starting with `Web Audio`). It does not match the released FLAC file, nor the WAV files of earlier versions |
| `recording_input_gain`, `recording_microphone`, `recording_profile_name`, `recording_profile_version` | recording settings |
| `audio_sample_rate`, `audio_channel_count` | audio format |

`stimulus_source`, `stimulus_asset` and `stimulus_asset_n_images` appear only for tasks that have them; the other fields are always present and can be empty. The same fields are columns of `phenotype/task/recording.tsv`, with one row per recording.

Each participant folder also has `sub-<participant_id>_sessions.tsv`: one row per session with `session_index`, `session_local_hour`, `session_seconds_since_first_session`, `session_status` and `session_duration`. It lists every session with released data, including sessions that have phenotype data but no released recordings and so no `ses-` folder. Its columns are described in `phenotype/task/session.json`.

## phenotype

The phenotype directory holds the participants' enrollment and demographic information, confounders, clinician-entered diagnosis forms, validated questionnaires, and per-session and per-recording information. Tables are grouped into `confounders`, `demographics`, `diagnosis`, `enrollment`, `questionnaire` and `task`.

- Each `.tsv` file is one table, keyed by `participant_id` and, where the data belongs to a session, a session column (`session_id`, or a column ending in `_session_id` such as `demographics_session_id`). Diagnosis and enrollment tables have no session column. A table only contains a row for a participant if at least one of its columns is not missing, and a participant can have more than one row when data were collected in more than one session.
- Each `.json` file describes the matching `.tsv` file. Under the table's name it gives the table's `description` and, in `data_elements`, an entry for each column with `description` (what the column holds), `question` (the question as worded on the form, where there is one), `choices` (the response options) and the value type.
- Checkbox columns (`<question>___<option>`, one per option of a select-all-that-apply question) are `1` when that option was selected and blank otherwise. The option is named in the column's `description` and `choices`.
- A `description` starting with `[DEPRECATED]` is a question no longer asked; its earlier answers are kept. `[RETIRED FORM]` marks a column of a questionnaire that is no longer collected.
- Free-text answers were released only after review, with identifying details replaced by `[REDACTED]`.
- Some answer options chosen by very few participants were combined.
- No calendar dates or clock times are released. Timing is given as intervals (for example days since an episode, or seconds since the participant's first session) or as the local hour of day.
- `enrollment/participant.tsv` lists every participant, with `site` (a pseudonymous site label), `some_data_collected_remotely` and `selected_language`. This dataset has no `participants.tsv` at the root.
- `enrollment/eligibility.tsv` holds `is_control_participant`, which records whether a participant was enrolled as a control.
- `task/recording.tsv` has one row per recording, with the sidecar fields above, the recording's order and timing within the session, and the device, platform and browser where the collection software recorded them.
- Some fields are released only in this controlled-access dataset, not in the PhysioNet release: 48 columns in `confounders/confounders.tsv`, `diagnosis/amyotrophic_lateral_sclerosis.tsv`, `diagnosis/essential_tremor.tsv`, `diagnosis/parkinsons_disease.tsv`, `enrollment/eligibility.tsv` and `questionnaire/psychiatric_history.tsv` (substance use, medications and psychiatric treatment).

## Audio quality and transcripts

`audio_quality_metrics.tsv` contains audio quality measures for each recording (clipping, silence, signal-to-noise ratio and amplitude modulation depth). Each `*_check` column is `True` when the recording was flagged by that check. The file has no `recording_id`; it joins to the other tables on `participant_id`, `session_id` and `task_name`.

`audio_quality_annotations.tsv` contains the measurements of an automated review of each recording: the estimated task extent (`task_start_s` to `task_end_s`), level, noise, signal-to-noise and speech-quality estimates, clipping and dropouts, speech-recognition agreement and speaker-consistency scores. The task extent is the part of the recording the review estimated to hold the requested task, 132.4 of the 156.8 hours in all. It is provided as an aid for analysis, for example to trim silence and other sound outside the task, and may leave out part of the task where the alignment is wrong.

`consensus_transcripts.parquet` contains a consensus of two speech recognizers' transcripts of each recording, with one row per word: each word's text, start and end time, whether it is a non-speech tag such as `[COUGH]`, how the two recognizers aligned, and each recognizer's own word. The rows of each recording are consecutive and in consensus order (`word_index`); in about 2% of recordings this order does not always follow the word start times, so sort by `start_s` where time order matters. Recordings in which no words were recognized have no rows.

Each of these files has a `.json` file of the same name that describes its columns.

## Usage

The audio can be read with any library that supports FLAC. For example, in Python with soundfile:

```python
import soundfile as sf
audio, sample_rate = sf.read(
    "sub-<participant_id>/ses-01/audio/sub-<participant_id>_ses-01_task-rainbow-passage.flac"
)
```

The phenotype tables can be loaded with any statistical analysis tool. For example, with pandas:

```python
import pandas as pd
recordings = pd.read_csv(
    "phenotype/task/recording.tsv",
    sep="\t",
    dtype={"participant_id": str, "recording_session_id": str},
)
```

To keep only the transcript words within each recording's task:

```python
import pandas as pd
annotations = pd.read_csv(
    "audio_quality_annotations.tsv", sep="\t", dtype={"participant_id": str, "session_id": str}
)
words = pd.read_parquet("consensus_transcripts.parquet")
words = words.merge(annotations[["recording_id", "task_start_s", "task_end_s"]], on="recording_id")
in_task = words[(words["start_s"] >= words["task_start_s"]) & (words["end_s"] <= words["task_end_s"])]
```
