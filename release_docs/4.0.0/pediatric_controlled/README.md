# Bridge2AI-Voice Pediatric Dataset - Raw Audio Pediatric Dataset

Version 2.0.0

This dataset contains voice recordings collected from pediatric participants (aged 2-17) as a part of the Bridge2AI Voice as a Biomarker of Health project, together with the participants' demographic, medical-history and questionnaire data. It contains 41,831 recordings (59.5 hours of audio) from 656 participants at a single site. Questionnaires were answered by the participant or by a parent or guardian.

This is the controlled-access audio release. Access and use are governed by the Bridge2AI-Voice Data Transfer and Use Agreement (see `LICENSE`). The pediatric dataset may only be used for non-commercial research. A feature-only release, with derived audio features but not the recordings, is available on PhysioNet under a separate agreement. Participant identifiers are the same in both releases and across versions of this dataset.

To cite this dataset, use the DOI of version 2.0.0 in the "B2AI-Voice Pediatric Dataset (Versioned)" table on Synapse (https://www.synapse.org/Synapse:syn74341287); each released version has its own DOI there.

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
- `session_id` is the session number within the participant (`01`), so a session is identified by the participant and session together. Folder names add the prefix (`ses-<session_id>`). Every participant has one session.
- `task_name` names the task item, for example `identifying-pictures-12`, `reading-passage-3` or `repeating-words-pond`. Each task name identifies the same item for every participant.
- `recording_id` identifies one recording. It links the audio sidecar and `phenotype/task/recording.tsv`.

Identifiers contain leading zeros, so read them as text.

## Audio

Each recording is a FLAC file (lossless compression), 16-bit PCM, 16 kHz, mono. The audio was converted to mono and resampled to 16 kHz with senselab. Where resampling pushed the waveform past full scale, the recording was scaled down to its original peak instead of being clipped. Recordings were manually reviewed for the voices of people other than the participant and for personally identifying information; recordings flagged in this review, and recordings not yet reviewed, are not included.

Next to each FLAC file is a JSON sidecar with the same name. It holds:

| Field | Meaning |
|---|---|
| `participant_id`, `session_id`, `task_name` | identifiers as above |
| `recording_id`, `recording_acoustic_task_id` | identifiers of the recording and of the task administration it belongs to |
| `language` | language of the task as presented (`en`) |
| `instructions` | instructions shown to the participant |
| `speech_type` | kind of speech the task asks for (for example `read`, `elicited`, `non-lexical`) |
| `stimulus_text`, `stimulus_source`, `stimulus_asset` | the stimulus, where the task has one: its text, where it comes from, and a link to the image shown |
| `recording_duration` | length of the recording, in seconds |
| `recording_size` | size in kilobytes of the original recording as exported from the collection platform. It does not match the released FLAC file, nor the WAV files of earlier versions, which were already converted |
| `recording_input_gain`, `recording_microphone`, `recording_profile_name`, `recording_profile_version` | recording settings; empty for this cohort |
| `audio_sample_rate`, `audio_channel_count` | audio format |

`stimulus_source` and `stimulus_asset` appear only for tasks that have them; the other fields are always present and can be empty. The same fields are columns of `phenotype/task/recording.tsv`, with one row per recording.

Each participant folder also has `sub-<participant_id>_sessions.tsv`: one row per session with `session_index`, `session_local_hour`, `session_seconds_since_first_session`, `session_status` and `session_duration`. Its columns are described in `phenotype/task/session.json`.

## phenotype

The phenotype directory holds enrollment information, demographics, medical history, validated questionnaires, and per-session and per-recording information. Tables are grouped into `enrollment`, `pediatric`, `pediatric_questionnaire` and `task`.

- Each `.tsv` file is one table, keyed by `participant_id` and, where the data belongs to a session, a session column (`session_id`, or a column ending in `_session_id` such as `peds_demographics_session_id`). Enrollment tables have no session column.
- Each `.json` file describes the matching `.tsv` file. Under the table's name it gives the table's `description` and, in `data_elements`, an entry for each column with `description` (what the column holds), `question` (the question as worded on the form, where there is one), `choices` (the response options) and the value type.
- Checkbox columns (`<question>___<option>`, one per option of a select-all-that-apply question) are `1` when that option was selected and blank otherwise. The option is named in the column's `description` and `choices`.
- No calendar dates or clock times are released. Timing is given as intervals (for example days from a surgery or procedure to the participant's first session) or as the local hour of day. Free-text responses are not released.
- `enrollment/participant.tsv` lists every participant, with `site` (a pseudonymous site label), `some_data_collected_remotely` and `selected_language`. This dataset has no `participants.tsv` at the root.
- `task/recording.tsv` has one row per recording, with the sidecar fields above and the recording's order and timing within the session. The device and browser columns are present but empty for this cohort.

## Audio quality

`audio_quality_metrics.tsv` contains audio quality measures for each recording (clipping, silence, signal-to-noise ratio and amplitude modulation depth). Each `*_check` column is `True` when the recording was flagged by that check. The file has no `recording_id`; it joins to the other tables on `participant_id`, `session_id` and `task_name`. `audio_quality_metrics.json` describes its columns.

## Usage

The audio can be read with any library that supports FLAC. For example, in Python with soundfile:

```python
import soundfile as sf
audio, sample_rate = sf.read(
    "sub-<participant_id>/ses-01/audio/sub-<participant_id>_ses-01_task-reading-passage-3.flac"
)
```

The phenotype tables can be loaded with any statistical analysis tool. For example, with pandas:

```python
import pandas as pd
demographics = pd.read_csv(
    "phenotype/pediatric/pediatric_demographics.tsv",
    sep="\t",
    dtype={"participant_id": str, "peds_demographics_session_id": str},
)
```
