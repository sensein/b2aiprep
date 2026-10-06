# Release Process

This document provides instructions for preparing and releasing the Bridge2AI Voice dataset for public and internal releases.

Required inputs:

- A RedCap dissemination export (CSV), one per cohort (adult, pediatric) and access tier. Both
  cohorts go through the same steps: the pediatric ReproSchema data collected before the app has
  been converted and merged into the pediatric RedCap project.
- A directory with audio recordings in .wav format

Note that there are implicit assumptions made regarding the structured data and the .wav files, e.g. it is assumed the .wav files have a UUID in their filename which is referenced in the structured data.

## Quick start

```sh
WORKING_DIR=${HOME}/data/bridge2ai/adult   # or .../pediatric: same commands

b2aiprep-cli redcap2bids $WORKING_DIR/redcap.csv --outdir $WORKING_DIR/bids --audiodir $WORKING_DIR/audio --sanitize_audio_format --date-shift-anchor $DATE_SHIFT_ANCHOR
b2aiprep-cli generate-audio-features $WORKING_DIR/bids $WORKING_DIR/bids --update
b2aiprep-cli run-quality-control-on-audios $WORKING_DIR/bids
b2aiprep-cli deidentify-bids-dataset $WORKING_DIR/bids $WORKING_DIR/de-identified-bids $WORKING_DIR/release_config
b2aiprep-cli create-bundled-dataset $WORKING_DIR/de-identified-bids $WORKING_DIR/bundled --skip_audio
```

---
### 1. Convert RedCap to BIDS

The next step in the pipeline reorganizes the dataset to follow the [BIDS format](https://bids.neuroimaging.io/index.html). Briefly, BIDS is a folder structure with sessions nested under unique folders for each subject and certain metadata files required.

The following command will parse the RedCap CSV into multiple phenotype CSVs and reorganize the audio data to follow the BIDS folder structure:

```
b2aiprep-cli redcap2bids <path/to/redcap_csv> \
    --outdir <path/to/bids/folder> \
    --audiodir <path/to/audio/files> \
    --sanitize_audio_format \
    --max-audio-workers=8 \
    --date-shift-anchor <YYYY-MM-DD> \
    --date-shift-log <path/outside/the/bids/folder/date_shift.json>
```

#### What reaches the BIDS tree: the field map

`src/b2aiprep/prepare/resources/bids_field_organization.csv` decides what happens to every RedCap
column through its `disposition` column (the older `delete` column is kept as a historical record
and is not read):

| disposition | redcap2bids | deidentify |
|---|---|---|
| `drop` | removed at ingest; never written | — |
| `internal` | written to the BIDS tree (e.g. form timestamps, shifted dates, postal codes) | removed |
| `review` | written | removed, unless a `column_value_reviews.json` manifest in the config folder has checked the values |
| `release` | written | kept |

Deidentify matches dispositions within each phenotype table, so a column name can be `release` in
one table and `internal` in another.

#### Date shifting

`--date-shift-anchor` is required and has no default. Every column marked `date_shift=YES` in the
field map is moved by a whole number of weeks, chosen per participant so their earliest session lands
within three days of the anchor (day of week kept). Shifted timestamps keep local wall-clock time
and the real UTC offset, so time of day and intervals stay exact. Each session is localized where it
happened: the participant's site (`enrollment_institution`) for sessions a data collector ran, or
the participant's postal code / single-zone state or province for self-administered sessions
(`*_via = Participant`). Values that cannot be parsed, and dates of participants with no session
start time, are blanked; real dates never reach the tree. See `src/b2aiprep/prepare/date_shift.py`.

Use the same anchor for both cohorts and for reruns so shifted dates stay comparable. The anchor is
not a secret (it can be read off any shifted table); what protects real dates is each participant's
offset, which is never stored. The run log always records the anchor and the participants left
unshifted; `--date-shift-log` additionally writes them as JSON and must be outside `--outdir`.

#### Metadata-only builds: `--skip-audio-copy`

`--skip-audio-copy` still reads `--audiodir` to decide which recordings exist, and writes exactly the
sidecars, `sessions.tsv` and `recording.tsv` a full build writes, but copies no audio. Use it to
regenerate phenotype tables and participant-level metadata for QA without touching audio (far fewer
files, much faster). Not for a release: the tree has no WAVs, so features and QC cannot run on it.

An optional `--max-audio-workers` controls the number of threads used for writing out audio files as writing of audio files is the speed bottleneck of this command. An optional `--sanitize_audio_format` can be used to sanitize the audio format into 16KHz mono-channel WAVs.

### 2. Feature Extraction

Run the following command to extract features:
```
b2aiprep-cli generate-audio-features \
    <path/to/input/bids_folder> \
    <path/to/output/folder> \
    --is_sequential True
```

Note: Due to the potential size and quantity of the audio files, this may take a while, and we typically run it on a cluster with slurm (sbatch).

Once the command is complete, you should have the following output:
```
/bids_dataset
├── README.md
├── dataset_description.json
├── phenotype/
│   ├── pediatric/
│   │       ├── pediatric_questionnaire.json
│   │       ├── pediatric_questionnaire.tsv
│   │       │
|   |      ...
├── sub-01/
│   ├── session-01/
│   │   └── audio/
│   │       ├── sub-01_session_task-audio.json
│   │       ├── sub-01_session_task-audio.wav
│   │       └── sub-01_session_task-audio.pt
├── sub-02/
│   ├── session-01/
│   │   └── audio/
│   │       ├── sub-02_session_task-audio.json
│   │       ├── sub-02_session_task-audio.wav
│   │       └── sub-02_session_task-audio.pt
│  ...
└── CHANGELOG.md
```

### 3. Quality Control

Run audio quality control checks on the BIDS directory to produce `audio_quality_metrics.tsv` and its companion `audio_quality_metrics.json` at the root of the BIDS directory:

```
b2aiprep-cli run-quality-control-on-audios <path/to/bids/folder>
```

These files summarize per-recording quality metrics (clipping, silence, SNR, amplitude modulation) that can be used to identify recordings that should be excluded before deidentification and release.

### 4. Deidentification

Once the data has been been reformatted into BIDS format and features have been extracted, we need to make sure to remove any entries that could have
sensitive information, referred to as deidentification. Deidentification requires creation of the following configuration files:

- `id_remapping.json` (File containing participant ids to change)
- `audio_tasks_to_include.json` (File containing list of audio tasks to include during deidentification)
- `audio_filestems_to_remove.json` (File containing a list of sensitive audio files to remove)
- `participants_to_include.json` (Optional: explicit allowlist of participant IDs to include. When present, only these participants appear in the output)
- `participants_to_remove.json` (Fallback: used only when `participants_to_include.json` is absent; the allowlist is derived by inverting this list against the input tree)

Create these files and place them in a folder, e.g. `deidentification_config` (the "config" folder).

#### (Optional) Generate or extend `id_remapping.json`

When preparing a new release, the participant list often contains IDs that aren't yet in the existing `id_remapping.json`. To bootstrap the file from scratch — or to add fresh pseudonyms for any new IDs while preserving the mappings already in use — run:

```sh
b2aiprep-cli generate-id-lookup-table <path/to/ids_file> <path/to/output/dir> \
    --load_lookup <path/to/existing/id_remapping.json>
```

The input file may be the RedCap export (`.csv`) or the BIDS `participants.tsv` produced in step 1 — either works, as long as it has a `record_id` or `participant_id` column. The command writes `id_lookup_table.json` to the output directory; entries from `--load_lookup` are kept verbatim, and only new IDs get new pseudonyms. Drop the `--load_lookup` flag when generating the very first lookup table for a release. Rename the resulting file to `id_remapping.json` (or copy it) and place it in your deidentification config folder before continuing.

After creating these files, you can run the deidentify dataset command:

```
b2aiprep-cli deidentify-bids-dataset <path/to/bids/folder> \
    <path/to/audio/output/folder> \
    <path/to/config>
```

The output will be remain in BIDS format. The primary changes are:

- participant IDs are modified
- a session is released when it has released audio or feature files, or rows in a questionnaire
  table; a session with nothing else (for example one abandoned after the microphone check) is not.
  A session without released audio has a `sessions.tsv` row but no `ses-*` directory
- sessions are renamed by `--session-labels` (see below); no original session ID is released
- per-participant `sub-<id>_sessions.tsv` is present with session metadata, including
  `session_index`, the session's place in the participant's released sessions by start time
- `internal` columns are removed from phenotype tables and `sessions.tsv`; `review` columns are removed
  too unless `column_value_reviews.json` has checked their values
- audio sidecar keys follow the field map's `audio_sidecar` table the same way; a key the table does not
  list is removed, and the run logs each such key with the number of sidecars it was removed from
- rows left with no publishable data once those columns are removed are dropped, and tables left with
  no rows are not written
- sensitive audio clips, particularly those which may contain protected health information, are removed
- features that can be used to identify individuals or re-create transcripts for sensitive audios (such as free-speech) are removed, but for only those files

#### Session labels

`redcap2bids` numbers each participant's sessions by start time (`session_index`, over every
session). `--session-labels` chooses the released names:

| value | label | a label can change when |
|---|---|---|
| `ordinal` (default) | `01`, `02`, … over the sessions any access tier could release (a file surviving the removal lists, whatever the task list or `--skip_audio_features`, or questionnaire rows); a tier that withholds one, such as a features-only session in the audio release, shows a gap | an earlier session becomes, or stops being, releasable in any tier |
| `index` | the session's `session_index`; gaps where a session is withheld | a session is added before, or removed from, the RedCap export |
| `uuid` | first 8 characters of the session ID, lower case (16 on a clash), as in v3.1 | never |

Pass `--session-id-map <path outside the output>` to record each released session's original ID and
label. The file holds original IDs and is never released; `scripts/session_label_crosswalk.py` turns
two releases' maps (or, for v3.1, `--old-uuid-labels`) into a publishable old-to-new label table.

#### QA deidentify builds

These options are for building trees to inspect, never for a release (the run logs a warning):

| option | effect | use it to |
|---|---|---|
| `--disposition-level internal` | keeps every column, including `internal` ones | QA every variable for the releasable participants (the participant filtering, ID remapping and audio rules still apply) |
| `--disposition-level review` | removes `internal` columns; passes `review` columns through unchecked | see what the release would contain once the reviewed fields are cleared |
| `--disposition-level release` | the release default when no review manifest is present | — |
| `--keep-shifted-dates` | keeps `date_shift=YES` columns even when `internal` columns are removed | discuss which derived values (time of day, ordinals, intervals) to compute from dates |
| `--skip_audio` | copies no audio; sidecars and `sessions.tsv` are still deidentified | deidentify a `--skip-audio-copy` tree |

### 5. Bundle

For simplicity, the BIDS dataset is bundled into a small set of files for publication and wide dissemination.
Run the following to create the bundled dataset:

```
b2aiprep-cli create-bundled-dataset <path/to/input/bids_folder> <path/to/output/bundled_folder>
```

Once this command is done, you will have a dataset ready to release! Congratulations!
