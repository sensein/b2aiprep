# Release Process

This document provides instructions for preparing and releasing the Bridge2AI Voice dataset for public and internal releases.

Required inputs:

- A RedCap CSV
    - If converting pediatric data, ReproSchema data must first be converted to the RedCap CSV format
- A directory with audio recordings in .wav format

Note that there are implicit assumptions made regarding the structured data and the .wav files, e.g. it is assumed the .wav files have a UUID in their filename which is referenced in the structured data.

## Quick start

```sh
WORKING_DIR=${HOME}/data/bridge2ai/pediatric

b2aiprep-cli reproschema-to-redcap $WORKING_DIR/Audio_300_Release $WORKING_DIR/Survey_300_Release $WORKING_DIR/redcap.csv
b2aiprep-cli redcap2bids $WORKING_DIR/redcap.csv --outdir $WORKING_DIR/bids --audiodir $WORKING_DIR/Audio_300_Release
b2aiprep-cli deidentify-bids-dataset $WORKING_DIR/bids $WORKING_DIR/de-identified-bids $WORKING_DIR/release_config
```

### 0. (Optional) Update the template files
---

```sh
python -m b2aiprep.cli update-bids-template --dry-run
```

---
### 1. (Optional) ReproSchema to RedCap conversion

This process is required for the pediatric cohort, but can be skipped for the adult cohort.

Pediatric data was collected using [reproschema-ui](https://github.com/ReproNim/reproschema-ui), which differs from the other protocols and requires additional preprocessing.

When first downloaded, the pediatric data consists of:

- A **Survey folder** (questionnaire responses).
- An **Audio folder** (voice recordings).

The following command parses the survey folder and audio folder, acquires necessary reference data with HTTP calls to GitHub, and outputs a RedCap CSV file.

```
b2aiprep-cli reproschema-to-redcap <path/to/audio/folder> <path/to/survey/folder> <path/to/output/redcap_csv>
```

Sanity check: The `record_id` column of the RedCap CSV should match the folder names in the survey data folder downloaded from reproschema-ui.

### 2. Convert RedCap to BIDS

The next step in the pipeline reorganizes the dataset to follow the [BIDS format](https://bids.neuroimaging.io/index.html). Briefly, BIDS is a folder structure with sessions nested under unique folders for each subject and certain metadata files required.

The following command will parse the RedCap CSV into multiple phenotype CSVs and reorganize the audio data to follow the BIDS folder structure:

```
b2aiprep-cli redcap2bids <path/to/redcap_csv> \
    <path/to/output/bids_folder> \
    --audiodir <path/to/audio/files>
```

An optional `--max-audio-workers` controls the number of threads used for writing out audio files as writing of audio files is the speed bottleneck of this command.

### 3. Deidentification

Once the data has been been reformatted into BIDS format, we need to make sure to remove any entries that could have
sensitive information, referred to as deidentification. Deidentification requires creation of the following configuration files:

- `audio_filestems_to_remove.json` (File containing a list of sensitive audio files to remove)
- `id_remapping.json` (File containing participant ids to change)
- `participants_to_remove.json` (File containing list of participants to remove)
- `sensitive_audio_tasks.json` (File containing list of sensitive audio tasks, necessary if not skipping audio feature deidentification)

Create these files and place them in a folder, e.g. `deidentification_config` (the "config" folder).
After creating these files, you can run the deidentify dataset command: 

```
b2aiprep-cli deidentify-bids-dataset <path/to/bids/folder> \
    <path/to/audio/output/folder> \
    <path/to/config>
```

The output will be remain in BIDS format. The primary changes are:

- participant IDs are modified
- sensitive columns are removed
- sensitive audio clips, particularly those which may contain protected health information, are removed
- features that can be used to identify individuals or re-create transcripts for sensitive audios (such as free-speech) are removed, but for only those files

### 4. Feature Extraction

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
├── README
├── dataset_description.json
├── participants.tsv
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
└── participants.json
```

### 5. Publish

For simplicity, the BIDS dataset is parsed into a small set of files for publication and wide dissemination. 
Run the following to create the derived dataset: 
```
b2aiprep-cli create-derived-dataset <path/to/input/bids_folder>  \ 
    <path/to/output/derive_folder> 
```

Once this command is done, you will have a dataset ready to release! Congratulations!
