# Release Process
This document provides instructions for preparing and releasing the Bridge2AI Voice dataset for public and internal releases.

## Table of Contents
1. [Data Retrieval](#data-retrieval)
2. [Data Formatting](#data-formatting)

---
### 1. Data Retrieval
The B2AIPrep library mainly requires two major things, a directory containing all the audio files, and secondly a redcap csv.
The redcap csv should contain indexes of questionnaire data as well as records of their audio recordings. The folder containing the
audio files should be of the following strucuture: 

/audio_files
├── 3e9d8f2f-973b-442b-9f9d-51cd32003288.wav
├── e582b7fa-8779-4663-bdef-8fb6c11e3f5d.wav
├── f5fcd7a7-2d6d-4a4f-bfbe-5f85973debf1.wav
├── 2d84d2ed-2446-42f6-9931-4bfb9be45c4b.wav

#### Pediatric Data Retrival


Pediatric data was collected using [reproschema-ui](https://github.com/ReproNim/reproschema-ui), which differs from the other protocols and requires additional preprocessing.

When first downloaded, the pediatric data consists of:
- A **Survey folder** (questionnaire responses).
- An **Audio folder** (voice recordings).


To organize the audio files into the format shown above, run:

```
b2aiprep-cli reproschema-audio-to-folder <path/to/audio/folder> <path/to/audio/output/folder>
```

Pediatric data that has been downloaded from reproschema-ui does not have a redcap csv associated with it, so we must generate one. This can be done using the following command:

```
b2aiprep-cli reproschema-to-redcap <path/to/audio/output/folder> <path/to/survey/folder> <path/to/output/redcap_csv> --participant_group subjectparticipant_basic_information_schema
```

In the end, we should have an audio folder with all the .wav files contained in it, and a redcap csv file.

Note: Before proceeding to the next step, please ensure that the record_id match the folder names downloaded from reproschema-ui.

### 2. Data Formatting & Feature Extraction
Once the redcap csv and audio files are ready, we can begin formatting the data and extraction feaatures from the audio files.
To do this, run the following command: 

```
b2aiprep-cli prepare-bids <path/to/output/folder> \
    --redcap_csv_path <path/to/redcap_csv> \
    --audio_dir_path <path/to/audio/files> \
    --is_sequential True
```
Note: Due to the potential size and quantity of the audio files, it is recommended to run this as a sbatch.

Once the command is complete, you should have the following bids-like structure:

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
└── participants.json


