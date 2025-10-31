# Release Process
This document provides instructions for preparing and releasing the Bridge2AI Voice dataset for public and internal releases.

---
### 1. Data Retrieval
The B2AIPrep library mainly requires two major things, a directory containing all the audio files, and secondly a redcap csv.
The redcap csv should contain indexes of questionnaire data as well as records of their audio recordings. The folder containing the
audio files should be of the following strucuture: 

```
/audio_files
├── 3e9d8f2f-973b-442b-9f9d-51cd32003288.wav
├── e582b7fa-8779-4663-bdef-8fb6c11e3f5d.wav
├── f5fcd7a7-2d6d-4a4f-bfbe-5f85973debf1.wav
├── 2d84d2ed-2446-42f6-9931-4bfb9be45c4b.wav
```

#### Pediatric Data Retrieval


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
To do this, run the following commands: 

Converting to Bids:
```
b2aiprep-cli redcap2bids <path/to/redcap_csv> \
    <path/to/output/bids_folder> \
    --audio_dir <path/to/audio/files>
```

Feature Extraction:
```
b2aiprep-cli generate-audio-features <path/to/redcap_csv> \
    <path/to/input/bids_folder> \
    <path/to/output/folder \
    --is_sequential True
```
Note: Due to the potential size and quantity of the audio files, it is recommended to run this as a sbatch.

Once the command is complete, you should have the following bids-like structure:
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
└── participants.json
```
### Processing Data for Release
Once the data has been been reformatted into a bids format, we need to make sure to remove any entries that could have
sensitive information. To do this, we must for create a **Config** folder.

The folder will contain 3 files:
- `audio_filestems_to_remove.json` (File containing a list of sensitive audio files to remove)
- `id_remapping.json` (File containing participant ids to change)
- `participants_to_remove.json` (File containing list of participants to remove)

Once the folder is generated, you can run the publish dataset command: 

```
b2aiprep-cli publish-bids-dataset <path/to/bids/folder> <path/to/audio/output/folder> <path/to/config>
```

The resulting output will be a update bids folder in the output directory with the sensitive participants and ids remove and renamed.

Note: Only audio files, metadata fils and tsv's will be in the resulting output. No `.pt` will be present in the the resulting output. 

At this point the resulting output folder is ready for release! 

