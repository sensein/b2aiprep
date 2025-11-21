# Release Process
This document provides instructions for preparing and releasing the Bridge2AI Voice dataset for public and internal releases.

## Quick start

```sh
WORKING_DIR=${HOME}/data/bridge2ai/pediatric

b2aiprep-cli reproschema-to-redcap $WORKING_DIR/Audio_300_Release $WORKING_DIR/Survey_300_Release $WORKING_DIR/redcap.csv
b2aiprep-cli redcap2bids $WORKING_DIR/redcap.csv --outdir $WORKING_DIR/bids --audiodir $WORKING_DIR/Audio_300_Release
b2aiprep-cli deidentify-bids-dataset $WORKING_DIR/bids $WORKING_DIR/Audio_300_Release $WORKING_DIR/release_config
```

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

Pediatric data that has been downloaded from reproschema-ui does not have a redcap csv associated with it, so we must generate one. This can be done using the following command:

```
b2aiprep-cli reproschema-to-redcap <path/to/audio/folder> <path/to/survey/folder> <path/to/output/redcap_csv>
```

In the end, we should have an audio folder with all the .wav files contained in it, and a redcap csv file.

Note: Before proceeding to the next step, please ensure that the record_id match the folder names downloaded from reproschema-ui.

**Optional**: We can run the `clean_redcap.py` script, located in the `external_scripts` folder, to fix any known reproschema to redcap issues.

```
python clean_redcap.py <path/to/redcap_csv> <path/to/output/fixed_redcap_csv>
```

### 2. Data Formatting & Feature Extraction
Once the redcap csv and audio files are ready, we can begin formatting the data and extraction feaatures from the audio files.
To do this, run the following commands: 

Converting to Bids:
```
b2aiprep-cli redcap2bids <path/to/redcap_csv> \
    <path/to/output/bids_folder> \
    --audiodir <path/to/audio/files>
```
### 3. Processing Data & De-identifying Data
Once the data has been been reformatted into a bids format, we need to make sure to remove any entries that could have
sensitive information. To do this, we must for create a **Config** folder.

The folder will contain 3 files:
- `audio_filestems_to_remove.json` (File containing a list of sensitive audio files to remove)
- `id_remapping.json` (File containing participant ids to change)
- `participants_to_remove.json` (File containing list of participants to remove)

Once the folder is generated, you can run the deidentify dataset command: 

```
b2aiprep-cli deidentify-bids-dataset <path/to/bids/folder> <path/to/audio/output/folder> <path/to/config>
```

The resulting output will be a update bids folder in the output directory with the sensitive participants and ids remove and renamed.

Note: Only audio files, metadata fils and tsv's will be in the resulting output.


### 4. Feature Extraction:
Run the following command to extract features: 
```
b2aiprep-cli generate-audio-features <path/to/input/bids_folder> \
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

### 5. Creating Derived Dataset
Run the following to create the derived dataset: 
```
b2aiprep-cli create-derived-dataset <path/to/input/bids_folder>  \ 
    <path/to/output/derive_folder> 
```
The following command will generate:
    1. `phenotype.tsv`- File containging all phenotype data for each participant
    2. `static_features.tsv`- File containging features extracted for each audio recording
    3. `mfcc.parquet` - Parquet file containing all the Mel-Frequency Cepstral Coefficients for each audio recording
    4. `spectrogram.parquet`- Parquet file conntaining all spectrograms for each audio recording
    

Once this command is done, you will have a dataset ready to release! Congratulations!
