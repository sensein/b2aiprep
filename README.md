# B2AI Prep

A simple Python package to prepare acoustic data for the Bridge2AI voice project.

**Caution:** this package is under activedevelopment and interfaces may change rapidly over the next few weeks.

## Installation
Requires a Python >= 3.10, <3.12 environment

```
pip install b2aiprep
```

## Usage
Four commands are available through the CLI:

```bash
b2aiprep-cli --help
```

1. Convert an audio file to features:

    The simplest form takes an audio file, a subject id, and a task name.

    ```bash
    b2aiprep-cli convert test_audio.wav s1 mpt
    ```

    It will save a pytorch `.pt` file with a dictionary of features. This can be
    loaded by `torch.load()`. The file is named following a simple convention:
    `sub-<subject_id>_task-<task_name>_md5-<checksum>_features.pt`

2. Batch process audio files

    This requires a CSV file, where each line is of the form:
    path/to/audio.wav

    This also supports a CSV file where each line is of the form:
    path/to/audio.wav,subject_id,task_name

    To generate this csv file from the Production directory pulled from wasabi, use command 3.

    ```bash
    b2aiprep-cli batchconvert filelist.csv --plugin cf n_procs=2 --outdir out --save_figures
    ```

    The above command uses pydra under the hood to parallel process the audio files.
    All outputs are currently stored in a single directory specified by the `--outdir`
    flag.

    One can also generate a hugging face dataset in the output directoryby specifying the
     `--dataset` flag.

3. Generate csv file to feed to batchconvert

    ```bash
    b2aiprep-cli createbatchcsv input_dir outfile
    ```

    The input directory should point to the location of the `Production` directory pulled from Wasabi e.g. `/Users/b2ai/production`.

    This directory can have subfolders for each institution, (e.g. `production/MIT`),
    and each subdirectory is expected to have all the `.wav` files from each institution.

    Outfile is the path to and name of the csv file to be generated, e.g. `audiofiles.csv`

4. Verify if two audio files are from the same speaker

    ```bash
    b2aiprep-cli test_audio1.wav test_audio2.wav --model 'speechbrain/spkrec-ecapa-voxceleb'
    ```

    This will use the speechbrain speaker recognition model to verify that the two
    audio files are from the same speaker.

    There is a notebook in the docs directory that can be used to interact with the library
    programmatically.

5. Convert the speaker in the source audio file (1st argument) into the speaker of the target audio file (2nd argument)
     and save the result in the output file (3rd argument)

    ```bash
    b2aiprep-cli convert-voice data/vc_source.wav data/vc_target.wav data/vc_output.wav
    ```

6. Transcribe the audio

    ```bash
    b2aiprep-cli transcribe data/vc_source.wav
    ```

    Or use a different model. Note that the large model may take some time to download.

    ```bash
    b2aiprep-cli transcribe data/vc_source.wav --model_id 'openai/whisper-large-v3' --return_timestamps true
    ```
