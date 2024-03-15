# B2AI Prep

A simple Python package to prepare acoustic data for the Bridge2AI voice project.

**Caution:** this package is still under development and may change rapidly over the next few weeks.

## Installation
Requires a Python >= 3.10 environment

```
pip install b2aiprep`
```

## Usage
Two commands are available through the CLI:

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

2. Verify if two audio files are from the same speaker

```bash
b2aiprep-cli test_audio1.wav test_audio2.wav --model 'speechbrain/spkrec-ecapa-voxceleb'
```

This will use the speechbrain speaker recognition model to verify that the two
audio files are from the same speaker.

There is a notebook in the docs directory that can be used to interact with the library
programmatically.
