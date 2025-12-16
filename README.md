# B2AI Prep

A simple Python package to prepare acoustic data for the Bridge2AI voice project.

**Caution:** this package is under active development and interfaces may change rapidly.

## Installation

Requires a Python == 3.11 environment.

```
pip install b2aiprep
```

## Usage

See commands available through the CLI:

```bash
b2aiprep-cli --help
```

## BIDS data

This package provides conversion from a RedCap/file-based custom structure into [BIDS](https://bids-standard.github.io/bids-starter-kit/folders_and_files/folders.html) for downstream analysis. In addition, utilities are provided for working with the BIDS-like structured data and to support data analysis of voice and questionnaire data.

See [RELEASE.md](RELEASE.md) for details on releasing a dataset using this package.

See the [tutorial.ipynb](docs/tutorial.ipynb) for a few use examples of data in the BIDS format.

### Output formats

The primary output structure is BIDS: nested sessions within individual specific folders.

```
/bids_dataset
├── README.md
├── dataset_description.json
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
```

A bundled version of the data is published from BIDS. The bundled version of the dataset contains parquet files with features derived from the audio.
