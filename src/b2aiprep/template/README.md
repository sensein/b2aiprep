# Project Directory Structure

This README provides an overview of the directory structure and the purpose of each type of file in this project.

## Root Directory

- **dataset_description.json**: A JSON file describing the dataset, including information such as the dataset's purpose, structure, and any relevant metadata.

### phenotype
The phenotype directory stores participant-specific data that is not directly related to specific audio tasks but is relevant for understanding participants' characteristics. Primarily it contains responses to questionnaires by the individual. Tables are grouped into subdirectories (for example `enrollment`, `demographics`, `diagnosis`, `questionnaire`, `task`).

- **phenotype/<group>/<table>.tsv**: A tab-separated values (TSV) file with one table of phenotype data, keyed by `participant_id` (and `session_id` where the data belongs to a session).

- **phenotype/<group>/<table>.json**: A JSON file describing each column of the matching TSV file.

- **phenotype/enrollment/participant.tsv** and **participant.json**: The list of participants and their enrollment information. This dataset has no `participants.tsv` at the root.

## Participant-Specific Directories

Each participant has a directory named `sub-<participant_id>`, containing session-specific data.

### Within `sub-<participant_id>`

### Session-Specific Directories

Each session directory is named `ses-<session_id>` and contains subdirectories for different types of data collected during that session.

#### Voice Data

- **audio/sub-<participant_id>_ses-<session_id>_task-<task_name>.json**: A JSON file containing metadata about the audio recording, including information such as recording identifiers, settings, and conditions.

- **audio/sub-<participant_id>_ses-<session_id>_task-<task_name>.wav**: The raw audio recording file for a specific task and run.
