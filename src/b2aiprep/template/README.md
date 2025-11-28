# Project Directory Structure

This README provides an overview of the directory structure and the purpose of each type of file in this project.

## Root Directory

### phenotype
The phenotype directory stores participant-specific data that is not directly related to specific audio tasks but is relevant for understanding participants' characteristics. Primarily it contains responses to questionnaires by the individual.

- **phenotype/<measurement_tool_name>.tsv**: A tab-separated values (TSV) file containing phenotype data collected using a specific measurement tool.

- **phenotype/<measurement_tool_name>.json**: A JSON file that provides metadata or additional information about the phenotype data collected using a specific measurement tool.

- **dataset_description.json**: A JSON file describing the dataset, including information such as the dataset's purpose, structure, and any relevant metadata.

- **participants.json**: A JSON file containing metadata about the participants involved in the study, including demographic information and other relevant details.

- **participants.tsv**: A TSV file listing the participants involved in the study, along with their corresponding IDs and other relevant information.

## Participant-Specific Directories

Each participant has a directory named `sub-<participant_id>`, containing session-specific data.

### Within `sub-<participant_id>`

### Session-Specific Directories

Each session directory is named `ses-<session_id>` and contains subdirectories for different types of data collected during that session.

#### Voice Data

- **audio/sub-<participant_id>_ses-<session_id>_task-<task_name>.json**: A JSON file containing metadata about the audio recording, including information such as recording identifiers, settings, and conditions.

- **audio/sub-<participant_id>_ses-<session_id>_task-<task_name>.wav**: The raw audio recording file for a specific task and run.
