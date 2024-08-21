# Project Directory Structure

This README provides an overview of the directory structure and the purpose of each type of file in this project.

## Root Directory

### phenotype
The phenotype directory stores participant-specific data that is not directly related to specific audio tasks but is relevant for understanding participants' characteristics.

- **phenotype/<measurement_tool_name>.tsv**: A tab-separated values (TSV) file containing phenotype data collected using a specific measurement tool.

- **phenotype/<measurement_tool_name>.json**: A JSON file that provides metadata or additional information about the phenotype data collected using a specific measurement tool.

- **CHANGES**: A file that documents the changes, updates, and revisions made to the project over time.

- **dataset_description.json**: A JSON file describing the dataset, including information such as the dataset's purpose, structure, and any relevant metadata.

- **participants.json**: A JSON file containing metadata about the participants involved in the study, including demographic information and other relevant details.

- **participants.tsv**: A TSV file listing the participants involved in the study, along with their corresponding IDs and other relevant information.

## Participant-Specific Directories

Each participant has a directory named `sub-<participant_id>`, containing session-specific data.

### Within `sub-<participant_id>`

- **sessions.tsv**: A TSV file listing all sessions for the participant, including session IDs and other relevant details.

### Session-Specific Directories

Each session directory is named `ses-<session_id>` and contains subdirectories for different types of data collected during that session.

#### Voice Data

- **voice/sub-<participant_id>_ses-<session_id>_task-<task_name>_run-<index>_transcript.txt**: A text file containing the transcript of the audio recording for a specific task and run.

- **voice/sub-<participant_id>_ses-<session_id>_task-<task_name>_run-<index>_metadata.json**: A JSON file containing metadata about the audio recording, including information such as recording settings and conditions.

- **voice/sub-<participant_id>_ses-<session_id>_task-<task_name>_run-<index>_features.pt**: A file containing extracted features from the audio recording, stored in a format suitable for further analysis.

- **voice/sub-<participant_id>_ses-<session_id>_task-<task_name>_run-<index>_audio.wav**: The raw audio recording file for a specific task and run.

#### Behavioral Data

- **beh/sub-<participant_id>_ses-<session_id>_task-<task_name>_run-<index>_response.json**: A JSON file containing the participant's responses for a specific task and run.

- **beh/sub-<participant_id>_ses-<session_id>_task-<task_name>_run-<index>_metadata.json**: A JSON file containing metadata about the behavioral data, including information such as task conditions and settings.

This structure helps to organize and manage the data collected for each participant and session, ensuring that all relevant information is easily accessible and well-documented.
