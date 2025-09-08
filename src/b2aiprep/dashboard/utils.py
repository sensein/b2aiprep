import argparse

import pandas as pd

import altair as alt
import streamlit as st

from b2aiprep.prepare.dataset import VBAIDataset


def parse_args(args):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Dashboard for audio data in BIDS format.")
    parser.add_argument("bids_dir", help="Folder with the BIDS data", default="output")
    return parser.parse_args(args)


def load_dataset(bids_dir):
    """Load the dataset from the given BIDS directory."""
    if not bids_dir.exists():
        raise ValueError(f"Folder {bids_dir} does not exist.")
    return VBAIDataset(bids_dir)

def load_dataset_with_participants_data(bids_dir):
    """Load the dataset and participants TSV data."""
    dataset = load_dataset(bids_dir)

    # Load participants.tsv if it exists
    participants_tsv = bids_dir / "participants.tsv"
    participants_df = None
    if participants_tsv.exists():
        participants_df = pd.read_csv(participants_tsv, sep="\t")

    return dataset, participants_df


def get_recording_summary_from_audio_files(dataset):
    """Get recording summary by scanning audio files directly since
    questionnaire structure has changed in the new dataset format."""
    recordings_data = []

    subjects = dataset.find_subjects()
    for subject_path in subjects:
        subject_id = subject_path.name.replace("sub-", "")
        sessions = dataset.find_sessions(subject_id)

        for session_path in sessions:
            session_id = session_path.name.replace("ses-", "")
            audio_files = dataset.find_audio(subject_id, session_id)

            for audio_file in audio_files:
                # Extract task from filename
                # Pattern: sub-{id}_ses-{session}_task-{task}.wav
                filename = audio_file.name
                if "_task-" in filename:
                    task_part = filename.split("_task-")[1].replace(".wav", "")
                    recordings_data.append(
                        {
                            "participant_id": subject_id,
                            "session_id": session_id,
                            "task_name": task_part,
                            "filename": filename,
                            "file_path": str(audio_file),
                        }
                    )

    return pd.DataFrame(recordings_data)


def display_summary(df, df_recordings):
    """Display summary statistics of the dataset."""
    n_sessions = df.shape[0]
    
    # Use participant_id or fall back to record_id for backward compatibility
    id_column = "participant_id" if "participant_id" in df.columns else "record_id"
    n_subjects = df[id_column].nunique()
    n_subj_gt_1 = df.groupby(id_column).size().gt(1).sum()
    max_num_sessions = df.groupby(id_column).size().max()
    n_recordings = df_recordings.shape[0]

    st.write(
        f"""
        * {n_subjects} subjects have participated.
        * {n_sessions} sessions have been conducted.
        * {n_subj_gt_1} subjects ({n_subj_gt_1/n_subjects:1.2%}) \
            have more than one session (max = {max_num_sessions}).
        * {n_recordings} recordings have been made.
        """
    )


def display_summary_with_tsv_data(dataset, participants_df=None):
    """Display summary statistics using TSV data and audio file scanning."""
    # Get recordings summary from audio files
    recordings_df = get_recording_summary_from_audio_files(dataset)

    if participants_df is not None:
        # Use participants TSV for subject count
        n_subjects = len(participants_df)
        n_sessions = recordings_df["session_id"].nunique() if not recordings_df.empty else 0
        n_recordings = len(recordings_df)

        # Check for multiple sessions per subject
        id_column = "participant_id" if "participant_id" in recordings_df.columns else "record_id"
        session_counts = recordings_df.groupby(id_column)["session_id"].nunique()
        n_subj_gt_1 = (session_counts > 1).sum()
        max_num_sessions = session_counts.max() if not session_counts.empty else 0
        percent_gt_1 = n_subj_gt_1 / n_subjects if n_subjects > 0 else 0.0
        st.write(
            f"""
            * {n_subjects} subjects have participated.
            * {n_sessions} unique sessions have been conducted.
            * {n_subj_gt_1} subjects ({percent_gt_1:1.2%}) \
                have more than one session (max = {max_num_sessions}).
            * {n_recordings} recordings have been made.
            """
        )

        return recordings_df
    else:
        display_summary(pd.DataFrame(), recordings_df)
        return recordings_df


def create_bar_chart(df, group_col, count_col, x_title, y_title):
    """Create a bar chart using Altair."""
    chart_data = df.groupby(group_col).size().reset_index(name=count_col)
    chart_data.columns = [y_title, x_title]
    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=x_title,
            y=y_title,
        )
        .properties(
            width=600,
            height=400,
        )
    )
    return chart
