import argparse

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


def display_summary(df, df_recordings):
    """Display summary statistics of the dataset."""
    n_sessions = df.shape[0]
    n_subjects = df["record_id"].nunique()
    n_subj_gt_1 = df.groupby("record_id").size().gt(1).sum()
    max_num_sessions = df.groupby("record_id").size().max()
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
