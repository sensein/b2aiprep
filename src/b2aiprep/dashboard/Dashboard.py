import argparse
import sys
from pathlib import Path

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


def initialize_session_state(dataset):
    """Initialize Streamlit session state."""
    if "dataset" not in st.session_state:
        st.session_state.dataset = dataset


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


def main():
    args = parse_args(sys.argv[1:])
    bids_dir = Path(args.bids_dir).resolve()
    dataset = load_dataset(bids_dir)
    initialize_session_state(dataset)

    st.set_page_config(
        page_title="b2ai voice",
        page_icon="👋",
    )

    st.write("# Bridge2AI Voice Data Dashboard")

    st.sidebar.success("Choose an option above.")

    st.markdown(
        """
        This dashboard allows you to explore the voice data collected by Bridge2AI.

        You should first load in the data below by providing a path to the BIDS-like
        formatted data folder. Once you've done that, you can explore the data.
        """
    )

    st.markdown("## Session information")

    df = dataset.load_and_pivot_questionnaire("sessionschema")
    df_recordings = dataset.load_and_pivot_questionnaire("recordingschema")

    display_summary(df, df_recordings)

    st.write("## Subjects by site")
    st.altair_chart(
        create_bar_chart(
            df[["record_id", "session_site"]].drop_duplicates(),
            "session_site",
            "count",
            "Number of participants",
            "Site",
        )
    )

    st.write("## Sessions by site")
    st.altair_chart(create_bar_chart(df, "session_site", "count", "Number of sessions", "Site"))

    st.write("## Session durations")
    session_durations = df["session_duration"].astype(float) / 3600.0
    session_durations = session_durations.dropna()
    hist = (
        alt.Chart(session_durations.reset_index())
        .mark_bar()
        .encode(
            alt.X("session_duration:Q", bin=alt.Bin(maxbins=20), title="Session duration (hours)"),
            y="count()",
        )
        .properties(
            width=600,
            height=400,
        )
    )
    st.altair_chart(hist)


if __name__ == "__main__":
    main()
