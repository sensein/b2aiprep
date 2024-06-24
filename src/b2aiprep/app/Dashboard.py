import argparse
from pathlib import Path
import sys

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from b2aiprep.dataset import VBAIDataset

def parse_args(args):
    parser = argparse.ArgumentParser('Dashboard for audio data in BIDS format.')
    parser.add_argument('bids_dir', help='Folder with the BIDS data', default='output')
    return parser.parse_args(args)

args = parse_args(sys.argv[1:])
bids_dir = Path(args.bids_dir).resolve()
if not bids_dir.exists():
    raise ValueError(f"Folder {bids_dir} does not exist.")

if 'bids_dir' not in st.session_state:
    st.session_state.bids_dir = bids_dir.as_posix()

dataset = VBAIDataset(st.session_state.bids_dir)

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

st.markdown(
    """## Session information"""
)

# every user has a sessionschema which we can get info for the users from
df = dataset.load_and_pivot_questionnaire('sessionschema')
n_sessions = df.shape[0]
n_subjects = df['record_id'].nunique()
n_subj_gt_1 = df.groupby('record_id').size().gt(1).sum()
n_subj_gt_3 = df.groupby('record_id').size().gt(3).sum()
max_num_sessions = df.groupby('record_id').size().max()
df_recordings = dataset.load_and_pivot_questionnaire('recordingschema')
n_recordings = df_recordings.shape[0]

st.write(
    f"""
    * {n_subjects} subjects have participated.
    * {n_sessions} sessions have been conducted.
    * {n_subj_gt_1} subjects ({n_subj_gt_1/n_subjects:1.2%}) have more than one session (max = {max_num_sessions}).
    * {n_recordings} recordings have been made.
    """
)

st.write(
    """
    ## Subjects by site
    """
)
n_participants_per_size = df[['record_id', 'session_site']].drop_duplicates().groupby('session_site').size()
n_participants_per_size = n_participants_per_size.reset_index(name='count')
n_participants_per_size.columns = ['Site', 'Number of participants']

site_chart = alt.Chart(
    n_participants_per_size
).mark_bar().encode(
    x='Number of participants',
    y='Site',
).properties(
    width=600,
    height=400,
)

st.altair_chart(site_chart)

st.write(
    """
    ## Sessions by site
    """
)

n_sessions_per_site = df.groupby('session_site').size()
n_sessions_per_site = n_sessions_per_site.reset_index(name='count')
n_sessions_per_site.columns = ['Site', 'Number of sessions']

site_chart = alt.Chart(
    n_sessions_per_site
).mark_bar().encode(
    x='Number of sessions',
    y='Site',
).properties(
    width=600,
    height=400,
)

st.altair_chart(site_chart)

session_durations = df['session_duration'].astype(float) / 3600.0
session_durations = session_durations.dropna()

st.write(
    """
    ## Session durations
    """
)

# altair histogram of session durations
hist = alt.Chart(session_durations.reset_index()).mark_bar().encode(
    alt.X("session_duration:Q", bin=alt.Bin(maxbins=20), title="Session duration (hours)"),
    y='count()',
).properties(
    width=600,
    height=400,
)

st.altair_chart(hist)