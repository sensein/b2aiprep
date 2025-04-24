from pathlib import Path

import altair as alt
import streamlit as st

from b2aiprep.prepare.dataset import VBAIDataset

from b2aiprep.dashboard.utils import create_bar_chart, display_summary, initialize_session_state, load_dataset

st.set_page_config(
    page_title="b2ai voice",
    page_icon="👋",
)


st.write("# Bridge2AI Voice Data Dashboard")

bids_dir = st.text_input(
    "Path to BIDS-like formatted data folder",
    value=st.session_state.get("bids_dir", ""),
    help="Path to the BIDS-like formatted data folder.",
)

if not bids_dir:
    st.stop()

bids_dir = Path(bids_dir).resolve()
dataset = load_dataset(bids_dir)

if dataset:
    st.success("BIDS formatted dataset found.")

st.session_state.dataset = dataset

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
