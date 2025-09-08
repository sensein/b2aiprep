from pathlib import Path

import altair as alt
import streamlit as st

from b2aiprep.prepare.dataset import VBAIDataset

from b2aiprep.dashboard.utils import create_bar_chart, display_summary, load_dataset

st.set_page_config(page_title="Summary", page_icon="📊")

st.write("# Bridge2AI Voice Data Dashboard")

bids_dir = st.sidebar.text_input(
    "Path to BIDS-like formatted data folder",
    value=st.session_state.get("bids_dir", ""),
    help="Path to the BIDS-like formatted data folder.",
)

if not bids_dir:
    st.stop()

bids_dir = Path(bids_dir).resolve()
dataset: VBAIDataset = load_dataset(bids_dir)

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

# cache the session data to avoid reloading on every interaction
@st.cache_data(ttl=3600, show_spinner=True)
def load_session_data():
    """Load session data from the dataset."""
    return dataset.load_and_pivot_questionnaire("sessionschema")

@st.cache_data(ttl=3600, show_spinner=True)
def load_recordings_data():
    """Load recordings data from the dataset."""
    return dataset.load_and_pivot_questionnaire("recordingschema")

# add loading indicator
df = load_session_data()
df_recordings = load_recordings_data()

display_summary(df, df_recordings)

if "session_site" in df.columns:
    st.write("## Subjects by site")
    st.altair_chart(
        create_bar_chart(
            df[["participant_id", "session_site"]].drop_duplicates(),
            "session_site",
            "count",
            "Number of participants",
            "Site",
        )
    )

    st.write("## Sessions by site")
    st.altair_chart(create_bar_chart(df, "session_site", "count", "Number of sessions", "Site"))

st.write("## Session durations")
session_durations = df["session_duration"].astype(float) / 60.0
session_durations = session_durations.dropna()
hist = (
    alt.Chart(session_durations.reset_index())
    .mark_bar()
    .encode(
        alt.X(
            "session_duration:Q",
            bin=alt.Bin(step=15),
            scale=alt.Scale(domain=[0, 360]),
            title="Session duration (minutes)"
        ),
        y="count()",
    )
    .configure_axis(
        labelFontSize=12,
        titleFontSize=14,
    )
)
st.altair_chart(hist)
