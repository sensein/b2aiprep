from pathlib import Path
import streamlit as st

from b2aiprep.dashboard.utils import load_dataset
from b2aiprep.prepare.dataset import VBAIDataset

st.set_page_config(page_title="Audio", page_icon="📊")

st.markdown("# Dataset validation")

if "bids_dir" not in st.session_state:
    st.session_state.bids_dir = ""

bids_dir = st.sidebar.text_input(
    "Path to BIDS-like formatted data folder",
    key="bids_dir",
    value=st.session_state.get("bids_dir", ""),
    placeholder="e.g. /path/to/bids/data",
    help="Path to the BIDS-like formatted data folder.",
)

bids_dir = Path(bids_dir).resolve()
dataset: VBAIDataset = load_dataset(bids_dir)

audio_files_exist = dataset.validate_audio_files_exist()

if audio_files_exist:
    st.success("All audio files exist.")
else:
    st.error("Some audio files are missing.")
