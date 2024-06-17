import streamlit as st
import pandas as pd

from b2aiprep.dataset import VBAIDataset

dataset = VBAIDataset(st.session_state.bids_dir)

st.set_page_config(page_title="Audio", page_icon="📊")

st.markdown("# Dataset validation")

audio_files_exist = dataset.validate_audio_files_exist()

if audio_files_exist:
    st.success("All audio files exist.")
else:
    st.error("Some audio files are missing.")