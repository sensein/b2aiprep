import streamlit as st
import pandas as pd

from b2aiprep.dataset import VBAIDataset

st.set_page_config(page_title="Audio", page_icon="📊")

@st.cache_data
def get_bids_data():
    # TODO: allow user to specify input folder input
    dataset = VBAIDataset('output')
    return dataset

st.markdown("# Dataset validation")

dataset = get_bids_data()
audio_files_exist = dataset.validate_audio_files_exist()

if audio_files_exist:
    st.success("All audio files exist.")
else:
    st.error("Some audio files are missing.")