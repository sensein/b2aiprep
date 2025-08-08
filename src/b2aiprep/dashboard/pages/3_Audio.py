import argparse
from pathlib import Path
import sys

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt

# from b2aiprep.process import Audio, specgram
from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.features_extraction.torchaudio import (
    extract_spectrogram_from_audios,
)
from senselab.audio.tasks.preprocessing.preprocessing import resample_audios

from b2aiprep.dashboard.utils import load_dataset
from b2aiprep.prepare.dataset import VBAIDataset

st.set_page_config(page_title="Audio", page_icon="📊")


if "bids_dir" not in st.session_state:
    st.session_state.bids_dir = ""

bids_dir = st.sidebar.text_input(
    "Path to BIDS-like formatted data folder",
    key="bids_dir",
    value=st.session_state.get("bids_dir", ""),
    placeholder="e.g. /path/to/bids/data",
    help="Path to the BIDS-like formatted data folder.",
)

if not bids_dir:
    st.stop()

bids_dir = Path(bids_dir).resolve()
dataset: VBAIDataset = load_dataset(bids_dir)

st.markdown("# Audio recordings")

subject_paths = dataset.find_subjects()
subjects = sorted([path.stem[4:] for path in subject_paths])
subject = st.selectbox("Choose a subject.", subjects)

session_paths = dataset.find_sessions(subject)
sessions = sorted([path.stem[4:] for path in session_paths])
session = st.selectbox("Choose a session.", sessions)

audio_files = sorted(dataset.find_audio(subject, session))
audio_task_names = [
    filename.stem[filename.stem.index('_task-')+6:] for filename in audio_files
]
audio_task_names = list(set(audio_task_names))
audio_task_names.sort()

if len(audio_task_names) == 0:
    st.write(
        f"""
        No audio recordings found for subject {subject} and session {session}.
        """
    )
    st.stop()
else:
    st.write(
        f"""
        {len(audio_task_names)} audio recordings for this session.
        """
    )

task_name = st.selectbox("Choose a task.", audio_task_names)
i = audio_task_names.index(task_name)

audio_file = audio_files[i]

# st.markdown(f"Playing {rec_name}")
st.audio(str(audio_file))

# configuration options
win_length = st.number_input("Window length", value=30)
hop_length = st.number_input("Hop length", value=15)
# audio_length = st.number_input("Audio length (seconds)", value=30)
nfft = 1024

audio = Audio(filepath=str(audio_file))
audio = resample_audios([audio], 16000)[0]

# set window and hop length to the same to not allow for good Griffin Lim reconstruction
features_specgram = extract_spectrogram_from_audios(
    audios=[audio], n_fft=nfft, hop_length=hop_length, win_length=win_length
)
features_specgram = features_specgram[0]["spectrogram"]  # torch.Tensor
# specgram(audio, win_length=win_length, hop_length=hop_length, n_fft=nfft)
# features_melfilterbank = melfilterbank(features_specgram, n_mels=n_mels)
# features_mfcc = MFCC(features_melfilterbank, n_coeff=n_coeff, compute_deltas=compute_deltas)
# features_opensmile = extract_opensmile(audio, opensmile_feature_set, opensmile_feature_level)

# plot the spectrogram
st.markdown("### Spectrogram")

# convert tensor to numpy
features_specgram = features_specgram.numpy()
# convert to db
log_spec = 10.0 * np.log10(np.maximum(features_specgram, 1e-10)).T
fig, ax = plt.subplots(1, 1)
ax.set_ylabel("Frequency (Hz)")
ax.matshow(log_spec, origin="lower", aspect="auto")

xlim = ax.get_xlim()
xticks = ax.get_xticks()
xticklabels = [f"{int(t * hop_length / 1000)}" for t in xticks]
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)
# reset the xlim, which may have been modified by setting the xticks
ax.set_xlim(xlim)
ax.set_xlabel("Time (s)")

# y-axis is frequency
ylim = ax.get_ylim()
yticks = ax.get_yticks()
# convert yticks into frequencies
frequencies = yticks / nfft * audio.sampling_rate
frequencies = [f"{int(f)}" for f in frequencies]
ax.set_yticks(yticks)
ax.set_yticklabels(frequencies)
ax.set_ylim(ylim)

# Display the image
st.pyplot(fig)
