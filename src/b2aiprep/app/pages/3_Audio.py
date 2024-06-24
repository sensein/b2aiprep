from matplotlib import pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import altair as alt

import matplotlib.cm as cm

from b2aiprep.dataset import VBAIDataset
from b2aiprep.process import Audio, specgram

dataset = VBAIDataset(st.session_state.bids_dir)

st.set_page_config(page_title="Audio", page_icon="📊")

st.markdown("# Audio recordings")


subject_paths = dataset.find_subjects()
subjects = [path.stem[4:] for path in subject_paths]
subject = st.selectbox(
    'Choose a subject.',
    subjects
)

session_paths = dataset.find_sessions(subject)
sessions = [path.stem[4:] for path in session_paths]
session = st.selectbox(
    'Choose a session.',
    sessions
)

audio_files = sorted(dataset.find_audio(subject, session))
audio_record_names = [
    filename.stem.split('_')[3].split('.')[0].replace('recording-', '')
    for filename in audio_files
]

if len(audio_record_names) > 0:
    st.write(
        f"""
        {len(audio_record_names)} audio recordings for this session.
        """
    )
    audio = st.selectbox(
        'Choose an audio file.',
        audio_record_names
    )
    i = audio_record_names.index(audio)

    audio_file = audio_files[i]

    # display a widget for playing the audio file
    task_name = audio_file.stem.split('_')[2]
    rec_name = audio_file.stem.split('_')[3]
    # st.markdown(f"Playing {rec_name}")
    st.audio(str(audio_file))


    # configuration options
    win_length = st.number_input('Window length', value=30)
    hop_length = st.number_input('Hop length', value=15)
    nfft = 1024

    audio = Audio.from_file(str(audio_file))
    audio = audio.to_16khz()
    # set window and hop length to the same to not allow for good Griffin Lim reconstruction
    features_specgram = specgram(audio, win_length=win_length, hop_length=hop_length, n_fft=nfft)
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
    ax.set_ylabel('Frequency (Hz)')
    ax.matshow(log_spec, origin="lower", aspect="auto")

    xlim = ax.get_xlim()
    xticks = ax.get_xticks()
    xticklabels = [f"{int(t * hop_length / 1000)}" for t in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    # reset the xlim, which may have been modified by setting the xticks
    ax.set_xlim(xlim)
    ax.set_xlabel('Time (s)')

    # y-axis is frequency
    ylim = ax.get_ylim()
    yticks = ax.get_yticks()
    # convert yticks into frequencies
    frequencies = yticks / nfft * audio.sample_rate
    frequencies = [f"{int(f)}" for f in frequencies]
    ax.set_yticks(yticks)
    ax.set_yticklabels(frequencies)
    ax.set_ylim(ylim)

    # Display the image
    st.pyplot(fig)
