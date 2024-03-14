"""Functions to prepare acoustic data for the Bridge2AI voice project."""

import numpy as np
import speechbrain.processing.features as spf
import torch
import torchaudio
from scipy.signal import butter
from speechbrain.augment.time_domain import Resample
from speechbrain.dataio.dataio import read_audio, read_audio_info


class Audio:
    def __init__(self, signal: torch.tensor, sample_rate: float):
        """Initialize audio object
        :param signal: is a Torch tensor
        :param sample_rate: is a float
        """
        self.signal = signal
        self.sample_rate = sample_rate

    @classmethod
    def from_file(cls, filename: str, channel: int = 0):
        """Load audio file

        If the file contains more than one channel of audio,
        only the first channel will be loaded.

        :param filename: Path to audio file
        :param channel: Channel to load (default: 0)
        :return: Initialized Audio object
        """
        signal = read_audio(filename)
        meta = read_audio_info(filename)
        if signal.shape[-1] > 1:
            signal = signal[:, [channel]]
        return cls(signal, meta.sample_rate)

    def to_16khz(self):
        """Resample audio to 16kHz and return new Audio object

        TODO: The default resampler does a poor job of taking care of
        aliasing. This should be replaced by a better antialiasing filter.
        """
        resampler = Resample(orig_freq=self.sample_rate, new_freq=16000)
        return Audio(resampler(self.signal.unsqueeze(0)).squeeze(0), 16000)


def specgram(
    audio: Audio, win_length: int = 25, hop_lenth: int = 10, log: bool = False
) -> torch.tensor:
    """Compute the spectrogram using STFT of the audio signal"""
    compute_STFT = spf.STFT(
        sample_rate=audio.sample_rate,
        win_length=win_length,
        hop_length=hop_lenth,
        n_fft=int(400 * audio.sample_rate / 16000),
    )
    stft = compute_STFT(audio.signal.unsqueeze(0))
    return spf.spectral_magnitude(stft.squeeze(), log=log)


def melfilterbank(specgram: torch.tensor, n_mels: int = 20) -> torch.tensor:
    """Compute the Mel filterbank of the spectrogram"""
    n_fft = 2 * (specgram.shape[-1] - 1)
    compute_fbanks = spf.Filterbank(n_mels=n_mels, n_fft=n_fft)
    return compute_fbanks(specgram.unsqueeze(0)).squeeze(0)


def MFCC(
    melfilterbank: torch.tensor, n_coeff: int = 20, compute_deltas: bool = True
) -> torch.tensor:
    """Compute the MFCC of the Mel filterbank

    Optionally compute delta features (2 levels)

    :param melfilterbank: Mel filterbank
    :param n_coeff: Number of coefficients
    :param compute_deltas: Whether to compute delta features
    """
    compute_mfccs = spf.DCT(input_size=melfilterbank.shape[-1], n_out=n_coeff)
    features = compute_mfccs(melfilterbank.unsqueeze(0))
    if compute_deltas:
        compute_deltas = spf.Deltas(input_size=features.shape[-1])
        delta1 = compute_deltas(features)
        delta2 = compute_deltas(delta1)
        features = torch.cat([features, delta1, delta2], dim=2)
    return features.squeeze()


def resample_iir(audio: Audio, lowcut: float, new_sample_rate: int, order: int = 5):
    """Resample audio using IIR filter"""
    b, a = butter(order, lowcut, btype="low", fs=audio.sample_rate)
    filtered = torchaudio.functional.filtfilt(
        audio.signal.unsqueeze(0),
        a_coeffs=torch.tensor(a.astype(np.float32)),
        b_coeffs=torch.tensor(b.astype(np.float32)),
    )
    resampler = Resample(orig_freq=audio.sample_rate, new_freq=16000)
    return Audio(resampler(filtered).squeeze(0), new_sample_rate)
