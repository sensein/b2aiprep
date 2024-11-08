import typing as t

import numpy as np
from scipy.signal import butter, filtfilt, resample_poly
import torch
import torchaudio

def resample_audio(waveform: np.ndarray, original_hz: int, target_hz: int, order: int = 5) -> np.ndarray:
    nyquist_orig = original_hz / 2
    nyquist_target = target_hz / 2

    # Design a low-pass filter to avoid aliasing when down-sampling
    if original_hz > target_hz:
        cutoff_freq = nyquist_target / nyquist_orig
        b, a = butter(order, cutoff_freq)
    elif original_hz < target_hz:
        cutoff_freq = nyquist_orig / nyquist_target
        b, a = butter(order, cutoff_freq)

    waveform = filtfilt(b, a, waveform)
    waveform = resample_poly(waveform, target_hz, original_hz, axis=0)

    return waveform


def extract_spectrogram(
    waveform: np.ndarray,
    n_fft: int = 1024,
    win_length: t.Optional[int] = None,
    hop_length: t.Optional[int] = None,
) -> np.ndarray:
    """Extract spectrograms from an audio waveform.

    Args:
        audios (List[Audio]): List of Audio objects.
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 bins. Default is 1024.
        win_length (int): Window size. Default is None, using n_fft.
        hop_length (int): Length of hop between STFT windows. Default is None, using win_length // 2.

    Returns:
        List[Dict[str, torch.Tensor]]: List of Dict objects containing spectrograms.
    """
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = win_length // 2
    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
    )
    return spectrogram(torch.from_numpy(waveform)).squeeze(0).numpy()


def extract_mel_spectrogram(
    waveform: np.ndarray,
    sampling_frequency: int,
    n_fft: t.Optional[int] = 1024,
    win_length: t.Optional[int] = None,
    hop_length: t.Optional[int] = None,
    n_mels: int = 128,
) -> np.ndarray:
    """Extract mel spectrogram from an audio waveform.

    Args:
        waveform (np.ndarray): Audio waveform.
        sampling_frequency (int): Sampling frequency of the audio.
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 bins. Default is 1024.
        win_length (int): Window size. Default is None, using n_fft.
        hop_length (int): Length of hop between STFT windows. Default is None, using win_length // 2.
        n_mels (int): Number of mel filter banks. Default is 128.
    
    Returns:
        np.ndarray: Mel spectrogram.

    """
    if win_length is None:
        win_length = n_fft
    if hop_length is None:
        hop_length = win_length // 2

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sampling_frequency,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    
    return mel_spectrogram(torch.from_numpy(waveform)).squeeze(0).numpy()


def extract_mfcc(
    waveform: np.ndarray,
    sampling_frequency: int,
    n_mfcc: int = 40,
    n_ftt: t.Optional[int] = 400,
    win_length: t.Optional[int] = None,
    hop_length: t.Optional[int] = None,
    n_mels: int = 128,
) -> np.ndarray:
    """Extract MFCCs from a list of audio objects.

    Args:
        audios (List[Audio]): List of Audio objects.
        n_mfcc (int): Number of MFCCs to return. Default is 40.
        n_ftt (int): Size of FFT, creates n_ftt // 2 + 1 bins. Default is 400.
        win_length (int): Window size. Default is None, using n_ftt.
        hop_length (int): Length of hop between STFT windows. Default is None, using win_length // 2.
        n_mels (int): Number of mel filter banks. Default is 128.

    Returns:
        List[Dict[str, torch.Tensor]]: List of Dict objects containing MFCCs.
    """
    if win_length is None:
        win_length = n_ftt
    if hop_length is None:
        hop_length = win_length // 2

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=sampling_frequency,
        n_mfcc=n_mfcc,
        melkwargs={"n_fft": n_ftt, "win_length": win_length, "hop_length": hop_length, "n_mels": n_mels},
    )
    return mfcc_transform(torch.from_numpy(waveform)).squeeze(0).numpy()


def extract_mel_filter_bank(
    spectrogram: np.ndarray,
    sampling_frequency: int,
    n_mels: int = 128,
    n_stft: int = 201,
) -> np.ndarray:
    """Extract mel filter bank from a spectrogram.

    Args:
        spectrogram (np.ndarray): Spectrogram.
        sampling_frequency (int): Sampling frequency of the audio.
        n_mels (int): Number of mel filter banks. Default is 128.
        n_stft (int): Number of STFT bins. Default is 201.

    Returns:
        np.ndarray: Mel filter bank.
    """
    melscale_transform = torchaudio.transforms.MelScale(
        sample_rate=sampling_frequency, n_mels=n_mels, n_stft=n_stft
    )
    return melscale_transform(torch.from_numpy(spectrogram)).squeeze(0).numpy()


def extract_pitch_from_audios(
    waveform: np.ndarray, sampling_frequency: int, freq_low: int = 85, freq_high: int = 3400
) -> np.ndarray:
    """Extract pitch frequency from an audio waveform.

    Args:
        waveform (np.ndarray): Audio waveform.
        sampling_frequency (int): Sampling frequency of the audio.
        freq_low (int): Lowest frequency to detect. Default is 85.
        freq_high (int): Highest frequency to detect. Default is 3400.
    
    Returns:
        np.ndarray: Pitch frequency.
    """
    return torchaudio.functional.detect_pitch_frequency(
        waveform, sample_rate=sampling_frequency, freq_low=freq_low, freq_high=freq_high
    ).squeeze(0).numpy()

