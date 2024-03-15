"""Functions to prepare acoustic data for the Bridge2AI voice project."""

import os
import typing as ty
from hashlib import md5
from pathlib import Path

import speechbrain.processing.features as spf
import torch
from scipy import signal
from speechbrain.augment.time_domain import Resample
from speechbrain.dataio.dataio import read_audio, read_audio_info
from speechbrain.inference.speaker import EncoderClassifier


class Audio:
    def __init__(self, signal: torch.tensor, sample_rate: int) -> None:
        """Initialize audio object
        :param signal: is a Torch tensor
        :param sample_rate: is a float
        """
        self.signal = signal
        self.sample_rate = int(sample_rate)

    @classmethod
    def from_file(cls, filename: Path, channel: int = 0) -> "Audio":
        """Load audio file

        If the file contains more than one channel of audio,
        only the first channel will be loaded.

        :param filename: Path to audio file
        :param channel: Channel to load (default: 0)
        :return: Initialized Audio object
        """
        signal = read_audio(filename)
        meta = read_audio_info(filename)
        if len(signal.shape) > 1 and signal.shape[-1] > 1:
            signal = signal[:, [channel]]
        else:
            signal = signal[:, None]
        return cls(signal, meta.sample_rate)

    def to_16khz(self) -> "Audio":
        """Resample audio to 16kHz and return new Audio object

        TODO: The default resampler does a poor job of taking care of
        aliasing. This should be replaced by a better antialiasing filter.
        """
        resampler = Resample(orig_freq=self.sample_rate, new_freq=16000)
        return Audio(resampler(self.signal.unsqueeze(0)).squeeze(0), 16000)


def embed_speaker(audio: Audio, model: str, device: ty.Optional[str] = None) -> torch.tensor:
    """Compute the speaker embedding of the audio signal"""
    classifier = EncoderClassifier.from_hparams(source=model, run_opts={"device": device})
    embeddings = classifier.encode_batch(audio.signal.T)
    return embeddings.squeeze()


def verify_speaker(
    audio1: Audio, audio2: Audio, model: str, device: ty.Optional[str] = None
) -> ty.Tuple[float, float]:
    from speechbrain.inference.speaker import SpeakerRecognition

    verification = SpeakerRecognition.from_hparams(source=model, run_opts={"device": device})
    score, prediction = verification.verify_batch(audio1.signal.T, audio2.signal.T)
    return score, prediction


def verify_speaker_from_files(
    file1: Path, file2: Path, model: str, device: ty.Optional[str] = None
) -> ty.Tuple[float, float]:
    audio1 = Audio.from_file(file1)
    audio2 = Audio.from_file(file2)
    return verify_speaker(audio1, audio2, model, device=device)


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


def resample_iir(audio: Audio, lowcut: float, new_sample_rate: int, order: int = 4) -> Audio:
    """Resample audio using IIR filter"""
    sos = signal.butter(order, lowcut, btype="low", output="sos", fs=new_sample_rate)
    filtered = torch.from_numpy(signal.sosfiltfilt(sos, audio.signal.squeeze()).copy()).float()
    resampler = Resample(orig_freq=audio.sample_rate, new_freq=new_sample_rate)
    return Audio(resampler(filtered.unsqueeze(0)).squeeze(0), new_sample_rate)


def to_features(
    filename: Path,
    subject: str,
    task: str,
    outdir: Path = Path(os.getcwd()),
    n_mels: int = 20,
    n_coeff: int = 20,
    compute_deltas: bool = True,
) -> ty.Tuple[dict, Path]:
    """Compute features from audio file

    :param filename: Path to audio file
    :param subject: Subject ID
    :param task: Task ID
    :param outdir: Output directory
    :param n_mels: Number of Mel bands
    :param n_coeff: Number of MFCC coefficients
    :param compute_deltas: Whether to compute delta features
    :return: Features dictionary
    :return: Path to features
    """
    with open(filename, "rb") as f:
        md5sum = md5(f.read()).hexdigest()
    audio = Audio.from_file(filename)
    audio = audio.to_16khz()
    features = specgram(audio)
    features_melfilterbank = melfilterbank(features, n_mels=n_mels)
    features_mfcc = MFCC(features_melfilterbank, n_coeff=n_coeff, compute_deltas=compute_deltas)
    features = {
        "specgram": features,
        "melfilterbank": features_melfilterbank,
        "mfcc": features_mfcc,
        "sample_rate": audio.sample_rate,
        "checksum": md5sum,
    }
    outfile = outdir / f"sub-{subject}_task-{task}_md5-{md5sum}_features.pt"
    torch.save(features, outfile)
    return features, outfile
