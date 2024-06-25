"""Functions to prepare acoustic data for the Bridge2AI voice project."""

import os
import typing as ty
import warnings
from hashlib import md5
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import opensmile
import speechbrain.processing.features as spf
import torch
from datasets import Dataset
from scipy import signal
from speechbrain.augment.time_domain import Resample
from speechbrain.dataio.dataio import read_audio, read_audio_info
from speechbrain.inference.speaker import EncoderClassifier
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

warnings.filterwarnings("ignore")


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
    audio1: Audio, audio2: Audio, model: str, model_rate: int, device: ty.Optional[str] = None
) -> ty.Tuple[float, float]:
    from speechbrain.inference.speaker import SpeakerRecognition

    if model_rate != audio1.sample_rate:
        audio1 = resample_iir(audio1, model_rate / 2 - 100, model_rate)
    if model_rate != audio2.sample_rate:
        audio2 = resample_iir(audio2, model_rate / 2 - 100, model_rate)
    verification = SpeakerRecognition.from_hparams(source=model, run_opts={"device": device})
    score, prediction = verification.verify_batch(audio1.signal.T, audio2.signal.T)
    return score, prediction


def verify_speaker_from_files(
    file1: Path, file2: Path, model: str, device: ty.Optional[str] = None
) -> ty.Tuple[float, float]:
    from speechbrain.inference.speaker import SpeakerRecognition

    verification = SpeakerRecognition.from_hparams(source=model, run_opts={"device": device})
    return verification.verify_files(file1, file2)


def specgram(
    audio: Audio,
    n_fft: ty.Optional[int] = None,
    win_length: int = 20,
    hop_length: int = 10,
    toDb: bool = False,
) -> torch.tensor:
    """Compute the spectrogram using STFT of the audio signal

    :param audio: Audio object
    :param n_fft: FFT window size
    :param win_length: Window length (ms)
    :param hop_length: Hop length (ms)
    :param toDb: If True, return the log of the power of the spectrogram
    :return: Spectrogram
    """
    compute_STFT = spf.STFT(
        sample_rate=audio.sample_rate,
        win_length=win_length,
        hop_length=hop_length,
        n_fft=n_fft or int(win_length * audio.sample_rate / 1000),
    )
    stft = compute_STFT(audio.signal.unsqueeze(0))
    spec = spf.spectral_magnitude(stft.squeeze(), power=1)
    if toDb:
        log_spec = 10.0 * torch.log10(torch.maximum(spec, torch.tensor(1e-10)))
        log_spec = torch.maximum(log_spec, log_spec.max() - 80)
        return log_spec
    else:
        return spec


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


def extract_opensmile(
    audio: Audio,
    feature_set: str = "eGeMAPSv02",
    feature_level: str = "Functionals",
) -> torch.tensor:
    """Extract features using opensmile"""
    smile = opensmile.Smile(
        feature_set=getattr(opensmile.FeatureSet, feature_set),
        feature_level=getattr(opensmile.FeatureLevel, feature_level),
    )
    return smile.process_signal(audio.signal.squeeze(), audio.sample_rate)


def plot_waveform(waveform, sr, title="Waveform", ax=None):
    time_axis = torch.arange(0, len(waveform)) / sr

    if ax is None:
        _, ax = plt.subplots(1, 1)
    ax.plot(time_axis, waveform, linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.matshow(specgram, origin="lower", aspect="auto", **kwargs)  # , interpolation="nearest")


def plot_save_figure(audio, log_spec, prefix, outdir):
    duration = len(audio.signal) / audio.sample_rate
    win_length = duration
    # for large plots determine a balance between the number of suplots
    # and padding required.
    if duration > 20:
        nplots = 100
        for win_length_try in range(10, 20):
            nplots_try = int(torch.ceil(torch.tensor(duration) / win_length_try))
            if nplots_try < nplots:
                nplots = nplots_try
                win_length = win_length_try
    else:
        nplots = 1
    freq_steps = 4
    freq_ticks = torch.arange(0, log_spec.shape[0], log_spec.shape[0] // (freq_steps - 1))
    freq_axis = (
        (torch.round(freq_ticks * audio.sample_rate / 2 / log_spec.shape[0])).numpy().astype(int)
    )

    signal = audio.signal
    sr = audio.sample_rate
    # This factor is used to decimate the waveform, which provides
    # the biggest speedup.
    decimate_factor = min(4 ** (int(len(signal) // (win_length * sr))), 80)
    signal = signal[::decimate_factor]
    sr = sr // decimate_factor

    if nplots > 1:
        # pad signal and spectrogram to fill up plot
        signal_pad = (nplots * win_length * sr) - len(signal)
        signal = torch.nn.ZeroPad1d((0, signal_pad))(signal.T).T
        spec_pad = int((len(signal) / sr) / (duration / log_spec.shape[1])) - log_spec.shape[1]
        log_spec = torch.nn.ZeroPad2d((0, spec_pad))(log_spec)

    N = len(signal)
    ymax = torch.abs(signal).max()

    fig, axs = plt.subplots(
        nplots * 2,
        1,
        figsize=(8, 3 * nplots),
        sharex=True,
    )
    fig.subplots_adjust(hspace=0.0)
    for idx in range(nplots):
        if nplots > 1:
            waveform = signal[(N // nplots * idx) : (N // nplots * (idx + 1))]
            spec = log_spec[
                :, (log_spec.shape[1] // nplots * idx) : (log_spec.shape[1] // nplots * (idx + 1))
            ]
        else:
            waveform = signal
            spec = log_spec
        # Plot waveform
        ax = axs[2 * idx + 0]
        ax.set_ylim([-ymax, ymax])
        timestamps = torch.arange(0, len(waveform)) / len(waveform) * spec.shape[1]
        ax.plot(timestamps, waveform, linewidth=1)
        ax.grid(True)
        # Plot spectrogram
        ax = axs[2 * idx + 1]
        ax.matshow(spec, origin="lower", aspect="auto")
        ax.set_yticks(freq_ticks)
        ax.set_yticklabels(freq_axis)
        ax.set_ylabel("Freq (Hz)")
        if idx == nplots - 1:
            ax.xaxis.set_ticks_position("bottom")
            xticks = torch.arange(0, spec.shape[1], spec.shape[1] // win_length)
            xticklabels = torch.round(xticks / spec.shape[1] * win_length).numpy().astype(int)
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            ax.set_xlabel("Time (s)")
    fig.suptitle(f"Waveform and spectrogram of {prefix}")
    fig.tight_layout()
    outfig = outdir / f"{prefix}_specgram.png"
    fig.savefig(outfig, bbox_inches="tight")
    plt.close(fig)
    return outfig


def to_features(
    filename: Path,
    subject: ty.Optional[str] = None,
    task: ty.Optional[str] = None,
    outdir: ty.Optional[Path] = None,
    save_figures: bool = False,
    stt_kwargs: ty.Optional[ty.Dict] = None,
    extract_text: bool = False,
    win_length: int = 20,
    hop_length: int = 10,
    n_mels: int = 20,
    n_coeff: int = 20,
    compute_deltas: bool = True,
    opensmile_feature_set: str = "eGeMAPSv02",
    opensmile_feature_level: str = "Functionals",
    return_features: bool = False,
    mpl_backend: str = "Agg",
    device: ty.Optional[str] = None,
) -> ty.Tuple[ty.Dict, Path, ty.Optional[Path]]:
    """Compute features from audio file

    :param filename: Path to audio file
    :param subject: Subject ID
    :param task: Task ID
    :param outdir: Output directory
    :param save_figures: Whether to save figures
    :param extract_text: Whether to extract text
    :param win_length: Window length (ms)
    :param hop_length: Hop length (ms)
    :param stt_kwargs: Keyword arguments for SpeechToText
    :param n_mels: Number of Mel bands
    :param n_coeff: Number of MFCC coefficients
    :param compute_deltas: Whether to compute delta features
    :param opensmile_feature_set: OpenSmile feature set
    :param opensmile_feature_level: OpenSmile feature level
    :param return_features: Whether to return features
    :param mpl_backend: matplotlib backend
    :param device: Acceleration device (e.g. "cuda" or "cpu" or "mps")
    :return: Features dictionary
    :return: Path to features
    :return: Path to figures
    """
    if mpl_backend is not None:
        import matplotlib

        matplotlib.use(mpl_backend)
    with open(filename, "rb") as f:
        md5sum = md5(f.read()).hexdigest()
    audio = Audio.from_file(str(filename))
    audio = audio.to_16khz()
    # set window and hop length to the same to not allow for good Griffin Lim reconstruction
    features_specgram = specgram(audio, win_length=win_length, hop_length=hop_length)
    features_melfilterbank = melfilterbank(features_specgram, n_mels=n_mels)
    features_mfcc = MFCC(features_melfilterbank, n_coeff=n_coeff, compute_deltas=compute_deltas)
    features_opensmile = extract_opensmile(audio, opensmile_feature_set, opensmile_feature_level)

    features = {
        "specgram": features_specgram,
        "melfilterbank": features_melfilterbank,
        "mfcc": features_mfcc,
        "opensmile": features_opensmile,
        "sample_rate": audio.sample_rate,
        "checksum": md5sum,
    }
    if extract_text:
        # Select the best device available
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            device = "mps" if torch.backends.mps.is_available() else device
        stt_kwargs_default = {
            "model_id": "openai/whisper-base",
            "max_new_tokens": 128,
            "chunk_length_s": 30,
            "batch_size": 16,
            "device": device,
        }
        if stt_kwargs is not None:
            stt_kwargs_default.update(**stt_kwargs)
        stt_kwargs = stt_kwargs_default
        stt = SpeechToText(**stt_kwargs)
        transcription = stt.transcribe(audio, language="en")
        features["transcription"] = transcription

    if subject is not None:
        if task is not None:
            prefix = f"sub-{subject}_task-{task}_md5-{md5sum}"
        else:
            prefix = f"sub-{subject}_md5-{md5sum}"
    else:
        prefix = Path(filename).stem
    if outdir is None:
        outdir = Path(os.getcwd())
    outfile = outdir / f"{prefix}_features.pt"
    torch.save(features, outfile)

    outfig = None
    if save_figures:
        # save spectogram as figure
        """
        log_spec = specgram(audio,
                            win_length=20,
                            hop_length=10,
                            toDb=True)
        """
        log_spec = 10.0 * torch.log10(torch.maximum(features_specgram, torch.tensor(1e-10)))
        log_spec = torch.maximum(log_spec, log_spec.max() - 80)
        outfig = plot_save_figure(audio, log_spec.T, prefix, outdir)

    return features if return_features else None, outfile, outfig

class SpeechToText:
    """
    A class for converting speech to text using a specified speech-to-text model.
    """

    def __init__(
        self,
        model_id: str = "openai/whisper-tiny",
        max_new_tokens: int = 128,
        chunk_length_s: int = 30,
        batch_size: int = 16,
        return_timestamps: Union[bool, str] = False,
        device: ty.Optional[str] = None,
    ) -> None:

        torch_dtype = torch.float32
        if device is not None:
            if device == "cpu":
                pass
            elif device == "cuda" and torch.cuda.is_available():
                # If CUDA is available, set use_gpu to True
                torch_dtype = torch.float16
            elif device == "mps" and torch.backends.mps.is_available():
                torch_dtype = torch.float16
            # If CUDA is not available, raise an error
            else:
                raise ValueError("CUDA is not available. Please use CPU.")

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.device = device
        self.torch_dtype = torch_dtype

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            max_new_tokens=max_new_tokens,
            chunk_length_s=chunk_length_s,
            batch_size=batch_size,
            return_timestamps=return_timestamps,
            torch_dtype=torch_dtype,
            device=device,
        )

    def transcribe(self, audio: Audio, language: Optional[str] = None):
        """
        Transcribes the given audio input into text.

        Parameters:
        - audio (Audio): The audio input to transcribe.
        - language (Optional[str]): The language of the audio input. If not None, the transcription
          will attempt to use this language. If None, the model will use its default (mulilingual).

        Returns:
        - dict: The transcription result, which includes the transcribed text. If timestamps were
          requested during initialization, they are also included in the result.

        TODO:
        - Add checking if the specified language is supported by the model.
        - Consider the outlet of the transcript (e.g., file, console, return object).
        """
        audio = audio.to_16khz()
        if language is not None:
            return self.pipe(audio.signal.squeeze().numpy(), generate_kwargs={"language": language})
        return self.pipe(audio.signal.squeeze().numpy())


def to_hf_dataset(generator, outdir: Path) -> None:
    # Create a Hugging Face dataset from the data
    ds = Dataset.from_generator(generator)
    ds.to_parquet(outdir / "b2aivoice.parquet")
