"""Functions to prepare acoustic data for the Bridge2AI voice project."""

import os
import typing as ty
import warnings
from hashlib import md5
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import opensmile
import speechbrain.processing.features as spf
import torch
import torchaudio
from datasets import Dataset
from scipy import signal
from speechbrain.augment.time_domain import Resample
from speechbrain.dataio.dataio import read_audio, read_audio_info
from speechbrain.inference.speaker import EncoderClassifier
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

try:
    from TTS.api import TTS
except ImportError:
    TTS = None

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
    audio: Audio, n_fft: int = 512, win_length: int = 20, hop_length: int = 10, toDb: bool = False
) -> torch.tensor:
    """Compute the spectrogram using STFT of the audio signal

    :param audio: Audio object
    :param n_fft: FFT window size
    :param win_length: Window length (ms)
    :param hop_length: Hop length (ms)
    :param toDb: If True, return the log of the power of the spectrogram
    :return: Spectrogram
    """
    spectrogram = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        win_length=int(audio.sample_rate * win_length / 1000),
        hop_length=int(audio.sample_rate * hop_length / 1000),
        power=2 if toDb else 1,
    )
    spec = spectrogram(audio.signal.squeeze()).T
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


def to_features(
    filename: Path,
    subject: ty.Optional[str] = None,
    task: ty.Optional[str] = None,
    outdir: ty.Optional[Path] = None,
    save_figures: bool = False,
    n_mels: int = 20,
    n_coeff: int = 20,
    compute_deltas: bool = True,
    opensmile_feature_set: str = "eGeMAPSv02",
    opensmile_feature_level: str = "Functionals",
    return_features: bool = False,
) -> ty.Tuple[ty.Dict, Path, ty.Optional[Path]]:
    """Compute features from audio file

    :param filename: Path to audio file
    :param subject: Subject ID
    :param task: Task ID
    :param outdir: Output directory
    :param save_figures: Whether to save figures
    :param n_mels: Number of Mel bands
    :param n_coeff: Number of MFCC coefficients
    :param compute_deltas: Whether to compute delta features
    :param opensmile_feature_set: OpenSmile feature set
    :param opensmile_feature_level: OpenSmile feature level
    :return: Features dictionary
    :return: Path to features
    """
    with open(filename, "rb") as f:
        md5sum = md5(f.read()).hexdigest()
    audio = Audio.from_file(str(filename))
    audio = audio.to_16khz()
    features_specgram = specgram(audio)
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

        # general log spectrogram for image
        features_specgram_log = specgram(audio, toDb=True)

        fig, axs = plt.subplots(2, 1)
        plot_waveform(
            audio.signal, sr=audio.sample_rate, title=f"Original waveform: {prefix}", ax=axs[0]
        )
        plot_spectrogram(features_specgram_log.T, title="spectrogram", ax=axs[1])
        fig.tight_layout()

        outfig = outdir / f"{prefix}_specgram.png"
        fig.savefig(outfig, bbox_inches="tight")
        plt.close(fig)

    return features if return_features else None, outfile, outfig


if TTS is not None:

    class VoiceConversion:
        def __init__(
            self,
            model_name: str = "voice_conversion_models/multilingual/vctk/freevc24",
            progress_bar: bool = True,
            device: ty.Optional[str] = None,
        ) -> None:
            """
            Initialize the Voice Conversion model.

            :param model_name: Name of the model to be used for voice conversion.
            :param use_gpu: Boolean indicating whether to use GPU for model computation.

            TODO: Add support for multiple devices.
            """
            use_gpu = False
            if device is not None and "cuda" in device:
                # If CUDA is available, set use_gpu to True
                if torch.cuda.is_available():
                    use_gpu = True
                # If CUDA is not available, raise an error
                else:
                    raise ValueError("CUDA is not available. Please use CPU.")

            self.tts = TTS(model_name=model_name, progress_bar=progress_bar, gpu=use_gpu)

        def convert_voice(self, source_file: str, target_file: str, output_file: str) -> None:
            """
            Converts the voice from the source audio file to match the voice in
            the target audio file.

            :param source_file: Path to the source audio file.
            :param target_file: Path to the target audio file.
            :param output_file: Path where the converted audio file will be saved.
            """
            if not os.path.exists(source_file):
                raise FileNotFoundError(f"The source file {source_file} does not exist.")
            if not os.path.exists(target_file):
                raise FileNotFoundError(f"The target file {target_file} does not exist.")

            # Perform voice conversion without modifying the source or target audio data directly.
            with torch.no_grad():
                self.tts.voice_conversion_to_file(
                    source_wav=source_file, target_wav=target_file, file_path=output_file
                )


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
        if device is not None and "cuda" in device:
            # If CUDA is available, set use_gpu to True
            if torch.cuda.is_available():
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

    def transcribe(self, audio: Audio, language: str = None):
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
    # Save the dataset to a JSON file
    ds.save_to_disk(outdir)
