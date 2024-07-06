"""Functions to prepare acoustic data for the Bridge2AI voice project."""
from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.speaker_embeddings import speechbrain
from senselab.audio.tasks.features_extraction.torchaudio import extract_spectrogram_from_audios
from senselab.audio.tasks.features_extraction.torchaudio import extract_mel_filter_bank_from_audios
from senselab.audio.tasks.features_extraction.torchaudio import extract_mfcc_from_audios
from senselab.audio.tasks.features_extraction.speaker_verification import verify_speaker
from senselab.audio.tasks.features_extraction.speaker_verification import verify_speaker_from_files
from senselab.audio.tasks.features_extraction import opensmile
from senselab.audio.tasks.plotting import plot_waveform
from senselab.audio.tasks.plotting import plot_specgram
from senselab.audio.tasks.speech_to_text.api import transcribe_audios



# def to_features(
#     filename: Path,
#     subject: ty.Optional[str] = None,
#     task: ty.Optional[str] = None,
#     outdir: ty.Optional[Path] = None,
#     save_figures: bool = False,
#     stt_kwargs: ty.Optional[ty.Dict] = None,
#     extract_text: bool = False,
#     win_length: int = 20,
#     hop_length: int = 10,
#     n_mels: int = 20,
#     n_coeff: int = 20,
#     compute_deltas: bool = True,
#     opensmile_feature_set: str = "eGeMAPSv02",
#     opensmile_feature_level: str = "Functionals",
#     return_features: bool = False,
#     mpl_backend: str = "Agg",
#     device: ty.Optional[str] = None,
# ) -> ty.Tuple[ty.Dict, Path, ty.Optional[Path]]:
#     """Compute features from audio file

#     :param filename: Path to audio file
#     :param subject: Subject ID
#     :param task: Task ID
#     :param outdir: Output directory
#     :param save_figures: Whether to save figures
#     :param extract_text: Whether to extract text
#     :param win_length: Window length (ms)
#     :param hop_length: Hop length (ms)
#     :param stt_kwargs: Keyword arguments for SpeechToText
#     :param n_mels: Number of Mel bands
#     :param n_coeff: Number of MFCC coefficients
#     :param compute_deltas: Whether to compute delta features
#     :param opensmile_feature_set: OpenSmile feature set
#     :param opensmile_feature_level: OpenSmile feature level
#     :param return_features: Whether to return features
#     :param mpl_backend: matplotlib backend
#     :param device: Acceleration device (e.g. "cuda" or "cpu" or "mps")
#     :return: Features dictionary
#     :return: Path to features
#     :return: Path to figures
#     """
#     if mpl_backend is not None:
#         import matplotlib

#         matplotlib.use(mpl_backend)
#     with open(filename, "rb") as f:
#         md5sum = md5(f.read()).hexdigest()
#     audio = Audio.from_file(str(filename))
#     audio = audio.to_16khz()
#     # set window and hop length to the same to not allow for good Griffin Lim reconstruction
#     features_specgram = specgram(audio, win_length=win_length, hop_length=hop_length)
#     features_melfilterbank = melfilterbank(features_specgram, n_mels=n_mels)
#     features_mfcc = MFCC(features_melfilterbank, n_coeff=n_coeff, compute_deltas=compute_deltas)
#     features_opensmile = extract_opensmile(audio, opensmile_feature_set, opensmile_feature_level)

#     features = {
#         "specgram": features_specgram,
#         "melfilterbank": features_melfilterbank,
#         "mfcc": features_mfcc,
#         "opensmile": features_opensmile,
#         "sample_rate": audio.sample_rate,
#         "checksum": md5sum,
#     }
#     if extract_text:
#         # Select the best device available
#         if device is None:
#             device = "cuda" if torch.cuda.is_available() else "cpu"
#             device = "mps" if torch.backends.mps.is_available() else device
#         stt_kwargs_default = {
#             "model_id": "openai/whisper-base",
#             "max_new_tokens": 128,
#             "chunk_length_s": 30,
#             "batch_size": 16,
#             "device": device,
#         }
#         if stt_kwargs is not None:
#             stt_kwargs_default.update(**stt_kwargs)
#         stt_kwargs = stt_kwargs_default
#         stt = SpeechToText(**stt_kwargs)
#         transcription = stt.transcribe(audio, language="en")
#         features["transcription"] = transcription

#     if subject is not None:
#         if task is not None:
#             prefix = f"sub-{subject}_task-{task}_md5-{md5sum}"
#         else:
#             prefix = f"sub-{subject}_md5-{md5sum}"
#     else:
#         prefix = Path(filename).stem
#     if outdir is None:
#         outdir = Path(os.getcwd())
#     outfile = outdir / f"{prefix}_features.pt"
#     torch.save(features, outfile)

#     outfig = None
#     if save_figures:
#         # save spectogram as figure
#         """
#         log_spec = specgram(audio,
#                             win_length=20,
#                             hop_length=10,
#                             toDb=True)
#         """
#         log_spec = 10.0 * torch.log10(torch.maximum(features_specgram, torch.tensor(1e-10)))
#         log_spec = torch.maximum(log_spec, log_spec.max() - 80)
#         outfig = plot_save_figure(audio, log_spec.T, prefix, outdir)

#     return features if return_features else None, outfile, outfig