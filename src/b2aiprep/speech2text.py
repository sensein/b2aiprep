import typing as ty

# Transcribes speech to text using OpenAI's whisper model
def transcribe_audio_whisper(audio: Audio, model="base": str):
    """
    Transcribes audio to text using OpenAI's whisper model.

    Args:
        audio (Audio). Audio object.
        model (str): Model to use for transcription. Defaults to "base". 
            See https://github.com/openai/whisper/ for a list of all available models.
    
    Returns:
        Result of the transcription.
    """
    import whisper
    
    model = whisper.load_model(model)
    result = model.transcribe(audio.signal.squeeze())
    return result


# Transcribes speech to text using the whisperX model
def transcribe_audio_whisperx(audio: Audio, model="base": str, device="cuda": str, batch_size=16: int, 
        compute_type="float16": str, force_alignment=True: bool, diarize=False: bool, hf_token=None: ty.Optional[str]):
    """
    Transcribes audio to text using OpenAI's whisper model.

    Args:
        audio (audio): Audio object.
        model (str): Model to use for transcription. Defaults to "base". 
            See https://github.com/openai/whisper/ for a list of all available models.
        device (str): Device to use for computation. Defaults to "cuda".
        batch_size (int): Batch size for transcription. Defaults to 16.
        compute_type (str): Type of computation to use. Defaults to "float16".
            Change to "int8" if low on GPU mem (may reduce accuracy)
	force_alignment (bool): Whether or not to perform forced alignment of the speech-to-text output
	diarize (bool): Whether or not to assign speaker labels to the text
	hf_token (str): A Huggingface auth token, required to perform speaker diarization
    
    Returns:
        Result of the transcription.
    """
    import whisperx

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model(model, device, compute_type=compute_type)

    # audio = whisperx.load_audio(audio.signal.squeeze())
    audio = audio.signal.squeeze().numpy()
    result = model.transcribe(audio, batch_size=batch_size)

    if force_alignment:
	    # 2. Align whisper output
	    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
	    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    if diarize:
	    # 3. Assign speaker labels
	    diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)

	    # add min/max number of speakers if known
	    diarize_segments = diarize_model(audio)
	    # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

	    result = whisperx.assign_word_speakers(diarize_segments, result)
    return result

