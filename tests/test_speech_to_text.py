from pathlib import Path

import pytest

from b2aiprep.process import Audio, SpeechToText


def test_transcribe():
    """
    Validates SpeechToText's ability to convert audio to text accurately.
    Checks if the transcription matches the expected output, considering known model discrepancies.
    """
    audio_path = str((Path(__file__).parent.parent / "data/vc_source.wav").absolute())
    audio_content = "If it isn't, it isn't."

    # Note: Should be "If it didn't, it didn't.", but that's what the model understands
    audio = Audio.from_file(audio_path)

    speech_to_text = SpeechToText()
    result = speech_to_text.transcribe(audio)
    text = result["text"]
    assert text.strip() == audio_content


def test_cuda_not_available():
    """
    Test behavior when CUDA is not available.
    A ValueError should be raised.

    To raise the exception, use "CUDA_VISIBLE_DEVICES="" pytest my_tests.py"
    """
    with pytest.raises(ValueError) as e:
        SpeechToText(device="cuda")
    assert str(e.value) == "CUDA is not available. Please use CPU."
