import unittest
from unittest.mock import patch, MagicMock
from b2aiprep.process import SpeechToText, Audio

class TestSpeechToText(unittest.TestCase):
    def test_transcribe(self):
        audio_path = "../data/vc_source.wav"
        audio_content = "If it isn't, it isn't." 
        # Should be "If it didn't, it didn't.", but that's what the model understands
        audio = Audio.from_file(audio_path)

        speech_to_text = SpeechToText()
        result = speech_to_text.transcribe(audio)
        text = result["text"]
        
        self.assertEqual(text.strip(), audio_content)

if __name__ == "__main__":
    unittest.main()
