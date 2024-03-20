import unittest
from unittest.mock import patch, MagicMock
from b2aiprep.process import VoiceConversion
import os


# run 'python -m unittest discover -s tests' to run all tests
class TestVoiceConversion(unittest.TestCase):
    @patch("b2aiprep.process.VoiceConversion")
    def test_convert_voice(self, mock_voice_conversion):
        source_file = "../data/vc_source.wav"
        target_file = "../data/vc_target.wav"
        output_file = "../data/vc_output.wav"

        if os.path.exists(output_file):
            os.remove(output_file)

        vc = VoiceConversion()
        vc.convert_voice(
            source_file=source_file, target_file=target_file, output_file=output_file
        )

        self.assertTrue(os.path.exists(output_file), "File does not exist")
        os.remove(output_file)

    @patch("b2aiprep.process.VoiceConversion.__init__", MagicMock(return_value=None))
    @patch("os.path.exists")
    def test_source_file_not_exist(self, mock_exists):
        source_file = "../data/vc_source.wav"
        target_file = "../data/vc_target.wav"
        output_file = "../data/vc_output.wav"

        # This line configures the mock_exists mock. 
        # The side_effect attribute is set to a lambda function that takes one argument (x, the file path). 
        # The lambda function returns False if the string "source" is in x, 
        # simulating that the source file does not exist. 
        # For any other path (e.g., the target file path), it returns True, simulating that the file does exist. 
        mock_exists.side_effect = lambda x: False if "source" in x else True

        vc = VoiceConversion()
        with self.assertRaises(FileNotFoundError):
            vc.convert_voice(
                source_file=source_file, target_file=target_file, output_file=output_file
            )

    @patch("b2aiprep.process.VoiceConversion.__init__", MagicMock(return_value=None))
    @patch("os.path.exists")
    def test_target_file_not_exist(self, mock_exists):
        source_file = "../data/vc_source.wav"
        target_file = "../data/vc_target.wav"
        output_file = "../data/vc_output.wav"

        mock_exists.side_effect = lambda x: False if "target" in x else True

        vc = VoiceConversion()
        with self.assertRaises(FileNotFoundError):
            vc.convert_voice(
                source_file=source_file, target_file=target_file, output_file=output_file
            )
