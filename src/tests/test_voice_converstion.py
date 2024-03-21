import pytest
import os
from b2aiprep.process import VoiceConversion

class TestVoiceConversion:
    """
    This class contains tests for the VoiceConversion class.
    """

    def test_convert_voice(self):
        """
        Test voice conversion from a source file to an output file
        using a target voice file. It checks if the output file is created.
        """
        source_file = "../../data/vc_source.wav"
        target_file = "../../data/vc_target.wav"
        output_file = "../../data/vc_output.wav"

        # Ensure output file does not exist before running the test
        if os.path.exists(output_file):
            os.remove(output_file)

        vc = VoiceConversion()
        vc.convert_voice(
            source_file=source_file, target_file=target_file, output_file=output_file
        )

        # Check if the output file now exists
        assert os.path.exists(output_file) is True
        
        # Cleanup after test
        os.remove(output_file)

    def test_source_file_not_exist(self):
        """
        Test behavior when the source file does not exist.
        A FileNotFoundError should be raised.
        """
        source_file = "fake_vc_source.wav"
        target_file = "../../data/vc_target.wav"
        output_file = "../../data/vc_output.wav"

        vc = VoiceConversion()
        with pytest.raises(FileNotFoundError) as e:
            vc.convert_voice(
                source_file=source_file, target_file=target_file, output_file=output_file
            )
        assert str(e.value) == f"The source file {source_file} does not exist."
        
    def test_target_file_not_exist(self):
        """
        Test behavior when the target file does not exist.
        A FileNotFoundError should be raised.
        """
        source_file = "../../data/vc_source.wav"
        target_file = "fake_vc_target.wav"
        output_file = "../../data/vc_output.wav"

        vc = VoiceConversion()
        with pytest.raises(FileNotFoundError) as e:
            vc.convert_voice(
                source_file=source_file, target_file=target_file, output_file=output_file
            )
        assert str(e.value) == f"The target file {target_file} does not exist."

    def test_cuda_not_available(self):
        """
        Test behavior when CUDA is not available.
        A ValueError should be raised.

        To raise the exception, use "CUDA_VISIBLE_DEVICES="" pytest test_voice_converstion.py"
        """
        with pytest.raises(ValueError) as e:
            VoiceConversion(device="cuda")
        assert str(e.value) == "CUDA is not available. Please use CPU."
