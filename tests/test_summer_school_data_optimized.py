import time
import os
from pathlib import Path
import logging
from b2aiprep.summer_school_data_optimized import (
    extract_features_workflow
)


logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


class parse_arguments():
    def __init__(self) -> None:
        self.ftf_redcap_file_path=Path("/Users/isaacbevers/sensein/b2ai-wrapper/b2ai-data/bridge2ai-voice-corpus-1/bridge2ai_voice_data.csv")
        self.updated_redcap_file_path=Path("/Users/isaacbevers/sensein/b2ai-wrapper/b2ai-data/Bridge2AIDEVELOPMENT_DATA_2024-06-03_1705.csv")
        self.audio_dir_path=Path("/Users/isaacbevers/sensein/b2ai-wrapper/b2ai-data/audio-data")
        self.tar_file_path=Path("/Users/isaacbevers/sensein/b2ai-wrapper/b2ai-data/summer_school_data_test_bundle.tar")
        self.bids_files_path=Path("/Users/isaacbevers/sensein/b2ai-wrapper/b2ai-data-bids-like")


args = parse_arguments()


def test_wav_to_features_output():
    """Tests that the output is formatted correctly."""
    #TODO: implement
    assert "Not implemented."


def test_extract_features_coverage():
    """Tests that extract features extracted features for every audio file."""
    #TODO: implement
    assert "Not implemented."


def test_extract_features_timing(benchmark, caplog, ):
    """Times the non-optimized iterative extract_features function."""
    result = benchmark(extract_features_workflow, args.bids_files_path)
    caplog.set_level(logging.INFO)
    _logger.info(str(result))
    return
