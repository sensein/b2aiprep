import pytest
import logging
from b2aiprep.summer_school_data import (
    extract_features_workflow
)

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    parser.addoption(
        "--bids-files-path", 
        action="store", 
        default=None, 
        help="Path to BIDS files"
    )

@pytest.fixture
def bids_files_path(request):
    return request.config.getoption("--bids-files-path")


def test_wav_to_features_output():
    """Tests that the output is formatted correctly."""
    #TODO: implement
    assert False, "Not implemented."


def test_extract_features_coverage():
    """Tests that extract features extracted features for every audio file."""
    #TODO: implement
    assert False, "Not implemented."


def test_extract_features_timing(benchmark, caplog, bids_files_path):
    """Times the non-optimized iterative extract_features function."""
    result = benchmark(extract_features_workflow, bids_files_path)
    caplog.set_level(logging.INFO)
    _logger.info(str(result))
    return
