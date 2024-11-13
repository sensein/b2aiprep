import logging
import os

import pytest

from b2aiprep.prepare.prepare import extract_features_workflow

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    parser.addoption("--bids-files-path", action="store", default=None, help="Path to BIDS files")


@pytest.fixture
def bids_files_path(request):
    return request.config.getoption("--bids-files-path")


@pytest.mark.skipif(
    os.getenv("CI") == "true", reason="Skipping benchmarking test in CI environment"
)
def test_extract_features_timing(benchmark, caplog, bids_files_path):
    caplog.set_level(logging.INFO)
    if bids_files_path is None:
        _logger.error("Please provide the path to BIDS files using --bids-files-path")
        return
    result = benchmark(extract_features_workflow, bids_files_path)
    _logger.info(str(result))
    assert result is not None, "Benchmark failed"
    assert len(result["features"]) > 0, "No features extracted"
