import pytest

def pytest_addoption(parser):
    print("Loading conftest.py")
    parser.addoption(
        "--bids-files-path", action="store", default=None, help="Path to BIDS files"
    )

@pytest.fixture
def bids_files_path(request):
    return request.config.getoption("--bids-files-path")