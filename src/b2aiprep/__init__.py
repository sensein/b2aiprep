from . import _version

__version__ = _version.get_versions()["version"]

from b2aiprep.prepare.prepare import redcap_to_bids

__all__ = ["redcap_to_bids", "__version__"]
