from . import _version
import warnings
warnings.filterwarnings("ignore")

__version__ = _version.get_versions()["version"]
