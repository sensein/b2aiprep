"""The adult release's hint builder and file-name reader, for senselab's triage drivers (``--hints`` this directory).

senselab loads this file to build each recording's hint (``build_hint``) and, for its identity check, to read a
recording's identity off its BIDS-like file name (``identity_from_name``); senselab itself reads no file name.
"""

from b2aiprep.prepare.triage_hints import build_hint, identity_from_name  # noqa: F401
