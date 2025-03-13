import platform
import sys

from setuptools import setup

import versioneer

# Make sure the system is ARM64 if on macOS
if platform.system() == "Darwin" and platform.machine() != "arm64":
    sys.stderr.write(
        "Error: This package requires an ARM64 architecture on macOS since pytorch 2.2.2+ does not support x86-64 on macOS\n"
    )
    sys.exit(1)

setup(version=versioneer.get_version(), cmdclass=versioneer.get_cmdclass())
