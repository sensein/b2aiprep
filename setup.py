from setuptools import setup

import versioneer

import warnings
warnings.filterwarnings("ignore")

setup(version=versioneer.get_version(), cmdclass=versioneer.get_cmdclass())
