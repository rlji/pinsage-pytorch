from __future__ import absolute_import, unicode_literals
from setuptools import setup, find_packages

VERSION = '0.1.0'

NAME = 'pinsage-pytorch'
DESCRIPTION = 'This is a PinSage pytorch library.'
URL = 'https://github.com/rlji/pinsage-pytorch'
EMAIL = 'me@example.com'
AUTHOR = 'rlji'

# What python versions are supported?
REQUIRES_PYTHON = ">=3.6"

# What packages are required for this module to be executed?
REQUIRED = [
    'dgl', 'pandas',
    'dask[complete]',
    'torch', 'numpy',
    'scipy', 'tqdm',
]

# What packages are optional?
EXTRAS = {
}


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
)
