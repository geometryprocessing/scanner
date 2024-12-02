from setuptools import setup, find_packages
from pathlib import Path

setup(
    name="scanner",
    version="0.0.1",
    authors="Giancarlo Pereira and Yidan Gao",
    author_email="giancarlo.pereira@nyu.edu",
    description="Python Library for 3D Scanning",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    keywords=[
        "scanner"
    ],
    url="https://github.com/geometryprocessing/scanner",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    test_suite="tests"
)