#!/usr/bin/env python3
"""
Setup script for the Fantasy Baseball Draft Tool.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="fantasy-baseball-draft-tool",
    version="0.1.0",
    author="Alexander Hicks",
    author_email="your.email@example.com",
    description="A tool for fantasy baseball draft analysis and player forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/fantasy-baseball-draft-tool",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "fantasy-baseball=src.draft_tool.cli:cli",
        ],
    },
)
