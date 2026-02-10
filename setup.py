"""Setup configuration for the Portfolio Risk Monitor package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh.readlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="portfolio-risk-monitor",
    version="1.0.0",
    author="Vishal Joshi",
    author_email="vishal.joshi@warwick.ac.uk",
    description=(
        "An ML system that predicts market downturns using "
        "technical indicators and NLP sentiment from financial news"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VishalKJ-ai/portfolio-risk-monitor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "risk-monitor=src.pipeline:main",
        ],
    },
)
