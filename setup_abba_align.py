"""
Setup script for ABBA-Align: Biblical Text Alignment Toolkit

This creates a standalone CLI tool for biblical text alignment
that can be used independently of the main ABBA project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="abba-align",
    version="1.0.0",
    author="ABBA Project Contributors",
    description="Advanced biblical text alignment toolkit with morphological and phrase analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/abba-align",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Religion",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Religion",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "click>=8.0.0",
        "rich>=10.0.0",  # For beautiful CLI output
        "tqdm>=4.62.0",  # For progress bars
        "jsonschema>=3.2.0",
        "lxml>=4.6.0",  # For XML parsing
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.9",
            "mypy>=0.910",
        ],
        "ml": [
            "scikit-learn>=0.24.0",
            "gensim>=4.0.0",  # For word embeddings
            "transformers>=4.0.0",  # For BERT-based models
            "torch>=1.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "abba-align=abba_align.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "abba_align": [
            "data/*.json",
            "schemas/*.json",
        ],
    },
)