#!/usr/bin/env python3
"""
Neuro-Symbolic Law Prover
Combines Graph Neural Networks with Z3 SMT solving for legal compliance verification.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neuro-symbolic-law-prover",
    version="0.1.0",
    author="Daniel Schmidt",
    author_email="daniel@terragonlabs.ai",
    description="Neuro-symbolic reasoning for automated legal compliance verification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/danieleschmidt/Photon-Neuromorphics-SDK",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Legal Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Other/Nonlisted Topic",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=1.12.0",
        "torch-geometric>=2.1.0",
        "transformers>=4.20.0",
        "z3-solver>=4.12.0",
        "spacy>=3.4.0",
        "networkx>=2.8",
        "numpy>=1.21.0",
        "pandas>=1.4.0",
        "scikit-learn>=1.1.0",
        "requests>=2.28.0",
        "beautifulsoup4>=4.11.0",
        "lxml>=4.9.0",
        "pydantic>=1.10.0",
        "jinja2>=3.1.0",
        "rich>=12.0.0",
        "click>=8.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.1.0",
            "pytest-cov>=3.0.0",
            "black>=22.3.0",
            "flake8>=4.0.1",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
            "jupyter>=1.0.0",
            "notebook>=6.4.0",
        ],
        "nlp": [
            "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.4.1/en_core_web_sm-3.4.1-py3-none-any.whl",
            "sentence-transformers>=2.2.0",
        ],
        "regulations": [
            "lxml>=4.9.0",
            "xmltodict>=0.13.0",
        ],
        "viz": [
            "matplotlib>=3.5.0",
            "plotly>=5.10.0",
            "graphviz>=0.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "neuro-law=neuro_symbolic_law.cli:main",
        ],
    },
)