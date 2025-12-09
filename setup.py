"""Setup script for Hypothesis Forge."""
from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="hypothesis-forge",
    version="1.0.0",
    description="AI-driven simulation engine for generating novel scientific hypotheses",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Joseph Jilovec",
    author_email="",
    url="https://github.com/josephjilovec/Hypothesis-Forge",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "torch>=2.0.0",
        "biopython>=1.81",
        "astropy>=5.3.0",
        "neo4j>=5.11.0",
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.1.0",
        "requests>=2.31.0",
        "feedparser>=6.0.10",
        "pyarrow>=12.0.0",
        "python-dotenv>=1.0.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "flake8>=6.1.0",
            "black>=23.9.0",
            "mypy>=1.6.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

