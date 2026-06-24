"""Setup script for StreamSplit."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="streamsplit",
    version="1.0.0",
    author="Minh K. Quan, Pubudu N. Pathirana",
    description=(
        "StreamSplit: Continuous Audio Representation Learning via "
        "Uncertainty-Guided Adaptive Splitting (MobiSys '26)"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mk3658/StreamSplit-AAAI",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "soundfile>=0.12.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
        ],
        "edge": [
            "torch>=2.0.0",
            "torchaudio>=2.0.0",
            "numpy>=1.24.0",
            "scipy>=1.10.0",
            "soundfile>=0.12.0",
            "pyyaml>=6.0",
            "psutil>=5.9.0",
        ],
    },
)
