"""
Setup script for Gemini CLI
Provides installation configuration for the command line interface
"""

from setuptools import setup, find_packages
from pathlib import Path
import sys

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read version from __init__.py
version_file = Path(__file__).parent / "gemini_cli" / "__init__.py"
version = "1.0.0"  # Default version
if version_file.exists():
    for line in version_file.read_text().splitlines():
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

# Python version check
if sys.version_info < (3, 8):
    sys.exit("Python 3.8 or higher is required")

# Base requirements (CPU-only versions)
base_requirements = [
    # Core dependencies
    "torch>=2.0.0+cpu",
    "torchvision>=0.15.0+cpu",
    "torchaudio>=2.0.0+cpu",

    # Model dependencies
    "einops>=0.7.0",
    "sentencepiece>=0.1.99",
    "transformers>=4.30.0",

    # CLI dependencies
    "rich>=13.0.0",
    "aiofiles>=23.0.0",
    "aiohttp>=3.8.0",
    "asyncio-mqtt>=0.16.0",

    # Utility dependencies
    "click>=8.0.0",
    "pydantic>=2.0.0",
    "pyyaml>=6.0",
    "toml>=0.10.0",
    "chardet>=5.0.0",
    "psutil>=5.9.0",

    # Optional but recommended
    "pillow>=9.0.0",
    "requests>=2.28.0",
    "packaging>=21.0",
]

# Extra requirements for different use cases
extra_requirements = {
    "dev": [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.0.0",
        "black>=23.0.0",
        "isort>=5.12.0",
        "flake8>=6.0.0",
        "mypy>=1.0.0",
        "pre-commit>=3.0.0",
    ],
    "test": [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.0.0",
        "pytest-mock>=3.10.0",
        "httpx>=0.24.0",
    ],
    "docs": [
        "sphinx>=6.0.0",
        "sphinx-rtd-theme>=1.2.0",
        "myst-parser>=2.0.0",
        "sphinx-autodoc-typehints>=1.23.0",
    ],
    "sandbox": [
        "docker>=6.0.0",
        "podman-py>=4.0.0",
    ],
    "all": [],  # Will be populated below
}

# Populate 'all' with all extra requirements
extra_requirements["all"] = list(set(
    req for extra_list in extra_requirements.values()
    for req in extra_list if req != []
))

setup(
    name="gemini-cli",
    version=version,
    description="Interactive AI assistant powered by Gemini OpenSource",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Gemini OpenSource Team",
    author_email="team@gemini-opensource.dev",
    url="https://github.com/kyegomez/Gemini",
    project_urls={
        "Homepage": "https://github.com/kyegomez/Gemini",
        "Documentation": "https://github.com/kyegomez/Gemini/docs",
        "Source": "https://github.com/kyegomez/Gemini",
        "Tracker": "https://github.com/kyegomez/Gemini/issues",
    },
    packages=find_packages(exclude=["tests", "tests.*", "docs", "examples"]),
    include_package_data=True,
    package_data={
        "gemini_cli": [
            "data/*.json",
            "themes/*.json",
            "templates/*.txt",
            "configs/*.yaml",
        ],
    },
    python_requires=">=3.8",
    install_requires=base_requirements,
    extras_require=extra_requirements,
    entry_points={
        "console_scripts": [
            "gemini-cli=gemini_cli.main:main",
            "gcli=gemini_cli.main:main",  # Short alias
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Shells",
        "Topic :: Utilities",
        "Environment :: Console",
        "Intended Audience :: End Users/Desktop",
    ],
    keywords=[
        "ai", "cli", "gemini", "assistant", "chatbot", "llm",
        "machine-learning", "natural-language-processing", "pytorch",
        "command-line", "interactive", "automation", "tools"
    ],
    license="MIT",
    zip_safe=False,
    platforms=["any"],

    # Additional metadata
    maintainer="Gemini OpenSource Team",
    maintainer_email="team@gemini-opensource.dev",

    # Dependency links for CPU-only torch
    dependency_links=[
        "https://download.pytorch.org/whl/cpu"
    ],

    # Options for different installation scenarios
    options={
        "bdist_wheel": {
            "universal": False,
        },
        "build_ext": {
            "inplace": True,
        },
    },
)