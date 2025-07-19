#!/usr/bin/env python3
"""
Setup configuration for Gemini CLI
Comprehensive package setup with all dependencies and entry points
"""

from setuptools import setup, find_packages
import os
from pathlib import Path

# Read version from version.py
version_file = Path(__file__).parent / "gemini_cli" / "utils" / "version.py"
version_dict = {}
if version_file.exists():
    with open(version_file) as f:
        exec(f.read(), version_dict)
    version = version_dict.get("__version__", "1.0.0")
else:
    version = "1.0.0"

# Read long description from README
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    with open(readme_file, encoding="utf-8") as f:
        long_description = f.read()

# Core dependencies with proper version constraints
install_requires = [
    # Core async and HTTP
    "aiohttp>=3.8.0,<4.0.0",
    "aiofiles>=22.1.0,<24.0.0",
    "async-timeout>=4.0.0,<5.0.0; python_version<'3.11'",

    # PyTorch ecosystem
    "torch>=2.0.0,<3.0.0",
    "torchvision>=0.15.0,<1.0.0",
    "torchaudio>=2.0.0,<3.0.0",

    # ML and AI dependencies
    "einops>=0.6.0,<1.0.0",
    "transformers>=4.30.0,<5.0.0",
    "sentencepiece>=0.1.99,<1.0.0",
    "tokenizers>=0.13.0,<1.0.0",

    # CLI and interface
    "rich>=13.0.0,<14.0.0",
    "click>=8.0.0,<9.0.0",
    "prompt-toolkit>=3.0.0,<4.0.0",

    # Data processing
    "numpy>=1.21.0,<2.0.0",
    "pandas>=1.5.0,<3.0.0",
    "pillow>=9.0.0,<11.0.0",

    # Utilities
    "requests>=2.28.0,<3.0.0",
    "chardet>=5.0.0,<6.0.0",
    "python-magic>=0.4.27,<1.0.0",
    "psutil>=5.9.0,<6.0.0",

    # Configuration and serialization
    "pyyaml>=6.0.0,<7.0.0",
    "toml>=0.10.0,<1.0.0",
    "configparser>=5.3.0,<6.0.0",

    # Advanced features
    "zeta-torch>=2.0.0,<3.0.0",
    "ring-attention-pytorch>=0.1.0,<1.0.0",
]

# Optional dependencies for enhanced features
extras_require = {
    "dev": [
        "pytest>=7.0.0,<8.0.0",
        "pytest-asyncio>=0.21.0,<1.0.0",
        "pytest-cov>=4.0.0,<5.0.0",
        "black>=23.0.0,<24.0.0",
        "isort>=5.12.0,<6.0.0",
        "flake8>=6.0.0,<7.0.0",
        "mypy>=1.0.0,<2.0.0",
        "pre-commit>=3.0.0,<4.0.0",
    ],
    "docs": [
        "sphinx>=6.0.0,<8.0.0",
        "sphinx-rtd-theme>=1.2.0,<2.0.0",
        "myst-parser>=1.0.0,<3.0.0",
    ],
    "audio": [
        "librosa>=0.10.0,<1.0.0",
        "soundfile>=0.12.0,<1.0.0",
        "scipy>=1.10.0,<2.0.0",
    ],
    "vision": [
        "opencv-python>=4.7.0,<5.0.0",
        "matplotlib>=3.7.0,<4.0.0",
        "seaborn>=0.12.0,<1.0.0",
    ],
    "sandbox": [
        "docker>=6.0.0,<7.0.0",
        "kubernetes>=26.0.0,<28.0.0",
    ],
    "telemetry": [
        "opentelemetry-api>=1.17.0,<2.0.0",
        "opentelemetry-sdk>=1.17.0,<2.0.0",
        "opentelemetry-exporter-otlp>=1.17.0,<2.0.0",
    ],
    "full": [
        # Include all optional dependencies
    ],
}

# Add all optional dependencies to 'full'
extras_require["full"] = sum(extras_require.values(), [])

setup(
    name="gemini-cli",
    version=version,
    author="Gemini OpenSource Team",
    author_email="team@gemini-opensource.org",
    description="Interactive AI assistant command line interface powered by Gemini OpenSource",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kyegomez/Gemini",
    packages=find_packages(include=["gemini_cli*", "gemini_torch*"]),
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
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "gemini-cli=gemini_cli.main:main",
            "gcli=gemini_cli.main:main",
            "gemini=gemini_cli.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "gemini_cli": [
            "core/*.py",
            "utils/*.py",
            "data/*",
            "templates/*",
            "themes/*",
        ],
        "gemini_torch": [
            "*.py",
            "data/*",
        ],
    },
    zip_safe=False,
    keywords=[
        "ai", "cli", "assistant", "gemini", "chat", "llm", "pytorch",
        "multimodal", "interactive", "shell", "automation", "tools"
    ],
    project_urls={
        "Bug Reports": "https://github.com/kyegomez/Gemini/issues",
        "Source": "https://github.com/kyegomez/Gemini",
        "Documentation": "https://github.com/kyegomez/Gemini/blob/main/README.md",
    },
)