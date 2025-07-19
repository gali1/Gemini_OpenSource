"""
Gemini CLI - Open Source Command Line Interface
Interactive AI assistant powered by Gemini OpenSource PyTorch implementation
"""

__version__ = "1.0.0"
__author__ = "Gemini OpenSource Team"
__description__ = "Interactive AI assistant command line interface"

from .main import main
from .core.cli import GeminiCLI
from .core.config import ConfigManager
from .model_interface import GeminiModelInterface

__all__ = [
    "main",
    "GeminiCLI",
    "ConfigManager",
    "GeminiModelInterface",
    "__version__"
]