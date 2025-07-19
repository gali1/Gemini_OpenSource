"""
Utility modules for Gemini CLI
Contains helper functions and classes used throughout the system
"""

from .version import get_version, get_build_info
from .colors import Colors, ColorTheme
from .formatting import (
    format_file_size,
    format_duration,
    format_timestamp,
    format_memory_usage,
    format_token_count,
    truncate_text,
    format_json,
    format_code_block
)
from .file_utils import FileUtils
from .shell import ShellMode
from .history import HistoryManager
from .theme import ThemeManager
from .stats import StatsTracker
from .sandbox import SandboxExecutor

__all__ = [
    # Version info
    "get_version",
    "get_build_info",

    # Colors and themes
    "Colors",
    "ColorTheme",
    "ThemeManager",

    # Formatting
    "format_file_size",
    "format_duration",
    "format_timestamp",
    "format_memory_usage",
    "format_token_count",
    "truncate_text",
    "format_json",
    "format_code_block",

    # File utilities
    "FileUtils",

    # Shell and interaction
    "ShellMode",
    "HistoryManager",

    # Statistics and monitoring
    "StatsTracker",

    # Sandbox execution
    "SandboxExecutor",
]