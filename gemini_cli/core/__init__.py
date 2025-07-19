"""
Core components for Gemini CLI
Contains the main classes and modules for CLI functionality
"""

from .cli import GeminiCLI
from .config import ConfigManager
from .display import DisplayManager
from .session import SessionManager
from .memory import MemoryManager
from .tools import ToolRegistry
from .extensions import ExtensionManager
from .file_discovery import FileDiscoveryService
from .checkpointing import CheckpointManager
from .telemetry import TelemetryManager
from .exceptions import (
    GeminiCLIError,
    ConfigurationError,
    ModelError,
    ToolExecutionError,
    FileOperationError,
    CheckpointError,
    ExtensionError,
    MemoryError,
    SessionError,
    TelemetryError,
    SandboxError,
    AuthenticationError,
    NetworkError,
    ValidationError,
    ResourceError,
    TimeoutError,
    PermissionError,
    ParseError,
    UnsupportedOperationError,
)

__all__ = [
    # Main classes
    "GeminiCLI",
    "ConfigManager",
    "DisplayManager",
    "SessionManager",
    "MemoryManager",
    "ToolRegistry",
    "ExtensionManager",
    "FileDiscoveryService",
    "CheckpointManager",
    "TelemetryManager",

    # Exceptions
    "GeminiCLIError",
    "ConfigurationError",
    "ModelError",
    "ToolExecutionError",
    "FileOperationError",
    "CheckpointError",
    "ExtensionError",
    "MemoryError",
    "SessionError",
    "TelemetryError",
    "SandboxError",
    "AuthenticationError",
    "NetworkError",
    "ValidationError",
    "ResourceError",
    "TimeoutError",
    "PermissionError",
    "ParseError",
    "UnsupportedOperationError",
]