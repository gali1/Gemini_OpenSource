"""
Exception classes for Gemini CLI
Defines hierarchy of exceptions used throughout the system
"""

from typing import Optional, Any, Dict


class GeminiCLIError(Exception):
    """Base exception class for all Gemini CLI errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class ConfigurationError(GeminiCLIError):
    """Raised when there are configuration-related errors."""
    pass


class ModelError(GeminiCLIError):
    """Raised when there are model-related errors."""
    pass


class ToolExecutionError(GeminiCLIError):
    """Raised when tool execution fails."""

    def __init__(self, message: str, tool_name: Optional[str] = None, tool_args: Optional[Dict[str, Any]] = None):
        details = {}
        if tool_name:
            details["tool_name"] = tool_name
        if tool_args:
            details["tool_args"] = tool_args

        super().__init__(message, details)
        self.tool_name = tool_name
        self.tool_args = tool_args


class FileOperationError(GeminiCLIError):
    """Raised when file operations fail."""

    def __init__(self, message: str, file_path: Optional[str] = None, operation: Optional[str] = None):
        details = {}
        if file_path:
            details["file_path"] = file_path
        if operation:
            details["operation"] = operation

        super().__init__(message, details)
        self.file_path = file_path
        self.operation = operation


class CheckpointError(GeminiCLIError):
    """Raised when checkpoint operations fail."""
    pass


class ExtensionError(GeminiCLIError):
    """Raised when extension operations fail."""

    def __init__(self, message: str, extension_name: Optional[str] = None):
        details = {}
        if extension_name:
            details["extension_name"] = extension_name

        super().__init__(message, details)
        self.extension_name = extension_name


class MemoryError(GeminiCLIError):
    """Raised when memory operations fail."""
    pass


class SessionError(GeminiCLIError):
    """Raised when session operations fail."""
    pass


class TelemetryError(GeminiCLIError):
    """Raised when telemetry operations fail."""
    pass


class SandboxError(GeminiCLIError):
    """Raised when sandbox operations fail."""
    pass


class AuthenticationError(GeminiCLIError):
    """Raised when authentication fails."""
    pass


class NetworkError(GeminiCLIError):
    """Raised when network operations fail."""

    def __init__(self, message: str, url: Optional[str] = None, status_code: Optional[int] = None):
        details = {}
        if url:
            details["url"] = url
        if status_code:
            details["status_code"] = status_code

        super().__init__(message, details)
        self.url = url
        self.status_code = status_code


class ValidationError(GeminiCLIError):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = value

        super().__init__(message, details)
        self.field = field
        self.value = value


class ResourceError(GeminiCLIError):
    """Raised when resource operations fail."""

    def __init__(self, message: str, resource_type: Optional[str] = None, resource_id: Optional[str] = None):
        details = {}
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id

        super().__init__(message, details)
        self.resource_type = resource_type
        self.resource_id = resource_id


class TimeoutError(GeminiCLIError):
    """Raised when operations timeout."""

    def __init__(self, message: str, timeout_seconds: Optional[float] = None):
        details = {}
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds

        super().__init__(message, details)
        self.timeout_seconds = timeout_seconds


class PermissionError(GeminiCLIError):
    """Raised when permission is denied."""

    def __init__(self, message: str, required_permission: Optional[str] = None):
        details = {}
        if required_permission:
            details["required_permission"] = required_permission

        super().__init__(message, details)
        self.required_permission = required_permission


class ParseError(GeminiCLIError):
    """Raised when parsing fails."""

    def __init__(self, message: str, input_data: Optional[str] = None, parser_type: Optional[str] = None):
        details = {}
        if input_data:
            details["input_data"] = input_data[:200] + "..." if len(input_data) > 200 else input_data
        if parser_type:
            details["parser_type"] = parser_type

        super().__init__(message, details)
        self.input_data = input_data
        self.parser_type = parser_type


class UnsupportedOperationError(GeminiCLIError):
    """Raised when an operation is not supported."""

    def __init__(self, message: str, operation: Optional[str] = None, reason: Optional[str] = None):
        details = {}
        if operation:
            details["operation"] = operation
        if reason:
            details["reason"] = reason

        super().__init__(message, details)
        self.operation = operation
        self.reason = reason


# Legacy aliases for backwards compatibility
ToolError = ToolExecutionError
FileError = FileOperationError