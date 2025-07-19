"""
Tool registry and management for Gemini CLI
Handles registration, discovery, and execution of all tools
"""

import asyncio
import json
import subprocess
import shlex
import os
import glob
import mimetypes
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from pathlib import Path
from abc import ABC, abstractmethod
import logging
import time
import re
import tempfile
import shutil

import aiohttp
import aiofiles
import torch
from PIL import Image
import requests

from .exceptions import ToolExecutionError, GeminiCLIError
from ..utils.file_utils import FileUtils
from ..utils.sandbox import SandboxExecutor


class BaseTool(ABC):
    """Base class for all tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"tool.{name}")

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """Get the JSON schema for this tool."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        pass

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate tool parameters."""
        return True  # Override in subclasses as needed

    def should_confirm(self, params: Dict[str, Any]) -> bool:
        """Check if execution should be confirmed by user."""
        return False  # Override in subclasses as needed


class ReadFileTool(BaseTool):
    """Tool for reading file contents."""

    def __init__(self):
        super().__init__("read_file", "Read the contents of a file")
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.max_lines = 2000

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": "read_file",
            "description": "Read and return the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the file to read"
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (0-based)",
                        "minimum": 0
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read",
                        "minimum": 1
                    }
                },
                "required": ["path"]
            }
        }

    async def execute(self, path: str, offset: int = 0, limit: Optional[int] = None, **kwargs) -> str:
        """Read file contents."""
        try:
            file_path = Path(path)

            if not file_path.exists():
                raise ToolExecutionError(f"File not found: {path}")

            if not file_path.is_file():
                raise ToolExecutionError(f"Path is not a file: {path}")

            # Check file size
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                raise ToolExecutionError(f"File too large: {file_size} bytes (max: {self.max_file_size})")

            # Detect file type
            mime_type, _ = mimetypes.guess_type(str(file_path))

            # Handle different file types
            if mime_type and mime_type.startswith('image/'):
                return await self._read_image_file(file_path)
            elif mime_type == 'application/pdf':
                return await self._read_pdf_file(file_path)
            elif self._is_binary_file(file_path):
                return f"Cannot display content of binary file: {path}"
            else:
                return await self._read_text_file(file_path, offset, limit)

        except Exception as e:
            self.logger.error(f"Error reading file {path}: {e}")
            raise ToolExecutionError(f"Failed to read file: {e}")

    async def _read_text_file(self, file_path: Path, offset: int, limit: Optional[int]) -> str:
        """Read text file with optional line range."""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = await f.readlines()

            total_lines = len(lines)

            # Apply offset and limit
            if offset > 0:
                lines = lines[offset:]

            if limit:
                lines = lines[:limit]
            elif len(lines) > self.max_lines:
                lines = lines[:self.max_lines]
                truncated = True
            else:
                truncated = False

            content = ''.join(lines)

            # Add truncation notice if needed
            if truncated or (limit and len(lines) == limit):
                start_line = offset + 1
                end_line = offset + len(lines)
                notice = f"[File content truncated: showing lines {start_line}-{end_line} of {total_lines} total lines]\n"
                content = notice + content

            return content

        except UnicodeDecodeError:
            # Try with different encoding
            try:
                async with aiofiles.open(file_path, 'r', encoding='latin-1') as f:
                    content = await f.read()
                return content
            except Exception:
                raise ToolExecutionError("Could not decode file contents")

    async def _read_image_file(self, file_path: Path) -> Dict[str, Any]:
        """Read image file and return as base64."""
        try:
            import base64

            async with aiofiles.open(file_path, 'rb') as f:
                image_data = await f.read()

            # Encode as base64
            encoded_data = base64.b64encode(image_data).decode('utf-8')

            # Get MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))

            return {
                "inlineData": {
                    "mimeType": mime_type or "image/jpeg",
                    "data": encoded_data
                }
            }

        except Exception as e:
            raise ToolExecutionError(f"Failed to read image file: {e}")

    async def _read_pdf_file(self, file_path: Path) -> Dict[str, Any]:
        """Read PDF file and return as base64."""
        try:
            import base64

            async with aiofiles.open(file_path, 'rb') as f:
                pdf_data = await f.read()

            encoded_data = base64.b64encode(pdf_data).decode('utf-8')

            return {
                "inlineData": {
                    "mimeType": "application/pdf",
                    "data": encoded_data
                }
            }

        except Exception as e:
            raise ToolExecutionError(f"Failed to read PDF file: {e}")

    def _is_binary_file(self, file_path: Path) -> bool:
        """Check if file is binary."""
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(8192)
                return b'\x00' in chunk
        except Exception:
            return True


class WriteFileTool(BaseTool):
    """Tool for writing file contents."""

    def __init__(self):
        super().__init__("write_file", "Write content to a file")

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": "write_file",
            "description": "Write content to a file, creating directories if needed",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["file_path", "content"]
            }
        }

    def should_confirm(self, params: Dict[str, Any]) -> bool:
        """File writing should be confirmed."""
        return True

    async def execute(self, file_path: str, content: str, **kwargs) -> str:
        """Write content to file."""
        try:
            path = Path(file_path)

            # Create parent directories if needed
            path.parent.mkdir(parents=True, exist_ok=True)

            # Check if file exists
            file_existed = path.exists()

            # Write content
            async with aiofiles.open(path, 'w', encoding='utf-8') as f:
                await f.write(content)

            if file_existed:
                return f"Successfully overwrote file: {file_path}"
            else:
                return f"Successfully created and wrote to new file: {file_path}"

        except Exception as e:
            self.logger.error(f"Error writing file {file_path}: {e}")
            raise ToolExecutionError(f"Failed to write file: {e}")


class ListDirectoryTool(BaseTool):
    """Tool for listing directory contents."""

    def __init__(self):
        super().__init__("list_directory", "List the contents of a directory")

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": "list_directory",
            "description": "List files and directories in a given path",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to the directory to list"
                    },
                    "ignore": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Glob patterns to ignore"
                    },
                    "respect_git_ignore": {
                        "type": "boolean",
                        "description": "Whether to respect .gitignore patterns",
                        "default": True
                    }
                },
                "required": ["path"]
            }
        }

    async def execute(self, path: str, ignore: Optional[List[str]] = None, respect_git_ignore: bool = True, **kwargs) -> str:
        """List directory contents."""
        try:
            dir_path = Path(path)

            if not dir_path.exists():
                raise ToolExecutionError(f"Directory not found: {path}")

            if not dir_path.is_dir():
                raise ToolExecutionError(f"Path is not a directory: {path}")

            # Get directory contents
            items = list(dir_path.iterdir())

            # Apply gitignore filtering
            if respect_git_ignore:
                items = self._filter_gitignore(items, dir_path)

            # Apply ignore patterns
            if ignore:
                items = self._filter_ignore_patterns(items, ignore)

            # Sort: directories first, then alphabetically
            items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))

            # Format output
            result_lines = [f"Directory listing for {path}:"]

            for item in items:
                if item.is_dir():
                    result_lines.append(f"[DIR] {item.name}")
                else:
                    size = item.stat().st_size
                    result_lines.append(f"{item.name} ({self._format_size(size)})")

            return '\n'.join(result_lines)

        except Exception as e:
            self.logger.error(f"Error listing directory {path}: {e}")
            raise ToolExecutionError(f"Failed to list directory: {e}")

    def _filter_gitignore(self, items: List[Path], base_path: Path) -> List[Path]:
        """Filter items based on .gitignore patterns."""
        gitignore_path = base_path / ".gitignore"
        if not gitignore_path.exists():
            return items

        try:
            with open(gitignore_path, 'r') as f:
                patterns = [line.strip() for line in f if line.strip() and not line.startswith('#')]

            filtered_items = []
            for item in items:
                if not self._matches_gitignore_patterns(item.name, patterns):
                    filtered_items.append(item)

            return filtered_items

        except Exception:
            return items

    def _matches_gitignore_patterns(self, name: str, patterns: List[str]) -> bool:
        """Check if name matches any gitignore pattern."""
        import fnmatch

        for pattern in patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False

    def _filter_ignore_patterns(self, items: List[Path], patterns: List[str]) -> List[Path]:
        """Filter items based on ignore patterns."""
        import fnmatch

        filtered_items = []
        for item in items:
            should_ignore = False
            for pattern in patterns:
                if fnmatch.fnmatch(item.name, pattern):
                    should_ignore = True
                    break

            if not should_ignore:
                filtered_items.append(item)

        return filtered_items

    def _format_size(self, size: int) -> str:
        """Format file size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"


class ShellTool(BaseTool):
    """Tool for executing shell commands."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("run_shell_command", "Execute shell commands")
        self.config = config
        self.sandbox_executor = SandboxExecutor(config) if config.get("sandbox", {}).get("enabled") else None

        # Load command restrictions
        self.allowed_commands = self._parse_command_restrictions(config.get("coreTools", []))
        self.blocked_commands = self._parse_command_restrictions(config.get("excludeTools", []))

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": "run_shell_command",
            "description": "Execute a shell command and return the result",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute"
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief description of what the command does"
                    },
                    "directory": {
                        "type": "string",
                        "description": "Directory to execute the command in (relative to project root)"
                    }
                },
                "required": ["command"]
            }
        }

    def should_confirm(self, params: Dict[str, Any]) -> bool:
        """Shell commands should usually be confirmed."""
        return not self.config.get("autoAccept", False)

    def _parse_command_restrictions(self, tools: List[str]) -> List[str]:
        """Parse command restrictions from tool configuration."""
        commands = []
        for tool in tools:
            if tool.startswith("run_shell_command(") and tool.endswith(")"):
                # Extract command from run_shell_command(command)
                command = tool[18:-1]  # Remove "run_shell_command(" and ")"
                commands.append(command)
            elif tool == "run_shell_command":
                # Wildcard - allow all commands
                commands.append("*")
        return commands

    def _is_command_allowed(self, command: str) -> bool:
        """Check if command is allowed based on restrictions."""
        # Check blocked commands first
        for blocked in self.blocked_commands:
            if blocked == "*" or command.startswith(blocked):
                return False

        # If no allowed commands specified, allow all (except blocked)
        if not self.allowed_commands:
            return True

        # Check if command matches any allowed pattern
        for allowed in self.allowed_commands:
            if allowed == "*" or command.startswith(allowed):
                return True

        return False

    async def execute(self, command: str, description: Optional[str] = None, directory: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Execute shell command."""
        try:
            # Validate command
            if not self._is_command_allowed(command):
                raise ToolExecutionError(f"Command not allowed: {command}")

            # Split command chains and validate each part
            command_parts = self._split_command_chain(command)
            for part in command_parts:
                if not self._is_command_allowed(part.strip()):
                    raise ToolExecutionError(f"Command chain contains blocked command: {part}")

            # Determine working directory
            if directory:
                work_dir = Path.cwd() / directory
                if not work_dir.exists():
                    raise ToolExecutionError(f"Directory not found: {directory}")
            else:
                work_dir = Path.cwd()

            # Execute command
            if self.sandbox_executor:
                result = await self.sandbox_executor.execute(command, work_dir)
            else:
                result = await self._execute_local(command, work_dir)

            return result

        except Exception as e:
            self.logger.error(f"Error executing command '{command}': {e}")
            raise ToolExecutionError(f"Command execution failed: {e}")

    def _split_command_chain(self, command: str) -> List[str]:
        """Split command chains on &&, ||, and ;"""
        # Simple splitting - could be enhanced for more complex parsing
        parts = []
        current = ""
        i = 0

        while i < len(command):
            if i < len(command) - 1:
                two_char = command[i:i+2]
                if two_char in ["&&", "||"]:
                    if current.strip():
                        parts.append(current.strip())
                    current = ""
                    i += 2
                    continue

            if command[i] == ';':
                if current.strip():
                    parts.append(current.strip())
                current = ""
            else:
                current += command[i]

            i += 1

        if current.strip():
            parts.append(current.strip())

        return parts

    async def _execute_local(self, command: str, work_dir: Path) -> Dict[str, Any]:
        """Execute command locally."""
        start_time = time.time()

        try:
            # Handle background processes
            background_pids = []
            if command.strip().endswith('&'):
                # Background process
                command = command.strip()[:-1].strip()

                process = await asyncio.create_subprocess_shell(
                    command,
                    cwd=work_dir,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True
                )

                background_pids.append(process.pid)

                return {
                    "command": command,
                    "directory": str(work_dir),
                    "stdout": "",
                    "stderr": "",
                    "exit_code": 0,
                    "execution_time": time.time() - start_time,
                    "background_pids": background_pids
                }

            # Regular command execution
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=work_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            return {
                "command": command,
                "directory": str(work_dir),
                "stdout": stdout.decode('utf-8', errors='replace'),
                "stderr": stderr.decode('utf-8', errors='replace'),
                "exit_code": process.returncode,
                "execution_time": time.time() - start_time,
                "background_pids": background_pids
            }

        except Exception as e:
            return {
                "command": command,
                "directory": str(work_dir),
                "stdout": "",
                "stderr": str(e),
                "exit_code": -1,
                "execution_time": time.time() - start_time,
                "error": str(e)
            }


class GlobTool(BaseTool):
    """Tool for finding files using glob patterns."""

    def __init__(self):
        super().__init__("glob", "Find files matching glob patterns")

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": "glob",
            "description": "Find files matching a glob pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern to match files"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default: current directory)"
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether matching should be case sensitive",
                        "default": False
                    },
                    "respect_git_ignore": {
                        "type": "boolean",
                        "description": "Whether to respect .gitignore patterns",
                        "default": True
                    }
                },
                "required": ["pattern"]
            }
        }

    async def execute(self, pattern: str, path: Optional[str] = None, case_sensitive: bool = False, respect_git_ignore: bool = True, **kwargs) -> str:
        """Find files matching glob pattern."""
        try:
            base_path = Path(path) if path else Path.cwd()

            if not base_path.exists():
                raise ToolExecutionError(f"Search path not found: {path}")

            # Use glob to find files
            if base_path.is_dir():
                search_pattern = str(base_path / pattern)
            else:
                search_pattern = pattern

            matches = glob.glob(search_pattern, recursive=True)

            # Convert to Path objects and filter
            file_paths = [Path(match) for match in matches if Path(match).is_file()]

            # Apply gitignore filtering
            if respect_git_ignore:
                file_paths = self._filter_gitignore(file_paths)

            # Sort by modification time (newest first)
            file_paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)

            if not file_paths:
                return f"No files found matching pattern: {pattern}"

            result_lines = [f"Found {len(file_paths)} file(s) matching \"{pattern}\" within {base_path}, sorted by modification time (newest first):"]
            result_lines.extend(str(p) for p in file_paths)

            return '\n'.join(result_lines)

        except Exception as e:
            self.logger.error(f"Error finding files with pattern '{pattern}': {e}")
            raise ToolExecutionError(f"File search failed: {e}")

    def _filter_gitignore(self, paths: List[Path]) -> List[Path]:
        """Filter paths based on gitignore patterns."""
        # Simple gitignore filtering - could be enhanced
        filtered = []
        common_ignores = {
            'node_modules', '.git', '.vscode', '.idea', '__pycache__',
            '.pytest_cache', '.coverage', 'dist', 'build', '.env'
        }

        for path in paths:
            # Check if any part of the path should be ignored
            parts = path.parts
            if not any(part in common_ignores for part in parts):
                filtered.append(path)

        return filtered


class SearchTextTool(BaseTool):
    """Tool for searching text content in files."""

    def __init__(self):
        super().__init__("search_file_content", "Search for text patterns in files")

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": "search_file_content",
            "description": "Search for a regex pattern within file contents",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression pattern to search for"
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in (default: current directory)"
                    },
                    "include": {
                        "type": "string",
                        "description": "Glob pattern for files to include in search"
                    }
                },
                "required": ["pattern"]
            }
        }

    async def execute(self, pattern: str, path: Optional[str] = None, include: Optional[str] = None, **kwargs) -> str:
        """Search for pattern in file contents."""
        try:
            base_path = Path(path) if path else Path.cwd()

            if not base_path.exists():
                raise ToolExecutionError(f"Search path not found: {path}")

            # Try git grep first for better performance
            if (base_path / ".git").exists():
                result = await self._git_grep(pattern, base_path, include)
                if result:
                    return result

            # Fallback to manual search
            return await self._manual_search(pattern, base_path, include)

        except Exception as e:
            self.logger.error(f"Error searching for pattern '{pattern}': {e}")
            raise ToolExecutionError(f"Text search failed: {e}")

    async def _git_grep(self, pattern: str, base_path: Path, include: Optional[str]) -> Optional[str]:
        """Use git grep for searching."""
        try:
            cmd = ["git", "grep", "-n", "-i", "--", pattern]
            if include:
                cmd.extend(["--", include])

            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=base_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                return self._format_grep_output(stdout.decode('utf-8'), pattern)

            return None

        except Exception:
            return None

    async def _manual_search(self, pattern: str, base_path: Path, include: Optional[str]) -> str:
        """Manual file search."""
        import fnmatch

        matches = []
        pattern_re = re.compile(pattern, re.IGNORECASE)

        # Get files to search
        if include:
            search_pattern = str(base_path / include)
            files = [Path(f) for f in glob.glob(search_pattern, recursive=True) if Path(f).is_file()]
        else:
            files = [f for f in base_path.rglob("*") if f.is_file() and self._should_search_file(f)]

        for file_path in files:
            try:
                async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = await f.readlines()

                for line_num, line in enumerate(lines, 1):
                    if pattern_re.search(line):
                        rel_path = file_path.relative_to(base_path)
                        matches.append(f"{rel_path}:{line_num}:{line.strip()}")

            except Exception:
                continue

        if not matches:
            return f"No matches found for pattern '{pattern}'"

        return self._format_search_results(matches, pattern, str(base_path), include)

    def _should_search_file(self, file_path: Path) -> bool:
        """Check if file should be searched."""
        # Skip binary files and common ignores
        if file_path.suffix.lower() in {'.pyc', '.pyo', '.so', '.dll', '.exe', '.bin'}:
            return False

        if any(part.startswith('.') for part in file_path.parts):
            return False

        try:
            # Quick binary check
            with open(file_path, 'rb') as f:
                chunk = f.read(512)
                if b'\x00' in chunk:
                    return False
        except Exception:
            return False

        return True

    def _format_grep_output(self, output: str, pattern: str) -> str:
        """Format git grep output."""
        lines = output.strip().split('\n')
        if not lines or not lines[0]:
            return f"No matches found for pattern '{pattern}'"

        return f"Found {len(lines)} matches for pattern '{pattern}':\n---\n" + '\n'.join(lines) + "\n---"

    def _format_search_results(self, matches: List[str], pattern: str, path: str, include: Optional[str]) -> str:
        """Format search results."""
        result = [f"Found {len(matches)} matches for pattern \"{pattern}\" in path \"{path}\""]
        if include:
            result[0] += f" (filter: \"{include}\")"
        result[0] += ":"
        result.append("---")

        # Group by file
        current_file = None
        for match in matches:
            parts = match.split(':', 2)
            if len(parts) >= 3:
                file_path, line_num, content = parts

                if file_path != current_file:
                    if current_file is not None:
                        result.append("---")
                    result.append(f"File: {file_path}")
                    current_file = file_path

                result.append(f"L{line_num}: {content}")

        result.append("---")
        return '\n'.join(result)


class EditTool(BaseTool):
    """Tool for editing files by replacing text."""

    def __init__(self):
        super().__init__("replace", "Replace text in a file")

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": "replace",
            "description": "Replace text in a file with new content",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Absolute path to the file to edit"
                    },
                    "old_string": {
                        "type": "string",
                        "description": "Exact text to replace (must be unique and include context)"
                    },
                    "new_string": {
                        "type": "string",
                        "description": "New text to replace with"
                    },
                    "expected_replacements": {
                        "type": "integer",
                        "description": "Expected number of replacements (default: 1)",
                        "default": 1
                    }
                },
                "required": ["file_path", "old_string", "new_string"]
            }
        }

    def should_confirm(self, params: Dict[str, Any]) -> bool:
        """File editing should be confirmed."""
        return True

    async def execute(self, file_path: str, old_string: str, new_string: str, expected_replacements: int = 1, **kwargs) -> str:
        """Replace text in file."""
        try:
            path = Path(file_path)

            # Handle new file creation
            if not old_string and not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(path, 'w', encoding='utf-8') as f:
                    await f.write(new_string)
                return f"Created new file: {file_path} with provided content."

            if not path.exists():
                raise ToolExecutionError(f"File not found: {file_path}")

            # Read current content
            async with aiofiles.open(path, 'r', encoding='utf-8') as f:
                content = await f.read()

            # Count occurrences
            occurrences = content.count(old_string)

            if occurrences == 0:
                raise ToolExecutionError(f"String not found in file: '{old_string[:50]}...'")

            if occurrences != expected_replacements:
                raise ToolExecutionError(f"Expected {expected_replacements} occurrences but found {occurrences}")

            # Perform replacement
            new_content = content.replace(old_string, new_string)

            # Write back to file
            async with aiofiles.open(path, 'w', encoding='utf-8') as f:
                await f.write(new_content)

            return f"Successfully modified file: {file_path} ({occurrences} replacements)."

        except Exception as e:
            self.logger.error(f"Error editing file {file_path}: {e}")
            raise ToolExecutionError(f"File edit failed: {e}")


class ReadManyFilesTool(BaseTool):
    """Tool for reading multiple files at once."""

    def __init__(self):
        super().__init__("read_many_files", "Read content from multiple files")
        self.max_total_size = 50 * 1024 * 1024  # 50MB total

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": "read_many_files",
            "description": "Read content from multiple files specified by paths or glob patterns",
            "parameters": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of file paths or glob patterns"
                    },
                    "exclude": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Glob patterns to exclude"
                    },
                    "include": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Additional glob patterns to include"
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Whether to search recursively",
                        "default": True
                    },
                    "useDefaultExcludes": {
                        "type": "boolean",
                        "description": "Whether to apply default exclusion patterns",
                        "default": True
                    },
                    "respect_git_ignore": {
                        "type": "boolean",
                        "description": "Whether to respect .gitignore patterns",
                        "default": True
                    }
                },
                "required": ["paths"]
            }
        }

    async def execute(self, paths: List[str], exclude: Optional[List[str]] = None, include: Optional[List[str]] = None, recursive: bool = True, useDefaultExcludes: bool = True, respect_git_ignore: bool = True, **kwargs) -> str:
        """Read multiple files."""
        try:
            all_patterns = paths + (include or [])
            exclude_patterns = exclude or []

            if useDefaultExcludes:
                exclude_patterns.extend([
                    "node_modules/**",
                    ".git/**",
                    "**/*.pyc",
                    "**/*.pyo",
                    "**/*.so",
                    "**/*.dll",
                    "**/*.exe",
                    "dist/**",
                    "build/**",
                    ".env"
                ])

            # Collect all files
            files = []
            for pattern in all_patterns:
                matches = glob.glob(pattern, recursive=recursive)
                files.extend(Path(match) for match in matches if Path(match).is_file())

            # Remove duplicates
            files = list(set(files))

            # Apply exclusions
            files = self._filter_files(files, exclude_patterns, respect_git_ignore)

            if not files:
                return "No files found matching the specified patterns."

            # Read files
            contents = []
            total_size = 0

            for file_path in sorted(files):
                try:
                    # Check file size
                    file_size = file_path.stat().st_size
                    if total_size + file_size > self.max_total_size:
                        contents.append(f"--- {file_path} ---\n[File skipped: would exceed size limit]")
                        continue

                    # Check if it's a supported file type
                    if self._is_supported_file(file_path):
                        async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            content = await f.read()
                        contents.append(f"--- {file_path} ---\n{content}")
                        total_size += file_size
                    else:
                        contents.append(f"--- {file_path} ---\n[Binary file or unsupported format]")

                except Exception as e:
                    contents.append(f"--- {file_path} ---\n[Error reading file: {e}]")

            if not contents:
                return "No files could be read."

            return '\n\n'.join(contents)

        except Exception as e:
            self.logger.error(f"Error reading multiple files: {e}")
            raise ToolExecutionError(f"Failed to read files: {e}")

    def _filter_files(self, files: List[Path], exclude_patterns: List[str], respect_git_ignore: bool) -> List[Path]:
        """Filter files based on exclusion patterns."""
        import fnmatch

        filtered = []
        for file_path in files:
            # Check exclude patterns
            should_exclude = False
            for pattern in exclude_patterns:
                if fnmatch.fnmatch(str(file_path), pattern) or fnmatch.fnmatch(file_path.name, pattern):
                    should_exclude = True
                    break

            if not should_exclude:
                filtered.append(file_path)

        return filtered

    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if file type is supported for text reading."""
        # Check for binary files
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(512)
                if b'\x00' in chunk:
                    return False
        except Exception:
            return False

        # Check file extensions
        text_extensions = {
            '.txt', '.md', '.py', '.js', '.ts', '.html', '.css', '.json', '.xml',
            '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.sh', '.bat',
            '.sql', '.go', '.rs', '.java', '.c', '.cpp', '.h', '.hpp', '.cs',
            '.php', '.rb', '.pl', '.r', '.scala', '.kt', '.swift', '.dart'
        }

        return file_path.suffix.lower() in text_extensions or not file_path.suffix


class WebFetchTool(BaseTool):
    """Tool for fetching web content."""

    def __init__(self):
        super().__init__("web_fetch", "Fetch content from URLs")

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": "web_fetch",
            "description": "Fetch and process content from web URLs",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Prompt containing URLs and instructions for processing"
                    }
                },
                "required": ["prompt"]
            }
        }

    def should_confirm(self, params: Dict[str, Any]) -> bool:
        """Web fetching should be confirmed."""
        return True

    async def execute(self, prompt: str, **kwargs) -> str:
        """Fetch content from URLs mentioned in prompt."""
        try:
            # Extract URLs from prompt
            urls = self._extract_urls(prompt)

            if not urls:
                raise ToolExecutionError("No URLs found in prompt")

            if len(urls) > 20:
                raise ToolExecutionError("Too many URLs (max: 20)")

            # Fetch content from URLs
            results = []
            async with aiohttp.ClientSession() as session:
                for url in urls:
                    try:
                        content = await self._fetch_url(session, url)
                        results.append(f"Content from {url}:\n{content}")
                    except Exception as e:
                        results.append(f"Failed to fetch {url}: {e}")

            return '\n\n---\n\n'.join(results)

        except Exception as e:
            self.logger.error(f"Error fetching web content: {e}")
            raise ToolExecutionError(f"Web fetch failed: {e}")

    def _extract_urls(self, text: str) -> List[str]:
        """Extract URLs from text."""
        url_pattern = re.compile(r'https?://[^\s<>"]+')
        return url_pattern.findall(text)

    async def _fetch_url(self, session: aiohttp.ClientSession, url: str) -> str:
        """Fetch content from a single URL."""
        timeout = aiohttp.ClientTimeout(total=30)

        async with session.get(url, timeout=timeout) as response:
            if response.status != 200:
                raise Exception(f"HTTP {response.status}")

            content_type = response.headers.get('content-type', '')

            if 'text' in content_type or 'html' in content_type:
                content = await response.text()
                # Basic HTML stripping
                if 'html' in content_type:
                    content = self._strip_html(content)
                return content[:10000]  # Limit content length
            else:
                return f"[Non-text content: {content_type}]"

    def _strip_html(self, html: str) -> str:
        """Basic HTML tag removal."""
        import re
        clean = re.compile('<.*?>')
        return re.sub(clean, '', html)


class WebSearchTool(BaseTool):
    """Tool for web search (placeholder - would need actual search API)."""

    def __init__(self):
        super().__init__("google_web_search", "Search the web")

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": "google_web_search",
            "description": "Perform a web search and return results",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        }

    async def execute(self, query: str, **kwargs) -> str:
        """Perform web search."""
        # This is a placeholder implementation
        # In a real implementation, you would integrate with a search API
        return f"Web search for '{query}' - This feature requires API integration"


class MemoryTool(BaseTool):
    """Tool for saving information to memory."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("save_memory", "Save information to memory")
        self.config = config
        self.memory_file = Path.home() / ".gemini" / "GEMINI.md"

    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": "save_memory",
            "description": "Save a fact or piece of information to memory",
            "parameters": {
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The fact or information to remember"
                    }
                },
                "required": ["fact"]
            }
        }

    async def execute(self, fact: str, **kwargs) -> str:
        """Save fact to memory."""
        try:
            # Ensure directory exists
            self.memory_file.parent.mkdir(exist_ok=True)

            # Read existing content
            if self.memory_file.exists():
                async with aiofiles.open(self.memory_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
            else:
                content = ""

            # Add fact to memories section
            memories_section = "## Gemini Added Memories\n\n"
            fact_entry = f"- {fact}\n"

            if memories_section in content:
                # Add to existing section
                content = content.replace(memories_section, memories_section + fact_entry)
            else:
                # Create new section
                content += f"\n{memories_section}{fact_entry}"

            # Write back
            async with aiofiles.open(self.memory_file, 'w', encoding='utf-8') as f:
                await f.write(content)

            return f"Memory saved: {fact}"

        except Exception as e:
            self.logger.error(f"Error saving memory: {e}")
            raise ToolExecutionError(f"Failed to save memory: {e}")


class ToolRegistry:
    """Registry for managing all available tools."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tools: Dict[str, BaseTool] = {}
        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialize and register all tools."""
        self.logger.info("Initializing tool registry...")

        # Register built-in tools
        await self._register_builtin_tools()

        # Register MCP tools
        await self._register_mcp_tools()

        # Register discovered tools
        await self._register_discovered_tools()

        self.logger.info(f"Registered {len(self.tools)} tools")

    async def _register_builtin_tools(self):
        """Register built-in tools."""
        builtin_tools = [
            ReadFileTool(),
            WriteFileTool(),
            ListDirectoryTool(),
            ShellTool(self.config),
            GlobTool(),
            SearchTextTool(),
            EditTool(),
            ReadManyFilesTool(),
            WebFetchTool(),
            WebSearchTool(),
            MemoryTool(self.config)
        ]

        # Apply tool filtering
        core_tools = self.config.get("coreTools", [])
        exclude_tools = self.config.get("excludeTools", [])

        for tool in builtin_tools:
            # Check if tool should be excluded
            if exclude_tools and tool.name in exclude_tools:
                continue

            # Check if tool is in core tools (if specified)
            if core_tools and tool.name not in core_tools:
                continue

            self.tools[tool.name] = tool

    async def _register_mcp_tools(self):
        """Register tools from MCP servers."""
        mcp_servers = self.config.get("mcpServers", {})

        for server_name, server_config in mcp_servers.items():
            try:
                # This would integrate with MCP protocol
                # For now, it's a placeholder
                self.logger.info(f"MCP server {server_name} - placeholder")
            except Exception as e:
                self.logger.warning(f"Failed to register MCP server {server_name}: {e}")

    async def _register_discovered_tools(self):
        """Register tools from discovery commands."""
        discovery_command = self.config.get("toolDiscoveryCommand")

        if discovery_command:
            try:
                # Execute discovery command
                process = await asyncio.create_subprocess_shell(
                    discovery_command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )

                stdout, stderr = await process.communicate()

                if process.returncode == 0:
                    # Parse tool definitions
                    tool_defs = json.loads(stdout.decode('utf-8'))
                    # Register discovered tools
                    # This would create DiscoveredTool instances
                    self.logger.info(f"Discovered {len(tool_defs)} tools")

            except Exception as e:
                self.logger.warning(f"Tool discovery failed: {e}")

    def get_tool(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self.tools.get(name)

    def has_tool(self, name: str) -> bool:
        """Check if tool exists."""
        return name in self.tools

    def list_tools(self, include_descriptions: bool = False) -> List[Dict[str, Any]]:
        """List all registered tools."""
        tools = []
        for tool in self.tools.values():
            tool_info = {
                "name": tool.name,
                "description": tool.description
            }

            if include_descriptions:
                tool_info["schema"] = tool.get_schema()

            tools.append(tool_info)

        return tools

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get schemas for all tools."""
        return [tool.get_schema() for tool in self.tools.values()]

    async def execute_tool(self, name: str, params: Dict[str, Any]) -> Any:
        """Execute a tool with given parameters."""
        tool = self.get_tool(name)
        if not tool:
            raise ToolExecutionError(f"Tool not found: {name}")

        # Validate parameters
        if not tool.validate_params(params):
            raise ToolExecutionError(f"Invalid parameters for tool {name}")

        # Execute tool
        try:
            result = await tool.execute(**params)
            return result
        except Exception as e:
            self.logger.error(f"Tool execution failed for {name}: {e}")
            raise ToolExecutionError(f"Tool {name} execution failed: {e}")

    def get_mcp_servers(self) -> List[Dict[str, Any]]:
        """Get MCP server information."""
        # Placeholder for MCP server status
        return []