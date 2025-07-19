"""
File discovery service for Gemini CLI
Handles file pattern matching, .gitignore processing, and @ command support
"""

import os
import glob
import fnmatch
import asyncio
from typing import Dict, Any, List, Optional, Set, Pattern, Union
from pathlib import Path
import logging
import re
import mimetypes

from .exceptions import GeminiCLIError


class GitIgnoreProcessor:
    """Processes .gitignore patterns for file filtering."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.GitIgnoreProcessor")
        self._gitignore_cache: Dict[Path, List[str]] = {}

    def load_gitignore_patterns(self, directory: Path) -> List[str]:
        """Load gitignore patterns for a directory."""
        if directory in self._gitignore_cache:
            return self._gitignore_cache[directory]

        patterns = []
        gitignore_file = directory / ".gitignore"

        if gitignore_file.exists():
            try:
                with open(gitignore_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            patterns.append(line)
            except Exception as e:
                self.logger.warning(f"Failed to read .gitignore in {directory}: {e}")

        # Add common ignore patterns
        patterns.extend([
            '.git/**',
            '.git/',
            '__pycache__/**',
            '*.pyc',
            '*.pyo',
            '.coverage',
            '.pytest_cache/**',
            'node_modules/**',
            '.vscode/',
            '.idea/',
            '.DS_Store'
        ])

        self._gitignore_cache[directory] = patterns
        return patterns

    def should_ignore(self, file_path: Path, base_directory: Path) -> bool:
        """Check if a file should be ignored based on gitignore patterns."""
        try:
            # Get gitignore patterns for the base directory
            patterns = self.load_gitignore_patterns(base_directory)

            # Get relative path
            try:
                relative_path = file_path.relative_to(base_directory)
                rel_path_str = str(relative_path)
                rel_path_posix = relative_path.as_posix()
            except ValueError:
                # File is outside base directory
                return False

            for pattern in patterns:
                # Normalize pattern
                pattern = pattern.rstrip('/')

                # Handle directory patterns
                if pattern.endswith('/'):
                    pattern = pattern[:-1] + '/**'

                # Check if pattern matches
                if self._matches_gitignore_pattern(rel_path_str, pattern) or \
                   self._matches_gitignore_pattern(rel_path_posix, pattern):
                    return True

                # Check individual path components
                for part in relative_path.parts:
                    if fnmatch.fnmatch(part, pattern):
                        return True

            return False

        except Exception as e:
            self.logger.warning(f"Error checking gitignore for {file_path}: {e}")
            return False

    def _matches_gitignore_pattern(self, path: str, pattern: str) -> bool:
        """Check if a path matches a gitignore pattern."""
        # Handle negation patterns
        if pattern.startswith('!'):
            return False  # Negation patterns need special handling

        # Handle directory-only patterns
        if pattern.endswith('/'):
            pattern = pattern[:-1]
            return fnmatch.fnmatch(path, pattern) or path.startswith(pattern + '/')

        # Handle glob patterns
        if '**' in pattern:
            # Convert ** to appropriate regex
            regex_pattern = pattern.replace('**', '.*').replace('*', '[^/]*').replace('?', '[^/]')
            regex_pattern = f"^{regex_pattern}$"
            try:
                return bool(re.match(regex_pattern, path))
            except re.error:
                return fnmatch.fnmatch(path, pattern)

        # Simple fnmatch
        return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern)


class FileDiscoveryService:
    """Service for discovering and filtering files based on patterns and rules."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.file_config = config.get("fileFiltering", {})
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.respect_gitignore = self.file_config.get("respectGitIgnore", True)
        self.enable_recursive_search = self.file_config.get("enableRecursiveFileSearch", True)
        self.max_files = self.file_config.get("maxFiles", 1000)
        self.max_file_size = self.file_config.get("maxFileSize", 10 * 1024 * 1024)  # 10MB

        # Components
        self.gitignore_processor = GitIgnoreProcessor()

        # File type mappings
        self.text_extensions = {
            '.txt', '.md', '.py', '.js', '.ts', '.html', '.css', '.json', '.xml',
            '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.sh', '.bat',
            '.sql', '.go', '.rs', '.java', '.c', '.cpp', '.h', '.hpp', '.cs',
            '.php', '.rb', '.pl', '.r', '.scala', '.kt', '.swift', '.dart',
            '.vue', '.jsx', '.tsx', '.scss', '.sass', '.less', '.styl',
            '.dockerfile', '.gitignore', '.gitattributes', '.editorconfig'
        }

        self.binary_extensions = {
            '.exe', '.dll', '.so', '.dylib', '.bin', '.obj', '.o', '.a', '.lib',
            '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg', '.webp',
            '.mp3', '.wav', '.ogg', '.flac', '.mp4', '.avi', '.mov', '.wmv'
        }

    async def discover_files(self, pattern: str, base_path: Optional[Path] = None) -> List[Path]:
        """Discover files matching a pattern."""
        if base_path is None:
            base_path = Path.cwd()

        try:
            # Handle different pattern types
            if self._is_glob_pattern(pattern):
                return await self._discover_glob_pattern(pattern, base_path)
            elif self._is_single_file(pattern, base_path):
                return await self._discover_single_file(pattern, base_path)
            elif self._is_directory(pattern, base_path):
                return await self._discover_directory_files(pattern, base_path)
            else:
                # Treat as glob pattern
                return await self._discover_glob_pattern(pattern, base_path)

        except Exception as e:
            self.logger.error(f"Error discovering files for pattern '{pattern}': {e}")
            raise GeminiCLIError(f"File discovery failed: {e}")

    async def discover_all_files(self, base_path: Optional[Path] = None) -> List[Path]:
        """Discover all relevant files in the base path."""
        if base_path is None:
            base_path = Path.cwd()

        return await self._discover_directory_files(".", base_path, recursive=True)

    def filter_files(
        self,
        files: List[Path],
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        file_types: Optional[List[str]] = None
    ) -> List[Path]:
        """Filter files based on various criteria."""
        filtered_files = []

        for file_path in files:
            # Check file type filter
            if file_types and not self._matches_file_types(file_path, file_types):
                continue

            # Check include patterns
            if include_patterns and not self._matches_any_pattern(file_path, include_patterns):
                continue

            # Check exclude patterns
            if exclude_patterns and self._matches_any_pattern(file_path, exclude_patterns):
                continue

            # Check gitignore
            if self.respect_gitignore and self._should_ignore_file(file_path):
                continue

            # Check file size
            if not self._is_file_size_ok(file_path):
                continue

            filtered_files.append(file_path)

            # Limit number of files
            if len(filtered_files) >= self.max_files:
                self.logger.warning(f"File limit reached ({self.max_files}), truncating results")
                break

        return filtered_files

    def is_text_file(self, file_path: Path) -> bool:
        """Check if a file is a text file."""
        # Check by extension first
        if file_path.suffix.lower() in self.text_extensions:
            return True

        if file_path.suffix.lower() in self.binary_extensions:
            return False

        # Check MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            if mime_type.startswith('text/'):
                return True
            if mime_type in ['application/json', 'application/xml', 'application/javascript']:
                return True

        # Check file content for binary markers
        try:
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    chunk = f.read(8192)
                    # Check for null bytes (common in binary files)
                    if b'\x00' in chunk:
                        return False

                    # Check for high ratio of non-printable characters
                    printable_chars = sum(1 for byte in chunk if 32 <= byte <= 126 or byte in [9, 10, 13])
                    if len(chunk) > 0 and printable_chars / len(chunk) < 0.7:
                        return False

                return True
        except Exception:
            return False

        return False

    def is_supported_file_type(self, file_path: Path) -> bool:
        """Check if file type is supported for reading."""
        # Text files are always supported
        if self.is_text_file(file_path):
            return True

        # Some binary files are supported
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            # Images and PDFs can be processed
            if mime_type.startswith('image/') or mime_type == 'application/pdf':
                return True

        return False

    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get detailed information about a file."""
        try:
            stat = file_path.stat()

            info = {
                "path": str(file_path),
                "name": file_path.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "is_text": self.is_text_file(file_path),
                "is_supported": self.is_supported_file_type(file_path),
                "extension": file_path.suffix,
                "mime_type": mimetypes.guess_type(str(file_path))[0]
            }

            return info

        except Exception as e:
            self.logger.warning(f"Failed to get file info for {file_path}: {e}")
            return {"path": str(file_path), "error": str(e)}

    async def _discover_glob_pattern(self, pattern: str, base_path: Path) -> List[Path]:
        """Discover files using glob pattern."""
        try:
            # Construct full pattern
            if base_path != Path.cwd():
                full_pattern = str(base_path / pattern)
            else:
                full_pattern = pattern

            # Use glob to find matches
            matches = glob.glob(full_pattern, recursive=self.enable_recursive_search)

            # Convert to Path objects and filter
            files = []
            for match in matches:
                file_path = Path(match)
                if file_path.is_file():
                    files.append(file_path)

            # Apply filtering
            return self.filter_files(files)

        except Exception as e:
            self.logger.error(f"Glob pattern discovery failed for '{pattern}': {e}")
            return []

    async def _discover_single_file(self, file_path: str, base_path: Path) -> List[Path]:
        """Discover a single file."""
        try:
            # Resolve path
            if Path(file_path).is_absolute():
                target_path = Path(file_path)
            else:
                target_path = base_path / file_path

            target_path = target_path.resolve()

            if target_path.exists() and target_path.is_file():
                # Apply filtering
                filtered = self.filter_files([target_path])
                return filtered
            else:
                self.logger.warning(f"File not found: {file_path}")
                return []

        except Exception as e:
            self.logger.error(f"Single file discovery failed for '{file_path}': {e}")
            return []

    async def _discover_directory_files(
        self,
        directory: str,
        base_path: Path,
        recursive: bool = True
    ) -> List[Path]:
        """Discover files in a directory."""
        try:
            # Resolve directory path
            if Path(directory).is_absolute():
                target_dir = Path(directory)
            else:
                target_dir = base_path / directory

            target_dir = target_dir.resolve()

            if not target_dir.exists() or not target_dir.is_dir():
                self.logger.warning(f"Directory not found: {directory}")
                return []

            files = []

            if recursive:
                # Recursive discovery
                for root, dirs, filenames in os.walk(target_dir):
                    # Filter directories to respect gitignore
                    if self.respect_gitignore:
                        dirs[:] = [d for d in dirs if not self._should_ignore_dir(Path(root) / d)]

                    for filename in filenames:
                        file_path = Path(root) / filename
                        if file_path.is_file():
                            files.append(file_path)

                            # Limit files to prevent memory issues
                            if len(files) >= self.max_files * 2:  # Allow for filtering
                                break

                    if len(files) >= self.max_files * 2:
                        break
            else:
                # Non-recursive discovery
                try:
                    for item in target_dir.iterdir():
                        if item.is_file():
                            files.append(item)
                except PermissionError:
                    self.logger.warning(f"Permission denied accessing directory: {target_dir}")

            # Apply filtering
            return self.filter_files(files)

        except Exception as e:
            self.logger.error(f"Directory discovery failed for '{directory}': {e}")
            return []

    def _is_glob_pattern(self, pattern: str) -> bool:
        """Check if string is a glob pattern."""
        glob_chars = {'*', '?', '[', ']', '{', '}'}
        return any(char in pattern for char in glob_chars)

    def _is_single_file(self, pattern: str, base_path: Path) -> bool:
        """Check if pattern refers to a single file."""
        if self._is_glob_pattern(pattern):
            return False

        # Check if it's an existing file
        if Path(pattern).is_absolute():
            return Path(pattern).is_file()
        else:
            return (base_path / pattern).is_file()

    def _is_directory(self, pattern: str, base_path: Path) -> bool:
        """Check if pattern refers to a directory."""
        if self._is_glob_pattern(pattern):
            return False

        # Check if it's an existing directory
        if Path(pattern).is_absolute():
            return Path(pattern).is_dir()
        else:
            return (base_path / pattern).is_dir()

    def _matches_file_types(self, file_path: Path, file_types: List[str]) -> bool:
        """Check if file matches any of the specified types."""
        for file_type in file_types:
            if file_type.lower() == 'text' and self.is_text_file(file_path):
                return True
            elif file_type.lower() == 'binary' and not self.is_text_file(file_path):
                return True
            elif file_type.startswith('.') and file_path.suffix.lower() == file_type.lower():
                return True
            elif file_type.lower() in str(file_path).lower():
                return True

        return False

    def _matches_any_pattern(self, file_path: Path, patterns: List[str]) -> bool:
        """Check if file matches any of the patterns."""
        path_str = str(file_path)
        name_str = file_path.name

        for pattern in patterns:
            if fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(name_str, pattern):
                return True

        return False

    def _should_ignore_file(self, file_path: Path) -> bool:
        """Check if file should be ignored based on gitignore."""
        if not self.respect_gitignore:
            return False

        # Find the appropriate base directory (git root or cwd)
        base_dir = self._find_git_root(file_path) or Path.cwd()

        return self.gitignore_processor.should_ignore(file_path, base_dir)

    def _should_ignore_dir(self, dir_path: Path) -> bool:
        """Check if directory should be ignored."""
        # Common directories to ignore
        ignore_dirs = {
            '.git', '.svn', '.hg', '__pycache__', '.pytest_cache',
            'node_modules', '.vscode', '.idea', 'dist', 'build',
            '.coverage', '.tox', '.mypy_cache'
        }

        return dir_path.name in ignore_dirs

    def _is_file_size_ok(self, file_path: Path) -> bool:
        """Check if file size is within limits."""
        try:
            size = file_path.stat().st_size
            return size <= self.max_file_size
        except Exception:
            return False

    def _find_git_root(self, file_path: Path) -> Optional[Path]:
        """Find the git root directory for a file."""
        current = file_path.parent if file_path.is_file() else file_path

        while current != current.parent:
            if (current / '.git').exists():
                return current
            current = current.parent

        return None

    async def search_files_content(
        self,
        query: str,
        file_patterns: Optional[List[str]] = None,
        base_path: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """Search for content within files."""
        if base_path is None:
            base_path = Path.cwd()

        # Discover files to search
        if file_patterns:
            all_files = []
            for pattern in file_patterns:
                files = await self.discover_files(pattern, base_path)
                all_files.extend(files)
        else:
            all_files = await self.discover_all_files(base_path)

        # Filter to text files only
        text_files = [f for f in all_files if self.is_text_file(f)]

        # Search content
        results = []
        query_lower = query.lower()

        for file_path in text_files[:100]:  # Limit search scope
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()

                matches = []
                for line_num, line in enumerate(lines, 1):
                    if query_lower in line.lower():
                        matches.append({
                            "line_number": line_num,
                            "content": line.strip(),
                            "context": self._get_line_context(lines, line_num - 1)
                        })

                if matches:
                    results.append({
                        "file": str(file_path),
                        "matches": matches[:10]  # Limit matches per file
                    })

            except Exception as e:
                self.logger.warning(f"Failed to search in {file_path}: {e}")
                continue

        return results

    def _get_line_context(self, lines: List[str], line_index: int, context_size: int = 2) -> List[str]:
        """Get context lines around a match."""
        start = max(0, line_index - context_size)
        end = min(len(lines), line_index + context_size + 1)

        return [line.strip() for line in lines[start:end]]

    def get_file_stats(self, base_path: Optional[Path] = None) -> Dict[str, Any]:
        """Get file statistics for the base path."""
        if base_path is None:
            base_path = Path.cwd()

        stats = {
            "total_files": 0,
            "text_files": 0,
            "binary_files": 0,
            "total_size": 0,
            "by_extension": {},
            "largest_files": []
        }

        try:
            files = []
            for root, dirs, filenames in os.walk(base_path):
                # Filter directories
                if self.respect_gitignore:
                    dirs[:] = [d for d in dirs if not self._should_ignore_dir(Path(root) / d)]

                for filename in filenames:
                    file_path = Path(root) / filename
                    if file_path.is_file() and not self._should_ignore_file(file_path):
                        files.append(file_path)

            for file_path in files:
                try:
                    file_size = file_path.stat().st_size
                    is_text = self.is_text_file(file_path)

                    stats["total_files"] += 1
                    stats["total_size"] += file_size

                    if is_text:
                        stats["text_files"] += 1
                    else:
                        stats["binary_files"] += 1

                    # Track by extension
                    ext = file_path.suffix.lower() or "(no extension)"
                    stats["by_extension"][ext] = stats["by_extension"].get(ext, 0) + 1

                    # Track largest files
                    stats["largest_files"].append((str(file_path), file_size))

                except Exception:
                    continue

            # Sort and limit largest files
            stats["largest_files"].sort(key=lambda x: x[1], reverse=True)
            stats["largest_files"] = stats["largest_files"][:10]

        except Exception as e:
            self.logger.error(f"Failed to get file stats: {e}")

        return stats