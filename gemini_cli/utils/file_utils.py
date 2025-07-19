"""
File utilities for Gemini CLI
Provides file operations, path manipulation, and file type detection
"""

import os
import shutil
import hashlib
import mimetypes
import tempfile
from typing import List, Dict, Any, Optional, Union, Tuple, Iterator
from pathlib import Path
import logging
import re
import chardet
from datetime import datetime


class FileUtils:
    """Utility class for file operations and management."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # File type mappings
        self.text_extensions = {
            '.txt', '.md', '.py', '.js', '.ts', '.html', '.css', '.json', '.xml',
            '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf', '.sh', '.bat',
            '.sql', '.go', '.rs', '.java', '.c', '.cpp', '.h', '.hpp', '.cs',
            '.php', '.rb', '.pl', '.r', '.scala', '.kt', '.swift', '.dart',
            '.vue', '.jsx', '.tsx', '.scss', '.sass', '.less', '.styl',
            '.dockerfile', '.gitignore', '.gitattributes', '.editorconfig',
            '.makefile', '.cmake', '.gradle', '.pom', '.package', '.lock'
        }

        self.code_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.toml': 'toml',
            '.sql': 'sql',
            '.go': 'go',
            '.rs': 'rust',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.cs': 'csharp',
            '.php': 'php',
            '.rb': 'ruby',
            '.pl': 'perl',
            '.r': 'r',
            '.scala': 'scala',
            '.kt': 'kotlin',
            '.swift': 'swift',
            '.dart': 'dart',
            '.sh': 'bash',
            '.bat': 'batch'
        }

        self.binary_extensions = {
            '.exe', '.dll', '.so', '.dylib', '.bin', '.obj', '.o', '.a', '.lib',
            '.zip', '.tar', '.gz', '.bz2', '.xz', '.7z', '.rar',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.svg', '.webp',
            '.mp3', '.wav', '.ogg', '.flac', '.mp4', '.avi', '.mov', '.wmv',
            '.ttf', '.otf', '.woff', '.woff2'
        }

    def is_text_file(self, file_path: Path) -> bool:
        """Check if a file is a text file."""
        try:
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
                if mime_type.startswith('application/') and any(
                    keyword in mime_type for keyword in ['script', 'source', 'code']
                ):
                    return True

            # Check file content for binary markers
            if file_path.is_file() and file_path.stat().st_size > 0:
                with open(file_path, 'rb') as f:
                    chunk = f.read(min(8192, file_path.stat().st_size))

                    # Check for null bytes (common in binary files)
                    if b'\x00' in chunk:
                        return False

                    # Try to decode as text
                    try:
                        chunk.decode('utf-8')
                        return True
                    except UnicodeDecodeError:
                        # Try with chardet
                        try:
                            result = chardet.detect(chunk)
                            if result and result.get('confidence', 0) > 0.8:
                                return True
                        except Exception:
                            pass

                        return False

            return True

        except Exception as e:
            self.logger.warning(f"Error checking if {file_path} is text file: {e}")
            return False

    def get_file_language(self, file_path: Path) -> Optional[str]:
        """Detect the programming language of a file."""
        extension = file_path.suffix.lower()

        # Direct extension mapping
        if extension in self.code_extensions:
            return self.code_extensions[extension]

        # Special cases based on filename
        filename = file_path.name.lower()

        if filename in ('makefile', 'makefile.am', 'makefile.in'):
            return 'make'
        elif filename in ('dockerfile', 'dockerfile.dev', 'dockerfile.prod'):
            return 'dockerfile'
        elif filename in ('cmakelists.txt',):
            return 'cmake'
        elif filename.endswith('.gradle'):
            return 'gradle'
        elif filename == 'build.gradle':
            return 'gradle'
        elif filename == 'pom.xml':
            return 'xml'
        elif filename in ('package.json', 'package-lock.json'):
            return 'json'
        elif filename.startswith('.'):
            # Configuration files
            if filename.endswith('rc') or filename.endswith('conf'):
                return 'conf'

        # Check shebang for scripts
        if self.is_text_file(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('#!'):
                        shebang = first_line[2:].strip()
                        if 'python' in shebang:
                            return 'python'
                        elif 'node' in shebang or 'nodejs' in shebang:
                            return 'javascript'
                        elif 'bash' in shebang or 'sh' in shebang:
                            return 'bash'
                        elif 'ruby' in shebang:
                            return 'ruby'
                        elif 'perl' in shebang:
                            return 'perl'
            except Exception:
                pass

        return None

    def calculate_file_hash(self, file_path: Path, algorithm: str = 'sha256') -> str:
        """Calculate hash of a file."""
        try:
            hash_func = hashlib.new(algorithm)

            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_func.update(chunk)

            return hash_func.hexdigest()

        except Exception as e:
            self.logger.error(f"Failed to calculate hash for {file_path}: {e}")
            raise

    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get comprehensive information about a file."""
        try:
            stat = file_path.stat()

            info = {
                "path": str(file_path),
                "name": file_path.name,
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime),
                "modified": datetime.fromtimestamp(stat.st_mtime),
                "accessed": datetime.fromtimestamp(stat.st_atime),
                "is_file": file_path.is_file(),
                "is_dir": file_path.is_dir(),
                "is_symlink": file_path.is_symlink(),
                "extension": file_path.suffix,
                "stem": file_path.stem,
                "parent": str(file_path.parent),
                "permissions": oct(stat.st_mode)[-3:]
            }

            if file_path.is_file():
                info.update({
                    "is_text": self.is_text_file(file_path),
                    "language": self.get_file_language(file_path),
                    "mime_type": mimetypes.guess_type(str(file_path))[0],
                    "encoding": self.detect_encoding(file_path)
                })

                # Calculate hash for smaller files
                if stat.st_size < 10 * 1024 * 1024:  # 10MB
                    try:
                        info["sha256"] = self.calculate_file_hash(file_path)
                    except Exception:
                        info["sha256"] = None

            return info

        except Exception as e:
            self.logger.error(f"Failed to get file info for {file_path}: {e}")
            return {"path": str(file_path), "error": str(e)}

    def detect_encoding(self, file_path: Path) -> Optional[str]:
        """Detect the encoding of a text file."""
        try:
            if not self.is_text_file(file_path):
                return None

            with open(file_path, 'rb') as f:
                raw_data = f.read(min(10240, file_path.stat().st_size))

            # Try UTF-8 first
            try:
                raw_data.decode('utf-8')
                return 'utf-8'
            except UnicodeDecodeError:
                pass

            # Use chardet for detection
            try:
                result = chardet.detect(raw_data)
                if result and result.get('confidence', 0) > 0.7:
                    return result['encoding']
            except Exception:
                pass

            # Fallback encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    raw_data.decode(encoding)
                    return encoding
                except UnicodeDecodeError:
                    continue

            return 'unknown'

        except Exception as e:
            self.logger.warning(f"Failed to detect encoding for {file_path}: {e}")
            return None

    def safe_read_file(self, file_path: Path, max_size: int = 10 * 1024 * 1024) -> Optional[str]:
        """Safely read a text file with size and encoding checks."""
        try:
            if not file_path.exists() or not file_path.is_file():
                return None

            file_size = file_path.stat().st_size
            if file_size > max_size:
                self.logger.warning(f"File {file_path} too large ({file_size} bytes)")
                return None

            if not self.is_text_file(file_path):
                return None

            encoding = self.detect_encoding(file_path)
            if not encoding or encoding == 'unknown':
                encoding = 'utf-8'

            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                return f.read()

        except Exception as e:
            self.logger.error(f"Failed to read file {file_path}: {e}")
            return None

    def safe_write_file(self, file_path: Path, content: str, encoding: str = 'utf-8', backup: bool = True) -> bool:
        """Safely write content to a file with optional backup."""
        try:
            # Create parent directories
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Create backup if file exists
            if backup and file_path.exists():
                backup_path = file_path.with_suffix(file_path.suffix + '.bak')
                shutil.copy2(file_path, backup_path)

            # Write to temporary file first
            temp_file = file_path.with_suffix(file_path.suffix + '.tmp')

            with open(temp_file, 'w', encoding=encoding) as f:
                f.write(content)

            # Atomic move
            temp_file.replace(file_path)

            return True

        except Exception as e:
            self.logger.error(f"Failed to write file {file_path}: {e}")
            return False

    def find_files(self, directory: Path, pattern: str = "*", recursive: bool = True, include_hidden: bool = False) -> List[Path]:
        """Find files matching a pattern."""
        try:
            files = []

            if recursive:
                iterator = directory.rglob(pattern)
            else:
                iterator = directory.glob(pattern)

            for path in iterator:
                if path.is_file():
                    # Skip hidden files unless requested
                    if not include_hidden and any(part.startswith('.') for part in path.parts):
                        continue
                    files.append(path)

            return sorted(files)

        except Exception as e:
            self.logger.error(f"Failed to find files in {directory}: {e}")
            return []

    def get_directory_size(self, directory: Path) -> int:
        """Calculate total size of a directory."""
        try:
            total_size = 0

            for path in directory.rglob('*'):
                if path.is_file():
                    try:
                        total_size += path.stat().st_size
                    except (OSError, IOError):
                        continue

            return total_size

        except Exception as e:
            self.logger.error(f"Failed to calculate directory size for {directory}: {e}")
            return 0

    def create_directory_tree(self, directory: Path, max_depth: int = 3, include_files: bool = True) -> Dict[str, Any]:
        """Create a tree representation of a directory."""
        try:
            def build_tree(path: Path, current_depth: int = 0) -> Dict[str, Any]:
                tree = {
                    "name": path.name,
                    "path": str(path),
                    "type": "directory" if path.is_dir() else "file",
                    "children": []
                }

                if path.is_dir() and current_depth < max_depth:
                    try:
                        children = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))

                        for child in children:
                            if child.is_dir() or (include_files and child.is_file()):
                                # Skip hidden files/directories
                                if not child.name.startswith('.'):
                                    tree["children"].append(build_tree(child, current_depth + 1))
                    except PermissionError:
                        tree["error"] = "Permission denied"

                elif path.is_file():
                    tree.update({
                        "size": path.stat().st_size,
                        "extension": path.suffix,
                        "is_text": self.is_text_file(path)
                    })

                return tree

            return build_tree(directory)

        except Exception as e:
            self.logger.error(f"Failed to create directory tree for {directory}: {e}")
            return {"name": directory.name, "error": str(e)}

    def compare_files(self, file1: Path, file2: Path) -> Dict[str, Any]:
        """Compare two files and return differences."""
        try:
            result = {
                "files_exist": file1.exists() and file2.exists(),
                "same_size": False,
                "same_content": False,
                "same_hash": False
            }

            if not result["files_exist"]:
                return result

            # Compare sizes
            size1 = file1.stat().st_size
            size2 = file2.stat().st_size
            result["same_size"] = size1 == size2
            result["size1"] = size1
            result["size2"] = size2

            if not result["same_size"]:
                return result

            # Compare hashes
            hash1 = self.calculate_file_hash(file1)
            hash2 = self.calculate_file_hash(file2)
            result["same_hash"] = hash1 == hash2
            result["hash1"] = hash1
            result["hash2"] = hash2

            result["same_content"] = result["same_hash"]

            return result

        except Exception as e:
            self.logger.error(f"Failed to compare files {file1} and {file2}: {e}")
            return {"error": str(e)}

    def backup_file(self, file_path: Path, backup_dir: Optional[Path] = None) -> Optional[Path]:
        """Create a backup of a file."""
        try:
            if not file_path.exists():
                return None

            if backup_dir is None:
                backup_dir = file_path.parent / ".backups"

            backup_dir.mkdir(exist_ok=True)

            # Create unique backup filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{file_path.stem}_{timestamp}{file_path.suffix}"
            backup_path = backup_dir / backup_name

            shutil.copy2(file_path, backup_path)

            return backup_path

        except Exception as e:
            self.logger.error(f"Failed to backup file {file_path}: {e}")
            return None

    def cleanup_backups(self, backup_dir: Path, keep_count: int = 10) -> int:
        """Clean up old backup files, keeping only the most recent ones."""
        try:
            if not backup_dir.exists():
                return 0

            # Group backups by original filename
            backup_groups = {}

            for backup_file in backup_dir.glob("*_*.*"):
                # Extract original filename from backup name
                parts = backup_file.stem.split('_')
                if len(parts) >= 3:  # name_date_time
                    original_name = '_'.join(parts[:-2])
                    if original_name not in backup_groups:
                        backup_groups[original_name] = []
                    backup_groups[original_name].append(backup_file)

            removed_count = 0

            for original_name, backups in backup_groups.items():
                # Sort by modification time (newest first)
                backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)

                # Remove old backups
                for backup in backups[keep_count:]:
                    try:
                        backup.unlink()
                        removed_count += 1
                    except Exception as e:
                        self.logger.warning(f"Failed to remove backup {backup}: {e}")

            return removed_count

        except Exception as e:
            self.logger.error(f"Failed to cleanup backups in {backup_dir}: {e}")
            return 0

    def normalize_path(self, path: Union[str, Path], base_path: Optional[Path] = None) -> Path:
        """Normalize a path, resolving relative paths and symlinks."""
        try:
            path = Path(path)

            # If path is relative and base_path is provided, resolve relative to base_path
            if not path.is_absolute() and base_path:
                path = base_path / path

            # Resolve symlinks and normalize
            return path.resolve()

        except Exception as e:
            self.logger.error(f"Failed to normalize path {path}: {e}")
            return Path(path)

    def is_safe_path(self, path: Path, allowed_roots: List[Path]) -> bool:
        """Check if a path is within allowed root directories."""
        try:
            resolved_path = path.resolve()

            for root in allowed_roots:
                try:
                    root_resolved = root.resolve()
                    if resolved_path.is_relative_to(root_resolved):
                        return True
                except (ValueError, OSError):
                    continue

            return False

        except Exception:
            return False

    def get_file_stats(self, directory: Path) -> Dict[str, Any]:
        """Get statistics about files in a directory."""
        try:
            stats = {
                "total_files": 0,
                "total_directories": 0,
                "total_size": 0,
                "text_files": 0,
                "binary_files": 0,
                "by_extension": {},
                "by_language": {},
                "largest_files": [],
                "newest_files": [],
                "oldest_files": []
            }

            all_files = []

            for path in directory.rglob('*'):
                try:
                    if path.is_file():
                        stats["total_files"] += 1

                        file_size = path.stat().st_size
                        stats["total_size"] += file_size

                        # Text vs binary
                        if self.is_text_file(path):
                            stats["text_files"] += 1
                        else:
                            stats["binary_files"] += 1

                        # By extension
                        ext = path.suffix.lower() or "(no extension)"
                        stats["by_extension"][ext] = stats["by_extension"].get(ext, 0) + 1

                        # By language
                        language = self.get_file_language(path)
                        if language:
                            stats["by_language"][language] = stats["by_language"].get(language, 0) + 1

                        # Collect for sorting
                        file_info = {
                            "path": path,
                            "size": file_size,
                            "modified": path.stat().st_mtime
                        }
                        all_files.append(file_info)

                    elif path.is_dir():
                        stats["total_directories"] += 1

                except (OSError, IOError):
                    continue

            # Sort files for top lists
            all_files.sort(key=lambda x: x["size"], reverse=True)
            stats["largest_files"] = [
                {"path": str(f["path"]), "size": f["size"]}
                for f in all_files[:10]
            ]

            all_files.sort(key=lambda x: x["modified"], reverse=True)
            stats["newest_files"] = [
                {"path": str(f["path"]), "modified": f["modified"]}
                for f in all_files[:10]
            ]

            all_files.sort(key=lambda x: x["modified"])
            stats["oldest_files"] = [
                {"path": str(f["path"]), "modified": f["modified"]}
                for f in all_files[:10]
            ]

            return stats

        except Exception as e:
            self.logger.error(f"Failed to get file stats for {directory}: {e}")
            return {"error": str(e)}