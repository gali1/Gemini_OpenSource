"""
Memory management for Gemini CLI
Handles GEMINI.md files, context loading, and hierarchical memory
"""

import asyncio
import re
import os
from typing import Dict, Any, List, Optional, Set, Tuple
from pathlib import Path
from datetime import datetime
import logging
import hashlib
import aiofiles

from .exceptions import GeminiCLIError


class MemoryImportProcessor:
    """Processes @file.md imports in GEMINI.md files."""

    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth
        self.logger = logging.getLogger(f"{__name__}.MemoryImportProcessor")

    async def process_imports(
        self,
        content: str,
        base_path: Path,
        debug_mode: bool = False,
        import_state: Optional[Set[Path]] = None
    ) -> str:
        """Process import statements in GEMINI.md content."""
        if import_state is None:
            import_state = set()

        if len(import_state) >= self.max_depth:
            self.logger.warning(f"Maximum import depth ({self.max_depth}) reached")
            return content

        # Find all @file.md imports
        import_pattern = r'@([^\s]+\.md)'
        imports = re.findall(import_pattern, content, re.IGNORECASE)

        if not imports:
            return content

        processed_content = content

        for import_path in imports:
            try:
                # Resolve import path
                resolved_path = self._resolve_import_path(import_path, base_path)

                if not resolved_path:
                    self.logger.warning(f"Could not resolve import path: {import_path}")
                    continue

                # Check for circular imports
                if resolved_path in import_state:
                    self.logger.warning(f"Circular import detected: {resolved_path}")
                    continue

                # Validate import path
                if not self._validate_import_path(resolved_path):
                    self.logger.warning(f"Import path validation failed: {resolved_path}")
                    continue

                # Read imported file
                try:
                    async with aiofiles.open(resolved_path, 'r', encoding='utf-8') as f:
                        imported_content = await f.read()

                    if debug_mode:
                        self.logger.debug(f"Successfully imported: {resolved_path}")

                    # Recursively process imports in the imported file
                    new_import_state = import_state | {resolved_path}
                    imported_content = await self.process_imports(
                        imported_content,
                        resolved_path.parent,
                        debug_mode,
                        new_import_state
                    )

                    # Replace the import statement with the content
                    import_statement = f"@{import_path}"
                    replacement = f"\n<!-- Imported from {import_path} -->\n{imported_content}\n<!-- End import from {import_path} -->\n"
                    processed_content = processed_content.replace(import_statement, replacement)

                except Exception as e:
                    self.logger.error(f"Failed to read imported file {resolved_path}: {e}")
                    # Replace with error comment
                    import_statement = f"@{import_path}"
                    error_replacement = f"\n<!-- Import error for {import_path}: {e} -->\n"
                    processed_content = processed_content.replace(import_statement, error_replacement)

            except Exception as e:
                self.logger.error(f"Error processing import {import_path}: {e}")
                continue

        return processed_content

    def _resolve_import_path(self, import_path: str, base_path: Path) -> Optional[Path]:
        """Resolve import path to absolute path."""
        try:
            # Handle different path formats
            if import_path.startswith('./'):
                # Relative to current directory
                resolved = base_path / import_path[2:]
            elif import_path.startswith('../'):
                # Relative to parent directory
                resolved = base_path / import_path
            elif import_path.startswith('/'):
                # Absolute path
                resolved = Path(import_path)
            else:
                # Relative to current directory (no prefix)
                resolved = base_path / import_path

            # Resolve and normalize
            resolved = resolved.resolve()

            if resolved.exists() and resolved.is_file():
                return resolved

            return None

        except Exception as e:
            self.logger.error(f"Failed to resolve import path {import_path}: {e}")
            return None

    def _validate_import_path(self, path: Path) -> bool:
        """Validate that import path is safe and allowed."""
        try:
            # Check file extension
            if not path.suffix.lower() == '.md':
                self.logger.warning(f"Import processor only supports .md files. Attempting to import: {path}")
                return False

            # Check if file exists and is readable
            if not path.exists() or not path.is_file():
                return False

            # Security check - ensure path is within allowed directories
            # This is a simplified check - could be enhanced based on configuration
            allowed_dirs = [
                Path.home() / ".gemini",
                Path.cwd(),
                Path.cwd().parent
            ]

            # Check if path is within any allowed directory
            path_resolved = path.resolve()
            for allowed_dir in allowed_dirs:
                try:
                    allowed_resolved = allowed_dir.resolve()
                    if path_resolved.is_relative_to(allowed_resolved):
                        return True
                except Exception:
                    continue

            # Also allow if it's in any subdirectory of current working directory
            cwd = Path.cwd().resolve()
            try:
                if path_resolved.is_relative_to(cwd):
                    return True
            except Exception:
                pass

            self.logger.warning(f"Import path outside allowed directories: {path}")
            return False

        except Exception as e:
            self.logger.error(f"Error validating import path {path}: {e}")
            return False


class MemoryManager:
    """Manages GEMINI.md files and hierarchical memory context."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_config = config.get("memory", {})
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.enabled = self.memory_config.get("enabled", True)
        self.max_memory_files = self.memory_config.get("maxMemoryFiles", 50)
        self.context_filename = config.get("contextFileName", "GEMINI.md")

        # State
        self.loaded_files: List[Path] = []
        self.memory_content = ""
        self.last_refresh = None

        # Import processor
        self.import_processor = MemoryImportProcessor()

        # File watching (for auto-refresh)
        self.watch_files: Set[Path] = set()

    async def initialize(self):
        """Initialize memory manager and load context."""
        if not self.enabled:
            self.logger.info("Memory management disabled")
            return

        self.logger.info("Initializing memory manager...")
        await self.refresh()
        self.logger.info(f"Loaded {len(self.loaded_files)} memory files")

    async def refresh(self):
        """Refresh memory context from all GEMINI.md files."""
        if not self.enabled:
            return

        try:
            # Find all context files
            context_files = await self._discover_context_files()

            # Load and process content
            memory_parts = []
            self.loaded_files = []

            for file_path, file_type in context_files:
                try:
                    content = await self._load_context_file(file_path)
                    if content:
                        memory_parts.append(self._format_file_content(file_path, content, file_type))
                        self.loaded_files.append(file_path)

                        # Add to watch list for auto-refresh
                        self.watch_files.add(file_path)

                except Exception as e:
                    self.logger.warning(f"Failed to load context file {file_path}: {e}")
                    continue

            # Combine all memory content
            self.memory_content = "\n\n".join(memory_parts)
            self.last_refresh = datetime.now()

            self.logger.debug(f"Refreshed memory with {len(self.loaded_files)} files")

        except Exception as e:
            self.logger.error(f"Failed to refresh memory: {e}")
            raise GeminiCLIError(f"Memory refresh failed: {e}")

    async def get_context(self) -> str:
        """Get the current memory context."""
        if not self.enabled:
            return ""

        # Auto-refresh if files have changed
        if await self._should_auto_refresh():
            await self.refresh()

        return self.memory_content

    async def add_memory(self, fact: str):
        """Add a fact to user's memory file."""
        try:
            memory_file = Path.home() / ".gemini" / self.context_filename
            memory_file.parent.mkdir(exist_ok=True)

            # Read existing content
            if memory_file.exists():
                async with aiofiles.open(memory_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
            else:
                content = f"# Personal Memory\n\nThis file contains personal information and preferences for Gemini CLI.\n\n"

            # Add new fact to memories section
            memories_section = "## Added Memories\n\n"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            fact_entry = f"- [{timestamp}] {fact}\n"

            if memories_section in content:
                # Add to existing section
                content = content.replace(memories_section, memories_section + fact_entry)
            else:
                # Create new section
                content += f"\n{memories_section}{fact_entry}"

            # Write back to file
            async with aiofiles.open(memory_file, 'w', encoding='utf-8') as f:
                await f.write(content)

            # Refresh memory to include new fact
            await self.refresh()

            self.logger.info(f"Added memory: {fact}")

        except Exception as e:
            self.logger.error(f"Failed to add memory: {e}")
            raise GeminiCLIError(f"Failed to add memory: {e}")

    def get_loaded_files(self) -> List[Path]:
        """Get list of loaded memory files."""
        return self.loaded_files.copy()

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "enabled": self.enabled,
            "loaded_files": len(self.loaded_files),
            "content_length": len(self.memory_content),
            "last_refresh": self.last_refresh.isoformat() if self.last_refresh else None,
            "watch_files": len(self.watch_files)
        }

    async def _discover_context_files(self) -> List[Tuple[Path, str]]:
        """Discover all context files in hierarchical order."""
        context_files = []

        # 1. Global context file (user home)
        global_file = Path.home() / ".gemini" / self.context_filename
        if global_file.exists():
            context_files.append((global_file, "global"))

        # 2. Project root and ancestor context files
        current_path = Path.cwd()
        project_files = []

        while current_path != current_path.parent:
            context_file = current_path / ".gemini" / self.context_filename
            if context_file.exists():
                project_files.append((context_file, "project"))

            # Check for context file in project root
            root_context_file = current_path / self.context_filename
            if root_context_file.exists():
                project_files.append((root_context_file, "project_root"))

            # Stop at git root or home directory
            if (current_path / ".git").exists() or current_path == Path.home():
                break

            current_path = current_path.parent

        # Add project files in reverse order (root first)
        context_files.extend(reversed(project_files))

        # 3. Sub-directory context files
        subdirectory_files = await self._find_subdirectory_context_files(Path.cwd())
        context_files.extend((f, "subdirectory") for f in subdirectory_files)

        # Limit total files
        if len(context_files) > self.max_memory_files:
            self.logger.warning(f"Too many context files ({len(context_files)}), limiting to {self.max_memory_files}")
            context_files = context_files[:self.max_memory_files]

        return context_files

    async def _find_subdirectory_context_files(self, base_path: Path) -> List[Path]:
        """Find context files in subdirectories."""
        context_files = []

        try:
            # Common directories to ignore
            ignore_dirs = {
                '.git', '.vscode', '.idea', '__pycache__', 'node_modules',
                '.pytest_cache', '.coverage', 'dist', 'build', '.env'
            }

            # Walk through subdirectories
            for root, dirs, files in os.walk(base_path):
                # Filter out ignored directories
                dirs[:] = [d for d in dirs if d not in ignore_dirs]

                root_path = Path(root)

                # Don't go too deep
                relative_depth = len(root_path.relative_to(base_path).parts)
                if relative_depth > 3:  # Limit depth
                    continue

                # Look for context files
                context_file = root_path / self.context_filename
                if context_file.exists() and context_file != base_path / self.context_filename:
                    context_files.append(context_file)

        except Exception as e:
            self.logger.warning(f"Error finding subdirectory context files: {e}")

        return context_files

    async def _load_context_file(self, file_path: Path) -> Optional[str]:
        """Load and process a context file."""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()

            # Process imports
            processed_content = await self.import_processor.process_imports(
                content,
                file_path.parent,
                debug_mode=self.config.get("debug", False)
            )

            return processed_content

        except Exception as e:
            self.logger.error(f"Failed to load context file {file_path}: {e}")
            return None

    def _format_file_content(self, file_path: Path, content: str, file_type: str) -> str:
        """Format file content with metadata."""
        relative_path = self._get_relative_path(file_path)

        header = f"<!-- Context from {file_type}: {relative_path} -->"
        footer = f"<!-- End context from {relative_path} -->"

        return f"{header}\n{content}\n{footer}"

    def _get_relative_path(self, file_path: Path) -> str:
        """Get relative path for display."""
        try:
            # Try to get path relative to current working directory
            return str(file_path.relative_to(Path.cwd()))
        except ValueError:
            try:
                # Try relative to home directory
                return str(file_path.relative_to(Path.home()))
            except ValueError:
                # Fall back to absolute path
                return str(file_path)

    async def _should_auto_refresh(self) -> bool:
        """Check if memory should be auto-refreshed."""
        if not self.watch_files:
            return False

        try:
            # Check if any watched files have been modified
            for file_path in self.watch_files:
                if file_path.exists():
                    file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if self.last_refresh is None or file_mtime > self.last_refresh:
                        return True

            return False

        except Exception as e:
            self.logger.warning(f"Error checking for file changes: {e}")
            return False

    def _calculate_content_hash(self, content: str) -> str:
        """Calculate hash of content for change detection."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

    async def export_memory(self, output_path: Path):
        """Export current memory context to a file."""
        try:
            content = await self.get_context()

            if not content:
                raise GeminiCLIError("No memory content to export")

            # Add export metadata
            export_content = f"""# Exported Memory Context
Generated: {datetime.now().isoformat()}
Files: {len(self.loaded_files)}

{content}
"""

            async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                await f.write(export_content)

            self.logger.info(f"Exported memory to: {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to export memory: {e}")
            raise GeminiCLIError(f"Memory export failed: {e}")

    async def validate_context_files(self) -> List[Dict[str, Any]]:
        """Validate all context files and return issues."""
        issues = []

        for file_path in self.loaded_files:
            try:
                if not file_path.exists():
                    issues.append({
                        "file": str(file_path),
                        "type": "missing",
                        "message": "File no longer exists"
                    })
                    continue

                # Check if file is readable
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()

                # Check for common issues
                if len(content) > 100000:  # 100KB
                    issues.append({
                        "file": str(file_path),
                        "type": "large_file",
                        "message": f"File is very large ({len(content)} characters)"
                    })

                # Check for import issues
                import_pattern = r'@([^\s]+\.md)'
                imports = re.findall(import_pattern, content, re.IGNORECASE)

                for import_path in imports:
                    resolved = self.import_processor._resolve_import_path(import_path, file_path.parent)
                    if not resolved:
                        issues.append({
                            "file": str(file_path),
                            "type": "broken_import",
                            "message": f"Cannot resolve import: {import_path}"
                        })

            except Exception as e:
                issues.append({
                    "file": str(file_path),
                    "type": "read_error",
                    "message": str(e)
                })

        return issues

    def clear_memory(self):
        """Clear all loaded memory content."""
        self.memory_content = ""
        self.loaded_files = []
        self.watch_files.clear()
        self.last_refresh = None
        self.logger.info("Memory cleared")

    async def search_memory(self, query: str) -> List[Dict[str, Any]]:
        """Search memory content for specific information."""
        if not self.memory_content:
            return []

        results = []
        query_lower = query.lower()

        # Split content by file sections
        sections = re.split(r'<!-- Context from .*? -->', self.memory_content)

        for i, section in enumerate(sections):
            if query_lower in section.lower():
                # Extract file information
                file_match = re.search(r'<!-- Context from (.*?): (.*?) -->', section)
                if file_match:
                    file_type, file_path = file_match.groups()
                else:
                    file_type, file_path = "unknown", f"section_{i}"

                # Find specific lines containing the query
                lines = section.split('\n')
                matching_lines = [
                    (line_num, line) for line_num, line in enumerate(lines, 1)
                    if query_lower in line.lower()
                ]

                if matching_lines:
                    results.append({
                        "file_type": file_type,
                        "file_path": file_path,
                        "matches": matching_lines[:5]  # Limit matches per file
                    })

        return results